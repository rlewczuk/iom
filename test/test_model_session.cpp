#include <doctest/doctest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

#include "../src/session_internal.hpp"

// Declared by the CPU driver in `src/cpu/queue.cpp`: process-wide
// post-acceptance failure latches consumed inside an already accepted host
// task. The linear latch is the gate/up projection seam of this stage; the
// SiLU latch is the existing seam of the later activation boundary.
namespace iom::cpu_detail {

void arm_linear_failure(std::size_t healthy_before, std::size_t failures) noexcept;
void clear_linear_failure() noexcept;
void arm_silu_failure() noexcept;
void clear_silu_failure() noexcept;

}  // namespace iom::cpu_detail

namespace {

using iom::session_detail::MlpStageFailure;
using iom::session_detail::MlpStageParams;
using iom::session_detail::MlpStageViews;
using iom::session_detail::MlpWorkspace;
using iom::session_detail::MlpWorkspaceRequirements;

// Value in every result store before a run, so a store that the stage must not
// publish is distinguishable from every reference value.
constexpr float kUntouched = 64.0F;

class HeapAllocator final : public iom::Allocator {
public:
    void* alloc(std::size_t bytes) override {
        return ::operator new(
                std::max<std::size_t>(bytes, 1), std::align_val_t{32});
    }

    void free(void* address) override {
        ::operator delete(address, std::align_val_t{32});
    }

    void reset() override {}
};

struct CpuFixture {
    HeapAllocator allocator;
    std::unique_ptr<iom::Device> device = iom::make_cpu_device(allocator);
};

// Disarms every armed post-acceptance occurrence on scope exit, so an
// interrupted case cannot leak a fault into a later one.
class FailureLatches final {
public:
    FailureLatches() = default;
    ~FailureLatches() {
        iom::cpu_detail::clear_linear_failure();
        iom::cpu_detail::clear_silu_failure();
    }

    FailureLatches(const FailureLatches&) = delete;
    FailureLatches& operator=(const FailureLatches&) = delete;
};

// ---------------------------------------------------------------------------
// BF16 boundary helpers. The comparison policy below is the engine's fixed
// BF16 reference policy: an observed value agrees with the independently
// computed FP64 equation rounded once to BF16 when it stays inside one BF16
// ULP or the fixed absolute/relative allowance.
// ---------------------------------------------------------------------------

[[nodiscard]] std::uint16_t mlp_bf16_bits(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<std::uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

[[nodiscard]] float mlp_bf16_value(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

[[nodiscard]] float mlp_bf16_round(double value) noexcept {
    return mlp_bf16_value(mlp_bf16_bits(static_cast<float>(value)));
}

[[nodiscard]] double mlp_bf16_ulp(double value) noexcept {
    if (value == 0.0 || std::fpclassify(value) == FP_SUBNORMAL) {
        return std::ldexp(1.0, -133);
    }
    const int exponent = std::ilogb(value);
    if (exponent < -126) {
        return std::ldexp(1.0, -133);
    }
    return std::ldexp(1.0, exponent - 7);
}

[[nodiscard]] bool mlp_agrees(float observed, float expected) noexcept {
    if (std::isnan(expected)) return std::isnan(observed);
    if (std::isinf(expected)) return observed == expected;
    if (!std::isfinite(observed)) return false;
    const double reference = static_cast<double>(expected);
    const double difference =
            std::fabs(static_cast<double>(observed) - reference);
    const double allowance = std::max(
            {mlp_bf16_ulp(reference), std::ldexp(1.0, -7),
             std::ldexp(1.0, -6) * std::fabs(reference)});
    return difference <= allowance;
}

// ---------------------------------------------------------------------------
// Tensor helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] iom::TensorSpec mlp_spec(
        std::span<const std::size_t> extents) {
    return iom::TensorSpec{
            iom::TensorShape(std::vector<std::size_t>(
                    extents.begin(), extents.end())),
            iom::DataType::BF16};
}

[[nodiscard]] iom::TensorSpec mlp_spec(
        std::initializer_list<std::size_t> extents) {
    return mlp_spec(std::span<const std::size_t>(extents));
}

[[nodiscard]] std::vector<std::byte> mlp_encode(
        std::span<const float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint16_t bits = mlp_bf16_bits(values[index]);
        bytes[2 * index] = static_cast<std::byte>(bits & 0xffu);
        bytes[2 * index + 1] = static_cast<std::byte>((bits >> 8) & 0xffu);
    }
    return bytes;
}

[[nodiscard]] std::vector<float> mlp_read(iom::Tensor& tensor) {
    const std::size_t elements = tensor.view().spec().shape.element_count();
    std::vector<std::byte> bytes(elements * sizeof(std::uint16_t));
    tensor.view().copy_to_host(bytes);
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const auto low = static_cast<std::uint8_t>(bytes[2 * index]);
        const auto high = static_cast<std::uint8_t>(bytes[2 * index + 1]);
        values[index] = mlp_bf16_value(
                static_cast<std::uint16_t>(low)
                | static_cast<std::uint16_t>(high) << 8);
    }
    return values;
}

void mlp_write(iom::Tensor& tensor, std::span<const float> values) {
    tensor.view().copy_from_host(mlp_encode(values));
}

// Writes the fixed pre-run contents of every result store through the logical
// boundary, leaving tiled padding untouched.
void mlp_fill(iom::Tensor& tensor, float value) {
    const std::size_t elements = tensor.view().spec().shape.element_count();
    mlp_write(tensor, std::vector<float>(elements, value));
}

// Replaces every physical padded row/column element of one tensor with a
// sentinel, so a logical result that reads padding cannot agree with an
// independently computed reference.
void mlp_poison_padding(iom::Tensor& tensor, std::uint16_t sentinel) {
    const iom::TensorSpec spec = tensor.view().spec();
    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    const std::size_t leading_rank = dimensions.size() - 2;
    const iom::TensorShape padded = spec.standard_padded_shape();
    const std::size_t rows = dimensions[leading_rank];
    const std::size_t columns = dimensions[leading_rank + 1];
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        planes *= dimensions[axis];
    }
    auto* storage = static_cast<std::byte*>(tensor.view().native_handle());
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < padded.dimension(leading_rank); ++row) {
            for (std::size_t column = 0;
                 column < padded.dimension(leading_rank + 1); ++column) {
                if (row < rows && column < columns) continue;
                const std::size_t slot =
                        iom::detail::standard_plane_slot(
                                spec, plane, row, column);
                storage[2 * slot] = static_cast<std::byte>(sentinel & 0xffu);
                storage[2 * slot + 1] =
                        static_cast<std::byte>((sentinel >> 8) & 0xffu);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One synthetic MLP bank: the logical host inputs, the device tensors that
// hold them, the seven distinct result stores, and the independent reference
// of every operation boundary.
// ---------------------------------------------------------------------------

struct MlpBank {
    std::size_t planes = 1;
    std::size_t rows = 0;
    std::size_t features = 0;
    std::size_t intermediate = 0;
    float epsilon = 1e-5F;

    std::vector<float> x2_values;
    std::vector<float> scale_values;
    std::vector<float> gate_weight_values;
    std::vector<float> up_weight_values;
    std::vector<float> down_weight_values;

    std::unique_ptr<iom::Tensor> staging;
    std::unique_ptr<iom::Tensor> x2;
    std::unique_ptr<iom::Tensor> scale;
    std::unique_ptr<iom::Tensor> gate_weight;
    std::unique_ptr<iom::Tensor> up_weight;
    std::unique_ptr<iom::Tensor> down_weight;
    std::unique_ptr<iom::Tensor> n2;
    std::unique_ptr<iom::Tensor> gate;
    std::unique_ptr<iom::Tensor> up;
    std::unique_ptr<iom::Tensor> activated;
    std::unique_ptr<iom::Tensor> product;
    std::unique_ptr<iom::Tensor> down;
    std::unique_ptr<iom::Tensor> next_x;

    std::vector<float> ref_n2;
    std::vector<float> ref_gate;
    std::vector<float> ref_up;
    std::vector<float> ref_activated;
    std::vector<float> ref_product;
    std::vector<float> ref_down;
    std::vector<float> ref_next_x;
};

// Deterministic nonsymmetric BF16-exact fixture value: every value is `k/8`
// for a small integer `k`, so inputs and weights are exactly representable and
// swapping two operands cannot agree.
[[nodiscard]] float mlp_value(
        std::uint32_t tag, std::uint32_t row, std::uint32_t column) noexcept {
    const std::uint32_t mixed =
            tag * 31u + row * 17u + column * 11u + row * column * 5u;
    return static_cast<float>(static_cast<int>(mixed % 29u) - 14) / 8.0F;
}

[[nodiscard]] std::vector<float> mlp_activation_values(
        std::size_t planes, std::size_t rows, std::size_t width,
        std::uint32_t tag) {
    std::vector<float> values(planes * rows * width);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                // The plane index participates, so an independent leading
                // plane can never be satisfied by broadcast state.
                values[(plane * rows + row) * width + column] = mlp_value(
                        tag + static_cast<std::uint32_t>(plane) * 101u,
                        static_cast<std::uint32_t>(row),
                        static_cast<std::uint32_t>(column));
            }
        }
    }
    return values;
}

// The independent host reference of every stored operation boundary.
void mlp_compute_reference(MlpBank& bank);

[[nodiscard]] MlpBank mlp_make_bank(
        iom::Device& device, std::size_t planes, std::size_t rows,
        std::size_t features, std::size_t intermediate, std::uint32_t tag) {
    MlpBank bank;
    bank.planes = planes;
    bank.rows = rows;
    bank.features = features;
    bank.intermediate = intermediate;

    const std::vector<std::size_t> activation_extents =
            planes == 1
            ? std::vector<std::size_t>{rows, features}
            : std::vector<std::size_t>{planes, rows, features};
    const std::vector<std::size_t> projection_extents =
            planes == 1
            ? std::vector<std::size_t>{rows, intermediate}
            : std::vector<std::size_t>{planes, rows, intermediate};

    bank.x2_values = mlp_activation_values(planes, rows, features, tag);
    bank.scale_values = mlp_activation_values(1, 1, features, tag + 300u);
    bank.gate_weight_values =
            mlp_activation_values(1, intermediate, features, tag + 400u);
    bank.up_weight_values =
            mlp_activation_values(1, intermediate, features, tag + 500u);
    bank.down_weight_values =
            mlp_activation_values(1, features, intermediate, tag + 600u);

    bank.staging = device.create_tensor(mlp_spec(activation_extents));
    bank.x2 = device.create_tensor(mlp_spec(activation_extents));
    bank.scale = device.create_tensor(mlp_spec({1, features}));
    bank.gate_weight =
            device.create_tensor(mlp_spec({intermediate, features}));
    bank.up_weight = device.create_tensor(mlp_spec({intermediate, features}));
    bank.down_weight =
            device.create_tensor(mlp_spec({features, intermediate}));
    bank.n2 = device.create_tensor(mlp_spec(activation_extents));
    bank.gate = device.create_tensor(mlp_spec(projection_extents));
    bank.up = device.create_tensor(mlp_spec(projection_extents));
    bank.activated = device.create_tensor(mlp_spec(projection_extents));
    bank.product = device.create_tensor(mlp_spec(projection_extents));
    bank.down = device.create_tensor(mlp_spec(activation_extents));
    bank.next_x = device.create_tensor(mlp_spec(activation_extents));

    mlp_write(*bank.staging, bank.x2_values);
    mlp_write(*bank.scale, bank.scale_values);
    mlp_write(*bank.gate_weight, bank.gate_weight_values);
    mlp_write(*bank.up_weight, bank.up_weight_values);
    mlp_write(*bank.down_weight, bank.down_weight_values);
    mlp_fill(*bank.n2, kUntouched);
    mlp_fill(*bank.gate, kUntouched);
    mlp_fill(*bank.up, kUntouched);
    mlp_fill(*bank.activated, kUntouched);
    mlp_fill(*bank.product, kUntouched);
    mlp_fill(*bank.down, kUntouched);
    mlp_fill(*bank.next_x, kUntouched);

    mlp_compute_reference(bank);
    return bank;
}

// The independent host reference of every stored operation boundary: the
// equations run in FP64, each boundary is rounded once to the BF16 store, and
// every consumer reads that stored value.
void mlp_compute_reference(MlpBank& bank) {
    const std::size_t features = bank.features;
    const std::size_t intermediate = bank.intermediate;
    const std::size_t rows = bank.rows;
    const double epsilon = static_cast<double>(bank.epsilon);
    const std::size_t activation_plane = rows * features;
    const std::size_t projection_plane = rows * intermediate;

    bank.ref_n2.assign(bank.planes * activation_plane, 0.0F);
    bank.ref_gate.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_up.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_activated.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_product.assign(bank.planes * projection_plane, 0.0F);
    bank.ref_down.assign(bank.planes * activation_plane, 0.0F);
    bank.ref_next_x.assign(bank.planes * activation_plane, 0.0F);

    for (std::size_t plane = 0; plane < bank.planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t activation_base =
                    plane * activation_plane + row * features;
            const std::size_t projection_base =
                    plane * projection_plane + row * intermediate;

            double square_sum = 0.0;
            for (std::size_t feature = 0; feature < features; ++feature) {
                const double value = bank.x2_values[activation_base + feature];
                square_sum += value * value;
            }
            const double inverse = 1.0
                    / std::sqrt(
                            square_sum / static_cast<double>(features)
                            + epsilon);
            for (std::size_t feature = 0; feature < features; ++feature) {
                bank.ref_n2[activation_base + feature] = mlp_bf16_round(
                        bank.x2_values[activation_base + feature] * inverse
                        * bank.scale_values[feature]);
            }

            for (std::size_t column = 0; column < intermediate; ++column) {
                double gate_sum = 0.0;
                double up_sum = 0.0;
                for (std::size_t feature = 0; feature < features; ++feature) {
                    const double normalized =
                            bank.ref_n2[activation_base + feature];
                    gate_sum += normalized
                            * bank.gate_weight_values[column * features
                                                      + feature];
                    up_sum += normalized
                            * bank.up_weight_values[column * features + feature];
                }
                bank.ref_gate[projection_base + column] =
                        mlp_bf16_round(gate_sum);
                bank.ref_up[projection_base + column] =
                        mlp_bf16_round(up_sum);
            }

            for (std::size_t column = 0; column < intermediate; ++column) {
                const double stored_gate = bank.ref_gate[projection_base + column];
                bank.ref_activated[projection_base + column] = mlp_bf16_round(
                        stored_gate / (1.0 + std::exp(-stored_gate)));
                bank.ref_product[projection_base + column] = mlp_bf16_round(
                        static_cast<double>(
                                bank.ref_activated[projection_base + column])
                        * static_cast<double>(
                                bank.ref_up[projection_base + column]));
            }

            for (std::size_t feature = 0; feature < features; ++feature) {
                double down_sum = 0.0;
                for (std::size_t column = 0; column < intermediate;
                     ++column) {
                    down_sum +=
                            static_cast<double>(
                                    bank.ref_product[projection_base + column])
                            * bank.down_weight_values[feature * intermediate
                                                      + column];
                }
                bank.ref_down[activation_base + feature] =
                        mlp_bf16_round(down_sum);
                bank.ref_next_x[activation_base + feature] = mlp_bf16_round(
                        bank.x2_values[activation_base + feature]
                        + static_cast<double>(
                                bank.ref_down[activation_base + feature]));
            }
        }
    }
}

[[nodiscard]] MlpStageViews mlp_make_views(
        iom::Tensor& x2, iom::Tensor& scale, iom::Tensor& gate_weight,
        iom::Tensor& up_weight, iom::Tensor& down_weight, iom::Tensor& n2,
        iom::Tensor& gate, iom::Tensor& up, iom::Tensor& activated,
        iom::Tensor& product, iom::Tensor& down, iom::Tensor& next_x) {
    return MlpStageViews{
            x2.view(),     scale.view(),   gate_weight.view(),
            up_weight.view(), down_weight.view(), n2.view(),
            gate.view(),   up.view(),      activated.view(),
            product.view(), down.view(),   next_x.view()};
}

[[nodiscard]] MlpStageViews mlp_views(const MlpBank& bank) {
    return mlp_make_views(
            *bank.x2, *bank.scale, *bank.gate_weight, *bank.up_weight,
            *bank.down_weight, *bank.n2, *bank.gate, *bank.up, *bank.activated,
            *bank.product, *bank.down, *bank.next_x);
}

[[nodiscard]] MlpStageParams mlp_params(const MlpBank& bank) noexcept {
    return MlpStageParams{
            bank.rows, bank.features, bank.intermediate, bank.epsilon};
}

// The request-scoped workspace of the two actual run banks. CPU reports the
// conventional zero requirement for every member, so the resolved members are
// the empty views the zero-workspace operations accept.
[[nodiscard]] MlpWorkspace mlp_workspace(
        iom::DeviceOps& operations, const MlpBank& prefill,
        const MlpBank& decode) {
    const MlpStageViews prefill_views = mlp_views(prefill);
    const MlpStageViews decode_views = mlp_views(decode);
    const MlpWorkspaceRequirements requirements =
            iom::session_detail::mlp_workspace_requirements(
                    operations, prefill_views, decode_views);
    return iom::session_detail::resolve_mlp_workspace(
            operations.device(), requirements, iom::RawWorkspaceView{});
}

// Submits the bank's first-residual producer: the exact copy of the
// caller-owned staging tensor into the residual store, which the stage must
// wait before it reads `x2`.
[[nodiscard]] iom::oid mlp_submit_first_residual(
        MlpBank& bank, iom::DeviceOps& operations) {
    const iom::oid producer =
            operations.copy(bank.staging->view(), bank.x2->view());
    REQUIRE(iom::oid_is_token(producer));
    return producer;
}

void mlp_drive(
        MlpBank& bank, iom::DeviceOps& operations, iom::oid producer,
        MlpWorkspace& workspace, MlpStageFailure& failure) {
    MlpStageViews views = mlp_views(bank);
    const MlpStageParams params = mlp_params(bank);
    const std::span<const iom::oid> readiness(&producer, 1);
    iom::session_detail::run_mlp_stage(
            operations, views, params, workspace, readiness, failure);
}

// Runs the stage and returns the failure message, or an empty string when the
// stage completed successfully.
[[nodiscard]] std::string mlp_failure_message(
        MlpBank& bank, iom::DeviceOps& operations, iom::oid producer,
        MlpWorkspace& workspace, MlpStageFailure& failure) {
    try {
        mlp_drive(bank, operations, producer, workspace, failure);
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

void mlp_check(
        iom::Tensor& store, std::span<const float> expected,
        const char* boundary) {
    const std::vector<float> observed = mlp_read(store);
    REQUIRE(observed.size() == expected.size());
    std::string first_mismatch;
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        if (mlp_agrees(observed[index], expected[index])) continue;
        ++mismatches;
        if (first_mismatch.empty()) {
            first_mismatch = std::string(boundary) + " element "
                    + std::to_string(index) + " observed "
                    + std::to_string(observed[index]) + " expected "
                    + std::to_string(expected[index]);
        }
    }
    CHECK_MESSAGE(mismatches == 0, first_mismatch);
}

// One result store must still hold its exact pre-run contents after a failed
// stage: no dependent stage was published.
void mlp_check_untouched(iom::Tensor& store, const char* boundary) {
    const std::vector<float> observed = mlp_read(store);
    const std::uint16_t sentinel = mlp_bf16_bits(kUntouched);
    std::string first_mismatch;
    for (std::size_t index = 0; index < observed.size(); ++index) {
        if (mlp_bf16_bits(observed[index]) == sentinel) continue;
        if (first_mismatch.empty()) {
            first_mismatch = std::string(boundary) + " element "
                    + std::to_string(index) + " changed to "
                    + std::to_string(observed[index]);
        }
    }
    CHECK_MESSAGE(first_mismatch.empty(), first_mismatch);
}

void mlp_check_reference(MlpBank& bank) {
    mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
    mlp_check(*bank.gate, bank.ref_gate, "gate projection");
    mlp_check(*bank.up, bank.ref_up, "up projection");
    mlp_check(*bank.activated, bank.ref_activated, "activated gate");
    mlp_check(*bank.product, bank.ref_product, "product");
    mlp_check(*bank.down, bank.ref_down, "down projection");
    mlp_check(*bank.next_x, bank.ref_next_x, "next residual");
}

// The first residual and every weight are read only: the stage never rewrites
// its inputs.
void mlp_check_inputs(const MlpBank& bank) {
    CHECK(mlp_read(*bank.x2) == bank.x2_values);
    CHECK(mlp_read(*bank.scale) == bank.scale_values);
    CHECK(mlp_read(*bank.gate_weight) == bank.gate_weight_values);
    CHECK(mlp_read(*bank.up_weight) == bank.up_weight_values);
    CHECK(mlp_read(*bank.down_weight) == bank.down_weight_values);
}

void mlp_fill_stores(MlpBank& bank) {
    mlp_fill(*bank.n2, kUntouched);
    mlp_fill(*bank.gate, kUntouched);
    mlp_fill(*bank.up, kUntouched);
    mlp_fill(*bank.activated, kUntouched);
    mlp_fill(*bank.product, kUntouched);
    mlp_fill(*bank.down, kUntouched);
    mlp_fill(*bank.next_x, kUntouched);
}

[[nodiscard]] std::vector<float> mlp_store_image(const MlpBank& bank) {
    std::vector<float> image;
    const std::unique_ptr<iom::Tensor>* const stores[7] = {
            &bank.n2, &bank.gate, &bank.up, &bank.activated, &bank.product,
            &bank.down, &bank.next_x};
    for (const std::unique_ptr<iom::Tensor>* store : stores) {
        const std::vector<float> values = mlp_read(**store);
        image.insert(image.end(), values.begin(), values.end());
    }
    return image;
}

}  // namespace

TEST_CASE("TinyLlama MLP stage matches an independent reference for prefill and R1 decode") {
    struct Shape {
        std::size_t rows;
        std::size_t features;
        std::size_t intermediate;
        std::uint32_t tag;
    };
    // A non-tile prefill run, the fixed one-row decode run, and a run past the
    // 16x16 tile row boundary.
    const Shape shapes[] = {
            {15, 8, 12, 1}, {1, 8, 12, 2}, {17, 8, 12, 3}};

    for (const Shape& shape : shapes) {
        CpuFixture fixture;
        MlpBank bank = mlp_make_bank(
                *fixture.device, 1, shape.rows, shape.features,
                shape.intermediate, shape.tag);
        auto operations = fixture.device->create_ops();
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        MlpStageFailure failure;
        mlp_drive(bank, *operations, producer, workspace, failure);

        CHECK_FALSE(failure.poisoned());
        mlp_check_reference(bank);
        mlp_check_inputs(bank);
    }
}

TEST_CASE("TinyLlama MLP stage isolates tiled padding and independent leading planes") {
    CpuFixture fixture;
    // Two identical banks of independent leading planes whose physical padding
    // carries different sentinels: one is the nonfinite pattern a padding read
    // would poison, the other a finite pattern.
    MlpBank poisoned = mlp_make_bank(*fixture.device, 2, 15, 8, 12, 4);
    MlpBank finite = mlp_make_bank(*fixture.device, 2, 15, 8, 12, 4);
    const std::unique_ptr<iom::Tensor>* const poisoned_stores[7] = {
            &poisoned.n2, &poisoned.gate, &poisoned.up, &poisoned.activated,
            &poisoned.product, &poisoned.down, &poisoned.next_x};
    const std::unique_ptr<iom::Tensor>* const finite_stores[7] = {
            &finite.n2, &finite.gate, &finite.up, &finite.activated,
            &finite.product, &finite.down, &finite.next_x};
    for (std::size_t index = 0; index < 7; ++index) {
        mlp_poison_padding(**poisoned_stores[index], 0x7F80u);
        mlp_poison_padding(**finite_stores[index], 0x3F00u);
    }
    // The read-only operands carry sentinels too, so a weight or activation
    // result that consumes padded rows or features cannot agree either.
    const std::unique_ptr<iom::Tensor>* const poisoned_inputs[5] = {
            &poisoned.x2, &poisoned.scale, &poisoned.gate_weight,
            &poisoned.up_weight, &poisoned.down_weight};
    const std::unique_ptr<iom::Tensor>* const finite_inputs[5] = {
            &finite.x2, &finite.scale, &finite.gate_weight, &finite.up_weight,
            &finite.down_weight};
    for (std::size_t index = 0; index < 5; ++index) {
        mlp_poison_padding(**poisoned_inputs[index], 0x7F80u);
        mlp_poison_padding(**finite_inputs[index], 0x7F00u);
    }

    auto operations = fixture.device->create_ops();
    const iom::oid poisoned_producer =
            mlp_submit_first_residual(poisoned, *operations);
    const iom::oid finite_producer =
            mlp_submit_first_residual(finite, *operations);
    MlpWorkspace workspace = mlp_workspace(*operations, poisoned, finite);
    MlpStageFailure poisoned_failure;
    MlpStageFailure finite_failure;
    mlp_drive(poisoned, *operations, poisoned_producer, workspace, poisoned_failure);
    mlp_drive(finite, *operations, finite_producer, workspace, finite_failure);

    CHECK_FALSE(poisoned_failure.poisoned());
    CHECK_FALSE(finite_failure.poisoned());
    mlp_check_reference(poisoned);
    mlp_check_reference(finite);
    CHECK(mlp_store_image(poisoned) == mlp_store_image(finite));
    // Both planes carry independent state: the leading plane is not broadcast.
    CHECK(mlp_read(*poisoned.gate) == poisoned.ref_gate);
}

TEST_CASE("TinyLlama MLP stage rejects inconsistent views, aliases, and readiness before submission") {
    CpuFixture fixture;
    MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 5);
    auto operations = fixture.device->create_ops();
    MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
    const MlpStageParams params = mlp_params(bank);
    const iom::oid producer = mlp_submit_first_residual(bank, *operations);

    SUBCASE("a mismatched weight view fails before any submission") {
        auto wrong_weight = fixture.device->create_tensor(
                mlp_spec({bank.intermediate + 1, bank.features}));
        MlpStageViews views = mlp_make_views(
                *bank.x2, *bank.scale, *wrong_weight, *bank.up_weight,
                *bank.down_weight, *bank.n2, *bank.gate, *bank.up,
                *bank.activated, *bank.product, *bank.down, *bank.next_x);
        MlpStageFailure failure;
        const std::span<const iom::oid> readiness(&producer, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK_FALSE(failure.poisoned());
        mlp_check_untouched(*bank.n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("an in-place activation alias fails instead of overwriting a store") {
        MlpStageViews views = mlp_make_views(
                *bank.x2, *bank.scale, *bank.gate_weight, *bank.up_weight,
                *bank.down_weight, *bank.n2, *bank.gate, *bank.up, *bank.gate,
                *bank.product, *bank.down, *bank.next_x);
        MlpStageFailure failure;
        const std::span<const iom::oid> readiness(&producer, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK(failure.poisoned());
        // The dependent stores were never published, and the gate store was
        // never overwritten in place by the activation.
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("an invalid readiness token is rejected without waiting it") {
        MlpStageFailure failure;
        MlpStageViews views = mlp_views(bank);
        const iom::oid invalid = 0;
        const std::span<const iom::oid> readiness(&invalid, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK_FALSE(failure.poisoned());
        mlp_check_untouched(*bank.n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("a producer from another queue is drained as a failure") {
        auto foreign_operations = fixture.device->create_ops();
        const iom::oid foreign = foreign_operations->copy(
                bank.staging->view(), bank.x2->view());
        REQUIRE(iom::oid_is_token(foreign));
        MlpStageFailure failure;
        MlpStageViews views = mlp_views(bank);
        const std::span<const iom::oid> readiness(&foreign, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        // A readiness producer that fails its wait is an accepted-work
        // failure: the execution stays poisoned and is not reused.
        CHECK(failure.poisoned());
        mlp_check_untouched(*bank.n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    SUBCASE("the checked workspace requirement keeps gate and up disjoint") {
        // The two actual banks contribute their checked member maxima; CPU
        // reports the conventional zero requirement, so the resolved members
        // stay the empty views the zero-workspace operations accept.
        const MlpStageViews prefill_views = mlp_views(bank);
        const MlpStageViews decode_views = mlp_views(bank);
        const MlpWorkspaceRequirements requirements =
                iom::session_detail::mlp_workspace_requirements(
                        *operations, prefill_views, decode_views);
        CHECK(requirements.total() == iom::WorkspaceRequirements{0, 1});
        const MlpWorkspace resolved =
                iom::session_detail::resolve_mlp_workspace(
                        *fixture.device, requirements, iom::RawWorkspaceView{});
        CHECK(resolved.gate.empty());
        CHECK(resolved.up.empty());
        CHECK(resolved.mul.empty());
        CHECK(resolved.down.empty());
        CHECK(resolved.residual.empty());

        // A backend with positive member requirements reserves two permanent
        // slices at the fixed 32-byte granularity: the reused gate slice
        // (shared by mul, down, and the second residual after their producer
        // completed) and the disjoint up slice.
        const MlpWorkspaceRequirements synthetic{
                {40, 32}, {8, 32}, {0, 1}, {16, 32}, {4, 32}};
        CHECK(synthetic.total() == iom::WorkspaceRequirements{96, 32});
    }
}

TEST_CASE("TinyLlama MLP stage drains both projection branches after an accepted failure") {
    CpuFixture fixture;
    auto operations = fixture.device->create_ops();
    FailureLatches latches;

    // One failing gate branch while the up branch succeeds.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 6);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_linear_failure(0, 1);
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU linear projection injected post-acceptance "
                             "failure") != std::string::npos,
                message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check(*bank.up, bank.ref_up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");

        // The poisoned execution is never reused: the preserved first failure
        // is rethrown without submitting anything, even though the latch is
        // already disarmed and the queue is healthy.
        const std::vector<float> before = mlp_store_image(bank);
        const std::string replay =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK(replay == message);
        CHECK(mlp_store_image(bank) == before);

        // Once every accepted OID has been drained, the same queue, tensors,
        // and resolved workspace are reusable.
        MlpStageFailure recovered;
        mlp_drive(bank, *operations, producer, workspace, recovered);
        CHECK_FALSE(recovered.poisoned());
        mlp_check_reference(bank);
    }

    // Both projection branches fail after acceptance.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 7);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_linear_failure(0, 2);
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU linear projection injected post-acceptance "
                             "failure") != std::string::npos,
                message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check_untouched(*bank.up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    // One failing up branch while the gate branch succeeds: the armed plan
    // lets the gate projection complete and fails the sibling branch after its
    // own acceptance.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 8);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_linear_failure(1, 1);
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU linear projection injected post-acceptance "
                             "failure") != std::string::npos,
                message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check(*bank.gate, bank.ref_gate, "gate projection");
        mlp_check_untouched(*bank.up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    // A rejected gate admission still submits, accepts, and drains the up
    // branch before the preserved first failure is rethrown.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 10);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);

        // A weight owner of another device is a plain BF16 [M,F] view, so the
        // rejection happens at the projection's own admission rather than in
        // the stage's local shape check.
        HeapAllocator foreign_allocator;
        std::unique_ptr<iom::Device> foreign_device =
                iom::make_cpu_device(foreign_allocator);
        auto foreign_weight = foreign_device->create_tensor(
                mlp_spec({bank.intermediate, bank.features}));
        MlpStageViews views = mlp_make_views(
                *bank.x2, *bank.scale, *foreign_weight, *bank.up_weight,
                *bank.down_weight, *bank.n2, *bank.gate, *bank.up,
                *bank.activated, *bank.product, *bank.down, *bank.next_x);

        const iom::oid producer = mlp_submit_first_residual(bank, *operations);
        MlpStageFailure failure;
        const MlpStageParams params = mlp_params(bank);
        const std::span<const iom::oid> readiness(&producer, 1);
        CHECK_THROWS_AS(
                iom::session_detail::run_mlp_stage(
                        *operations, views, params, workspace, readiness,
                        failure),
                std::invalid_argument);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check_untouched(*bank.gate, "gate projection");
        mlp_check(*bank.up, bank.ref_up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }

    // A failing accepted activation boundary stops every dependent stage.
    {
        MlpBank bank = mlp_make_bank(*fixture.device, 1, 15, 8, 12, 9);
        mlp_fill_stores(bank);
        MlpWorkspace workspace = mlp_workspace(*operations, bank, bank);
        const iom::oid producer = mlp_submit_first_residual(bank, *operations);

        iom::cpu_detail::arm_silu_failure();
        MlpStageFailure failure;
        const std::string message =
                mlp_failure_message(bank, *operations, producer, workspace, failure);
        CHECK_MESSAGE(
                message.find("CPU SiLU injected") != std::string::npos, message);
        CHECK(failure.poisoned());
        mlp_check(*bank.n2, bank.ref_n2, "post-attention norm");
        mlp_check(*bank.gate, bank.ref_gate, "gate projection");
        mlp_check(*bank.up, bank.ref_up, "up projection");
        mlp_check_untouched(*bank.activated, "activated gate");
        mlp_check_untouched(*bank.product, "product");
        mlp_check_untouched(*bank.down, "down projection");
        mlp_check_untouched(*bank.next_x, "next residual");
    }
}

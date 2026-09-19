#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>
#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_copy_storage.hpp"
#include "backend/backend_conformance_embedding.hpp"
#include "backend/backend_conformance_linear.hpp"
#include "backend/backend_conformance_other.hpp"
#include "backend/backend_conformance_rmsnorm.hpp"
#include "backend/backend_conformance_rope.hpp"

#include "backend/backend_conformance_add.hpp"
#include "backend/backend_conformance_model_loading.hpp"
#include "iom/alloc.hpp"
#include "iom/cpu/device.hpp"

namespace {


// Recording allocator over 32-byte-aligned blocks. While the gate is armed —
// inside transfers, transforms, and queue operations — every alloc or free is
// a conformance failure: no case may allocate operands there.
class GatedAllocator final : public iom::Allocator {
public:
    explicit GatedAllocator(const iom_conformance::TrafficGate& gate) : gate_(gate) {}

    void* alloc(std::size_t size) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage allocated inside a transfer, transform, or "
                "operation");
        void* block = ::operator new(size, std::align_val_t(32));
        live_.insert(block);
        ++traffic_;
        return block;
    }

    void free(void* buffer) override {
        CHECK_MESSAGE(
                !gate_.armed(),
                "tensor storage freed inside a transfer, transform, or "
                "operation");
        const auto found = live_.find(buffer);
        REQUIRE_MESSAGE(
                found != live_.end(),
                "allocator freed an address it never handed out");
        live_.erase(found);
        ++traffic_;
        ::operator delete(buffer, std::align_val_t(32));
    }

    void reset() override {
        ++traffic_;
    }

    [[nodiscard]] std::size_t traffic() const noexcept {
        return traffic_;
    }

private:
    const iom_conformance::TrafficGate& gate_;
    std::unordered_set<void*> live_;
    std::size_t traffic_ = 0;
};

// One independent CPU reference device, one candidate device, and one foreign
// device instance of the same backend ordinal, each over its own allocator.
struct CpuDevices {
    iom_conformance::TrafficGate gate;
    GatedAllocator reference_allocator{gate};
    GatedAllocator candidate_allocator{gate};
    GatedAllocator foreign_allocator{gate};
    std::unique_ptr<iom::Device> reference = iom::make_cpu_device(reference_allocator);
    std::unique_ptr<iom::Device> candidate = iom::make_cpu_device(candidate_allocator);
    std::unique_ptr<iom::Device> foreign = iom::make_cpu_device(foreign_allocator);



    [[nodiscard]] iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};
// Submit only derived temporary views so the worker can only succeed by using
// the value-captured snapshot. The source owner is released by the caller
// before the dependent copy is waited, exercising registry retention.
iom::oid submit_temporary_rope(
        iom::DeviceOps& queue, iom::Tensor& source,
        iom::Tensor& destination) {
    return queue.rope(
            source.view().slice(0, 0, 1),
            const_cast<iom::TensorView&>(
                    static_cast<const iom::TensorView&>(
                            destination.view().slice(0, 0, 1))),
            0, 10000.0);
}


}  // namespace

void require_storage_outside_view_is_zero(
        const iom::TensorSpec& owner_spec, const iom::TensorView& view) {
    std::vector<unsigned char> touched(
            owner_spec.tiled_storage_nbytes(), 0);
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    const std::span<const std::size_t> strides =
            view.plane_strides();
    REQUIRE_EQ(dimensions.size(), 4);
    for (std::size_t first = 0; first < dimensions[0]; ++first) {
        for (std::size_t second = 0; second < dimensions[1]; ++second) {
            const std::size_t plane =
                    view.plane_offset()
                    + first * strides[0] + second * strides[1];
            for (std::size_t row = 0; row < dimensions[2]; ++row) {
                for (std::size_t column = 0;
                     column < dimensions[3]; ++column) {
                    const std::size_t byte =
                            iom::detail::standard_plane_slot(
                                    owner_spec, plane, row, column);
                    REQUIRE_LT(byte, touched.size());
                    touched[byte] = 1;
                }
            }
        }
    }

    const auto* storage = static_cast<const unsigned char*>(
            view.native_handle());
    for (std::size_t byte = 0; byte < touched.size(); ++byte) {
        if (touched[byte] == 0) {
            CHECK_EQ(storage[byte], 0);
        }
    }
}

void require_subbyte_storage_scope_is_zero(
        const iom::TensorSpec& owner_spec, const iom::TensorView& view) {
    const std::size_t bits =
            iom::detail::leaf_bits(owner_spec.data_type);
    std::vector<unsigned char> touched(
            owner_spec.tiled_storage_nbytes(), 0);
    const std::span<const std::size_t> dimensions =
            view.spec().shape.dimensions();
    REQUIRE_EQ(dimensions.size(), 2);
    for (std::size_t row = 0; row < dimensions[0]; ++row) {
        for (std::size_t column = 0; column < dimensions[1]; ++column) {
            const std::size_t bit =
                    iom::detail::standard_plane_slot(
                            owner_spec, view.plane_offset(), row, column)
                    * bits;
            for (std::size_t offset = 0; offset < bits; ++offset) {
                touched[(bit + offset) / 8] |=
                        static_cast<unsigned char>(
                                1u << ((bit + offset) % 8));
            }
        }
    }

    const auto* storage = static_cast<const unsigned char*>(
            view.native_handle());
    for (std::size_t byte = 0; byte < touched.size(); ++byte) {
        CHECK_EQ(
                static_cast<unsigned char>(storage[byte] & ~touched[byte]),
                0);
    }
}
TEST_CASE("Device::supported_data_types returns the per-backend 23-entry span") {
    std::vector<std::byte> storage(1024);
    iom::LinearAllocator allocator(storage.data(), storage.size());
    const std::unique_ptr<iom::Device> candidate =
            iom::make_cpu_device(allocator);
    iom_conformance::require_standard_capabilities(
            candidate->supported_data_types());
}


// The CPU reference instantiation passes every shared case for every declared
// leaf type. Each entry point runs in its own test case with freshly and
// independently allocated devices, allocators, and host buffers.

TEST_CASE("CPU conformance: storage and host transfers for every leaf type") {
    CpuDevices devices;
    iom_conformance::run_storage_and_transfer_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: standard storage oracle covers every leaf width and padded shape") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            oracle, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("CPU conformance: storage oracle identifies perturbed transfer map") {
    CpuDevices devices;
    const std::span<const iom::DataType> supported =
            devices.candidate->supported_data_types();
    REQUIRE(supported.size() >= 1);
    const std::span<const iom::DataType> one_type =
            supported.subspan(0, 1);

    iom_conformance::CpuStorageOracle direct;
    iom_conformance::PermutingStorageOracle perturbed(
            direct, iom_conformance::swap_first_adjacent_slots);
    CHECK_FALSE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, perturbed, &devices.gate,
            false, false));

    iom_conformance::PermutingStorageOracle identity(
            direct, [](std::size_t logical) { return logical; });
    REQUIRE(iom_conformance::run_storage_oracle_conformance(
            devices.conformance(), one_type, identity, &devices.gate));
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: asynchronous copies against the CPU reference") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::run_async_copy_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate, &oracle);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: copy validation fails before writes and sequences") {
    CpuDevices devices;
    iom_conformance::run_copy_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: common binary validation and lifetime policy") {
    CpuDevices devices;
    iom_conformance::run_binary_request_conformance(
            devices.conformance(), nullptr);
    iom_conformance::run_binary_rank_boundary_conformance(
            *devices.candidate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: all binary operations through the real queue") {
    CpuDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, nullptr, true, true);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: binary MUL SUB and floating DIV values through real queue") {
    CpuDevices devices;
    for (const auto operation : {
                 iom_conformance::BinaryOperation::add,
                 iom_conformance::BinaryOperation::mul,
                 iom_conformance::BinaryOperation::sub,
                 iom_conformance::BinaryOperation::div}) {
        iom_conformance::run_binary_value_conformance(
                *devices.candidate, operation);
    }
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: transfer failures keep metadata and ownership") {
    CpuDevices devices;
    iom_conformance::run_transfer_error_conformance(
            devices.conformance(), devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: deferred queue lifetime and stability") {
    CpuDevices devices;
    iom_conformance::run_lifetime_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: binary operations are supported") {
    CpuDevices devices;
    iom_conformance::run_compute_capability_conformance(
            *devices.candidate, devices.candidate->supported_data_types(),
            &devices.gate, "CPU", true, true);
    CHECK_FALSE(devices.gate.armed());
}

// CPU's declared embedding expectation covers the complete 23-payload/12-index
// matrix and the fixed `{0, 1}` scratch policy.
constexpr iom_conformance::EmbeddingDeclaration kCpuEmbeddingDeclaration{
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom_conformance::kEmbeddingPayloadSpan,
        iom_conformance::kEmbeddingIdSpan,
        iom::WorkspaceRequirements{0, 1}};

TEST_CASE("CPU conformance: embedding lookup reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::run_embedding_conformance(
            devices.conformance(), kCpuEmbeddingDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
    // The declared zero requirement is the whole CPU scratch contract: a
    // positive CPU raw workspace stays impossible, so no embedding request can
    // ever be handed positive CPU scratch.
    CHECK_THROWS_AS(
            devices.candidate->create_workspace(1), std::invalid_argument);
}

// CPU's declared linear expectation: the complete 21-leaf applicable matrix,
// one scalar recurrence covering every applicable leaf (including BF16, whose
// CPU path is that same recurrence), and the fixed `{0, 1}` scratch path. The
// CPU port implements that whole matrix, so the implemented span is the
// complete applicable leaf span and every declared leaf is compared
// numerically against the independent reference.
const iom_conformance::LinearDeclaration kCpuLinearDeclaration{
        iom_conformance::kLinearLeafSpan,
        iom_conformance::kLinearLeafSpan,
        iom_conformance::kNoLinearSpan,
        iom_conformance::kLinearLeafSpan,
        {},
        iom_conformance::LinearWorkspacePath::zero,
        iom_conformance::LinearWorkspacePath::zero};

TEST_CASE("CPU conformance: linear projection reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::run_linear_conformance(
            devices.conformance(), kCpuLinearDeclaration, &devices.gate,
            &oracle);
    CHECK_FALSE(devices.gate.armed());
}

// CPU's declared RMSNorm expectation: the nine applicable signed floating
// leaves, the frozen `{0, 1}` workspace requirement, and positive execution
// through the real queue, with host observation of the logical results and of
// deliberately poisoned tile padding.
TEST_CASE("CPU conformance: RMSNorm reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::CpuStorageOracle oracle;
    iom_conformance::RmsNormConformanceConfig config{
            devices.conformance(),
            iom_conformance::kRmsNormAllLeafSpan,
            &devices.gate,
            &oracle};
    // The CPU backend exposes no testing seam at all: `src/cpu` carries no
    // fault-injection hook, and a CPU RMSNorm request runs to completion on the
    // shared FIFO host worker, so no existing seam can produce an accepted
    // CPU RMSNorm failure. The harness records that gap explicitly. The minimal
    // test-only seam that would close it is a `cpu_detail` fault consumed
    // inside the enqueued host task (after acceptance, before the row loop)
    // under an `IOM_ENABLE_TESTING` guard.
    config.native_failure = iom_conformance::RmsNormNativeFailureSeam{
            {},
            {},
            "cpu_detail",
            "src/cpu exposes no fault-injection seam; an accepted CPU RMSNorm "
            "request cannot be made to fail after acceptance"};
    iom_conformance::run_rmsnorm_conformance(config);
    CHECK_FALSE(devices.gate.armed());
}
// The CPU port covers the complete nine-leaf RoPE span. The runner owns the
// independent split-half oracle and exercises transformed views, tails,
// poisoned padding, absolute positions, and the fixed zero-workspace query
// through the real asynchronous CPU queue.
TEST_CASE("CPU conformance: RoPE reference, admission, and lifetime") {
    CpuDevices devices;
    iom_conformance::rope_reference::run_rope_conformance(
            *devices.candidate,
            iom_conformance::rope_reference::kRopeCpuExpectedSupported);
    CHECK_FALSE(devices.gate.armed());
}
TEST_CASE("CPU conformance: RoPE keeps inapplicable leaves explicitly unsupported") {
    CpuDevices devices;
    auto queue = devices.candidate->create_ops();
    for (const iom::DataType data_type
            : iom_conformance::rope_reference::kRopeUnsupportedDataTypes) {
        const iom::TensorSpec leaf_shape{
                iom::TensorShape{{1, 1, 1, 16}}, data_type};
        auto source = devices.candidate->create_tensor(leaf_shape);
        auto destination = devices.candidate->create_tensor(leaf_shape);
        CHECK_EQ(
                queue->rope(source->view(), destination->view(), 0, 1.0),
                iom::to_oid(iom::OidError::Unsupported));
        CHECK_THROWS_AS(
                (void)queue->rope_workspace_requirements(
                        source->view(), destination->view(), 0, 1.0),
                std::runtime_error);
    }
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU RoPE retains owners and preserves FIFO ordering for temporary views") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 18}}, iom::DataType::F32};
    std::unique_ptr<iom::Tensor> source =
            devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> first =
            devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> second =
            devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0xC0DE);
    const std::vector<std::byte> zero(
            spec.logical_nbytes(), std::byte{0});
    iom_conformance::copy_from_host(source->view(), pattern);
    iom_conformance::copy_from_host(first->view(), zero);
    iom_conformance::copy_from_host(second->view(), zero);

    const iom::TensorView first_slice = first->view().slice(0, 0, 1);
    const iom::TensorView second_slice = second->view().slice(0, 0, 1);
    const std::size_t slice_bytes = first_slice.spec().logical_nbytes();
    const std::vector<std::byte> expected_slice(
            pattern.begin(),
            pattern.begin() + static_cast<std::ptrdiff_t>(slice_bytes));

    auto queue = devices.candidate->create_ops();
    const iom::oid rope_token =
            submit_temporary_rope(*queue, *source, *first);
    REQUIRE(iom::oid_is_token(rope_token));
    // The worker must retain source's owner registration after this reset.
    source.reset();

    const iom::oid copy_token = queue->copy(
            first_slice,
            const_cast<iom::TensorView&>(
                    static_cast<const iom::TensorView&>(second_slice)));
    REQUIRE(iom::oid_is_token(copy_token));
    REQUIRE_NOTHROW(queue->wait(copy_token));
    CHECK_NOTHROW(queue->wait(rope_token));
    CHECK_NOTHROW(queue->wait(copy_token));

    const std::vector<std::byte> observed_first =
            iom_conformance::read_logical(first_slice);
    const std::vector<std::byte> observed_second =
            iom_conformance::read_logical(second_slice);
    CHECK(observed_second == observed_first);

    // Position zero is an encoded-bit identity. Checking each head's first
    // logical row also proves that the temporary-view slice addressed the
    // expected independent head planes rather than a contiguous owner base.
    const std::size_t rank = first_slice.spec().shape.rank();
    const std::size_t heads = first_slice.spec().shape.dimension(rank - 3);
    const std::size_t rows = first_slice.spec().shape.dimension(rank - 2);
    const std::size_t width = first_slice.spec().shape.dimension(rank - 1);
    const std::size_t row_bytes = width * sizeof(float);
    const std::size_t head_bytes = rows * row_bytes;
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t offset = head * head_bytes;
        for (std::size_t byte = 0; byte < row_bytes; ++byte) {
            CHECK_EQ(observed_first[offset + byte], expected_slice[offset + byte]);
        }
    }
    CHECK_FALSE(devices.gate.armed());
    // CPU has no fault-injection seam for an accepted worker failure; repeat
    // waits above cover the successful retained-completion path.
}



TEST_CASE("CPU conformance: full shared suite composes every shared case") {
    CpuDevices devices;
    iom_conformance::run_backend_conformance(
            devices.conformance(),
            devices.candidate->supported_data_types().subspan(0, 1),
            &devices.gate, nullptr, true, true);
    CHECK_FALSE(devices.gate.armed());
}

// The CPU reference instantiation of the shared model-loading scenario. Each
// synthetic checkpoint is loaded through the production configuration and
// mapped-source API, realized on the CPU reference and candidate devices, and
// read back bit-for-bit against the fixture's independent role bytes. The
// candidate binding reports `{0, 1}`, so it realizes with the default empty
// scratch and never calls the CPU's unsupported positive workspace factory.
TEST_CASE("CPU model loading realizes every published weight role of each synthetic checkpoint") {
    CpuDevices devices;
    iom_conformance::run_model_loading_conformance(devices.conformance());
}

// Opt-in real-checkpoint loading, compiled only with
// `IOM_TEST_REAL_MODEL_LOADING=ON`: the ordinary CPU suite reads no model
// directory, assumes no default path, and downloads nothing. The case is this
// driver's selected CPU device over the caller's own allocator, which is the
// CPU ownership path; no second full-weight binding is created for a reference
// payload.
#ifdef IOM_TEST_REAL_MODEL_LOADING
TEST_CASE("CPU real model loading") {
    CpuDevices devices;
    iom_conformance::run_real_model_loading(*devices.candidate);
}
#endif

// A deliberately perturbed candidate layout or view map must surface as a
// bit-for-bit logical-byte mismatch; the independently generated encodings
// make round-trip cancellation impossible.
TEST_CASE("conformance harness detects perturbed candidate bytes") {
    iom_conformance::TrafficGate gate;
    GatedAllocator reference_allocator(gate);
    GatedAllocator candidate_allocator(gate);
    auto reference = iom::make_cpu_device(reference_allocator);
    auto candidate = iom::make_cpu_device(candidate_allocator);

    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::U8};
    auto reference_tensor = reference->create_tensor(spec);
    auto candidate_tensor = candidate->create_tensor(spec);

    const std::vector<std::byte> seeded = iom_conformance::encode_logical(spec, 7);
    iom_conformance::copy_from_host(reference_tensor->view(), seeded);
    iom_conformance::copy_from_host(candidate_tensor->view(), seeded);
    REQUIRE_FALSE(iom_conformance::first_logical_mismatch(
            candidate_tensor->view(), seeded)
                          .has_value());

    // Flip one byte of the candidate storage at the shared layout helper's
    // slot for logical element zero.
    const std::size_t element_zero_slot = iom::detail::standard_layout_slot(
            spec, iom_conformance::span_of({0, 0, 0, 0}));
    auto* storage =
            static_cast<std::byte*>(candidate_tensor->view().native_handle());
    storage[element_zero_slot] ^= std::byte{0xFF};

    const std::optional<std::size_t> layout_mismatch =
            iom_conformance::first_logical_mismatch(
                    candidate_tensor->view(), seeded);
    REQUIRE(layout_mismatch.has_value());
    CHECK_EQ(*layout_mismatch, element_zero_slot);

    iom::TensorView stepped_window = candidate_tensor->view().slice(0, 1, 1);
    const iom::TensorView earlier_window =
            candidate_tensor->view().slice(0, 0, 1);
    const std::vector<std::byte> stepped_pattern =
            iom_conformance::encode_logical(stepped_window.spec(), 8);
    const std::vector<std::byte> earlier_pattern =
            iom_conformance::encode_logical(earlier_window.spec(), 9);
    iom_conformance::copy_from_host(stepped_window, stepped_pattern);
    CHECK_FALSE(iom_conformance::first_logical_mismatch(
            stepped_window, stepped_pattern)
                        .has_value());
    CHECK(iom_conformance::first_logical_mismatch(stepped_window, earlier_pattern)
                  .has_value());
}

TEST_CASE("CPU conformance: tile-blocked copy preserves logical and physical window scope") {
    CpuDevices devices;
    const iom::TensorSpec owner_spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    auto reference = devices.candidate->create_tensor(owner_spec);
    auto candidate = devices.candidate->create_tensor(owner_spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(owner_spec, 0x1234);
    iom_conformance::copy_from_host(reference->view(), pattern);
    std::memset(
            candidate->view().native_handle(), 0,
            owner_spec.tiled_storage_nbytes());

    const iom::TensorView source =
            reference->view().slice(0, 0, 2, 2);
    iom::TensorView destination =
            candidate->view().slice(0, 1, 2, 1);
    const std::vector<std::byte> expected =
            iom_conformance::read_logical(source);
    const std::size_t candidate_traffic =
            devices.candidate_allocator.traffic();
    devices.gate.setup_complete();
    auto queue = devices.candidate->create_ops();
    const iom::oid token = queue->copy(source, destination);
    queue->wait(token);
    CHECK_EQ(
            devices.candidate_allocator.traffic(), candidate_traffic);
    devices.gate.case_complete();

    iom_conformance::require_logical_bytes(
            destination, expected, "tile-blocked window copy");
    require_storage_outside_view_is_zero(owner_spec, destination);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: blocked byte-aligned copy matches `std::memcpy` byte-for-byte") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{4096, 4096}}, iom::DataType::F32};
    auto reference = devices.candidate->create_tensor(spec);
    auto candidate = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x5678);
    iom_conformance::copy_from_host(reference->view(), pattern);
    std::memset(
            candidate->view().native_handle(), 0,
            spec.tiled_storage_nbytes());

    const std::size_t candidate_traffic =
            devices.candidate_allocator.traffic();
    devices.gate.setup_complete();
    auto queue = devices.candidate->create_ops();
    const iom::oid token =
            queue->copy(reference->view(), candidate->view());
    queue->wait(token);
    CHECK_EQ(
            devices.candidate_allocator.traffic(), candidate_traffic);
    devices.gate.case_complete();

    CHECK_EQ(
            std::memcmp(
                    reference->view().native_handle(),
                    candidate->view().native_handle(),
                    spec.tiled_storage_nbytes()),
            0);
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: blocked sub-byte copy preserves LSB-first packing and zero tail bits") {
    CpuDevices devices;
    for (const iom::DataType data_type : {
                 iom::DataType::I2,
                 iom::DataType::F6_E2M3,
                 iom::DataType::F6_E3M2}) {
        const iom::TensorSpec spec{
                iom::TensorShape{{1, 17}}, data_type};
        auto reference = devices.candidate->create_tensor(spec);
        auto candidate = devices.candidate->create_tensor(spec);
        const std::vector<std::byte> pattern =
                iom_conformance::encode_logical(spec, 0x9ABC);
        iom_conformance::copy_from_host(reference->view(), pattern);
        std::memset(
                candidate->view().native_handle(), 0,
                spec.tiled_storage_nbytes());

        const std::size_t candidate_traffic =
                devices.candidate_allocator.traffic();
        devices.gate.setup_complete();
        auto queue = devices.candidate->create_ops();
        const iom::oid token =
                queue->copy(reference->view(), candidate->view());
        queue->wait(token);
        CHECK_EQ(
                devices.candidate_allocator.traffic(), candidate_traffic);
        devices.gate.case_complete();

        iom_conformance::require_logical_bytes(
                candidate->view(), pattern, "sub-byte blocked copy");
        require_subbyte_storage_scope_is_zero(
                spec, candidate->view());
        CHECK_FALSE(devices.gate.armed());
    }
}

TEST_CASE("CPU conformance: blocked copy walks lockstep planes without scratch allocation") {
    CpuDevices devices;
    const iom::TensorSpec owner_spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    auto reference = devices.candidate->create_tensor(owner_spec);
    auto candidate = devices.candidate->create_tensor(owner_spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(owner_spec, 0xDEF0);
    iom_conformance::copy_from_host(reference->view(), pattern);
    std::memset(
            candidate->view().native_handle(), 0,
            owner_spec.tiled_storage_nbytes());

    const iom::TensorView source =
            reference->view().slice(0, 0, 2, 2);
    iom::TensorView destination =
            candidate->view().slice(0, 1, 2, 1);
    const std::vector<std::byte> expected =
            iom_conformance::read_logical(source);
    const std::size_t candidate_traffic =
            devices.candidate_allocator.traffic();
    devices.gate.setup_complete();
    auto queue = devices.candidate->create_ops();
    const iom::oid token = queue->copy(source, destination);
    queue->wait(token);
    CHECK_EQ(
            devices.candidate_allocator.traffic(), candidate_traffic);
    devices.gate.case_complete();

    iom_conformance::require_logical_bytes(
            destination, expected, "lockstep plane copy");
    CHECK_FALSE(devices.gate.armed());
}

namespace {
// Submits the copy from a frame that returns before the caller waits: the
// derived-view temporaries die when this function returns, so a queue that
// stored their addresses would leave the worker dereferencing dead stack
// storage. No named local binds either view; `copy`'s destination
// parameter is a non-const reference, so the rvalue destination view is
// bound through const_cast — the view object is never modified, the worker
// writes through the owner's storage.
iom::oid submit_temporary_copy(
        iom::DeviceOps& queue, iom::Tensor& t, iom::Tensor& u,
        std::size_t half) {
    return queue.copy(
            t.view().slice(0, 0, half),
            const_cast<iom::TensorView&>(
                    static_cast<const iom::TensorView&>(
                            u.view().slice(0, 0, half))));
}
}  // namespace

TEST_CASE("CPU copy survives derived-view temporaries") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{4, 3, 17, 33}}, iom::DataType::U8};
    std::unique_ptr<iom::Tensor> t = devices.candidate->create_tensor(spec);
    std::unique_ptr<iom::Tensor> u = devices.candidate->create_tensor(spec);
    const std::vector<std::byte> pattern =
            iom_conformance::encode_logical(spec, 0x5A7C);
    iom_conformance::copy_from_host(t->view(), pattern);
    std::memset(
            u->view().native_handle(), 0, spec.tiled_storage_nbytes());

    auto queue = devices.candidate->create_ops();
    const std::vector<std::size_t> dims = {4, 3, 17, 33};
    const iom::oid token =
            submit_temporary_copy(*queue, *t, *u, dims[0] / 2);
    REQUIRE_NOTHROW(queue->wait(token));

    const iom::TensorView expected_view = t->view().slice(0, 0, dims[0] / 2);
    const iom::TensorView actual_view = u->view().slice(0, 0, dims[0] / 2);
    const std::vector<std::byte> expected =
            iom_conformance::read_logical(expected_view);
    iom_conformance::require_logical_bytes(
            actual_view, expected, "temporary-view copy");
    CHECK_FALSE(devices.gate.armed());
}

TEST_CASE("CPU conformance: raw workspace contract accepts only the empty owner") {
    CpuDevices devices;

    // The empty workspace is valid on every backend and performs no
    // allocation: the traffic gate stays silent through creation and
    // destruction.
    devices.gate.setup_complete();
    {
        const std::unique_ptr<iom::RawWorkspace> workspace =
                devices.candidate->create_workspace(0);
        REQUIRE(workspace != nullptr);
        CHECK(workspace->empty());
        CHECK_EQ(workspace->byte_size(), 0);
        CHECK(&workspace->device() == devices.candidate.get());
        CHECK(workspace->backend_kind() == iom::BackendKind::CPU);

        const iom::RawWorkspaceView view = workspace->view();
        CHECK(view.owner_identity() == workspace.get());
        CHECK(&view.device() == devices.candidate.get());
        CHECK_EQ(view.byte_size(), 0);
        CHECK_FALSE(view.empty());
    }
    devices.gate.case_complete();
    CHECK_FALSE(devices.gate.armed());

    // Positive scratch is unsupported CPU device storage: no dummy
    // native storage is manufactured.
    CHECK_THROWS_AS(
            devices.candidate->create_workspace(1), std::invalid_argument);
    CHECK_THROWS_AS(
            devices.candidate->create_workspace(32), std::invalid_argument);
    CHECK_THROWS_AS(
            devices.reference->create_workspace(4096),
            std::invalid_argument);
}

TEST_CASE("CPU conformance: workspace requirement queries report exact zero") {
    CpuDevices devices;
    const iom::TensorSpec spec{
            iom::TensorShape{{2, 3, 17, 33}}, iom::DataType::F32};
    auto lhs = devices.candidate->create_tensor(spec);
    auto rhs = devices.candidate->create_tensor(spec);
    auto out = devices.candidate->create_tensor(spec);
    auto queue = devices.candidate->create_ops();

    const iom::WorkspaceRequirements zero{0, 1};
    CHECK(queue->add_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->mul_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->sub_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(queue->div_workspace_requirements(
                  lhs->view(), rhs->view(), out->view()) == zero);
    CHECK(lhs->view().copy_from_host_workspace_requirements() == zero);
    CHECK(lhs->view().copy_to_host_workspace_requirements() == zero);

    // Validation matches the binary operation: foreign-device views are
    // invalid input before any requirement is reported.
    auto foreign = devices.foreign->create_tensor(spec);
    CHECK_THROWS_AS((void)
            queue->add_workspace_requirements(
                    foreign->view(), rhs->view(), out->view()),
            std::invalid_argument);

    // Pure queries cause no allocator, registration, lease, token, or
    // queue traffic.
    devices.gate.setup_complete();
    (void)queue->add_workspace_requirements(
            lhs->view(), rhs->view(), out->view());
    (void)lhs->view().copy_from_host_workspace_requirements();
    (void)lhs->view().copy_to_host_workspace_requirements();
    devices.gate.case_complete();
    CHECK_FALSE(devices.gate.armed());
}

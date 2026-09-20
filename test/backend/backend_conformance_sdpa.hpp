#pragma once

// Backend-neutral SDPA conformance.  Drivers provide only their device triple,
// the explicit SDPA capability span, optional storage observation, and an
// optional native accepted-failure seam.  All admission, numerical, ownership,
// workspace, and queue assertions live in this one matrix.

#include "backend_conformance_common.hpp"
#include "backend_conformance_oracle.hpp"
#include "backend_conformance_sdpa_reference.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::sdpa),
              iom::oid (iom::DeviceOps::*)(
                      const iom::TensorView&, const iom::TensorView&,
                      const iom::TensorView&, iom::TensorView&, std::size_t,
                      std::size_t, iom::RawWorkspaceView) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::sdpa_workspace_requirements),
              iom::WorkspaceRequirements (iom::DeviceOps::*)(
                      const iom::TensorView&, const iom::TensorView&,
                      const iom::TensorView&, const iom::TensorView&,
                      std::size_t, std::size_t)>);

// A backend-native fault must be consumed after the operation has returned a
// positive OID.  Empty seams are explicit coverage gaps, never fake failures.
struct SdpaNativeFailureSeam {
    std::function<void()> arm;
    std::function<void()> clear;
    std::string_view name;
    std::string_view unavailable_reason;

    [[nodiscard]] bool available() const noexcept {
        return static_cast<bool>(arm);
    }
};

// Optional logical hooks let a driver retain a native transfer/observation
// policy without teaching this harness about accelerator headers.  The native
// storage oracle below is used for full tiled-storage padding observations.
struct SdpaObservationHooks {
    std::function<void(iom::TensorView&, std::span<const std::byte>)>
            seed_logical;
    std::function<std::vector<std::byte>(const iom::TensorView&)>
            observe_logical;
};

struct SdpaConformanceConfig {
    ConformanceDevices devices;
    // A driver's own immutable statement of the leaves its SDPA port queues.
    // The current contract permits BF16 only; the CPU driver declares BF16
    // support, while accelerator drivers may keep an empty span until their
    // own ports land.
    std::span<const iom::DataType> supported_leaves;
    bool support_enabled = false;
    ConformanceObserver* observer = nullptr;
    AcceleratorStorageOracle* native_storage = nullptr;
    SdpaObservationHooks observation{};
    SdpaNativeFailureSeam native_failure{};
};

namespace sdpa_detail {

inline constexpr iom::oid invalid_oid = iom::to_oid(iom::OidError::InvalidArgument);
inline constexpr iom::oid unsupported_oid = iom::to_oid(iom::OidError::Unsupported);
inline constexpr iom::oid overflow_oid = iom::to_oid(iom::OidError::Overflow);
inline constexpr iom::oid resource_oid = iom::to_oid(iom::OidError::ResourceExhausted);


inline iom::TensorSpec q_spec(
        const SdpaReferenceCase& item, iom::DataType type) {
    std::vector<std::size_t> dimensions = item.leading_dimensions;
    dimensions.push_back(item.hq);
    dimensions.push_back(item.rows);
    dimensions.push_back(item.head_dim);
    return {iom::TensorShape{std::move(dimensions)}, type};
}

inline iom::TensorSpec kv_spec(
        const SdpaReferenceCase& item, iom::DataType type) {
    std::vector<std::size_t> dimensions = item.leading_dimensions;
    dimensions.push_back(item.hkv);
    dimensions.push_back(item.capacity);
    dimensions.push_back(item.head_dim);
    return {iom::TensorShape{std::move(dimensions)}, type};
}

inline iom::TensorSpec out_spec(
        const SdpaReferenceCase& item, iom::DataType type) {
    std::vector<std::size_t> dimensions = item.leading_dimensions;
    dimensions.push_back(item.rows);
    dimensions.push_back(item.hq * item.head_dim);
    return {iom::TensorShape{std::move(dimensions)}, type};
}

inline std::vector<std::byte> pack_bits(
        iom::DataType type, std::span<const std::uint64_t> values) {
    const std::size_t bits = sdpa_oracle::bit_width(type);
    if (bits == 0) {
        throw std::invalid_argument("SDPA test packing requires a floating leaf");
    }
    if (values.size() > std::numeric_limits<std::size_t>::max() / bits) {
        throw std::overflow_error("SDPA test packing overflows");
    }
    std::vector<std::byte> packed((values.size() * bits + 7) / 8, std::byte{0});
    for (std::size_t index = 0; index < values.size(); ++index) {
        write_bits(
                reinterpret_cast<unsigned char*>(packed.data()), index * bits,
                bits, values[index]);
    }
    return packed;
}

inline std::vector<std::byte> pack_output_sentinel(
        const iom::TensorSpec& spec, std::byte value) {
    std::vector<std::byte> bytes(spec.logical_nbytes(), value);
    const std::size_t bit_count =
            spec.shape.element_count() * sdpa_oracle::bit_width(spec.data_type);
    const std::size_t remainder = bit_count % 8;
    if (remainder != 0 && !bytes.empty()) {
        const unsigned int mask = (1u << remainder) - 1u;
        bytes.back() = std::byte{
                static_cast<unsigned char>(
                        std::to_integer<unsigned int>(bytes.back()) & mask)};
    }
    return bytes;
}

inline std::vector<std::byte> logical_case_bytes(
        iom::DataType type, std::span<const std::uint64_t> values) {
    return pack_bits(type, values);
}

inline std::vector<std::byte> case_output_bytes(
        const iom::TensorSpec& spec, std::byte value) {
    return pack_output_sentinel(spec, value);
}

inline std::vector<std::byte> physical_case_storage(
        const iom::TensorSpec& owner, const iom::TensorView& view,
        std::span<const std::byte> logical, std::uint64_t salt) {
    std::vector<std::byte> storage(
            owner.tiled_storage_nbytes(),
            static_cast<std::byte>(salt & 0xffu));
    apply_standard_tiled_view(view, owner, logical, storage);
    return storage;
}

inline void seed_logical(
        const SdpaConformanceConfig& config, iom::TensorView& view,
        const iom::TensorSpec& owner, std::span<const std::byte> logical,
        std::uint64_t physical_salt = 0) {
    if (config.native_storage != nullptr) {
        config.native_storage->set_owner_spec(owner);
        std::vector<std::byte> storage = physical_case_storage(
                owner, view, logical, physical_salt);
        config.native_storage->seed(view, storage);
    } else if (config.observation.seed_logical) {
        config.observation.seed_logical(view, logical);
    } else {
        copy_from_host(view, logical);
    }
}

inline std::vector<std::byte> observe_logical(
        const SdpaConformanceConfig& config, const iom::TensorView& view,
        const iom::TensorSpec& owner) {
    if (config.native_storage != nullptr) {
        config.native_storage->set_owner_spec(owner);
        const std::vector<std::byte> storage =
                config.native_storage->observe(view);
        return decode_standard_tiled_view(view, owner, storage);
    }
    if (config.observation.observe_logical) {
        return config.observation.observe_logical(view);
    }
    return read_logical(view);
}

class CaseWindow final {
public:
    explicit CaseWindow(ConformanceObserver* observer) : observer_(observer) {
        if (observer_ != nullptr) observer_->setup_complete();
    }

    CaseWindow(const CaseWindow&) = delete;
    CaseWindow& operator=(const CaseWindow&) = delete;

    void complete() noexcept {
        if (observer_ != nullptr && armed_) {
            observer_->case_complete();
            armed_ = false;
        }
    }

    ~CaseWindow() { complete(); }

private:
    ConformanceObserver* observer_ = nullptr;
    bool armed_ = true;
};

class QuantizationQualification final {
public:
    QuantizationQualification(
            iom::Tensor& q, iom::Tensor& k, iom::Tensor& v, iom::Tensor& out,
            iom::QuantizationFormat format)
            : specs_{
                      qualify(q, format), qualify(k, format), qualify(v, format),
                      qualify(out, format)} {}

    QuantizationQualification(const QuantizationQualification&) = delete;
    QuantizationQualification& operator=(const QuantizationQualification&) = delete;

    ~QuantizationQualification() {
        for (iom::TensorSpec* spec : specs_) {
            if (spec != nullptr) spec->quantization = iom::QuantizationFormat::NONE;
        }
    }

private:
    static iom::TensorSpec* qualify(
            iom::Tensor& owner, iom::QuantizationFormat format) {
        iom::TensorSpec& spec = const_cast<iom::TensorSpec&>(owner.view().spec());
        spec.quantization = format;
        return &spec;
    }

    std::array<iom::TensorSpec*, 4> specs_{};
};

struct OwnedOperands {
    std::unique_ptr<iom::Tensor> q;
    std::unique_ptr<iom::Tensor> k;
    std::unique_ptr<iom::Tensor> v;
    std::unique_ptr<iom::Tensor> out;

    OwnedOperands(
            iom::Device& device, const iom::TensorSpec& q_spec,
            const iom::TensorSpec& k_spec, const iom::TensorSpec& v_spec,
            const iom::TensorSpec& out_spec)
            : q(device.create_tensor(q_spec)), k(device.create_tensor(k_spec)),
              v(device.create_tensor(v_spec)), out(device.create_tensor(out_spec)) {}
};

inline OwnedOperands make_operands(
        iom::Device& device, const SdpaReferenceCase& item,
        iom::DataType type) {
    const iom::TensorSpec q = q_spec(item, type);
    const iom::TensorSpec k = kv_spec(item, type);
    const iom::TensorSpec out = out_spec(item, type);
    return {device, q, k, k, out};
}

inline void expect_query_exception(
        iom::DeviceOps& queue, const iom::TensorView& q,
        const iom::TensorView& k, const iom::TensorView& v,
        const iom::TensorView& out, std::size_t a, std::size_t L,
        bool unsupported, bool overflow) {
    bool caught = false;
    std::string message;
    try {
        (void)queue.sdpa_workspace_requirements(q, k, v, out, a, L);
    } catch (const std::overflow_error& error) {
        caught = overflow;
        message = error.what();
    } catch (const std::invalid_argument& error) {
        caught = !unsupported && !overflow;
        message = error.what();
    } catch (const std::runtime_error& error) {
        caught = unsupported;
        message = error.what();
    }
    CAPTURE(message);
    CHECK_MESSAGE(caught, "SDPA requirement query returned the wrong failure category");
}

inline void expect_submission_failure(
        iom::DeviceOps& queue, const iom::TensorView& q,
        const iom::TensorView& k, const iom::TensorView& v,
        iom::TensorView& out, std::size_t a, std::size_t L,
        iom::oid expected) {
    CHECK_EQ(queue.sdpa(q, k, v, out, a, L), expected);
}

inline void prove_no_admission_effect(
        iom::DeviceOps& queue, iom::TensorView& out,
        std::span<const std::byte> before, std::string_view context) {
    (void)before;
    (void)context;
    const iom::oid probe = queue.copy(out, out);
    REQUIRE(iom::oid_is_token(probe));
    CHECK_EQ(token_sequence(probe), std::uint64_t{1});
    CHECK_NOTHROW(queue.wait(probe));
    CHECK_NOTHROW(queue.wait(probe));
}

inline void check_reference_output(
        iom::DataType type, std::span<const std::byte> logical,
        std::span<const SdpaReferenceValue> expected,
        std::string_view context) {
    const std::size_t bits = sdpa_oracle::bit_width(type);
    REQUIRE_EQ(logical.size(), (expected.size() * bits + 7) / 8);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::uint64_t actual = read_storage_bits(
                logical.data(), index * bits, bits);
        CHECK_MESSAGE(
                matches(type, actual, expected[index]),
                context << ": output mismatch at logical index " << index);
    }
}

inline void mutate_unread_values(
        SdpaReferenceCase& item, std::uint64_t salt) {
    const std::size_t planes = sdpa_oracle::leading_plane_count(
            std::span<const std::size_t>(item.leading_dimensions));
    const std::uint64_t value = sdpa_oracle::value_bits(
            item.data_type, salt == 0 ? 123.0 : -77.0);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < item.hkv; ++head) {
            for (std::size_t token = 0; token < item.capacity; ++token) {
                const bool future = item.kind
                                            == SdpaReferenceCaseKind::future_token_perturbation
                                    && token > item.position;
                const bool tail = item.kind
                                          == SdpaReferenceCaseKind::capacity_tail_perturbation
                                  && token >= item.length;
                if (!future && !tail) continue;
                for (std::size_t feature = 0; feature < item.head_dim; ++feature) {
                    const std::size_t index = sdpa_oracle::kv_offset(
                            item, plane, head, token, feature);
                    item.k_bits[index] = value;
                    item.v_bits[index] = value;
                }
            }
        }
    }
}

inline std::vector<std::byte> run_reference_case(
        const SdpaConformanceConfig& config, const SdpaReferenceCase& item,
        std::string_view label, std::uint64_t physical_salt = 0) {
    const std::vector<SdpaReferenceValue> expected = evaluate(item);
    OwnedOperands operands = make_operands(
            config.devices.candidate, item, item.data_type);
    const iom::TensorSpec q_owner = operands.q->view().spec();
    const iom::TensorSpec k_owner = operands.k->view().spec();
    const iom::TensorSpec v_owner = operands.v->view().spec();
    const iom::TensorSpec out_owner = operands.out->view().spec();
    const std::vector<std::byte> q_bytes = logical_case_bytes(
            item.data_type, item.q_bits);
    const std::vector<std::byte> k_bytes = logical_case_bytes(
            item.data_type, item.k_bits);
    const std::vector<std::byte> v_bytes = logical_case_bytes(
            item.data_type, item.v_bits);
    const std::vector<std::byte> out_before =
            case_output_bytes(out_owner, std::byte{0xA5});
    seed_logical(config, operands.q->view(), q_owner, q_bytes, physical_salt);
    seed_logical(config, operands.k->view(), k_owner, k_bytes, physical_salt);
    seed_logical(config, operands.v->view(), v_owner, v_bytes, physical_salt);
    copy_from_host(operands.out->view(), out_before);

    auto queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements first =
            queue->sdpa_workspace_requirements(
                    operands.q->view(), operands.k->view(), operands.v->view(),
                    operands.out->view(), item.position, item.length);
    CHECK_EQ(
            first,
            queue->sdpa_workspace_requirements(
                    operands.q->view(), operands.k->view(), operands.v->view(),
                    operands.out->view(), item.position, item.length));
    std::unique_ptr<iom::RawWorkspace> workspace;
    if (first.bytes != 0) {
        workspace = config.devices.candidate.create_workspace(first.bytes);
        REQUIRE(workspace != nullptr);
    } else {
        CHECK_EQ(first.alignment, std::size_t{1});
    }
    CaseWindow window(config.observer);
    const iom::oid token = workspace
            ? queue->sdpa(
                      operands.q->view(), operands.k->view(), operands.v->view(),
                      operands.out->view(), item.position, item.length,
                      workspace->view())
            : queue->sdpa(
                      operands.q->view(), operands.k->view(), operands.v->view(),
                      operands.out->view(), item.position, item.length);
    REQUIRE_MESSAGE(iom::oid_is_token(token), label << ": supported SDPA did not accept");
    CHECK_NOTHROW(queue->wait(token));
    CHECK_NOTHROW(queue->wait(token));
    window.complete();

    const std::vector<std::byte> observed = observe_logical(
            config, operands.out->view(), out_owner);
    check_reference_output(item.data_type, observed, expected, label);
    return observed;
}
// Exercise independently transformed leading planes and K/V head strides.
// The K owner selects every other head plane while the V owner selects a
// contiguous head window, then both views are permuted over their leading
// dimensions. This keeps the logical fixture unchanged while ensuring the
// worker must honor each captured view's own plane strides.
inline void run_transformed_reference_case(
        const SdpaConformanceConfig& config) {
    const SdpaReferenceCase item =
            sdpa_oracle::make_mixed_gqa_case(iom::DataType::BF16);
    const std::vector<SdpaReferenceValue> expected = evaluate(item);
    const iom::TensorSpec q_owner_spec = q_spec(item, item.data_type);
    const iom::TensorSpec out_owner_spec = out_spec(item, item.data_type);
    std::vector<std::size_t> expanded_dimensions = item.leading_dimensions;
    expanded_dimensions.push_back(item.hkv * 2);
    expanded_dimensions.push_back(item.capacity);
    expanded_dimensions.push_back(item.head_dim);
    const iom::TensorSpec expanded_kv_spec{
            iom::TensorShape{std::move(expanded_dimensions)},
            item.data_type};
    auto q_owner = config.devices.candidate.create_tensor(q_owner_spec);
    auto k_owner = config.devices.candidate.create_tensor(expanded_kv_spec);
    auto v_owner = config.devices.candidate.create_tensor(expanded_kv_spec);
    auto out_owner = config.devices.candidate.create_tensor(out_owner_spec);

    const std::size_t head_axis = item.leading_dimensions.size();
    const std::array<std::size_t, 3> qkv_order{1, 0, 2};
    const std::array<std::size_t, 2> out_order{1, 0};
    iom::TensorView k_slice =
            k_owner->view().slice(head_axis, 0, item.hkv, 2);
    iom::TensorView v_slice =
            v_owner->view().slice(head_axis, 0, item.hkv, 1);
    iom::TensorView q_view = q_owner->view().permute(
            std::span<const std::size_t>(qkv_order));
    iom::TensorView k_view = k_slice.permute(
            std::span<const std::size_t>(qkv_order));
    iom::TensorView v_view = v_slice.permute(
            std::span<const std::size_t>(qkv_order));
    iom::TensorView out_view = out_owner->view().permute(
            std::span<const std::size_t>(out_order));

    seed_logical(
            config, q_view, q_owner_spec,
            logical_case_bytes(item.data_type, item.q_bits));
    seed_logical(
            config, k_view, expanded_kv_spec,
            logical_case_bytes(item.data_type, item.k_bits));
    seed_logical(
            config, v_view, expanded_kv_spec,
            logical_case_bytes(item.data_type, item.v_bits));
    copy_from_host(
            out_view, case_output_bytes(out_owner_spec, std::byte{0xA5}));

    auto queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements requirements =
            queue->sdpa_workspace_requirements(
                    q_view, k_view, v_view, out_view, item.position,
                    item.length);
    CHECK_EQ(
            requirements,
            queue->sdpa_workspace_requirements(
                    q_view, k_view, v_view, out_view, item.position,
                    item.length));
    auto workspace = config.devices.candidate.create_workspace(
            requirements.bytes);
    REQUIRE(workspace != nullptr);
    CaseWindow window(config.observer);
    const iom::oid token = queue->sdpa(
            q_view, k_view, v_view, out_view, item.position, item.length,
            workspace->view());
    REQUIRE_MESSAGE(
            iom::oid_is_token(token),
            "transformed SDPA fixture did not accept");
    CHECK_NOTHROW(queue->wait(token));
    CHECK_NOTHROW(queue->wait(token));
    window.complete();

    const std::vector<std::byte> observed =
            observe_logical(config, out_view, out_owner_spec);
    check_reference_output(
            item.data_type, observed, expected,
            "transformed K/V head strides and leading permutation");
}


inline void run_supported_numeric_matrix(
        const SdpaConformanceConfig& config) {
    const bool has_observation =
            config.native_storage != nullptr
            || static_cast<bool>(config.observation.observe_logical)
            || static_cast<bool>(config.observation.seed_logical);
    REQUIRE(has_observation);
    for (const iom::DataType type : config.supported_leaves) {
        for (const SdpaReferenceCase& item : sdpa_reference_cases(type)) {
            CAPTURE(sdpa_reference_case_label(item));
            const std::string label = sdpa_reference_case_label(item);
            const std::vector<std::byte> baseline =
                    run_reference_case(config, item, label);
            if (item.kind == SdpaReferenceCaseKind::future_token_perturbation
                    || item.kind
                               == SdpaReferenceCaseKind::capacity_tail_perturbation) {
                SdpaReferenceCase perturbed = item;
                mutate_unread_values(perturbed, 1);
                const std::vector<std::byte> changed = run_reference_case(
                        config, perturbed, label + " unread perturbation");
                CHECK_MESSAGE(
                        baseline == changed,
                        label << ": unread future/capacity storage changed output");
            }
            if (item.kind == SdpaReferenceCaseKind::physical_padding_perturbation) {
                const std::vector<std::byte> changed = run_reference_case(
                        config, item, label + " physical padding perturbation",
                        item.physical_padding_salt);
                CHECK_MESSAGE(
                        baseline == changed,
                        label << ": physical padding changed output");
            }
        }

        // Full causal recomputation and one-row cached/incremental calls must
        // agree bit-for-bit for every row and every grouped query head.
        const SdpaReferenceCase full =
                sdpa_oracle::make_cached_incremental_case(type);
        const std::vector<std::byte> full_observed =
                run_reference_case(config, full, "cached/full reference");
        const std::size_t bits = sdpa_oracle::bit_width(type);
        const std::size_t row_values = full.hq * full.head_dim;
        for (std::size_t row = 0; row < full.rows; ++row) {
            const SdpaReferenceCase incremental = sdpa_incremental_slice(full, row);
            const std::vector<std::byte> one = run_reference_case(
                    config, incremental, "cached/incremental row");
            for (std::size_t index = 0; index < row_values; ++index) {
                const std::uint64_t expected = read_storage_bits(
                        full_observed.data(),
                        (row * row_values + index) * bits, bits);
                const std::uint64_t actual = read_storage_bits(
                        one.data(), index * bits, bits);
                CHECK_EQ(actual, expected);
            }
        }
        if (type == iom::DataType::BF16) {
            run_transformed_reference_case(config);
        }
    }
}

inline void run_supported_workspace_conformance(
        const SdpaConformanceConfig& config) {
    const SdpaReferenceCase item = sdpa_oracle::make_ones_case(
            iom::DataType::BF16);
    OwnedOperands operands = make_operands(
            config.devices.candidate, item, iom::DataType::BF16);
    const std::vector<std::byte> q_bytes = logical_case_bytes(
            item.data_type, item.q_bits);
    const std::vector<std::byte> k_bytes = logical_case_bytes(
            item.data_type, item.k_bits);
    const std::vector<std::byte> v_bytes = logical_case_bytes(
            item.data_type, item.v_bits);
    seed_logical(config, operands.q->view(), operands.q->view().spec(), q_bytes);
    seed_logical(config, operands.k->view(), operands.k->view().spec(), k_bytes);
    seed_logical(config, operands.v->view(), operands.v->view().spec(), v_bytes);
    copy_from_host(
            operands.out->view(),
            pack_output_sentinel(operands.out->view().spec(), std::byte{0x4D}));
    auto queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements requirements =
            queue->sdpa_workspace_requirements(
                    operands.q->view(), operands.k->view(), operands.v->view(),
                    operands.out->view(), item.position, item.length);
    CHECK_EQ(
            requirements,
            queue->sdpa_workspace_requirements(
                    operands.q->view(), operands.k->view(), operands.v->view(),
                    operands.out->view(), item.position, item.length));
    if (requirements.bytes == 0) {
        CHECK_EQ(requirements.alignment, std::size_t{1});
        return;
    }

    auto workspace = config.devices.candidate.create_workspace(requirements.bytes);
    REQUIRE(workspace != nullptr);
    const iom::RawWorkspaceView valid_workspace = workspace->view();
    auto short_workspace = config.devices.candidate.create_workspace(
            requirements.bytes);
    REQUIRE(short_workspace != nullptr);
    CaseWindow window(config.observer);
    const iom::oid first = queue->sdpa(
            operands.q->view(), operands.k->view(), operands.v->view(),
            operands.out->view(), item.position, item.length, valid_workspace);
    REQUIRE(iom::oid_is_token(first));

    // The same live range cannot be leased by accepted work a second time.
    const iom::oid overlap = queue->sdpa(
            operands.q->view(), operands.k->view(), operands.v->view(),
            operands.out->view(), item.position, item.length, valid_workspace);
    CHECK_EQ(overlap, resource_oid);
    CHECK_NOTHROW(queue->wait(first));
    CHECK_NOTHROW(queue->wait(first));

    // Exact capacity is accepted, while a shorter borrowed range is rejected
    // synchronously before another sequence is consumed.
    const iom::RawWorkspaceView short_view = short_workspace->view().subrange(
            0, requirements.bytes - 1);
    const iom::oid short_result = queue->sdpa(
            operands.q->view(), operands.k->view(), operands.v->view(),
            operands.out->view(), item.position, item.length, short_view);
    CHECK_EQ(short_result, invalid_oid);
    const iom::oid probe = queue->copy(operands.out->view(), operands.out->view());
    REQUIRE(iom::oid_is_token(probe));
    CHECK_NOTHROW(queue->wait(probe));
    window.complete();
}

inline void run_supported_failure_conformance(
        const SdpaConformanceConfig& config) {
    const SdpaNativeFailureSeam& seam = config.native_failure;
    if (!seam.available()) {
        REQUIRE_MESSAGE(
                !seam.unavailable_reason.empty(),
                "an unavailable SDPA accepted-failure seam must state why");
        std::printf(
                "sdpa-native-failure-record seam=%.*s coverage=unavailable reason=%.*s\n",
                static_cast<int>(seam.name.size()), seam.name.data(),
                static_cast<int>(seam.unavailable_reason.size()),
                seam.unavailable_reason.data());
        return;
    }

    const SdpaReferenceCase item = sdpa_oracle::make_ones_case(
            iom::DataType::BF16);
    const std::vector<SdpaReferenceValue> expected = evaluate(item);
    OwnedOperands operands = make_operands(
            config.devices.candidate, item, iom::DataType::BF16);
    seed_logical(config, operands.q->view(), operands.q->view().spec(),
                 logical_case_bytes(item.data_type, item.q_bits));
    seed_logical(config, operands.k->view(), operands.k->view().spec(),
                 logical_case_bytes(item.data_type, item.k_bits));
    seed_logical(config, operands.v->view(), operands.v->view().spec(),
                 logical_case_bytes(item.data_type, item.v_bits));
    copy_from_host(
            operands.out->view(),
            pack_output_sentinel(operands.out->view().spec(), std::byte{0x77}));
    auto queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements requirements =
            queue->sdpa_workspace_requirements(
                    operands.q->view(), operands.k->view(), operands.v->view(),
                    operands.out->view(), item.position, item.length);
    std::unique_ptr<iom::RawWorkspace> workspace;
    if (requirements.bytes != 0) {
        workspace = config.devices.candidate.create_workspace(requirements.bytes);
    }
    CaseWindow window(config.observer);
    const auto submit = [&]() {
        return workspace
                ? queue->sdpa(
                          operands.q->view(), operands.k->view(),
                          operands.v->view(), operands.out->view(),
                          item.position, item.length, workspace->view())
                : queue->sdpa(
                          operands.q->view(), operands.k->view(),
                          operands.v->view(), operands.out->view(),
                          item.position, item.length);
    };
    const iom::oid control = submit();
    REQUIRE(iom::oid_is_token(control));
    CHECK_NOTHROW(queue->wait(control));

    seam.arm();
    const iom::oid failed = submit();
    REQUIRE(iom::oid_is_token(failed));
    expect_repeated_runtime_failure(*queue, failed);
    seam.clear();

    const iom::oid recovered = submit();
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    window.complete();
    const std::vector<std::byte> output = observe_logical(
            config, operands.out->view(), operands.out->view().spec());
    check_reference_output(
            iom::DataType::BF16, output, expected, "SDPA recovered request");
}

inline void run_invalid_case(
        const SdpaConformanceConfig& config, OwnedOperands& operands,
        std::size_t a, std::size_t L, bool unsupported, bool overflow,
        iom::oid expected, std::string_view label,
        std::span<const std::byte> before_override = {},
        bool verify_output = true) {
    CAPTURE(label);
    std::vector<std::byte> before_storage;
    std::span<const std::byte> before = before_override;
    if (before.empty()) {
        before_storage = read_logical(operands.out->view());
        before = before_storage;
    }
    auto queue = config.devices.candidate.create_ops();
    iom::TensorView q = operands.q->view();
    iom::TensorView k = operands.k->view();
    iom::TensorView v = operands.v->view();
    iom::TensorView out = operands.out->view();
    CaseWindow window(config.observer);
    expect_query_exception(*queue, q, k, v, out, a, L, unsupported, overflow);
    expect_submission_failure(*queue, q, k, v, out, a, L, expected);
    prove_no_admission_effect(*queue, out, before, label);
    window.complete();
    if (verify_output) {
        require_logical_bytes(out, before, label);
    }
}

inline void run_unsupported_valid_case(
        const SdpaConformanceConfig& config, OwnedOperands& operands,
        std::size_t a, std::size_t L, std::string_view label) {
    CAPTURE(label);
    const std::byte sentinel =
            operands.out->view().spec().data_type == iom::DataType::BOOL
            ? std::byte{0x00}
            : std::byte{0xC3};
    const std::vector<std::byte> seeded =
            pack_output_sentinel(operands.out->view().spec(), sentinel);
    copy_from_host(operands.out->view(), seeded);
    const std::vector<std::byte> before =
            read_logical(operands.out->view());
    auto queue = config.devices.candidate.create_ops();
    iom::TensorView q = operands.q->view();
    iom::TensorView k = operands.k->view();
    iom::TensorView v = operands.v->view();
    iom::TensorView out = operands.out->view();
    CaseWindow window(config.observer);
    expect_query_exception(*queue, q, k, v, out, a, L, true, false);
    const std::unique_ptr<iom::RawWorkspace> empty_workspace =
            config.devices.candidate.create_workspace(0);
    REQUIRE(empty_workspace != nullptr);
    CHECK_EQ(
            queue->sdpa(q, k, v, out, a, L, empty_workspace->view()),
            unsupported_oid);
    CHECK_EQ(queue->sdpa(q, k, v, out, a, L), unsupported_oid);
    prove_no_admission_effect(*queue, out, before, label);
    window.complete();
    require_logical_bytes(out, before, label);
}

inline void run_sdpa_negative_matrix(const SdpaConformanceConfig& config) {
    const SdpaReferenceCase base =
            sdpa_oracle::make_mixed_gqa_case(iom::DataType::BF16);
    {
        OwnedOperands operands(
                config.devices.candidate,
                iom::TensorSpec{iom::TensorShape{{3, 5}},
                                iom::DataType::BF16},
                iom::TensorSpec{iom::TensorShape{{3, 5}},
                                iom::DataType::BF16},
                iom::TensorSpec{iom::TensorShape{{3, 5}},
                                iom::DataType::BF16},
                iom::TensorSpec{iom::TensorShape{{3, 20}},
                                iom::DataType::BF16});
        copy_from_host(
                operands.out->view(),
                pack_output_sentinel(operands.out->view().spec(), std::byte{0x01}));
        run_invalid_case(config, operands, 0, 1, false, false, invalid_oid,
                         "wrong SDPA ranks");
    }
    {
        const iom::TensorSpec q{
                iom::TensorShape{{2, 3, 5}}, iom::DataType::BF16};
        const iom::TensorSpec k{
                iom::TensorShape{{1, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec v{
                iom::TensorShape{{1, 6, 5}}, iom::DataType::BF16};
        const iom::TensorSpec out{
                iom::TensorShape{{3, 10}}, iom::DataType::BF16};
        OwnedOperands operands(config.devices.candidate, q, k, v, out);
        copy_from_host(
                operands.out->view(),
                pack_output_sentinel(operands.out->view().spec(), std::byte{0x02}));
        run_invalid_case(config, operands, 0, 1, false, false, invalid_oid,
                         "key/value inequality");
    }
    {
        const iom::TensorSpec q{
                iom::TensorShape{{2, 1, 3, 5}}, iom::DataType::BF16};
        const iom::TensorSpec k{
                iom::TensorShape{{3, 1, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec v{
                iom::TensorShape{{3, 1, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec out{
                iom::TensorShape{{2, 3, 10}}, iom::DataType::BF16};
        OwnedOperands operands(config.devices.candidate, q, k, v, out);
        copy_from_host(
                operands.out->view(),
                pack_output_sentinel(operands.out->view().spec(), std::byte{0x03}));
        run_invalid_case(config, operands, 0, 1, false, false, invalid_oid,
                         "mismatched leading tuples");
    }
    {
        const iom::TensorSpec q{
                iom::TensorShape{{3, 3, 5}}, iom::DataType::BF16};
        const iom::TensorSpec k{
                iom::TensorShape{{2, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec v{
                iom::TensorShape{{2, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec out{
                iom::TensorShape{{3, 15}}, iom::DataType::BF16};
        OwnedOperands operands(config.devices.candidate, q, k, v, out);
        copy_from_host(
                operands.out->view(),
                pack_output_sentinel(operands.out->view().spec(), std::byte{0x04}));
        run_invalid_case(config, operands, 0, 1, false, false, invalid_oid,
                         "non-divisible GQA heads");
    }
    {
        const iom::TensorSpec q{
                iom::TensorShape{{2, 3, 5}}, iom::DataType::BF16};
        const iom::TensorSpec k{
                iom::TensorShape{{1, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec v{
                iom::TensorShape{{1, 7, 5}}, iom::DataType::BF16};
        const iom::TensorSpec out{
                iom::TensorShape{{3, 11}}, iom::DataType::BF16};
        OwnedOperands operands(config.devices.candidate, q, k, v, out);
        copy_from_host(
                operands.out->view(),
                pack_output_sentinel(operands.out->view().spec(), std::byte{0x06}));
        run_invalid_case(config, operands, 0, 1, false, false, invalid_oid,
                         "wrong output width");
    }

    {
        OwnedOperands operands = make_operands(
                config.devices.candidate, base, iom::DataType::BF16);
        run_invalid_case(config, operands, 0, 0, false, false, invalid_oid,
                         "empty initialized visibility");
        run_invalid_case(config, operands, 0, base.capacity + 1, false, false,
                         invalid_oid, "initialized length exceeds capacity");
        run_invalid_case(config, operands, base.capacity, 1, false, false,
                         invalid_oid, "causal offset exceeds capacity");
        run_invalid_case(config, operands, base.capacity - 1, 1, false, false,
                         invalid_oid, "query rows exceed causal window");
    }
    {
        OwnedOperands candidate_operands = make_operands(
                config.devices.candidate, base, iom::DataType::BF16);
        auto foreign = config.devices.foreign.create_tensor(
                q_spec(base, iom::DataType::BF16));
        copy_from_host(
                candidate_operands.out->view(),
                pack_output_sentinel(
                        candidate_operands.out->view().spec(), std::byte{0x07}));
        const std::vector<std::byte> before =
                read_logical(candidate_operands.out->view());
        auto queue = config.devices.candidate.create_ops();
        iom::TensorView q = foreign->view();
        iom::TensorView k = candidate_operands.k->view();
        iom::TensorView v = candidate_operands.v->view();
        iom::TensorView out = candidate_operands.out->view();
        CaseWindow window(config.observer);
        expect_query_exception(*queue, q, k, v, out, base.position, base.length,
                               false, false);
        expect_submission_failure(
                *queue, q, k, v, out, base.position, base.length, invalid_oid);
        prove_no_admission_effect(*queue, out, before, "foreign-device SDPA view");
        window.complete();
        require_logical_bytes(out, before, "foreign-device SDPA view");
    }
    {
        OwnedOperands operands = make_operands(
                config.devices.candidate, base, iom::DataType::BF16);
        copy_from_host(
                operands.out->view(),
                pack_output_sentinel(operands.out->view().spec(), std::byte{0x08}));
        const std::vector<std::byte> before =
                read_logical(operands.out->view());
        {
            QuantizationQualification qualification(
                    *operands.q, *operands.k, *operands.v, *operands.out,
                    iom::QuantizationFormat::OCP_MXFP4);
            run_invalid_case(config, operands, base.position, base.length, true,
                             false, unsupported_oid, "non-NONE quantization",
                             before, false);
        }
        require_logical_bytes(
                operands.out->view(), before, "non-NONE quantization");
    }
    {
        const iom::TensorSpec tiny{
                iom::TensorShape{{1, 1, 1}}, iom::DataType::BF16};
        OwnedOperands operands(
                config.devices.candidate, tiny, tiny, tiny,
                iom::TensorSpec{iom::TensorShape{{1, 1}},
                                iom::DataType::BF16});
        iom::TensorView output_alias = operands.q->view().reshape_leading({});
        copy_from_host(
                operands.q->view(),
                pack_output_sentinel(operands.q->view().spec(), std::byte{0x09}));
        const std::vector<std::byte> before = read_logical(output_alias);
        auto queue = config.devices.candidate.create_ops();
        iom::TensorView k = operands.k->view();
        iom::TensorView v = operands.v->view();
        CaseWindow window(config.observer);
        expect_query_exception(
                *queue, operands.q->view(), k, v, output_alias, 0, 1, false,
                false);
        expect_submission_failure(
                *queue, operands.q->view(), k, v, output_alias, 0, 1,
                invalid_oid);
        prove_no_admission_effect(
                *queue, output_alias, before, "output/input overlap");
        window.complete();
        require_logical_bytes(output_alias, before, "output/input overlap");
    }
    {
        OwnedOperands operands = make_operands(
                config.devices.candidate, base, iom::DataType::BF16);
        auto& q_spec_mut = const_cast<iom::TensorSpec&>(operands.q->view().spec());
        auto& k_spec_mut = const_cast<iom::TensorSpec&>(operands.k->view().spec());
        auto& v_spec_mut = const_cast<iom::TensorSpec&>(operands.v->view().spec());
        auto& out_spec_mut = const_cast<iom::TensorSpec&>(operands.out->view().spec());
        const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
        q_spec_mut.shape = iom::TensorShape{{huge, 1, huge}};
        k_spec_mut.shape = iom::TensorShape{{1, huge, huge}};
        v_spec_mut.shape = iom::TensorShape{{1, huge, huge}};
        out_spec_mut.shape = iom::TensorShape{{1, 1}};
        auto queue = config.devices.candidate.create_ops();
        iom::TensorView q = operands.q->view();
        iom::TensorView k = operands.k->view();
        iom::TensorView v = operands.v->view();
        iom::TensorView out = operands.out->view();
        CAPTURE("overflow dimensions");
        CaseWindow window(config.observer);
        expect_query_exception(*queue, q, k, v, out, 0, 1, false, true);
        expect_submission_failure(*queue, q, k, v, out, 0, 1, overflow_oid);
        // Restore owner specifications before the no-admission probe and
        // before any backend destructor derives the allocation extent.
        q_spec_mut.shape = q_spec(base, iom::DataType::BF16).shape;
        k_spec_mut.shape = kv_spec(base, iom::DataType::BF16).shape;
        v_spec_mut.shape = kv_spec(base, iom::DataType::BF16).shape;
        out_spec_mut.shape = out_spec(base, iom::DataType::BF16).shape;
        const iom::oid probe = queue->copy(out, out);
        REQUIRE(iom::oid_is_token(probe));
        CHECK_EQ(token_sequence(probe), std::uint64_t{1});
        CHECK_NOTHROW(queue->wait(probe));
        window.complete();
    }

    for (const iom::DataType type : kSdpaCurrentUnsupportedDataTypes) {
        SdpaReferenceCase item = sdpa_oracle::make_causal_prefix_case(type);
        OwnedOperands operands = make_operands(
                config.devices.candidate, item, type);
        run_unsupported_valid_case(
                config, operands, item.position, item.length,
                std::string("current unsupported ")
                        + std::string(sdpa_oracle::leaf_name(type)));
    }
    for (const iom::DataType type : kSdpaInapplicableDataTypes) {
        // BOOL and integer leaves are valid storage leaves, but never SDPA
        // arithmetic leaves; the common facade must reject them explicitly.
        // The independent reference intentionally has no arithmetic for these
        // leaves; only the public admission path is exercised here.
        OwnedOperands operands(
                config.devices.candidate,
                iom::TensorSpec{iom::TensorShape{{1, 1, 1}}, type},
                iom::TensorSpec{iom::TensorShape{{1, 2, 1}}, type},
                iom::TensorSpec{iom::TensorShape{{1, 2, 1}}, type},
                iom::TensorSpec{iom::TensorShape{{1, 1}}, type});
        run_unsupported_valid_case(
                config, operands, 0, 1,
                std::string("inapplicable ")
                        + std::string(sdpa_oracle::leaf_name(type)));
    }

    // A zero-extented TensorShape is rejected before a tensor or queue exists.
    CHECK_THROWS_AS(
            (void)config.devices.candidate.create_tensor(
                    iom::TensorSpec{iom::TensorShape{{1, 0, 1}},
                                    iom::DataType::BF16}),
            std::invalid_argument);
}

}  // namespace sdpa_detail

inline void run_sdpa_conformance(const SdpaConformanceConfig& config) {
    if (config.support_enabled) {
        REQUIRE_FALSE(config.supported_leaves.empty());
        for (const iom::DataType type : config.supported_leaves) {
            REQUIRE_MESSAGE(
                    sdpa_currently_supported(type),
                    "SDPA support declaration contains a non-current leaf");
        }
        const bool has_observation =
                config.native_storage != nullptr
                || static_cast<bool>(config.observation.observe_logical)
                || static_cast<bool>(config.observation.seed_logical);
        REQUIRE(has_observation);
    } else {
        REQUIRE(config.supported_leaves.empty());
    }

    const SdpaReferenceSelfCheckReport report = sdpa_reference_self_check();
    CHECK(report.checks > 0);
    for (const std::string& failure : report.failures) {
        CHECK_MESSAGE(false, "independent SDPA oracle: " << failure);
    }
    REQUIRE(report.ok());

    sdpa_detail::run_sdpa_negative_matrix(config);
    if (config.support_enabled) {
        sdpa_detail::run_supported_numeric_matrix(config);
        sdpa_detail::run_supported_workspace_conformance(config);
        sdpa_detail::run_supported_failure_conformance(config);
    } else {
        // The current four drivers deliberately expose no SDPA capability. The
        // valid BF16 request and every recognized/non-applicable leaf therefore
        // remain explicit Unsupported, with no output, workspace, or sequence
        // side effects; the negative matrix above still exercises admission.
        const SdpaReferenceCase item =
                sdpa_oracle::make_causal_prefix_case(iom::DataType::BF16);
        sdpa_detail::OwnedOperands operands = sdpa_detail::make_operands(
                config.devices.candidate, item, iom::DataType::BF16);
        sdpa_detail::run_unsupported_valid_case(
                config, operands, item.position, item.length,
                "current BF16 SDPA capability");
    }
}

}  // namespace iom_conformance

#pragma once

// Backend-neutral conformance for DeviceOps::cache_append.  The fixture owns
// the logical reference model and all admission/lifetime cases; backend ports
// only provide their devices, capability span, native storage oracle, and an
// optional accepted-failure seam.

#include "backend_conformance_common.hpp"
#include "backend_conformance_oracle.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Independent scenario matrix and stable backend-port contract.
// ---------------------------------------------------------------------------

// The operation applies to the complete standard payload matrix.  TTNN's
// driver may omit only F8_E8M0; the runner checks that profile explicitly.

struct CacheAppendScenario {
    std::string label;
    std::vector<std::size_t> leading_dimensions;
    std::size_t heads = 0;
    std::size_t source_rows = 0;
    std::size_t destination_rows = 0;
    std::size_t features = 0;
    std::size_t offset = 0;

    [[nodiscard]] std::size_t rank() const noexcept {
        return leading_dimensions.size() + 3;
    }

    [[nodiscard]] std::vector<std::size_t> dimensions(
            bool destination) const {
        std::vector<std::size_t> result = leading_dimensions;
        result.push_back(heads);
        result.push_back(destination ? destination_rows : source_rows);
        result.push_back(features);
        return result;
    }

    [[nodiscard]] iom::TensorSpec source_spec(iom::DataType type) const {
        return iom::TensorSpec{
                iom::TensorShape{dimensions(false)}, type};
    }

    [[nodiscard]] iom::TensorSpec destination_spec(iom::DataType type) const {
        return iom::TensorSpec{
                iom::TensorShape{dimensions(true)}, type};
    }

    [[nodiscard]] bool valid() const noexcept {
        return rank() >= 3 && rank() <= 8 && heads != 0
                && source_rows != 0 && destination_rows != 0
                && features != 0 && offset <= destination_rows
                && source_rows <= destination_rows - offset;
    }
};

inline const std::vector<CacheAppendScenario>& cache_append_scenarios() {
    // The matrix deliberately keeps every final extent nonzero while crossing
    // both tile boundaries and the small feature/row boundaries that tend to
    // expose transposed planes or an accidental numeric conversion.
    static const std::vector<CacheAppendScenario> scenarios = {
            {"rank three: one row at offset one", {}, 1, 1, 17, 33, 1},
            {"rank four: fifteen rows at offset seventeen", {4}, 15, 15,
             32, 17, 17},
            {"rank five: sixteen rows with a tail", {2, 3}, 16, 16, 33, 31,
             16},
            {"rank six: seventeen rows at offset fifteen", {2, 1, 4}, 17, 17,
             34, 15, 15},
            {"rank seven: thirty-one heads and one row", {2, 3, 2, 1}, 31, 1,
             33, 16, 17},
            {"rank eight: nested leading planes", {4, 1, 2, 1, 3}, 17, 15,
             40, 1, 16},
    };
    return scenarios;
}

struct CacheAppendOrderedScenario {
    std::string label;
    std::vector<std::size_t> leading_dimensions;
    std::size_t heads = 0;
    std::size_t first_rows = 0;
    std::size_t first_offset = 0;
    std::size_t second_rows = 0;
    std::size_t second_offset = 0;
    std::size_t destination_rows = 0;
    std::size_t features = 0;
    bool second_reads_first = false;

    [[nodiscard]] CacheAppendScenario first() const {
        return {label + ": first append", leading_dimensions, heads,
                first_rows, destination_rows, features, first_offset};
    }
    [[nodiscard]] CacheAppendScenario second() const {
        return {label + ": second append", leading_dimensions, heads,
                second_rows, destination_rows, features, second_offset};
    }
};

inline const std::vector<CacheAppendOrderedScenario>&
cache_append_ordered_scenarios() {
    static const std::vector<CacheAppendOrderedScenario> scenarios = {
            {"two ordered appends", {2, 3}, 17, 1, 1, 15, 16, 33, 17},
            {"ordered append data dependency", {2, 1, 2}, 16, 16, 1, 40,
             0, 40, 31, true},
    };
    return scenarios;
}

struct CacheAppendViewCase {
    std::string label;
    std::function<iom::TensorView(const iom::TensorView&)> build;
};

// Explicit append view cases use only public leading-axis transforms.  Unlike
// the broad copy matrix, this list deliberately names a non-identity
// permutation and the stepped/nested families so a missing family is a hard
// fixture error rather than a silent omission.
inline std::vector<CacheAppendViewCase> cache_append_view_cases(
        const iom::TensorSpec& owner_spec) {
    const std::span<const std::size_t> dimensions =
            owner_spec.shape.dimensions();
    const std::size_t leading = dimensions.size() - 2;
    std::vector<CacheAppendViewCase> result;
    const auto add = [&](std::string label,
                         std::function<iom::TensorView(
                                 const iom::TensorView&)> build) {
        result.push_back({std::move(label), std::move(build)});
    };
    add("full", [](const iom::TensorView& full) { return full; });

    if (leading >= 1 && dimensions[0] >= 4) {
        const std::size_t first = dimensions[0];
        add("stepped slice of first leading axis",
            [first](const iom::TensorView& full) {
                return full.slice(0, 0, first / 2, 2);
            });
        add("interior dense slice of first leading axis",
            [first](const iom::TensorView& full) {
                return full.slice(0, 1, first - 1);
            });
    }
    if (leading >= 1 && dimensions[0] >= 2) {
        add("selected first leading plane",
            [](const iom::TensorView& full) {
                return full.select(0, 0);
            });
    }
    if (leading >= 2) {
        std::vector<std::size_t> order(leading);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::swap(order.front(), order.back());
        add("non-identity leading permutation",
            [order](const iom::TensorView& full) {
                return full.permute(
                        std::span<const std::size_t>{order});
            });
    }
    if (leading >= 1) {
        std::size_t product = 1;
        for (std::size_t axis = 0; axis < leading; ++axis) {
            product *= dimensions[axis];
        }
        add("reshaped leading planes",
            [product](const iom::TensorView& full) {
                return full.reshape_leading(span_of({product}));
            });
    }
    if (leading >= 3 && dimensions[0] >= 2 && dimensions[1] >= 3) {
        std::vector<std::size_t> order(leading - 1);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::swap(order.front(), order.back());
        add("nested select stepped slice permutation",
            [order](const iom::TensorView& full) {
                return full.select(0, 1)
                        .slice(0, 0, 2, 2)
                        .permute(std::span<const std::size_t>{order});
            });
    }
    if (leading >= 3 && dimensions[2] >= 4) {
        const std::size_t second = dimensions[1];
        const std::size_t third = dimensions[2];
        std::vector<std::size_t> reshaped{2 * second, third};
        reshaped.insert(
                reshaped.end(), dimensions.begin() + 3,
                dimensions.begin() + static_cast<std::ptrdiff_t>(leading));
        add("nested dense slice reshape select step",
            [reshaped = std::move(reshaped), second, third](
                    const iom::TensorView& full) {
                return full.slice(0, 0, 2)
                        .reshape_leading(
                                std::span<const std::size_t>{reshaped})
                        .select(0, 2 * second - 1)
                        .slice(0, 0, third / 2, 2);
            });
    }

    const auto has = [&](std::string_view needle) {
        return std::find_if(
                       result.begin(), result.end(),
                       [needle](const CacheAppendViewCase& candidate) {
                           return candidate.label.find(needle)
                                   != std::string::npos;
                       })
                != result.end();
    };
    if (leading >= 1 && dimensions[0] >= 4) {
        REQUIRE_MESSAGE(
                has("stepped slice"),
                "cache append view matrix lost its stepped case");
    }
    if (leading >= 2) {
        REQUIRE_MESSAGE(
                has("non-identity leading permutation"),
                "cache append view matrix lost its permutation case");
    }
    if (leading >= 3 && dimensions[0] >= 2 && dimensions[1] >= 3) {
        REQUIRE_MESSAGE(
                has("nested select"),
                "cache append view matrix lost its nested case");
    }
    return result;
}

struct CacheAppendFaultSeam {
    // Arm a failure that is observed only after a valid append is accepted.
    // The callback belongs to the backend port and may inspect its concrete
    // queue type; the common harness sees only the public DeviceOps contract.
    std::function<void(iom::DeviceOps&)> arm_post_acceptance_failure;
};

struct CacheAppendStateSnapshot {
    std::size_t registrations = 0;
    std::size_t leases = 0;
    std::size_t submissions = 0;

    friend bool operator==(
            const CacheAppendStateSnapshot& lhs,
            const CacheAppendStateSnapshot& rhs) = default;
};

struct CacheAppendObservation {
    // Optional backend-owned counters.  A configured observer must report
    // identical values around rejected validation and repeated queries.
    std::function<CacheAppendStateSnapshot(const iom::DeviceOps&)>
            snapshot;
};

struct CacheAppendConformanceConfig {
    ConformanceDevices devices;
    std::span<const iom::DataType> supported_types;
    AcceleratorStorageOracle* storage_oracle = nullptr;
    CacheAppendFaultSeam fault_seam;
    CacheAppendObservation observation;
    ConformanceObserver* observer = nullptr;
    // TTNN's caller-owned host workspace address domain is intentionally
    // separate from opaque device tensor handles.  Its port sets this false;
    // standard CPU/GPU/SYCL ports leave it true to exercise range overlap.
    bool workspace_overlap_supported = true;
};

// A backend port arms this window after all fixture owners and resources for a
// case exist.  Tensor transfers, transforms, validation, and submissions then
// run while the backend's allocator gate is armed; teardown disarms it before
// owners are released.
class CacheAppendCaseWindow final {
public:
    explicit CacheAppendCaseWindow(ConformanceObserver* observer)
            : observer_(observer) {
        if (observer_ != nullptr) {
            observer_->setup_complete();
        }
    }

    ~CacheAppendCaseWindow() {
        if (observer_ != nullptr) {
            observer_->case_complete();
        }
    }

    CacheAppendCaseWindow(const CacheAppendCaseWindow&) = delete;
    CacheAppendCaseWindow& operator=(const CacheAppendCaseWindow&) = delete;
    CacheAppendCaseWindow(CacheAppendCaseWindow&&) = delete;
    CacheAppendCaseWindow& operator=(CacheAppendCaseWindow&&) = delete;

private:
    ConformanceObserver* observer_;
};

// Qualify the live owner specifications, not independent view copies.  The
// common admission path rejects a view/owner mismatch before it reaches the
// operation's quantization capability decision, and an unrestored format can
// make backend tensor teardown fail while deriving its storage extent.
class CacheAppendQuantizationQualification final {
public:
    CacheAppendQuantizationQualification(
            iom::Tensor& source, iom::Tensor& destination)
            : specs_{qualify(source), qualify(destination)} {}

    ~CacheAppendQuantizationQualification() {
        for (iom::TensorSpec* spec : specs_) {
            spec->quantization = iom::QuantizationFormat::NONE;
        }
    }

    CacheAppendQuantizationQualification(
            const CacheAppendQuantizationQualification&) = delete;
    CacheAppendQuantizationQualification& operator=(
            const CacheAppendQuantizationQualification&) = delete;
    CacheAppendQuantizationQualification(
            CacheAppendQuantizationQualification&&) = delete;
    CacheAppendQuantizationQualification& operator=(
            CacheAppendQuantizationQualification&&) = delete;

private:
    [[nodiscard]] static iom::TensorSpec* qualify(iom::Tensor& owner) {
        iom::TensorSpec& spec =
                const_cast<iom::TensorSpec&>(owner.view().spec());
        spec.quantization = iom::QuantizationFormat::OCP_MXFP4;
        return &spec;
    }

    std::array<iom::TensorSpec*, 2> specs_;
};
// A live workspace owner over a caller-selected address makes range-overlap
// admission observable without allocating backend scratch.  The same helper
// also produces a view whose owner is stale after the helper returns.
class CacheAppendConformanceWorkspace final : public iom::RawWorkspace {
public:
    CacheAppendConformanceWorkspace(
            iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

inline iom::RawWorkspaceView cache_append_dead_workspace_view(
        iom::Device& device, void* address, std::size_t bytes) {
    CacheAppendConformanceWorkspace dead(device, address, bytes);
    return dead.view();
}



inline bool cache_append_type_is_advertised(
        std::span<const iom::DataType> supported, iom::DataType type) {
    return std::find(supported.begin(), supported.end(), type)
            != supported.end();
}

inline bool cache_append_matches_capability_profile(
        std::span<const iom::DataType> actual) {
    const bool omit_e8m0 = actual.size() == kStandardCapabilityOracle.size() - 1;
    if (!omit_e8m0 && actual.size() != kStandardCapabilityOracle.size()) {
        return false;
    }
    std::size_t actual_index = 0;
    for (const iom::DataType expected : kStandardCapabilityOracle) {
        if (omit_e8m0 && expected == iom::DataType::F8_E8M0) {
            continue;
        }
        if (actual_index == actual.size() || actual[actual_index] != expected) {
            return false;
        }
        ++actual_index;
    }
    return actual_index == actual.size();
}

inline void require_cache_append_capabilities(
        std::span<const iom::DataType> supported,
        std::span<const iom::DataType> storable) {
    REQUIRE_MESSAGE(
            cache_append_matches_capability_profile(storable),
            "cache append candidate must advertise the standard 23-leaf "
            "profile or exactly that profile without F8_E8M0");
    REQUIRE_MESSAGE(
            cache_append_matches_capability_profile(supported),
            "cache append configuration must use the exact candidate profile");
    for (const iom::DataType type : supported) {
        CAPTURE(static_cast<int>(type));
        CHECK(cache_append_type_is_advertised(storable, type));
    }
}


// ---------------------------------------------------------------------------
// Independent logical-byte model.
// ---------------------------------------------------------------------------

inline std::vector<std::byte> encode_cache_append_logical(
        const iom::TensorSpec& spec, std::uint64_t salt) {
    const std::size_t count = spec.shape.element_count();
    const std::size_t bits = bits_of(spec.data_type);
    std::vector<std::byte> result(spec.logical_nbytes(), std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(result.data());
    for (std::size_t linear = 0; linear < count; ++linear) {
        write_bits(
                base, linear * bits, bits,
                cache_append_element_pattern(spec.data_type, linear, salt));
    }
    return result;
}

inline void require_cache_append_view_shapes(
        const iom::TensorSpec& source,
        const iom::TensorSpec& destination, std::size_t a) {
    const std::span<const std::size_t> source_dimensions =
            source.shape.dimensions();
    const std::span<const std::size_t> destination_dimensions =
            destination.shape.dimensions();
    if (source_dimensions.size() != destination_dimensions.size()
            || source_dimensions.size() < 3
            || source.data_type != destination.data_type
            || source.quantization != destination.quantization) {
        throw std::invalid_argument("cache append logical model shape mismatch");
    }
    for (std::size_t axis = 0; axis + 3 < source_dimensions.size(); ++axis) {
        if (source_dimensions[axis] != destination_dimensions[axis]) {
            throw std::invalid_argument("cache append logical leading mismatch");
        }
    }
    if (source_dimensions[source_dimensions.size() - 3]
                != destination_dimensions[destination_dimensions.size() - 3]
            || source_dimensions.back() != destination_dimensions.back()) {
        throw std::invalid_argument("cache append logical head/features mismatch");
    }
    if (a > destination_dimensions[destination_dimensions.size() - 2]
            || source_dimensions[source_dimensions.size() - 2]
                    > destination_dimensions[destination_dimensions.size() - 2]
                            - a) {
        throw std::invalid_argument("cache append logical row range mismatch");
    }
}

// Copy exact encoded payload bits from source rows into a destination byte
// image.  This function intentionally does not use TensorView, a physical
// mapper, arithmetic, or numeric conversion; it is the row-mapping oracle.
inline std::vector<std::byte> cache_append_expected_logical(
        const iom::TensorSpec& source_spec,
        const iom::TensorSpec& destination_spec, std::size_t a,
        std::span<const std::byte> source,
        std::span<const std::byte> destination_before) {
    require_cache_append_view_shapes(source_spec, destination_spec, a);
    const std::size_t source_bytes = source_spec.logical_nbytes();
    const std::size_t destination_bytes = destination_spec.logical_nbytes();
    if (source.size() != source_bytes
            || destination_before.size() != destination_bytes) {
        throw std::invalid_argument("cache append logical model byte size mismatch");
    }

    std::vector<std::byte> result(
            destination_before.begin(), destination_before.end());
    const std::span<const std::size_t> source_dimensions =
            source_spec.shape.dimensions();
    const std::span<const std::size_t> destination_dimensions =
            destination_spec.shape.dimensions();
    const std::size_t rank = source_dimensions.size();
    const std::size_t source_rows = source_dimensions[rank - 2];
    const std::size_t destination_rows = destination_dimensions[rank - 2];
    const std::size_t features = source_dimensions.back();
    const std::size_t bits = bits_of(source_spec.data_type);
    const std::size_t count = source_spec.shape.element_count();

    for (std::size_t linear = 0; linear < count; ++linear) {
        std::size_t rest = linear;
        const std::size_t column = rest % features;
        rest /= features;
        const std::size_t row = rest % source_rows;
        rest /= source_rows;
        const std::size_t destination_linear =
                (rest * destination_rows + a + row) * features + column;
        const std::uint64_t value = read_storage_bits(
                source.data(), linear * bits, bits);
        write_bits(
                reinterpret_cast<unsigned char*>(result.data()),
                destination_linear * bits, bits, value);
    }
    return result;
}

inline constexpr std::byte kCacheAppendDestinationSentinel{0xC7};
inline constexpr std::byte kCacheAppendSourceSentinel{0x35};

struct CacheAppendLogicalModel {
    iom::TensorSpec source_spec;
    iom::TensorSpec destination_spec;
    std::size_t offset = 0;
    std::vector<std::byte> source;
    std::vector<std::byte> destination_before;
    std::vector<std::byte> destination_after;
};

inline CacheAppendLogicalModel make_cache_append_logical_model(
        const CacheAppendScenario& scenario, iom::DataType type,
        std::uint64_t salt) {
    REQUIRE(scenario.valid());
    CacheAppendLogicalModel model{
            scenario.source_spec(type), scenario.destination_spec(type),
            scenario.offset};
    model.source = encode_cache_append_logical(model.source_spec, salt);
    model.destination_before.assign(
            model.destination_spec.logical_nbytes(),
            kCacheAppendDestinationSentinel);
    model.destination_after = cache_append_expected_logical(
            model.source_spec, model.destination_spec, model.offset,
            model.source, model.destination_before);
    return model;
}

inline bool cache_append_logical_equal(
        std::span<const std::byte> lhs, std::span<const std::byte> rhs) {
    return lhs.size() == rhs.size()
            && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

// ---------------------------------------------------------------------------
// Device runner helpers.
// ---------------------------------------------------------------------------

inline std::unique_ptr<iom::RawWorkspace> cache_append_workspace(
        iom::Device& device, iom::WorkspaceRequirements requirements) {
    if (requirements.bytes == 0) {
        return nullptr;
    }
    return device.create_workspace(requirements.bytes);
}

inline iom::oid submit_cache_append(
        iom::DeviceOps& queue, const iom::TensorView& source,
        iom::TensorView& destination, std::size_t offset,
        const std::unique_ptr<iom::RawWorkspace>& workspace) {
    if (workspace == nullptr) {
        return queue.cache_append(source, destination, offset);
    }
    return queue.cache_append(
            source, destination, offset, workspace->view());
}

inline iom::WorkspaceRequirements query_cache_append_twice(
        iom::DeviceOps& queue, const iom::TensorView& source,
        const iom::TensorView& destination, std::size_t offset) {
    const iom::WorkspaceRequirements first =
            queue.cache_append_workspace_requirements(
                    source, destination, offset);
    CHECK_EQ(
            first,
            queue.cache_append_workspace_requirements(
                    source, destination, offset));
    return first;
}

inline void require_cache_append_storage(
        AcceleratorStorageOracle& oracle, const iom::TensorSpec& owner_spec,
        const iom::TensorView& owner_view,
        std::span<const std::byte> expected, std::string_view context) {
    oracle.set_owner_spec(owner_spec);
    const std::vector<std::byte> actual = oracle.observe(owner_view);
    REQUIRE_MESSAGE(
            actual.size() == expected.size(),
            context << ": storage byte count differs");
    CHECK_MESSAGE(
            std::equal(actual.begin(), actual.end(), expected.begin()),
            context << ": canonical physical bytes differ");
}

inline bool cache_append_view_is_usable(
        const iom::TensorView& source, const iom::TensorView& destination) {
    const std::span<const std::size_t> source_dimensions =
            source.spec().shape.dimensions();
    const std::span<const std::size_t> destination_dimensions =
            destination.spec().shape.dimensions();
    if (source_dimensions.size() != destination_dimensions.size()
            || source_dimensions.size() < 3
            || source_dimensions.size() > 8
            || source.spec().data_type != destination.spec().data_type
            || source.spec().quantization
                    != destination.spec().quantization
            || source_dimensions.back() != destination_dimensions.back()) {
        return false;
    }
    for (std::size_t axis = 0; axis + 2 < source_dimensions.size(); ++axis) {
        if (source_dimensions[axis] != destination_dimensions[axis]) {
            return false;
        }
    }
    return true;
}
inline void run_cache_append_view_case(
        const CacheAppendConformanceConfig& config,
        const CacheAppendScenario& scenario, iom::DataType type,
        const CacheAppendViewCase& view_case, std::uint64_t salt) {
    iom::TensorSpec source_owner_spec = scenario.source_spec(type);
    iom::TensorSpec destination_owner_spec = scenario.destination_spec(type);
    auto reference_source = config.devices.reference.create_tensor(
            source_owner_spec);
    auto reference_destination = config.devices.reference.create_tensor(
            destination_owner_spec);
    auto candidate_source = config.devices.candidate.create_tensor(
            source_owner_spec);
    auto candidate_destination = config.devices.candidate.create_tensor(
            destination_owner_spec);

    iom::TensorView reference_source_view =
            view_case.build(reference_source->view());
    iom::TensorView reference_destination_view =
            view_case.build(reference_destination->view());
    iom::TensorView candidate_source_view =
            view_case.build(candidate_source->view());
    iom::TensorView candidate_destination_view =
            view_case.build(candidate_destination->view());
    const CacheAppendCaseWindow window(config.observer);
    REQUIRE_MESSAGE(
            cache_append_view_is_usable(
                    candidate_source_view, candidate_destination_view),
            view_case.label << ": candidate view pair is not usable");
    REQUIRE_MESSAGE(
            reference_source_view.spec() == candidate_source_view.spec(),
            view_case.label << ": reference and candidate source specs differ");
    REQUIRE_MESSAGE(
            reference_destination_view.spec()
                    == candidate_destination_view.spec(),
            view_case.label
                    << ": reference and candidate destination specs differ");

    std::vector<std::byte> source_logical;
    std::vector<std::byte> destination_before;
    std::vector<std::byte> source_storage;
    std::vector<std::byte> destination_storage;
    std::vector<std::byte> expected_destination_storage;
    if (config.storage_oracle != nullptr) {
        source_storage = encode_cache_append_storage(
                source_owner_spec, salt, kCacheAppendSourceSentinel);
        destination_storage = encode_cache_append_storage(
                destination_owner_spec, salt + 1,
                kCacheAppendDestinationSentinel);
        expected_destination_storage = destination_storage;
        config.storage_oracle->set_owner_spec(source_owner_spec);
        config.storage_oracle->seed(
                candidate_source->view(), source_storage);
        config.storage_oracle->set_owner_spec(destination_owner_spec);
        config.storage_oracle->seed(
                candidate_destination->view(), destination_storage);
        source_logical = decode_standard_tiled_view(
                candidate_source_view, source_owner_spec, source_storage);
        destination_before = decode_standard_tiled_view(
                candidate_destination_view, destination_owner_spec,
                destination_storage);
    } else {
        source_logical = encode_cache_append_logical(
                candidate_source_view.spec(), salt);
        destination_before = encode_logical(
                candidate_destination_view.spec(), salt + 1);
        copy_from_host(candidate_source_view, source_logical);
        copy_from_host(candidate_destination_view, destination_before);
    }
    const std::vector<std::byte> expected_destination =
            cache_append_expected_logical(
                    candidate_source_view.spec(),
                    candidate_destination_view.spec(), scenario.offset,
                    source_logical, destination_before);

    copy_from_host(reference_source_view, source_logical);
    copy_from_host(reference_destination_view, destination_before);
    require_logical_bytes(
            candidate_destination_view, destination_before,
            std::string("cache append destination before ")
                    + view_case.label);

    auto reference_queue = config.devices.reference.create_ops();
    auto candidate_queue = config.devices.candidate.create_ops();
    const std::optional<CacheAppendStateSnapshot> candidate_state_before =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*candidate_queue))
            : std::nullopt;
    const iom::WorkspaceRequirements reference_requirements =
            query_cache_append_twice(
                    *reference_queue, reference_source_view,
                    reference_destination_view, scenario.offset);
    const iom::WorkspaceRequirements candidate_requirements =
            query_cache_append_twice(
                    *candidate_queue, candidate_source_view,
                    candidate_destination_view, scenario.offset);
    if (candidate_state_before) {
        CHECK(
                *candidate_state_before
                == config.observation.snapshot(*candidate_queue));
    }
    // SYCL's native staging requirement is intentionally different from the
    // direct CPU/GPU paths, so each device's pure query is checked
    // independently rather than compared across backends.
    const std::vector<std::byte> after_query =
            read_logical(candidate_destination_view);
    CHECK(cache_append_logical_equal(after_query, destination_before));

    // The CPU reference executes cache_append with its internal staging path.
    // Keep its pure requirement query above, but do not turn that query into
    // a positive raw-workspace allocation on the reference device.
    (void)reference_requirements;
    std::unique_ptr<iom::RawWorkspace> reference_workspace;
    std::unique_ptr<iom::RawWorkspace> candidate_workspace =
            cache_append_workspace(
                    config.devices.candidate, candidate_requirements);
    iom::oid reference_token = 0;
    iom::oid candidate_token = 0;
    {
        const iom::TensorView temporary_reference_source = reference_source_view;
        iom::TensorView temporary_reference_destination =
                reference_destination_view;
        const iom::TensorView temporary_candidate_source = candidate_source_view;
        iom::TensorView temporary_candidate_destination =
                candidate_destination_view;
        reference_token = submit_cache_append(
                *reference_queue, temporary_reference_source,
                temporary_reference_destination, scenario.offset,
                reference_workspace);
        candidate_token = submit_cache_append(
                *candidate_queue, temporary_candidate_source,
                temporary_candidate_destination, scenario.offset,
                candidate_workspace);
    }
    REQUIRE(iom::oid_is_token(reference_token));
    REQUIRE(iom::oid_is_token(candidate_token));
    reference_queue->wait(reference_token);
    reference_queue->wait(reference_token);
    candidate_queue->wait(candidate_token);
    candidate_queue->wait(candidate_token);
    require_logical_bytes(
            reference_destination_view, expected_destination,
            std::string("reference cache append ") + view_case.label);
    require_logical_bytes(
            candidate_destination_view, expected_destination,
            std::string("candidate cache append ") + view_case.label);

    if (config.storage_oracle != nullptr) {
        apply_cache_append_storage(
                candidate_source_view, source_owner_spec,
                candidate_destination_view, destination_owner_spec,
                scenario.offset, source_storage,
                expected_destination_storage);
        require_cache_append_storage(
                *config.storage_oracle, destination_owner_spec,
                candidate_destination->view(), expected_destination_storage,
                std::string("candidate cache append storage ")
                        + view_case.label);
        require_cache_append_storage(
                *config.storage_oracle, source_owner_spec,
                candidate_source->view(), source_storage,
                std::string("candidate cache append source storage ")
                        + view_case.label);
    }
}

inline void run_cache_append_ordered_case(
        const CacheAppendConformanceConfig& config,
        const CacheAppendOrderedScenario& ordered, iom::DataType type) {
    const CacheAppendScenario first = ordered.first();
    const CacheAppendScenario second = ordered.second();
    REQUIRE(first.valid());
    REQUIRE(second.valid());
    const iom::TensorSpec source_one_spec = first.source_spec(type);
    const iom::TensorSpec source_two_spec = second.source_spec(type);
    const iom::TensorSpec destination_spec = first.destination_spec(type);
    if (ordered.second_reads_first) {
        REQUIRE(source_two_spec == destination_spec);
    }
    auto source_one = config.devices.candidate.create_tensor(source_one_spec);
    auto source_two = config.devices.candidate.create_tensor(source_two_spec);
    auto destination = config.devices.candidate.create_tensor(destination_spec);
    iom::TensorView source_one_view = source_one->view();
    iom::TensorView source_two_view = source_two->view();
    iom::TensorView destination_view = destination->view();
    const CacheAppendCaseWindow window(config.observer);

    std::vector<std::byte> source_one_bytes =
            encode_cache_append_logical(source_one_spec, 0x101);
    std::vector<std::byte> source_two_bytes =
            encode_cache_append_logical(source_two_spec, 0x202);
    std::vector<std::byte> destination_before =
            encode_logical(destination_spec, 0x303);
    std::vector<std::byte> source_one_storage;
    std::vector<std::byte> source_two_storage;
    std::vector<std::byte> source_two_expected_storage;
    std::vector<std::byte> destination_storage;
    std::vector<std::byte> after_first_storage;
    std::vector<std::byte> after_second_storage;
    if (config.storage_oracle != nullptr) {
        source_one_storage = encode_cache_append_storage(
                source_one_spec, 0x101, kCacheAppendSourceSentinel);
        source_two_storage = encode_cache_append_storage(
                source_two_spec, 0x202, kCacheAppendSourceSentinel);
        destination_storage = encode_cache_append_storage(
                destination_spec, 0x303, kCacheAppendDestinationSentinel);
        config.storage_oracle->set_owner_spec(source_one_spec);
        config.storage_oracle->seed(source_one_view, source_one_storage);
        config.storage_oracle->set_owner_spec(source_two_spec);
        config.storage_oracle->seed(source_two_view, source_two_storage);
        config.storage_oracle->set_owner_spec(destination_spec);
        config.storage_oracle->seed(destination_view, destination_storage);
        source_one_bytes = decode_standard_tiled_view(
                source_one_view, source_one_spec, source_one_storage);
        source_two_bytes = decode_standard_tiled_view(
                source_two_view, source_two_spec, source_two_storage);
        destination_before = decode_standard_tiled_view(
                destination_view, destination_spec, destination_storage);
    } else {
        copy_from_host(source_one_view, source_one_bytes);
        copy_from_host(source_two_view, source_two_bytes);
        copy_from_host(destination_view, destination_before);
    }
    const std::vector<std::byte> after_first =
            cache_append_expected_logical(
                    source_one_spec, destination_spec, first.offset,
                    source_one_bytes, destination_before);
    if (ordered.second_reads_first) {
        // The second source owner has the same full shape as the destination.
        // A queued copy from the first append's destination makes the second
        // append observe the first append's result without aliasing owners.
        source_two_bytes = after_first;
    }
    const std::vector<std::byte> after_second =
            cache_append_expected_logical(
                    source_two_spec, destination_spec, second.offset,
                    source_two_bytes, after_first);
    if (config.storage_oracle != nullptr) {
        after_first_storage = destination_storage;
        apply_cache_append_storage(
                source_one_view, source_one_spec, destination_view,
                destination_spec, first.offset, source_one_storage,
                after_first_storage);
        source_two_expected_storage = source_two_storage;
        if (ordered.second_reads_first) {
            apply_standard_tiled_view(
                    destination_view, destination_spec, after_first,
                    source_two_expected_storage);
        }
        after_second_storage = after_first_storage;
        apply_cache_append_storage(
                source_two_view, source_two_spec, destination_view,
                destination_spec, second.offset, source_two_expected_storage,
                after_second_storage);
    }

    auto queue = config.devices.candidate.create_ops();
    const std::optional<CacheAppendStateSnapshot> first_state_before =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*queue))
            : std::nullopt;
    const iom::WorkspaceRequirements first_requirements =
            query_cache_append_twice(
                    *queue, source_one_view, destination_view,
                    first.offset);
    if (first_state_before) {
        CHECK(*first_state_before == config.observation.snapshot(*queue));
    }
    const std::optional<CacheAppendStateSnapshot> second_state_before =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*queue))
            : std::nullopt;
    const iom::WorkspaceRequirements second_requirements =
            query_cache_append_twice(
                    *queue, source_two_view, destination_view,
                    second.offset);
    if (second_state_before) {
        CHECK(*second_state_before == config.observation.snapshot(*queue));
    }
    std::unique_ptr<iom::RawWorkspace> first_workspace =
            cache_append_workspace(
                    config.devices.candidate, first_requirements);
    std::unique_ptr<iom::RawWorkspace> second_workspace =
            cache_append_workspace(
                    config.devices.candidate, second_requirements);
    const iom::oid first_token = submit_cache_append(
            *queue, source_one_view, destination_view, first.offset,
            first_workspace);
    REQUIRE(iom::oid_is_token(first_token));
    iom::oid dependency_token = 0;
    if (ordered.second_reads_first) {
        dependency_token = queue->copy(destination_view, source_two_view);
        REQUIRE(iom::oid_is_token(dependency_token));
    }
    const iom::oid second_token = submit_cache_append(
            *queue, source_two_view, destination_view, second.offset,
            second_workspace);
    REQUIRE(iom::oid_is_token(second_token));
    CHECK_LT(token_sequence(first_token), token_sequence(second_token));
    if (ordered.second_reads_first) {
        CHECK_LT(
                token_sequence(first_token),
                token_sequence(dependency_token));
        CHECK_LT(
                token_sequence(dependency_token),
                token_sequence(second_token));
    }
    queue->wait(second_token);
    require_logical_bytes(
            destination_view, after_second,
            std::string("ordered cache append final ") + ordered.label);
    if (ordered.second_reads_first) {
        require_logical_bytes(
                source_two_view, source_two_bytes,
                std::string("ordered cache append dependency source ")
                        + ordered.label);
    }
    if (config.storage_oracle != nullptr) {
        require_cache_append_storage(
                *config.storage_oracle, destination_spec, destination_view,
                after_second_storage,
                std::string("ordered cache append final storage ")
                        + ordered.label);
        require_cache_append_storage(
                *config.storage_oracle, source_one_spec, source_one_view,
                source_one_storage,
                std::string("ordered cache append source one storage ")
                        + ordered.label);
        require_cache_append_storage(
                *config.storage_oracle, source_two_spec, source_two_view,
                ordered.second_reads_first ? source_two_expected_storage
                                            : source_two_storage,
                std::string("ordered cache append source two storage ")
                        + ordered.label);
    }
}

enum class CacheAppendValidationError {
    invalid_argument,
    overflow,
};

inline void require_cache_append_query_error(
        iom::DeviceOps& queue, const iom::TensorView& source,
        iom::TensorView& destination, std::size_t offset,
        CacheAppendValidationError error) {
    switch (error) {
        case CacheAppendValidationError::invalid_argument:
            CHECK_THROWS_AS(
                    (void)queue.cache_append_workspace_requirements(
                            source, destination, offset),
                    std::invalid_argument);
            break;
        case CacheAppendValidationError::overflow:
            CHECK_THROWS_AS(
                    (void)queue.cache_append_workspace_requirements(
                            source, destination, offset),
                    std::overflow_error);
            break;
    }
}

inline void run_cache_append_invalid_case(
        const CacheAppendConformanceConfig& config,
        const iom::TensorView& source, iom::TensorView& destination,
        std::size_t offset, std::string_view context,
        CacheAppendValidationError query_error =
                CacheAppendValidationError::invalid_argument,
        iom::OidError submission_error = iom::OidError::InvalidArgument) {
    auto queue = config.devices.candidate.create_ops();
    const CacheAppendCaseWindow window(config.observer);
    const std::vector<std::byte> before = read_logical(destination);
    const std::optional<CacheAppendStateSnapshot> state_before =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*queue))
            : std::nullopt;
    require_cache_append_query_error(
            *queue, source, destination, offset, query_error);
    CHECK_EQ(
            queue->cache_append(source, destination, offset),
            iom::to_oid(submission_error));
    if (state_before) {
        CHECK(*state_before == config.observation.snapshot(*queue));
    }
    require_logical_bytes(destination, before, context);
}

inline void run_cache_append_admission_conformance(
        const CacheAppendConformanceConfig& config) {
    const iom::DataType type = iom::DataType::F32;
    const iom::TensorSpec source_spec{
            iom::TensorShape{{2, 3, 17, 17}}, type};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{2, 3, 33, 17}}, type};
    auto source = config.devices.candidate.create_tensor(source_spec);
    auto destination = config.devices.candidate.create_tensor(destination_spec);
    copy_from_host(
            source->view(), encode_cache_append_logical(source_spec, 0x411));
    copy_from_host(destination->view(), encode_logical(destination_spec, 0x422));

    run_cache_append_invalid_case(
            config, source->view(), destination->view(), 34,
            "cache append offset greater than capacity");
    run_cache_append_invalid_case(
            config, source->view(), destination->view(), 17,
            "cache append rows exceed remaining capacity");

    auto mismatched_leading = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{3, 3, 17, 17}}, type});
    run_cache_append_invalid_case(
            config, mismatched_leading->view(), destination->view(), 1,
            "cache append mismatched leading tuple");

    auto mismatched_heads = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{2, 4, 17, 17}}, type});
    run_cache_append_invalid_case(
            config, mismatched_heads->view(), destination->view(), 1,
            "cache append mismatched heads");

    auto mismatched_features = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{2, 3, 17, 16}}, type});
    run_cache_append_invalid_case(
            config, mismatched_features->view(), destination->view(), 1,
            "cache append mismatched features");

    auto other_type = config.devices.candidate.create_tensor(
            iom::TensorSpec{
                    iom::TensorShape{{2, 3, 17, 17}}, iom::DataType::I16});
    run_cache_append_invalid_case(
            config, other_type->view(), destination->view(), 1,
            "cache append mismatched data type");

    auto rank_two_source = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{17, 17}}, type});
    auto rank_two_destination = config.devices.candidate.create_tensor(
            iom::TensorSpec{iom::TensorShape{{33, 17}}, type});
    run_cache_append_invalid_case(
            config, rank_two_source->view(), rank_two_destination->view(), 1,
            "cache append rank two operands");
    CHECK_THROWS_AS(
            config.devices.candidate.create_tensor(iom::TensorSpec{
                    iom::TensorShape{{1, 1, 1, 1, 1, 1, 1, 1, 1}}, type}),
            std::invalid_argument);

    {
        iom::TensorView unknown_source_view = source->view();
        iom::TensorSpec& unknown_source_spec =
                const_cast<iom::TensorSpec&>(unknown_source_view.spec());
        const iom::DataType previous = unknown_source_spec.data_type;
        unknown_source_spec.data_type = static_cast<iom::DataType>(200);
        run_cache_append_invalid_case(
                config, unknown_source_view, destination->view(), 1,
                "cache append unknown data type");
        unknown_source_spec.data_type = previous;
    }
    {
        iom::TensorView zero_source_view = source->view();
        iom::TensorSpec& zero_source_spec =
                const_cast<iom::TensorSpec&>(zero_source_view.spec());
        auto dimensions = zero_source_spec.shape.dimensions();
        std::size_t* mutable_dimensions =
                const_cast<std::size_t*>(dimensions.data());
        const std::size_t previous = mutable_dimensions[0];
        mutable_dimensions[0] = 0;
        run_cache_append_invalid_case(
                config, zero_source_view, destination->view(), 1,
                "cache append zero extent");
        mutable_dimensions[0] = previous;
    }
    {
        iom::TensorView overflow_source_view = source->view();
        auto strides = overflow_source_view.plane_strides();
        REQUIRE(!strides.empty());
        std::size_t* mutable_strides =
                const_cast<std::size_t*>(strides.data());
        const std::size_t previous = mutable_strides[0];
        mutable_strides[0] = std::numeric_limits<std::size_t>::max();
        run_cache_append_invalid_case(
                config, overflow_source_view, destination->view(), 1,
                "cache append overflowing plane stride",
                CacheAppendValidationError::overflow,
                iom::OidError::Overflow);
        mutable_strides[0] = previous;
    }
    const std::vector<std::byte> quantized_before =
            read_logical(destination->view());
    {
        CacheAppendQuantizationQualification qualified(
                *source, *destination);
        const iom::TensorView quantized_source_view = source->view();
        iom::TensorView quantized_destination_view = destination->view();
        auto quantized_queue = config.devices.candidate.create_ops();
        const CacheAppendCaseWindow window(config.observer);
        const std::optional<CacheAppendStateSnapshot> state_before =
                config.observation.snapshot
                ? std::optional<CacheAppendStateSnapshot>(
                          config.observation.snapshot(*quantized_queue))
                : std::nullopt;
        CHECK_THROWS_AS(
                (void)quantized_queue->cache_append_workspace_requirements(
                        quantized_source_view, quantized_destination_view, 1),
                std::runtime_error);
        CHECK_EQ(
                quantized_queue->cache_append(
                        quantized_source_view, quantized_destination_view, 1),
                iom::to_oid(iom::OidError::Unsupported));
        if (state_before) {
            CHECK(*state_before == config.observation.snapshot(*quantized_queue));
        }
    }
    require_logical_bytes(
            destination->view(), quantized_before,
            "cache append quantized operands");

    auto foreign_source = config.devices.foreign.create_tensor(source_spec);
    run_cache_append_invalid_case(
            config, foreign_source->view(), destination->view(), 1,
            "cache append foreign device");

    const iom::TensorSpec overlap_spec{
            iom::TensorShape{{2, 3, 17, 33}}, type};
    auto overlap = config.devices.candidate.create_tensor(overlap_spec);
    copy_from_host(
            overlap->view(), encode_cache_append_logical(overlap_spec, 0x433));
    run_cache_append_invalid_case(
            config, overlap->view(), overlap->view(), 0,
            "cache append identical owner overlap");

    const iom::TensorView full = destination->view();
    const iom::TensorView selected_source = full.select(0, 0);
    iom::TensorView selected_destination = destination->view().select(0, 0);
    run_cache_append_invalid_case(
            config, selected_source, selected_destination, 1,
            "cache append aliases through leading select");

    const iom::TensorView disjoint_source_window =
            overlap->view().select(0, 0);
    iom::TensorView disjoint_destination_window =
            overlap->view().select(0, 1);
    run_cache_append_invalid_case(
            config, disjoint_source_window, disjoint_destination_window, 0,
            "cache append same owner disjoint windows");
    // Query and submission of a valid case are side-effect free until the
    // caller accepts the operation.  A zero-workspace direct path is checked
    // explicitly; positive-workspace backends receive all identity/range
    // probes that can be expressed by the public workspace API.
    const iom::TensorSpec valid_source_spec{
            iom::TensorShape{{2, 3, 1, 17}}, type};
    const iom::TensorSpec valid_destination_spec{
            iom::TensorShape{{2, 3, 17, 17}}, type};
    auto valid_source = config.devices.candidate.create_tensor(valid_source_spec);
    auto valid_destination =
            config.devices.candidate.create_tensor(valid_destination_spec);
    const std::vector<std::byte> valid_source_bytes =
            encode_cache_append_logical(valid_source_spec, 0x444);
    const std::vector<std::byte> valid_destination_bytes =
            encode_logical(valid_destination_spec, 0x445);
    copy_from_host(valid_source->view(), valid_source_bytes);
    copy_from_host(valid_destination->view(), valid_destination_bytes);
    auto queue = config.devices.candidate.create_ops();
    const std::optional<CacheAppendStateSnapshot> state_before_query =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*queue))
            : std::nullopt;
    const iom::WorkspaceRequirements requirements =
            query_cache_append_twice(
                    *queue, valid_source->view(), valid_destination->view(), 1);
    if (state_before_query) {
        CHECK(*state_before_query == config.observation.snapshot(*queue));
    }
    CHECK(cache_append_logical_equal(
            read_logical(valid_destination->view()), valid_destination_bytes));
    if (requirements.bytes == 0) {
        const std::vector<std::byte> expected_after =
                cache_append_expected_logical(
                        valid_source_spec, valid_destination_spec, 1,
                        valid_source_bytes, valid_destination_bytes);
        auto zero_owner = config.devices.candidate.create_workspace(0);
        auto foreign_zero_owner = config.devices.foreign.create_workspace(0);
        REQUIRE(zero_owner != nullptr);
        REQUIRE(foreign_zero_owner != nullptr);
        CacheAppendConformanceWorkspace overlap_zero_owner(
                config.devices.candidate,
                valid_source->view().native_handle(), 0);
        CacheAppendConformanceWorkspace misaligned_zero_owner(
                config.devices.candidate, reinterpret_cast<void*>(0x1), 0);
        const iom::RawWorkspaceView stale_zero_view =
                cache_append_dead_workspace_view(
                        config.devices.candidate,
                        valid_source->view().native_handle(), 0);
        const auto submit_zero_workspace =
                [&](const iom::RawWorkspaceView& workspace,
                    std::string_view context) {
                    copy_from_host(
                            valid_destination->view(), valid_destination_bytes);
                    const std::optional<CacheAppendStateSnapshot> before =
                            config.observation.snapshot
                            ? std::optional<CacheAppendStateSnapshot>(
                                      config.observation.snapshot(*queue))
                            : std::nullopt;
                    const iom::oid token = queue->cache_append(
                            valid_source->view(), valid_destination->view(),
                            1, workspace);
                    REQUIRE(iom::oid_is_token(token));
                    queue->wait(token);
                    require_logical_bytes(
                            valid_destination->view(), expected_after, context);
                    if (before) {
                        const CacheAppendStateSnapshot after =
                                config.observation.snapshot(*queue);
                        CHECK_EQ(after.registrations, before->registrations);
                        CHECK_EQ(after.leases, before->leases);
                        CHECK_EQ(after.submissions, before->submissions + 1);
                    }
                };
        submit_zero_workspace(
                zero_owner->view(), "cache append zero workspace owner");
        submit_zero_workspace(
                foreign_zero_owner->view(),
                "cache append zero foreign workspace owner");
        submit_zero_workspace(
                stale_zero_view, "cache append zero stale workspace view");
        submit_zero_workspace(
                overlap_zero_owner.view(),
                "cache append zero overlapping workspace range");
        submit_zero_workspace(
                misaligned_zero_owner.view(),
                "cache append zero misaligned workspace range");
    } else {
        const std::size_t alignment_quantum = 32;
        REQUIRE(
                requirements.alignment
                <= std::numeric_limits<std::size_t>::max()
                        - (alignment_quantum - 1));
        std::size_t aligned_offset =
                ((requirements.alignment + alignment_quantum - 1)
                 / alignment_quantum)
                * alignment_quantum;
        aligned_offset = std::max(aligned_offset, alignment_quantum);
        REQUIRE(
                requirements.bytes
                <= std::numeric_limits<std::size_t>::max()
                        - aligned_offset);
        const std::size_t exact_owner_bytes =
                requirements.bytes + aligned_offset;

        auto undersized = config.devices.candidate.create_workspace(
                requirements.bytes - 1);
        REQUIRE(undersized != nullptr);
        CHECK_EQ(
                queue->cache_append(
                        valid_source->view(), valid_destination->view(), 1,
                        undersized->view()),
                iom::to_oid(iom::OidError::InvalidArgument));
        CHECK(cache_append_logical_equal(
                read_logical(valid_destination->view()),
                valid_destination_bytes));

        auto exact = config.devices.candidate.create_workspace(
                exact_owner_bytes);
        REQUIRE(exact != nullptr);
        const iom::RawWorkspaceView aligned_subrange =
                exact->view().subrange(aligned_offset, requirements.bytes);
        const iom::oid token = queue->cache_append(
                valid_source->view(), valid_destination->view(), 1,
                aligned_subrange);
        REQUIRE(iom::oid_is_token(token));
        queue->wait(token);
        const std::vector<std::byte> after_exact =
                read_logical(valid_destination->view());
        if (config.workspace_overlap_supported
                && requirements.alignment != 0 && requirements.bytes != 0) {
            const std::uintptr_t operand_base =
                    reinterpret_cast<std::uintptr_t>(
                            valid_source->view().native_handle());
            const std::size_t operand_bytes =
                    valid_source_spec.tiled_storage_nbytes();
            const std::uintptr_t limit =
                    std::numeric_limits<std::uintptr_t>::max();
            const std::uintptr_t remainder =
                    operand_base % requirements.alignment;
            const std::uintptr_t aligned_base =
                    remainder == 0
                    ? operand_base
                    : operand_base
                            + (requirements.alignment - remainder);
            if (aligned_base >= operand_base
                    && operand_bytes <= limit - operand_base
                    && aligned_base < operand_base + operand_bytes
                    && requirements.bytes <= limit - aligned_base) {
                CacheAppendConformanceWorkspace overlapping_workspace(
                        config.devices.candidate,
                        reinterpret_cast<void*>(aligned_base),
                        requirements.bytes);
                CHECK_EQ(
                        queue->cache_append(
                                valid_source->view(),
                                valid_destination->view(), 1,
                                overlapping_workspace.view()),
                        iom::to_oid(iom::OidError::InvalidArgument));
                CHECK(cache_append_logical_equal(
                        read_logical(valid_destination->view()), after_exact));
            }
        }

        auto foreign = config.devices.foreign.create_workspace(
                requirements.bytes);
        REQUIRE(foreign != nullptr);
        CHECK_EQ(
                queue->cache_append(
                        valid_source->view(), valid_destination->view(), 1,
                        foreign->view()),
                iom::to_oid(iom::OidError::InvalidArgument));
        CHECK(cache_append_logical_equal(
                read_logical(valid_destination->view()), after_exact));
        auto stale_owner = config.devices.candidate.create_workspace(
                requirements.bytes);
        REQUIRE(stale_owner != nullptr);
        const iom::RawWorkspaceView stale = stale_owner->view();
        stale_owner.reset();
        CHECK_EQ(
                queue->cache_append(
                        valid_source->view(), valid_destination->view(), 1,
                        stale),
                iom::to_oid(iom::OidError::InvalidArgument));
        CHECK(cache_append_logical_equal(
                read_logical(valid_destination->view()), after_exact));

        if (requirements.alignment > alignment_quantum) {
            auto misaligned_owner = config.devices.candidate.create_workspace(
                    requirements.bytes + alignment_quantum);
            REQUIRE(misaligned_owner != nullptr);
            const iom::RawWorkspaceView misaligned =
                    misaligned_owner->view().subrange(
                            alignment_quantum, requirements.bytes);
            CHECK_EQ(
                    queue->cache_append(
                            valid_source->view(), valid_destination->view(), 1,
                            misaligned),
                    iom::to_oid(iom::OidError::InvalidArgument));
        }
    }
}

inline void run_cache_append_capability_rejections(
        const CacheAppendConformanceConfig& config) {
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    for (const iom::DataType type : kStandardCapabilityOracle) {
        const bool storable = cache_append_type_is_advertised(
                config.devices.candidate.supported_data_types(), type);
        if (cache_append_type_is_advertised(
                    config.supported_types, type)) {
            CHECK(storable);
            continue;
        }
        CAPTURE(static_cast<int>(type));
        const iom::TensorSpec source_spec{
                iom::TensorShape{{1, 1, 1, 17}}, type};
        const iom::TensorSpec destination_spec{
                iom::TensorShape{{1, 1, 17, 17}}, type};
        if (!storable) {
            CHECK_THROWS(config.devices.candidate.create_tensor(source_spec));
            continue;
        }
        auto source = config.devices.candidate.create_tensor(source_spec);
        auto destination =
                config.devices.candidate.create_tensor(destination_spec);
        auto queue = config.devices.candidate.create_ops();
        CHECK_EQ(
                queue->cache_append(source->view(), destination->view(), 1),
                unsupported);
        CHECK_THROWS_AS(
                (void)queue->cache_append_workspace_requirements(
                        source->view(), destination->view(), 1),
                std::runtime_error);
    }
}
inline void run_cache_append_failure_conformance(
        const CacheAppendConformanceConfig& config) {
    if (!config.fault_seam.arm_post_acceptance_failure) {
        return;
    }
    const iom::DataType type = iom::DataType::F32;
    const iom::TensorSpec source_spec{
            iom::TensorShape{{1, 1, 1, 17}}, type};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{1, 1, 17, 17}}, type};
    auto source = config.devices.candidate.create_tensor(source_spec);
    auto destination = config.devices.candidate.create_tensor(destination_spec);
    const std::vector<std::byte> source_bytes =
            encode_cache_append_logical(source_spec, 0x551);
    copy_from_host(source->view(), source_bytes);
    const std::vector<std::byte> before =
            encode_logical(destination_spec, 0x552);
    copy_from_host(destination->view(), before);
    auto queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements requirements =
            query_cache_append_twice(
                    *queue, source->view(), destination->view(), 1);
    std::unique_ptr<iom::RawWorkspace> workspace =
            cache_append_workspace(
                    config.devices.candidate, requirements);
    {
        const CacheAppendCaseWindow window(config.observer);
        config.fault_seam.arm_post_acceptance_failure(*queue);
        const iom::oid token = submit_cache_append(
                *queue, source->view(), destination->view(), 1, workspace);
        REQUIRE(iom::oid_is_token(token));
        expect_repeated_runtime_failure(*queue, token);
        require_logical_bytes(
                destination->view(), before,
                "cache append retained failure leaves destination unchanged");
    }
    auto recovery_destination =
            config.devices.candidate.create_tensor(destination_spec);
    const std::vector<std::byte> recovery_before =
            encode_logical(destination_spec, 0x553);
    copy_from_host(recovery_destination->view(), recovery_before);
    auto recovery_queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements recovery_requirements =
            query_cache_append_twice(
                    *recovery_queue, source->view(),
                    recovery_destination->view(), 1);
    auto recovery_workspace = cache_append_workspace(
            config.devices.candidate, recovery_requirements);
    {
        const CacheAppendCaseWindow window(config.observer);
        const iom::oid recovery_token = submit_cache_append(
                *recovery_queue, source->view(), recovery_destination->view(),
                1, recovery_workspace);
        REQUIRE(iom::oid_is_token(recovery_token));
        recovery_queue->wait(recovery_token);
        recovery_queue->wait(recovery_token);
        require_logical_bytes(
                recovery_destination->view(),
                cache_append_expected_logical(
                        source_spec, destination_spec, 1, source_bytes,
                        recovery_before),
                "cache append post-failure independent reuse");
    }
}
inline void run_cache_append_pipeline_conformance(
        const CacheAppendConformanceConfig& config) {
    constexpr iom::DataType type = iom::DataType::BF16;
    const iom::TensorSpec source_spec{
            iom::TensorShape{{2, 3, 1, 17}}, type};
    const iom::TensorSpec destination_spec{
            iom::TensorShape{{2, 3, 17, 17}}, type};
    auto source = config.devices.candidate.create_tensor(source_spec);
    auto staging = config.devices.candidate.create_tensor(source_spec);
    auto destination = config.devices.candidate.create_tensor(destination_spec);
    auto consumer = config.devices.candidate.create_tensor(destination_spec);

    std::vector<std::byte> source_bytes =
            encode_cache_append_logical(source_spec, 0x651);
    std::vector<std::byte> destination_before =
            encode_logical(destination_spec, 0x652);
    std::vector<std::byte> consumer_before =
            encode_logical(destination_spec, 0x653);
    std::vector<std::byte> source_storage;
    std::vector<std::byte> staging_storage;
    std::vector<std::byte> destination_storage;
    std::vector<std::byte> consumer_storage;
    std::vector<std::byte> expected_staging_storage;
    std::vector<std::byte> expected_destination_storage;
    std::vector<std::byte> expected_consumer_storage;
    if (config.storage_oracle != nullptr) {
        source_storage = encode_cache_append_storage(
                source_spec, 0x651, kCacheAppendSourceSentinel);
        staging_storage = encode_cache_append_storage(
                source_spec, 0x654, kCacheAppendSourceSentinel);
        destination_storage = encode_cache_append_storage(
                destination_spec, 0x652, kCacheAppendDestinationSentinel);
        consumer_storage = encode_cache_append_storage(
                destination_spec, 0x653, std::byte{0x6D});
        config.storage_oracle->set_owner_spec(source_spec);
        config.storage_oracle->seed(source->view(), source_storage);
        config.storage_oracle->seed(staging->view(), staging_storage);
        config.storage_oracle->set_owner_spec(destination_spec);
        config.storage_oracle->seed(
                destination->view(), destination_storage);
        config.storage_oracle->seed(consumer->view(), consumer_storage);
        source_bytes = decode_standard_tiled_view(
                source->view(), source_spec, source_storage);
        destination_before = decode_standard_tiled_view(
                destination->view(), destination_spec, destination_storage);
        consumer_before = decode_standard_tiled_view(
                consumer->view(), destination_spec, consumer_storage);
    } else {
        copy_from_host(source->view(), source_bytes);
        copy_from_host(
                staging->view(),
                std::vector<std::byte>(
                        source_spec.logical_nbytes(),
                        kCacheAppendSourceSentinel));
        copy_from_host(destination->view(), destination_before);
        copy_from_host(consumer->view(), consumer_before);
    }
    const std::vector<std::byte> expected_destination =
            cache_append_expected_logical(
                    source_spec, destination_spec, 1, source_bytes,
                    destination_before);
    if (config.storage_oracle != nullptr) {
        expected_staging_storage = staging_storage;
        apply_standard_tiled_view(
                source->view(), source_spec, source_bytes,
                expected_staging_storage);
        expected_destination_storage = destination_storage;
        apply_cache_append_storage(
                staging->view(), source_spec, destination->view(),
                destination_spec, 1, expected_staging_storage,
                expected_destination_storage);
        expected_consumer_storage = consumer_storage;
        apply_standard_tiled_view(
                consumer->view(), destination_spec, expected_destination,
                expected_consumer_storage);
    }

    auto queue = config.devices.candidate.create_ops();
    const std::optional<CacheAppendStateSnapshot> pipeline_state_before =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*queue))
            : std::nullopt;
    const iom::WorkspaceRequirements requirements =
            query_cache_append_twice(
                    *queue, staging->view(), destination->view(), 1);
    if (pipeline_state_before) {
        CHECK(*pipeline_state_before == config.observation.snapshot(*queue));
    }
    auto workspace = cache_append_workspace(
            config.devices.candidate, requirements);
    iom::oid producer = 0;
    iom::oid append = 0;
    iom::oid consumer_token = 0;
    {
        const CacheAppendCaseWindow window(config.observer);
        producer = queue->copy(source->view(), staging->view());
        REQUIRE(iom::oid_is_token(producer));
        {
            const iom::TensorView temporary_source = staging->view();
            iom::TensorView temporary_destination = destination->view();
            append = submit_cache_append(
                    *queue, temporary_source, temporary_destination, 1,
                    workspace);
        }
        REQUIRE(iom::oid_is_token(append));
        consumer_token = queue->copy(
                destination->view(), consumer->view());
        REQUIRE(iom::oid_is_token(consumer_token));
        CHECK_LT(token_sequence(producer), token_sequence(append));
        CHECK_LT(token_sequence(append), token_sequence(consumer_token));
        queue->wait(consumer_token);
        queue->wait(consumer_token);
        require_logical_bytes(
                staging->view(), source_bytes,
                "cache append pipeline producer staging");
        require_logical_bytes(
                destination->view(), expected_destination,
                "cache append pipeline destination");
        require_logical_bytes(
                consumer->view(), expected_destination,
                "cache append pipeline consumer");
        if (config.storage_oracle != nullptr) {
            require_cache_append_storage(
                    *config.storage_oracle, source_spec, staging->view(),
                    expected_staging_storage,
                    "cache append pipeline producer staging storage");
            require_cache_append_storage(
                    *config.storage_oracle, destination_spec,
                    destination->view(), expected_destination_storage,
                    "cache append pipeline destination storage");
            require_cache_append_storage(
                    *config.storage_oracle, destination_spec, consumer->view(),
                    expected_consumer_storage,
                    "cache append pipeline consumer storage");
        }
    }

    // The same workspace owner is intentionally retained after the first
    // drain and supplied to a second accepted append.
    auto second_destination =
            config.devices.candidate.create_tensor(destination_spec);
    std::vector<std::byte> second_destination_storage;
    std::vector<std::byte> expected_second_storage;
    std::vector<std::byte> second_destination_before = destination_before;
    if (config.storage_oracle != nullptr) {
        second_destination_storage = encode_cache_append_storage(
                destination_spec, 0x655, kCacheAppendDestinationSentinel);
        config.storage_oracle->set_owner_spec(destination_spec);
        config.storage_oracle->seed(
                second_destination->view(), second_destination_storage);
        second_destination_before = decode_standard_tiled_view(
                second_destination->view(), destination_spec,
                second_destination_storage);
        expected_second_storage = second_destination_storage;
        apply_cache_append_storage(
                staging->view(), source_spec, second_destination->view(),
                destination_spec, 1, expected_staging_storage,
                expected_second_storage);
    } else {
        copy_from_host(second_destination->view(), second_destination_before);
    }
    const std::vector<std::byte> expected_second_destination =
            cache_append_expected_logical(
                    source_spec, destination_spec, 1, source_bytes,
                    second_destination_before);
    const std::optional<CacheAppendStateSnapshot> reuse_state_before =
            config.observation.snapshot
            ? std::optional<CacheAppendStateSnapshot>(
                      config.observation.snapshot(*queue))
            : std::nullopt;
    const iom::WorkspaceRequirements second_requirements =
            query_cache_append_twice(
                    *queue, staging->view(), second_destination->view(), 1);
    if (reuse_state_before) {
        CHECK(*reuse_state_before == config.observation.snapshot(*queue));
    }
    CHECK_EQ(requirements, second_requirements);
    iom::oid second_append = 0;
    {
        const CacheAppendCaseWindow window(config.observer);
        const iom::TensorView temporary_source = staging->view();
        iom::TensorView temporary_destination = second_destination->view();
        second_append = submit_cache_append(
                *queue, temporary_source, temporary_destination, 1, workspace);
        REQUIRE(iom::oid_is_token(second_append));
        CHECK_LT(token_sequence(consumer_token), token_sequence(second_append));
        queue->wait(second_append);
        queue->wait(second_append);
        require_logical_bytes(
                second_destination->view(), expected_second_destination,
                "cache append pipeline reused workspace");
        if (config.storage_oracle != nullptr) {
            require_cache_append_storage(
                    *config.storage_oracle, destination_spec,
                    second_destination->view(), expected_second_storage,
                    "cache append pipeline reused workspace storage");
        }
    }

    // Submit from temporary owner objects and release those owners before
    // draining. The consumer view is snapshotted while the owners are live;
    // completion must still observe the accepted append's retained storage.
    auto ephemeral_source = config.devices.candidate.create_tensor(source_spec);
    auto ephemeral_destination =
            config.devices.candidate.create_tensor(destination_spec);
    auto ephemeral_consumer =
            config.devices.candidate.create_tensor(destination_spec);
    std::vector<std::byte> ephemeral_source_storage;
    std::vector<std::byte> ephemeral_destination_storage;
    std::vector<std::byte> ephemeral_consumer_storage;
    std::vector<std::byte> expected_ephemeral_consumer_storage;
    if (config.storage_oracle != nullptr) {
        ephemeral_source_storage = encode_cache_append_storage(
                source_spec, 0x651, kCacheAppendSourceSentinel);
        ephemeral_destination_storage = encode_cache_append_storage(
                destination_spec, 0x652, kCacheAppendDestinationSentinel);
        ephemeral_consumer_storage = encode_cache_append_storage(
                destination_spec, 0x656, std::byte{0x6D});
        config.storage_oracle->set_owner_spec(source_spec);
        config.storage_oracle->seed(
                ephemeral_source->view(), ephemeral_source_storage);
        config.storage_oracle->set_owner_spec(destination_spec);
        config.storage_oracle->seed(
                ephemeral_destination->view(), ephemeral_destination_storage);
        config.storage_oracle->seed(
                ephemeral_consumer->view(), ephemeral_consumer_storage);
        expected_ephemeral_consumer_storage = ephemeral_consumer_storage;
        apply_standard_tiled_view(
                ephemeral_consumer->view(), destination_spec,
                expected_destination, expected_ephemeral_consumer_storage);
    } else {
        copy_from_host(ephemeral_source->view(), source_bytes);
        copy_from_host(ephemeral_destination->view(), destination_before);
        copy_from_host(
                ephemeral_consumer->view(),
                std::vector<std::byte>(
                        destination_spec.logical_nbytes(), std::byte{0x6D}));
    }
    auto ephemeral_queue = config.devices.candidate.create_ops();
    const iom::WorkspaceRequirements ephemeral_requirements =
            query_cache_append_twice(
                    *ephemeral_queue, ephemeral_source->view(),
                    ephemeral_destination->view(), 1);
    auto ephemeral_workspace = cache_append_workspace(
            config.devices.candidate, ephemeral_requirements);
    iom::oid ephemeral_append = 0;
    iom::oid ephemeral_consumer_token = 0;
    {
        const CacheAppendCaseWindow window(config.observer);
        const iom::TensorView temporary_source = ephemeral_source->view();
        iom::TensorView temporary_destination = ephemeral_destination->view();
        ephemeral_append = submit_cache_append(
                *ephemeral_queue, temporary_source, temporary_destination, 1,
                ephemeral_workspace);
        REQUIRE(iom::oid_is_token(ephemeral_append));
        const iom::TensorView destination_snapshot =
                ephemeral_destination->view();
        ephemeral_consumer_token = ephemeral_queue->copy(
                destination_snapshot, ephemeral_consumer->view());
        REQUIRE(iom::oid_is_token(ephemeral_consumer_token));
    }
    ephemeral_source.reset();
    ephemeral_destination.reset();
    ephemeral_workspace.reset();
    ephemeral_queue->wait(ephemeral_consumer_token);
    ephemeral_queue->wait(ephemeral_consumer_token);
    require_logical_bytes(
            ephemeral_consumer->view(), expected_destination,
            "cache append temporary owner lifetime");
    if (config.storage_oracle != nullptr) {
        require_cache_append_storage(
                *config.storage_oracle, destination_spec,
                ephemeral_consumer->view(), expected_ephemeral_consumer_storage,
                "cache append temporary owner lifetime storage");
    }
}


inline void run_cache_append_conformance(
        const CacheAppendConformanceConfig& config) {
    require_cache_append_capabilities(
            config.supported_types,
            config.devices.candidate.supported_data_types());
    run_cache_append_capability_rejections(config);

    for (const iom::DataType type : config.supported_types) {
        CAPTURE(static_cast<int>(type));
        for (const CacheAppendScenario& scenario : cache_append_scenarios()) {
            CAPTURE(scenario.label);
            const iom::TensorSpec owner_spec = scenario.source_spec(type);
            for (const CacheAppendViewCase& view_case :
                 cache_append_view_cases(owner_spec)) {
                CAPTURE(view_case.label);
                run_cache_append_view_case(
                        config, scenario, type, view_case,
                        static_cast<std::uint64_t>(type) * 1000
                                + scenario.rank());
            }
        }
    }

    for (const CacheAppendOrderedScenario& ordered :
         cache_append_ordered_scenarios()) {
        CAPTURE(ordered.label);
        run_cache_append_ordered_case(config, ordered, iom::DataType::BF16);
    }
    run_cache_append_admission_conformance(config);
    run_cache_append_pipeline_conformance(config);
    run_cache_append_failure_conformance(config);
}

}  // namespace iom_conformance

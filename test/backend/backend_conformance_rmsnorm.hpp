#pragma once

// Shared RMSNorm conformance harness (change 006-tinyllama / 05-rms-normalization / 10).
//
// One backend-neutral owner of every shared RMS normalization scenario of the
// `RMS normalization` contract in docs/BACKEND_CONTRACT.md. The harness never
// switches on `BackendKind` and includes no accelerator header, so a backend
// driver includes it unchanged and supplies only
//
//   * its own `ConformanceDevices` triple,
//   * the explicit span of leaves its port actually queues,
//   * an optional phase observer (the traffic gate that proves no allocation,
//     registration, submission, or transfer happens inside a case), and
//   * an optional backend-native storage observer used to poison physical tile
//     padding, which only the driver can reach.
//
// The expected values come from the independent oracle of
// `backend_conformance_rmsnorm_reference.hpp`: production RMSNorm, its codec,
// a backend kernel, or a host roundtrip never serve as their own oracle, and
// the fixed comparison policy (`matches`) is used verbatim rather than
// re-derived here.
//
// The harness drives the frozen operation exactly as a consumer would:
//
//   oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
//               float eps, RawWorkspaceView workspace = {}) noexcept;
//   WorkspaceRequirements rmsnorm_workspace_requirements(
//           const TensorView& x, const TensorView& scale,
//           const TensorView& out, float eps);
//
// `run_rmsnorm_conformance` runs the independent analytic self-check, the
// numeric reference matrix for every declared leaf (the oracle's own fixture
// classes, non-tile feature widths, the run boundary sizes, ranks two through
// eight with multiple independent leading planes, padded-physical invariance,
// and the accepted epsilon boundaries), and the common contract matrix (query
// purity and determinism, workspace policy, admission rejection without
// effects, alias rules, checked overflow, capability rejection, ownership and
// queue order, and retained post-acceptance failure identity).
//
// A port whose own `RMS normalization` record documents deviations from the
// frozen policy declares them through `RmsNormComparisonMode`, and every
// element such a declaration observes is counted in its
// `RmsNormComparisonRecord`; a contract-exact port declares nothing and the
// frozen comparison policy applies unchanged to every element. The
// accepted-failure scenario runs against the candidate's own queue through the
// driver's `RmsNormNativeFailureSeam`, and a backend whose testing seams cannot
// reach its RMSNorm path declares that gap explicitly instead of passing
// silently.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_oracle.hpp"
#include "backend/backend_conformance_rmsnorm_reference.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Driver capability statements.
// ---------------------------------------------------------------------------

// The nine applicable ordinary signed floating leaves. CPU, CUDA, and ROCm
// queue all nine.
inline constexpr std::array<iom::DataType, 9> kRmsNormAllLeafSpan = {
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN,
        iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::BF16,
        iom::DataType::F32,
        iom::DataType::F64,
};

// The eight non-`F64` leaves: the span of a port whose `F64` leaf is gated on
// a device fact the selected device does not report.
inline constexpr std::array<iom::DataType, 8> kRmsNormNonF64LeafSpan = {
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN,
        iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::BF16,
        iom::DataType::F32,
};

// The two leaves a native facility that cannot consume the seven encoded
// carriers can queue.
inline constexpr std::array<iom::DataType, 2> kRmsNormWideLeafSpan = {
        iom::DataType::BF16,
        iom::DataType::F32,
};

// The explicit empty span of an unported port.
inline constexpr std::span<const iom::DataType> kNoRmsNormSpan{};

// Accepted epsilon values: the whole finite nonnegative domain, including both
// boundaries. Zero is the quiet-NaN boundary of an all-zero row, and the
// largest finite float is the far end of the admitted range.
inline constexpr std::array<float, 4> kRmsNormValidEpsilon = {
        0.0F,
        std::numeric_limits<float>::denorm_min(),
        1.0e-5F,
        std::numeric_limits<float>::max()};

// Rejected epsilon values: negative values and NaN are outside the finite
// nonnegative range, so they are admission failures rather than device data.
inline constexpr std::array<float, 4> kRmsNormInvalidEpsilon = {
        -1.0F,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};

// The byte pattern a case pre-fills an output owner with, so an element the
// operation never wrote cannot cancel against an expected value.
inline constexpr std::byte kRmsNormOutputSentinel{0xA5};

// The byte patterns of the padded-physical runs. The two sentinels differ in
// every bit, so a reduction that reads tile padding cannot produce the same
// logical output for both.
inline constexpr std::byte kRmsNormPaddingPoison{0x5A};
inline constexpr std::byte kRmsNormPaddingClear{0x00};

// Declared comparison capability of one port. Exactly the deviations a port
// records in the `RMS normalization` section of docs/BACKEND_CONTRACT.md may
// be declared here; the frozen comparison policy of the independent oracle is
// unchanged for every other port and for every undeclared clause, and a
// declared deviation is observed and counted instead of silently skipped.
struct RmsNormComparisonMode {
    // Whether the port evaluates each row in the leaf's direct accumulator
    // domain and encodes the result exactly once (contract clause 2). A
    // native facility that computes the row at reduced fidelity declares it
    // here: a finite `F32` expectation is then compared at that declared
    // fidelity instead of the frozen `F32` bound, and every such element is
    // counted. No other backend may declare this.
    bool direct_accumulator = true;
    // Whether the port produces the contract's special value classes —
    // signed zeros, quiet NaNs, and infinities (contract clause 3). A native
    // facility that cannot declares it here: an expectation whose class is
    // not `finite` is observed and counted instead of asserted.
    bool special_values = true;
};

// Explicit record of every element a declared deviation observed rather than
// asserted. A contract-exact run leaves both counts at zero.
struct RmsNormComparisonRecord {
    std::size_t observed_precision_elements = 0;
    std::size_t observed_special_value_elements = 0;
};

// Backend-native accepted-failure seam of one driver. `arm` arms a native
// failure on the candidate's own real RMSNorm submission path and `clear`
// disarms it again (the backend's own `none` value); the optional `name`
// identifies the seam for the record. Consumption is proven behaviorally by
// the shared scenario rather than by an accessor: the identical request
// succeeds while the seam is disarmed, and with the seam armed the accepted
// token fails and keeps failing — a fault that never reached the RMSNorm path
// would leave that wait succeeding, and a fault that broke admission would not
// return a positive token at all. A driver whose backend exposes no testing
// seam that reaches its RMSNorm path supplies an empty seam and states the
// reason; the harness then records and prints that coverage gap instead of
// passing silently, and never fakes a native failure.
struct RmsNormNativeFailureSeam {
    std::function<void()> arm;
    std::function<void()> clear;
    // Seam identity for the record, e.g. "cuda event_record".
    std::string_view name;
    // Non-empty exactly when no seam reaches this backend's RMSNorm path.
    std::string_view unavailable_reason;

    [[nodiscard]] bool available() const noexcept {
        return static_cast<bool>(arm);
    }
};

// One driver's explicit RMSNorm conformance configuration. The declared span
// is the driver's own statement of what its port queues, never an echo of a
// runtime capability report; the harness fails when a declared leaf is not
// accepted or an undeclared applicable leaf is.
struct RmsNormConformanceConfig {
    // The backend-neutral device triple every conformance driver provides.
    ConformanceDevices devices;
    // The leaves this driver's port actually queues.
    std::span<const iom::DataType> supported_leaves;
    // Optional phase observer; the CPU driver passes its traffic gate.
    ConformanceObserver* observer = nullptr;
    // Optional backend-native storage observer. Without it the harness cannot
    // poison physical tile padding, and the padded-invariance case fails
    // instead of silently observing nothing.
    AcceleratorStorageOracle* native_storage = nullptr;
    // Declared deviations of this port, exactly as its own contract record
    // states them. A contract-exact port leaves every flag set.
    RmsNormComparisonMode comparison{};
    // Receives the explicit counts of the elements those deviations observed.
    RmsNormComparisonRecord* comparison_record = nullptr;
    // This driver's backend-native accepted-failure seam.
    RmsNormNativeFailureSeam native_failure{};
};

[[nodiscard]] inline bool rmsnorm_declares(
        std::span<const iom::DataType> leaves, iom::DataType leaf) {
    return std::find(leaves.begin(), leaves.end(), leaf) != leaves.end();
}

// The leaves of `candidates` the candidate device can actually store, which is
// the only set a fixture may instantiate. The device's own capability table is
// the existing, backend-neutral answer; a fixture never assumes a leaf the
// device would refuse to allocate.
[[nodiscard]] inline std::vector<iom::DataType> rmsnorm_storable(
        std::span<const iom::DataType> candidates,
        std::span<const iom::DataType> supported) {
    std::vector<iom::DataType> leaves;
    for (const iom::DataType leaf : candidates) {
        if (rmsnorm_declares(supported, leaf)) {
            leaves.push_back(leaf);
        }
    }
    return leaves;
}

// The first declared leaf the device can store. A driver whose declared span
// names no storable leaf has a broken declaration and is reported as such.
[[nodiscard]] inline iom::DataType rmsnorm_first_storable(
        std::span<const iom::DataType> declared,
        std::span<const iom::DataType> supported) {
    for (const iom::DataType leaf : declared) {
        if (rmsnorm_declares(supported, leaf)) {
            return leaf;
        }
    }
    throw std::invalid_argument(
            "the declared RMSNorm span names no leaf the device can store");
}

// ---------------------------------------------------------------------------
// Fixture geometry, seeding, and observation.
// ---------------------------------------------------------------------------

[[nodiscard]] inline iom::TensorSpec rmsnorm_spec(
        std::vector<std::size_t> dimensions, iom::DataType leaf) {
    return iom::TensorSpec{iom::TensorShape{std::move(dimensions)}, leaf};
}

// The `[leading..., R, F]` specification of one fixture.
[[nodiscard]] inline iom::TensorSpec rmsnorm_full_spec(
        std::span<const std::size_t> leading, std::size_t rows,
        std::size_t features, iom::DataType leaf) {
    std::vector<std::size_t> dimensions(leading.begin(), leading.end());
    dimensions.push_back(rows);
    dimensions.push_back(features);
    return rmsnorm_spec(std::move(dimensions), leaf);
}

// The checked plane count of one leading tuple.
[[nodiscard]] inline std::size_t rmsnorm_plane_count(
        std::span<const std::size_t> leading) {
    std::size_t planes = 1;
    for (const std::size_t extent : leading) {
        planes *= extent;
    }
    return planes;
}

// Contiguous row-major logical bytes of one raw leaf code per element, using
// the harness's least-significant-bit-first host encoding.
[[nodiscard]] inline std::vector<std::byte> rmsnorm_pack_codes(
        std::span<const std::uint64_t> codes, iom::DataType leaf) {
    const std::size_t bits = bits_of(leaf);
    std::vector<std::byte> bytes((codes.size() * bits + 7) / 8, std::byte{0});
    auto* base = reinterpret_cast<unsigned char*>(bytes.data());
    for (std::size_t index = 0; index < codes.size(); ++index) {
        write_bits(base, index * bits, bits, codes[index]);
    }
    return bytes;
}

// One raw leaf code of a contiguous logical image.
[[nodiscard]] inline std::uint64_t rmsnorm_code_at(
        std::span<const std::byte> image, iom::DataType leaf,
        std::size_t index) {
    const std::size_t bits = bits_of(leaf);
    if ((index + 1) * bits > image.size() * 8) {
        throw std::invalid_argument(
                "RMSNorm logical image is too small for its element index");
    }
    return read_storage_bits(image.data(), index * bits, bits);
}

// Canonical physical image of one logical fixture: every logical element
// receives its own raw code at the canonical slot of its coordinates, and
// every padded bit keeps the caller's sentinel pattern. The physical padding
// therefore carries a deliberate value a reduction must never read.
[[nodiscard]] inline std::vector<std::byte> rmsnorm_padded_image(
        const iom::TensorSpec& spec, std::span<const std::uint64_t> codes,
        std::byte padding) {
    spec.validate();
    const std::size_t bits = bits_of(spec.data_type);
    const std::size_t slots = canonical_padded_element_count(spec);
    if (codes.size() != spec.shape.element_count()) {
        throw std::invalid_argument(
                "an RMSNorm physical image needs one code per logical element");
    }
    if (spec.tiled_storage_nbytes() * 8 < slots * bits) {
        throw std::invalid_argument(
                "an RMSNorm physical image needs the canonical padded slots");
    }
    std::vector<std::byte> image(spec.tiled_storage_nbytes(), padding);
    const std::span<const std::size_t> dimensions = spec.shape.dimensions();
    std::vector<std::size_t> coordinates(dimensions.size());
    for (std::size_t linear = 0; linear < codes.size(); ++linear) {
        std::size_t rest = linear;
        for (std::size_t axis = dimensions.size(); axis-- > 0;) {
            coordinates[axis] = rest % dimensions[axis];
            rest /= dimensions[axis];
        }
        const std::size_t slot = canonical_layout_slot(
                spec, std::span<const std::size_t>{coordinates});
        write_storage_bits(image.data(), slot * bits, bits, codes[linear]);
    }
    return image;
}

// Text of one observed or expected leaf value for a failure report: the class
// and the raw encoding, so a class disagreement and a numeric disagreement are
// distinguishable.
[[nodiscard]] inline std::string rmsnorm_value_text(
        iom::DataType leaf, std::uint64_t bits) {
    const RmsNormReferenceClass value_class =
            leaf == iom::DataType::F64
                    ? rmsnorm_oracle::classify(
                              rmsnorm_oracle::decode_f64(bits))
                    : rmsnorm_oracle::classify(
                              rmsnorm_oracle::decode_f32(leaf, bits));
    std::string text(rmsnorm_reference_class_name(value_class));
    text += " bits=0x";
    constexpr char kDigits[] = "0123456789ABCDEF";
    const std::size_t width = bits_of(leaf);
    for (std::size_t index = (width + 3) / 4; index-- > 0;) {
        const std::size_t shift = index * 4;
        const unsigned nibble = shift >= 64
                ? 0u
                : static_cast<unsigned>((bits >> shift) & 0xFu);
        text.push_back(kDigits[nibble]);
    }
    return text;
}

// The frozen comparison policy of the independent oracle with one port's
// declared deviations applied. A declared deviation never loosens a
// comparison for another port: it either compares the element at the declared
// compute fidelity or observes it, and the caller reports the count.
[[nodiscard]] inline bool rmsnorm_matches_declared(
        iom::DataType leaf, std::uint64_t actual_bits,
        const RmsNormReferenceValue& expected,
        const RmsNormComparisonMode& mode,
        RmsNormComparisonRecord& record) noexcept {
    if (expected.value_class != RmsNormReferenceClass::finite) {
        if (mode.special_values) {
            return matches(leaf, actual_bits, expected);
        }
        ++record.observed_special_value_elements;
        return true;
    }
    if (mode.direct_accumulator || leaf != iom::DataType::F32) {
        return matches(leaf, actual_bits, expected);
    }
    // The declared reduced-fidelity path evaluates the row at BF16 compute
    // fidelity, so a finite `F32` result must stay within two adjacent BF16
    // destination encodings of the reference. The element is counted, and the
    // observed value is decoded and re-encoded independently of production.
    ++record.observed_precision_elements;
    const float actual = rmsnorm_oracle::decode_f32(
            iom::DataType::F32, actual_bits & 0xFFFFFFFFu);
    const float reference = rmsnorm_oracle::decode_f32(
            iom::DataType::F32, expected.bits & 0xFFFFFFFFu);
    const std::uint64_t actual_bf16 =
            rmsnorm_oracle::encode_f32(iom::DataType::BF16, actual) & 0xFFFFu;
    const std::uint64_t reference_bf16 =
            rmsnorm_oracle::encode_f32(iom::DataType::BF16, reference)
            & 0xFFFFu;
    return rmsnorm_oracle::ordered_distance(
                   actual_bf16, reference_bf16, 16)
           <= 2;
}

// Compares one logical output image element by element under the frozen policy
// of the independent oracle. The returned text is empty when every element
// conforms; otherwise it names the case, the first mismatches with both
// values, and the total count.
[[nodiscard]] inline std::string rmsnorm_compare(
        iom::DataType leaf, std::span<const std::byte> image,
        std::span<const RmsNormReferenceValue> expected,
        std::string_view label, const RmsNormComparisonMode& mode = {},
        RmsNormComparisonRecord* record = nullptr) {
    constexpr std::size_t kDetails = 4;
    RmsNormComparisonRecord local;
    RmsNormComparisonRecord& observed = record != nullptr ? *record : local;
    std::string details;
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::uint64_t observed_bits =
                rmsnorm_code_at(image, leaf, index);
        if (rmsnorm_matches_declared(
                    leaf, observed_bits, expected[index], mode, observed)) {
            continue;
        }
        ++mismatches;
        if (mismatches <= kDetails) {
            details += "\n  element " + std::to_string(index) + ": expected "
                       + rmsnorm_value_text(leaf, expected[index].bits)
                       + ", observed "
                       + rmsnorm_value_text(leaf, observed_bits);
        }
    }
    if (mismatches == 0) {
        return {};
    }
    return std::string(label) + ": " + std::to_string(mismatches) + " of "
           + std::to_string(expected.size())
           + " elements disagree with the independent RMSNorm oracle"
           + details;
}

// One fixture's comparison. It is a non-fatal check so that a nonconforming
// port reports its complete failing-case inventory in one run instead of
// stopping at the first fixture; the queued-submission infrastructure itself
// remains fatal, because a rejected or failing submission invalidates every
// later comparison.
inline void check_rmsnorm_image(
        iom::DataType leaf, std::span<const std::byte> image,
        std::span<const RmsNormReferenceValue> expected,
        std::string_view label, const RmsNormComparisonMode& mode = {},
        RmsNormComparisonRecord* record = nullptr) {
    const std::string report =
            rmsnorm_compare(leaf, image, expected, label, mode, record);
    CHECK_MESSAGE(report.empty(), report);
}

// Arms one case's observer window and disarms it on every exit path, including
// a failing assertion and a thrown exception. A case declares its window after
// its tensors and its seeding and closes it before its readbacks, so the gate
// guards exactly the queue phase and never stays armed across teardown.
class RmsNormCaseWindow final {
public:
    explicit RmsNormCaseWindow(ConformanceObserver* observer)
            : observer_(observer) {
        if (observer_ != nullptr) {
            observer_->setup_complete();
        }
    }
    ~RmsNormCaseWindow() {
        if (observer_ != nullptr) {
            observer_->case_complete();
        }
    }
    RmsNormCaseWindow(const RmsNormCaseWindow&) = delete;
    RmsNormCaseWindow& operator=(const RmsNormCaseWindow&) = delete;
    RmsNormCaseWindow(RmsNormCaseWindow&&) = delete;
    RmsNormCaseWindow& operator=(RmsNormCaseWindow&&) = delete;

private:
    ConformanceObserver* observer_;
};

// Qualifies the three live owner view specifications for the recognized
// non-`NONE` quantization probe and restores them to `NONE` when its scope
// ends, on every exit path — including a failing assertion and a thrown
// exception — so no operand is ever destroyed while its specification carries
// a recognized grouped format. The accelerator tensor destructors derive their
// storage extent from the owner specification inside a `noexcept` destructor,
// so an unrestored qualification would abort instead of reporting.
class RmsNormQuantizationQualification final {
public:
    RmsNormQuantizationQualification(
            iom::Tensor& x, iom::Tensor& scale, iom::Tensor& out)
            : specs_{qualify(x), qualify(scale), qualify(out)} {}

    ~RmsNormQuantizationQualification() {
        for (iom::TensorSpec* spec : specs_) {
            spec->quantization = iom::QuantizationFormat::NONE;
        }
    }

    RmsNormQuantizationQualification(const RmsNormQuantizationQualification&) =
            delete;
    RmsNormQuantizationQualification& operator=(
            const RmsNormQuantizationQualification&) = delete;
    RmsNormQuantizationQualification(RmsNormQuantizationQualification&&) =
            delete;
    RmsNormQuantizationQualification& operator=(
            RmsNormQuantizationQualification&&) = delete;

private:
    [[nodiscard]] static iom::TensorSpec* qualify(iom::Tensor& owner) {
        iom::TensorSpec& spec =
                const_cast<iom::TensorSpec&>(owner.view().spec());
        spec.quantization = iom::QuantizationFormat::OCP_MXFP4;
        return &spec;
    }

    std::array<iom::TensorSpec*, 3> specs_;
};

// Submits one RMSNorm request from views that die when this frame returns: the
// queue must snapshot the request metadata instead of retaining a borrowed
// view, so the wait outside this frame still observes the stored result.
[[nodiscard]] inline iom::oid submit_rmsnorm_from_temporary_views(
        iom::DeviceOps& queue, iom::Tensor& x, iom::Tensor& scale,
        iom::Tensor& out, float eps) {
    const iom::TensorView x_view = x.view();
    const iom::TensorView scale_view = scale.view();
    iom::TensorView out_view = out.view();
    return queue.rmsnorm(x_view, scale_view, out_view, eps);
}

// ---------------------------------------------------------------------------
// Shared doubles.
// ---------------------------------------------------------------------------

// The device-less nonempty workspace of the workspace-policy probe. The frozen
// requirement is exactly `{0, 1}`, so any supplied workspace view with a live
// owner must be rejected before dispatch, and only an owned view can prove
// that `RawWorkspaceView::empty` is the admitted predicate. A CPU device
// cannot manufacture positive scratch, so the probe owns the range itself.
class RmsNormWorkspaceDouble final : public iom::RawWorkspace {
public:
    RmsNormWorkspaceDouble(
            const iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

// The common `DeviceOps` RMSNorm path of one declared span: the shared
// admission, ownership, queue-order, and retained-failure double. It exercises
// exactly the common machinery a real port inherits, and it never touches
// native compute or native storage.
class CommonRmsNormQueue final : public iom::DeviceOps {
public:
    enum class Failure { none, post_acceptance };

    struct RmsNormRecord {
        std::uint64_t sequence = 0;
        iom::detail::BinaryEntryRegistration entries;
        bool retained_failure = false;
    };

    CommonRmsNormQueue(
            const iom::Device& device, std::span<const iom::DataType> declared)
            : iom::DeviceOps(device), declared_(declared) {}

    void inject_failure(Failure failure) noexcept {
        next_failure_ = failure;
    }

    [[nodiscard]] const std::vector<RmsNormRecord>& rmsnorm_records()
            const noexcept {
        return records_;
    }

    [[nodiscard]] const std::vector<std::string_view>& submissions()
            const noexcept {
        return submissions_;
    }

    [[nodiscard]] std::size_t registered_at(const void* address) const {
        return registry_.registry.snapshot_for(const_cast<void*>(address))
                .size();
    }

    // Plays the in-order completion of one accepted submission: an RMSNorm
    // record releases or invalidates its owners exactly as a proven-completion
    // worker would, and a view-less probe submission only completes its
    // sequence.
    void finish(std::uint64_t sequence) {
        for (const RmsNormRecord& record : records_) {
            if (record.sequence != sequence) {
                continue;
            }
            (void)iom::detail::release_or_invalidate_binary_entries(
                    registry_.registry, record.entries,
                    record.retained_failure, !record.retained_failure);
            complete(sequence);
            return;
        }
        if (std::find(plain_.begin(), plain_.end(), sequence)
                != plain_.end()) {
            complete(sequence);
            return;
        }
        throw std::invalid_argument("unknown common RMSNorm sequence");
    }

    // A view-less submission that observes queue identity and sequence
    // allocation without reusing an accepted operation.
    iom::oid probe() {
        const iom::oid token = submit([this](std::uint64_t) {
            submissions_.push_back("probe");
        });
        plain_.push_back(token_sequence(token));
        return token;
    }

protected:
    // The double's immutable capability is exactly its declared span, so the
    // shared facade observes the same categories the driver's own queue
    // reports for the same span.
    [[nodiscard]] bool rmsnorm_supported(
            iom::DataType data_type) const override {
        return rmsnorm_declares(declared_, data_type);
    }

    iom::oid rmsnorm_impl(const RmsnormRequest& request) override {
        const Failure failure = std::exchange(next_failure_, Failure::none);
        iom::detail::Fence fence;
        fence.invoke = [](const iom::detail::Fence&) noexcept {
            return iom::detail::FenceResult::pending();
        };
        return submit_rmsnorm(
                request, registry_, queue_id_, fence,
                [this, failure](
                        std::uint64_t sequence, const RmsnormRequest&,
                        iom::detail::BinaryEntryRegistration entries) {
                    records_.push_back(RmsNormRecord{
                            sequence, entries,
                            failure == Failure::post_acceptance});
                    submissions_.push_back("rmsnorm");
                    if (failure == Failure::post_acceptance) {
                        commit_failure(
                                sequence,
                                std::make_exception_ptr(std::runtime_error(
                                        "common RMSNorm retained failure")));
                    }
                });
    }

private:
    iom::detail::RegistryState registry_;
    iom::detail::QueueId queue_id_ =
            iom::detail::allocate_queue_id(registry_);
    std::span<const iom::DataType> declared_;
    Failure next_failure_ = Failure::none;
    std::vector<RmsNormRecord> records_;
    std::vector<std::string_view> submissions_;
    std::vector<std::uint64_t> plain_;
};

// ---------------------------------------------------------------------------
// Numeric reference conformance.
// ---------------------------------------------------------------------------

// Runs exactly one fixture through the candidate's queue: the `[leading...,
// R, F]` operands and the rank-two `[1, F]` scale are created, seeded from the
// case's own raw codes, submitted once with the empty workspace, and waited.
// `observed` receives the logical readback of the output.
inline void run_rmsnorm_fixture(
        const RmsNormConformanceConfig& config, iom::DeviceOps& queue,
        const RmsNormReferenceCase& reference_case,
        std::span<const std::size_t> leading,
        std::vector<std::byte>& observed) {
    const iom::DataType leaf = reference_case.data_type;
    const iom::TensorSpec spec = rmsnorm_full_spec(
            leading, reference_case.rows, reference_case.features, leaf);
    const iom::TensorSpec scale_spec =
            rmsnorm_spec({1, reference_case.features}, leaf);
    auto x = config.devices.candidate.create_tensor(spec);
    auto scale = config.devices.candidate.create_tensor(scale_spec);
    auto out = config.devices.candidate.create_tensor(spec);
    copy_from_host(
            x->view(), rmsnorm_pack_codes(reference_case.x_bits, leaf));
    copy_from_host(
            scale->view(),
            rmsnorm_pack_codes(reference_case.scale_bits, leaf));
    copy_from_host(
            out->view(),
            std::vector<std::byte>(
                    spec.logical_nbytes(), kRmsNormOutputSentinel));
    iom::oid token = 0;
    {
        const RmsNormCaseWindow window(config.observer);
        token = queue.rmsnorm(
                x->view(), scale->view(), out->view(), reference_case.eps);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue.wait(token));
    }
    observed = read_logical(out->view());
}

// One fixture of the oracle's own matrix, compared under the fixed policy. A
// declared leaf the queue rejects, or an expected value a real submission
// cannot reproduce, fails here: neither is numeric conformance. The label
// names the leaf, its fixture class, and the geometry, so one backend's
// failure inventory is readable without cross-referencing the fixture tables.
inline void run_rmsnorm_reference_case(
        const RmsNormConformanceConfig& config, iom::DeviceOps& queue,
        const RmsNormReferenceCase& reference_case,
        std::span<const std::size_t> leading,
        std::string_view geometry_label) {
    std::vector<std::byte> observed;
    run_rmsnorm_fixture(config, queue, reference_case, leading, observed);
    const std::vector<RmsNormReferenceValue> expected =
            evaluate(reference_case);
    check_rmsnorm_image(
            reference_case.data_type, observed, expected,
            std::string(rmsnorm_oracle::leaf_name(reference_case.data_type))
                    + " " + rmsnorm_reference_case_label(reference_case) + " "
                    + std::string(geometry_label),
            config.comparison, config.comparison_record);
}

// The rank-two-through-eight geometry sweep: one, two, four, and eight
// independent leading planes across the run boundary sizes and both non-tile
// feature widths.
struct RmsNormRankGeometry {
    std::vector<std::size_t> leading;
    std::size_t rows;
    std::size_t features;
};

[[nodiscard]] inline const std::array<RmsNormRankGeometry, 7>&
rmsnorm_rank_geometries() {
    static const std::array<RmsNormRankGeometry, 7> geometries{{
            {{}, 17, 17},
            {{2}, 1, 33},
            {{2, 1}, 15, 17},
            {{2, 1, 2}, 16, 33},
            {{2, 1, 2, 1}, 17, 17},
            {{2, 1, 2, 1, 2}, 17, 33},
            {{2, 1, 2, 1, 2, 1}, 16, 17},
    }};
    return geometries;
}

// One native-seeded fixture run: the physical padding of every operand carries
// a deliberate sentinel, and the logical readbacks plus the canonical physical
// image of the input are observed through the driver's own storage access.
struct RmsNormNativeRun {
    std::vector<std::byte> output;
    std::vector<std::byte> input;
    std::vector<std::byte> input_image;
};

inline void run_rmsnorm_native_fixture(
        const RmsNormConformanceConfig& config, iom::DeviceOps& queue,
        const RmsNormReferenceCase& reference_case,
        std::span<const std::size_t> leading, std::byte padding,
        RmsNormNativeRun& run) {
    AcceleratorStorageOracle& native = *config.native_storage;
    const iom::DataType leaf = reference_case.data_type;
    const iom::TensorSpec spec = rmsnorm_full_spec(
            leading, reference_case.rows, reference_case.features, leaf);
    const iom::TensorSpec scale_spec =
            rmsnorm_spec({1, reference_case.features}, leaf);
    auto x = config.devices.candidate.create_tensor(spec);
    auto scale = config.devices.candidate.create_tensor(scale_spec);
    auto out = config.devices.candidate.create_tensor(spec);
    native.set_owner_spec(spec);
    native.seed(
            x->view(),
            rmsnorm_padded_image(spec, reference_case.x_bits, padding));
    native.set_owner_spec(scale_spec);
    native.seed(
            scale->view(),
            rmsnorm_padded_image(
                    scale_spec, reference_case.scale_bits, padding));
    native.set_owner_spec(spec);
    native.seed(
            out->view(),
            std::vector<std::byte>(spec.tiled_storage_nbytes(), padding));
    iom::oid token = 0;
    {
        const RmsNormCaseWindow window(config.observer);
        token = queue.rmsnorm(
                x->view(), scale->view(), out->view(), reference_case.eps);
        REQUIRE(iom::oid_is_token(token));
        CHECK_NOTHROW(queue.wait(token));
    }
    run.input = read_logical(x->view());
    run.output = read_logical(out->view());
    native.set_owner_spec(spec);
    run.input_image = native.observe(x->view());
}

// Padded-physical invariance: the identical logical fixture runs twice with
// differently poisoned tile padding. Both logical outputs must equal the
// independent oracle and each other, the operation must not have written its
// input, and the perturbation itself is proven to have happened — a backend
// that read padding would change its result, and a harness whose seeding
// silently failed would report a vacuous pass instead.
inline void run_rmsnorm_padding_invariance(
        const RmsNormConformanceConfig& config, iom::DeviceOps& queue,
        const RmsNormReferenceCase& reference_case,
        std::span<const std::size_t> leading) {
    REQUIRE(config.native_storage != nullptr);
    const std::string label =
            std::string(rmsnorm_oracle::leaf_name(reference_case.data_type))
            + " " + rmsnorm_reference_case_label(reference_case)
            + " padded physical storage";
    RmsNormNativeRun poisoned;
    RmsNormNativeRun cleared;
    run_rmsnorm_native_fixture(
            config, queue, reference_case, leading, kRmsNormPaddingPoison,
            poisoned);
    run_rmsnorm_native_fixture(
            config, queue, reference_case, leading, kRmsNormPaddingClear,
            cleared);
    const std::vector<RmsNormReferenceValue> expected =
            evaluate(reference_case);
    CHECK_MESSAGE(
            poisoned.input_image != cleared.input_image,
            label << ": the two runs did not perturb the physical padding");
    const std::vector<std::byte> input_bytes = rmsnorm_pack_codes(
            reference_case.x_bits, reference_case.data_type);
    CHECK_MESSAGE(
            poisoned.input == input_bytes,
            label << ": the operation wrote its input operand");
    check_rmsnorm_image(
            reference_case.data_type, poisoned.output, expected,
            label + " poison 0x5A", config.comparison,
            config.comparison_record);
    check_rmsnorm_image(
            reference_case.data_type, cleared.output, expected,
            label + " poison 0x00", config.comparison,
            config.comparison_record);
    CHECK_MESSAGE(
            poisoned.output == cleared.output,
            label << ": the logical output depends on physical padding");
}

// The complete numeric matrix of every declared leaf: the oracle's own cases,
// the rank sweep, the accepted epsilon boundaries, and the padded-physical
// invariance runs.
inline void run_rmsnorm_reference_conformance(
        const RmsNormConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    const std::span<const iom::DataType> storable =
            candidate.supported_data_types();
    auto queue = candidate.create_ops();
    for (const iom::DataType leaf : config.supported_leaves) {
        REQUIRE_MESSAGE(
                rmsnorm_declares(storable, leaf),
                "a declared RMSNorm leaf the device cannot store is a "
                "configuration defect");
        // The oracle's fixture matrix: constant, signed-zero, zero-epsilon
        // zero-row, mixed, scaled, NaN row, infinity row, nonfinite scale,
        // destination saturation and underflow, accumulator overflow and
        // underflow, and the complete boundary-size sweep of independent
        // planes, runs `1,15,16,17`, and non-tile features.
        for (const RmsNormReferenceCase& reference_case :
             rmsnorm_reference_cases(leaf)) {
            const std::vector<std::size_t> leading =
                    reference_case.planes == 1
                            ? std::vector<std::size_t>{}
                            : std::vector<std::size_t>{
                                      reference_case.planes};
            run_rmsnorm_reference_case(
                    config, *queue, reference_case, leading,
                    "rank " + std::to_string(leading.size() + 2));
        }
        // Ranks two through eight, each with its own leading tuple and
        // boundary extents. The same pattern content reaches every rank, so a
        // leading-axis or plane-stride regression cannot hide.
        for (const RmsNormRankGeometry& geometry :
             rmsnorm_rank_geometries()) {
            const std::size_t planes =
                    rmsnorm_plane_count(geometry.leading);
            const RmsNormReferenceCase reference_case =
                    rmsnorm_oracle::make_case(
                            RmsNormReferenceCaseKind::boundary_sizes, leaf,
                            planes, geometry.rows, geometry.features, 1.0e-5F,
                            rmsnorm_oracle::pattern_content(
                                    planes, geometry.rows,
                                    geometry.features),
                            rmsnorm_oracle::scale_content(
                                    geometry.features));
            run_rmsnorm_reference_case(
                    config, *queue, reference_case, geometry.leading,
                    "rank " + std::to_string(geometry.leading.size() + 2)
                            + " planes " + std::to_string(planes));
        }
        // The accepted epsilon boundaries, with nonzero content so the
        // zero-epsilon run is the arithmetic boundary rather than the
        // all-zero quiet-NaN decision.
        for (const float eps : kRmsNormValidEpsilon) {
            constexpr std::size_t kPlanes = 2;
            constexpr std::size_t kRows = 2;
            constexpr std::size_t kFeatures = 33;
            const RmsNormReferenceCase reference_case =
                    rmsnorm_oracle::make_case(
                            RmsNormReferenceCaseKind::mixed, leaf, kPlanes,
                            kRows, kFeatures, eps,
                            rmsnorm_oracle::pattern_content(
                                    kPlanes, kRows, kFeatures),
                            rmsnorm_oracle::scale_content(kFeatures));
            run_rmsnorm_reference_case(
                    config, *queue, reference_case,
                    std::vector<std::size_t>{kPlanes},
                    "epsilon " + std::to_string(eps));
        }
        // The padded-physical runs: a closed-form ones row that proves the
        // divisor is the logical `F` rather than the padded width, and a
        // two-tile row of independently poisoned padding.
        {
            constexpr std::size_t kPlanes = 2;
            constexpr std::size_t kRows = 17;
            constexpr std::size_t kFeatures = 17;
            run_rmsnorm_padding_invariance(
                    config, *queue,
                    rmsnorm_oracle::make_case(
                            RmsNormReferenceCaseKind::ones, leaf, kPlanes,
                            kRows, kFeatures, 1.0e-5F,
                            std::vector<double>(
                                    kPlanes * kRows * kFeatures, 1.0),
                            std::vector<double>(kFeatures, 1.0)),
                    std::vector<std::size_t>{kPlanes});
        }
        run_rmsnorm_padding_invariance(
                config, *queue,
                rmsnorm_oracle::make_boundary_sizes_case(leaf, 17, 33),
                std::vector<std::size_t>{3});
    }
}

// ---------------------------------------------------------------------------
// Common contract conformance.
// ---------------------------------------------------------------------------

// Disarms an armed backend-native fault on every exit path, so a failing
// assertion can never leak an armed fault into a later case.
class RmsNormNativeFailureGuard final {
public:
    explicit RmsNormNativeFailureGuard(
            const RmsNormNativeFailureSeam& seam) noexcept
            : seam_(seam) {
        seam_.arm();
        armed_ = true;
    }
    ~RmsNormNativeFailureGuard() {
        if (armed_) {
            seam_.clear();
        }
    }
    RmsNormNativeFailureGuard(const RmsNormNativeFailureGuard&) = delete;
    RmsNormNativeFailureGuard& operator=(const RmsNormNativeFailureGuard&) =
            delete;
    RmsNormNativeFailureGuard(RmsNormNativeFailureGuard&&) = delete;
    RmsNormNativeFailureGuard& operator=(RmsNormNativeFailureGuard&&) = delete;

private:
    const RmsNormNativeFailureSeam& seam_;
    bool armed_ = false;
};

// The accepted-failure scenario of one backend-native seam, run against the
// candidate's own queue so the real RMSNorm path — not a test double — carries
// the failure. The contract requires a `noexcept` facade to return a positive
// token for accepted work and to keep that token failing forever after a
// runtime failure; the buffer content of a failed accepted request is
// unspecified and is therefore not asserted, while the token's failure
// identity, its repetition, and the queue's recovery are.
inline void run_rmsnorm_native_failure_conformance(
        const RmsNormConformanceConfig& config) {
    const RmsNormNativeFailureSeam& seam = config.native_failure;
    if (!seam.available()) {
        // A backend whose testing seams cannot reach its RMSNorm path is a
        // recorded coverage gap, never a silent pass: the driver must state
        // why, and the gap is printed with the exact missing seam.
        REQUIRE_MESSAGE(
                !seam.unavailable_reason.empty(),
                "a backend without a native RMSNorm accepted-failure seam "
                "must state the reason");
        std::printf(
                "rmsnorm-native-failure-record seam=%s coverage=unavailable "
                "reason=%.*s\n",
                std::string(seam.name).c_str(),
                static_cast<int>(seam.unavailable_reason.size()),
                seam.unavailable_reason.data());
        return;
    }
    iom::Device& candidate = config.devices.candidate;
    const std::span<const iom::DataType> storable =
            candidate.supported_data_types();
    const iom::DataType leaf =
            rmsnorm_first_storable(config.supported_leaves, storable);
    constexpr std::size_t kPlanes = 2;
    constexpr std::size_t kRows = 2;
    constexpr std::size_t kFeatures = 33;
    const RmsNormReferenceCase reference_case = rmsnorm_oracle::make_case(
            RmsNormReferenceCaseKind::mixed, leaf, kPlanes, kRows, kFeatures,
            1.0e-5F,
            rmsnorm_oracle::pattern_content(kPlanes, kRows, kFeatures),
            rmsnorm_oracle::scale_content(kFeatures));
    const std::vector<RmsNormReferenceValue> expected =
            evaluate(reference_case);
    const iom::TensorSpec spec =
            rmsnorm_spec({kPlanes, kRows, kFeatures}, leaf);
    const iom::TensorSpec scale_spec = rmsnorm_spec({1, kFeatures}, leaf);
    auto x = candidate.create_tensor(spec);
    auto scale = candidate.create_tensor(scale_spec);
    auto out = candidate.create_tensor(spec);
    copy_from_host(x->view(), rmsnorm_pack_codes(reference_case.x_bits, leaf));
    copy_from_host(
            scale->view(),
            rmsnorm_pack_codes(reference_case.scale_bits, leaf));
    copy_from_host(
            out->view(),
            std::vector<std::byte>(
                    spec.logical_nbytes(), kRmsNormOutputSentinel));
    auto queue = candidate.create_ops();
    std::printf(
            "rmsnorm-native-failure-record seam=%s coverage=native "
            "reason=\n",
            std::string(seam.name).c_str());
    // 1. The identical request completes while the seam is disarmed: the
    //    fixture, the queue, and the declared leaf are healthy, so the failure
    //    observed below can only come from the armed native fault.
    iom::oid healthy = 0;
    {
        const RmsNormCaseWindow window(config.observer);
        healthy = queue->rmsnorm(
                x->view(), scale->view(), out->view(), reference_case.eps);
        REQUIRE(iom::oid_is_token(healthy));
        CHECK_NOTHROW(queue->wait(healthy));
    }
    check_rmsnorm_image(
            leaf, read_logical(out->view()), expected,
            "unarmed native request", config.comparison,
            config.comparison_record);
    // 2. The armed native fault is consumed by the real RMSNorm path: the
    //    submission is still accepted with a positive token, and that token
    //    keeps reporting the identical established error.
    iom::oid failed = 0;
    {
        const RmsNormNativeFailureGuard guard(seam);
        const RmsNormCaseWindow window(config.observer);
        failed = queue->rmsnorm(
                x->view(), scale->view(), out->view(), reference_case.eps);
        REQUIRE(iom::oid_is_token(failed));
        CHECK_EQ(token_sequence(failed), token_sequence(healthy) + 1);
        // The accepted token failing here, and failing identically on every
        // repeat, is the consumption proof: an armed fault that never reached
        // the real RMSNorm path would leave this wait succeeding, and the
        // disarmed control above shows the fixture itself is healthy.
        expect_repeated_runtime_failure(*queue, failed);
    }
    // 3. Recovery: after the seam is disarmed the queue still accepts and
    //    completes work, and the failed token keeps its established error, so
    //    nothing was re-published as valid and no owner was released early.
    {
        const RmsNormCaseWindow window(config.observer);
        const iom::oid recovered = queue->rmsnorm(
                x->view(), scale->view(), out->view(), reference_case.eps);
        REQUIRE(iom::oid_is_token(recovered));
        CHECK_EQ(token_sequence(recovered), token_sequence(failed) + 1);
        CHECK_NOTHROW(queue->wait(recovered));
        expect_repeated_runtime_failure(*queue, failed);
    }
    check_rmsnorm_image(
            leaf, read_logical(out->view()), expected,
            "recovered native request", config.comparison,
            config.comparison_record);
}

// The query, workspace, admission, alias, overflow, capability, ownership,
// queue-order, and retained-failure cases. They run against the candidate's
// own tensors, its real queue, and the shared double, so they observe the
// common `DeviceOps` contract every port inherits without duplicating any
// backend's implementation.
inline void run_rmsnorm_common_conformance(
        const RmsNormConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const iom::oid overflow = iom::to_oid(iom::OidError::Overflow);
    const iom::WorkspaceRequirements zero{0, 1};
    const std::span<const iom::DataType> storable =
            candidate.supported_data_types();
    const iom::DataType leaf =
            rmsnorm_first_storable(config.supported_leaves, storable);
    constexpr std::size_t kPlanes = 2;
    constexpr std::size_t kRows = 17;
    constexpr std::size_t kFeatures = 33;
    const iom::TensorSpec spec = rmsnorm_spec({kPlanes, kRows, kFeatures}, leaf);
    const iom::TensorSpec scale_spec = rmsnorm_spec({1, kFeatures}, leaf);
    auto queue = candidate.create_ops();
    // The last token the real queue accepted, so in-order sequence allocation
    // is asserted against the observed predecessor instead of an absolute
    // number.
    iom::oid last_accepted = 0;

    // The pure requirement query: the exact `{0, 1}` policy, deterministic,
    // validated with the submission's own categories, and free of every
    // observable effect.
    {
        auto x = candidate.create_tensor(spec);
        auto scale = candidate.create_tensor(scale_spec);
        auto out = candidate.create_tensor(spec);
        auto foreign = config.devices.foreign.create_tensor(spec);
        const std::vector<std::byte> seeded =
                encode_logical(out->view().spec(), 0x51u);
        copy_from_host(out->view(), seeded);
        CommonRmsNormQueue queue_double(candidate, config.supported_leaves);
        {
            const RmsNormCaseWindow window(config.observer);
            for (const float eps : kRmsNormValidEpsilon) {
                CHECK(
                        queue_double.rmsnorm_workspace_requirements(
                                x->view(), scale->view(), out->view(), eps)
                        == zero);
            }
            CHECK(
                    queue_double.rmsnorm_workspace_requirements(
                            x->view(), scale->view(), out->view(), 1.0e-5F)
                    == queue_double.rmsnorm_workspace_requirements(
                            x->view(), scale->view(), out->view(), 1.0e-5F));
            // The query rejects malformed requests with the same categories
            // as the submission instead of reporting a requirement.
            for (const float eps : kRmsNormInvalidEpsilon) {
                CHECK_THROWS_AS(
                        (void)queue_double.rmsnorm_workspace_requirements(
                                x->view(), scale->view(), out->view(), eps),
                        std::invalid_argument);
            }
            CHECK_THROWS_AS(
                    (void)queue_double.rmsnorm_workspace_requirements(
                            foreign->view(), scale->view(), out->view(),
                            1.0e-5F),
                    std::invalid_argument);
            CHECK_THROWS_AS(
                    (void)queue_double.rmsnorm_workspace_requirements(
                            x->view(), x->view(), out->view(), 1.0e-5F),
                    std::invalid_argument);
        }
        CHECK(queue_double.rmsnorm_records().empty());
        CHECK(queue_double.submissions().empty());
        CHECK_EQ(
                queue_double.registered_at(x->view().native_handle()),
                std::size_t{0});
        CHECK_EQ(
                queue_double.registered_at(out->view().native_handle()),
                std::size_t{0});
        require_logical_bytes(out->view(), seeded, "output after pure queries");
        // The consumed sequence proves that no query registered, leased,
        // submitted, or otherwise changed queue state.
        CHECK_EQ(token_sequence(queue_double.probe()), 1);
    }

    // A supplied workspace is rejected before dispatch. The frozen
    // requirement is `{0, 1}`, so only the empty default view is admissible,
    // and even a zero-byte view with a live owner is a supplied workspace.
    {
        auto x = candidate.create_tensor(spec);
        auto scale = candidate.create_tensor(scale_spec);
        auto out = candidate.create_tensor(spec);
        const std::vector<std::byte> seeded =
                encode_logical(out->view().spec(), 0x52u);
        copy_from_host(out->view(), seeded);
        iom::TensorView x_view = x->view();
        const iom::TensorView scale_view = scale->view();
        iom::TensorView out_view = out->view();
        std::array<std::byte, 64> scratch{};
        RmsNormWorkspaceDouble nonempty(
                candidate, scratch.data(), scratch.size());
        RmsNormWorkspaceDouble zero_bytes(candidate, scratch.data(), 0);
        // Capability precedes workspace admission, so a nonempty workspace
        // never masks an unported leaf. The probes are allocated before the
        // observer window: an operand is never created inside an armed gate.
        const std::vector<iom::DataType> rejected = rmsnorm_storable(
                kRmsNormUnsupportedDataTypes, storable);
        std::unique_ptr<iom::Tensor> unsupported_x;
        std::unique_ptr<iom::Tensor> unsupported_scale;
        std::unique_ptr<iom::Tensor> unsupported_out;
        if (!rejected.empty()) {
            unsupported_x = candidate.create_tensor(
                    rmsnorm_spec(
                            {kPlanes, kRows, kFeatures}, rejected.front()));
            unsupported_scale = candidate.create_tensor(
                    rmsnorm_spec({1, kFeatures}, rejected.front()));
            unsupported_out = candidate.create_tensor(
                    rmsnorm_spec(
                            {kPlanes, kRows, kFeatures}, rejected.front()));
        }
        CommonRmsNormQueue queue_double(candidate, config.supported_leaves);
        {
            const RmsNormCaseWindow window(config.observer);
            CHECK_EQ(
                    queue->rmsnorm(
                            x_view, scale_view, out_view, 1.0e-5F,
                            nonempty.view()),
                    invalid);
            CHECK_EQ(
                    queue->rmsnorm(
                            x_view, scale_view, out_view, 1.0e-5F,
                            zero_bytes.view()),
                    invalid);
            CHECK_EQ(
                    queue_double.rmsnorm(
                            x_view, scale_view, out_view, 1.0e-5F,
                            nonempty.view()),
                    invalid);
            if (unsupported_x != nullptr) {
                iom::TensorView unsupported_x_view = unsupported_x->view();
                const iom::TensorView unsupported_scale_view =
                        unsupported_scale->view();
                iom::TensorView unsupported_out_view = unsupported_out->view();
                CHECK_EQ(
                        queue->rmsnorm(
                                unsupported_x_view, unsupported_scale_view,
                                unsupported_out_view, 1.0e-5F,
                                nonempty.view()),
                        unsupported);
            }
        }
        CHECK(queue_double.rmsnorm_records().empty());
        require_logical_bytes(
                out->view(), seeded, "output after workspace rejection");
    }

    // The admission matrix. Every rejection reports its established category,
    // mutates no output byte, and consumes no token; the double additionally
    // proves that no owner was registered and no submission was recorded.
    {
        auto x = candidate.create_tensor(spec);
        auto scale = candidate.create_tensor(scale_spec);
        auto out = candidate.create_tensor(spec);
        auto short_out = candidate.create_tensor(
                rmsnorm_spec({kPlanes, kRows - 1, kFeatures}, leaf));
        auto wide_scale = candidate.create_tensor(
                rmsnorm_spec({2, kFeatures}, leaf));
        auto narrow_scale = candidate.create_tensor(
                rmsnorm_spec({1, kFeatures - 1}, leaf));
        auto thin_scale = candidate.create_tensor(
                rmsnorm_spec({1, kFeatures, 1}, leaf));
        auto foreign = config.devices.foreign.create_tensor(spec);
        auto wide_owner = candidate.create_tensor(
                rmsnorm_spec({3, kRows, kFeatures}, leaf));
        // A scale of another leaf is a specification mismatch; the probe is
        // allocated up front so no operand is ever created inside the armed
        // observer window.
        std::unique_ptr<iom::Tensor> mixed_scale;
        for (const iom::DataType other : storable) {
            if (other != leaf) {
                mixed_scale = candidate.create_tensor(
                        rmsnorm_spec({1, kFeatures}, other));
                break;
            }
        }
        const std::vector<std::byte> seeded =
                encode_logical(out->view().spec(), 0x53u);
        const std::vector<std::byte> wide_seeded =
                encode_logical(wide_owner->view().spec(), 0x54u);
        copy_from_host(out->view(), seeded);
        copy_from_host(
                short_out->view(),
                encode_logical(short_out->view().spec(), 0x54u));
        CommonRmsNormQueue queue_double(candidate, config.supported_leaves);
        const auto reject =
                [&](const iom::TensorView& x_view,
                    const iom::TensorView& scale_view,
                    iom::TensorView& out_view, float eps, iom::oid expected,
                    std::string_view what) {
                    const std::vector<std::byte> before =
                            read_logical(out_view);
                    const iom::oid observed =
                            queue->rmsnorm(x_view, scale_view, out_view, eps);
                    CHECK_MESSAGE(
                            observed == expected,
                            what << ": expected category "
                                 << static_cast<unsigned long long>(expected)
                                 << ", observed "
                                 << static_cast<unsigned long long>(observed));
                    CHECK_MESSAGE(
                            read_logical(out_view) == before,
                            what << ": the rejection mutated the output");
                    CHECK_EQ(
                            queue_double.rmsnorm(
                                    x_view, scale_view, out_view, eps),
                            expected);
                    CHECK(queue_double.rmsnorm_records().empty());
                    CHECK(queue_double.submissions().empty());
                    CHECK_EQ(
                            queue_double.registered_at(
                                    x_view.native_handle()),
                            std::size_t{0});
                    CHECK_EQ(
                            queue_double.registered_at(
                                    out_view.native_handle()),
                            std::size_t{0});
                };
        {
            const RmsNormCaseWindow window(config.observer);
            iom::TensorView out_view = out->view();
            iom::TensorView short_out_view = short_out->view();
            const iom::TensorView scale_view = scale->view();
            const iom::TensorView wide_scale_view = wide_scale->view();
            const iom::TensorView narrow_scale_view = narrow_scale->view();
            const iom::TensorView thin_scale_view = thin_scale->view();
            const iom::TensorView foreign_view = foreign->view();
            iom::TensorView foreign_out_view = foreign->view();
            // Rank and extent bounds are rejected at full-shape formation,
            // before any tensor, allocator traffic, or native effect can
            // exist, so no rank-one or rank-nine request ever reaches the
            // facade.
            CHECK_THROWS_AS(
                    (iom::TensorShape{std::vector<std::size_t>{kFeatures}}),
                    std::invalid_argument);
            {
                std::vector<std::size_t> dims(9u, 1);
                dims[7] = kRows;
                dims[8] = kFeatures;
                CHECK_THROWS_AS(
                        (iom::TensorShape{dims}), std::invalid_argument);
            }
            // A zero runtime extent is a malformed view: it is rejected by
            // the checked operand path before any capability or queue effect.
            iom::TensorView zero_extent = x->view();
            const_cast<std::size_t*>(
                    zero_extent.spec().shape.dimensions().data())[0] = 0;
            reject(
                    zero_extent, scale_view, out_view, 1.0e-5F, invalid,
                    "a zero runtime extent is rejected");
            reject(
                    x->view(), scale_view, short_out_view, 1.0e-5F, invalid,
                    "a mismatched x/out shape is rejected");
            reject(
                    x->view(), thin_scale_view, out_view, 1.0e-5F, invalid,
                    "a rank-three scale is rejected");
            reject(
                    x->view(), wide_scale_view, out_view, 1.0e-5F, invalid,
                    "a [2,F] scale is rejected");
            reject(
                    x->view(), narrow_scale_view, out_view, 1.0e-5F, invalid,
                    "a scale of the wrong feature extent is rejected");
            reject(
                    foreign_view, scale_view, out_view, 1.0e-5F, invalid,
                    "a foreign-device input is rejected");
            reject(
                    x->view(), scale_view, foreign_out_view, 1.0e-5F, invalid,
                    "a foreign-device output is rejected");
            for (const float eps : kRmsNormInvalidEpsilon) {
                reject(
                        x->view(), scale_view, out_view, eps, invalid,
                        "a nonfinite or negative epsilon is rejected");
            }
            if (mixed_scale != nullptr) {
                const iom::TensorView mixed_scale_view = mixed_scale->view();
                reject(
                        x->view(), mixed_scale_view, out_view, 1.0e-5F,
                        invalid, "a scale of another leaf is rejected");
            }
            // Checked plane addressing overflows instead of wrapping: the
            // mutated stride keeps every earlier structural check satisfied,
            // so the checked view arithmetic is what fails.
            iom::TensorView wide = wide_owner->view();
            const_cast<std::size_t*>(wide.plane_strides().data())[0] =
                    std::numeric_limits<std::size_t>::max();
            iom::TensorView wide_out = wide_owner->view();
            reject(
                    wide, scale_view, wide_out, 1.0e-5F, overflow,
                    "a checked plane address overflow is rejected");
        }
        CHECK(queue_double.rmsnorm_records().empty());
        require_logical_bytes(
                out->view(), seeded, "output after admission rejections");

        // Quantization is a capability rejection, so all three operands carry
        // the same recognized format. The qualification is scoped and removed
        // again before any operand is destroyed.
        {
            auto grouped_x = candidate.create_tensor(spec);
            auto grouped_scale = candidate.create_tensor(scale_spec);
            auto grouped_out = candidate.create_tensor(spec);
            {
                const RmsNormQuantizationQualification qualified(
                        *grouped_x, *grouped_scale, *grouped_out);
                iom::TensorView grouped_out_view = grouped_out->view();
                CHECK_EQ(
                        queue->rmsnorm(
                                grouped_x->view(), grouped_scale->view(),
                                grouped_out_view, 1.0e-5F),
                        unsupported);
                CHECK_EQ(
                        queue_double.rmsnorm(
                                grouped_x->view(), grouped_scale->view(),
                                grouped_out_view, 1.0e-5F),
                        unsupported);
            }
        }

        // Every leaf of the frozen applicability matrix the driver does not
        // declare is a capability rejection on a valid-shaped request, never
        // coerced arithmetic: BOOL, the twelve integer leaves, and `F8_E8M0`
        // are inapplicable everywhere, and an applicable leaf the port does
        // not implement is equally `Unsupported`.
        std::vector<iom::DataType> rejected_leaves = rmsnorm_storable(
                kRmsNormUnsupportedDataTypes, storable);
        for (const iom::DataType applicable : kRmsNormApplicableDataTypes) {
            if (!rmsnorm_declares(config.supported_leaves, applicable)) {
                rejected_leaves.push_back(applicable);
            }
        }
        for (const iom::DataType rejected_leaf : rejected_leaves) {
            if (!rmsnorm_declares(storable, rejected_leaf)) {
                continue;
            }
            const iom::TensorSpec rejected_spec =
                    rmsnorm_spec({kPlanes, kRows, kFeatures}, rejected_leaf);
            const iom::TensorSpec rejected_scale =
                    rmsnorm_spec({1, kFeatures}, rejected_leaf);
            auto rejected_x = candidate.create_tensor(rejected_spec);
            auto rejected_scale_tensor =
                    candidate.create_tensor(rejected_scale);
            auto rejected_out = candidate.create_tensor(rejected_spec);
            iom::TensorView rejected_out_view = rejected_out->view();
            CHECK_EQ(
                    queue->rmsnorm(
                            rejected_x->view(), rejected_scale_tensor->view(),
                            rejected_out_view, 1.0e-5F),
                    unsupported);
            CHECK_THROWS_AS(
                    (void)queue->rmsnorm_workspace_requirements(
                            rejected_x->view(),
                            rejected_scale_tensor->view(),
                            rejected_out_view, 1.0e-5F),
                    std::runtime_error);
            CHECK_EQ(
                    queue_double.rmsnorm(
                            rejected_x->view(), rejected_scale_tensor->view(),
                            rejected_out_view, 1.0e-5F),
                    unsupported);
        }
        CHECK(queue_double.rmsnorm_records().empty());
        require_logical_bytes(
                out->view(), seeded, "output after capability rejections");

        // Output aliasing is rejected for every form, including a
        // transformed-looking window of the same owner; x/scale read overlap
        // stays valid. The probe owners are allocated before the observer
        // window.
        auto split = candidate.create_tensor(spec);
        auto alias_owner = candidate.create_tensor(
                rmsnorm_spec({1, kFeatures}, leaf));
        auto alias_out = candidate.create_tensor(
                rmsnorm_spec({1, kFeatures}, leaf));
        {
            const RmsNormCaseWindow window(config.observer);
            iom::TensorView x_view = x->view();
            const iom::TensorView scale_view = scale->view();
            iom::TensorView scale_out_view = scale->view();
            CHECK_EQ(
                    queue->rmsnorm(x_view, scale_view, x_view, 1.0e-5F),
                    invalid);
            CHECK_EQ(
                    queue->rmsnorm(x_view, scale_view, scale_out_view,
                                   1.0e-5F),
                    invalid);
            const iom::TensorView first_plane = split->view().select(0, 0);
            iom::TensorView second_plane = split->view().select(0, 1);
            CHECK_EQ(
                    queue->rmsnorm(first_plane, scale_view, second_plane,
                                   1.0e-5F),
                    invalid);
            const iom::oid aliased = queue->rmsnorm(
                    alias_owner->view(), alias_owner->view(),
                    alias_out->view(), 1.0e-5F);
            REQUIRE(iom::oid_is_token(aliased));
            CHECK_NOTHROW(queue->wait(aliased));
            last_accepted = aliased;
        }
        CHECK(queue_double.rmsnorm_records().empty());
        require_logical_bytes(
                out->view(), seeded, "output after alias rejections");
    }

    // Ownership, in-order queue behavior, snapshot lifetime, and retained
    // failure identity.
    {
        const iom::TensorSpec order_spec =
                rmsnorm_spec({kPlanes, 2, 8}, leaf);
        const iom::TensorSpec order_scale = rmsnorm_spec({1, 8}, leaf);
        const RmsNormReferenceCase order_case = rmsnorm_oracle::make_case(
                RmsNormReferenceCaseKind::mixed, leaf, kPlanes, 2, 8, 1.0e-5F,
                rmsnorm_oracle::pattern_content(kPlanes, 2, 8),
                rmsnorm_oracle::scale_content(8));
        const std::vector<RmsNormReferenceValue> expected =
                evaluate(order_case);
        auto x = candidate.create_tensor(order_spec);
        auto scale = candidate.create_tensor(order_scale);
        auto out = candidate.create_tensor(order_spec);
        auto consumer = candidate.create_tensor(order_spec);
        copy_from_host(
                x->view(), rmsnorm_pack_codes(order_case.x_bits, leaf));
        copy_from_host(
                scale->view(),
                rmsnorm_pack_codes(order_case.scale_bits, leaf));
        {
            const RmsNormCaseWindow window(config.observer);
            const iom::oid producer = queue->rmsnorm(
                    x->view(), scale->view(), out->view(), order_case.eps);
            REQUIRE(iom::oid_is_token(producer));
            REQUIRE(iom::oid_is_token(last_accepted));
            CHECK_EQ(
                    token_sequence(producer),
                    token_sequence(last_accepted) + 1);
            last_accepted = producer;
            // A consumer submitted behind the producer observes its stored
            // result once its own wait returns: the queue is in order.
            const iom::oid consumer_token =
                    queue->copy(out->view(), consumer->view());
            REQUIRE(iom::oid_is_token(consumer_token));
            CHECK_NOTHROW(queue->wait(producer));
            CHECK_NOTHROW(queue->wait(consumer_token));
            CHECK_NOTHROW(queue->wait(consumer_token));
        }
        {
            const std::vector<std::byte> observed =
                    read_logical(consumer->view());
            check_rmsnorm_image(
                    leaf, observed, expected,
                    "producer order through a queued consumer");
        }

        // Views die before the wait: the queue snapshotted the request
        // metadata instead of retaining the caller's borrowed views.
        {
            auto temp_x = candidate.create_tensor(order_spec);
            auto temp_scale = candidate.create_tensor(order_scale);
            auto temp_out = candidate.create_tensor(order_spec);
            copy_from_host(
                    temp_x->view(),
                    rmsnorm_pack_codes(order_case.x_bits, leaf));
            copy_from_host(
                    temp_scale->view(),
                    rmsnorm_pack_codes(order_case.scale_bits, leaf));
            const iom::oid token = submit_rmsnorm_from_temporary_views(
                    *queue, *temp_x, *temp_scale, *temp_out, order_case.eps);
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
            const std::vector<std::byte> observed =
                    read_logical(temp_out->view());
            check_rmsnorm_image(
                    leaf, observed, expected,
                    "request snapshot survives dying views");
        }

        // The double's registration, release, quarantine, and repeated
        // failure identity. Owners of a proven completion are released, and
        // the owners of an accepted failure stay retained so a repeated wait
        // observes the same established error.
        CommonRmsNormQueue queue_double(candidate, config.supported_leaves);
        {
            const RmsNormCaseWindow window(config.observer);
            const iom::oid accepted = queue_double.rmsnorm(
                    x->view(), scale->view(), out->view(), order_case.eps);
            REQUIRE(iom::oid_is_token(accepted));
            CHECK_EQ(token_sequence(accepted), 1);
            REQUIRE_EQ(queue_double.rmsnorm_records().size(), std::size_t{1});
            CHECK_EQ(
                    queue_double.rmsnorm_records().back().entries.count,
                    std::size_t{3});
            CHECK_EQ(
                    queue_double.registered_at(x->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue_double.registered_at(scale->view().native_handle()),
                    std::size_t{1});
            CHECK_EQ(
                    queue_double.registered_at(out->view().native_handle()),
                    std::size_t{1});
            // The query stays pure while the queue is occupied: no record, no
            // registration, and exactly one consumed sequence.
            CHECK(
                    queue_double.rmsnorm_workspace_requirements(
                            x->view(), scale->view(), out->view(),
                            order_case.eps)
                    == zero);
            CHECK_EQ(queue_double.rmsnorm_records().size(), std::size_t{1});
            CHECK_EQ(token_sequence(queue_double.probe()), 2);
            queue_double.finish(1);
            CHECK_NOTHROW(queue_double.wait(accepted));
            CHECK_EQ(
                    queue_double.registered_at(x->view().native_handle()),
                    std::size_t{0});
            CHECK_EQ(
                    queue_double.registered_at(out->view().native_handle()),
                    std::size_t{0});

            queue_double.inject_failure(
                    CommonRmsNormQueue::Failure::post_acceptance);
            const iom::oid failed = queue_double.rmsnorm(
                    x->view(), scale->view(), out->view(), order_case.eps);
            REQUIRE(iom::oid_is_token(failed));
            CHECK_EQ(token_sequence(failed), 3);
            queue_double.finish(token_sequence(failed));
            expect_repeated_runtime_failure(queue_double, failed);
            CHECK_EQ(
                    queue_double.registered_at(x->view().native_handle()),
                    std::size_t{1});
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------

// The complete shared RMSNorm suite for one driver: the independent oracle's
// analytic self-check, the numeric reference matrix, and the common contract
// matrix. A driver supplies setup, its own supported span, and its native
// access; no driver copies a scenario.
inline void run_rmsnorm_conformance(const RmsNormConformanceConfig& config) {
    REQUIRE_FALSE(config.supported_leaves.empty());
    // A declared deviation must be recorded: an unrecorded one would hide the
    // elements it observed, which is exactly what this harness must never do.
    const bool contract_exact = config.comparison.direct_accumulator
                                && config.comparison.special_values;
    REQUIRE(contract_exact == (config.comparison_record == nullptr));
    {
        const RmsNormReferenceSelfCheckReport report =
                rmsnorm_reference_self_check();
        CHECK(report.checks > 0);
        for (const std::string& failure : report.failures) {
            CHECK_MESSAGE(false, "independent RMSNorm oracle: " << failure);
        }
        REQUIRE(report.ok());
    }
    run_rmsnorm_reference_conformance(config);
    run_rmsnorm_native_failure_conformance(config);
    run_rmsnorm_common_conformance(config);
}

}  // namespace iom_conformance

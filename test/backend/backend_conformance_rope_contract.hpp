#pragma once

// Shared RoPE admission and failure conformance (change 006-tinyllama /
// 06-rotary-position-encoding / 10).
//
// The numerical/reference runner in backend_conformance_rope.hpp owns the
// split-half oracle, the high-precision reference, and the fixed tolerance
// policy. This header deliberately owns only the public contract around that
// operation: capability admission, workspace and query purity, error
// categories and their precedence, queue ordering, copied metadata,
// ownership, and retained asynchronous failures. No reference value, angle,
// sine, cosine, or tolerance is computed here; the position-zero identity it
// observes is the frozen contract's own bitwise clause.
//
// The header never switches on `BackendKind` and includes no accelerator
// header, so each backend driver includes it unchanged and supplies only
//
//   * its own `ConformanceDevices` triple,
//   * the explicit span of leaves its port actually queues (never an echo of
//     a runtime capability report),
//   * an optional phase observer (the traffic gate that proves no operand is
//     allocated inside a case), and
//   * its own backend-native accepted-failure seam, or an explicit record
//     that no seam reaches the port.

#include "backend_conformance_common.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// The fourteen recognized leaves that have no RoPE semantics. Applicable
// leaves a driver omits from its explicit supported span (SYCL `F64`) are
// rejected through the same runner below.
inline constexpr std::array<iom::DataType, 14>
        kRopeContractInapplicableDataTypes = {
                iom::DataType::BOOL,
                iom::DataType::I2, iom::DataType::U2,
                iom::DataType::I4, iom::DataType::U4,
                iom::DataType::I8, iom::DataType::U8,
                iom::DataType::I16, iom::DataType::U16,
                iom::DataType::I32, iom::DataType::U32,
                iom::DataType::I64, iom::DataType::U64,
                iom::DataType::F8_E8M0,
        };

// The exact admissible workspace requirement of the frozen RoPE contract.
inline constexpr iom::WorkspaceRequirements kRopeContractWorkspace{0, 1};

// The largest admitted `a + R - 1`, the binary32 position bound of the
// frozen contract.
inline constexpr std::size_t kRopeContractMaxPosition =
        (std::size_t{1} << 24) - 1;

// A backend-native accepted-failure seam. `arm` installs a native failure on
// the candidate's own real RoPE submission path and `clear` restores the
// backend's own `none` value. A driver whose backend exposes no seam that
// reaches its RoPE path supplies an empty seam and states why; the runner then
// records and prints that coverage gap instead of passing silently, and never
// fakes a native failure.
struct RopeNativeFailureSeam {
    std::function<void()> arm;
    std::function<void()> clear;
    // Seam identity for the record, e.g. "sycl_detail::SubmissionFault".
    std::string_view name;
    // Non-empty exactly when no seam reaches this backend's RoPE path.
    std::string_view unavailable_reason;

    [[nodiscard]] bool available() const noexcept {
        return static_cast<bool>(arm) && static_cast<bool>(clear);
    }
};

struct RopeContractConformanceConfig {
    ConformanceDevices devices;
    // The driver's declaration of the leaves its port queues. Every declared
    // leaf is submitted through the real queue by the runner below, and every
    // applicable leaf it omits must be rejected as unsupported.
    std::span<const iom::DataType> supported_leaves;
    // Optional phase observer; the CPU driver passes its traffic gate.
    ConformanceObserver* observer = nullptr;
    RopeNativeFailureSeam native_failure{};
};

namespace rope_contract_detail {

// Arms one observer only for the scope of a single case, so a failing
// assertion can never leave the gate armed for a later case.
class CaseWindow final {
public:
    explicit CaseWindow(ConformanceObserver* observer) : observer_(observer) {
        if (observer_ != nullptr) {
            observer_->setup_complete();
        }
    }

    ~CaseWindow() {
        if (observer_ != nullptr) {
            observer_->case_complete();
        }
    }

    CaseWindow(const CaseWindow&) = delete;
    CaseWindow& operator=(const CaseWindow&) = delete;

private:
    ConformanceObserver* observer_;
};

// The device-less workspace of the workspace-policy probe. The frozen
// requirement is exactly `{0, 1}`, so only the empty default view is
// admissible and any view with a live owner must be rejected before dispatch.
// A CPU device cannot manufacture positive scratch, so the probe owns the
// range itself and reports the caller's address.
class TestWorkspace final : public iom::RawWorkspace {
public:
    TestWorkspace(
            const iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }

    void* address_;
};

// Replaces one owner's complete specification and restores the exact saved
// specification on every exit path, including a failing assertion and a
// thrown transfer error, so no operand is ever destroyed while its metadata
// is malformed. An accelerator tensor destructor derives its storage extent
// from the owner specification inside a `noexcept` destructor, so an
// unrestored replacement would abort instead of reporting.
class OwnerSpecRestore final {
public:
    OwnerSpecRestore(iom::Tensor& owner, const iom::TensorSpec& replacement)
        : spec_(const_cast<iom::TensorSpec&>(owner.view().spec())),
          saved_(spec_) {
        spec_ = replacement;
    }

    ~OwnerSpecRestore() {
        spec_ = saved_;
    }

    OwnerSpecRestore(const OwnerSpecRestore&) = delete;
    OwnerSpecRestore& operator=(const OwnerSpecRestore&) = delete;

private:
    iom::TensorSpec& spec_;
    iom::TensorSpec saved_;
};

[[nodiscard]] inline bool contains(
        std::span<const iom::DataType> leaves, iom::DataType leaf) {
    return std::find(leaves.begin(), leaves.end(), leaf) != leaves.end();
}

[[nodiscard]] inline iom::TensorSpec make_spec(
        std::initializer_list<std::size_t> dimensions, iom::DataType leaf) {
    return iom::TensorSpec{
            iom::TensorShape{std::vector<std::size_t>(
                    dimensions.begin(), dimensions.end())},
            leaf};
}

// Every leaf that must be rejected: the fourteen semantically inapplicable
// leaves, plus every applicable leaf the driver does not declare.
[[nodiscard]] inline std::vector<iom::DataType> rejected_leaves(
        const RopeContractConformanceConfig& config) {
    const std::span<const iom::DataType> actual =
            config.devices.candidate.supported_data_types();
    std::vector<iom::DataType> result;
    for (const iom::DataType leaf : kRopeContractInapplicableDataTypes) {
        REQUIRE_MESSAGE(
                contains(actual, leaf),
                "the candidate capability table must retain every RoPE leaf");
        result.push_back(leaf);
    }
    for (const iom::DataType leaf : {
                 iom::DataType::F4_E2M1,
                 iom::DataType::F6_E2M3,
                 iom::DataType::F6_E3M2,
                 iom::DataType::F8_E4M3FN,
                 iom::DataType::F8_E5M2,
                 iom::DataType::F16,
                 iom::DataType::BF16,
                 iom::DataType::F32,
                 iom::DataType::F64,
         }) {
        REQUIRE_MESSAGE(
                contains(actual, leaf),
                "the candidate capability table must retain every RoPE leaf");
        if (!contains(config.supported_leaves, leaf)) {
            result.push_back(leaf);
        }
    }
    return result;
}

// The declared `F32` leaf, which every retained backend queues. The
// contract-level cases use it so a case never depends on a leaf a port gates.
[[nodiscard]] inline iom::DataType required_f32_leaf(
        const RopeContractConformanceConfig& config) {
    REQUIRE_MESSAGE(
            contains(config.supported_leaves, iom::DataType::F32),
            "every retained backend must declare the F32 RoPE path");
    return iom::DataType::F32;
}

// Each declared leaf is a real queued path: the pure query reports exactly
// `{0, 1}` twice without consuming a sequence, and the identical submission is
// then accepted as sequence one and waitable repeatedly.
inline void run_supported_leaf_paths(
        const RopeContractConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    const std::span<const iom::DataType> actual =
            candidate.supported_data_types();
    REQUIRE_FALSE(config.supported_leaves.empty());
    for (const iom::DataType leaf : config.supported_leaves) {
        CAPTURE(static_cast<int>(leaf));
        REQUIRE_MESSAGE(
                contains(actual, leaf),
                "a declared RoPE leaf is absent from the candidate capability table");
        auto input = candidate.create_tensor(make_spec({1, 1, 1, 16}, leaf));
        auto output = candidate.create_tensor(make_spec({1, 1, 1, 16}, leaf));
        // A fresh queue per leaf keeps the "no query consumed a sequence"
        // proof independent of every earlier declared leaf.
        auto queue = candidate.create_ops();
        {
            CaseWindow window(config.observer);
            CHECK(
                    queue->rope_workspace_requirements(
                            input->view(), output->view(), 0, 10000.0)
                    == kRopeContractWorkspace);
            CHECK(
                    queue->rope_workspace_requirements(
                            input->view(), output->view(), 0, 10000.0)
                    == kRopeContractWorkspace);
            const iom::oid token = queue->rope(
                    input->view(), output->view(), 0, 10000.0);
            REQUIRE(iom::oid_is_token(token));
            // The two pure queries consumed no sequence.
            CHECK_EQ(token_sequence(token), std::uint64_t{1});
            CHECK_NOTHROW(queue->wait(token));
            CHECK_NOTHROW(queue->wait(token));
        }
    }
}

// Every recognized-but-inapplicable leaf and every applicable leaf the driver
// does not declare reports the semantic capability rejection after admission,
// consumes no sequence, mutates no output, and shows the same category from
// the pure query. Structural and device faults keep their earlier categories.
inline void run_capability_rejections(
        const RopeContractConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    iom::Device& foreign = config.devices.foreign;
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid overflow = iom::to_oid(iom::OidError::Overflow);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const std::vector<iom::DataType> rejected = rejected_leaves(config);
    auto queue = candidate.create_ops();
    for (const iom::DataType leaf : rejected) {
        CAPTURE(static_cast<int>(leaf));
        const iom::TensorSpec leaf_spec = make_spec({1, 1, 1, 16}, leaf);
        auto input = candidate.create_tensor(leaf_spec);
        auto output = candidate.create_tensor(leaf_spec);
        const std::vector<std::byte> before = read_logical(output->view());
        {
            CaseWindow window(config.observer);
            CHECK_EQ(
                    queue->rope(input->view(), output->view(), 0, 10000.0),
                    unsupported);
        }
        CHECK(read_logical(output->view()) == before);
        CHECK_THROWS_AS(
                (void)queue->rope_workspace_requirements(
                        input->view(), output->view(), 0, 10000.0),
                std::runtime_error);
    }

    // The inapplicable capability is classified after every prior admission
    // check, so each earlier fault keeps its own category instead of becoming
    // Unsupported. `BOOL` is inapplicable on every retained backend.
    auto precedence_input = candidate.create_tensor(
            make_spec({2, 3, 2, 16}, iom::DataType::BOOL));
    auto precedence_output = candidate.create_tensor(
            make_spec({2, 3, 2, 16}, iom::DataType::BOOL));
    auto malformed_odd = candidate.create_tensor(
            make_spec({2, 3, 2, 15}, iom::DataType::BOOL));
    auto malformed_odd_output = candidate.create_tensor(
            make_spec({2, 3, 2, 15}, iom::DataType::BOOL));
    auto foreign_capability = foreign.create_tensor(
            make_spec({2, 3, 2, 16}, iom::DataType::BOOL));
    {
        CaseWindow window(config.observer);
        // Structural: an odd width precedes the inapplicable leaf.
        CHECK_EQ(
                queue->rope(
                        malformed_odd->view(), malformed_odd_output->view(), 0,
                        10000.0),
                invalid);
        // Parameter policy precedes the inapplicable leaf.
        CHECK_EQ(
                queue->rope(
                        precedence_input->view(), precedence_output->view(), 0,
                        0.5),
                invalid);
        // The position bound precedes the inapplicable leaf.
        CHECK_EQ(
                queue->rope(
                        precedence_input->view(), precedence_output->view(),
                        kRopeContractMaxPosition + 1, 10000.0),
                invalid);
        // The checked position range precedes the inapplicable leaf.
        CHECK_EQ(
                queue->rope(
                        precedence_input->view(), precedence_output->view(),
                        std::numeric_limits<std::size_t>::max(), 10000.0),
                overflow);
        // Exact device identity precedes the inapplicable leaf.
        CHECK_EQ(
                queue->rope(
                        precedence_input->view(),
                        foreign_capability->view(), 0, 10000.0),
                invalid);
        // Conservative output aliasing precedes the inapplicable leaf.
        CHECK_EQ(
                queue->rope(
                        precedence_input->view(), precedence_input->view(), 0,
                        10000.0),
                invalid);
    }

    // Every rejection above was negative and consumed no sequence: the first
    // accepted supported request on this queue is still sequence one.
    auto supported_input = candidate.create_tensor(
            make_spec({1, 1, 1, 16}, required_f32_leaf(config)));
    auto supported_output = candidate.create_tensor(
            make_spec({1, 1, 1, 16}, required_f32_leaf(config)));
    const iom::oid token = queue->rope(
            supported_input->view(), supported_output->view(), 0, 10000.0);
    REQUIRE(iom::oid_is_token(token));
    CHECK_EQ(token_sequence(token), std::uint64_t{1});
    CHECK_NOTHROW(queue->wait(token));
}

// Pure-query determinism and the fixed zero-workspace contract. A pure query
// consumes no sequence, and every supplied view with a live owner — local,
// zero-byte, foreign, overlapping, or stale — is rejected before dispatch.
inline void run_query_and_workspace_conformance(
        const RopeContractConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    iom::Device& foreign = config.devices.foreign;
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::TensorSpec f32_spec =
            make_spec({2, 3, 2, 16}, required_f32_leaf(config));

    {
        auto input = candidate.create_tensor(f32_spec);
        auto output = candidate.create_tensor(f32_spec);
        auto queue = candidate.create_ops();
        {
            CaseWindow window(config.observer);
            CHECK(
                    queue->rope_workspace_requirements(
                            input->view(), output->view(), 15, 1.25)
                    == kRopeContractWorkspace);
            CHECK(
                    queue->rope_workspace_requirements(
                            input->view(), output->view(), 15, 1.25)
                    == kRopeContractWorkspace);
            const iom::oid token = queue->rope(
                    input->view(), output->view(), 0, 1.0);
            REQUIRE(iom::oid_is_token(token));
            // Neither pure query registered, leased, allocated, or consumed a
            // sequence, so the first accepted request is sequence one.
            CHECK_EQ(token_sequence(token), std::uint64_t{1});
            CHECK_NOTHROW(queue->wait(token));
        }
    }

    {
        auto input = candidate.create_tensor(f32_spec);
        auto output = candidate.create_tensor(f32_spec);
        const std::vector<std::byte> before = read_logical(output->view());
        std::array<std::byte, 64> scratch{};
        // The only admissible supplied workspace is the empty default view, so
        // even a zero-byte view with a live owner is a supplied workspace.
        TestWorkspace nonempty(candidate, scratch.data(), scratch.size());
        TestWorkspace zero_bytes(candidate, scratch.data(), 0);
        TestWorkspace foreign_workspace(foreign, scratch.data(), scratch.size());
        TestWorkspace overlapping_workspace(
                candidate, const_cast<void*>(input->view().native_handle()),
                scratch.size());
        auto stale_owner = std::make_unique<TestWorkspace>(
                candidate, scratch.data(), scratch.size());
        const iom::RawWorkspaceView stale_view = stale_owner->view();
        stale_owner.reset();
        auto queue = candidate.create_ops();
        {
            CaseWindow window(config.observer);
            CHECK_EQ(
                    queue->rope(
                            input->view(), output->view(), 0, 1.0,
                            nonempty.view()),
                    invalid);
            CHECK_EQ(
                    queue->rope(
                            input->view(), output->view(), 0, 1.0,
                            zero_bytes.view()),
                    invalid);
            CHECK_EQ(
                    queue->rope(
                            input->view(), output->view(), 0, 1.0,
                            foreign_workspace.view()),
                    invalid);
            CHECK_EQ(
                    queue->rope(
                            input->view(), output->view(), 0, 1.0,
                            overlapping_workspace.view()),
                    invalid);
            CHECK_EQ(
                    queue->rope(
                            input->view(), output->view(), 0, 1.0, stale_view),
                    invalid);
        }
        CHECK(read_logical(output->view()) == before);
        const iom::oid token = queue->rope(
                input->view(), output->view(), 0, 1.0);
        REQUIRE(iom::oid_is_token(token));
        CHECK_EQ(token_sequence(token), std::uint64_t{1});
        CHECK_NOTHROW(queue->wait(token));
    }
}

// The admission matrix: rank and extent bounds, tuple/dtype/layout/
// quantization/device/alias faults, unknown enumeration values, the parameter
// and position boundaries, and the checked position overflow. Every case is a
// negative category with no output mutation and no consumed sequence, and the
// exact admitted boundary is observed on a fresh queue.
inline void run_admission_matrix(
        const RopeContractConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    iom::Device& foreign = config.devices.foreign;
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid overflow = iom::to_oid(iom::OidError::Overflow);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const iom::TensorSpec f32_spec =
            make_spec({2, 3, 2, 16}, required_f32_leaf(config));
    auto input = candidate.create_tensor(f32_spec);
    auto output = candidate.create_tensor(f32_spec);
    auto foreign_input = foreign.create_tensor(f32_spec);
    auto foreign_output = foreign.create_tensor(f32_spec);
    auto leading_output = candidate.create_tensor(
            make_spec({4, 3, 2, 16}, iom::DataType::F32));
    auto short_output = candidate.create_tensor(
            make_spec({2, 3, 1, 16}, iom::DataType::F32));
    auto mixed_output = candidate.create_tensor(
            make_spec({2, 3, 2, 16}, iom::DataType::F16));
    auto odd_input = candidate.create_tensor(
            make_spec({2, 3, 2, 15}, iom::DataType::F32));
    auto odd_output = candidate.create_tensor(
            make_spec({2, 3, 2, 15}, iom::DataType::F32));
    auto rank_two_input = candidate.create_tensor(
            make_spec({2, 16}, iom::DataType::F32));
    auto rank_two_output = candidate.create_tensor(
            make_spec({2, 16}, iom::DataType::F32));
    auto native_rank_input = candidate.create_tensor(
            make_spec({2, 3, 2}, iom::DataType::F32));
    auto native_rank_output = candidate.create_tensor(
            make_spec({2, 3, 2}, iom::DataType::F32));
    auto two_row_input = candidate.create_tensor(
            make_spec({2, 3, 2, 16}, iom::DataType::F32));
    auto two_row_output = candidate.create_tensor(
            make_spec({2, 3, 2, 16}, iom::DataType::F32));
    // `R = 1` puts every logical element at the explicit position `a`, so the
    // admitted requests below are the frozen position-zero bitwise identity
    // for the whole tensor and no angle or trigonometric value is recomputed
    // here.
    auto identity_input = candidate.create_tensor(
            make_spec({2, 3, 1, 16}, iom::DataType::F32));
    auto identity_output = candidate.create_tensor(
            make_spec({2, 3, 1, 16}, iom::DataType::F32));
    auto quantized_input = candidate.create_tensor(f32_spec);
    auto quantized_output = candidate.create_tensor(f32_spec);

    // Every operand is created before the phase observer is armed: a case may
    // never allocate or release tensor storage through the device allocator.
    const std::vector<std::byte> seeded =
            encode_logical(identity_input->view().spec(), 0x7Au);
    copy_from_host(identity_input->view(), seeded);
    copy_from_host(
            identity_output->view(),
            std::vector<std::byte>(
                    identity_output->view().spec().logical_nbytes(),
                    kReadbackSentinel));
    const std::vector<std::byte> quantized_before =
            read_logical(quantized_output->view());
    auto queue = candidate.create_ops();
    const auto reject = [&](const iom::TensorView& x_view,
                            iom::TensorView& out_view, iom::oid expected,
                            std::string_view label) {
        const std::vector<std::byte> before = read_logical(out_view);
        const iom::oid observed = queue->rope(x_view, out_view, 0, 1.0);
        CHECK_MESSAGE(observed == expected, label);
        CHECK_MESSAGE(read_logical(out_view) == before, label);
    };
    // The same rejection, submitted with an explicit position and base so the
    // parameter and position categories are exercised directly.
    const auto reject_with = [&](const iom::TensorView& x_view,
                                 iom::TensorView& out_view, std::size_t a,
                                 double theta, iom::oid expected,
                                 std::string_view label) {
        const std::vector<std::byte> before = read_logical(out_view);
        CHECK_MESSAGE(
                queue->rope(x_view, out_view, a, theta) == expected, label);
        CHECK_MESSAGE(read_logical(out_view) == before, label);
    };

    {
        CaseWindow window(config.observer);

        // Rank two is the reachable lower rank of a materialized owner, and
        // rank nine is rejected while the full shape is formed, before any
        // owner, storage, or queue state can exist.
        reject(
                rank_two_input->view(), rank_two_output->view(), invalid,
                "rank two");
        std::vector<std::size_t> rank_nine(9, 1);
        rank_nine[7] = 2;
        rank_nine[8] = 16;
        CHECK_THROWS_AS((iom::TensorShape{rank_nine}), std::invalid_argument);

        // Three is the smallest admitted rank.
        const iom::oid native_rank = queue->rope(
                native_rank_input->view(), native_rank_output->view(), 0, 1.0);
        CHECK(iom::oid_is_token(native_rank));
        CHECK_EQ(token_sequence(native_rank), std::uint64_t{1});

        reject(input->view(), leading_output->view(), invalid, "leading tuple");
        reject(odd_input->view(), odd_output->view(), invalid, "odd D");
        reject(input->view(), short_output->view(), invalid, "shape mismatch");
        reject(input->view(), mixed_output->view(), invalid, "dtype mismatch");
        reject(input->view(), foreign_output->view(), invalid, "foreign output");
        reject(foreign_input->view(), output->view(), invalid, "foreign input");
        reject(input->view(), input->view(), invalid, "same-owner alias");

        // A pair of disjoint-looking views of one owner is still a
        // conservative output alias and must not be admitted.
        const iom::TensorView first_plane = input->view().select(0, 0);
        iom::TensorView second_plane = input->view().select(0, 1);
        reject(
                first_plane, second_plane, invalid, "same-owner plane overlap");

        // Zero H, R, a leading extent, and D are malformed metadata. The
        // owner's full view is never changed: a copied view carries the
        // malformed metadata for exactly the rejected request.
        for (const std::size_t axis : {
                     std::size_t{0}, std::size_t{1}, std::size_t{2},
                     std::size_t{3}}) {
            iom::TensorView zero_extent = input->view();
            const_cast<std::size_t*>(
                    zero_extent.spec().shape.dimensions().data())[axis] = 0;
            reject(zero_extent, output->view(), invalid, "zero extent");
        }

        // An unknown leaf value and an unknown quantization value are
        // malformed input rather than a capability miss.
        {
            iom::TensorSpec unknown_leaf = input->view().spec();
            unknown_leaf.data_type = static_cast<iom::DataType>(255);
            const OwnerSpecRestore restore(*input, unknown_leaf);
            reject(input->view(), output->view(), invalid, "unknown leaf");
        }
        {
            iom::TensorSpec unknown_quantization = input->view().spec();
            unknown_quantization.quantization =
                    static_cast<iom::QuantizationFormat>(9999);
            const OwnerSpecRestore restore(*input, unknown_quantization);
            reject(
                    input->view(), output->view(), invalid,
                    "unknown quantization");
        }

        // A zero plane stride is malformed layout. The copied view restores
        // itself and never changes the owner's metadata.
        iom::TensorView malformed_layout = input->view();
        std::size_t* malformed_stride = const_cast<std::size_t*>(
                malformed_layout.plane_strides().data());
        const std::size_t saved_stride = malformed_stride[0];
        malformed_stride[0] = 0;
        reject(malformed_layout, output->view(), invalid, "zero plane stride");
        malformed_stride[0] = saved_stride;

        // One recognized grouped quantization against `NONE` is a
        // specification mismatch; the same recognized format on both sides is
        // a semantic capability rejection after every earlier check. The
        // qualification is restored before the request returns in every case.
        {
            iom::TensorSpec grouped = quantized_input->view().spec();
            grouped.quantization = iom::QuantizationFormat::OCP_MXFP4;
            const OwnerSpecRestore restore(*quantized_input, grouped);
            reject(
                    quantized_input->view(), quantized_output->view(), invalid,
                    "quantization mismatch");
        }
        {
            iom::TensorSpec grouped = quantized_input->view().spec();
            grouped.quantization = iom::QuantizationFormat::OCP_MXFP4;
            iom::TensorSpec grouped_out = quantized_output->view().spec();
            grouped_out.quantization = iom::QuantizationFormat::OCP_MXFP4;
            const OwnerSpecRestore restore_input(*quantized_input, grouped);
            const OwnerSpecRestore restore_output(*quantized_output, grouped_out);
            CHECK_EQ(
                    queue->rope(
                            quantized_input->view(), quantized_output->view(),
                            0, 1.0),
                    unsupported);
            CHECK_THROWS_AS(
                    (void)queue->rope_workspace_requirements(
                            quantized_input->view(), quantized_output->view(),
                            0, 1.0),
                    std::runtime_error);
        }
        // The rejected requests mutated no output byte under qualification.
        CHECK(read_logical(quantized_output->view()) == quantized_before);

        // The admissible parameter domain is exactly `[1, float_max]`. Every
        // admitted request here is the position-zero identity of the seeded
        // input, which the check after this window observes.
        const double float_max =
                static_cast<double>(std::numeric_limits<float>::max());
        for (const double theta : {
                 float_max,
                 1.0,
                 1.25,
         }) {
            const iom::oid accepted = queue->rope(
                    identity_input->view(), identity_output->view(), 0, theta);
            CHECK(iom::oid_is_token(accepted));
            CHECK_NOTHROW(queue->wait(accepted));
        }
        for (const double theta : {
                 0.5,
                 -0.0,
                 std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::quiet_NaN(),
                 float_max * 2.0,
         }) {
            reject_with(
                    identity_input->view(), identity_output->view(), 0, theta,
                    invalid, "invalid theta");
        }

        // The checked position range: the whole-range overflow is an
        // Overflow, a representable position above the bound is invalid, and
        // the exact admitted maximum is `a + R - 1`.
        CHECK_THROWS_AS(
                (void)queue->rope_workspace_requirements(
                        two_row_input->view(), two_row_output->view(),
                        std::numeric_limits<std::size_t>::max(), 1.0),
                std::overflow_error);
        reject_with(
                two_row_input->view(), two_row_output->view(),
                std::numeric_limits<std::size_t>::max(), 1.0, overflow,
                "position range overflow");
        reject_with(
                input->view(), output->view(), kRopeContractMaxPosition + 1,
                1.0, invalid, "position above the bound");
        reject_with(
                two_row_input->view(), two_row_output->view(),
                kRopeContractMaxPosition, 1.0, invalid,
                "position bound with R=2");
        // The exact admitted `a + R - 1` boundary is valid; the position
        // above the bound is rejected without touching this request.
        reject_with(
                identity_input->view(), identity_output->view(),
                kRopeContractMaxPosition + 1, 1.0, invalid,
                "position above the bound");
        const iom::oid boundary = queue->rope(
                two_row_input->view(), two_row_output->view(),
                kRopeContractMaxPosition - 1, 1.0);
        CHECK(iom::oid_is_token(boundary));
        CHECK_NOTHROW(queue->wait(boundary));

        // The admitted requests wrote the frozen position-zero identity of
        // the seeded input over the sentinel-filled output.
        CHECK(read_logical(identity_output->view()) == seeded);
    }
}

// Queue ordering, independent Q/K calls, copied metadata, owner destruction,
// and the release/reuse transition after a proven completion.
inline void run_owner_and_queue_conformance(
        const RopeContractConformanceConfig& config) {
    iom::Device& candidate = config.devices.candidate;
    const iom::TensorSpec query_spec =
            make_spec({2, 2, 3, 16}, iom::DataType::F32);
    const iom::TensorSpec key_spec =
            make_spec({2, 1, 3, 16}, iom::DataType::F32);

    // Q and K are independent calls with unequal head counts, and acceptance
    // order is the in-order OID order.
    {
        auto query = candidate.create_tensor(query_spec);
        auto query_output = candidate.create_tensor(query_spec);
        auto key = candidate.create_tensor(key_spec);
        auto key_output = candidate.create_tensor(key_spec);
        auto queue = candidate.create_ops();
        {
            CaseWindow window(config.observer);
            const iom::oid query_token = queue->rope(
                    query->view(), query_output->view(), 0, 10000.0);
            const iom::oid key_token = queue->rope(
                    key->view(), key_output->view(), 3, 10000.0);
            REQUIRE(iom::oid_is_token(query_token));
            REQUIRE(iom::oid_is_token(key_token));
            CHECK_EQ(
                    token_sequence(key_token),
                    token_sequence(query_token) + std::uint64_t{1});
            CHECK_NOTHROW(queue->wait(query_token));
            CHECK_NOTHROW(queue->wait(key_token));
            CHECK_NOTHROW(queue->wait(key_token));
        }
    }

    // The accepted request keeps a value-copied view snapshot and its owner
    // registration: neither a later edit of the caller's view copy nor the
    // destruction of the source owner changes the stored request, and the
    // position-zero encoding is the frozen bitwise identity of the retained
    // source bytes (`R = 1` puts every logical element at position zero).
    // This scenario deliberately runs without an armed traffic window:
    // releasing a registered owner quarantines its storage through the device
    // allocator, which is not an operation-time allocation.
    {
        const iom::TensorSpec identity_spec =
                make_spec({2, 2, 1, 16}, iom::DataType::F32);
        auto source = candidate.create_tensor(identity_spec);
        auto output = candidate.create_tensor(identity_spec);
        const std::vector<std::byte> seeded =
                encode_logical(source->view().spec(), 0x91u);
        copy_from_host(source->view(), seeded);
        copy_from_host(
                output->view(),
                std::vector<std::byte>(
                        output->view().spec().logical_nbytes(),
                        kReadbackSentinel));
        auto queue = candidate.create_ops();
        iom::TensorView source_view = source->view();
        iom::TensorView output_view = output->view();
        const iom::oid token = queue->rope(source_view, output_view, 0, 1.0);
        REQUIRE(iom::oid_is_token(token));
        CHECK_EQ(token_sequence(token), std::uint64_t{1});
        {

            const std::span<const std::size_t> strides =
                    source_view.plane_strides();
            if (!strides.empty()) {
                const_cast<std::size_t*>(strides.data())[0] = strides[0] + 1;
            }
        }
        source.reset();
        CHECK_NOTHROW(queue->wait(token));
        CHECK_NOTHROW(queue->wait(token));
        CHECK(read_logical(output->view()) == seeded);
    }

    // A proven completion releases both owners, so the same owners accept a
    // later request in order.
    {
        auto input = candidate.create_tensor(query_spec);
        auto output = candidate.create_tensor(query_spec);
        auto queue = candidate.create_ops();
        const iom::oid first = queue->rope(
                input->view(), output->view(), 0, 1.0);
        REQUIRE(iom::oid_is_token(first));
        CHECK_NOTHROW(queue->wait(first));
        const iom::oid second = queue->rope(
                input->view(), output->view(), 1, 1.0);
        REQUIRE(iom::oid_is_token(second));
        CHECK_EQ(token_sequence(second), token_sequence(first) + 1);
        CHECK_NOTHROW(queue->wait(second));
    }
}

// Disarms an armed backend-native fault on every exit path, so a failing
// assertion can never leak an armed fault into a later case.
class NativeFailureGuard final {
public:
    explicit NativeFailureGuard(const RopeNativeFailureSeam& seam)
        : seam_(seam) {
        seam_.arm();
        armed_ = true;
    }

    ~NativeFailureGuard() {
        if (armed_) {
            seam_.clear();
        }
    }

    NativeFailureGuard(const NativeFailureGuard&) = delete;
    NativeFailureGuard& operator=(const NativeFailureGuard&) = delete;

private:
    const RopeNativeFailureSeam& seam_;
    bool armed_ = false;
};

// The accepted-failure scenario of one backend-native seam, run against the
// candidate's own queue so the real RoPE path — never a test double — carries
// the failure. The contract requires a `noexcept` facade to return a positive
// token for accepted work and to keep that token failing identically forever;
// the buffer content of a failed accepted request is unspecified, so it is
// never read while the failure is undrained, and the queue's recovery and the
// later reuse transition are asserted instead.
inline void run_native_failure_conformance(
        const RopeContractConformanceConfig& config) {
    const RopeNativeFailureSeam& seam = config.native_failure;
    if (!seam.available()) {
        // A backend whose testing seams cannot reach its RoPE path is a
        // recorded coverage gap, never a silent pass: the driver states why,
        // and the gap is printed with the exact missing seam.
        const bool identified =
                !seam.name.empty() && !seam.unavailable_reason.empty();
        REQUIRE_MESSAGE(
                identified,
                "a backend without a native RoPE accepted-failure seam must state why");
        std::printf(
                "rope-native-failure-record seam=%.*s coverage=unavailable "
                "reason=%.*s\n",
                static_cast<int>(seam.name.size()), seam.name.data(),
                static_cast<int>(seam.unavailable_reason.size()),
                seam.unavailable_reason.data());
        return;
    }

    iom::Device& candidate = config.devices.candidate;
    const iom::TensorSpec spec = make_spec({1, 1, 1, 16}, iom::DataType::F32);
    auto input = candidate.create_tensor(spec);
    auto output = candidate.create_tensor(spec);
    auto queue = candidate.create_ops();
    std::printf(
            "rope-native-failure-record seam=%.*s coverage=native reason=\n",
            static_cast<int>(seam.name.size()), seam.name.data());

    // 1. The identical request completes while the seam is disarmed, so the
    //    fixture, the queue, and the path are healthy; the accepted token
    //    establishes the sequence baseline.
    iom::oid healthy = 0;
    {
        CaseWindow window(config.observer);
        healthy = queue->rope(input->view(), output->view(), 0, 10000.0);
        REQUIRE(iom::oid_is_token(healthy));
        CHECK_NOTHROW(queue->wait(healthy));
    }

    // 2. The armed native fault is consumed after acceptance: the submission
    //    is still a positive OID in order, and that token keeps reporting the
    //    identical established error on every repeat wait. A fault that never
    //    reached the real path would leave this wait succeeding, and the
    //    disarmed control above shows the fixture itself is healthy.
    iom::oid failed = 0;
    {
        NativeFailureGuard guard(seam);
        CaseWindow window(config.observer);
        failed = queue->rope(input->view(), output->view(), 1, 10000.0);
        REQUIRE(iom::oid_is_token(failed));
        CHECK_EQ(
                token_sequence(failed),
                token_sequence(healthy) + std::uint64_t{1});
        // The pure query stays pure and correct while the failure is
        // undrained, and consumes no sequence of its own.
        CHECK(
                queue->rope_workspace_requirements(
                        input->view(), output->view(), 1, 10000.0)
                == kRopeContractWorkspace);
        expect_repeated_runtime_failure(*queue, failed);
    }

    // 3. Recovery: once the failure has been drained the queue accepts and
    //    completes the same owners again, and the failed token keeps its
    //    established error, so nothing was re-published as valid.
    {
        CaseWindow window(config.observer);
        const iom::oid recovered = queue->rope(
                input->view(), output->view(), 2, 10000.0);
        REQUIRE(iom::oid_is_token(recovered));
        CHECK_EQ(
                token_sequence(recovered),
                token_sequence(failed) + std::uint64_t{1});
        CHECK_NOTHROW(queue->wait(recovered));
        expect_repeated_runtime_failure(*queue, failed);
    }
}

}  // namespace rope_contract_detail

// The complete shared RoPE contract and failure suite of this leaf:
// capability admission, the unsupported matrix and its precedence, query and
// workspace purity, the admission matrix, ownership and queue semantics, and
// the backend-native accepted-failure identity.
inline void run_rope_contract_conformance(
        const RopeContractConformanceConfig& config) {
    REQUIRE_FALSE(config.supported_leaves.empty());
    rope_contract_detail::run_supported_leaf_paths(config);
    rope_contract_detail::run_capability_rejections(config);
    rope_contract_detail::run_query_and_workspace_conformance(config);
    rope_contract_detail::run_admission_matrix(config);
    rope_contract_detail::run_owner_and_queue_conformance(config);
    rope_contract_detail::run_native_failure_conformance(config);
}

}  // namespace iom_conformance

#include "session_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "iom/device.hpp"
#include "iom_internal.hpp"

namespace iom::session_detail {
namespace {

// The base alignment every device guarantees for a caller-owned raw workspace
// range, and the granularity at which the stage's fixed slices are reserved.
constexpr std::size_t kMlpSliceAlignment = 32;

[[nodiscard]] std::span<const std::size_t> mlp_extents(
        const TensorView& view) noexcept {
    return view.spec().shape.dimensions();
}

// Leading extents of one activation view: everything before the final two
// matrix axes. These stay independent and are never broadcast.
[[nodiscard]] std::span<const std::size_t> mlp_leading_extents(
        const TensorView& view) noexcept {
    const std::span<const std::size_t> dimensions = mlp_extents(view);
    return dimensions.first(dimensions.size() - 2);
}

// Final two extents of one activation view: the logical matrix axes.
[[nodiscard]] std::span<const std::size_t> mlp_matrix_extents(
        const TensorView& view) noexcept {
    const std::span<const std::size_t> dimensions = mlp_extents(view);
    return dimensions.subspan(dimensions.size() - 2);
}

[[nodiscard]] bool mlp_same_extents(
        std::span<const std::size_t> lhs,
        std::span<const std::size_t> rhs) noexcept {
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index] != rhs[index]) return false;
    }
    return true;
}

// One checked plain-BF16 view requirement. `expected` holds the required final
// two extents; `message` names the view and the violated boundary.
void mlp_require_view(
        const TensorView& view, std::span<const std::size_t> expected,
        const char* message) {
    if (view.spec().data_type != DataType::BF16
            || view.spec().quantization != QuantizationFormat::NONE
            || !mlp_same_extents(mlp_matrix_extents(view), expected)) {
        throw std::invalid_argument(message);
    }
}

void mlp_validate_views(
        const MlpStageViews& views, const MlpStageParams& params) {
    if (params.rows == 0 || params.features == 0
            || params.intermediate == 0) {
        throw std::invalid_argument(
                "TinyLlama MLP stage requires nonzero row, feature, and "
                "intermediate sizes");
    }
    if (!std::isfinite(params.rms_epsilon) || params.rms_epsilon < 0.0F) {
        throw std::invalid_argument(
                "TinyLlama MLP stage requires a finite nonnegative epsilon");
    }

    const std::size_t residual_extents[2] = {params.rows, params.features};
    const std::size_t projection_extents[2] = {
            params.rows, params.intermediate};
    const std::size_t scale_extents[2] = {1, params.features};
    const std::size_t gate_weight_extents[2] = {
            params.intermediate, params.features};
    const std::size_t down_weight_extents[2] = {
            params.features, params.intermediate};

    mlp_require_view(
            views.x2, residual_extents,
            "TinyLlama MLP stage first residual must be a plain BF16 [R,F] "
            "view");
    mlp_require_view(
            views.n2, residual_extents,
            "TinyLlama MLP stage norm store must be a plain BF16 [R,F] view");
    mlp_require_view(
            views.down, residual_extents,
            "TinyLlama MLP stage down store must be a plain BF16 [R,F] view");
    mlp_require_view(
            views.next_x, residual_extents,
            "TinyLlama MLP stage next-residual store must be a plain BF16 "
            "[R,F] view");
    mlp_require_view(
            views.gate, projection_extents,
            "TinyLlama MLP stage gate store must be a plain BF16 [R,M] view");
    mlp_require_view(
            views.up, projection_extents,
            "TinyLlama MLP stage up store must be a plain BF16 [R,M] view");
    mlp_require_view(
            views.activated_gate, projection_extents,
            "TinyLlama MLP stage activated-gate store must be a plain BF16 "
            "[R,M] view");
    mlp_require_view(
            views.product, projection_extents,
            "TinyLlama MLP stage product store must be a plain BF16 [R,M] "
            "view");
    mlp_require_view(
            views.post_attention_scale, scale_extents,
            "TinyLlama MLP stage post-attention scale must be a plain BF16 "
            "[1,F] view");
    mlp_require_view(
            views.gate_weight, gate_weight_extents,
            "TinyLlama MLP stage gate weight must be a plain BF16 [M,F] view");
    mlp_require_view(
            views.up_weight, gate_weight_extents,
            "TinyLlama MLP stage up weight must be a plain BF16 [M,F] view");
    mlp_require_view(
            views.down_weight, down_weight_extents,
            "TinyLlama MLP stage down weight must be a plain BF16 [F,M] view");

    const TensorView* const activations[8] = {
            &views.x2, &views.n2, &views.gate, &views.up,
            &views.activated_gate, &views.product, &views.down, &views.next_x};
    for (std::size_t index = 1; index < 8; ++index) {
        if (!mlp_same_extents(
                    mlp_leading_extents(*activations[0]),
                    mlp_leading_extents(*activations[index]))) {
            throw std::invalid_argument(
                    "TinyLlama MLP stage activation views must share one "
                    "independent leading tuple");
        }
    }
}

// Category of one negative admission OID, mapped back to the established
// exception class the facades map from. Negative OIDs are synchronous errors
// that must never be waited, and the mapped exception is constructed only when
// it is thrown.
[[noreturn]] void mlp_throw_admission_failure(oid token) {
    switch (token) {
        case to_oid(OidError::InvalidArgument):
            throw std::invalid_argument(
                    "TinyLlama MLP stage rejected an operand as invalid input");
        case to_oid(OidError::Unsupported):
            throw detail::UnsupportedOperation{};
        case to_oid(OidError::Overflow):
            throw std::overflow_error(
                    "TinyLlama MLP stage rejected an operand as overflowing");
        case to_oid(OidError::ResourceExhausted):
            throw std::bad_alloc{};
        case to_oid(OidError::DeviceError):
            throw std::runtime_error(
                    "TinyLlama MLP stage failed on its device");
        case to_oid(OidError::InternalError):
            throw std::logic_error("TinyLlama MLP stage failed internally");
        default:
            throw std::logic_error(
                    "TinyLlama MLP stage received an invalid operation "
                    "result");
    }
}

// First-failure preservation with complete draining. Every accepted OID is
// attempted even when an earlier wait throws, a later success never masks an
// earlier failure, and a negative admission OID is recorded without ever being
// waited.
class MlpFailureSpool {
public:
    explicit MlpFailureSpool(MlpStageFailure& record) noexcept
        : record_(&record) {}

    // Records one negative admission OID as the first actionable failure and
    // reports whether the token was accepted.
    [[nodiscard]] bool accept(oid token) noexcept {
        if (oid_is_token(token)) return true;
        if (first_ == nullptr) {
            admission_ = token;
        }
        return false;
    }

    void wait(DeviceOps& operations, oid token) noexcept {
        try {
            operations.wait(token);
        } catch (...) {
            capture(std::current_exception());
        }
    }

    [[nodiscard]] bool failed() const noexcept {
        return first_ != nullptr || admission_ != 0;
    }

    // Records the preserved first failure and rethrows it.
    [[noreturn]] void publish() {
        if (first_ != nullptr) {
            record_->record(first_);
            std::rethrow_exception(first_);
        }
        if (admission_ != 0) {
            record_->record_admission(admission_);
            mlp_throw_admission_failure(admission_);
        }
        throw std::logic_error(
                "TinyLlama MLP stage has no failure to publish");
    }

private:
    // A wait failure is only remembered while no earlier failure exists, so
    // the recorded failure is always the first actionable one.
    void capture(std::exception_ptr failure) noexcept {
        if (failure != nullptr && first_ == nullptr && admission_ == 0) {
            first_ = std::move(failure);
        }
    }

    MlpStageFailure* record_;
    std::exception_ptr first_;
    oid admission_ = 0;
};

// Rounds one positive byte count up to the fixed slice granularity with
// checked arithmetic; a zero count stays zero.
[[nodiscard]] std::size_t mlp_aligned_extent(std::size_t bytes) {
    if (bytes == 0) return 0;
    const std::size_t mask = kMlpSliceAlignment - 1;
    return detail::checked_add(
                   bytes, mask, "post-attention MLP workspace extent")
            & ~mask;
}

// Fixed member layout of the resolved workspace. The reused slice holds the
// gate projection and is reused by mul, down, and the second residual only
// after that producer and its readers completed; the up projection keeps its
// own disjoint slice because both projections are enqueued before either is
// waited.
struct MlpSliceLayout {
    std::size_t reused_bytes;
    std::size_t up_offset;
    std::size_t up_bytes;
};

[[nodiscard]] MlpSliceLayout mlp_slice_layout(
        const MlpWorkspaceRequirements& requirements) {
    const std::size_t reused = std::max(
            std::max(requirements.gate.bytes, requirements.mul.bytes),
            std::max(requirements.down.bytes, requirements.residual.bytes));
    return MlpSliceLayout{
            reused, mlp_aligned_extent(reused), requirements.up.bytes};
}

[[nodiscard]] WorkspaceRequirements mlp_wider(
        WorkspaceRequirements lhs, WorkspaceRequirements rhs) noexcept {
    return WorkspaceRequirements{
            std::max(lhs.bytes, rhs.bytes),
            std::max(lhs.alignment, rhs.alignment)};
}

// The five member requirements of one actual run bank. Every query runs the
// exact admission validation of its operation and reports the requirement
// without allocation, registration, lease, or submission.
[[nodiscard]] MlpWorkspaceRequirements mlp_bank_requirements(
        DeviceOps& operations, const MlpStageViews& views) {
    const std::span<const std::size_t> gate_extents =
            mlp_matrix_extents(views.gate);
    const std::size_t rows = gate_extents[0];
    const std::size_t intermediate = gate_extents[1];
    const std::size_t features = mlp_matrix_extents(views.down)[1];
    const std::size_t projection_extents[2] = {rows, intermediate};
    const std::size_t residual_extents[2] = {rows, features};
    mlp_require_view(
            views.gate, projection_extents,
            "TinyLlama MLP stage gate store must be a plain BF16 [R,M] view");
    mlp_require_view(
            views.down, residual_extents,
            "TinyLlama MLP stage down store must be a plain BF16 [R,F] view");

    return MlpWorkspaceRequirements{
            operations.linear_workspace_requirements(
                    views.n2, views.gate_weight, views.gate, 0, rows,
                    LinearOutputLayout::ordinary, 1, intermediate),
            operations.linear_workspace_requirements(
                    views.n2, views.up_weight, views.up, 0, rows,
                    LinearOutputLayout::ordinary, 1, intermediate),
            operations.mul_workspace_requirements(
                    views.activated_gate, views.up, views.product),
            operations.linear_workspace_requirements(
                    views.product, views.down_weight, views.down, 0, rows,
                    LinearOutputLayout::ordinary, 1, features),
            operations.add_workspace_requirements(
                    views.x2, views.down, views.next_x)};
}

// One member view of the reserved workspace. A zero-byte member keeps the
// empty default view, which is the only admissible workspace for the
// zero-requirement operations; a positive member must be a live, sufficient,
// aligned range of the exact device.
[[nodiscard]] RawWorkspaceView mlp_member_view(
        const Device& device, RawWorkspaceView workspace, std::size_t offset,
        WorkspaceRequirements requirement) {
    if (requirement.bytes == 0) return RawWorkspaceView{};
    return detail::WorkspaceValidation::validated(
            device, workspace.subrange(offset, requirement.bytes),
            requirement.bytes, requirement.alignment,
            std::span<const TensorView>{});
}

}  // namespace

void MlpStageFailure::record(std::exception_ptr failure) noexcept {
    if (failure != nullptr && admission_ == 0 && first_ == nullptr) {
        first_ = std::move(failure);
    }
}

void MlpStageFailure::record_admission(oid token) noexcept {
    if (token < 0 && admission_ == 0 && first_ == nullptr) {
        admission_ = token;
    }
}

[[noreturn]] void MlpStageFailure::rethrow_first() const {
    if (first_ != nullptr) {
        std::rethrow_exception(first_);
    }
    if (admission_ != 0) {
        mlp_throw_admission_failure(admission_);
    }
    throw std::logic_error("TinyLlama MLP stage failure record is empty");
}

[[nodiscard]] WorkspaceRequirements MlpWorkspaceRequirements::total() const {
    const MlpSliceLayout layout = mlp_slice_layout(*this);
    const std::size_t bytes = detail::checked_add(
            layout.up_offset, mlp_aligned_extent(layout.up_bytes),
            "post-attention MLP workspace requirement");
    if (bytes == 0) return WorkspaceRequirements{0, 1};
    const std::size_t alignment = std::max(
            {kMlpSliceAlignment, gate.alignment, up.alignment, mul.alignment,
             down.alignment, residual.alignment});
    if ((alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(
                "TinyLlama MLP stage requires a power-of-two workspace "
                "alignment");
    }
    return WorkspaceRequirements{bytes, alignment};
}

[[nodiscard]] MlpWorkspaceRequirements mlp_workspace_requirements(
        DeviceOps& operations, const MlpStageViews& prefill,
        const MlpStageViews& decode) {
    const MlpWorkspaceRequirements prefill_requirements =
            mlp_bank_requirements(operations, prefill);
    const MlpWorkspaceRequirements decode_requirements =
            mlp_bank_requirements(operations, decode);
    return MlpWorkspaceRequirements{
            mlp_wider(prefill_requirements.gate, decode_requirements.gate),
            mlp_wider(prefill_requirements.up, decode_requirements.up),
            mlp_wider(prefill_requirements.mul, decode_requirements.mul),
            mlp_wider(prefill_requirements.down, decode_requirements.down),
            mlp_wider(
                    prefill_requirements.residual,
                    decode_requirements.residual)};
}

[[nodiscard]] MlpWorkspace resolve_mlp_workspace(
        const Device& device, const MlpWorkspaceRequirements& requirements,
        RawWorkspaceView workspace) {
    const WorkspaceRequirements total = requirements.total();
    if (total.bytes == 0) {
        // Nothing is required, so the supplied range is neither validated nor
        // leased and every member keeps the empty default view.
        return MlpWorkspace{};
    }
    const MlpSliceLayout layout = mlp_slice_layout(requirements);
    (void)detail::WorkspaceValidation::validated(
            device, workspace, total.bytes, total.alignment,
            std::span<const TensorView>{});

    // Aggregate construction only: a workspace view is never reassignable, so
    // every member is built once over its reserved range.
    return MlpWorkspace{
            mlp_member_view(device, workspace, 0, requirements.gate),
            mlp_member_view(
                    device, workspace, layout.up_offset, requirements.up),
            mlp_member_view(device, workspace, 0, requirements.mul),
            mlp_member_view(device, workspace, 0, requirements.down),
            mlp_member_view(device, workspace, 0, requirements.residual)};
}

void run_mlp_stage(DeviceOps& operations, MlpStageViews& views,
                   const MlpStageParams& params, const MlpWorkspace& workspace,
                   std::span<const oid> readiness, MlpStageFailure& failure) {
    // A poisoned execution is never reused: the preserved first failure is
    // rethrown without submitting anything.
    if (failure.poisoned()) {
        failure.rethrow_first();
    }
    mlp_validate_views(views, params);
    for (const oid token : readiness) {
        if (!oid_is_token(token)) {
            throw std::invalid_argument(
                    "TinyLlama MLP stage readiness requires accepted producer "
                    "tokens");
        }
    }

    MlpFailureSpool spool(failure);

    // The supplied first-residual producer must complete before the stage
    // reads `x2`, and every readiness producer is attempted even when an
    // earlier one fails.
    for (const oid token : readiness) {
        spool.wait(operations, token);
    }
    if (spool.failed()) spool.publish();

    // Post-attention RMSNorm into its own store; RMSNorm consumes the empty
    // workspace under its zero-workspace contract.
    const oid norm = operations.rmsnorm(
            views.x2, views.post_attention_scale, views.n2,
            params.rms_epsilon);
    if (!spool.accept(norm)) spool.publish();
    spool.wait(operations, norm);
    if (spool.failed()) spool.publish();

    // Gate and up are independent branches of the same ready normalization:
    // both are submitted before either is waited, and both accepted branches
    // are attempted even when the first wait reports failure.
    const oid gate = operations.linear(
            views.n2, views.gate_weight, views.gate, 0, params.rows,
            LinearOutputLayout::ordinary, 1, params.intermediate,
            workspace.gate);
    const oid up = operations.linear(
            views.n2, views.up_weight, views.up, 0, params.rows,
            LinearOutputLayout::ordinary, 1, params.intermediate,
            workspace.up);
    const bool gate_accepted = spool.accept(gate);
    const bool up_accepted = spool.accept(up);
    if (gate_accepted) spool.wait(operations, gate);
    if (up_accepted) spool.wait(operations, up);
    if (spool.failed()) spool.publish();

    // Existing BF16 SiLU of the stored gate, then the existing multiply in the
    // `SiLU(gate), up` operand order. Neither boundary is fused.
    const oid activated = operations.silu(views.gate, views.activated_gate);
    if (!spool.accept(activated)) spool.publish();
    spool.wait(operations, activated);
    if (spool.failed()) spool.publish();

    const oid product = operations.mul(
            views.activated_gate, views.up, views.product, workspace.mul);
    if (!spool.accept(product)) spool.publish();
    spool.wait(operations, product);
    if (spool.failed()) spool.publish();

    // Ordinary down projection with the Hugging Face `[F,M]` weight, then the
    // second residual `x2 + down` into the next layer input.
    const oid down = operations.linear(
            views.product, views.down_weight, views.down, 0, params.rows,
            LinearOutputLayout::ordinary, 1, params.features, workspace.down);
    if (!spool.accept(down)) spool.publish();
    spool.wait(operations, down);
    if (spool.failed()) spool.publish();

    const oid residual = operations.add(
            views.x2, views.down, views.next_x, workspace.residual);
    if (!spool.accept(residual)) spool.publish();
    spool.wait(operations, residual);
    if (spool.failed()) spool.publish();
}

/*
 * Private TinyLlama session implementation.
 *
 * This translation unit owns the numerical stages of the session forward
 * path. Every stage is a private, allocation-free helper over caller-owned
 * views, the session's single `DeviceOps` queue, supplied parameters, and
 * caller-owned workspace; nothing here creates a public stage API, a graph,
 * a scheduler, or an owner.
 */

namespace {

    using detail::UnsupportedOperation;
    using detail::checked_add;
    using detail::checked_mul;

    // Supplied head-planar geometry `[H,R,D]` of one rank-three view.
    struct HeadPlane {
        std::size_t heads = 0;
        std::size_t rows = 0;
        std::size_t features = 0;
    };

    // Supplied matrix geometry `[R,F]` of one rank-two view.
    struct Matrix {
        std::size_t rows = 0;
        std::size_t features = 0;
    };

    // Checked stage facts derived before any submission.
    struct CheckedStage {
        std::size_t merged_width = 0;
        std::size_t initialized_length = 0;
    };

    // One supplied view of the stage, with the role name used in rejection
    // text and whether the stage writes it.
    struct StageOwnership {
        const TensorView* view;
        const char* role;
        bool store;
    };

    [[noreturn]] void reject(
            std::string_view role, std::string_view requirement) {
        throw std::invalid_argument(
                "cache attention stage " + std::string(role) + " "
                + std::string(requirement));
    }

    /**
     * Stage-wide conservative storage disjointness. The K/V cache appends,
     * SDPA, the output projection, and the first residual run in sequence over
     * the supplied views, so a store sharing storage with another supplied
     * view can destroy a value that an earlier or later phase still needs: V
     * rows would overwrite the K cache, the merged attention would overwrite
     * the residual input, and so on. Every store must therefore live in
     * distinct owner storage from every other supplied view, which is the same
     * conservative owner-overlap policy the operation facades apply locally.
     * Read/read aliasing stays legal, exactly as those contracts allow.
     */
    void require_distinct_stores(
            const std::array<StageOwnership, 10>& views) {
        for (std::size_t first = 0; first < views.size(); ++first) {
            for (std::size_t second = first + 1; second < views.size();
                 ++second) {
                if (!views[first].store && !views[second].store) continue;
                if (views[first].view->owner_identity()
                        != views[second].view->owner_identity()) {
                    continue;
                }
                reject(views[first].role,
                       std::string("must not share owner storage with the ")
                               + views[second].role);
            }
        }
    }

    [[nodiscard]] const TensorSpec& require_bf16(
            const Device& device, const TensorView& view,
            std::string_view role) {
        if (&view.device() != &device) {
            reject(role, "belongs to a different device");
        }
        const TensorSpec& spec = view.spec();
        if (spec.data_type != DataType::BF16
                || spec.quantization != QuantizationFormat::NONE) {
            reject(role, "must be an unquantized BF16 tensor");
        }
        return spec;
    }

    [[nodiscard]] HeadPlane require_heads(
            const Device& device, const TensorView& view,
            std::string_view role) {
        const std::span<const std::size_t> dimensions =
                require_bf16(device, view, role).shape.dimensions();
        if (dimensions.size() != 3) {
            reject(role, "must have rank three");
        }
        return {dimensions[0], dimensions[1], dimensions[2]};
    }

    [[nodiscard]] Matrix require_matrix(
            const Device& device, const TensorView& view,
            std::string_view role) {
        const std::span<const std::size_t> dimensions =
                require_bf16(device, view, role).shape.dimensions();
        if (dimensions.size() != 2) {
            reject(role, "must have rank two");
        }
        return {dimensions[0], dimensions[1]};
    }

    /**
     * Checked request admission. Everything here is derived from supplied
     * values and borrowed view metadata, so a rejection happens before the
     * stage submits an operation, mutates a cache row, or publishes a
     * length.
     */
    [[nodiscard]] CheckedStage validate_stage(
            const DeviceOps& ops, const CacheAttentionStageRequest& request,
            const CacheAttentionStageState& state) {
        if (state.failed) {
            throw std::logic_error(
                    "cache attention stage refuses a failed session state");
        }
        if (request.R == 0) {
            throw std::invalid_argument(
                    "cache attention stage requires nonzero rows");
        }
        if (request.C == 0) {
            throw std::invalid_argument(
                    "cache attention stage requires a nonzero cache "
                    "capacity");
        }
        // The order is contractual: the capacity relation is established
        // before `a+R` is formed, so the sum can never wrap.
        if (request.a > request.C) {
            throw std::invalid_argument(
                    "cache attention stage offset exceeds the cache "
                    "capacity");
        }
        const std::size_t initialized_length = checked_add(
                request.a, request.R,
                "cache attention stage initialized length overflows");
        if (initialized_length > request.C) {
            throw std::invalid_argument(
                    "cache attention stage rows exceed the cache capacity");
        }
        if (state.initialized_length != request.a) {
            throw std::invalid_argument(
                    "cache attention stage offset must continue the "
                    "published cache prefix");
        }

        const Device& device = ops.device();
        const HeadPlane q =
                require_heads(device, request.rotated_q, "rotated Q");
        const std::size_t merged_width = checked_mul(
                q.heads, q.features,
                "cache attention stage merged attention width overflows");
        if (merged_width == 0) {
            reject("rotated Q", "must have nonzero head and feature extents");
        }
        if (q.rows != request.R) {
            reject("rotated Q", "must have exactly R rows");
        }

        const HeadPlane k =
                require_heads(device, request.rotated_k, "rotated K");
        const HeadPlane v =
                require_heads(device, request.rotated_v, "rotated V");
        if (k.heads != v.heads || k.rows != v.rows
                || k.features != v.features) {
            reject("rotated K",
                   "and rotated V must share head, row, and feature "
                   "extents");
        }
        const std::size_t kv_width = checked_mul(
                k.heads, k.features,
                "cache attention stage K/V merge width overflows");
        if (kv_width == 0) {
            reject("rotated K/V", "must have nonzero head and feature extents");
        }
        if (k.rows != request.R) {
            reject("rotated K/V", "must have exactly R rows");
        }
        if (k.features != q.features) {
            reject("rotated K/V", "must use the query head width");
        }
        if (q.heads % k.heads != 0) {
            reject("rotated Q",
                   "head count must be a multiple of the KV head count");
        }

        // Distinct append boundaries: each cache keeps its own owner,
        // head geometry, feature width, and capacity row extent.
        const HeadPlane k_cache =
                require_heads(device, request.k_cache, "K cache");
        const HeadPlane v_cache =
                require_heads(device, request.v_cache, "V cache");
        if (k_cache.heads != v_cache.heads
                || k_cache.rows != v_cache.rows
                || k_cache.features != v_cache.features) {
            reject("K cache",
                   "and V cache must share head, row, and feature extents");
        }
        if (k_cache.rows != request.C) {
            reject("K cache",
                   "row extent must equal the supplied cache capacity");
        }
        if (k_cache.heads != k.heads
                || k_cache.features != k.features) {
            reject("K cache", "geometry must match the appended K/V rows");
        }

        const Matrix input =
                require_matrix(device, request.residual_input,
                               "residual input");
        if (input.rows != request.R || input.features != merged_width) {
            reject("residual input", "must be [R,F]");
        }
        const Matrix weight =
                require_matrix(device, request.o_weight, "output weight");
        if (weight.rows != merged_width
                || weight.features != merged_width) {
            reject("output weight", "must be [F,F]");
        }
        const Matrix merged =
                require_matrix(device, request.attention_merged,
                               "merged attention");
        if (merged.rows != request.R || merged.features != merged_width) {
            reject("merged attention", "must be [R,F]");
        }
        const Matrix projected =
                require_matrix(device, request.attention_output,
                               "attention output");
        if (projected.rows != request.R
                || projected.features != merged_width) {
            reject("attention output", "must be [R,F]");
        }
        const Matrix residual =
                require_matrix(device, request.residual_output,
                               "residual output");
        if (residual.rows != request.R
                || residual.features != merged_width) {
            reject("residual output", "must be [R,F]");
        }

        // Every supplied view carries its stage role and whether the stage
        // writes it. Both caches are stores, so distinct K and V owners are
        // enforced here rather than by the individual append facades, and the
        // same check rejects every cross-phase alias that would let a later
        // store destroy an earlier input (for example the merged attention
        // overwriting the residual input before the closing residual).
        require_distinct_stores(std::array<StageOwnership, 10>{{
                {&request.rotated_q, "rotated Q", false},
                {&request.rotated_k, "rotated K", false},
                {&request.rotated_v, "rotated V", false},
                {&request.k_cache, "K cache", true},
                {&request.v_cache, "V cache", true},
                {&request.residual_input, "residual input", false},
                {&request.o_weight, "output weight", false},
                {&request.attention_merged, "merged attention", true},
                {&request.attention_output, "attention output", true},
                {&request.residual_output, "residual output", true}}});

        return {merged_width, initialized_length};
    }

    /**
     * Mirrors the facade's synchronous error mapping so a rejected
     * submission surfaces through the established exception categories
     * instead of a bare negative token.
     */
    [[noreturn]] void throw_oid_failure(
            oid value, std::string_view role) {
        if (!oid_is_error(value)) {
            throw std::logic_error(
                    "cache attention stage received a non-error oid");
        }
        const std::string what =
                "cache attention stage " + std::string(role);
        switch (static_cast<OidError>(value)) {
            case OidError::InvalidArgument:
                throw std::invalid_argument(
                        what + " rejected its operands");
            case OidError::Unsupported:
                throw UnsupportedOperation();
            case OidError::Overflow:
                throw std::overflow_error(what + " overflowed");
            case OidError::ResourceExhausted:
                throw std::bad_alloc();
            case OidError::DeviceError:
                throw std::runtime_error(what + " failed on its device");
            case OidError::InternalError:
                break;
        }
        throw std::runtime_error(what + " failed internally");
    }

    [[nodiscard]] std::exception_ptr oid_failure(
            oid value, std::string_view role) noexcept {
        try {
            throw_oid_failure(value, role);
        } catch (...) {
            return std::current_exception();
        }
    }

    /**
     * Waits one submitted step and returns its first failure without
     * throwing, so a caller can record one failure and still complete
     * every remaining mandatory wait.
     */
    [[nodiscard]] std::exception_ptr completed(
            DeviceOps& ops, oid token, std::string_view role) noexcept {
        if (oid_is_error(token)) {
            return oid_failure(token, role);
        }
        try {
            ops.wait(token);
        } catch (...) {
            return std::current_exception();
        }
        return nullptr;
    }

}  // namespace

void run_cache_attention_stage(
        DeviceOps& ops, CacheAttentionStageRequest& request,
        RawWorkspaceView workspace, CacheAttentionStageState& state) {
    const CheckedStage stage = validate_stage(ops, request, state);

    // Every downstream operation is preflighted with its actual operands
    // and scratch before the first submission. SDPA, the output projection,
    // and the first residual are three strictly sequential consumers of one
    // caller-owned range, so that range must satisfy the largest of their
    // queried requirements: a backend whose projection or residual needs
    // positive staging rejects an insufficient range here, before any cache
    // row is written or a prefix is published.
    const WorkspaceRequirements sdpa_requirement =
            ops.sdpa_workspace_requirements(
                    request.rotated_q, request.k_cache, request.v_cache,
                    request.attention_merged, request.a,
                    stage.initialized_length);
    const WorkspaceRequirements projection_requirement =
            ops.linear_workspace_requirements(
                    request.attention_merged, request.o_weight,
                    request.attention_output, 0, request.R,
                    LinearOutputLayout::ordinary, 1, stage.merged_width);
    const WorkspaceRequirements residual_requirement =
            ops.add_workspace_requirements(
                    request.residual_input, request.attention_output,
                    request.residual_output);
    const WorkspaceRequirements combined_requirement{
            std::max({sdpa_requirement.bytes, projection_requirement.bytes,
                      residual_requirement.bytes}),
            std::max({sdpa_requirement.alignment,
                      projection_requirement.alignment,
                      residual_requirement.alignment})};

    // The shared validator compares the caller range against one borrowed
    // operand at a time: its checks are pairwise, and copying a borrowed
    // `TensorView` would allocate its plane-stride vector. Every operand
    // read or written by those three operations is checked, so no later
    // rejection can leave accepted append work behind.
    const auto validate_against_operand = [&](const TensorView& operand) {
        const TensorView* borrowed = &operand;
        (void)detail::WorkspaceValidation::validated(
                ops.device(), workspace, combined_requirement.bytes,
                combined_requirement.alignment,
                std::span<const TensorView>(borrowed, 1));
    };
    validate_against_operand(request.rotated_q);
    validate_against_operand(request.k_cache);
    validate_against_operand(request.v_cache);
    validate_against_operand(request.attention_merged);
    validate_against_operand(request.o_weight);
    validate_against_operand(request.attention_output);
    validate_against_operand(request.residual_input);
    validate_against_operand(request.residual_output);

    // An operation whose queried requirement is zero consumes no scratch
    // and accepts only the empty default; each positive requirement gets the
    // validated reusable range, which stays live until its own wait.
    const RawWorkspaceView sdpa_workspace =
            sdpa_requirement.bytes == 0 ? RawWorkspaceView{} : workspace;
    const RawWorkspaceView projection_workspace =
            projection_requirement.bytes == 0 ? RawWorkspaceView{}
                                              : workspace;
    const RawWorkspaceView residual_workspace =
            residual_requirement.bytes == 0 ? RawWorkspaceView{}
                                            : workspace;

    // K and V have distinct cache owners and distinct OIDs. Both
    // submissions are attempted, and both accepted OIDs are waited, even
    // when the first result already failed.
    std::array<oid, 2> appends{0, 0};
    appends[0] =
            ops.cache_append(request.rotated_k, request.k_cache, request.a);
    appends[1] =
            ops.cache_append(request.rotated_v, request.v_cache, request.a);

    std::exception_ptr failure;
    for (std::size_t index = 0; index < appends.size(); ++index) {
        const std::string_view role =
                index == 0 ? "K append" : "V append";
        if (oid_is_error(appends[index])) {
            if (failure == nullptr) {
                failure = oid_failure(appends[index], role);
            }
            continue;
        }
        if (std::exception_ptr waited =
                    completed(ops, appends[index], role);
                waited != nullptr && failure == nullptr) {
            failure = std::move(waited);
        }
    }
    if (failure != nullptr) {
        // A possibly written physical cache row is never rolled back,
        // retried, or reused, and no new prefix is published.
        state.failed = true;
        std::rethrow_exception(failure);
    }

    // Both independent append barriers succeeded: publish exactly the
    // checked prefix and read no cache row beyond it.
    state.initialized_length = stage.initialized_length;

    failure = completed(
            ops,
            ops.sdpa(
                    request.rotated_q, request.k_cache, request.v_cache,
                    request.attention_merged, request.a,
                    stage.initialized_length, sdpa_workspace),
            "SDPA");
    if (failure == nullptr) {
        failure = completed(
                ops,
                ops.linear(
                        request.attention_merged, request.o_weight,
                        request.attention_output, 0, request.R,
                        LinearOutputLayout::ordinary, 1,
                        stage.merged_width, projection_workspace),
                "output projection");
    }
    if (failure == nullptr) {
        failure = completed(
                ops,
                ops.add(
                        request.residual_input, request.attention_output,
                        request.residual_output, residual_workspace),
                "first residual");
    }
    if (failure != nullptr) {
        state.failed = true;
        std::rethrow_exception(failure);
    }
}

}  // namespace iom::session_detail

#pragma once

// Private session implementation seams.
//
// This header is deliberately not installed: it is included only by
// `src/session.cpp` and the focused `test/test_model_session.cpp`, so session
// stage types, scheduling helpers, readiness/error state, and numerical
// boundaries stay implementation-local. No stage framework, graph object, or
// numerical seam is published from `include/iom/session.hpp` or any other
// public header.

#include <cstddef>
#include <exception>
#include <span>

#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom::session_detail {

// ---------------------------------------------------------------------------
// Post-attention SwiGLU MLP stage (leaf 05-mlp-stage).
//
// One private, allocation-free stage implements the post-attention half of a
// decoder layer: RMS normalization of the first residual, independent gate and
// up projections, the existing BF16 SiLU, the existing multiply, the down
// projection, and the second residual. The stage consumes only caller-owned
// views, the session's one `DeviceOps` queue, fixed scalar/row parameters,
// caller-supplied readiness producers, and one fixed caller-owned workspace;
// it creates no tensor, workspace, host buffer, or asynchronous task, and the
// session's decoder-layer routine is its direct caller.
//
// The stage is used unchanged by an exact-`R` prefill run and by the fixed
// `R1=1` decode run: the caller supplies the selected run bank's views and the
// matching row parameters, so no view is retargeted and no work is sized from
// `R` during execution.
// ---------------------------------------------------------------------------

/**
 * Caller-owned views of one post-attention MLP execution.
 *
 * Every view belongs to the same queue device, is a plain BF16
 * (`QuantizationFormat::NONE`) view with rank two through eight, and is
 * disjoint from every view it is not the declared output of. The activation
 * views describe the exact logical run bank of the selected run, so the
 * projection row window is always `s=0, R=rows`: `n2`, `gate`, `up`,
 * `activated_gate`, `product`, `down`, and `next_x` are seven distinct stores
 * and no view aliases another, in place or otherwise.
 *
 * `x2` is the first residual `[..,R,F]` and is only read; the stage never
 * rewrites it, and `next_x` is written only by the closing residual add.
 * `post_attention_scale` is the layer post-attention scale `[1,F]`.
 * `gate_weight`/`up_weight` are `[M,F]`, `down_weight` is `[F,M]`, and every
 * ordinary linear call consumes the supplied Hugging Face `[out,in]` rows
 * unchanged, so no transpose, final-axis view transform, or padded row is
 * introduced. Independent leading planes stay independent: the stage never
 * broadcasts state between planes.
 */
struct MlpStageViews {
    TensorView x2;
    TensorView post_attention_scale;
    TensorView gate_weight;
    TensorView up_weight;
    TensorView down_weight;
    TensorView n2;
    TensorView gate;
    TensorView up;
    TensorView activated_gate;
    TensorView product;
    TensorView down;
    TensorView next_x;
};

/** Fixed scalar and row parameters of one stage execution. */
struct MlpStageParams {
    std::size_t rows;          // logical R of the selected run bank
    std::size_t features;      // F, the RMSNorm and residual width
    std::size_t intermediate;  // M, the gate/up width
    float rms_epsilon;         // finite nonnegative post-attention epsilon
};

/**
 * The five pre-sliced caller-owned raw workspace members of one stage
 * execution. `gate` and `up` are disjoint because both projections are
 * enqueued before either is waited; `mul`, `down`, and `residual` are strictly
 * serialized behind the completed gate/up work, so they reuse the `gate` slice
 * only after its producer and readers have completed. RMSNorm and SiLU
 * consume the empty view under their existing zero-workspace contracts, so
 * they need no member.
 */
struct MlpWorkspace {
    RawWorkspaceView gate;
    RawWorkspaceView up;
    RawWorkspaceView mul;
    RawWorkspaceView down;
    RawWorkspaceView residual;
};

/**
 * Checked per-member workspace maxima over the actual exact-`R` prefill and
 * fixed-`R1` decode views. Request setup folds `total()` into the session's
 * reusable operation workspace and keeps the members for `resolve`.
 */
struct MlpWorkspaceRequirements {
    WorkspaceRequirements gate;
    WorkspaceRequirements up;
    WorkspaceRequirements mul;
    WorkspaceRequirements down;
    WorkspaceRequirements residual;

    /**
     * Combined contribution of this stage to the request's reusable operation
     * workspace: the two permanent slices (the reused gate slice and the
     * disjoint up slice) with checked alignment arithmetic. A stage whose
     * members are all zero reports the conventional zero requirement
     * `{0, 1}` and consumes no workspace.
     */
    [[nodiscard]] WorkspaceRequirements total() const;
};

/**
 * Sticky first-failure record of one stage execution.
 *
 * An accepted asynchronous failure, a drained wait failure, or a negative
 * admission OID is recorded here as the first actionable failure: later
 * successes never mask it, every accepted OID is still drained, and the
 * enclosing session keeps its execution poisoned. Re-entering the stage with
 * a poisoned record rethrows that same first failure without submitting
 * anything, so a failed execution is never silently retried or reused.
 */
class MlpStageFailure {
public:
    [[nodiscard]] bool poisoned() const noexcept {
        return first_ != nullptr || admission_ != 0;
    }

    /** Keeps the first recorded wait failure and ignores later ones. */
    void record(std::exception_ptr failure) noexcept;

    /** Keeps the first recorded negative admission OID and ignores later ones. */
    void record_admission(oid token) noexcept;

    /** Rethrows the recorded first failure; requires a poisoned record. */
    [[noreturn]] void rethrow_first() const;

private:
    std::exception_ptr first_;
    oid admission_ = 0;
};

/**
 * Pure checked workspace maxima of the gate, up, mul, down, and residual
 * members over the actual exact-`R` prefill and fixed-`R1` decode views.
 * Requests the same requirements the stage later passes to its operations and
 * has no allocation, registration, lease, submission, or backend effect.
 */
[[nodiscard]] MlpWorkspaceRequirements mlp_workspace_requirements(
        DeviceOps& operations, const MlpStageViews& prefill,
        const MlpStageViews& decode);

/**
 * Reserves the five fixed members from one caller-owned raw workspace. A
 * positive member requirement must be a live range of the exact device with
 * sufficient capacity, aligned base, and required alignment; a zero-byte
 * member stays the empty default view. No allocation, query, or submission
 * happens here.
 */
[[nodiscard]] MlpWorkspace resolve_mlp_workspace(
        const Device& device, const MlpWorkspaceRequirements& requirements,
        RawWorkspaceView workspace);

/**
 * Executes the private stage: waits every supplied readiness producer, then
 * submits post-attention RMSNorm, the independent gate/up projections (both
 * accepted branches attempted and drained), SiLU, multiply, down projection,
 * and the second residual, waiting every direct producer before its consumer.
 *
 * A negative admission OID is never waited but is preserved as the first
 * failure; a failed wait never suppresses another accepted wait, and no
 * dependent stage is submitted after any failure. On failure the recorded
 * first failure is rethrown after the drain, leaving the result stores
 * unusable or unchanged as promised by the session.
 *
 * The stage only reads the supplied view objects; they are mutable lvalues
 * because every operation facade takes its output views by reference.
 */
void run_mlp_stage(DeviceOps& operations, MlpStageViews& views,
                   const MlpStageParams& params, const MlpWorkspace& workspace,
                   std::span<const oid> readiness, MlpStageFailure& failure);

}  // namespace iom::session_detail

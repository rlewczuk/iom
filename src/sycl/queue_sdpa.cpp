#include "queue_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "runtime.hpp"
#include "sdpa_stage.hpp"
#include "../iom_internal.hpp"

namespace iom::sycl_detail {
namespace {

constexpr std::size_t kSdpaScratchAlignment = 32;

[[nodiscard]] std::size_t checked_leading_planes(
        const DeviceOps::SdpaRequest& request) {
    const std::size_t leading_rank = request.q.rank - 3;
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        planes = detail::checked_mul(
                planes, request.q.dimensions[axis],
                "SYCL SDPA leading plane count overflows");
    }
    return planes;
}

// One standard tiled plane of the packed scratch and output segments holds
// `tiles(rows) * tiles(columns)` sixteen-by-sixteen slots.
[[nodiscard]] std::size_t plane_slot_span(
        std::size_t rows, std::size_t columns, const char* what) {
    const std::size_t tile_rows = detail::padded_extent(rows, what)
            / TensorSpec::TILE;
    const std::size_t tile_columns = detail::padded_extent(columns, what)
            / TensorSpec::TILE;
    return detail::checked_mul(
            detail::checked_mul(tile_rows, tile_columns, what),
            TensorSpec::TILE * TensorSpec::TILE, what);
}

// The two native matrix stages address Q, K, and V through the standard tiled
// plane slot of each operand's own final two extents, so their transformed
// leading-plane maps are copied verbatim from the value-owned view snapshots:
// the leading tuple keeps one plane-unit stride per axis, and the head axis
// stays a separate axis whose stride comes from the same snapshot (one plane on
// a dense view, or the selected step of a sliced head axis).  The score,
// probability, and PV segments are the packed scratch planes this queue owns,
// so their strides are the validated padded row-major layout.
[[nodiscard]] SdpaMatrixGeometry build_sdpa_matrix_geometry(
        const DeviceOps::SdpaRequest& request) {
    const std::size_t leading_rank = request.q.rank - 3;
    const std::size_t padded_rows = detail::padded_extent(
            request.R, "SYCL SDPA row padding overflows");
    const std::size_t padded_keys = detail::padded_extent(
            request.L, "SYCL SDPA key padding overflows");
    const std::size_t padded_depth = detail::padded_extent(
            request.D, "SYCL SDPA depth padding overflows");

    SdpaMatrixGeometry geometry;
    geometry.planes = checked_leading_planes(request);
    geometry.Hq = request.Hq;
    geometry.Hkv = request.Hkv;
    geometry.R = request.R;
    geometry.C = request.C;
    geometry.D = request.D;
    geometry.L = request.L;
    geometry.a = request.a;
    geometry.grouping = request.grouping;
    geometry.leading_rank = leading_rank;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        geometry.leading_dimensions[axis] = request.q.dimensions[axis];
        geometry.q_leading_strides[axis] = request.q.plane_strides[axis];
        geometry.k_leading_strides[axis] = request.k.plane_strides[axis];
        geometry.v_leading_strides[axis] = request.v.plane_strides[axis];
    }
    geometry.q_plane_offset = request.q.plane_offset;
    geometry.k_plane_offset = request.k.plane_offset;
    geometry.v_plane_offset = request.v.plane_offset;
    geometry.q_head_stride = request.q.plane_strides[leading_rank];
    geometry.k_head_stride = request.k.plane_strides[leading_rank];
    geometry.v_head_stride = request.v.plane_strides[leading_rank];

    geometry.padded_rows = padded_rows;
    geometry.padded_keys = padded_keys;
    geometry.padded_depth = padded_depth;
    const std::uint64_t score_row = padded_keys;
    const std::uint64_t score_head = static_cast<std::uint64_t>(
            detail::checked_mul(
                    padded_rows, padded_keys,
                    "SYCL SDPA score stride overflows"));
    geometry.scores_row_stride = score_row;
    geometry.scores_head_stride = score_head;
    geometry.scores_plane_stride = static_cast<std::uint64_t>(
            detail::checked_mul(
                    request.Hq, static_cast<std::size_t>(score_head),
                    "SYCL SDPA score plane stride overflows"));
    // Probabilities carry the identical padded layout in the BF16 segment.
    geometry.probability_row_stride = score_row;
    geometry.probability_head_stride = score_head;
    geometry.probability_plane_stride = geometry.scores_plane_stride;
    const std::uint64_t pv_row = padded_depth;
    const std::uint64_t pv_head = static_cast<std::uint64_t>(
            detail::checked_mul(
                    padded_rows, padded_depth,
                    "SYCL SDPA PV stride overflows"));
    geometry.pv_row_stride = pv_row;
    geometry.pv_head_stride = pv_head;
    geometry.pv_plane_stride = static_cast<std::uint64_t>(
            detail::checked_mul(
                    request.Hq, static_cast<std::size_t>(pv_head),
                    "SYCL SDPA PV plane stride overflows"));
    return geometry;
}

[[nodiscard]] SdpaNonmatrixRequest build_sdpa_nonmatrix_request(
        const DeviceOps::SdpaRequest& request,
        const SdpaMatrixScratchLayout& layout, float* scores,
        SdpaBf16* probabilities, const float* pv, SdpaBf16* head) {
    const std::size_t leading_rank = request.q.rank - 3;
    const std::size_t out_leading_rank = request.out.rank - 2;
    if (out_leading_rank != leading_rank) {
        throw std::invalid_argument(
                "SYCL SDPA output leading rank does not match its operands");
    }
    const std::size_t padded_rows = detail::padded_extent(
            request.R, "SYCL SDPA row padding overflows");
    const std::size_t padded_keys = detail::padded_extent(
            request.L, "SYCL SDPA key padding overflows");
    const std::size_t padded_depth = detail::padded_extent(
            request.D, "SYCL SDPA depth padding overflows");

    SdpaNonmatrixRequest nonmatrix;
    nonmatrix.scores = scores;
    nonmatrix.p_bf16 = probabilities;
    nonmatrix.v_bf16 = static_cast<const SdpaBf16*>(request.v.native_handle);
    nonmatrix.pv_fp32 = pv;
    nonmatrix.head_bf16 = head;
    nonmatrix.merged_bf16 =
            static_cast<SdpaBf16*>(request.out.native_handle);
    // V is a read-only caller operand consumed at the native matrix boundary
    // through its own standard tiled plane slot, so no linear stride set
    // describes it.  The recorded extent therefore covers exactly the selected
    // tiled planes of the owner - one plane span per leading plane and per KV
    // head, with single-element key/feature steps - which can never exceed the
    // owner's own storage.  Operand/workspace/output disjointness stays owned
    // by common admission and by the workspace validator.
    const std::size_t v_plane_slots = plane_slot_span(
            request.C, request.D, "SYCL SDPA V plane slots overflow");
    nonmatrix.v_plane_stride = v_plane_slots;
    nonmatrix.v_head_stride = v_plane_slots;
    nonmatrix.v_key_stride = 1;
    nonmatrix.v_feature_stride = 1;

    // Every supplied base pointer is already the alignment-32 base of its own
    // packed scratch segment in the caller workspace, so the per-segment plane
    // offsets stay zero - the documented normal packed-scratch case.  Re-adding
    // the layout offsets here would displace each segment by its own workspace
    // offset and push PV and head staging outside the leased range.
    nonmatrix.score_plane_offset = 0;
    nonmatrix.p_plane_offset = 0;
    nonmatrix.v_plane_offset = 0;
    nonmatrix.pv_plane_offset = 0;
    nonmatrix.head_plane_offset = 0;
    const std::size_t output_plane_slots = plane_slot_span(
            request.R, request.output_width,
            "SYCL SDPA output plane slots overflow");
    nonmatrix.merged_plane_offset = detail::checked_mul(
            request.out.plane_offset, output_plane_slots,
            "SYCL SDPA output plane offset overflows");

    nonmatrix.plane_count = checked_leading_planes(request);
    nonmatrix.query_heads = request.Hq;
    nonmatrix.kv_heads = request.Hkv;
    nonmatrix.rows = request.R;
    nonmatrix.capacity = request.C;
    nonmatrix.head_dim = request.D;
    nonmatrix.initialized_length = request.L;
    nonmatrix.causal_offset = request.a;
    nonmatrix.grouping = request.grouping;
    // A zero scale asks the device stage to derive the checked positive
    // 1/sqrt(head_dim) in FP32.
    nonmatrix.score_scale = 0.0F;

    nonmatrix.score_plane_stride = detail::checked_mul(
            detail::checked_mul(
                    request.Hq, padded_rows,
                    "SYCL SDPA score plane stride overflows"),
            padded_keys, "SYCL SDPA score plane stride overflows");
    nonmatrix.score_head_stride = detail::checked_mul(
            padded_rows, padded_keys,
            "SYCL SDPA score head stride overflows");
    nonmatrix.score_row_stride = padded_keys;
    nonmatrix.score_key_stride = 1;

    nonmatrix.p_plane_stride = nonmatrix.score_plane_stride;
    nonmatrix.p_head_stride = nonmatrix.score_head_stride;
    nonmatrix.p_row_stride = padded_keys;
    nonmatrix.p_key_stride = 1;

    nonmatrix.pv_plane_stride = detail::checked_mul(
            detail::checked_mul(
                    request.Hq, padded_rows,
                    "SYCL SDPA PV plane stride overflows"),
            padded_depth, "SYCL SDPA PV plane stride overflows");
    nonmatrix.pv_head_stride = detail::checked_mul(
            padded_rows, padded_depth,
            "SYCL SDPA PV head stride overflows");
    nonmatrix.pv_row_stride = padded_depth;
    nonmatrix.pv_feature_stride = 1;

    nonmatrix.head_plane_stride = nonmatrix.pv_plane_stride;
    nonmatrix.head_head_stride = nonmatrix.pv_head_stride;
    nonmatrix.head_row_stride = padded_depth;
    nonmatrix.head_feature_stride = 1;

    // The merged destination is the caller's standard 16x16 tiled output
    // plane: `launch_sdpa_merge_output` derives the row/column slot from
    // `rows`, `query_heads`, and `head_dim`, and only the independently
    // transformed leading planes are addressed here, in element slots.
    nonmatrix.merged_leading_rank = out_leading_rank;
    for (std::size_t axis = 0; axis < out_leading_rank; ++axis) {
        nonmatrix.merged_leading_dimensions[axis] =
                request.out.dimensions[axis];
        nonmatrix.merged_leading_strides[axis] = detail::checked_mul(
                request.out.plane_strides[axis], output_plane_slots,
                "SYCL SDPA output leading stride overflows");
    }
    return nonmatrix;
}

}  // namespace

WorkspaceRequirements sdpa_stage_workspace_requirements(
        const DeviceOps::SdpaRequest& request) {
    const SdpaMatrixScratchLayout layout = sdpa_matrix_scratch_layout(
            checked_leading_planes(request), request.Hq, request.R,
            request.L, request.D);
    return {layout.bytes, kSdpaScratchAlignment};
}

SdpaStagePlan lower_sdpa_stages(const DeviceOps::SdpaRequest& request) {
    const SdpaMatrixScratchLayout layout = sdpa_matrix_scratch_layout(
            checked_leading_planes(request), request.Hq, request.R,
            request.L, request.D);
    auto* const workspace = static_cast<unsigned char*>(
            detail::WorkspaceValidation::address(request.workspace));
    if (workspace == nullptr
            || request.workspace.byte_size() < layout.bytes) {
        throw std::invalid_argument(
                "SYCL SDPA caller workspace is not usable");
    }
    if (reinterpret_cast<std::uintptr_t>(workspace)
                    % kSdpaScratchAlignment
            != 0) {
        throw std::invalid_argument(
                "SYCL SDPA caller workspace is not 32-byte aligned");
    }

    const SdpaMatrixGeometry geometry = build_sdpa_matrix_geometry(request);
    auto* const scores =
            reinterpret_cast<float*>(workspace + layout.scores_offset);
    auto* const probabilities = reinterpret_cast<SdpaBf16*>(
            workspace + layout.probability_offset);
    auto* const pv = reinterpret_cast<float*>(workspace + layout.pv_offset);
    auto* const head = reinterpret_cast<SdpaBf16*>(
            workspace + layout.head_offset);

    SdpaStagePlan plan;
    plan.qk.geometry = geometry;
    plan.qk.q = static_cast<const unsigned char*>(
            request.q.native_handle);
    plan.qk.k = static_cast<const unsigned char*>(
            request.k.native_handle);
    plan.qk.scores = scores;
    plan.pv.geometry = geometry;
    plan.pv.p_bf16 = reinterpret_cast<const unsigned char*>(probabilities);
    plan.pv.v = static_cast<const unsigned char*>(
            request.v.native_handle);
    plan.pv.pv_fp32 = pv;
    plan.nonmatrix = build_sdpa_nonmatrix_request(
            request, layout, scores, probabilities, pv, head);

    // Both leaf validators accept the complete lowered plan before any
    // sequence, owner registration, workspace lease, output mutation, or
    // native submission exists.
    validate_sdpa_matrix_geometry(plan.qk.geometry);
    validate_sdpa_nonmatrix_request(plan.nonmatrix);
    return plan;
}

sycl::event launch_sdpa_stages(
        sycl::queue& queue, const SdpaStagePlan& plan) {
    const sycl::event qk = launch_sdpa_qk(queue, plan.qk);
    (void)launch_sdpa_scale_softmax(queue, plan.nonmatrix, qk);
    const sycl::event pv = launch_sdpa_pv(queue, plan.pv);
    return launch_sdpa_merge_output(queue, plan.nonmatrix, pv);
}

WorkspaceRequirements SyclQueue::sdpa_workspace_requirements_impl(
        const SdpaRequest& request) {
    // Pure capability decision of the implemented SDPA leaf set.  The common
    // facade has already accepted only the current BF16 leaf, so the only
    // remaining capability is the immutable device fact this queue was built
    // with: the queried subgroup-16 BF16/BF16/FP32 joint-matrix facility that
    // the native QK/PV stages queue.  The hook performs no allocation,
    // registration, lease, sequence, submission, queue/arena inspection, or
    // operand access.
    if (!sdpa_supported_) {
        throw detail::UnsupportedOperation();
    }
    return sdpa_stage_workspace_requirements(request);
}

oid SyclQueue::sdpa_impl(const SdpaRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    // Capability and the complete lowered plan are decided before the
    // submission sequence, the four-owner registration, the workspace lease,
    // the fixed queue resources, and any output mutation, so an unported device
    // and an unrepresentable request are both rejected repeatably without side
    // effects.
    if (!sdpa_supported_) {
        throw detail::UnsupportedOperation();
    }
    const SdpaStagePlan plan = lower_sdpa_stages(request);
    if (consume_submission_fault(SubmissionFault::state_allocation)) {
        throw std::bad_alloc();
    }
    auto state = std::make_shared<SyclFenceState>();
    if (consume_submission_fault(SubmissionFault::fence_construction)) {
        throw std::bad_alloc();
    }
    detail::Fence fence = build_sycl_fence(state);
    return submit_sdpa(
            request, *state_, registry_queue_id_, fence,
            [this, state, plan](
                    std::uint64_t sequence, const SdpaRequest& captured,
                    detail::SdpaEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.sdpa_plan.emplace(plan);
                task.sdpa_entries = entries;
                task.sdpa_lease = captured.workspace_lease;
                worker_.submit_copy(std::move(task));
            });
}

void SyclQueue::execute_sdpa(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    {
        SyclSequenceOutcome outcome;
        outcome.state = task.state;
        outcome.workspace_lease = task.sdpa_lease;
        outcome.sdpa_entries = task.sdpa_entries;
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence, std::move(outcome));
        if (!inserted) {
            throw std::logic_error(
                    "duplicate SYCL outstanding-work sequence");
        }
    }

    bool native_attempted = false;
    try {
        const auto completion_slot = completion_pool_->try_acquire();
        if (!completion_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_completion_slot(
                *completion_pool_, *completion_slot);
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }
        // The four joined stages are descriptors and device addresses only, so
        // no metadata slot is consumed.  Once the first native enqueue is
        // attempted the accepted outcome is retained even if a later enqueue
        // throws: the runtime may have submitted work before reporting the
        // error, and only a successful drain proves completion.
        native_attempted = true;
        sycl::event event = launch_sdpa_stages(queue_, *task.sdpa_plan);
        if (launch_calls.kernel_launched != nullptr) {
            launch_calls.kernel_launched();
        }
        task.state->set_event(std::move(event));
        // The SDPA conformance seam models a semantic post-acceptance failure:
        // all four stages are enqueued, so native completion stays proven and
        // the retained failure remains observable on every later wait while the
        // caller workspace lease is released with that proof.
        if (consume_submission_fault(
                    SubmissionFault::sdpa_post_acceptance_failure)) {
            task.state->mark_completion_proven();
            throw std::runtime_error(
                    "injected SYCL SDPA post-acceptance failure");
        }
        if (consume_submission_fault(SubmissionFault::post_launch)) {
            throw std::runtime_error(
                    "injected SYCL post-launch failure");
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (!native_attempted) {
            task.state->mark_completion_proven();
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_.erase(task.sequence);
            }
            task.fence = nullptr;
            task.state.reset();
            throw;
        }
        task.state->set_failure(failure);
    }
}

}  // namespace iom::sycl_detail

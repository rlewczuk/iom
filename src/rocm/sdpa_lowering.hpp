#pragma once

// ---------------------------------------------------------------------------
// Causal grouped-query scaled dot-product attention: the ROCm lowering.
//
// Common admission has already validated and snapshotted the request this
// translation-unit-private seam consumes: ranks, leading tuples, GQA
// divisibility, the causal window, exact device identity, dtype,
// quantization, aliasing, and the caller-owned workspace range. The seam
// itself owns only the backend-private lowering. It carves the conservative
// six-segment scratch out of that caller workspace, sequences the completed
// native matrix stages (src/rocm/sdpa_matrix.hpp) and nonmatrix stages
// (src/rocm/sdpa_softmax.hpp) on the queue's existing nonblocking stream, and
// copies the merged result into the caller's standard-tiled output. Nothing
// here allocates, stages through the host, creates a stream or its own event,
// or synchronizes; the shared queue records the completion event after every
// stage of this chain is enqueued.
//
// The seam is duck-typed over the request so this header never names the
// shared queue's protected request type, and it is deliberately kept out of
// line with the rest of the backend: the lowering is compiled only into the
// ROCm translation units that include it.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "copy.hpp"
#include "sdpa_matrix.hpp"
#include "sdpa_softmax.hpp"
#include "../iom_internal.hpp"

#define IOM_SDPA_GLOBAL __global__
#define IOM_SDPA_LAUNCH(kernel, blocks, threads, stream, ...) \
    hipLaunchKernelGGL( \
            kernel, dim3(blocks), dim3(threads), 0, stream, __VA_ARGS__)

namespace iom::rocm_detail {
namespace {

constexpr unsigned int kSdpaStoreThreads = 256;
constexpr unsigned int kSdpaStoreMaxBlocks = 65535;

[[nodiscard]] std::size_t sdpa_checked_add(
        std::size_t left, std::size_t right, const char* message) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::overflow_error(message);
    }
    return left + right;
}

[[nodiscard]] std::size_t sdpa_checked_mul(
        std::size_t left, std::size_t right, const char* message) {
    if (right != 0
            && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(message);
    }
    return left * right;
}

[[nodiscard]] std::size_t sdpa_align32(
        std::size_t value, const char* message) {
    const std::size_t bumped = sdpa_checked_add(value, 31, message);
    return bumped - bumped % 32;
}

[[nodiscard]] std::size_t sdpa_pad16(
        std::size_t value, const char* message) {
    const std::size_t bumped = sdpa_checked_add(value, 15, message);
    return bumped - bumped % 16;
}

// Merged staging -> caller output. The merged region is the packed row-major
// [planes, R, Hq*D] BF16 result the nonmatrix merge stage produced; the
// destination is the caller's standard 16x16 tile-major owner, reached
// through the snapshotted plane offset and leading plane strides, so GQA
// heads, transformed leading tuples, and independent planes are honored
// without a broadcast.
struct SdpaOutputStoreMetadata {
    const std::uint16_t* merged = nullptr;
    unsigned char* out = nullptr;
    std::uint64_t plane_offset = 0;
    std::uint64_t plane_count = 0;
    std::uint64_t rows = 0;
    std::uint64_t width = 0;
    std::uint32_t leading_rank = 0;
    std::uint64_t leading_dimensions[kSdpaMatrixMaxLeadingRank]{};
    std::uint64_t leading_strides[kSdpaMatrixMaxLeadingRank]{};
};

static_assert(std::is_trivially_copyable_v<SdpaOutputStoreMetadata>);

// One logical output element per iteration: the merged value is copied to the
// logical slot `out[plane, row, h * D + d]` of the tile-major owner. Only
// logical cells are written; tile padding, capacity tail, and every cell
// outside the logical region keep their previous bytes.
IOM_SDPA_GLOBAL void sdpa_output_store_kernel(
        SdpaOutputStoreMetadata m) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
            * static_cast<std::uint64_t>(blockDim.x)
            + static_cast<std::uint64_t>(threadIdx.x);
    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x)
            * static_cast<std::uint64_t>(gridDim.x);
    const std::uint64_t plane_elements = m.rows * m.width;
    const std::uint64_t total = m.plane_count * plane_elements;
    const std::uint64_t tile_columns = (m.width + 15) / 16;
    const std::uint64_t tile_rows = (m.rows + 15) / 16;
    for (std::uint64_t work = index; work < total; work += stride) {
        const std::uint64_t plane = work / plane_elements;
        const std::uint64_t in_plane = work % plane_elements;
        const std::uint64_t row = in_plane / m.width;
        const std::uint64_t column = in_plane % m.width;
        std::uint64_t destination_plane = m.plane_offset;
        std::uint64_t rest = plane;
        for (std::uint32_t axis = m.leading_rank; axis-- > 0;) {
            const std::uint64_t coordinate =
                    rest % m.leading_dimensions[axis];
            rest /= m.leading_dimensions[axis];
            destination_plane += coordinate * m.leading_strides[axis];
        }
        const std::uint64_t slot =
                (destination_plane * tile_rows * tile_columns
                 + (row / 16) * tile_columns + column / 16) * 256
                + (row % 16) * 16 + column % 16;
        reinterpret_cast<std::uint16_t*>(m.out)[slot] = m.merged[work];
    }
}

// The conservative six-segment layout of one admitted request. The five stage
// segments are exactly the matrix leaf's own checked query, so the reported
// requirement, the stages' internal scratch-range checks, and the pointers
// the stages receive are one arithmetic; the total adds the merged staging
// region after them. Every product, `pad16`, byte conversion, 32-byte
// alignment, segment addition, and offset is checked.
struct RocmSdpaLayout {
    SdpaMatrixWorkspaceRequirements matrix{};
    std::size_t Mp = 0;
    std::size_t Lp = 0;
    std::size_t Dp = 0;
    std::size_t merged_offset = 0;
    std::size_t merged_bytes = 0;
    std::size_t bytes = 0;
};

[[nodiscard]] RocmSdpaLayout sdpa_layout(
        std::size_t planes, std::size_t Hq, std::size_t Hkv,
        std::size_t R, std::size_t L, std::size_t D,
        std::size_t output_width) {
    RocmSdpaLayout result;
    result.matrix = sdpa_matrix_workspace_requirements(
            planes, Hq, Hkv, R, L, D);
    result.Mp = sdpa_pad16(R, "ROCm SDPA row padding overflows");
    result.Lp = sdpa_pad16(L, "ROCm SDPA length padding overflows");
    result.Dp = sdpa_pad16(D, "ROCm SDPA feature padding overflows");
    result.merged_offset = sdpa_align32(
            result.matrix.bytes, "ROCm SDPA merged offset overflows");
    const std::size_t merged_elements = sdpa_checked_mul(
            sdpa_checked_mul(
                    planes, R, "ROCm SDPA merged extent overflows"),
            output_width, "ROCm SDPA merged extent overflows");
    result.merged_bytes = sdpa_align32(
            sdpa_checked_mul(
                    merged_elements, sizeof(std::uint16_t),
                    "ROCm SDPA merged bytes overflow"),
            "ROCm SDPA merged alignment overflows");
    result.bytes = sdpa_align32(
            sdpa_checked_add(
                    result.merged_offset, result.merged_bytes,
                    "ROCm SDPA workspace bytes overflow"),
            "ROCm SDPA workspace alignment overflows");
    return result;
}

struct RocmSdpaGeometry {
    std::size_t leading_rank = 0;
    std::size_t planes = 0;
    std::size_t ordinal = 0;
    std::size_t width = 0;
    RocmSdpaLayout layout{};
};

// Snapshot one operand as the matrix stage contract expects it: the owning
// native handle, the selected plane, the independent leading coordinates, and
// the head axis kept separate so GQA never broadcasts a query head.
template <typename View>
[[nodiscard]] SdpaMatrixView sdpa_matrix_view(
        const View& view, std::size_t leading_rank, std::size_t planes,
        std::size_t heads) {
    SdpaMatrixView result;
    result.native_handle = view.native_handle;
    result.plane_offset = view.plane_offset;
    result.leading_rank = leading_rank;
    result.plane_count = planes;
    result.head_count = heads;
    result.head_stride = view.plane_strides[leading_rank];
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        result.leading_dimensions[axis] = view.dimensions[axis];
        result.leading_strides[axis] = view.plane_strides[axis];
    }
    return result;
}

// The request is a template parameter because the shared queue's immutable
// request type is a protected backend hook type; the lowering reads it by
// value-copied snapshot and never widens or retains it.
template <typename Request>
[[nodiscard]] RocmSdpaGeometry sdpa_geometry(const Request& request) {
    if (request.q.rank < 3 || request.q.rank > 8
            || request.k.rank != request.q.rank
            || request.v.rank != request.q.rank
            || request.out.rank + 1 != request.q.rank) {
        throw std::invalid_argument(
                "ROCm SDPA request ranks are inconsistent");
    }
    const std::size_t leading_rank = request.q.rank - 3;
    if (leading_rank > kSdpaMatrixMaxLeadingRank) {
        throw std::invalid_argument(
                "ROCm SDPA leading rank exceeds the fixed ABI");
    }
    std::size_t planes = 1;
    for (std::size_t axis = 0; axis < leading_rank; ++axis) {
        planes = sdpa_checked_mul(
                planes, request.q.dimensions[axis],
                "ROCm SDPA leading plane count overflows");
    }
    const std::size_t width = sdpa_checked_mul(
            request.Hq, request.D, "ROCm SDPA output width overflows");
    if (width != request.output_width) {
        throw std::invalid_argument(
                "ROCm SDPA output width is inconsistent");
    }
    if (request.q.device_identity == nullptr) {
        throw std::invalid_argument("ROCm SDPA request has no device");
    }
    const std::uint32_t ordinal =
            request.q.device_identity->backend_device();
    if (ordinal
            > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
                "ROCm SDPA device ordinal exceeds the native ABI");
    }
    RocmSdpaGeometry geometry;
    geometry.leading_rank = leading_rank;
    geometry.planes = planes;
    geometry.ordinal = static_cast<std::size_t>(ordinal);
    geometry.width = width;
    geometry.layout = sdpa_layout(
            planes, request.Hq, request.Hkv, request.R, request.L,
            request.D, width);
    return geometry;
}

// The nonmatrix stages address exactly the packed scratch the matrix stages
// produce and consume: contiguous per (plane, head) blocks of
// pad16(R) x pad16(L) scores/probabilities and pad16(R) x pad16(D) PV values,
// with the merged region packed as [planes, R, Hq*D]. One element-strided
// description serves both stage families, so no conversion pass, extra
// segment, or hidden allocation sits between them. The scale is the frozen
// FP32 `1/sqrt(D)` factor every formed score is multiplied by.
template <typename Request>
[[nodiscard]] SdpaSoftmaxRequest sdpa_softmax_request(
        const Request& request, const RocmSdpaGeometry& geometry,
        unsigned char* const base) {
    const RocmSdpaLayout& layout = geometry.layout;
    const std::uint64_t Mp = static_cast<std::uint64_t>(layout.Mp);
    const std::uint64_t Lp = static_cast<std::uint64_t>(layout.Lp);
    const std::uint64_t Dp = static_cast<std::uint64_t>(layout.Dp);
    const std::uint64_t width = static_cast<std::uint64_t>(geometry.width);
    // One element-strided description per packed segment. The plane offset is
    // `plane * Hq + head` in the packed family order, so the plane stride is
    // the full per-plane head block and the head stride is one block: the
    // strided layout sums the two independently and must never conflate them.
    const std::uint64_t score_head = static_cast<std::uint64_t>(
            sdpa_checked_mul(
                    layout.Mp, layout.Lp,
                    "ROCm SDPA score head size overflows"));
    const std::uint64_t score_plane = static_cast<std::uint64_t>(
            sdpa_checked_mul(
                    static_cast<std::size_t>(score_head),
                    static_cast<std::size_t>(request.Hq),
                    "ROCm SDPA score plane size overflows"));
    const std::uint64_t pv_head = static_cast<std::uint64_t>(
            sdpa_checked_mul(
                    layout.Mp, layout.Dp,
                    "ROCm SDPA PV head size overflows"));
    const std::uint64_t pv_plane = static_cast<std::uint64_t>(
            sdpa_checked_mul(
                    static_cast<std::size_t>(pv_head),
                    static_cast<std::size_t>(request.Hq),
                    "ROCm SDPA PV plane size overflows"));
    const std::uint64_t merged_plane = static_cast<std::uint64_t>(
            sdpa_checked_mul(
                    request.R, geometry.width,
                    "ROCm SDPA merged plane size overflows"));

    SdpaSoftmaxRequest softmax{};
    softmax.planes = static_cast<std::uint64_t>(geometry.planes);
    softmax.hq = static_cast<std::uint64_t>(request.Hq);
    softmax.hkv = static_cast<std::uint64_t>(request.Hkv);
    softmax.rows = static_cast<std::uint64_t>(request.R);
    softmax.capacity = static_cast<std::uint64_t>(request.C);
    softmax.head_dim = static_cast<std::uint64_t>(request.D);
    softmax.a = static_cast<std::uint64_t>(request.a);
    softmax.L = static_cast<std::uint64_t>(request.L);
    softmax.mask = sdpa_required_mask;
    softmax.scale = 1.0F / std::sqrt(static_cast<float>(request.D));
    softmax.scores = reinterpret_cast<const float*>(
            base + layout.matrix.scores_offset);
    softmax.p_bf16 = reinterpret_cast<std::uint16_t*>(
            base + layout.matrix.p_bf16_offset);
    softmax.pv_bf16 = reinterpret_cast<const std::uint16_t*>(
            base + layout.matrix.pv_bf16_offset);
    softmax.merged = reinterpret_cast<std::uint16_t*>(
            base + layout.merged_offset);
    softmax.scores_layout = SdpaStageLayout{
            0, score_plane, score_head, Lp, 1, Mp, Lp};
    softmax.probability_layout = softmax.scores_layout;
    softmax.pv_layout = SdpaStageLayout{
            0, pv_plane, pv_head, Dp, 1, Mp, Dp};
    softmax.merged_layout = SdpaStageLayout{
            0, merged_plane, 0, width, 1,
            static_cast<std::uint64_t>(request.R), width};
    return softmax;
}

}  // namespace

// Pure conservative requirement of one admitted request: alignment 32 over
// the checked six-segment sum. It allocates nothing, registers no owner,
// takes no lease, consumes no OID, touches no queue or native handle, and
// depends on no prior completion; the capability gate has already run in the
// shared query path before this hook is reached.
template <typename Request>
WorkspaceRequirements gpu_policy::sdpa_workspace_requirements(
        const Request& request) {
    return WorkspaceRequirements{
            sdpa_geometry(request).layout.bytes, 32};
}

// The joined lowering. Every stage is enqueued in order on this queue's
// existing nonblocking stream; the caller's already validated and leased
// workspace supplies all six segments, and the completion event belongs to
// the shared queue, which records it after this returns.
template <typename Request>
void gpu_policy::launch_sdpa(
        gpu_policy::stream_type stream, const Request& request) {
    const RocmSdpaGeometry geometry = sdpa_geometry(request);
    const RocmSdpaLayout& layout = geometry.layout;
    if (request.workspace_requirements.bytes != layout.bytes
            || request.workspace_requirements.alignment != 32) {
        throw std::invalid_argument(
                "ROCm SDPA workspace layout is inconsistent");
    }
    void* const workspace = detail::WorkspaceValidation::address(
            request.workspace);
    if (workspace == nullptr
            || request.workspace.byte_size() < layout.bytes) {
        throw std::invalid_argument("ROCm SDPA workspace is unavailable");
    }
    auto* const base = static_cast<unsigned char*>(workspace);
    auto* const kv_pack = base + layout.matrix.kv_pack_offset;

    // QK: pack Q and the initialized K prefix, then the native BF16 WMMA
    // product into the packed FP32 score segment.
    SdpaMatrixCompletionState completion{};
    SdpaQkRequest qk{};
    qk.q = sdpa_matrix_view(
            request.q, geometry.leading_rank, geometry.planes, request.Hq);
    qk.k = sdpa_matrix_view(
            request.k, geometry.leading_rank, geometry.planes, request.Hkv);
    qk.scratch.q_pack = base;
    qk.scratch.q_pack_bytes = layout.matrix.q_pack_bytes;
    qk.scratch.kv_pack = kv_pack;
    qk.scratch.kv_pack_bytes = layout.matrix.kv_pack_bytes;
    qk.scratch.scores = base + layout.matrix.scores_offset;
    qk.scratch.scores_bytes = layout.matrix.scores_bytes;
    qk.completion = &completion;
    qk.device_ordinal = geometry.ordinal;
    qk.Hq = request.Hq;
    qk.Hkv = request.Hkv;
    qk.R = request.R;
    qk.C = request.C;
    qk.D = request.D;
    qk.a = request.a;
    qk.L = request.L;
    qk.grouping = request.grouping;
    if (launch_sdpa_qk(stream, qk) != SdpaMatrixResult::submitted
            || !completion.qk_submitted) {
        throw detail::UnsupportedOperation();
    }
    // Accepted-failure checkpoint: an armed test fault throws here, after the
    // native QK stage is enqueued and before any consumer stage, so the
    // conformance suite proves post-acceptance stage failure semantics.
    sdpa_stage_checkpoint();

    // Scale, mask, stable special-value softmax, and the single RNE BF16
    // probability boundary.
    const SdpaSoftmaxRequest softmax =
            sdpa_softmax_request(request, geometry, base);
    launch_sdpa_softmax(stream, softmax);

    // PV: reuse the same K/V pack for the initialized V prefix, consume the
    // BF16 probabilities, and accumulate FP32 into planar BF16 PV.
    SdpaPvRequest pv{};
    pv.v = sdpa_matrix_view(
            request.v, geometry.leading_rank, geometry.planes, request.Hkv);
    pv.scratch.p_bf16 = base + layout.matrix.p_bf16_offset;
    pv.scratch.p_bf16_bytes = layout.matrix.p_bf16_bytes;
    pv.scratch.kv_pack = kv_pack;
    pv.scratch.kv_pack_bytes = layout.matrix.kv_pack_bytes;
    pv.scratch.pv_bf16 = base + layout.matrix.pv_bf16_offset;
    pv.scratch.pv_bf16_bytes = layout.matrix.pv_bf16_bytes;
    pv.completion = &completion;
    pv.device_ordinal = geometry.ordinal;
    pv.Hq = request.Hq;
    pv.Hkv = request.Hkv;
    pv.R = request.R;
    pv.C = request.C;
    pv.D = request.D;
    pv.a = request.a;
    pv.L = request.L;
    pv.grouping = request.grouping;
    if (launch_sdpa_pv(stream, pv) != SdpaMatrixResult::submitted
            || !completion.pv_submitted) {
        throw detail::UnsupportedOperation();
    }

    // Merge the planar PV values into the packed logical [P, R, Hq*D]
    // staging region, then copy only logical cells into the caller's
    // standard-tiled output.
    launch_sdpa_merge(stream, softmax);

    SdpaOutputStoreMetadata store{};
    store.merged = reinterpret_cast<const std::uint16_t*>(
            base + layout.merged_offset);
    store.out = static_cast<unsigned char*>(request.out.native_handle);
    store.plane_offset = request.out.plane_offset;
    store.plane_count = geometry.planes;
    store.rows = request.R;
    store.width = geometry.width;
    store.leading_rank = static_cast<std::uint32_t>(geometry.leading_rank);
    for (std::size_t axis = 0; axis < geometry.leading_rank; ++axis) {
        store.leading_dimensions[axis] = request.out.dimensions[axis];
        store.leading_strides[axis] = request.out.plane_strides[axis];
    }
    const std::size_t total = sdpa_checked_mul(
            geometry.planes,
            sdpa_checked_mul(
                    request.R, geometry.width,
                    "ROCm SDPA store extent overflows"),
            "ROCm SDPA store extent overflows");
    const std::size_t blocks = sdpa_checked_add(
            total, kSdpaStoreThreads - 1,
            "ROCm SDPA store launch count overflows")
            / kSdpaStoreThreads;
    IOM_SDPA_LAUNCH(
            sdpa_output_store_kernel,
            static_cast<unsigned int>(
                    blocks < kSdpaStoreMaxBlocks ? blocks
                                                 : kSdpaStoreMaxBlocks),
            kSdpaStoreThreads, stream, store);
    check_hip("HIP SDPA output store kernel launch", hipGetLastError());
}


}  // namespace iom::rocm_detail

#undef IOM_SDPA_LAUNCH
#undef IOM_SDPA_GLOBAL

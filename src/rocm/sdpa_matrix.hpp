#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>


namespace iom::rocm_detail {

// Q/K/V have at most five independent leading axes before their head axis
// (the public tensor rank is capped at eight).  The arrays are deliberately
// owned by each request so a deferred queue callback never retains a borrowed
// TensorView or host span.
inline constexpr std::size_t kSdpaMatrixMaxLeadingRank = 5;

enum class SdpaMatrixCapability : std::uint8_t {
    unsupported,
    gfx1201_wave32_wmma,
};

enum class SdpaMatrixResult : std::uint8_t {
    unsupported,
    submitted,
};


// Checked conservative caller-owned layout for the five stage segments.  The
// K/V pack is reused sequentially by QK and PV, so it occurs once in the
// total.  Every offset is 32-byte aligned and every product/round-up is
// checked before this value is returned.
struct SdpaMatrixWorkspaceRequirements {
    std::size_t alignment = 32;
    std::size_t q_pack_offset = 0;
    std::size_t q_pack_bytes = 0;
    std::size_t kv_pack_offset = 0;
    std::size_t kv_pack_bytes = 0;
    std::size_t scores_offset = 0;
    std::size_t scores_bytes = 0;
    std::size_t p_bf16_offset = 0;
    std::size_t p_bf16_bytes = 0;
    std::size_t pv_bf16_offset = 0;
    std::size_t pv_bf16_bytes = 0;
    std::size_t bytes = 0;
};
// A value-owned snapshot of one head-planar BF16 operand.  Plane offsets and
// strides are element-plane units, as returned by TensorView.  `head_stride`
// is the stride of the head axis and is kept separate from independent leading
// plane strides so GQA never accidentally broadcasts a query head.
struct SdpaMatrixView {
    const void* native_handle = nullptr;
    std::size_t plane_offset = 0;
    std::size_t leading_rank = 0;
    std::size_t plane_count = 1;
    std::size_t head_count = 0;
    std::size_t head_stride = 0;
    std::array<std::size_t, kSdpaMatrixMaxLeadingRank>
            leading_dimensions{};
    std::array<std::size_t, kSdpaMatrixMaxLeadingRank>
            leading_strides{};
};


struct SdpaQkScratch {
    // q_pack is [P,Hq,Mp,Dp] and kv_pack is [P,Hkv,Lp,Dp], both standard
    // row-major 16x16-tiled BF16 payloads.  scores is [P,Hq,Mp,Lp] FP32.
    void* q_pack = nullptr;
    std::size_t q_pack_bytes = 0;
    void* kv_pack = nullptr;
    std::size_t kv_pack_bytes = 0;
    void* scores = nullptr;
    std::size_t scores_bytes = 0;
};

struct SdpaPvScratch {
    // p_bf16 is [P,Hq,Mp,Lp], kv_pack is reused for [P,Hkv,Lp,Dp], and
    // pv_bf16 is [P,Hq,Mp,Dp].  All payloads are standard tiled BF16.
    const void* p_bf16 = nullptr;
    std::size_t p_bf16_bytes = 0;
    void* kv_pack = nullptr;
    std::size_t kv_pack_bytes = 0;
    void* pv_bf16 = nullptr;
    std::size_t pv_bf16_bytes = 0;
};

// The completion marker belongs to the integration caller.  The matrix leaf
// never creates an event or queue; it only records which native stage was
// accepted so the caller can retain owners and its workspace lease through the
// queue's terminal completion boundary.
struct SdpaMatrixCompletionState {
    hipEvent_t event = nullptr;
    std::uint64_t sequence = 0;
    bool qk_submitted = false;
    bool pv_submitted = false;
};

struct SdpaQkRequest {
    SdpaMatrixView q;
    SdpaMatrixView k;
    SdpaQkScratch scratch;
    SdpaMatrixCompletionState* completion = nullptr;
    std::size_t device_ordinal = 0;
    std::size_t Hq = 0;
    std::size_t Hkv = 0;
    std::size_t R = 0;
    std::size_t C = 0;
    std::size_t D = 0;
    std::size_t a = 0;
    std::size_t L = 0;
    std::size_t grouping = 0;
};

struct SdpaPvRequest {
    SdpaMatrixView v;
    SdpaPvScratch scratch;
    SdpaMatrixCompletionState* completion = nullptr;
    std::size_t device_ordinal = 0;
    std::size_t Hq = 0;
    std::size_t Hkv = 0;
    std::size_t R = 0;
    std::size_t C = 0;
    std::size_t D = 0;
    std::size_t a = 0;
    std::size_t L = 0;
    std::size_t grouping = 0;
};

// Pure checked layout query; it performs no device or queue operation.
[[nodiscard]] SdpaMatrixWorkspaceRequirements
sdpa_matrix_workspace_requirements(
        std::size_t planes, std::size_t Hq, std::size_t Hkv,
        std::size_t R, std::size_t L, std::size_t D);

// This is a pure capability query.  Only an exact gfx1201 device with a
// wave32 WMMA image compiled for the matching builtin is admitted.  In
// particular, gfx1036 and any target for which execution evidence is absent
// remain Unsupported before a native launch.
[[nodiscard]] SdpaMatrixCapability sdpa_matrix_capability(
        int device_ordinal) noexcept;

// The two entries are deliberately independent: QK emits FP32 scores and PV
// consumes an explicit BF16 probability boundary.  They launch only on the
// caller-provided stream, never allocate, create a queue, register owners, or
// inspect public views.  Invalid immutable metadata or a short/misaligned
// caller scratch range throws before a kernel launch; an unproved device
// returns Unsupported without changing the completion marker.
[[nodiscard]] SdpaMatrixResult launch_sdpa_qk(
        hipStream_t stream, const SdpaQkRequest& request);
[[nodiscard]] SdpaMatrixResult launch_sdpa_pv(
        hipStream_t stream, const SdpaPvRequest& request);

}  // namespace iom::rocm_detail

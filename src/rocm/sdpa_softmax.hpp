#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <type_traits>

namespace iom::rocm_detail {

// The nonmatrix stages are deliberately private to the ROCm SDPA lowering.
// Every stride and offset is measured in elements of the pointed-to buffer, not
// bytes.  `rows` and `columns` describe the physical region that the stage may
// write; the logical extents live in SdpaSoftmaxRequest.
struct SdpaStageLayout final {
    std::uint64_t plane_offset = 0;
    std::uint64_t plane_stride = 0;
    std::uint64_t head_stride = 0;
    std::uint64_t row_stride = 0;
    std::uint64_t column_stride = 1;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
};

// The two masks are explicit so a caller cannot accidentally turn an
// initialized-prefix request into an unbounded cache read.  A valid request
// requires both bits.  Physical probability rows/columns are retained in the
// probability layout and are canonicalized to +0 as well.
enum SdpaMaskBits : std::uint32_t {
    sdpa_mask_initialized_prefix = 1u << 0,
    sdpa_mask_causal = 1u << 1,
};
inline constexpr std::uint32_t sdpa_required_mask =
        sdpa_mask_initialized_prefix | sdpa_mask_causal;

// Immutable, caller-owned device stage contract.  `scores` contains formed
// FP32 QK values, `p_bf16` is the explicit RNE BF16 probability boundary,
// `pv_bf16` contains per-query-head BF16 PV values, and `merged` is the
// device-local [planes, rows, hq * head_dim] result.  None of the pointers is
// allocated, retained, or interpreted through host metadata by the entry
// points.
struct SdpaSoftmaxRequest final {
    std::uint64_t planes = 1;
    std::uint64_t hq = 0;
    std::uint64_t hkv = 0;
    std::uint64_t rows = 0;
    std::uint64_t capacity = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t a = 0;
    std::uint64_t L = 0;
    std::uint32_t mask = sdpa_required_mask;
    float scale = 0.0F;

    const float* scores = nullptr;
    std::uint16_t* p_bf16 = nullptr;
    const std::uint16_t* pv_bf16 = nullptr;
    std::uint16_t* merged = nullptr;

    SdpaStageLayout scores_layout{};
    SdpaStageLayout probability_layout{};
    SdpaStageLayout pv_layout{};
    SdpaStageLayout merged_layout{};
};

static_assert(std::is_trivially_copyable_v<SdpaStageLayout>);
static_assert(std::is_trivially_copyable_v<SdpaSoftmaxRequest>);

// These validators are host-side checked-layout seams.  They do no allocation,
// device inspection, stream work, or pointer dereference.  The stage launchers
// call the relevant validator immediately before enqueuing their own kernel.
void validate_sdpa_softmax_request(const SdpaSoftmaxRequest& request);
void validate_sdpa_merge_request(const SdpaSoftmaxRequest& request);

// Enqueue only the requested device-local stage on the supplied stream.  The
// calls are independent: a matrix producer may submit QK/PV between the
// probability and merge stages without a host round trip or queue ownership
// change.
void launch_sdpa_softmax(
        hipStream_t stream, const SdpaSoftmaxRequest& request);
void launch_sdpa_merge(
        hipStream_t stream, const SdpaSoftmaxRequest& request);

}  // namespace iom::rocm_detail

#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace iom::sycl_detail {

// Backend-private BF16 carrier.  The nonmatrix stage deliberately keeps the
// carrier as raw bits so its conversion boundary is explicit and independent
// of any optional SYCL BF16 arithmetic extension.
using SdpaBf16 = std::uint16_t;

// Exact BF16 matrix-operand DAZ rule.  The helper is usable by the native PV
// kernel without allocating or materializing a second V range.
[[nodiscard]] inline SdpaBf16 sdpa_bf16_daz(SdpaBf16 bits) noexcept {
    const SdpaBf16 exponent = static_cast<SdpaBf16>(bits & 0x7f80u);
    const SdpaBf16 fraction = static_cast<SdpaBf16>(bits & 0x007fu);
    return exponent == 0 && fraction != 0
            ? static_cast<SdpaBf16>(bits & 0x8000u) : bits;
}

// Immutable device-stage description.  All strides are in elements.  Scores,
// probabilities, PV, and head staging are independent caller-owned scratch
// ranges; V remains a read-only caller operand.  The score and probability
// ranges use logical [plane, head, row, key] coordinates (their row/key
// pitches may include the fixed Rp/Lp padding); PV and head staging use
// [plane, head, row, feature] coordinates; V uses [plane, kv-head, key,
// feature]; merged output uses [plane, row, head, feature].
struct SdpaNonmatrixRequest final {
    float* scores = nullptr;
    SdpaBf16* p_bf16 = nullptr;
    const SdpaBf16* v_bf16 = nullptr;
    const float* pv_fp32 = nullptr;
    SdpaBf16* head_bf16 = nullptr;
    SdpaBf16* merged_bf16 = nullptr;

    // Offsets are element offsets from each supplied base pointer.  Zero is
    // the normal packed-scratch case; nonzero values preserve transformed
    // leading-plane views without retaining a caller TensorView.
    std::size_t score_plane_offset = 0;
    std::size_t p_plane_offset = 0;
    std::size_t v_plane_offset = 0;
    std::size_t pv_plane_offset = 0;
    std::size_t head_plane_offset = 0;
    std::size_t merged_plane_offset = 0;

    std::size_t plane_count = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t rows = 0;
    std::size_t capacity = 0;
    std::size_t head_dim = 0;
    std::size_t initialized_length = 0;
    std::size_t causal_offset = 0;
    std::size_t grouping = 0;

    // A zero scale asks the device stage to derive 1/sqrt(head_dim) in FP32.
    // Integrations may provide the already checked positive FP32 scale.
    float score_scale = 0.0F;

    std::size_t score_plane_stride = 0;
    std::size_t score_head_stride = 0;
    std::size_t score_row_stride = 0;
    std::size_t score_key_stride = 0;

    std::size_t p_plane_stride = 0;
    std::size_t p_head_stride = 0;
    std::size_t p_row_stride = 0;
    std::size_t p_key_stride = 0;

    std::size_t v_plane_stride = 0;
    std::size_t v_head_stride = 0;
    std::size_t v_key_stride = 0;
    std::size_t v_feature_stride = 0;

    std::size_t pv_plane_stride = 0;
    std::size_t pv_head_stride = 0;
    std::size_t pv_row_stride = 0;
    std::size_t pv_feature_stride = 0;

    std::size_t head_plane_stride = 0;
    std::size_t head_head_stride = 0;
    std::size_t head_row_stride = 0;
    std::size_t head_feature_stride = 0;

    std::size_t merged_plane_stride = 0;
    std::size_t merged_row_stride = 0;
    std::size_t merged_head_stride = 0;
    std::size_t merged_feature_stride = 0;
};

// Host-side representation and range checks shared by every stage entry.  It
// performs no device query, allocation, registration, queue submission, or
// data access; integration remains responsible for exact-device identity and
// workspace leasing.
void validate_sdpa_nonmatrix_request(
        const SdpaNonmatrixRequest& request);

// Scale and mask formed FP32 QK scores, compute the fixed special-value-aware
// stable softmax for every logical row, and write one BF16-RNE probability for
// each logical key.  Only included keys are read; future keys and capacity or
// physical tails are left untouched.  The overload with a dependency is used
// by the in-order integration chain; neither overload waits on the host.
[[nodiscard]] sycl::event launch_sdpa_scale_softmax(
        sycl::queue& queue, const SdpaNonmatrixRequest& request);
[[nodiscard]] sycl::event launch_sdpa_scale_softmax(
        sycl::queue& queue, const SdpaNonmatrixRequest& request,
        const sycl::event& dependency);
// The native PV loader applies sdpa_bf16_daz to each participating V operand
// at the matrix boundary.  No V scratch range is materialized.


// Convert FP32 PV into BF16 head staging once, then merge the logical head
// coordinates into [plane,row,head,feature] output with canonical +0 stores.
// No padded row/feature or excluded output cell is read or written.
[[nodiscard]] sycl::event launch_sdpa_merge_output(
        sycl::queue& queue, const SdpaNonmatrixRequest& request);
[[nodiscard]] sycl::event launch_sdpa_merge_output(
        sycl::queue& queue, const SdpaNonmatrixRequest& request,
        const sycl::event& dependency);

}  // namespace iom::sycl_detail

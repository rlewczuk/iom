#pragma once

#include <cstddef>

#include "device_internal.hpp"

// Backend-private boundary of the isolated AVX-512 BF16 SDPA QK stage. The
// entry point is defined in `avx512_bf16_sdpa_qk.cpp`, the single translation
// unit registered through `iom_add_avx512_bf16_source()`, and is therefore
// visible only to the baseline CPU workers that call it after
// `avx512_bf16_available()` has confirmed this build and this process may
// enter target code.
namespace iom {
namespace cpu_detail {

// Forms the FP32 scaled scores of one logical (query head, row) pair with the
// native AVX-512 BF16 pair-dot instructions and stores them into the same
// caller score workspace the scalar QK stage writes.
//
// `q_head_plane` and `k_head_plane` are the plane indices of the grouped query
// head and its KV head, `row` and the token are logical row coordinates of
// `request.q`/`request.k`, and only tokens in `[0, visible_limit)` are
// addressed. `scores[token]` receives `dot / scale`.
//
// Returns true only when every visible score was formed by executed
// VDPBF16PS accumulation and is finite. It returns false when the row is not
// numerically safe for the native association — an out-of-band or nonfinite
// Q/K code, a Q/K product envelope that the parent's ordered FP32 recurrence
// could overflow, or a formed score outside the FP32 finite range — and the
// caller must recompute the whole row with the compliant scalar worker, which
// overwrites every visible score the native path may already have written.
[[nodiscard]] bool avx512_bf16_sdpa_qk_row(
        const SdpaRequest& request, std::size_t q_head_plane,
        std::size_t k_head_plane, std::size_t row, std::size_t visible_limit,
        float scale, float* scores) noexcept;

}  // namespace cpu_detail
}  // namespace iom
#pragma once

#include <cstddef>

#include "iom/tensor.hpp"

// Backend-private entry point of the CPU's native BF16 linear dot worker.
// Nothing here is public API: the baseline `src/cpu/queue.cpp` worker reaches
// this function only after `iom::cpu_detail::avx512_bf16_available()` approved
// this host, and the definition lives in the separately ISA-targeted
// `src/cpu/avx512_bf16_linear.cpp` that only an `AVX_512_BF16_ENABLED` build
// registers through `iom_add_avx512_bf16_source()`. The signature carries only
// scalar types and the public `TensorSpec`, so the baseline caller keeps its
// own ABI and the target flags stay confined to the source that needs them.
namespace iom {
namespace cpu_detail {

// Projects one logical BF16 output element — exactly the coordinates the
// linear worker's own traversal walks — with the AVX-512 BF16 paired dot
// product, accumulating the BF16 products in FP32 and encoding the single
// destination value once with the named-format round-to-nearest-ties-to-even
// rules.
//
// `x_row` and `w_row` are the byte addresses of feature zero of the two
// participating logical rows, exactly as the caller's checked
// `logical_element_bits()` views compute them, and `features` is the reduction
// extent `I` those rows share. The caller owns that logical addressing; this
// function owns the pair packing, the accumulation, the tail/padding masking,
// the numerical safety decision, and the output store.
//
// Answers `false` without writing anything when the participating values are
// not numerically safe for the instruction: a subnormal or nonfinite BF16
// operand (the instruction replaces input denormals with zero and owns its own
// special-value rules), an operand magnitude range whose ordered FP32
// recurrence could overflow while a lane-wise reduction stays finite, or an
// FP32 accumulation that reached the hardware's flush-to-zero range. The caller
// must then project that element with the contract's scalar recurrence, inside
// the same accepted queued worker, and record the fallback path. `Native` work
// is recorded only for an element whose stored value is the one this function
// produced.
[[nodiscard]] bool avx512_bf16_linear_dot(
        const unsigned char* x_row, const unsigned char* w_row,
        std::size_t features, const TensorSpec& out_spec,
        unsigned char* out_base, std::size_t out_plane, std::size_t out_row,
        std::size_t out_column);

}  // namespace cpu_detail
}  // namespace iom
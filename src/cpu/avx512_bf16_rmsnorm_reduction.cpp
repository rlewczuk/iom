// Target-isolated AVX-512 BF16 RMSNorm row reduction.
//
// This translation unit is compiled with the AVX512F/AVX512BF16 target options
// and is the only definition site of `avx512_bf16_rmsnorm_reduce`. It performs
// the RMSNorm first pass's ordinary BF16 vector arithmetic: decode 16 logical
// BF16 features per 32-byte tile row into 16 FP32 lanes, square them, and
// reduce their FP32 sum. It emits no BF16 dot instruction: `VDPBF16PS` is a
// paired-BF16 dot product with its own denormal and accumulation policy, while
// the RMSNorm contract needs ordinary FP32 multiply and add in the accumulator
// domain, which is exactly what `_mm512_mul_ps`/`_mm512_add_ps` provide.
//
// The layout work is the shared standard 16x16 tiled mapping: `features` are
// logical columns, a 16-feature block of one row is one physically contiguous
// 32-byte tile row, and the next block of the same row starts one 512-byte tile
// later. Only logical lanes participate; the tile padding beyond a feature tail
// is masked out of both the finite-input test and the reduction, so poisoned
// padding cannot change a row's inverse. No global FTZ, DAZ, or fast-math state
// is touched: the vector multiply, add, and compare keep the thread's ordinary
// MXCSR, so FP32 subnormal products and sums stay gradual.

#include "avx512_bf16.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "transfer_helpers.hpp"

namespace iom {
namespace cpu_detail {

namespace {

// A BF16 tile row holds `TensorSpec::TILE` features in 32 contiguous bytes, and
// the next tile column of the same logical row begins one whole tile later.
// Both numbers follow from the standard tiled layout, so the vector block width
// is the tile width rather than a second constant to keep in step.
constexpr std::size_t kBlockFeatures = TensorSpec::TILE;
constexpr std::size_t kBf16TileBytes =
        TensorSpec::TILE * TensorSpec::TILE * sizeof(std::uint16_t);

static_assert(
        kBlockFeatures == 16,
        "one BF16 vector block is one 16-feature tile row of the standard "
        "tiled layout");

// Decodes one 32-byte tile row of BF16 codes into 16 FP32 lanes. Placing a
// BF16 code in the high half of an FP32 word is exact for every code: signed
// zeros, subnormals, infinities, and NaNs keep their class and their
// representable value, so the lanes below are ordinary FP32 operands.
[[nodiscard]] inline __m512 decode_block(const unsigned char* bytes) noexcept {
    const __m128i low =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes));
    const __m128i high =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes + 16));
    const __m512i codes =
            _mm512_inserti32x4(_mm512_castsi128_si512(low), high, 1);
    return _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_castsi512_si256(codes)), 16));
}

// Folds 16 FP32 lanes into one FP32 sum with a fixed pairwise tree. Every step
// is one ordinary FP32 addition, so the running sum keeps the accumulator
// domain's gradual behavior and the reduction order is the same for every call
// instead of depending on a compiler helper's internals.
[[nodiscard]] inline float sum_lanes(const __m512 values) noexcept {
    const __m512i bits = _mm512_castps_si512(values);
    __m128 total = _mm_add_ps(
            _mm_castsi128_ps(_mm512_castsi512_si128(bits)),
            _mm_castsi128_ps(_mm512_extracti32x4_epi32(bits, 1)));
    total = _mm_add_ps(
            total, _mm_castsi128_ps(_mm512_extracti32x4_epi32(bits, 2)));
    total = _mm_add_ps(
            total, _mm_castsi128_ps(_mm512_extracti32x4_epi32(bits, 3)));
    total = _mm_add_ps(total, _mm_movehl_ps(total, total));
    total = _mm_add_ss(total, _mm_shuffle_ps(total, total, 1));
    return _mm_cvtss_f32(total);
}

}  // namespace

bool avx512_bf16_rmsnorm_reduce(
        const unsigned char* x_base, const TensorSpec& spec,
        std::size_t plane, std::size_t row, std::size_t features,
        float epsilon, float& inverse) {
    // The first logical feature of the row in the owner's tiled storage. Every
    // later block of the same row is reached by stepping one tile further, so
    // the checked plane/row/column mapping runs once per row.
    const unsigned char* block =
            x_base + logical_element_bits(spec, plane, row, 0) / 8;
    const __m512 sign_mask = _mm512_set1_ps(-0.0F);
    const __m512 infinity =
            _mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512 accumulator = _mm512_setzero_ps();
    for (std::size_t column = 0; column < features;
         column += kBlockFeatures, block += kBf16TileBytes) {
        const std::size_t remaining = features - column;
        // A full block addresses 16 logical features; a feature tail addresses
        // only its own logical lanes and masks the tile padding beside them.
        const __mmask16 logical =
                remaining >= kBlockFeatures
                        ? static_cast<__mmask16>(0xFFFFU)
                        : static_cast<__mmask16>(
                                  (std::uint32_t{1} << remaining) - 1U);
        const __m512 values = decode_block(block);
        // A non-finite logical feature leaves the row to the caller's
        // sequential recurrence: the contract fixes exact special-value
        // classes for an infinite or NaN-poisoned row, and the scalar ordering
        // that already produces them is not re-derived from a lane-parallel
        // sum. Masked padding lanes are never tested, so a poisoned padding
        // code cannot decline the row.
        //
        // The magnitude test clears the sign bit with the AVX512F integer
        // bitwise operation rather than `_mm512_andnot_ps`, whose 512-bit form
        // belongs to AVX512DQ and is therefore outside this source's isolated
        // `-mavx512f -mavx512bf16` target options.
        const __m512 magnitude = _mm512_castsi512_ps(_mm512_andnot_si512(
                _mm512_castps_si512(sign_mask),
                _mm512_castps_si512(values)));
        const __mmask16 unordered =
                _mm512_mask_cmp_ps_mask(logical, values, values, _CMP_UNORD_Q);
        const __mmask16 unbounded =
                _mm512_mask_cmp_ps_mask(logical, magnitude, infinity, _CMP_GE_OQ);
        if ((unordered | unbounded) != 0) {
            return false;
        }
        accumulator = _mm512_add_ps(
                accumulator, _mm512_maskz_mul_ps(logical, values, values));
#if defined(IOM_AVX512_BF16_TESTING)
        // Executed work of this block: its FP32 squares and their running sum
        // have run in vector lanes. Selection alone is never recorded.
        avx512_bf16_test_record(
                Avx512Bf16Stage::RmsReduction, Avx512Bf16Path::Native);
#endif
    }
    // The contract's derivation stays in the accumulator domain with exactly
    // the operations the scalar recurrence uses, so the mean, the epsilon
    // addition, the square root, and the reciprocal round the same way for a
    // vectorized row as for its scalar neighbor.
    const float sum = sum_lanes(accumulator);
    const float mean = sum / static_cast<float>(features);
    const float shifted = mean + epsilon;
    const float root = std::sqrt(shifted);
    inverse = 1.0F / root;
    return true;
}

}  // namespace cpu_detail
}  // namespace iom
// Isolated AVX-512 BF16 linear dot worker.
//
// This translation unit is the only one compiled with the `-mavx512f
// -mavx512bf16` target options, and only an `AVX_512_BF16_ENABLED` build
// registers it (see `iom_add_avx512_bf16_source()` in `CMakeLists.txt`). It
// therefore contains the complete native BF16 arithmetic of one CPU linear
// output element: BF16 pair packing, the paired dot instruction, the tail and
// tile-padding masking that keeps padding out of every value, the
// exception-free numerical safety decision, and the single named-format output
// encode. Baseline translation units keep the portable command line and reach
// this code only through the runtime eligibility gate in `avx512_bf16.hpp`,
// which supplies the two rows' checked logical byte addresses.

#include "avx512_bf16_linear.hpp"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../shared/scalar_add.hpp"
#include "avx512_bf16.hpp"
#include "transfer_helpers.hpp"

namespace iom {
namespace cpu_detail {

namespace {

// Bytes of one 16x16 BF16 tile: 256 elements of two bytes each. The standard
// logical mapping stores a plane's tiles row-major inside the plane, so the
// tile that holds the next 16 features of a logical row is exactly one tile
// further, i.e. `kBf16TileBytes` past the current one. A 16-feature step inside
// a row therefore advances the row's first tile column by this fixed stride
// instead of by 32 bytes: a logical row is *not* physically contiguous across
// tile boundaries.
constexpr std::size_t kBf16TileBytes =
        TensorSpec::TILE * TensorSpec::TILE * (16 / 8);

// Features one `VDPBF16PS` operand pair consumes: a 512-bit BF16 register
// holds 32 values, paired into the 16 FP32 destination lanes.
constexpr std::size_t kBf16FeaturesPerDot = 2 * TensorSpec::TILE;

// Smallest accumulated magnitude this native path stores.
//
// `VDPBF16PS` flushes a subnormal FP32 result to zero, so a partial sum that
// passed through the FP32 subnormal range (below `2^-126`) can differ from the
// contract's gradual recurrence by up to one flush per accumulation step. That
// absolute error is irrelevant for an ordinary dot, and this bound keeps only
// the dots where it could matter on the compliant path: a final magnitude
// below `2^-96` lies within `2^30` of the FP32 flush boundary, so a recomputed
// element is cheap insurance for underflowing and cancellation-collapsed sums
// while every ordinary result keeps the native route.
constexpr float kBf16DotFlushGuard = 0x1p-96f;

// Element mask of the first `extent` BF16 values of one 32-byte chunk, for
// `extent` in `[0, 16]`. Zeroing the lanes at or beyond the logical tail keeps
// every tile-padding value out of both the dot and the safety test, so padding
// bytes are read (they are inside the tile that holds the row) but never used
// as values.
[[nodiscard]] inline __m256i bf16_tail_mask(std::size_t extent) noexcept {
    return _mm256_cmpgt_epi16(
            _mm256_set1_epi16(static_cast<short>(extent)),
            _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                              14, 15));
}

// Absolute BF16 codes of one loaded chunk: the sign bit removed, so the 16-bit
// lanes order by magnitude and the largest code is the largest magnitude.
[[nodiscard]] inline __m256i bf16_magnitudes(__m256i values) noexcept {
    return _mm256_and_si256(values, _mm256_set1_epi16(0x7FFF));
}

// Lanes whose BF16 magnitude the paired dot cannot represent faithfully for
// this operation:
//   * a subnormal value (exponent field zero with a nonzero significand), which
//     the instruction's input-denormal handling replaces with zero although the
//     contract's recurrence is gradual; and
//   * a nonfinite value (exponent field all ones), whose NaN and infinity rules
//     are the instruction's own rather than the named format's.
// Zero and every value with an exponent field in `1..254` is exactly
// representable in FP32, so those lanes are safe.
[[nodiscard]] inline __m256i bf16_unsafe_lanes(
        __m256i magnitudes) noexcept {
    const __m256i zero =
            _mm256_cmpeq_epi16(magnitudes, _mm256_setzero_si256());
    const __m256i subnormal = _mm256_cmpgt_epi16(
            _mm256_set1_epi16(0x0080), magnitudes);
    const __m256i nonfinite = _mm256_cmpgt_epi16(
            magnitudes, _mm256_set1_epi16(0x7F7F));
    return _mm256_andnot_si256(
            zero, _mm256_or_si256(subnormal, nonfinite));
}

// Largest absolute code of one accumulated block magnitude vector.
[[nodiscard]] inline std::uint32_t bf16_max_code(__m256i magnitudes) noexcept {
    __m128i folded = _mm_max_epu16(
            _mm256_castsi256_si128(magnitudes),
            _mm256_extracti128_si256(magnitudes, 1));
    folded = _mm_max_epu16(folded, _mm_srli_si128(folded, 8));
    folded = _mm_max_epu16(folded, _mm_srli_si128(folded, 4));
    folded = _mm_max_epu16(folded, _mm_srli_si128(folded, 2));
    return static_cast<std::uint32_t>(_mm_extract_epi16(folded, 0));
}

// Whether the contract's ordered FP32 recurrence could overflow to infinity for
// this dot even though the lane-wise native reduction might stay finite.
//
// Cancelling large finite products are the hazard: the reassociated native
// lanes can end small while the recurrence the contract mandates passed through
// the FP32 range, so a finite native result would disagree with the contract on
// the result class. This screen is conservative and needs no extra data pass of
// its own: every BF16 value with exponent field `e` satisfies
// `abs(value) < 2^(e-126)`, so the absolute sum of `features` products stays
// below `2^(e_x + e_w - 252 + log2(features))`, and an ordered partial sum can
// only exceed the FP32 maximum, just below `2^128`, when that bound reaches
// `2^128`. Exponent fields and `bit_width(features)`, an upper bound of
// `log2(features)`, are compared as integers, so the decision is exact, cheap,
// and independent of the accumulation order it guards.
[[nodiscard]] inline bool bf16_overflow_risk(
        std::uint32_t x_code, std::uint32_t w_code,
        std::size_t features) noexcept {
    const std::uint32_t bound_exponent =
            (x_code >> 7) + (w_code >> 7)
            + static_cast<std::uint32_t>(std::bit_width(features));
    return bound_exponent >= 128 + 252;
}

// The 16 FP32 lane partial sums of one accumulated dot reduced to the single
// dot product value, entirely in FP32. The lane split is an association order,
// which the contract permits for a native BF16 reduction; no partial product
// and no partial sum is ever rounded through BF16.
[[nodiscard]] inline float bf16_horizontal_sum(__m512 values) noexcept {
    const __m256 halves = _mm256_add_ps(
            _mm512_castps512_ps256(values),
            _mm256_castsi256_ps(_mm512_extracti64x4_epi64(
                    _mm512_castps_si512(values), 1)));
    const __m128 quarters = _mm_add_ps(
            _mm256_castps256_ps128(halves),
            _mm256_extractf128_ps(halves, 1));
    const __m128 pairs = _mm_add_ps(
            quarters, _mm_shuffle_ps(quarters, quarters, 0x4E));
    const __m128 total = _mm_add_ps(
            pairs, _mm_shuffle_ps(pairs, pairs, 0xB1));
    return _mm_cvtss_f32(total);
}

// Loads the 32-byte chunk of one logical row that starts at feature `first`,
// keeping only its first `extent` values.
//
// The chunk start is exact: a row's 16-bit values are contiguous from a
// tile-row start, so the 32 bytes read stay inside the tile that holds the row
// — including the tile's last row and a partial tail — and never cross the
// 512-byte tile boundary. The load is deliberately unaligned: the owner base is
// guaranteed only 32 bytes, and a chunk start is 32-byte aligned relative to
// that base, so neither an aligned load nor a 64-byte load is safe here.
[[nodiscard]] inline __m256i bf16_load_chunk(
        const unsigned char* chunk, std::size_t extent) noexcept {
    return _mm256_and_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(chunk)),
            bf16_tail_mask(extent));
}

}  // namespace

bool avx512_bf16_linear_dot(
        const unsigned char* x_row, const unsigned char* w_row,
        std::size_t features, const TensorSpec& out_spec,
        unsigned char* out_base, std::size_t out_plane, std::size_t out_row,
        std::size_t out_column) {
    __m512 accumulator = _mm512_setzero_ps();
    __m256i unsafe = _mm256_setzero_si256();
    // Largest absolute code of each operand over the whole reduction, used once
    // after the loop by the overflow screen below.
    __m256i x_magnitude = _mm256_setzero_si256();
    __m256i w_magnitude = _mm256_setzero_si256();
    for (std::size_t first = 0; first < features;
         first += kBf16FeaturesPerDot) {
        const unsigned char* const x_chunk =
                x_row + (first / TensorSpec::TILE) * kBf16TileBytes;
        const unsigned char* const w_chunk =
                w_row + (first / TensorSpec::TILE) * kBf16TileBytes;
        const std::size_t remaining = features - first;
        const std::size_t low_extent = remaining < TensorSpec::TILE
                ? remaining : TensorSpec::TILE;
        const __m256i x_low = bf16_load_chunk(x_chunk, low_extent);
        const __m256i w_low = bf16_load_chunk(w_chunk, low_extent);
        const __m256i x_low_magnitudes = bf16_magnitudes(x_low);
        const __m256i w_low_magnitudes = bf16_magnitudes(w_low);
        x_magnitude = _mm256_max_epu16(x_magnitude, x_low_magnitudes);
        w_magnitude = _mm256_max_epu16(w_magnitude, w_low_magnitudes);
        unsafe = _mm256_or_si256(
                unsafe, _mm256_or_si256(
                                bf16_unsafe_lanes(x_low_magnitudes),
                                bf16_unsafe_lanes(w_low_magnitudes)));
        // Both operands of a pair must describe the same feature positions, so
        // the two products of every destination lane are the two products of
        // one feature pair, and the 16 lanes together cover the block's
        // features exactly once whatever order the instruction pairs them in.
        //
        // The half of a short operand that lies beyond the block is defined as
        // zero rather than left as an unspecified register half: the
        // instruction consumes all 32 BF16 slots of both operands and the
        // reduction below sums all 16 FP32 lanes, so a stale upper half would
        // enter the stored result.
        __m512i left = _mm512_inserti64x4(
                _mm512_setzero_si512(), x_low, 0);
        __m512i right = _mm512_inserti64x4(
                _mm512_setzero_si512(), w_low, 0);
        if (remaining > TensorSpec::TILE) {
            // The block's second tile column exists precisely because the
            // logical row continues past feature 16, and it is one tile further
            // inside the same plane.
            const std::size_t high_remaining = remaining - TensorSpec::TILE;
            const std::size_t high_extent =
                    high_remaining < TensorSpec::TILE
                    ? high_remaining : TensorSpec::TILE;
            const __m256i x_high = bf16_load_chunk(
                    x_chunk + kBf16TileBytes, high_extent);
            const __m256i w_high = bf16_load_chunk(
                    w_chunk + kBf16TileBytes, high_extent);
            const __m256i x_high_magnitudes = bf16_magnitudes(x_high);
            const __m256i w_high_magnitudes = bf16_magnitudes(w_high);
            x_magnitude = _mm256_max_epu16(x_magnitude, x_high_magnitudes);
            w_magnitude = _mm256_max_epu16(w_magnitude, w_high_magnitudes);
            unsafe = _mm256_or_si256(
                    unsafe, _mm256_or_si256(
                                    bf16_unsafe_lanes(x_high_magnitudes),
                                    bf16_unsafe_lanes(w_high_magnitudes)));
            left = _mm512_inserti64x4(left, x_high, 1);
            right = _mm512_inserti64x4(right, w_high, 1);
        }
        accumulator = _mm512_dpbf16_ps(
                accumulator,
                reinterpret_cast<__m512bh>(left),
                reinterpret_cast<__m512bh>(right));
    }

    const float sum = bf16_horizontal_sum(accumulator);
    if (!_mm256_testz_si256(unsafe, unsafe)) {
        // A subnormal or nonfinite participating value: this element's dot
        // belongs to the contract's gradual scalar recurrence.
        return false;
    }
    if (bf16_overflow_risk(
                bf16_max_code(x_magnitude), bf16_max_code(w_magnitude),
                features)) {
        // Finite operands large enough for cancelling products: the ordered
        // FP32 recurrence can leave the finite range where this lane-wise
        // reduction does not, so the recurrence owns this element's result
        // class.
        return false;
    }
    if (!std::isfinite(sum) || std::fabs(sum) < kBf16DotFlushGuard) {
        // The FP32 accumulation reached the hardware's flush range or
        // overflowed, so its association is no longer numerically equivalent to
        // the contract's recurrence for this element.
        return false;
    }
#if defined(IOM_AVX512_BF16_TESTING)
    // Recorded only here: the vector dot above has executed and its result is
    // the value this call stores, so the counter can never report a dispatch
    // selection, an unsafe discarding, or a fallback element as native work.
    avx512_bf16_test_record(Avx512Bf16Stage::Linear, Avx512Bf16Path::Native);
#endif
    // One destination encode, in the same FP32 carrier domain and with the same
    // named-format rules the scalar recurrence uses: the accumulated FP32 sum is
    // exact in `long double`, so the shared codec contributes exactly the
    // contract's single round-to-nearest, ties-to-even rounding step.
    store_logical_element(
            out_base, out_spec, out_plane, out_row, out_column,
            detail::scalar_add_detail::encode_small(
                    static_cast<long double>(sum),
                    detail::scalar_add_detail::format(DataType::BF16)));
    return true;
}

}  // namespace cpu_detail
}  // namespace iom
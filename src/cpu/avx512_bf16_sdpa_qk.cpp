// Target-isolated BF16 SDPA QK stage.
//
// This translation unit is the only SDPA source compiled with the isolated
// `-mavx512f -mavx512bf16` target options: it is registered through
// `iom_add_avx512_bf16_source()` and reaches the vector pair-dot instructions
// only from `avx512_bf16_available()`-guarded callers, so an unsupported CPU
// or OS never enters this code. Every value it forms is produced by executed
// VDPBF16PS accumulation into gradual FP32 lanes; the surrounding scalar
// score classification, softmax, and PV stages are untouched.
//
// Nothing in this file is public API, and the pair-dot is a per-request
// decision: one call covers one logical (query head, row) pair of a request
// that common admission already validated.

#include "avx512_bf16_sdpa_qk.hpp"

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "avx512_bf16.hpp"
#include "transfer_helpers.hpp"

namespace iom::cpu_detail {
namespace {

// One standard tile span. The shared 16x16 tiled layout keeps a logical row's
// features contiguous inside a tile row, so a span of 16 consecutive features
// starting at a tile-aligned feature is 16 packed BF16 codes in 32 contiguous
// bytes.
constexpr std::size_t kSpanCodes = 16;
// One native pair-dot operand: 16 packed BF16 pairs, i.e. 32 logical features
// in a single 512-bit register.
constexpr std::size_t kOperandCodes = 2 * kSpanCodes;

// Native safety band, expressed on the BF16 exponent field. With the field's
// bias of 127 a field of 64 means |v| >= 2^-63 and a field of 189 means
// |v| < 2^63. Two in-band operands therefore multiply to an exponent field sum
// in [128, 378], i.e. an FP32 magnitude in [2^-126, 2^126]: every individual
// product is a normal FP32 value, so the pair-dot can neither flush a subnormal
// product (the contract's FTZ-sensitive case) nor overflow one. A zero exponent
// field is the signed zero/subnormal group that the matrix DAZ rule rewrites to
// a signed zero contribution, so it is always safe; every other field outside
// the band, including the 255 field of an infinity or NaN, is not.
constexpr unsigned kMinimumBandExponent = 64;
constexpr unsigned kMaximumBandExponent = 189;

// Exponent budget of the parent's ordered FP32 recurrence. The scalar worker
// sums the Q/K products in increasing feature order, so a row may only be formed
// natively when the whole product envelope is finite under *every* association:
//
//   sum_d |q_d * k_d| <= D * max|q| * max|k|
//                     < D * 2^(e_q - 126) * 2^(e_k - 126)
//
// where `e_q` and `e_k` are the largest participating BF16 exponent fields.
// That envelope stays at or below 2^126 exactly when
// `e_q + e_k + ceil(log2 D) <= 378`. Beyond the budget the ordered recurrence
// can reach an infinity that the pair lanes cancel away — head dimension 32,
// every Q value 2^62 and a K token of sixteen +2^62 followed by sixteen -2^62
// reaches 16 * 2^124 = 2^128 in feature order — so such a row is recomputed by
// the scalar worker and keeps the contract's formed-score classification.
constexpr unsigned kOverflowSafeExponentBudget = 378;

// Bit patterns that keep a code's sign while dropping its magnitude: the low
// 16-bit code of a 32-bit lane uses the upper mask, and the high code uses the
// lower mask, so the two rewrites never interfere.
constexpr int kLowCodeKeepMask = static_cast<int>(0xFFFF8000u);
constexpr int kHighCodeKeepMask = static_cast<int>(0x8000FFFFu);

// The BF16 pair operand of the native dot instruction. `__m512bh` has no
// integer-vector cast intrinsic of its own (`_mm512_castsi512_ph` is the FP16
// one), so the packed codes are reinterpreted with the bit-preserving
// conversion the contract's exact code preservation requires.
[[nodiscard]] __m512bh as_bf16_pairs(__m512i codes) noexcept {
    return std::bit_cast<__m512bh>(codes);
}

// One 512-bit operand after the matrix DAZ rule, with the largest BF16 exponent
// field it carries (zero for a zero or subnormal code, which the DAZ rule
// rewrites to a zero contribution) and the band verdict.
struct OperandProfile {
    __m512i codes;
    unsigned max_exponent;
    bool out_of_band;
};

// Applies the contract's matrix DAZ to 32 packed BF16 codes and reports the
// operand's largest exponent field together with whether any code is outside the
// native safety band.
//
// The DAZ rule is exactly an exponent-field property of a BF16 code: field zero
// covers a signed zero and every BF16 subnormal, and `D(x)` rewrites both to
// `copysign(+0, x)`, which is what keeping only the sign bit produces. The
// rewrite is therefore identical to `matrix_daz` on the codec-decoded value for
// every 16-bit pattern, including the sign of a stored zero.
[[nodiscard]] OperandProfile daz_and_classify(__m512i raw) noexcept {
    const __m512i exponent_mask = _mm512_set1_epi32(0x000000FF);
    const __m512i low_exponent =
            _mm512_and_si512(_mm512_srli_epi32(raw, 7), exponent_mask);
    const __m512i high_exponent =
            _mm512_and_si512(_mm512_srli_epi32(raw, 23), exponent_mask);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i minimum =
            _mm512_set1_epi32(static_cast<int>(kMinimumBandExponent));
    const __m512i maximum =
            _mm512_set1_epi32(static_cast<int>(kMaximumBandExponent));
    const auto outside = [&](const __m512i exponent) {
        const __mmask16 nonzero = _mm512_cmpneq_epi32_mask(exponent, zero);
        const __mmask16 below = _mm512_cmplt_epi32_mask(exponent, minimum);
        const __mmask16 above = _mm512_cmpgt_epi32_mask(exponent, maximum);
        return static_cast<__mmask16>(nonzero & (below | above));
    };
    const __mmask16 low_daz = _mm512_cmpeq_epi32_mask(low_exponent, zero);
    const __mmask16 high_daz = _mm512_cmpeq_epi32_mask(high_exponent, zero);
    __m512i codes = _mm512_mask_mov_epi32(
            raw, low_daz,
            _mm512_and_si512(raw, _mm512_set1_epi32(kLowCodeKeepMask)));
    codes = _mm512_mask_mov_epi32(
            codes, high_daz,
            _mm512_and_si512(raw, _mm512_set1_epi32(kHighCodeKeepMask)));
    return {
            codes,
            std::max(
                    _mm512_reduce_max_epu32(low_exponent),
                    _mm512_reduce_max_epu32(high_exponent)),
            static_cast<bool>(
                    static_cast<__mmask16>(
                            outside(low_exponent) | outside(high_exponent)))};
}

// Loads one standard tile span of `span` (1..16) consecutive logical BF16
// codes of one row, starting at the tile-aligned `column`.
[[nodiscard]] __m256i load_span(
        const SdpaView& view, std::span<const std::size_t> dimensions,
        std::size_t plane, std::size_t row, std::size_t column,
        std::size_t span) noexcept {
    const unsigned char* base = view.native_handle;
    const std::size_t byte_offset = cpu_detail::logical_element_bits(
            dimensions, DataType::BF16, plane, row, column) / 8;
    if (span == kSpanCodes) {
        return _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(base + byte_offset));
    }
    // A short span is read one logical code at a time so no tile padding code
    // of this row is ever loaded.
    alignas(32) std::array<std::uint16_t, kSpanCodes> staged{};
    for (std::size_t index = 0; index < span; ++index) {
        std::memcpy(
                &staged[index],
                base + byte_offset + index * sizeof(std::uint16_t),
                sizeof(std::uint16_t));
    }
    return _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(staged.data()));
}

// Loads exactly `count` (1..32) consecutive logical BF16 codes of one row,
// starting at a tile-aligned `column`, into a 512-bit operand. Unused lanes stay
// zero, so they contribute nothing to the pair-dot.
[[nodiscard]] __m512i load_codes(
        const SdpaView& view, std::size_t plane, std::size_t row,
        std::size_t column, std::size_t count) noexcept {
    const std::span<const std::size_t> dimensions(
            view.dimensions.data(), view.rank);
    const std::size_t first =
            count < kSpanCodes ? count : kSpanCodes;
    const __m256i low =
            load_span(view, dimensions, plane, row, column, first);
    __m256i high = _mm256_setzero_si256();
    if (count > kSpanCodes) {
        high = load_span(
                view, dimensions, plane, row, column + kSpanCodes,
                count - kSpanCodes);
    }
    return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
}

// Logical feature window of one pair-dot operand.
struct OperandWindow {
    std::size_t offset;
    std::size_t count;
};

[[nodiscard]] OperandWindow next_window(
        std::size_t offset, std::size_t features) noexcept {
    const std::size_t remaining = features - offset;
    return {offset,
            remaining < kOperandCodes ? remaining : kOperandCodes};
}

// `ceil(log2(features))`, the exponent carry of the product-count bound.
[[nodiscard]] unsigned ceiling_log2(std::size_t features) noexcept {
    return static_cast<unsigned>(std::bit_width(features - 1));
}

// Whether the parent's ordered recurrence could overflow its FP32 accumulator
// for the product envelope of one visible token.
[[nodiscard]] bool recurrence_may_overflow(
        unsigned q_exponent, unsigned k_exponent,
        unsigned dimension_carry) noexcept {
    return q_exponent + k_exponent + dimension_carry
            > kOverflowSafeExponentBudget;
}

}  // namespace

bool avx512_bf16_sdpa_qk_row(
        const SdpaRequest& request, std::size_t q_head_plane,
        std::size_t k_head_plane, std::size_t row, std::size_t visible_limit,
        float scale, float* scores) noexcept {
    const std::size_t features = request.D;
    if (features == 0 || visible_limit == 0) return false;
    const unsigned dimension_carry = ceiling_log2(features);

    for (std::size_t token = 0; token < visible_limit; ++token) {
        __m512 accumulator = _mm512_setzero_ps();
        unsigned q_exponent = 0;
        unsigned k_exponent = 0;
        bool out_of_band = false;
        for (std::size_t offset = 0; offset < features;) {
            const OperandWindow window = next_window(offset, features);
            const OperandProfile q = daz_and_classify(load_codes(
                    request.q, q_head_plane, row, window.offset,
                    window.count));
            const OperandProfile k = daz_and_classify(load_codes(
                    request.k, k_head_plane, token, window.offset,
                    window.count));
            q_exponent = std::max(q_exponent, q.max_exponent);
            k_exponent = std::max(k_exponent, k.max_exponent);
            out_of_band = out_of_band || q.out_of_band || k.out_of_band;
            accumulator = _mm512_dpbf16_ps(
                    accumulator, as_bf16_pairs(q.codes),
                    as_bf16_pairs(k.codes));
            offset += window.count;
        }
        // A nonfinite or out-of-band Q/K code is never native work: the row
        // falls back before its score is stored, and the compliant scalar
        // worker recomputes every visible score.
        if (out_of_band) return false;
        // The whole product envelope must stay inside the FP32 finite range
        // under every association before the native pair lanes may stand in for
        // the parent's ordered recurrence.
        if (recurrence_may_overflow(
                    q_exponent, k_exponent, dimension_carry)) {
            return false;
        }

        // The one FP32 division that forms the scaled score, in the same
        // gradual FP32 domain as the scalar stage's `dot / scale`.
        const float formed = _mm512_reduce_add_ps(accumulator) / scale;
        // A row whose formed score leaves the finite FP32 range keeps the
        // existing scalar classification and is never native work.
        if (!std::isfinite(formed)) return false;
        scores[token] = formed;
    }

    // Recorded after the pair-dot loop above executed for this row, never for
    // selecting the kernel.
#if defined(IOM_AVX512_BF16_TESTING)
    avx512_bf16_test_record(Avx512Bf16Stage::SdpaQk, Avx512Bf16Path::Native);
#endif
    return true;
}

}  // namespace iom::cpu_detail
// Isolated AVX-512 BF16 target source for binary ADD and SUB.
//
// This translation unit is compiled only through iom_add_avx512_bf16_source(),
// which applies -mavx512f/-mavx512bf16 to this file alone, and is entered only
// from an accepted CPU queue task that already confirmed
// iom::cpu_detail::avx512_bf16_available(). It owns no scalar numerical
// semantics of its own: every lane whose result the vector arithmetic cannot
// reproduce exactly is handed to the baseline codec routine the caller passes
// in, and the calling route selects the floating-point control word the
// contract's arithmetic needs through the two helpers below.

#include "avx512_bf16.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace iom {
namespace cpu_detail {

namespace {

using Lanes = std::uint16_t[TensorSpec::TILE];

// Loads one operand span as 16 BF16 lanes, reading logical leaves only. A
// broadcast operand repeats the row's single logical leaf; a span that fills
// its tile row is loaded straight from storage; a shorter logical tail is
// gathered lane by lane into a lane buffer first, because the slots beside a
// logical row inside its tile are padding.
[[nodiscard]] __m256i load_span(
        const std::uint16_t* source, bool broadcast, unsigned lanes) noexcept {
    if (broadcast) {
        return _mm256_set1_epi16(static_cast<short>(source[0]));
    }
    if (lanes == TensorSpec::TILE) {
        return _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(source));
    }
    Lanes lanes_buffer{};
    for (unsigned lane = 0; lane < lanes; ++lane) {
        lanes_buffer[lane] = source[lane];
    }
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lanes_buffer));
}

// Exact BF16 decode of 16 lanes into FP32. A BF16 pattern is exactly the high
// half of the FP32 pattern of the same value -- finite, subnormal, infinity, or
// zero -- so widening the shift back is lossless and independent of the
// current denormal mode; nothing is flushed or rounded here. NaN patterns
// decode to FP32 NaNs, which the exceptional-lane mask removes before any
// arithmetic.
[[nodiscard]] __m512 decode(__m256i bits) noexcept {
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(bits), 16));
}

// Encodes 16 FP32 lanes to BF16 with exactly one round-to-nearest-even
// conversion, matching the scalar codec's `encode` for every value that is not
// a NaN. Adding 0x7FFF plus the first discarded mantissa bit rounds the 16
// discarded bits to nearest-even; a carry propagates into the exponent field,
// so an overflowing rounding becomes the infinity encoding exactly as the
// codec's overflow path does, and gradual subnormal results round within the
// subnormal encoding because the FP32 exponent field reaches zero with the
// same value the codec's subnormal branch computes.
[[nodiscard]] __m256i encode_rne(__m512 value) noexcept {
    const __m512i bits = _mm512_castps_si512(value);
    const __m512i rounded = _mm512_add_epi32(
            _mm512_add_epi32(bits, _mm512_set1_epi32(0x7FFF)),
            _mm512_and_si512(
                    _mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1)));
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
}

// Stores exactly `lanes` leaves. A span that fills its tile row is stored
// directly; a shorter logical tail is materialized in a lane buffer first so
// the padding beside it stays untouched.
void store_span(std::uint16_t* out, unsigned lanes, __m256i packed) noexcept {
    if (lanes == TensorSpec::TILE) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), packed);
        return;
    }
    Lanes lanes_buffer;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes_buffer), packed);
    std::memcpy(out, lanes_buffer, lanes * sizeof(std::uint16_t));
}

}  // namespace

std::uint32_t avx512_bf16_mxcsr() noexcept {
    return _mm_getcsr();
}

void avx512_bf16_set_mxcsr(std::uint32_t control) noexcept {
    _mm_setcsr(control);
}

void avx512_bf16_binary_span(
        bool subtract, const std::uint16_t* lhs, const std::uint16_t* rhs,
        std::uint16_t* out, unsigned lanes, bool lhs_broadcast,
        bool rhs_broadcast, Avx512Bf16ScalarBinary scalar) noexcept {
    const __m512 x = decode(load_span(lhs, lhs_broadcast, lanes));
    const __m512 y = decode(load_span(rhs, rhs_broadcast, lanes));

    // The lanes whose result the scalar codec defines rather than plain IEEE
    // FP32 arithmetic: any NaN operand, and the infinity pair that is an
    // invalid operation for this operation -- opposite signs for ADD, like
    // signs for SUB. Both decode to the codec's canonical quiet NaN, whose
    // class and sign come from the codec rather than from the hardware's
    // invalid-operation default, so those lanes stay with the codec.
    const auto infinity = [](float sign) {
        return _mm512_set1_ps(sign * std::numeric_limits<float>::infinity());
    };
    const __m512 positive_infinity = infinity(1.0f);
    const __m512 negative_infinity = infinity(-1.0f);
    const __mmask16 x_positive_infinity =
            _mm512_cmp_ps_mask(x, positive_infinity, _CMP_EQ_OQ);
    const __mmask16 x_negative_infinity =
            _mm512_cmp_ps_mask(x, negative_infinity, _CMP_EQ_OQ);
    const __mmask16 y_positive_infinity =
            _mm512_cmp_ps_mask(y, positive_infinity, _CMP_EQ_OQ);
    const __mmask16 y_negative_infinity =
            _mm512_cmp_ps_mask(y, negative_infinity, _CMP_EQ_OQ);
    const __mmask16 like_signs = static_cast<__mmask16>(
            (x_positive_infinity & y_positive_infinity)
            | (x_negative_infinity & y_negative_infinity));
    const __mmask16 unlike_signs = static_cast<__mmask16>(
            (x_positive_infinity & y_negative_infinity)
            | (x_negative_infinity & y_positive_infinity));
    const __mmask16 invalid_infinity_pair =
            subtract ? like_signs : unlike_signs;
    const __mmask16 span = static_cast<__mmask16>((1u << lanes) - 1u);
    const __mmask16 codec_lanes = static_cast<__mmask16>(
            (_mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q)
             | _mm512_cmp_ps_mask(y, y, _CMP_UNORD_Q)
             | invalid_infinity_pair)
            & span);
    const __mmask16 vector_lanes = static_cast<__mmask16>(span & ~codec_lanes);

    // Both operand values are in registers before this store: a lane that
    // aliases the output reads its own inputs, not a partial result.
    const __m512 result = subtract
            ? _mm512_mask_sub_ps(x, vector_lanes, x, y)
            : _mm512_mask_add_ps(x, vector_lanes, x, y);

#if defined(IOM_AVX512_BF16_TESTING)
    const Avx512Bf16Stage stage = subtract ? Avx512Bf16Stage::BinarySub
                                           : Avx512Bf16Stage::BinaryAdd;
    // Native work is recorded inside the executed vector loop, after the
    // arithmetic above ran for at least one lane of this span.
    if (vector_lanes != 0) {
        avx512_bf16_test_record(stage, Avx512Bf16Path::Native);
    }
#endif

    const __m256i packed = encode_rne(result);
    if (codec_lanes == 0) {
        store_span(out, lanes, packed);
        return;
    }

#if defined(IOM_AVX512_BF16_TESTING)
    // The codec runs for exactly the lanes that need it.
    avx512_bf16_test_record(stage, Avx512Bf16Path::Fallback);
#endif

    Lanes results;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), packed);
    for (unsigned lane = 0; lane < lanes; ++lane) {
        if ((codec_lanes & (1u << lane)) == 0) {
            continue;
        }
        results[lane] = static_cast<std::uint16_t>(scalar(
                DataType::BF16,
                lhs_broadcast ? lhs[0] : lhs[lane],
                rhs_broadcast ? rhs[0] : rhs[lane]));
    }
    std::memcpy(out, results, lanes * sizeof(std::uint16_t));
}

}  // namespace cpu_detail
}  // namespace iom
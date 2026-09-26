// AVX-512 BF16 SiLU element arithmetic.
//
// This translation unit is one of the isolated AVX-512 BF16 sources: the
// project registers it through `iom_add_avx512_bf16_source()` so that only this
// file receives the AVX512F/AVX512BF16 target options, and it is compiled only
// when `AVX_512_BF16_ENABLED` is ON. `CpuQueue::silu_elements` reaches the row
// entry point below after its own per-request eligibility decision, so no
// baseline translation unit contains target code and an ineligible request
// never enters this file.
//
// The kernel keeps the operation's contract exactly:
//
//  * BF16 storage is the standard 16x16 tile layout, so one complete logical
//    tile row is 16 contiguous BF16 leaves in 32 bytes: exactly one 256-bit
//    vector. The byte offset of a tile row is a multiple of 32 from the storage
//    base, but the base is only guaranteed 32-byte aligned for allocator-owned
//    storage, so every access uses the unaligned load/store form.
//  * BF16 to FP32 decode is the exact widening shift. The FP32 encoding of a
//    BF16 value is that value's encoding shifted into the high half, so sign,
//    exponent, subnormal, infinity, NaN, and signed zero all decode without
//    rounding.
//  * The FP32 branches are the scalar ones. A nonnegative value uses
//    `x / (1 + exp(-x))`; a negative value keeps the underflow-safe
//    `((x * t) * t) / (1 + t * t)` with `t = exp(x / 2)` in the written
//    left-associated order, so representable negative tails survive. `exp()`
//    stays the shared scalar source; this file vectorizes the arithmetic, not
//    the transcendental. Neither FTZ nor fast-math behavior is enabled, and no
//    branch is reassociated.
//  * Signed zeros need no special case: both branches preserve the input sign,
//    and `-0 / 2` is `-0`.
//  * Nonfinite values are the scalar evaluator's contract. A NaN maps to NaN,
//    and `-infinity` must become negative zero, which neither branch produces,
//    so a group holding one evaluates that element with
//    `src/shared/scalar_silu.hpp`. A feature tail that forms no complete tile
//    row stays with the same scalar evaluator, so it is bit-identical to the
//    portable path.
//  * Exactly one effective RNE encode per logical element happens at the
//    store: `_mm512_cvtneps_pbh` is the round-to-nearest-even FP32-to-BF16
//    conversion, and the target format is never an intermediate. That
//    instruction treats an FP32 subnormal input as zero, while every BF16
//    subnormal result comes from exactly that range, so a lane whose vector
//    result is a nonzero subnormal takes the shared codec's RNE encode of that
//    same FP32 value instead; the flushed conversion result is never
//    observable.
//  * Only logical elements of the selected plane, run, and feature are read or
//    written. Tile padding is never addressed, and tail elements go through the
//    shared checked logical-element helpers.

#include "avx512_bf16.hpp"
#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <span>

#include "../shared/scalar_silu.hpp"

// The target options this translation unit is compiled with (AVX512F implies
// FMA) also let the compiler contract a multiplication and the following
// addition into one rounded operation, and GCC enables contraction by default.
// The operation's contract fixes the FP32 evaluation order of both stable
// branches, and its tolerance is one BF16 ULP, so this file keeps every one of
// its own and the shared evaluator's FP32 operations separately rounded; the
// pragma is GCC-specific, and a compiler that ignores it still only changes the
// denominator's last FP32 rounding, which no BF16 output class or sign can
// observe.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace iom {
namespace cpu_detail {

namespace {

// One complete tile row of the standard layout: `TensorSpec::TILE` logical
// BF16 leaves, which is one 256-bit vector and one FP32 conversion pair. A
// group of logical features is contiguous in storage exactly when it starts at
// a multiple of the tile width, which is how the row loop below selects its
// groups.
constexpr std::size_t kGroupFeatures = TensorSpec::TILE;
constexpr std::size_t kLeafBytes = sizeof(std::uint16_t);

// The BF16 leaf's established FP32 evaluation domain: the portable worker
// instantiates the shared scalar evaluator with exactly this carrier for every
// leaf below F64, so the scalar fallback below is the same evaluator on the
// same domain.
using SiluCarrier = CpuCarrierTraits<float>;

// The named-format codec of that same evaluation domain. Its encoding rules are
// the portable path's, so a value this kernel cannot convert itself is encoded
// exactly as the scalar path encodes it.
using SiluCodec = detail::scalar_binary_codec_detail::Codec<SiluCarrier>;
using SiluFormat = detail::scalar_binary_codec_detail::Format;

// Test-build-only stage observation. A production build compiles this call away
// together with the seam it names.
void record_silu(Avx512Bf16Path path) noexcept {
#if defined(IOM_AVX512_BF16_TESTING)
    avx512_bf16_test_record(Avx512Bf16Stage::Silu, path);
#else
    (void)path;
#endif
}

// Exact BF16 to FP32 widening of one whole group.
[[nodiscard]] __m512 decode_group(const unsigned char* bytes) noexcept {
    const __m256i packed =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bytes));
    const __m512i widened = _mm512_cvtepu16_epi32(packed);
    return _mm512_castsi512_ps(_mm512_slli_epi32(widened, 16));
}

// One raw BF16 leaf from its two little-endian bytes, which is exactly what
// `load_bits` reads for an aligned 16-bit logical leaf.
[[nodiscard]] std::uint64_t raw_leaf(const unsigned char* bytes) noexcept {
    return static_cast<std::uint64_t>(bytes[0])
            | (static_cast<std::uint64_t>(bytes[1]) << 8);
}

void store_raw_leaf(unsigned char* bytes, std::uint64_t raw) noexcept {
    bytes[0] = static_cast<unsigned char>(raw & 0xFFu);
    bytes[1] = static_cast<unsigned char>((raw >> 8) & 0xFFu);
}

// One complete 16-feature group of a logical feature row.
//
// The vector part is the FP32 arithmetic of both stable branches plus the one
// RNE BF16 output encode. The exponentials come from the shared scalar source
// one lane at a time, and the lane's `exp` argument is the branch's own
// argument: `-x` for a nonnegative finite value and `x * 0.5` for a negative
// one. A NaN or an infinity is the scalar evaluator's contract, so such a lane
// keeps its vector result only until the scalar value replaces it.
void silu_group(const unsigned char* x_bytes, unsigned char* y_bytes) noexcept {
    const __m512 x = decode_group(x_bytes);
    const __mmask16 negative =
            _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    const __m512 magnitude = _mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(x), _mm512_set1_epi32(0x7FFFFFFF)));
    const __mmask16 special = static_cast<__mmask16>(
            _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q)
            | _mm512_cmp_ps_mask(
                      magnitude,
                      _mm512_set1_ps(
                              std::numeric_limits<float>::infinity()),
                      _CMP_EQ_OQ));

    const __m512 argument = _mm512_mask_blend_ps(
            negative, _mm512_sub_ps(_mm512_setzero_ps(), x),
            _mm512_mul_ps(x, _mm512_set1_ps(0.5f)));
    alignas(64) float arguments[kGroupFeatures];
    alignas(64) float weights[kGroupFeatures];
    _mm512_store_ps(arguments, argument);
    for (std::size_t lane = 0; lane < kGroupFeatures; ++lane) {
        weights[lane] = ((special >> lane) & 1u) != 0
                ? 0.0f : std::exp(arguments[lane]);
    }

    const __m512 weight = _mm512_load_ps(weights);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 nonnegative_result =
            _mm512_div_ps(x, _mm512_add_ps(one, weight));
    // The negative branch keeps the contract's written left-associated
    // `(x * t) * t` numerator and `1 + t * t` denominator.
    const __m512 negative_result = _mm512_div_ps(
            _mm512_mul_ps(_mm512_mul_ps(x, weight), weight),
            _mm512_add_ps(one, _mm512_mul_ps(weight, weight)));
    const __m512 result =
            _mm512_mask_blend_ps(negative, nonnegative_result, negative_result);
    // One RNE encode per logical element, written straight into the group's
    // 32-byte tile row. `_mm512_cvtneps_pbh` yields the packed BF16 vector
    // type, which reinterprets bit-for-bit as the integer vector the store
    // takes.
    _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(y_bytes),
            std::bit_cast<__m256i>(_mm512_cvtneps_pbh(result)));
    const __m512i result_magnitude = _mm512_and_si512(
            _mm512_castps_si512(result),
            _mm512_set1_epi32(0x7FFFFFFF));
    const __mmask16 subnormal_result = static_cast<__mmask16>(
            _mm512_cmplt_epu32_mask(
                    result_magnitude, _mm512_set1_epi32(0x00800000))
            & _mm512_cmpneq_epi32_mask(
                      result_magnitude, _mm512_setzero_si512()));
    record_silu(Avx512Bf16Path::Native);
    if (subnormal_result != 0) {
        // `_mm512_cvtneps_pbh` treats an FP32 subnormal input as zero, but the
        // whole BF16 subnormal range is exactly where the named format's own
        // rounding decides the result, so those lanes take the shared codec's
        // single RNE encode of the value the vector arithmetic produced. The
        // flushed conversion result above is never observable.
        alignas(64) float results[kGroupFeatures];
        _mm512_store_ps(results, result);
        const SiluFormat format = SiluCodec::format(DataType::BF16);
        for (std::size_t lane = 0; lane < kGroupFeatures; ++lane) {
            if (((subnormal_result >> lane) & 1u) == 0) continue;
            store_raw_leaf(
                    y_bytes + lane * kLeafBytes,
                    SiluCodec::encode(results[lane], format));
        }
    }

    for (std::size_t lane = 0; lane < kGroupFeatures; ++lane) {
        if (((special >> lane) & 1u) == 0) continue;
        const unsigned char* x_leaf = x_bytes + lane * kLeafBytes;
        store_raw_leaf(
                y_bytes + lane * kLeafBytes,
                detail::scalar_silu_detail::scalar_silu<SiluCarrier>(
                        DataType::BF16, raw_leaf(x_leaf)));
        record_silu(Avx512Bf16Path::Fallback);
    }
}

}  // namespace

// One logical feature row of one run in one selected plane pair. Complete
// 16-feature groups of the row are one tile row each and run through the vector
// kernel; the non-multiple tail keeps the shared scalar evaluator, the shared
// checked layout helpers, and the single RNE encode the portable worker uses,
// so a tail element is bit-identical to the unaccelerated path and no tile
// padding bit is addressed.
void avx512_bf16_silu_row(
        const unsigned char* x_base, unsigned char* y_base,
        std::span<const std::size_t> dimensions, std::size_t x_plane,
        std::size_t y_plane, std::size_t run, std::size_t features) {
    std::size_t feature = 0;
    for (; feature + kGroupFeatures <= features; feature += kGroupFeatures) {
        const std::size_t x_bit = logical_element_bits(
                dimensions, DataType::BF16, x_plane, run, feature);
        const std::size_t y_bit = logical_element_bits(
                dimensions, DataType::BF16, y_plane, run, feature);
        silu_group(x_base + x_bit / 8, y_base + y_bit / 8);
    }
    for (; feature < features; ++feature) {
        store_logical_element(
                y_base, dimensions, DataType::BF16, y_plane, run, feature,
                detail::scalar_silu_detail::scalar_silu<SiluCarrier>(
                        DataType::BF16,
                        load_logical_element(
                                x_base, dimensions, DataType::BF16, x_plane,
                                run, feature)));
        record_silu(Avx512Bf16Path::Fallback);
    }
}

}  // namespace cpu_detail
}  // namespace iom
#include "avx512_bf16_rope.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "avx512_bf16.hpp"
#include "transfer_helpers.hpp"

namespace iom {
namespace cpu_detail {

namespace {

// One chunk of pairs fills exactly one lane mask.
constexpr std::uint32_t kAllPairLanes =
        (std::uint32_t{1} << kAvx512Bf16RopePairs) - 1u;

[[nodiscard]] std::uint32_t pair_lane_mask(std::size_t count) noexcept {
    return count >= kAvx512Bf16RopePairs
            ? kAllPairLanes
            : (std::uint32_t{1} << count) - 1u;
}

// Byte offset of one logical BF16 element inside the standard 16x16 tiled
// owner storage the worker's scalar path addresses. Every 16-bit leaf offset
// is byte aligned, so the shared bit offset divided by eight is exact.
[[nodiscard]] std::size_t element_byte_offset(
        std::span<const std::size_t> dimensions, std::size_t plane,
        std::size_t row, std::size_t column) noexcept {
    return logical_element_bits(
                   dimensions, DataType::BF16, plane, row, column)
            / 8;
}

// A chunk that starts on a tile boundary and fills the tile row is one
// contiguous 32-byte span of the standard layout; every other chunk keeps
// only logical elements in view and is addressed element by element.
[[nodiscard]] bool fills_tile_row(
        std::size_t column, std::size_t count) noexcept {
    return count == TensorSpec::TILE && column % TensorSpec::TILE == 0;
}

// Loads logical BF16 elements `[column, column + count)` of one row of one
// plane into `values`, which the caller has sized for a whole chunk. Lanes
// outside `count` keep their previous content, so the caller clears them
// before the vector load.
void load_pair_run(
        const unsigned char* base, std::span<const std::size_t> dimensions,
        std::size_t plane, std::size_t row, std::size_t column,
        std::size_t count, std::uint16_t* values) noexcept {
    if (fills_tile_row(column, count)) {
        std::memcpy(values,
                    base + element_byte_offset(
                                   dimensions, plane, row, column),
                    TensorSpec::TILE * sizeof(std::uint16_t));
        return;
    }
    for (std::size_t lane = 0; lane < count; ++lane) {
        std::memcpy(values + lane,
                    base + element_byte_offset(
                                   dimensions, plane, row, column + lane),
                    sizeof(std::uint16_t));
    }
}

// Writes the encoded BF16 elements of the lanes set in `stored_lanes` to the
// same logical destination coordinates. Lanes the caller declined are never
// addressed, so their destination elements keep the caller's own content.
void store_pair_run(
        unsigned char* base, std::span<const std::size_t> dimensions,
        std::size_t plane, std::size_t row, std::size_t column,
        const std::uint16_t* values, std::size_t count,
        std::uint32_t stored_lanes) noexcept {
    if (fills_tile_row(column, count) && stored_lanes == kAllPairLanes) {
        std::memcpy(base + element_byte_offset(
                                  dimensions, plane, row, column),
                    values, TensorSpec::TILE * sizeof(std::uint16_t));
        return;
    }
    for (std::size_t lane = 0; lane < count; ++lane) {
        if (((stored_lanes >> lane) & 1u) == 0) continue;
        std::memcpy(base + element_byte_offset(
                                  dimensions, plane, row, column + lane),
                    values + lane, sizeof(std::uint16_t));
    }
}

// BF16 is the upper half of an FP32 value, so zero-extending the raw sixteen
// bits and shifting them into the exponent position decodes every sign,
// subnormal, zero, and finite value exactly; NaN and infinity keep a payload
// and a class this kernel only declines.
[[nodiscard]] __m512 decode_pair_bits(__m256i raw) noexcept {
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

// Lanes whose BF16 encoding must stay with the codec: a NaN (whose payload
// the codec canonicalizes), an infinity or an out-of-range magnitude, and a
// subnormal magnitude are exactly what round-to-nearest-even cannot be assumed
// to share between the packing conversion and the codec. Zero is not
// subnormal here: both paths encode a signed zero as its own sign.
[[nodiscard]] __mmask16 unsafe_encode_lanes(__m512 value) noexcept {
    const __m512 magnitude = _mm512_abs_ps(value);
    const __mmask16 unordered =
            _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
    const __mmask16 too_large = _mm512_cmp_ps_mask(
            magnitude, _mm512_set1_ps(std::numeric_limits<float>::max()),
            _CMP_GT_OQ);
    const __mmask16 subnormal = _mm512_cmp_ps_mask(
            magnitude, _mm512_set1_ps(std::numeric_limits<float>::min()),
            _CMP_LT_OQ)
            & _mm512_cmp_ps_mask(
                    value, _mm512_setzero_ps(), _CMP_NEQ_OQ);
    return static_cast<__mmask16>(
            unordered | too_large | subnormal);
}

// The vector counterpart of the scalar pair's volatile intermediates. GCC and
// Clang may contract a product and a following sum into a fused multiply-add,
// which would replace the contract's separately rounded operations, so each
// product is held in its own register across an empty asm. Other compilers
// keep intrinsics separate on their own.
void keep_products_separate(__m512& value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __asm__("" : "+v"(value));
#else
    (void)value;
#endif
}

#if defined(IOM_AVX512_BF16_TESTING)
// Records one native pair for every lane the vector arithmetic encoded. The
// call sits after the vector work, so a decline or a dispatch decision can
// never be counted as executed arithmetic.
void record_native_pairs(std::size_t count, std::uint32_t stored_lanes) noexcept {
    for (std::size_t lane = 0; lane < count; ++lane) {
        if (((stored_lanes >> lane) & 1u) != 0) {
            avx512_bf16_test_record(
                    Avx512Bf16Stage::Rope, Avx512Bf16Path::Native);
        }
    }
}
#endif

}  // namespace

std::uint32_t avx512_bf16_rope_pairs(
        const unsigned char* x_base, unsigned char* out_base,
        std::span<const std::size_t> x_dimensions,
        std::span<const std::size_t> out_dimensions, std::size_t x_plane,
        std::size_t out_plane, std::size_t row, std::size_t first_pair,
        std::size_t count, const float* sine, const float* cosine) noexcept {
    const std::size_t half = x_dimensions.back() / 2;
    const std::uint32_t active = pair_lane_mask(count);

    // The chunk's scratch is cleared first: lanes outside `count` then carry
    // defined zeros into the vector operations and are masked out below.
    alignas(64) std::uint16_t first_bits[kAvx512Bf16RopePairs] = {};
    alignas(64) std::uint16_t second_bits[kAvx512Bf16RopePairs] = {};
    alignas(64) float sine_values[kAvx512Bf16RopePairs] = {};
    alignas(64) float cosine_values[kAvx512Bf16RopePairs] = {};
    for (std::size_t lane = 0; lane < count; ++lane) {
        sine_values[lane] = sine[lane];
        cosine_values[lane] = cosine[lane];
    }
    load_pair_run(
            x_base, x_dimensions, x_plane, row, first_pair, count, first_bits);
    load_pair_run(
            x_base, x_dimensions, x_plane, row, first_pair + half, count,
            second_bits);

    const __m512 first = decode_pair_bits(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(first_bits)));
    const __m512 second = decode_pair_bits(
            _mm256_load_si256(reinterpret_cast<const __m256i*>(second_bits)));
    const __m512 sine_vector = _mm512_load_ps(sine_values);
    const __m512 cosine_vector = _mm512_load_ps(cosine_values);

    __m512 first_cosine_product = _mm512_mul_ps(first, cosine_vector);
    __m512 second_sine_product = _mm512_mul_ps(second, sine_vector);
    __m512 second_cosine_product = _mm512_mul_ps(second, cosine_vector);
    __m512 first_sine_product = _mm512_mul_ps(first, sine_vector);
    keep_products_separate(first_cosine_product);
    keep_products_separate(second_sine_product);
    keep_products_separate(second_cosine_product);
    keep_products_separate(first_sine_product);
    const __m512 first_output =
            _mm512_sub_ps(first_cosine_product, second_sine_product);
    const __m512 second_output =
            _mm512_add_ps(second_cosine_product, first_sine_product);

    const __mmask16 declined = static_cast<__mmask16>(
            (unsafe_encode_lanes(first) | unsafe_encode_lanes(second)
             | unsafe_encode_lanes(first_output)
             | unsafe_encode_lanes(second_output))
            & static_cast<__mmask16>(active));
    const std::uint32_t declined_lanes =
            static_cast<std::uint32_t>(declined) & active;
    const std::uint32_t stored = active & ~declined_lanes;
    if (stored != 0) {
        // The packing conversion takes the lower half of its result from its
        // second operand and the upper half from its first, so the first-half
        // columns of the chunk are the second argument and the second-half
        // columns its first. Each half is then stored at its own logical
        // column run, in lane order.
        const __m512bh encoded =
                _mm512_cvtne2ps_pbh(second_output, first_output);
        alignas(64) std::uint16_t packed[2 * kAvx512Bf16RopePairs] = {};
        std::memcpy(packed, &encoded, sizeof(packed));
        store_pair_run(
                out_base, out_dimensions, out_plane, row, first_pair, packed,
                count, stored);
        store_pair_run(
                out_base, out_dimensions, out_plane, row, first_pair + half,
                packed + kAvx512Bf16RopePairs, count, stored);
    }

#if defined(IOM_AVX512_BF16_TESTING)
    record_native_pairs(count, stored);
#endif
    return declined_lanes;
}

}  // namespace cpu_detail
}  // namespace iom
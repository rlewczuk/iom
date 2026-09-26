// Native AVX-512 BF16 multiplication of one standard-tile row.
//
// The CPU queue reaches this unit only for the BF16 leaf of `mul`, only from an
// accepted queued worker, and only after `avx512_bf16_available()` reported an
// eligible CPU and OS vector state. Its exported entry is declared in
// `avx512_bf16.hpp` and carries a scalar ABI, so no baseline caller and no
// public type ever names a target vector type; this file is the only place
// where the AVX512F/AVX512BF16 target options apply.
//
// One standard-tile row is exactly 16 BF16 leaves in 16 contiguous owner slots,
// which is one 256-bit vector, so a row is processed without any packing or
// realignment. The row's arithmetic is FP32: both operands decode exactly into
// FP32, one multiply produces the exact product, and one RNE conversion encodes
// the row. The scalar codec (src/shared/scalar_add.hpp) remains the authority
// for every lane this unit cannot reproduce exactly, and the caller is the one
// that runs it.
//
// Contract fit. BF16 and FP32 share an exponent range and an eight-bit exponent
// field, so placing a leaf's bits in the high half of an FP32 lane decodes the
// leaf exactly: finite values including signed zero and subnormals, infinity,
// and NaN keep their class, sign, and payload, and the operation is integer
// work that consults no rounding mode, denormal mode, or MXCSR state. A product
// of two BF16 significands carries at most 16 significant bits, which fits the
// FP32 significand, so the FP32 product is exact whenever it is a normal FP32
// value -- the same value the codec's extended-domain product holds. The
// FP32-to-BF16 conversion instruction rounds to nearest even but treats
// denormal inputs as zero and flushes denormal results, so only lanes whose
// product is a normal FP32 value below 2^127 are committed: their BF16 result
// is a normal finite leaf, the conversion is the contract's single RNE encode,
// and no lane reaches the denormal behavior. Every other lane -- NaN,
// infinity, `0 * infinity`, signed zero, a product that underflows to subnormal
// or zero, and a product that could round to infinity -- is left to the scalar
// codec, which owns gradual underflow, the special-value classes, and the
// signed-zero result.
#include "avx512_bf16.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace iom {
namespace cpu_detail {

namespace {

// One standard-tile row: the 16 logical features of a row hold 16 contiguous
// BF16 leaves, which is exactly one 256-bit vector of leaves and one 512-bit
// vector of decoded FP32 lanes.
constexpr std::size_t kRowLanes = 16;

// Bit width of one BF16 leaf, which is also the shift that moves a leaf into
// the FP32 exponent and significand fields.
constexpr unsigned kLeafBits = 16;

// FP32 exponent-field bounds of a lane this unit may commit. The field is at
// least one, so the product is a normal FP32 value: no denormal reaches the
// conversion, and the codec's gradual underflow stays with the codec because a
// smaller product is excluded. The field is at most 253, so the product stays
// below 2^127: it cannot round to infinity, and its BF16 result is a normal
// finite leaf rather than a flushed denormal.
constexpr unsigned kMinNormalExponent = 1;
constexpr unsigned kMaxNormalExponent = 253;
constexpr std::uint32_t kExponentFieldMask = 0xFFu;

static_assert(
        sizeof(__m256i) * 8 == kRowLanes * kLeafBits,
        "one standard-tile row must be exactly one 256-bit BF16 vector");

// Decode one row of BF16 leaves to FP32 values by placing each leaf's bits in
// the high half of its FP32 lane. The mapping is exact for every leaf pattern
// and involves no floating-point instruction, so no rounding, denormal, or
// MXCSR behavior participates in the decode.
[[nodiscard]] __m512 decode_row(__m256i leaves) noexcept {
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(leaves), kLeafBits));
}

// The lanes whose product this unit may commit. Testing the unsigned
// difference `exponent - kMinNormalExponent <= kMaxNormalExponent -
// kMinNormalExponent` is the field range [kMinNormalExponent,
// kMaxNormalExponent] in one subtract and one unsigned compare, and it rejects
// every lane whose MUL semantics the conversion would not preserve.
[[nodiscard]] __mmask16 usable_lanes(__m512 product) noexcept {
    const __m512i exponent = _mm512_and_si512(
            _mm512_srli_epi32(_mm512_castps_si512(product), 23),
            _mm512_set1_epi32(kExponentFieldMask));
    return _mm512_cmp_epu32_mask(
            _mm512_sub_epi32(
                    exponent, _mm512_set1_epi32(kMinNormalExponent)),
            _mm512_set1_epi32(kMaxNormalExponent - kMinNormalExponent),
            _MM_CMPINT_LE);
}

// The 16 BF16 leaves at a bit offset of the shared little-endian tiled layout,
// and the inverse store. Leaves and whole rows are moved as raw bits: the
// addressing is byte-granular, so a row is copied from a 32-byte-aligned owner
// whose base may be misaligned for a wider access, exactly like the codec's
// per-leaf loads and stores.
[[nodiscard]] __m256i load_row(
        const unsigned char* base, std::size_t bit) noexcept {
    __m256i leaves{};
    std::memcpy(&leaves, base + bit / 8, sizeof(leaves));
    return leaves;
}

// One leaf at an arbitrary bit offset. Only a column-broadcast operand uses it:
// the leaf's own slot is the operand's single feature, and every lane of the
// row reuses its value.
[[nodiscard]] std::uint16_t load_broadcast_leaf(
        const unsigned char* base, std::size_t bit) noexcept {
    std::uint16_t leaf = 0;
    std::memcpy(&leaf, base + bit / 8, sizeof(leaf));
    return leaf;
}

}  // namespace

bool avx512_bf16_binary_mul_row(
        const unsigned char* lhs, std::size_t lhs_bit,
        const unsigned char* rhs, std::size_t rhs_bit,
        unsigned char* out, std::size_t out_bit,
        bool lhs_broadcast, bool rhs_broadcast) noexcept {
    const __m256i lhs_leaves = lhs_broadcast
            ? _mm256_set1_epi16(static_cast<short>(
                      load_broadcast_leaf(lhs, lhs_bit)))
            : load_row(lhs, lhs_bit);
    const __m256i rhs_leaves = rhs_broadcast
            ? _mm256_set1_epi16(static_cast<short>(
                      load_broadcast_leaf(rhs, rhs_bit)))
            : load_row(rhs, rhs_bit);

    const __m512 product = _mm512_mul_ps(
            decode_row(lhs_leaves), decode_row(rhs_leaves));
    if (usable_lanes(product) != static_cast<__mmask16>(0xFFFFu)) {
        // At least one lane needs the codec's semantics, and the caller
        // recomputes the whole row. Both operands were read and nothing was
        // written, so a declined row leaves even an exact in-place alias intact.
        return false;
    }

    // Fixed round-to-nearest-even, applied once to a row of exact normal
    // products, is the contract's single encode of the mathematical result.
    const __m256bh encoded = _mm512_cvtneps_pbh(product);
    std::memcpy(out + out_bit / 8, &encoded, sizeof(encoded));

#if defined(IOM_AVX512_BF16_TESTING)
    // Committed SIMD work, recorded after the vector arithmetic ran: the row's
    // encoded result is what the caller observes.
    avx512_bf16_test_record(
            Avx512Bf16Stage::BinaryMul, Avx512Bf16Path::Native);
#endif
    return true;
}

}  // namespace cpu_detail
}  // namespace iom
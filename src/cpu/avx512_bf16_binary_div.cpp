// Isolated AVX-512 BF16 target source of the CPU binary DIV route: one full
// tile run of sixteen BF16 lanes per call.
//
// The kernel owns exactly one tile-column group of one output row. It loads
// both operands into registers before the destination store, classifies every
// lane from those operands alone, and only then divides and converts. Lanes the
// contract keeps off the vector path are excluded *before* the divide and the
// conversion, their operands replaced by one, and their value is produced by
// the same scalar codec the portable CPU path uses, inside the accepted queued
// worker. That boundary is what keeps the operation's contract intact:
//
//  - BF16-to-FP32 decode is a bit-exact exponent shift, so signed zeros,
//    subnormals, infinities and NaN payloads reach the kernel unchanged.
//  - A lane crosses the vector divide only when both operands are normal and
//    the difference of their FP32 exponent fields is inside a fixed window that
//    keeps the quotient several binades inside the normal FP32 range. Such a
//    lane can raise neither the invalid (0/0, infinity/infinity), the
//    divide-by-zero, the denormal-operand, the overflow nor the underflow SIMD
//    exception; excluded lanes are never divided or converted at all, so an
//    inherited unmasked FP-exception environment cannot signal on them.
//  - The single RNE conversion is VCVTNEPS2BF16, which treats denormal inputs
//    as zero and flushes denormal results, so it is only ever fed lanes that
//    cannot be either: a normal FP32 value is at least 2^-126, which is the
//    smallest normal BF16 value, and rounds up within the normal BF16 range.
//  - The scalar codec evaluates the quotient in the host `long double` carrier
//    and encodes it once. For a kept lane the exact quotient is (a / b) * 2^e
//    with 8-bit a and b, so its significand is never closer than
//    1 / (512 * 255) to a BF16 rounding boundary - orders of magnitude further
//    than an FP32 rounding error (2^-24) can move it. The FP32 quotient
//    therefore RNE-encodes to the same BF16 bits as the exact one, and a
//    quotient inside the exponent window cannot round differently at the BF16
//    overflow, underflow or exponent-carry boundaries either.
//  - Every exceptional and every underflowing lane keeps the scalar reference
//    as its authority: NaN (including the exact quiet-NaN encoding sign), 0/0,
//    infinity/infinity, zero quotients, infinities, overflow, and any lane
//    with a subnormal operand, a subnormal result or an out-of-window exponent
//    difference.
//
// The entry point has a scalar ABI: no vector type crosses it, so the baseline
// queue may call it after establishing eligibility, and only a build that
// compiles these isolated sources defines it at all.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "../shared/scalar_add.hpp"
#include "avx512_bf16.hpp"
#include "transfer_helpers.hpp"

namespace iom {
namespace cpu_detail {

namespace {

// A standard tile row holds sixteen BF16 leaves, so one run is one 256-bit
// operand pair and one 512-bit quotient.
constexpr std::size_t kRunLanes = 16;
constexpr std::size_t kLaneBits = 16;
constexpr __mmask16 kAllLanes = 0xFFFF;

// Difference of the two FP32 exponent fields inside which the quotient of two
// normal operands is provably normal. With significands in [1, 2) the quotient
// lies in (2^(d - 1), 2^(d + 1)) for a field difference of d, whose biases
// cancel, so |d| <= 120 keeps it at least five binades inside the normal FP32
// range - away from the underflow, the subnormal and the overflow boundary.
constexpr int kExponentWindow = 120;

// Lanes that may cross the vector divide and the BF16 conversion: both operands
// normal, so no zero, subnormal, infinity or NaN reaches the arithmetic, and an
// exponent difference inside the window above, so the quotient cannot leave the
// normal FP32 range. Everything else is excluded before the arithmetic runs.
[[nodiscard]] __mmask16 safe_lanes(
        __m512i lhs_bits, __m512i rhs_bits) noexcept {
    const __m512i exponent_mask = _mm512_set1_epi32(0xFF);
    const __m512i lhs_exponent = _mm512_and_si512(
            _mm512_srli_epi32(lhs_bits, 23), exponent_mask);
    const __m512i rhs_exponent = _mm512_and_si512(
            _mm512_srli_epi32(rhs_bits, 23), exponent_mask);
    // A normal FP32 exponent field is 1..254.
    const auto normal = [](__m512i exponent) {
        return _mm512_cmple_epu32_mask(
                _mm512_sub_epi32(exponent, _mm512_set1_epi32(1)),
                _mm512_set1_epi32(253));
    };
    const __m512i difference =
            _mm512_abs_epi32(_mm512_sub_epi32(lhs_exponent, rhs_exponent));
    return normal(lhs_exponent) & normal(rhs_exponent)
            & _mm512_cmple_epu32_mask(
                      difference, _mm512_set1_epi32(kExponentWindow));
}

}  // namespace

void avx512_bf16_div_run(
        const unsigned char* lhs, std::size_t lhs_bit,
        const unsigned char* rhs, std::size_t rhs_bit,
        unsigned char* out, std::size_t out_bit) noexcept {
    const __m256i lhs_packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(lhs + lhs_bit / 8));
    const __m256i rhs_packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(rhs + rhs_bit / 8));
    // BF16 is the high half of the matching FP32 encoding, so the decode is a
    // zero-extend plus one shift, and both signed zeros and subnormals survive
    // exactly.
    const __m512i lhs_bits = _mm512_slli_epi32(
            _mm512_cvtepu16_epi32(lhs_packed), 16);
    const __m512i rhs_bits = _mm512_slli_epi32(
            _mm512_cvtepu16_epi32(rhs_packed), 16);
    // Every lane's fate is decided from the two operands alone, before any
    // arithmetic runs: an excluded lane is not divided and not converted, so
    // the invalid, divide-by-zero, denormal-operand, overflow and underflow
    // SIMD exceptions cannot be raised for it even under an unmasked
    // environment. Both of its operands are replaced by one, which makes the
    // vector work below exact on that lane; its value is then written by the
    // scalar codec instead.
    const __mmask16 safe = safe_lanes(lhs_bits, rhs_bits);
    const __m512 one = _mm512_set1_ps(1.0F);
    const __m512 quotient = _mm512_div_ps(
            _mm512_mask_blend_ps(safe, one, _mm512_castsi512_ps(lhs_bits)),
            _mm512_mask_blend_ps(safe, one, _mm512_castsi512_ps(rhs_bits)));
    // Both operands are already in registers, so the store cannot disturb
    // them: the exact alias case stays defined.
    _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(out + out_bit / 8),
            reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(quotient)));
    if (safe != kAllLanes) {
        alignas(32) std::uint16_t lhs_lane[kRunLanes];
        alignas(32) std::uint16_t rhs_lane[kRunLanes];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lhs_lane), lhs_packed);
        _mm256_store_si256(reinterpret_cast<__m256i*>(rhs_lane), rhs_packed);
        for (std::size_t lane = 0; lane < kRunLanes; ++lane) {
            if ((safe & (static_cast<__mmask16>(1) << lane)) != 0) {
                continue;
            }
            store_bits(
                    out, out_bit + lane * kLaneBits, kLaneBits,
                    detail::scalar_div(
                            DataType::BF16,
                            static_cast<std::uint64_t>(lhs_lane[lane]),
                            static_cast<std::uint64_t>(rhs_lane[lane])));
        }
    }
#if defined(IOM_AVX512_BF16_TESTING)
    // The stage reports the work it produced, never the selection: a run with a
    // vector-produced lane is vector work, and a lane the scalar reference had
    // to produce is fallback work.
    if (safe != 0) {
        avx512_bf16_test_record(
                Avx512Bf16Stage::BinaryDiv, Avx512Bf16Path::Native);
    }
    if (safe != kAllLanes) {
        avx512_bf16_test_record(
                Avx512Bf16Stage::BinaryDiv, Avx512Bf16Path::Fallback);
    }
#endif
}

}  // namespace cpu_detail
}  // namespace iom
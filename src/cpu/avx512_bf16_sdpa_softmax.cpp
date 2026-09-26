// Isolated AVX-512 BF16 SDPA softmax/probability stage.
//
// Only this translation unit carries the AVX512F/AVX512BF16 target options, and
// a portable build does not contain it at all. Baseline `src/cpu/sdpa.cpp`
// reaches it through the detector `avx512_bf16_available()` and keeps its own
// scalar probability loop for an ineligible process or a build without the
// isolated sources.
//
// The stage owns the ordinary stable-softmax row's second half: the FP32
// normalization `exp(score - max) / sum` for the visible tokens, the single
// BF16 round-to-nearest conversion of each probability, and the stores into the
// row's existing caller-owned BF16 probability workspace segment. The scalar QK
// producer and the scalar PV consumer around it are untouched, the FP32
// exponential and its increasing-order sum stay in the baseline row loop, and
// the special-score rows never enter this stage.

#include "avx512_bf16.hpp"

#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../shared/scalar_binary_codec.hpp"

#include <immintrin.h>

namespace iom {
namespace cpu_detail {
namespace {

using Bf16Codec = detail::scalar_binary_codec_detail::Codec<
        CpuCarrierTraits<float>>;

// One ZMM of FP32 lanes: this many probabilities are normalized and converted
// per vector step.
constexpr std::size_t kVectorLanes = 16;

// Smallest positive FP32 normal, 2^-126. The native conversion treats a
// smaller input as zero and flushes a smaller result, while the BF16 codec
// converts an FP32 subnormal with gradual underflow -- including a subnormal
// that rounds up to the smallest BF16 normal. A lane below this bound is
// therefore not contract-compatible with the native conversion and takes the
// compliant scalar path instead.
constexpr float kSmallestNormal = 0x1p-126f;

// Whether the native conversion of every lane selected by `lanes` is the same
// single BF16 RNE conversion the codec produces: zero and normal FP32 values
// convert identically, while an FP32 subnormal or a nonfinite probability does
// not.
[[nodiscard]] bool native_conversion_compatible(
        __m512 probability, __mmask16 lanes) noexcept {
    const __m512 magnitude = _mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(probability),
            _mm512_set1_epi32(0x7fffffff)));
    const __mmask16 zero = _mm512_cmp_ps_mask(
            magnitude, _mm512_setzero_ps(), _CMP_EQ_OQ);
    const __mmask16 subnormal = _mm512_cmp_ps_mask(
            magnitude, _mm512_set1_ps(kSmallestNormal), _CMP_LT_OQ) & ~zero;
    const __mmask16 unordered = _mm512_cmp_ps_mask(
            probability, probability, _CMP_UNORD_Q);
    return ((subnormal | unordered) & lanes) == 0;
}

// Store one vector of converted probabilities as little-endian 16-bit BF16
// values, the same byte layout cpu_detail::store_bits() writes. `lanes` is the
// chunk lane count, at most kVectorLanes: a full chunk writes its 32 bytes
// directly, while a row tail copies only the 1..kVectorLanes-1 converted lanes
// it owns, so no byte outside the row's probability segment is touched.
void store_native_probabilities(
        unsigned char* destination, __m512 probability,
        std::size_t lanes) noexcept {
    assert(lanes <= kVectorLanes);
    const __m256bh converted = _mm512_cvtneps_pbh(probability);
    if (lanes == kVectorLanes) {
        _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(destination),
                reinterpret_cast<const __m256i&>(converted));
        return;
    }
    alignas(32) std::uint16_t bits[kVectorLanes];
    std::memcpy(bits, &converted, sizeof(bits));
    std::memcpy(destination, bits, lanes * sizeof(std::uint16_t));
}

}  // namespace

void avx512_bf16_sdpa_softmax_probabilities(
        const float* exponentials, std::size_t visible, float sum,
        unsigned char* workspace, std::size_t probability_base,
        std::size_t row_index) noexcept {
    unsigned char* const row = workspace + probability_base
            + row_index * sizeof(std::uint16_t);
    const __m512 divisor = _mm512_set1_ps(sum);

    // The whole visible range is normalized and converted by the vector path
    // unless a lane is not contract-compatible with the native conversion; that
    // row is then completed by the compliant scalar path below, which
    // reproduces the codec's gradual-underflow conversion for every lane.
    bool fallback = false;
    std::size_t token = 0;
    while (token < visible) {
        const std::size_t remaining = visible - token;
        // The number of lanes this chunk stores: a full chunk is all
        // kVectorLanes lanes, and only a row's final chunk stores the lanes it
        // actually owns.
        const std::size_t chunk = remaining >= kVectorLanes
                ? kVectorLanes
                : remaining;
        const __mmask16 lanes = remaining >= kVectorLanes
                ? static_cast<__mmask16>(0xffffu)
                : static_cast<__mmask16>(
                          (std::uint32_t{1} << remaining) - 1u);
        const __m512 numerator = _mm512_maskz_loadu_ps(
                lanes, exponentials + token);
        const __m512 probability = _mm512_div_ps(numerator, divisor);
        if (!native_conversion_compatible(probability, lanes)) {
            fallback = true;
            break;
        }
        store_native_probabilities(
                row + token * sizeof(std::uint16_t), probability, chunk);
#if defined(IOM_AVX512_BF16_TESTING)
        // Recorded inside the arithmetic loop, after this chunk's FP32 division
        // and its one-time BF16 conversion have executed.
        avx512_bf16_test_record(
                Avx512Bf16Stage::SdpaSoftmax, Avx512Bf16Path::Native);
#endif
        token += kVectorLanes;
    }

    if (fallback) {
        for (std::size_t index = 0; index < visible; ++index) {
            const volatile float normalized = exponentials[index] / sum;
            cpu_detail::store_bits(
                    row + index * sizeof(std::uint16_t), 0, 16,
                    Bf16Codec::encode(
                            normalized, Bf16Codec::format(DataType::BF16)));
        }
#if defined(IOM_AVX512_BF16_TESTING)
        avx512_bf16_test_record(
                Avx512Bf16Stage::SdpaSoftmax, Avx512Bf16Path::Fallback);
#endif
    }
}

}  // namespace cpu_detail
}  // namespace iom
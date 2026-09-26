// Target-specific BF16 RMSNorm normalization and store pass. This translation
// unit is compiled with the isolated AVX512F/AVX512BF16 options and is entered
// only after `avx512_bf16_available()` accepted the host for this request,
// because its entry point and every vector instruction below assume the
// AVX-512 BF16 vector state the detector proved.
//
// The second RMSNorm pass is the operation's FP32 domain: the row inverse comes
// from the (unchanged) scalar reduction, every normalization and scale product
// is rounded in FP32, and each logical output is encoded exactly once. The
// vector loop covers a whole 16-feature group of the standard 16x16 tiled
// layout, whose sixteen 16-bit BF16 codes are one contiguous 32-byte tile row,
// and the scalar codec still owns the feature tail and every lane whose value
// the ISA conversion cannot encode under the contract.

#include "avx512_bf16.hpp"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../shared/scalar_add.hpp"
#include "transfer_helpers.hpp"

namespace iom {
namespace cpu_detail {

namespace {

// One logical feature group of a row is exactly one tile row of the standard
// 16x16 tiled layout for the 16-bit BF16 leaf: `TensorSpec::TILE` contiguous
// little-endian BF16 codes whose bit offset is `TensorSpec::TILE * 16` bits,
// i.e. a multiple of the group size, so an aligned 256-bit access is in bounds
// and covers exactly the sixteen logical features. Consecutive groups of one
// row are the tile rows of consecutive tile columns, one whole tile apart.
constexpr std::size_t kGroupFeatures = TensorSpec::TILE;
constexpr std::size_t kGroupBytes = kGroupFeatures * 2;
constexpr std::size_t kTileBytes =
        TensorSpec::TILE * TensorSpec::TILE * 2;
constexpr std::size_t kGroupMask = (std::size_t{1} << kGroupFeatures) - 1;

// The BF16 leaf encoding, as the operation's codec names it.
using StoreFormat = detail::scalar_add_detail::Format;

[[nodiscard]] StoreFormat store_format() noexcept {
    return detail::scalar_add_detail::format(DataType::BF16);
}

// The mask of the ISA conversion's unsupported input class: a zero exponent
// field covers both signed zeros and FP32 subnormals, which is exactly the
// range whose contract result is a BF16 signed zero or subnormal -- the
// conversion flushes both -- and the unordered comparison covers NaN results,
// whose stored form is canonical positive and therefore not the instruction's
// own NaN encoding. Every other lane is a normal finite value or an infinity,
// for which the fixed round-to-nearest-even conversion reproduces the
// contract's single encode exactly.
[[nodiscard]] __mmask16 scalar_lane_mask(
        __m512 result, __m512i exponent_field) noexcept {
    const __mmask16 zero_exponent = static_cast<__mmask16>(
            ~_mm512_test_epi32_mask(
                    _mm512_castps_si512(result), exponent_field));
    return zero_exponent
            | _mm512_cmp_ps_mask(result, result, _CMP_UNORD_Q);
}

// BF16 codes of one feature group widened to FP32. A BF16 value is the upper
// half of its FP32 encoding, so the widening is a 16-bit left shift: every
// finite BF16 value -- signed zeros, subnormals, and the largest finite
// magnitude -- and every infinity/NaN pattern becomes the exact FP32 value
// with the same sign and payload. No rounding and no denormal flush occurs.
[[nodiscard]] __m512 widen_group(
        const unsigned char* base, std::size_t byte_offset) noexcept {
    const __m256i codes = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(base + byte_offset));
    return _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(codes), 16));
}

// The compliant scalar encode of one already-computed FP32 result, exactly as
// the portable row pass evaluates it: a NaN result is stored in its canonical
// positive form and the value is encoded once with BF16 round-to-nearest-even.
[[nodiscard]] std::uint16_t encode_scalar_result(
        float value, const StoreFormat& format) noexcept {
    if (std::isnan(value)) {
        value = std::fabs(value);
    }
    return static_cast<std::uint16_t>(
            detail::scalar_add_detail::encode_small(
                    static_cast<long double>(value), format));
}

// One logical feature decoded through the portable codec, used by the scalar
// feature tail so its arithmetic matches the portable row pass step for step.
[[nodiscard]] float decode_scalar_feature(
        const unsigned char* base, const TensorSpec& spec,
        const StoreFormat& format, std::size_t plane, std::size_t row,
        std::size_t column) {
    return static_cast<float>(detail::scalar_add_detail::decode_small(
            load_logical_element(base, spec, plane, row, column), format));
}

void store_scalar_feature(
        unsigned char* base, const TensorSpec& spec, const StoreFormat& format,
        std::size_t plane, std::size_t row, std::size_t column,
        float value) {
    store_logical_element(
            base, spec, plane, row, column,
            encode_scalar_result(value, format));
}

#if defined(IOM_AVX512_BF16_TESTING)
// Executed-work observation of one stored logical feature: the SIMD group whose
// vector arithmetic produced a stored lane, and the compliant scalar store path
// of one feature. Production builds contain neither call.
void record_native_store() noexcept {
    avx512_bf16_test_record(
            Avx512Bf16Stage::RmsStore, Avx512Bf16Path::Native);
}
void record_scalar_store() noexcept {
    avx512_bf16_test_record(
            Avx512Bf16Stage::RmsStore, Avx512Bf16Path::Fallback);
}
#else
void record_native_store() noexcept {}
void record_scalar_store() noexcept {}
#endif

}  // namespace

void avx512_bf16_rmsnorm_store(
        unsigned char* out_base, const TensorSpec& out_spec,
        std::size_t out_plane, const unsigned char* x_base,
        const TensorSpec& x_spec, std::size_t x_plane,
        const unsigned char* scale_base, const TensorSpec& scale_spec,
        std::size_t scale_plane, std::size_t row, std::size_t features,
        float inverse) {
    const StoreFormat format = store_format();
    const __m512 inverse_vector = _mm512_set1_ps(inverse);
    const __m512i exponent_field = _mm512_set1_epi32(0x7F800000);
    const std::size_t groups = features / kGroupFeatures;
    // The first group of the row fixes the tiled addressing for the whole row:
    // every group is one tile further along, so the checked slot arithmetic
    // runs once per operand instead of once per feature, and the tile stride
    // (not an assumption that the row is contiguous) carries the loop.
    std::size_t out_bytes =
            logical_element_bits(out_spec, out_plane, row, 0) / 8;
    std::size_t x_bytes =
            logical_element_bits(x_spec, x_plane, row, 0) / 8;
    std::size_t scale_bytes =
            logical_element_bits(scale_spec, scale_plane, 0, 0) / 8;
    for (std::size_t group = 0; group < groups; ++group) {
        const __m512 value = widen_group(x_base, x_bytes);
        const __m512 scale = widen_group(scale_base, scale_bytes);
        // The operation's expression in its accumulator domain: the
        // normalization and the scale multiplication are two separately
        // rounded FP32 products, exactly as the portable pass evaluates them.
        const __m512 result =
                _mm512_mul_ps(_mm512_mul_ps(value, inverse_vector), scale);
        const __mmask16 scalar_lanes =
                scalar_lane_mask(result, exponent_field);
        alignas(kGroupBytes) unsigned char encoded[kGroupBytes];
        const __m256bh packed = _mm512_cvtneps_pbh(result);
        std::memcpy(encoded, &packed, kGroupBytes);
        if (scalar_lanes != 0) {
            alignas(kGroupFeatures * sizeof(float)) float lanes[kGroupFeatures];
            _mm512_store_ps(lanes, result);
            unsigned int pending = scalar_lanes;
            while (pending != 0) {
                const auto lane =
                        static_cast<unsigned>(std::countr_zero(pending));
                pending &= pending - 1;
                const std::uint16_t codes =
                        encode_scalar_result(lanes[lane], format);
                std::memcpy(encoded + lane * 2, &codes, 2);
                record_scalar_store();
            }
        }
        _mm256_store_si256(
                reinterpret_cast<__m256i*>(out_base + out_bytes),
                _mm256_load_si256(
                        reinterpret_cast<const __m256i*>(encoded)));
        // The group is stored before its vector work is reported: a lane the
        // ISA conversion produced is the evidence, and a group whose every lane
        // needed the scalar codec is reported as the fallback it was.
        if ((scalar_lanes & kGroupMask) != kGroupMask) {
            record_native_store();
        }
        out_bytes += kTileBytes;
        x_bytes += kTileBytes;
        scale_bytes += kTileBytes;
    }
    // The logical feature tail is not a full tile row, so it is stored by the
    // portable codec's own decode, arithmetic, and single encode.
    for (std::size_t column = groups * kGroupFeatures; column < features;
         ++column) {
        const float value = decode_scalar_feature(
                x_base, x_spec, format, x_plane, row, column);
        const float scale = decode_scalar_feature(
                scale_base, scale_spec, format, scale_plane, 0, column);
        const float normalized = value * inverse;
        store_scalar_feature(
                out_base, out_spec, format, out_plane, row, column,
                normalized * scale);
        record_scalar_store();
    }
}

}  // namespace cpu_detail
}  // namespace iom

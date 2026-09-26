// Isolated AVX-512 BF16 probability-value stage of the CPU SDPA worker.
//
// Only this translation unit is compiled with the AVX512F/AVX512BF16 target
// options, so the paired-dot instructions stay behind the runtime gate in
// `avx512_bf16_dispatch.cpp`. The stage consumes the same BF16 probability
// workspace and logical V view the scalar PV recurrence does, applies the
// contract's matrix DAZ function to both matrix operands, accumulates each
// destination lane in FP32 with VDPBF16PS, and encodes each result once with
// the shared BF16 round-to-nearest-even codec.
//
// Intel documents VDPBF16PS as treating input denormals as zero and flushing
// output denormals while ignoring MXCSR (SDM Vol. 2C, VDPBF16PS). Operand DAZ
// therefore matches the SDPA contract by itself, but the output flush does not:
// a representable subnormal result must survive. The stage accepts a row only
// while every nonzero product it will form is at least 2^-100 in magnitude, so
// no product, partial sum, or final result of that row can be subnormal and no
// flush can change a representable value; anything below that scale - and every
// request whose operands the gate cannot bound - is handed back to the caller's
// gradual scalar recurrence.

#include "avx512_bf16.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "device_internal.hpp"
#include "transfer_helpers.hpp"

#include "../shared/scalar_binary_codec.hpp"

namespace iom {
namespace cpu_detail {
namespace {

using Bf16Codec = detail::scalar_binary_codec_detail::Codec<
        CpuCarrierTraits<float>>;

// One 512-bit dot product accumulates into sixteen FP32 lanes, and the standard
// storage tile is sixteen elements wide, so a feature block, a destination
// vector, and a V tile row all have the same width.
constexpr std::size_t kLanes = 16;
constexpr std::size_t kTile = TensorSpec::TILE;
constexpr std::size_t kTileBytes = kTile * kTile * sizeof(std::uint16_t);
constexpr std::size_t kTileRowBytes = kTile * sizeof(std::uint16_t);

// The stage holds one FP32 accumulator per feature block of a group, so a group
// covers at most this many features and a wider row repeats the token loop for
// its next group of blocks. Every head width in the retained models (64 to 256)
// stays inside one group.
constexpr std::size_t kGroupBlocks = 16;
constexpr std::size_t kGroupFeatures = kGroupBlocks * kLanes;

// Magnitude bits of the two BF16 lanes of a 32-bit element, and the "this lane
// holds no live operand" sentinel the magnitude minima use. Every live
// magnitude is at least the smallest BF16 normal, so the sentinel may be any
// larger value; 0x7FFF is the largest finite magnitude.
constexpr std::uint32_t kMagnitudeMask = 0x7FFF7FFF;
constexpr std::uint16_t kMagnitudeSentinel = 0x7FFF;

// Lower bound on every nonzero product of an accepted row, as a binary
// exponent. A product of two BF16 operands has at most sixteen significant
// bits, so every nonzero product, partial sum, and final result of an accepted
// row is a multiple of 2^-115 and therefore normal: the hardware's operand DAZ
// and result flush cannot alter a representable value. A row whose smallest
// live operand magnitudes fall below this scale is produced by the scalar
// recurrence instead.
constexpr int kMinimumProductExponent = -100;
// Exponent fields of the two operand minima add up to at least
// 254 + kMinimumProductExponent for their product to reach 2^-100, because a
// BF16 value with exponent field e is at least 2^(e - 127).
constexpr unsigned int kRequiredExponentSum =
        static_cast<unsigned int>(254 + kMinimumProductExponent);

// The matrix DAZ function of the SDPA contract for one BF16 operand: a
// subnormal magnitude becomes the signed zero of the same sign, and every other
// pattern - including infinities and NaNs - is returned unchanged.
[[nodiscard]] std::uint16_t daz_operand(std::uint16_t bits) noexcept {
    const bool subnormal = (bits & 0x7F80u) == 0 && (bits & 0x007Fu) != 0;
    return subnormal ? static_cast<std::uint16_t>(bits & 0x8000u) : bits;
}

// The same function for all 32 packed BF16 lanes of a 512-bit vector. AVX512F
// has no 16-bit lane comparison, so the two halves of every 32-bit element are
// tested with 32-bit masks; AVX512BW is not part of the isolated target
// options.
[[nodiscard]] __m512i daz_operands(__m512i packed) noexcept {
    const __mmask16 low_subnormal = _mm512_testn_epi32_mask(
                                            packed,
                                            _mm512_set1_epi32(0x00007F80))
            & _mm512_test_epi32_mask(
                    packed, _mm512_set1_epi32(0x0000007F));
    const __mmask16 high_subnormal = _mm512_testn_epi32_mask(
                                             packed,
                                             _mm512_set1_epi32(0x7F800000))
            & _mm512_test_epi32_mask(
                    packed, _mm512_set1_epi32(0x007F0000));
    packed = _mm512_mask_and_epi32(
            packed, low_subnormal, packed,
            _mm512_set1_epi32(static_cast<int>(0xFFFF8000u)));
    return _mm512_mask_and_epi32(
            packed, high_subnormal, packed,
            _mm512_set1_epi32(static_cast<int>(0x8000FFFFu)));
}

// One 512-bit value reinterpreted as the packed BF16 operand vector
// VDPBF16PS consumes. Both vector types cover the same 64 bytes.
[[nodiscard]] __m512bh as_bf16_operands(__m512i packed) noexcept {
    return std::bit_cast<__m512bh>(packed);
}

// The 512-bit BF16 operand of one token pair and one feature block: lane i
// holds (value[t][i], value[t + 1][i]), which is the pair VDPBF16PS multiplies
// against the repeated probability pair. `first` and `second` are one 16
// feature tile row each; a zero row contributes nothing for an unpaired last
// token.
[[nodiscard]] __m512i interleave_rows(
        __m256i first, __m256i second) noexcept {
    const __m256i low = _mm256_unpacklo_epi16(first, second);
    const __m256i high = _mm256_unpackhi_epi16(first, second);
    const __m256i lower = _mm256_permute2x128_si256(low, high, 0x20);
    const __m256i upper = _mm256_permute2x128_si256(low, high, 0x31);
    return _mm512_inserti64x4(
            _mm512_castsi256_si512(lower), upper, 1);
}

// One V tile row of a feature block. A full block takes its 32 packed bytes
// from the tile; a tail block reads only its live features so no tile padding
// element is ever touched, leaving the dead lanes zero.
[[nodiscard]] __m256i load_operand_row(
        const unsigned char* row, std::size_t live_features) noexcept {
    if (live_features >= kLanes) {
        return _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row));
    }
    alignas(32) std::uint16_t values[kLanes] = {};
    for (std::size_t feature = 0; feature < live_features; ++feature) {
        std::memcpy(
                &values[feature],
                row + feature * sizeof(std::uint16_t),
                sizeof(std::uint16_t));
    }
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
}

// Byte address of the 16 feature tile row holding `token` for feature block
// `block`: the plane's tile grid is row-major over (token tile, feature tile)
// and every tile is 16x16 elements, so consecutive blocks are consecutive
// tiles and consecutive tokens are consecutive 32-byte rows inside them.
[[nodiscard]] const unsigned char* tile_grid_pointer(
        const unsigned char* plane_base, std::size_t tile_columns,
        std::size_t token) noexcept {
    return plane_base
            + (token / kTile) * tile_columns * kTileBytes
            + (token % kTile) * kTileRowBytes;
}

// Keep the smallest live magnitude of a packed operand vector: a zero magnitude
// is replaced by the sentinel so a plain unsigned minimum reports the smallest
// non-zero operand and reports the sentinel only when the vector is all zeros.
inline void fold_operand_magnitudes(
        __m512i packed, __m512i& minimum) noexcept {
    const __m512i magnitudes = _mm512_and_si512(
            packed, _mm512_set1_epi32(static_cast<int>(kMagnitudeMask)));
    const __m512i zero = _mm512_setzero_si512();
    const __m512i low = _mm512_and_si512(
            magnitudes, _mm512_set1_epi32(0x0000FFFF));
    const __m512i high = _mm512_srli_epi32(magnitudes, 16);
    const __m512i low_live = _mm512_mask_set1_epi32(
            low, _mm512_cmpeq_epi32_mask(low, zero),
            static_cast<int>(kMagnitudeSentinel));
    const __m512i high_live = _mm512_mask_set1_epi32(
            high, _mm512_cmpeq_epi32_mask(high, zero),
            static_cast<int>(kMagnitudeSentinel));
    minimum = _mm512_min_epu32(
            minimum, _mm512_min_epu32(low_live, high_live));
}

// Smallest live magnitude a fold left in `minimum`, or zero when every folded
// operand was zero.
[[nodiscard]] std::uint16_t minimum_magnitude(__m512i minimum) noexcept {
    const std::uint32_t value = _mm512_reduce_min_epu32(minimum);
    return value >= kMagnitudeSentinel
            ? 0 : static_cast<std::uint16_t>(value);
}

// Record one live probability magnitude in the row minimum.
inline void fold_probability(
        std::uint16_t operand, std::uint16_t& minimum) noexcept {
    const std::uint16_t magnitude =
            static_cast<std::uint16_t>(operand & 0x7FFFu);
    if (magnitude != 0 && magnitude < minimum) {
        minimum = magnitude;
    }
}

// The 32-bit lane of one token pair: its two BF16 halves are (p, q) in the
// order interleave_rows() places the matching V operands. An unpaired last
// token multiplies a zero, and both probabilities are DAZ-adjusted first.
[[nodiscard]] std::uint32_t probability_pair(
        const unsigned char* probability_row, std::size_t token,
        bool paired, std::uint16_t& minimum) noexcept {
    std::uint16_t first = 0;
    std::memcpy(
            &first, probability_row + token * sizeof(std::uint16_t),
            sizeof(first));
    first = daz_operand(first);
    fold_probability(first, minimum);
    if (!paired) {
        return first;
    }
    std::uint16_t second = 0;
    std::memcpy(
            &second,
            probability_row + (token + 1) * sizeof(std::uint16_t),
            sizeof(second));
    second = daz_operand(second);
    fold_probability(second, minimum);
    return static_cast<std::uint32_t>(first)
            | (static_cast<std::uint32_t>(second) << 16);
}

// Whether every nonzero product of a group is at least 2^-100 in magnitude,
// which is what lets the native stage prove that no flush can change a
// representable result. A group without a live probability or without a live V
// operand has no native work at all and defers to the scalar recurrence.
[[nodiscard]] bool native_group_safe(
        std::uint16_t probability_minimum,
        std::uint16_t value_minimum) noexcept {
    if (probability_minimum == 0 || value_minimum == 0) {
        return false;
    }
    const unsigned int probability_exponent = probability_minimum >> 7;
    const unsigned int value_exponent = value_minimum >> 7;
    return probability_exponent + value_exponent
            >= kRequiredExponentSum;
}

// Features one block of a group still covers: every block but the last of a
// group is full width.
[[nodiscard]] std::size_t live_lanes(
        std::size_t group_features, std::size_t block) noexcept {
    const std::size_t remaining = group_features - block * kLanes;
    return remaining < kLanes ? remaining : kLanes;
}

// Store the produced FP32 lanes of one feature block as BF16 outputs. A
// numerical zero is canonical `+0`, and the shared codec performs the single
// round-to-nearest-even encode the contract requires - it preserves
// representable subnormal outputs, unlike the hardware BF16 conversion.
void store_block(
        const SdpaRequest& request, std::span<const std::size_t> out_dimensions,
        std::size_t out_plane, std::size_t row, std::size_t out_column,
        __m512 accumulator, std::size_t lane_count) noexcept {
    const auto format = Bf16Codec::format(DataType::BF16);
    alignas(64) float lanes[kLanes];
    _mm512_store_ps(lanes, accumulator);
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        float value = lanes[lane];
        // Canonicalize a -0 accumulator to +0 before the final BF16 encode.
        if (value == 0.0f) value = 0.0f;
        const std::size_t bit = logical_element_bits(
                out_dimensions, DataType::BF16, out_plane, row,
                out_column + lane);
        store_bits(
                request.out.native_handle, bit,
                sizeof(std::uint16_t) * 8,
                Bf16Codec::encode(value, format));
    }
}

}  // namespace

bool sdpa_pv_bf16_row(
        const SdpaRequest& request, const unsigned char* probability_row,
        std::size_t v_head_plane, std::size_t out_plane, std::size_t row,
        std::size_t out_column, std::size_t visible_limit) noexcept {
    const std::size_t features = request.D;
    if (features == 0 || visible_limit == 0 || probability_row == nullptr
            || request.v.rank < 2 || request.out.rank < 2
            || request.v.native_handle == nullptr
            || request.out.native_handle == nullptr) {
        return false;
    }

    // The V view's tile grid, exactly as cpu_detail::logical_element_bits maps
    // it: the leading axes fold into the plane slot the caller passes, and the
    // final two logical axes are the token and feature coordinates.
    const std::size_t value_columns = request.v.dimensions[request.v.rank - 1];
    const std::size_t value_rows = request.v.dimensions[request.v.rank - 2];
    const std::size_t value_tile_columns =
            value_columns / kTile + (value_columns % kTile != 0);
    const std::size_t value_tile_rows =
            value_rows / kTile + (value_rows % kTile != 0);
    const unsigned char* const value_base = request.v.native_handle
            + v_head_plane * value_tile_rows * value_tile_columns
                    * kTileBytes;

    const std::span<const std::size_t> out_dimensions(
            request.out.dimensions.data(), request.out.rank);

    for (std::size_t group = 0; group < features; group += kGroupFeatures) {
        const std::size_t group_features =
                features - group < kGroupFeatures
                ? features - group : kGroupFeatures;
        const std::size_t block_count = group_features / kLanes
                + (group_features % kLanes != 0);
        const std::size_t first_block = group / kLanes;

        __m512 accumulator[kGroupBlocks];
        for (std::size_t block = 0; block < block_count; ++block) {
            accumulator[block] = _mm512_setzero_ps();
        }
        // Both operand minima start at the sentinel, so a group whose folded
        // operands are all zero keeps `native_group_safe` false and defers to
        // the scalar recurrence.
        __m512i value_minimum =
                _mm512_set1_epi32(static_cast<int>(kMagnitudeSentinel));
        std::uint16_t probability_minimum = kMagnitudeSentinel;

        for (std::size_t token = 0; token < visible_limit; token += 2) {
            const bool paired = token + 1 < visible_limit;
            const unsigned char* const first_base = tile_grid_pointer(
                    value_base, value_tile_columns, token)
                    + first_block * kTileBytes;
            const unsigned char* second_base = nullptr;
            if (paired) {
                second_base = tile_grid_pointer(
                                      value_base, value_tile_columns,
                                      token + 1)
                        + first_block * kTileBytes;
            }
            const std::uint32_t pair = probability_pair(
                    probability_row, token, paired, probability_minimum);
            const __m512bh operand = as_bf16_operands(
                    _mm512_set1_epi32(static_cast<int>(pair)));
            for (std::size_t block = 0; block < block_count; ++block) {
                const std::size_t lane_count =
                        live_lanes(group_features, block);
                const __m256i first = load_operand_row(
                        first_base + block * kTileBytes, lane_count);
                const __m256i second = paired
                        ? load_operand_row(
                                  second_base + block * kTileBytes, lane_count)
                        : _mm256_setzero_si256();
                const __m512i packed = daz_operands(
                        interleave_rows(first, second));
                fold_operand_magnitudes(packed, value_minimum);
                accumulator[block] = _mm512_dpbf16_ps(
                        accumulator[block], operand,
                        as_bf16_operands(packed));
            }
        }

        if (!native_group_safe(
                    probability_minimum,
                    minimum_magnitude(value_minimum))) {
#if defined(IOM_AVX512_BF16_TESTING)
            avx512_bf16_test_record(
                    Avx512Bf16Stage::SdpaPv, Avx512Bf16Path::Fallback);
#endif
            return false;
        }

        for (std::size_t block = 0; block < block_count; ++block) {
            store_block(
                    request, out_dimensions, out_plane, row,
                    out_column + group + block * kLanes, accumulator[block],
                    live_lanes(group_features, block));
        }
#if defined(IOM_AVX512_BF16_TESTING)
        // The stage reports work only after this group's vector dots executed.
        avx512_bf16_test_record(
                Avx512Bf16Stage::SdpaPv, Avx512Bf16Path::Native);
#endif
    }
    return true;
}

}  // namespace cpu_detail
}  // namespace iom
#pragma once

// Independent host-side SDPA reference (change 006-tinyllama / 08-causal-
// grouped-attention / 02).  This header deliberately calls no production
// codec, tensor-layout, address, queue, or SDPA helper.  It owns the named
// floating encodings, the scalar recurrence, the causal/GQA index mapping,
// special-value policy, fixed fixtures, and the comparison contract consumed
// by the future shared SDPA suite.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Public expected-value and fixture types.
// ---------------------------------------------------------------------------

enum class SdpaReferenceClass : std::uint8_t {
    finite,
    positive_zero,
    negative_zero,
    positive_infinity,
    negative_infinity,
    quiet_nan,
};

inline constexpr std::string_view sdpa_reference_class_name(
        SdpaReferenceClass value_class) noexcept {
    switch (value_class) {
        case SdpaReferenceClass::finite: return "finite";
        case SdpaReferenceClass::positive_zero: return "+0";
        case SdpaReferenceClass::negative_zero: return "-0";
        case SdpaReferenceClass::positive_infinity: return "+inf";
        case SdpaReferenceClass::negative_infinity: return "-inf";
        case SdpaReferenceClass::quiet_nan: return "nan";
    }
    return "unknown";
}

struct SdpaReferenceValue {
    std::uint64_t bits = 0;
    SdpaReferenceClass value_class = SdpaReferenceClass::finite;
};

enum class SdpaReferenceCaseKind : std::uint8_t {
    ones,
    mixed_gqa,
    causal_prefix,
    multiple_planes,
    boundary_rows,
    exact_capacity,
    signed_zeros,
    special_scores,
    included_zero_nonfinite,
    gradual_underflow,
    future_token_perturbation,
    capacity_tail_perturbation,
    physical_padding_perturbation,
    cached_incremental,
};

inline constexpr std::string_view sdpa_reference_case_kind_name(
        SdpaReferenceCaseKind kind) noexcept {
    switch (kind) {
        case SdpaReferenceCaseKind::ones: return "ones";
        case SdpaReferenceCaseKind::mixed_gqa: return "mixed_gqa";
        case SdpaReferenceCaseKind::causal_prefix: return "causal_prefix";
        case SdpaReferenceCaseKind::multiple_planes: return "multiple_planes";
        case SdpaReferenceCaseKind::boundary_rows: return "boundary_rows";
        case SdpaReferenceCaseKind::exact_capacity: return "exact_capacity";
        case SdpaReferenceCaseKind::signed_zeros: return "signed_zeros";
        case SdpaReferenceCaseKind::special_scores: return "special_scores";
        case SdpaReferenceCaseKind::included_zero_nonfinite:
            return "included_zero_nonfinite";
        case SdpaReferenceCaseKind::gradual_underflow: return "gradual_underflow";
        case SdpaReferenceCaseKind::future_token_perturbation:
            return "future_token_perturbation";
        case SdpaReferenceCaseKind::capacity_tail_perturbation:
            return "capacity_tail_perturbation";
        case SdpaReferenceCaseKind::physical_padding_perturbation:
            return "physical_padding_perturbation";
        case SdpaReferenceCaseKind::cached_incremental: return "cached_incremental";
    }
    return "unknown";
}

// q is [B..., Hq, R, D], k/v are [B..., Hkv, C, D], and the output is
// [B..., R, Hq*D].  The three raw vectors use contiguous row-major logical
// indexing and retain only the low bits of the named leaf.  An empty leading
// tuple is valid and means one independent plane.
struct SdpaReferenceCase {
    SdpaReferenceCaseKind kind = SdpaReferenceCaseKind::ones;
    iom::DataType data_type = iom::DataType::BF16;
    std::vector<std::size_t> leading_dimensions;
    std::size_t hq = 0;
    std::size_t hkv = 0;
    std::size_t rows = 0;       // R
    std::size_t head_dim = 0;   // D
    std::size_t capacity = 0;   // C
    std::size_t position = 0;   // a, initialized prefix before row zero
    std::size_t length = 0;     // L, readable initialized prefix
    std::vector<std::uint64_t> q_bits;
    std::vector<std::uint64_t> k_bits;
    std::vector<std::uint64_t> v_bits;
    // Physical padding is deliberately outside the logical reference.  This
    // marker lets a fixture name the perturbation without giving it influence.
    std::uint64_t physical_padding_salt = 0;
};

inline std::string sdpa_reference_case_label(
        const SdpaReferenceCase& reference_case) {
    std::string label(sdpa_reference_case_kind_name(reference_case.kind));
    label += " Hq=" + std::to_string(reference_case.hq);
    label += " Hkv=" + std::to_string(reference_case.hkv);
    label += " R=" + std::to_string(reference_case.rows);
    label += " D=" + std::to_string(reference_case.head_dim);
    label += " C=" + std::to_string(reference_case.capacity);
    label += " a=" + std::to_string(reference_case.position);
    label += " L=" + std::to_string(reference_case.length);
    return label;
}

// ---------------------------------------------------------------------------
// Semantic and current-support classification.
// ---------------------------------------------------------------------------

inline constexpr std::array<iom::DataType, 9> kSdpaApplicableDataTypes = {
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN,
        iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::BF16,
        iom::DataType::F32,
        iom::DataType::F64,
};

inline constexpr std::array<iom::DataType, 1> kSdpaCurrentSupportedDataTypes = {
        iom::DataType::BF16};

inline constexpr std::array<iom::DataType, 8> kSdpaCurrentUnsupportedDataTypes = {
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN,
        iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::F32,
        iom::DataType::F64,
};

inline constexpr std::array<iom::DataType, 14> kSdpaInapplicableDataTypes = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F8_E8M0,
};

static_assert(kSdpaApplicableDataTypes.size()
                              + kSdpaInapplicableDataTypes.size()
                      == 23,
              "SDPA classification must cover every declared leaf");
static_assert(kSdpaCurrentSupportedDataTypes.size()
                              + kSdpaCurrentUnsupportedDataTypes.size()
                      == kSdpaApplicableDataTypes.size(),
              "SDPA current matrix must partition the nine semantic leaves");

enum class SdpaDataTypeClass : std::uint8_t {
    inapplicable,
    unsupported,
    current_supported,
};

inline constexpr SdpaDataTypeClass sdpa_data_type_classification(
        iom::DataType type) noexcept {
    switch (type) {
        case iom::DataType::BF16: return SdpaDataTypeClass::current_supported;
        case iom::DataType::F4_E2M1:
        case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2:
        case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2:
        case iom::DataType::F16:
        case iom::DataType::F32:
        case iom::DataType::F64: return SdpaDataTypeClass::unsupported;
        case iom::DataType::BOOL:
        case iom::DataType::I2: case iom::DataType::U2:
        case iom::DataType::I4: case iom::DataType::U4:
        case iom::DataType::I8: case iom::DataType::U8:
        case iom::DataType::I16: case iom::DataType::U16:
        case iom::DataType::I32: case iom::DataType::U32:
        case iom::DataType::I64: case iom::DataType::U64:
        case iom::DataType::F8_E8M0: return SdpaDataTypeClass::inapplicable;
    }
    return SdpaDataTypeClass::inapplicable;
}

inline constexpr bool sdpa_currently_supported(iom::DataType type) noexcept {
    return sdpa_data_type_classification(type)
           == SdpaDataTypeClass::current_supported;
}

// ---------------------------------------------------------------------------
// Independent named-format codec and scalar helpers.
// ---------------------------------------------------------------------------

namespace sdpa_oracle {

struct FloatFormat {
    std::uint8_t bits = 0;
    std::uint8_t exponent_bits = 0;
    std::uint8_t mantissa_bits = 0;
    std::int16_t bias = 0;
    bool finite_only = false;
    bool has_infinity = false;
};

inline constexpr FloatFormat float_format(iom::DataType type) noexcept {
    switch (type) {
        case iom::DataType::F4_E2M1: return {4, 2, 1, 1, true, false};
        case iom::DataType::F6_E2M3: return {6, 2, 3, 1, true, false};
        case iom::DataType::F6_E3M2: return {6, 3, 2, 3, true, false};
        case iom::DataType::F8_E4M3FN: return {8, 4, 3, 7, true, false};
        case iom::DataType::F8_E5M2: return {8, 5, 2, 15, false, true};
        case iom::DataType::F16: return {16, 5, 10, 15, false, true};
        case iom::DataType::BF16: return {16, 8, 7, 127, false, true};
        case iom::DataType::F32: return {32, 8, 23, 127, false, true};
        case iom::DataType::F64: return {64, 11, 52, 1023, false, true};
        default: return {};
    }
}

inline constexpr std::uint64_t exponent_mask(const FloatFormat& format) noexcept {
    return (std::uint64_t{1} << format.exponent_bits) - 1;
}

inline constexpr std::uint64_t mantissa_mask(const FloatFormat& format) noexcept {
    return (std::uint64_t{1} << format.mantissa_bits) - 1;
}

inline constexpr std::uint64_t finite_exponent_max(
        const FloatFormat& format) noexcept {
    return format.finite_only && format.exponent_bits < 4
                   ? exponent_mask(format)
                   : exponent_mask(format) - 1;
}

inline constexpr bool has_nan_encoding(const FloatFormat& format) noexcept {
    return !format.finite_only || format.exponent_bits >= 4;
}

inline constexpr bool represents_nan(iom::DataType type) noexcept {
    return has_nan_encoding(float_format(type));
}

inline constexpr bool represents_infinity(iom::DataType type) noexcept {
    return float_format(type).has_infinity;
}

inline constexpr std::string_view leaf_name(iom::DataType type) noexcept {
    switch (type) {
        case iom::DataType::F4_E2M1: return "F4_E2M1";
        case iom::DataType::F6_E2M3: return "F6_E2M3";
        case iom::DataType::F6_E3M2: return "F6_E3M2";
        case iom::DataType::F8_E4M3FN: return "F8_E4M3FN";
        case iom::DataType::F8_E5M2: return "F8_E5M2";
        case iom::DataType::F16: return "F16";
        case iom::DataType::BF16: return "BF16";
        case iom::DataType::F32: return "F32";
        case iom::DataType::F64: return "F64";
        default: return "inapplicable";
    }
}

template <typename Carrier>
inline Carrier decode_carrier(
        std::uint64_t raw, const FloatFormat& format) noexcept {
    const std::uint64_t exponent =
            (raw >> format.mantissa_bits) & exponent_mask(format);
    const std::uint64_t mantissa = raw & mantissa_mask(format);
    const bool negative =
            ((raw >> (format.exponent_bits + format.mantissa_bits)) & 1) != 0;
    if (exponent == exponent_mask(format) && has_nan_encoding(format)) {
        if (format.has_infinity && mantissa == 0) {
            return negative ? -std::numeric_limits<Carrier>::infinity()
                            : std::numeric_limits<Carrier>::infinity();
        }
        return std::numeric_limits<Carrier>::quiet_NaN();
    }
    const Carrier magnitude = exponent != 0
            ? std::ldexp(
                      static_cast<Carrier>((std::uint64_t{1}
                                            << format.mantissa_bits)
                                           + mantissa),
                      static_cast<int>(exponent) - format.bias
                              - static_cast<int>(format.mantissa_bits))
            : std::ldexp(
                      static_cast<Carrier>(mantissa),
                      1 - format.bias - static_cast<int>(format.mantissa_bits));
    return negative ? -magnitude : magnitude;
}

template <typename Carrier>
inline std::uint64_t round_to_nearest_even(Carrier value) noexcept {
    const Carrier integral = std::floor(value);
    const Carrier remainder = value - integral;
    const std::uint64_t truncated = static_cast<std::uint64_t>(integral);
    return truncated
           + ((remainder > static_cast<Carrier>(0.5)
               || (remainder == static_cast<Carrier>(0.5)
                   && (truncated & 1)))
                      ? std::uint64_t{1}
                      : std::uint64_t{0});
}

template <typename Carrier>
inline std::uint64_t encode_carrier(
        Carrier value, const FloatFormat& format) noexcept {
    const std::uint64_t exponent_max = exponent_mask(format);
    const std::uint64_t fraction_max = mantissa_mask(format);
    const std::uint64_t finite_max = finite_exponent_max(format);
    const std::uint64_t sign = std::signbit(value) ? std::uint64_t{1} : 0;
    const auto pack = [sign, &format](
                              std::uint64_t exponent,
                              std::uint64_t mantissa) noexcept {
        return (sign << (format.exponent_bits + format.mantissa_bits))
               | (exponent << format.mantissa_bits) | mantissa;
    };
    // NaN payload/sign is outside the contract.  Pin a positive canonical
    // payload so expected bits are host-independent; finite-only narrow leaves
    // saturate to their maximum finite encoding instead.
    if (std::isnan(value)) {
        if (!has_nan_encoding(format)) {
            return (finite_max << format.mantissa_bits) | fraction_max;
        }
        return (exponent_max << format.mantissa_bits)
               | (format.finite_only ? fraction_max
                                     : std::uint64_t{1}
                                               << (format.mantissa_bits - 1));
    }
    value = std::fabs(value);
    const std::uint64_t overflow = format.has_infinity
            ? pack(exponent_max, 0)
            : pack(finite_max, fraction_max);
    if (std::isinf(value)) return overflow;
    if (value == 0) return sign << (format.exponent_bits + format.mantissa_bits);

    int exponent = 0;
    (void)std::frexp(value, &exponent);
    --exponent;
    const int min_sub =
            1 - format.bias - static_cast<int>(format.mantissa_bits);
    const int max_exp = static_cast<int>(finite_max) - format.bias;
    if (exponent < min_sub + static_cast<int>(format.mantissa_bits)) {
        const std::uint64_t quantized = round_to_nearest_even(
                std::ldexp(value, -min_sub));
        return quantized == (std::uint64_t{1} << format.mantissa_bits)
                ? pack(1, 0)
                : pack(0, quantized);
    }
    if (exponent > max_exp) return overflow;
    std::uint64_t fraction = round_to_nearest_even(
            std::ldexp(value, static_cast<int>(format.mantissa_bits) - exponent)
            - static_cast<Carrier>(std::uint64_t{1}
                                   << format.mantissa_bits));
    if (fraction == (std::uint64_t{1} << format.mantissa_bits)) {
        ++exponent;
        fraction = 0;
    }
    if (exponent > max_exp) return overflow;
    return pack(static_cast<std::uint64_t>(exponent + format.bias), fraction);
}

inline float decode_f32(iom::DataType type, std::uint64_t raw) noexcept {
    return decode_carrier<float>(raw, float_format(type));
}

inline std::uint64_t encode_f32(iom::DataType type, float value) noexcept {
    return encode_carrier<float>(value, float_format(type));
}

inline double decode_f64(std::uint64_t raw) noexcept {
    return decode_carrier<double>(raw, float_format(iom::DataType::F64));
}

inline std::uint64_t encode_f64(double value) noexcept {
    return encode_carrier<double>(value, float_format(iom::DataType::F64));
}

inline std::uint64_t value_bits(iom::DataType type, double value) noexcept {
    return type == iom::DataType::F64
                   ? encode_f64(value)
                   : encode_f32(type, static_cast<float>(value));
}

inline std::size_t bit_width(iom::DataType type) noexcept {
    return float_format(type).bits;
}

template <typename Carrier>
inline SdpaReferenceClass classify(Carrier value) noexcept {
    if (std::isnan(value)) return SdpaReferenceClass::quiet_nan;
    if (std::isinf(value)) {
        return std::signbit(value) ? SdpaReferenceClass::negative_infinity
                                   : SdpaReferenceClass::positive_infinity;
    }
    if (value == 0) {
        return std::signbit(value) ? SdpaReferenceClass::negative_zero
                                   : SdpaReferenceClass::positive_zero;
    }
    return SdpaReferenceClass::finite;
}

inline std::uint64_t ordered_distance(
        std::uint64_t left, std::uint64_t right, std::uint8_t bits) noexcept {
    const std::uint64_t mask =
            bits >= 64 ? ~std::uint64_t{0}
                        : (std::uint64_t{1} << bits) - 1;
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    const auto ordered = [mask, sign](std::uint64_t pattern) noexcept {
        const std::uint64_t value = pattern & mask;
        return (value & sign) ? (~value & mask) : (value | sign);
    };
    const std::uint64_t first = ordered(left);
    const std::uint64_t second = ordered(right);
    return first > second ? first - second : second - first;
}

// Matrix-operand denormals-are-zero for BF16's FP32 matrix path.  The
// comparison is made in FP32 and the sign of a flushed operand is preserved.
inline float matrix_daz(float value) noexcept {
    return value != 0.0f && std::fabs(value) < 0x1p-126f
                   ? std::copysign(0.0f, value)
                   : value;
}

inline float sdpa_matrix_daz(float value) noexcept { return matrix_daz(value); }

// Force each declared FP32/FP64 operation to materialize in its accumulator
// domain.  The volatile round trip prevents an accidental extended-precision
// recurrence or hidden contraction in the host oracle.
template <typename Carrier>
inline Carrier materialize(Carrier value) noexcept {
    volatile Carrier rounded = value;
    return rounded;
}

template <typename Carrier>
inline Carrier add(Carrier left, Carrier right) noexcept {
    return materialize(left + right);
}

template <typename Carrier>
inline Carrier sub(Carrier left, Carrier right) noexcept {
    return materialize(left - right);
}

template <typename Carrier>
inline Carrier mul(Carrier left, Carrier right) noexcept {
    return materialize(left * right);
}

template <typename Carrier>
inline Carrier div(Carrier left, Carrier right) noexcept {
    return materialize(left / right);
}

template <typename Carrier>
inline Carrier exp(Carrier value) noexcept {
    return materialize(std::exp(value));
}

template <typename Carrier>
inline Carrier sqrt(Carrier value) noexcept {
    return materialize(std::sqrt(value));
}

inline std::size_t checked_mul(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error("SDPA reference size multiplication overflow");
    }
    return left * right;
}

inline std::size_t checked_add(std::size_t left, std::size_t right) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error("SDPA reference size addition overflow");
    }
    return left + right;
}

inline std::size_t leading_plane_count(
        std::span<const std::size_t> dimensions) {
    std::size_t count = 1;
    for (const std::size_t extent : dimensions) {
        if (extent == 0) {
            throw std::invalid_argument(
                    "SDPA reference leading dimensions must be nonzero");
        }
        count = checked_mul(count, extent);
    }
    return count;
}

inline std::size_t q_offset(
        const SdpaReferenceCase& reference_case, std::size_t plane,
        std::size_t head, std::size_t row, std::size_t feature) {
    const std::size_t index = checked_add(
            checked_mul(plane, reference_case.hq), head);
    return checked_add(
            checked_mul(
                    checked_add(checked_mul(index, reference_case.rows), row),
                    reference_case.head_dim),
            feature);
}

inline std::size_t kv_offset(
        const SdpaReferenceCase& reference_case, std::size_t plane,
        std::size_t head, std::size_t token, std::size_t feature) {
    const std::size_t index = checked_add(
            checked_mul(plane, reference_case.hkv), head);
    return checked_add(
            checked_mul(
                    checked_add(checked_mul(index, reference_case.capacity), token),
                    reference_case.head_dim),
            feature);
}

inline std::size_t output_offset(
        const SdpaReferenceCase& reference_case, std::size_t plane,
        std::size_t row, std::size_t head, std::size_t feature) {
    const std::size_t index = checked_add(
            checked_mul(plane, reference_case.rows), row);
    return checked_add(
            checked_mul(
                    checked_add(checked_mul(index, reference_case.hq), head),
                    reference_case.head_dim),
            feature);
}

inline void validate_case(const SdpaReferenceCase& reference_case) {
    const FloatFormat format = float_format(reference_case.data_type);
    if (format.bits == 0) {
        throw std::invalid_argument("SDPA reference requires a floating leaf");
    }
    if (reference_case.hq == 0 || reference_case.hkv == 0
            || reference_case.rows == 0 || reference_case.head_dim == 0
            || reference_case.capacity == 0) {
        throw std::invalid_argument("SDPA reference extents must be nonzero");
    }
    if (reference_case.hq % reference_case.hkv != 0) {
        throw std::invalid_argument("SDPA reference requires Hq divisible by Hkv");
    }
    if (reference_case.length == 0
            || reference_case.length > reference_case.capacity) {
        throw std::invalid_argument("SDPA reference requires 0 < L <= C");
    }
    if (reference_case.position >= reference_case.capacity) {
        throw std::invalid_argument("SDPA reference requires a < C");
    }
    if (reference_case.rows
            > reference_case.capacity - reference_case.position) {
        throw std::invalid_argument("SDPA reference requires checked R <= C-a");
    }
    const std::size_t planes = leading_plane_count(
            std::span<const std::size_t>(reference_case.leading_dimensions));
    const std::size_t q_count = checked_mul(
            checked_mul(checked_mul(planes, reference_case.hq),
                        reference_case.rows),
            reference_case.head_dim);
    const std::size_t kv_count = checked_mul(
            checked_mul(checked_mul(planes, reference_case.hkv),
                        reference_case.capacity),
            reference_case.head_dim);
    if (reference_case.q_bits.size() != q_count
            || reference_case.k_bits.size() != kv_count
            || reference_case.v_bits.size() != kv_count) {
        throw std::invalid_argument(
                "SDPA reference raw bits do not match exact logical extents");
    }
    // T(r) must be nonempty.  With a >= 0 this is implied by L > 0, but keep
    // the explicit check because it is part of the public admission contract.
    for (std::size_t row = 0; row < reference_case.rows; ++row) {
        bool nonempty = false;
        for (std::size_t token = 0; token < reference_case.length; ++token) {
            if (token <= reference_case.position + row) {
                nonempty = true;
                break;
            }
        }
        if (!nonempty) {
            throw std::invalid_argument("SDPA reference has an empty causal row");
        }
    }
}

// Encode and classify one final destination value.  Stored numerical zeros
// are canonicalized by the evaluator before this helper is called.
inline SdpaReferenceValue stored_value(
        iom::DataType type, double value) noexcept {
    const std::uint64_t bits = value_bits(type, value);
    const auto value_class = type == iom::DataType::F64
            ? classify(decode_f64(bits))
            : classify(decode_f32(type, bits));
    return {bits, value_class};
}

inline SdpaReferenceValue stored_value(
        iom::DataType type, float value) noexcept {
    const std::uint64_t bits = encode_f32(type, value);
    return {bits, classify(decode_f32(type, bits))};
}

}  // namespace sdpa_oracle

// ---------------------------------------------------------------------------
// Deterministic fixture generation.
// ---------------------------------------------------------------------------

namespace sdpa_oracle {

inline constexpr std::array<double, 12> kPatternValues = {
        1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0};

inline double pattern_value(
        std::size_t plane, std::size_t head, std::size_t row,
        std::size_t coordinate, std::size_t salt) noexcept {
    return kPatternValues[(plane * 37 + head * 17 + row * 11
                           + coordinate * 5 + salt)
                          % kPatternValues.size()];
}

inline SdpaReferenceCase make_case(
        SdpaReferenceCaseKind kind, iom::DataType type,
        std::vector<std::size_t> leading_dimensions, std::size_t hq,
        std::size_t hkv, std::size_t rows, std::size_t head_dim,
        std::size_t capacity, std::size_t position, std::size_t length,
        const std::vector<double>& q_values,
        const std::vector<double>& k_values,
        const std::vector<double>& v_values,
        std::uint64_t physical_padding_salt = 0) {
    SdpaReferenceCase reference_case;
    reference_case.kind = kind;
    reference_case.data_type = type;
    reference_case.leading_dimensions = std::move(leading_dimensions);
    reference_case.hq = hq;
    reference_case.hkv = hkv;
    reference_case.rows = rows;
    reference_case.head_dim = head_dim;
    reference_case.capacity = capacity;
    reference_case.position = position;
    reference_case.length = length;
    reference_case.physical_padding_salt = physical_padding_salt;
    reference_case.q_bits.reserve(q_values.size());
    reference_case.k_bits.reserve(k_values.size());
    reference_case.v_bits.reserve(v_values.size());
    for (const double value : q_values) {
        reference_case.q_bits.push_back(value_bits(type, value));
    }
    for (const double value : k_values) {
        reference_case.k_bits.push_back(value_bits(type, value));
    }
    for (const double value : v_values) {
        reference_case.v_bits.push_back(value_bits(type, value));
    }
    validate_case(reference_case);
    return reference_case;
}

inline std::vector<double> fill_q(
        std::size_t planes, std::size_t hq, std::size_t rows,
        std::size_t head_dim, std::size_t salt) {
    std::vector<double> values(
            planes * hq * rows * head_dim, 0.0);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < hq; ++head) {
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t feature = 0; feature < head_dim; ++feature) {
                    values[q_offset(
                            SdpaReferenceCase{.hq = hq,
                                              .hkv = 1,
                                              .rows = rows,
                                              .head_dim = head_dim},
                            plane, head, row, feature)] =
                            pattern_value(plane, head, row, feature, salt);
                }
            }
        }
    }
    return values;
}

inline std::vector<double> fill_kv(
        std::size_t planes, std::size_t hkv, std::size_t capacity,
        std::size_t head_dim, std::size_t salt) {
    std::vector<double> values(planes * hkv * capacity * head_dim, 0.0);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < hkv; ++head) {
            for (std::size_t token = 0; token < capacity; ++token) {
                for (std::size_t feature = 0; feature < head_dim; ++feature) {
                    values[((plane * hkv + head) * capacity + token)
                                   * head_dim
                           + feature] =
                            pattern_value(plane, head, token, feature, salt);
                }
            }
        }
    }
    return values;
}

inline SdpaReferenceCase make_ones_case(iom::DataType type) {
    constexpr std::size_t planes = 2;
    constexpr std::size_t hq = 4;
    constexpr std::size_t hkv = 2;
    constexpr std::size_t rows = 2;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 5;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 4;
    return make_case(
            SdpaReferenceCaseKind::ones, type, {planes}, hq, hkv, rows,
            head_dim, capacity, position, length,
            std::vector<double>(planes * hq * rows * head_dim, 1.0),
            std::vector<double>(planes * hkv * capacity * head_dim, 1.0),
            std::vector<double>(planes * hkv * capacity * head_dim, 1.0));
}

inline SdpaReferenceCase make_mixed_gqa_case(iom::DataType type) {
    constexpr std::size_t planes = 2;
    constexpr std::size_t hq = 4;
    constexpr std::size_t hkv = 2;
    constexpr std::size_t rows = 3;
    constexpr std::size_t head_dim = 5;
    constexpr std::size_t capacity = 7;
    constexpr std::size_t position = 2;
    constexpr std::size_t length = 5;
    std::vector<double> q = fill_q(planes, hq, rows, head_dim, 1);
    std::vector<double> k = fill_kv(planes, hkv, capacity, head_dim, 3);
    std::vector<double> v = fill_kv(planes, hkv, capacity, head_dim, 7);
    return make_case(
            SdpaReferenceCaseKind::mixed_gqa, type, {2, 1}, hq, hkv, rows,
            head_dim, capacity, position, length, q, k, v);
}

inline SdpaReferenceCase make_causal_prefix_case(iom::DataType type) {
    constexpr std::size_t hq = 2;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 3;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 7;
    constexpr std::size_t position = 2;
    constexpr std::size_t length = 4;
    std::vector<double> q = fill_q(1, hq, rows, head_dim, 2);
    std::vector<double> k = fill_kv(1, hkv, capacity, head_dim, 5);
    std::vector<double> v = fill_kv(1, hkv, capacity, head_dim, 9);
    return make_case(
            SdpaReferenceCaseKind::causal_prefix, type, {}, hq, hkv, rows,
            head_dim, capacity, position, length, q, k, v);
}

inline SdpaReferenceCase make_multiple_planes_case(iom::DataType type) {
    constexpr std::size_t planes = 6;
    constexpr std::size_t hq = 3;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 6;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 4;
    return make_case(
            SdpaReferenceCaseKind::multiple_planes, type, {2, 3}, hq, hkv,
            rows, head_dim, capacity, position, length,
            fill_q(planes, hq, rows, head_dim, 13),
            fill_kv(planes, hkv, capacity, head_dim, 15),
            fill_kv(planes, hkv, capacity, head_dim, 19));
}

inline SdpaReferenceCase make_boundary_rows_case(
        iom::DataType type, std::size_t rows) {
    constexpr std::size_t hq = 2;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t head_dim = 7;
    constexpr std::size_t capacity = 19;
    constexpr std::size_t position = 2;
    constexpr std::size_t length = 17;
    return make_case(
            SdpaReferenceCaseKind::boundary_rows, type, {}, hq, hkv, rows,
            head_dim, capacity, position, length,
            fill_q(1, hq, rows, head_dim, rows),
            fill_kv(1, hkv, capacity, head_dim, 23 + rows),
            fill_kv(1, hkv, capacity, head_dim, 29 + rows));
}

inline SdpaReferenceCase make_exact_capacity_case(iom::DataType type) {
    constexpr std::size_t hq = 2;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 17;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 17;
    constexpr std::size_t position = 0;
    constexpr std::size_t length = 17;
    return make_case(
            SdpaReferenceCaseKind::exact_capacity, type, {}, hq, hkv, rows,
            head_dim, capacity, position, length,
            fill_q(1, hq, rows, head_dim, 31),
            fill_kv(1, hkv, capacity, head_dim, 37),
            fill_kv(1, hkv, capacity, head_dim, 41));
}

inline SdpaReferenceCase make_signed_zero_case(iom::DataType type) {
    constexpr std::size_t hq = 1;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t capacity = 3;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 2;
    const std::vector<double> q = {
            +0.0, -0.0,
            -0.0, +0.0};
    const std::vector<double> k = {
            +0.0, -0.0,
            -0.0, +0.0,
            +1.0, -1.0};
    const std::vector<double> v = {
            +0.0, -0.0,
            -0.0, +0.0,
            +7.0, -7.0};
    return make_case(
            SdpaReferenceCaseKind::signed_zeros, type, {}, hq, hkv, rows,
            head_dim, capacity, position, length, q, k, v);
}

inline SdpaReferenceCase make_special_scores_case(iom::DataType type) {
    constexpr std::size_t hq = 1;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 3;
    constexpr std::size_t head_dim = 2;
    constexpr std::size_t capacity = 5;
    constexpr std::size_t position = 2;
    constexpr std::size_t length = 2;
    std::vector<double> q(rows * head_dim, 1.0);
    std::vector<double> k(capacity * head_dim, 1.0);
    std::vector<double> v(capacity * head_dim, 0.0);
    v[0] = 2.0; v[1] = 1.0;
    v[2] = 4.0; v[3] = 3.0;
    v[4] = 8.0; v[5] = 7.0;
    if (represents_infinity(type)) {
        // Row zero has two +inf scores, row one has all -inf scores, and row
        // two has a NaN score.  No inf*inf product is formed.
        q[0] = 1.0; q[1] = 0.0;
        k[0] = std::numeric_limits<double>::infinity();
        k[2] = std::numeric_limits<double>::infinity();
        q[2] = -1.0;
        q[3] = 0.0;
        q[4] = std::numeric_limits<double>::quiet_NaN();
        q[5] = 0.0;
        k[4] = 1.0; k[5] = 0.0;
    } else if (represents_nan(type)) {
        q[4] = std::numeric_limits<double>::quiet_NaN();
    }
    return make_case(
            SdpaReferenceCaseKind::special_scores, type, {}, hq, hkv, rows,
            head_dim, capacity, position, length, q, k, v);
}

inline SdpaReferenceCase make_included_zero_nonfinite_case(
        iom::DataType type) {
    constexpr std::size_t hq = 1;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 1;
    constexpr std::size_t head_dim = 1;
    constexpr std::size_t capacity = 2;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 2;
    const std::vector<double> q = {1.0};
    const std::vector<double> k = {
            -std::numeric_limits<double>::infinity(), 0.0};
    const std::vector<double> v = {
            std::numeric_limits<double>::infinity(), 1.0};
    return make_case(
            SdpaReferenceCaseKind::included_zero_nonfinite, type, {}, hq, hkv,
            rows, head_dim, capacity, position, length, q, k, v);
}

inline SdpaReferenceCase make_gradual_underflow_case() {
    constexpr iom::DataType type = iom::DataType::BF16;
    constexpr std::size_t hq = 1;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 1;
    constexpr std::size_t head_dim = 1;
    constexpr std::size_t capacity = 2;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 2;
    const std::vector<double> q = {+0.0};
    const std::vector<double> k = {+0.0, +0.0};
    const std::vector<double> v = {
            std::ldexp(1.0, -126), +0.0};
    return make_case(
            SdpaReferenceCaseKind::gradual_underflow, type, {}, hq, hkv,
            rows, head_dim, capacity, position, length, q, k, v);
}

inline SdpaReferenceCase make_future_token_case(iom::DataType type) {
    constexpr std::size_t hq = 2;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 1;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 8;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 6;
    return make_case(
            SdpaReferenceCaseKind::future_token_perturbation, type, {}, hq, hkv,
            rows, head_dim, capacity, position, length,
            fill_q(1, hq, rows, head_dim, 47),
            fill_kv(1, hkv, capacity, head_dim, 53),
            fill_kv(1, hkv, capacity, head_dim, 59));
}

inline SdpaReferenceCase make_capacity_tail_case(iom::DataType type) {
    constexpr std::size_t hq = 2;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 8;
    constexpr std::size_t position = 1;
    constexpr std::size_t length = 4;
    return make_case(
            SdpaReferenceCaseKind::capacity_tail_perturbation, type, {}, hq,
            hkv, rows, head_dim, capacity, position, length,
            fill_q(1, hq, rows, head_dim, 61),
            fill_kv(1, hkv, capacity, head_dim, 67),
            fill_kv(1, hkv, capacity, head_dim, 71));
}

inline SdpaReferenceCase make_padding_case(iom::DataType type) {
    SdpaReferenceCase reference_case = make_case(
            SdpaReferenceCaseKind::physical_padding_perturbation, type,
            {2}, 2, 1, 2, 3, 6, 1, 4,
            fill_q(2, 2, 2, 3, 73), fill_kv(2, 1, 6, 3, 79),
            fill_kv(2, 1, 6, 3, 83), 0xD00D);
    return reference_case;
}

inline SdpaReferenceCase make_cached_incremental_case(iom::DataType type) {
    constexpr std::size_t hq = 2;
    constexpr std::size_t hkv = 1;
    constexpr std::size_t rows = 4;
    constexpr std::size_t head_dim = 3;
    constexpr std::size_t capacity = 4;
    constexpr std::size_t position = 0;
    constexpr std::size_t length = 4;
    return make_case(
            SdpaReferenceCaseKind::cached_incremental, type, {}, hq, hkv,
            rows, head_dim, capacity, position, length,
            fill_q(1, hq, rows, head_dim, 89),
            fill_kv(1, hkv, capacity, head_dim, 97),
            fill_kv(1, hkv, capacity, head_dim, 101));
}

}  // namespace sdpa_oracle

inline constexpr std::array<std::size_t, 4> kSdpaBoundaryRows = {1, 15, 16, 17};

// ---------------------------------------------------------------------------
// Consumable scalar reference.
// ---------------------------------------------------------------------------

inline std::vector<SdpaReferenceValue> evaluate(
        const SdpaReferenceCase& reference_case) {
    if (sdpa_data_type_classification(reference_case.data_type)
        == SdpaDataTypeClass::inapplicable) {
        return {};
    }
    sdpa_oracle::validate_case(reference_case);
    const std::size_t planes = sdpa_oracle::leading_plane_count(
            std::span<const std::size_t>(reference_case.leading_dimensions));
    const std::size_t output_count = sdpa_oracle::checked_mul(
            sdpa_oracle::checked_mul(
                    sdpa_oracle::checked_mul(planes, reference_case.rows),
                    reference_case.hq),
            reference_case.head_dim);
    std::vector<SdpaReferenceValue> values(output_count);
    const bool f64 = reference_case.data_type == iom::DataType::F64;
    const bool bf16 = reference_case.data_type == iom::DataType::BF16;
    const std::size_t group = reference_case.hq / reference_case.hkv;

    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < reference_case.hq; ++head) {
            const std::size_t kv_head = head / group;
            for (std::size_t row = 0; row < reference_case.rows; ++row) {
                if (f64) {
                    std::vector<double> scores(reference_case.length,
                                               -std::numeric_limits<double>::infinity());
                    for (std::size_t token = 0; token < reference_case.length;
                         ++token) {
                        if (token > reference_case.position + row) continue;
                        double dot = 0.0;
                        for (std::size_t feature = 0;
                             feature < reference_case.head_dim; ++feature) {
                            const double q = sdpa_oracle::decode_f64(
                                    reference_case.q_bits[sdpa_oracle::q_offset(
                                            reference_case, plane, head, row,
                                            feature)]);
                            const double k = sdpa_oracle::decode_f64(
                                    reference_case.k_bits[sdpa_oracle::kv_offset(
                                            reference_case, plane, kv_head,
                                            token, feature)]);
                            dot = sdpa_oracle::add(dot, sdpa_oracle::mul(q, k));
                        }
                        const double root = sdpa_oracle::sqrt(
                                static_cast<double>(reference_case.head_dim));
                        scores[token] = sdpa_oracle::div(dot, root);
                    }
                    std::vector<double> probabilities(reference_case.length, 0.0);
                    bool has_nan = false;
                    std::size_t positive_infinities = 0;
                    for (std::size_t token = 0; token < reference_case.length;
                         ++token) {
                        has_nan = has_nan || std::isnan(scores[token]);
                        positive_infinities +=
                                scores[token] == std::numeric_limits<double>::infinity();
                    }
                    const bool all_negative_infinity =
                            positive_infinities == 0 && !has_nan
                            && std::all_of(
                                    scores.begin(), scores.end(),
                                    [](double value) {
                                        return std::isinf(value)
                                               && std::signbit(value);
                                    });
                    if (!has_nan && !all_negative_infinity) {
                        if (positive_infinities != 0) {
                            const double probability = sdpa_oracle::div(
                                    1.0, static_cast<double>(positive_infinities));
                            for (std::size_t token = 0;
                                 token < reference_case.length; ++token) {
                                if (scores[token]
                                    == std::numeric_limits<double>::infinity()) {
                                    probabilities[token] = probability;
                                }
                            }
                        } else {
                            const double maximum = *std::max_element(
                                    scores.begin(), scores.end());
                            double sum = 0.0;
                            for (std::size_t token = 0;
                                 token < reference_case.length; ++token) {
                                if (std::isinf(scores[token])
                                    && std::signbit(scores[token])) {
                                    continue;
                                }
                                probabilities[token] = sdpa_oracle::exp(
                                        sdpa_oracle::sub(scores[token], maximum));
                                sum = sdpa_oracle::add(sum, probabilities[token]);
                            }
                            for (std::size_t token = 0;
                                 token < reference_case.length; ++token) {
                                probabilities[token] = sdpa_oracle::div(
                                        probabilities[token], sum);
                            }
                        }
                    }
                    for (std::size_t feature = 0;
                         feature < reference_case.head_dim; ++feature) {
                        double result = 0.0;
                        if (has_nan || all_negative_infinity) {
                            result = std::numeric_limits<double>::quiet_NaN();
                        } else {
                            for (std::size_t token = 0;
                                 token < reference_case.length; ++token) {
                                if (token > reference_case.position + row) continue;
                                const double v = sdpa_oracle::decode_f64(
                                        reference_case.v_bits[sdpa_oracle::kv_offset(
                                                reference_case, plane, kv_head,
                                                token, feature)]);
                                result = sdpa_oracle::add(
                                        result,
                                        sdpa_oracle::mul(probabilities[token], v));
                            }
                        }
                        if (result == 0.0) result = 0.0;
                        values[sdpa_oracle::output_offset(
                                reference_case, plane, row, head, feature)] =
                                sdpa_oracle::stored_value(reference_case.data_type,
                                                           result);
                    }
                    continue;
                }

                std::vector<float> scores(reference_case.length,
                                          -std::numeric_limits<float>::infinity());
                for (std::size_t token = 0; token < reference_case.length;
                     ++token) {
                    if (token > reference_case.position + row) continue;
                    float dot = 0.0f;
                    for (std::size_t feature = 0;
                         feature < reference_case.head_dim; ++feature) {
                        float q = sdpa_oracle::decode_f32(
                                reference_case.data_type,
                                reference_case.q_bits[sdpa_oracle::q_offset(
                                        reference_case, plane, head, row,
                                        feature)]);
                        float k = sdpa_oracle::decode_f32(
                                reference_case.data_type,
                                reference_case.k_bits[sdpa_oracle::kv_offset(
                                        reference_case, plane, kv_head, token,
                                        feature)]);
                        if (bf16) {
                            q = sdpa_oracle::matrix_daz(q);
                            k = sdpa_oracle::matrix_daz(k);
                        }
                        dot = sdpa_oracle::add(dot, sdpa_oracle::mul(q, k));
                    }
                    const float root = sdpa_oracle::sqrt(
                            static_cast<float>(reference_case.head_dim));
                    scores[token] = sdpa_oracle::div(dot, root);
                }
                std::vector<float> probabilities(reference_case.length, 0.0f);
                bool has_nan = false;
                std::size_t positive_infinities = 0;
                for (std::size_t token = 0; token < reference_case.length;
                     ++token) {
                    has_nan = has_nan || std::isnan(scores[token]);
                    positive_infinities +=
                            scores[token] == std::numeric_limits<float>::infinity();
                }
                const bool all_negative_infinity =
                        positive_infinities == 0 && !has_nan
                        && std::all_of(
                                scores.begin(), scores.end(),
                                [](float value) {
                                    return std::isinf(value)
                                           && std::signbit(value);
                                });
                if (!has_nan && !all_negative_infinity) {
                    if (positive_infinities != 0) {
                        const float probability = sdpa_oracle::div(
                                1.0f, static_cast<float>(positive_infinities));
                        for (std::size_t token = 0;
                             token < reference_case.length; ++token) {
                            if (scores[token]
                                == std::numeric_limits<float>::infinity()) {
                                probabilities[token] = probability;
                            }
                        }
                    } else {
                        const float maximum = *std::max_element(
                                scores.begin(), scores.end());
                        float sum = 0.0f;
                        for (std::size_t token = 0;
                             token < reference_case.length; ++token) {
                            if (std::isinf(scores[token])
                                && std::signbit(scores[token])) {
                                continue;
                            }
                            probabilities[token] = sdpa_oracle::exp(
                                    sdpa_oracle::sub(scores[token], maximum));
                            sum = sdpa_oracle::add(sum, probabilities[token]);
                        }
                        for (std::size_t token = 0;
                             token < reference_case.length; ++token) {
                            probabilities[token] = sdpa_oracle::div(
                                    probabilities[token], sum);
                        }
                    }
                }
                if (bf16) {
                    for (float& probability : probabilities) {
                        if (probability == 0.0f) {
                            probability = 0.0f;
                            continue;
                        }
                        probability = sdpa_oracle::decode_f32(
                                iom::DataType::BF16,
                                sdpa_oracle::encode_f32(
                                        iom::DataType::BF16, probability));
                        probability = sdpa_oracle::matrix_daz(probability);
                    }
                }
                for (std::size_t feature = 0;
                     feature < reference_case.head_dim; ++feature) {
                    float result = 0.0f;
                    if (has_nan || all_negative_infinity) {
                        result = std::numeric_limits<float>::quiet_NaN();
                    } else {
                        for (std::size_t token = 0;
                             token < reference_case.length; ++token) {
                            if (token > reference_case.position + row) continue;
                            float v = sdpa_oracle::decode_f32(
                                    reference_case.data_type,
                                    reference_case.v_bits[sdpa_oracle::kv_offset(
                                            reference_case, plane, kv_head,
                                            token, feature)]);
                            if (bf16) v = sdpa_oracle::matrix_daz(v);
                            result = sdpa_oracle::add(
                                    result,
                                    sdpa_oracle::mul(probabilities[token], v));
                        }
                    }
                    if (result == 0.0f) result = 0.0f;
                    values[sdpa_oracle::output_offset(
                            reference_case, plane, row, head, feature)] =
                            sdpa_oracle::stored_value(reference_case.data_type,
                                                       result);
                }
            }
        }
    }
    return values;
}

inline bool matches(
        iom::DataType type, std::uint64_t actual_bits,
        const SdpaReferenceValue& expected) noexcept {
    const sdpa_oracle::FloatFormat format = sdpa_oracle::float_format(type);
    if (format.bits == 0) return false;
    const std::uint64_t mask = format.bits >= 64
            ? ~std::uint64_t{0}
            : (std::uint64_t{1} << format.bits) - 1;
    actual_bits &= mask;
    const std::uint64_t expected_bits = expected.bits & mask;
    const SdpaReferenceClass actual_class = type == iom::DataType::F64
            ? sdpa_oracle::classify(sdpa_oracle::decode_f64(actual_bits))
            : sdpa_oracle::classify(
                      sdpa_oracle::decode_f32(type, actual_bits));
    if (actual_class != expected.value_class) return false;
    if (expected.value_class != SdpaReferenceClass::finite) return true;
    if (type != iom::DataType::BF16) return actual_bits == expected_bits;

    const float actual = sdpa_oracle::decode_f32(type, actual_bits);
    const float reference = sdpa_oracle::decode_f32(type, expected_bits);
    const float next = std::nextafter(reference,
                                      std::numeric_limits<float>::infinity());
    const float ulp = std::fabs(next - reference);
    const float absolute_error = std::fabs(actual - reference);
    return sdpa_oracle::ordered_distance(actual_bits, expected_bits, 16) <= 2
           || absolute_error
                      <= std::max(2.0f * ulp, std::ldexp(1.0f, -10));
}

inline bool matches(
        iom::DataType type, const SdpaReferenceValue& actual,
        const SdpaReferenceValue& expected) noexcept {
    return matches(type, actual.bits, expected);
}

inline std::vector<SdpaReferenceCase> sdpa_reference_cases(
        iom::DataType type) {
    if (type != iom::DataType::BF16) {
        return {};
    }
    std::vector<SdpaReferenceCase> cases;
    cases.push_back(sdpa_oracle::make_ones_case(type));
    cases.push_back(sdpa_oracle::make_mixed_gqa_case(type));
    cases.push_back(sdpa_oracle::make_causal_prefix_case(type));
    cases.push_back(sdpa_oracle::make_multiple_planes_case(type));
    for (const std::size_t rows : kSdpaBoundaryRows) {
        cases.push_back(sdpa_oracle::make_boundary_rows_case(type, rows));
    }
    cases.push_back(sdpa_oracle::make_exact_capacity_case(type));
    cases.push_back(sdpa_oracle::make_signed_zero_case(type));
    if (sdpa_oracle::represents_nan(type)) {
        cases.push_back(sdpa_oracle::make_special_scores_case(type));
    }
    if (sdpa_oracle::represents_infinity(type)) {
        cases.push_back(sdpa_oracle::make_included_zero_nonfinite_case(type));
    }
    if (type == iom::DataType::BF16) {
        cases.push_back(sdpa_oracle::make_gradual_underflow_case());
    }
    cases.push_back(sdpa_oracle::make_future_token_case(type));
    cases.push_back(sdpa_oracle::make_capacity_tail_case(type));
    cases.push_back(sdpa_oracle::make_padding_case(type));
    cases.push_back(sdpa_oracle::make_cached_incremental_case(type));
    return cases;
}

// ---------------------------------------------------------------------------
// Backend-free analytic checks.
// ---------------------------------------------------------------------------

struct SdpaReferenceSelfCheckReport {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    [[nodiscard]] bool ok() const noexcept { return failures.empty(); }
};

inline void sdpa_require(
        SdpaReferenceSelfCheckReport& report, bool condition,
        std::string message) {
    ++report.checks;
    if (!condition) report.failures.push_back(std::move(message));
}

inline bool sdpa_all_equal(
        std::span<const SdpaReferenceValue> left,
        std::span<const SdpaReferenceValue> right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].bits != right[index].bits
                || left[index].value_class != right[index].value_class) {
            return false;
        }
    }
    return true;
}

inline SdpaReferenceCase sdpa_incremental_slice(
        const SdpaReferenceCase& full, std::size_t row) {
    if (row >= full.rows) {
        throw std::invalid_argument("SDPA incremental row is out of range");
    }
    SdpaReferenceCase one = full;
    one.kind = SdpaReferenceCaseKind::cached_incremental;
    one.rows = 1;
    one.position = row;
    one.length = row + 1;
    one.q_bits.clear();
    one.q_bits.reserve(full.hq * full.head_dim);
    for (std::size_t head = 0; head < full.hq; ++head) {
        for (std::size_t feature = 0; feature < full.head_dim; ++feature) {
            one.q_bits.push_back(full.q_bits[sdpa_oracle::q_offset(
                    full, 0, head, row, feature)]);
        }
    }
    // K/V keep exactly the full cache image, including unread capacity tail.
    sdpa_oracle::validate_case(one);
    return one;
}

inline SdpaReferenceSelfCheckReport sdpa_reference_self_check() {
    SdpaReferenceSelfCheckReport report;
    const auto name = [](iom::DataType type) {
        return std::string(sdpa_oracle::leaf_name(type));
    };
    const auto occurrences = [](std::span<const iom::DataType> values,
                                iom::DataType type) {
        std::size_t count = 0;
        for (const iom::DataType value : values) count += value == type ? 1 : 0;
        return count;
    };

    sdpa_require(
            report,
            kSdpaApplicableDataTypes.size()
                            + kSdpaInapplicableDataTypes.size()
                    == 23,
            "SDPA lists must partition all 23 declared leaves");
    for (const iom::DataType type : kSdpaApplicableDataTypes) {
        const std::string leaf = name(type);
        sdpa_require(
                report,
                sdpa_data_type_classification(type)
                        != SdpaDataTypeClass::inapplicable,
                leaf + ": semantic leaf became inapplicable");
        sdpa_require(
                report,
                occurrences(kSdpaApplicableDataTypes, type) == 1
                        && occurrences(kSdpaInapplicableDataTypes, type) == 0,
                leaf + ": semantic leaf is not classified exactly once");
        sdpa_require(
                report,
                occurrences(kSdpaCurrentSupportedDataTypes, type)
                                + occurrences(kSdpaCurrentUnsupportedDataTypes, type)
                        == 1,
                leaf + ": current matrix must classify exactly once");
        sdpa_require(
                report,
                type == iom::DataType::BF16
                        ? !sdpa_reference_cases(type).empty()
                        : sdpa_reference_cases(type).empty(),
                leaf + ": current fixture classification is inconsistent");
    }
    for (const iom::DataType type : kSdpaInapplicableDataTypes) {
        sdpa_require(
                report,
                sdpa_data_type_classification(type)
                        == SdpaDataTypeClass::inapplicable,
                "inapplicable leaf classified as arithmetic");
        sdpa_require(
                report,
                sdpa_reference_cases(type).empty(),
                "inapplicable leaf unexpectedly received fixtures");
    }
    sdpa_require(
            report,
            sdpa_data_type_classification(iom::DataType::BF16)
                    == SdpaDataTypeClass::current_supported,
            "BF16 must be the only current SDPA-supported leaf");
    for (const iom::DataType type : kSdpaCurrentUnsupportedDataTypes) {
        sdpa_require(
                report,
                sdpa_data_type_classification(type)
                        == SdpaDataTypeClass::unsupported,
                name(type) + ": future leaf must remain explicitly unsupported");
    }
    // Every current fixture is evaluated from its own exact raw image, and
    // every expected encoding is canonical and matches itself.
    for (const iom::DataType type : kSdpaCurrentSupportedDataTypes) {
        for (const SdpaReferenceCase& reference_case :
             sdpa_reference_cases(type)) {
            const std::vector<SdpaReferenceValue> expected = evaluate(reference_case);
            const std::size_t expected_count = sdpa_oracle::checked_mul(
                    sdpa_oracle::checked_mul(
                            sdpa_oracle::leading_plane_count(
                                    std::span<const std::size_t>(
                                            reference_case.leading_dimensions)),
                            reference_case.rows),
                    sdpa_oracle::checked_mul(reference_case.hq,
                                             reference_case.head_dim));
            const std::string label = name(type) + " "
                                      + sdpa_reference_case_label(reference_case);
            sdpa_require(
                    report, expected.size() == expected_count,
                    label + ": output count does not match [B,R,Hq*D]");
            for (const SdpaReferenceValue& value : expected) {
                sdpa_require(
                        report,
                        matches(type, value.bits, value),
                        label + ": an expected value must match itself");
            }
        }
    }

    // Equal scores and unit V are the closed-form one identity; it also
    // exercises every GQA group and the two independent leading planes.
    {
        const SdpaReferenceCase ones = sdpa_oracle::make_ones_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> expected = evaluate(ones);
        for (const SdpaReferenceValue& value : expected) {
            sdpa_require(
                    report,
                    value.bits == sdpa_oracle::value_bits(
                                          iom::DataType::BF16, 1.0),
                    "all-ones SDPA must average to one");
        }
    }

    // Distinct KV heads must be visible through g(h), while paired query heads
    // must use the same KV head and exact leading-plane independence.
    {
        const SdpaReferenceCase mixed = sdpa_oracle::make_mixed_gqa_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> expected = evaluate(mixed);
        const auto output = [&](std::size_t plane, std::size_t row,
                                std::size_t head, std::size_t feature) {
            return expected[sdpa_oracle::output_offset(
                    mixed, plane, row, head, feature)];
        };
        bool paired_difference = false;
        bool group_difference = false;
        for (std::size_t plane = 0; plane < 2; ++plane) {
            for (std::size_t row = 0; row < mixed.rows; ++row) {
                for (std::size_t feature = 0; feature < mixed.head_dim;
                     ++feature) {
                    paired_difference = paired_difference
                            || output(plane, row, 0, feature).bits
                                       != output(plane, row, 1, feature).bits;
                    group_difference = group_difference
                            || output(plane, row, 1, feature).bits
                                       != output(plane, row, 2, feature).bits;
                }
            }
        }
        sdpa_require(report, paired_difference,
                     "mixed GQA fixture must retain query-head distinctions");
        sdpa_require(report, group_difference,
                     "GQA mapping must distinguish KV head groups");
    }

    // The output is independent of masked future tokens, unread capacity tail,
    // and physical padding.  These comparisons are deliberately bitwise, not
    // tolerance comparisons.
    {
        SdpaReferenceCase future = sdpa_oracle::make_future_token_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> before = evaluate(future);
        for (std::size_t head = 0; head < future.hq; ++head) {
            for (std::size_t feature = 0; feature < future.head_dim; ++feature) {
                future.k_bits[sdpa_oracle::kv_offset(
                        future, 0, 0, 5, feature)] =
                        sdpa_oracle::value_bits(
                                future.data_type, 123.0 + feature);
                future.v_bits[sdpa_oracle::kv_offset(
                        future, 0, 0, 5, feature)] =
                        sdpa_oracle::value_bits(
                                future.data_type, -77.0 - feature);
            }
        }
        const std::vector<SdpaReferenceValue> after = evaluate(future);
        sdpa_require(report, sdpa_all_equal(before, after),
                     "future-token perturbation must not alter causal output");

        SdpaReferenceCase tail = sdpa_oracle::make_capacity_tail_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> tail_before = evaluate(tail);
        for (std::size_t token = tail.length; token < tail.capacity; ++token) {
            for (std::size_t feature = 0; feature < tail.head_dim; ++feature) {
                tail.k_bits[sdpa_oracle::kv_offset(
                        tail, 0, 0, token, feature)] =
                        sdpa_oracle::value_bits(
                                tail.data_type,
                                std::numeric_limits<double>::quiet_NaN());
                tail.v_bits[sdpa_oracle::kv_offset(
                        tail, 0, 0, token, feature)] =
                        sdpa_oracle::value_bits(
                                tail.data_type,
                                std::numeric_limits<double>::infinity());
            }
        }
        sdpa_require(report, sdpa_all_equal(tail_before, evaluate(tail)),
                     "capacity-tail perturbation must not be read");

        SdpaReferenceCase padding = sdpa_oracle::make_padding_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> padding_before = evaluate(padding);
        padding.physical_padding_salt ^= 0xFFFF;
        sdpa_require(report, sdpa_all_equal(padding_before, evaluate(padding)),
                     "physical padding perturbation must be ignored");
    }

    // Causal prefix, exact capacity, nonzero a, generic L<a+R, and row-size
    // boundaries are all pinned by the generated matrix.
    {
        const SdpaReferenceCase prefix = sdpa_oracle::make_causal_prefix_case(
                iom::DataType::BF16);
        sdpa_require(report, prefix.length < prefix.capacity,
                     "causal fixture must leave capacity tail unread");
        sdpa_require(report, prefix.length < prefix.position + prefix.rows,
                     "causal fixture must cover generic L<a+R");
        const SdpaReferenceCase exact = sdpa_oracle::make_exact_capacity_case(
                iom::DataType::BF16);
        sdpa_require(report, exact.length == exact.capacity,
                     "exact-capacity fixture must use L=C");
        sdpa_require(report, exact.rows <= exact.capacity - exact.position,
                     "exact-capacity fixture must satisfy checked R<=C-a");
        sdpa_require(report, kSdpaBoundaryRows.size() == 4,
                     "boundary matrix must include R=1,15,16,17");
    }

    // Formed-score special policy and included-only V semantics.
    {
        const SdpaReferenceCase special = sdpa_oracle::make_special_scores_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> expected = evaluate(special);
        const std::size_t row0 = sdpa_oracle::output_offset(special, 0, 0, 0, 0);
        const std::size_t row1 = sdpa_oracle::output_offset(special, 0, 1, 0, 0);
        const std::size_t row2 = sdpa_oracle::output_offset(special, 0, 2, 0, 0);
        sdpa_require(report,
                     expected[row0].bits
                             == sdpa_oracle::value_bits(
                                     iom::DataType::BF16, 3.0),
                     "+inf scores must receive equal probability");
        sdpa_require(report,
                     expected[row1].value_class
                             == SdpaReferenceClass::quiet_nan,
                     "all -inf scores must produce NaN output");
        sdpa_require(report,
                     expected[row2].value_class
                             == SdpaReferenceClass::quiet_nan,
                     "a formed NaN score must poison its row");

        const SdpaReferenceCase zero_nonfinite =
                sdpa_oracle::make_included_zero_nonfinite_case(
                        iom::DataType::BF16);
        sdpa_require(report,
                     evaluate(zero_nonfinite)[0].value_class
                             == SdpaReferenceClass::quiet_nan,
                     "included +0 probability times infinity must poison output");
    }

    // DAZ boundaries are exact and local to BF16 matrix operands.  In
    // particular, the smallest BF16 normal survives, while a subnormal
    // probability is not flushed before its BF16 conversion.
    {
        const float tiny = std::ldexp(1.0f, -127);
        const float normal = std::ldexp(1.0f, -126);
        sdpa_require(report, sdpa_oracle::matrix_daz(+0.0f) == +0.0f,
                     "DAZ must preserve +0");
        sdpa_require(report, sdpa_oracle::matrix_daz(-0.0f) == -0.0f
                             && std::signbit(sdpa_oracle::matrix_daz(-0.0f)),
                     "DAZ must preserve signed zero");
        sdpa_require(report,
                     sdpa_oracle::matrix_daz(tiny) == +0.0f,
                     "DAZ must flush +2^-127");
        sdpa_require(report,
                     sdpa_oracle::matrix_daz(-tiny) == -0.0f
                             && std::signbit(sdpa_oracle::matrix_daz(-tiny)),
                     "DAZ must flush -2^-127 with its sign");
        sdpa_require(report,
                     sdpa_oracle::matrix_daz(normal) == normal,
                     "DAZ must preserve the 2^-126 boundary");
        sdpa_require(report,
                     std::isinf(sdpa_oracle::matrix_daz(
                             std::numeric_limits<float>::infinity())),
                     "DAZ must preserve infinity");
        sdpa_require(report,
                     std::isnan(sdpa_oracle::matrix_daz(
                             std::numeric_limits<float>::quiet_NaN())),
                     "DAZ must preserve NaN");

        const std::vector<SdpaReferenceValue> underflow = evaluate(
                sdpa_oracle::make_gradual_underflow_case());
        sdpa_require(report, underflow.size() == 1,
                     "underflow falsifier must produce one output");
        sdpa_require(report,
                     underflow[0].bits == 0x0040,
                     "2^-127 FP32 product must survive final BF16 RNE");
    }

    // Cached one-row calls must reproduce the corresponding rows of full
    // causal recomputation exactly, including GQA head selection.
    {
        const SdpaReferenceCase full = sdpa_oracle::make_cached_incremental_case(
                iom::DataType::BF16);
        const std::vector<SdpaReferenceValue> full_values = evaluate(full);
        for (std::size_t row = 0; row < full.rows; ++row) {
            const SdpaReferenceCase incremental =
                    sdpa_incremental_slice(full, row);
            const std::vector<SdpaReferenceValue> one = evaluate(incremental);
            for (std::size_t head = 0; head < full.hq; ++head) {
                for (std::size_t feature = 0;
                     feature < full.head_dim; ++feature) {
                    const SdpaReferenceValue expected = full_values[
                            sdpa_oracle::output_offset(
                                    full, 0, row, head, feature)];
                    const SdpaReferenceValue actual = one[
                            sdpa_oracle::output_offset(
                                    incremental, 0, 0, head, feature)];
                    sdpa_require(report,
                                 expected.bits == actual.bits
                                         && expected.value_class
                                                    == actual.value_class,
                                 "cached incremental row differs from full causal row");
                }
            }
        }
    }

    // Exact final-store RNE and the fixed finite acceptance margin are pinned
    // before any backend is measured.  A deliberately perturbed expectation
    // outside both the two-code and absolute margin must fail.
    {
        const iom::DataType type = iom::DataType::BF16;
        const std::uint64_t one = sdpa_oracle::value_bits(type, 1.0);
        const std::uint64_t upper = sdpa_oracle::value_bits(type, 1.0 + 0x1p-7);
        const SdpaReferenceCase tie = sdpa_oracle::make_case(
                SdpaReferenceCaseKind::mixed_gqa, type, {}, 1, 1, 1, 1, 2,
                1, 2, {0.0}, {0.0, 0.0}, {1.0, 1.0 + 0x1p-7});
        const std::vector<SdpaReferenceValue> tie_value = evaluate(tie);
        sdpa_require(report, tie_value[0].bits == one,
                     "final BF16 RNE must choose the even midpoint encoding");
        (void)upper;
        SdpaReferenceValue expected{one, SdpaReferenceClass::finite};
        sdpa_require(report,
                     matches(type, sdpa_oracle::value_bits(type, 1.0 + 0x1p-7),
                             expected),
                     "one BF16 destination ULP must remain within tolerance");
        sdpa_require(report,
                     !matches(type,
                              sdpa_oracle::value_bits(type, 1.0 + 3.0 * 0x1p-7),
                              expected),
                     "a deliberately perturbed expected value must fail");
    }

    // Malformed structure fails closed before any arithmetic and never gets a
    // chance to coerce an inapplicable leaf to BF16.
    {
        SdpaReferenceCase malformed = sdpa_oracle::make_ones_case(
                iom::DataType::BF16);
        malformed.length = malformed.capacity + 1;
        bool rejected = false;
        try {
            (void)evaluate(malformed);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        sdpa_require(report, rejected,
                     "an L>C fixture must be rejected before evaluation");
        sdpa_require(report, sdpa_data_type_classification(
                                      iom::DataType::I8)
                                 == SdpaDataTypeClass::inapplicable,
                     "integer leaves must remain inapplicable");
    }

    return report;
}

}  // namespace iom_conformance

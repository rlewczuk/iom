#pragma once

// Independent RoPE oracle (change 006-tinyllama / 06-rotary-position-encoding / 02).
//
// This header deliberately does not include, call, or mirror any production
// codec, address calculation, kernel, or backend capability declaration.  It
// owns the raw named-format codec, the split-half equation, fixed comparison
// policy, deterministic fixtures, and the small callable success-path runner
// consumed by later backend conformance leaves.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "backend/backend_conformance_oracle.hpp"

namespace iom_conformance::rope_reference {

// ---------------------------------------------------------------------------
// Public fixture and expected-value API.
// ---------------------------------------------------------------------------

enum class RopeReferenceClass : std::uint8_t {
    finite,
    positive_zero,
    negative_zero,
    positive_infinity,
    negative_infinity,
    quiet_nan,
};


enum class RopeReferenceCaseKind : std::uint8_t {
    position_zero,
    theta_one,
    theta_10000,
    run_1,
    run_15,
    run_16,
    run_17,
    adjacent_pair,
    reset_position,
    split_prefill_decode,
    leading_plane_head_mapping,
    padding_isolation,
    intermediate_bf16_rounding,
    wrong_rne,
    wide_magnitude_boundaries,
};

inline constexpr std::string_view rope_reference_case_kind_name(
        RopeReferenceCaseKind kind) noexcept {
    switch (kind) {
        case RopeReferenceCaseKind::position_zero: return "position_zero";
        case RopeReferenceCaseKind::theta_one: return "theta_one";
        case RopeReferenceCaseKind::theta_10000: return "theta_10000";
        case RopeReferenceCaseKind::run_1: return "run_1";
        case RopeReferenceCaseKind::run_15: return "run_15";
        case RopeReferenceCaseKind::run_16: return "run_16";
        case RopeReferenceCaseKind::run_17: return "run_17";
        case RopeReferenceCaseKind::adjacent_pair: return "adjacent_pair";
        case RopeReferenceCaseKind::reset_position: return "reset_position";
        case RopeReferenceCaseKind::split_prefill_decode:
            return "split_prefill_decode";
        case RopeReferenceCaseKind::leading_plane_head_mapping:
            return "leading_plane_head_mapping";
        case RopeReferenceCaseKind::padding_isolation:
            return "padding_isolation";
        case RopeReferenceCaseKind::intermediate_bf16_rounding:
            return "intermediate_bf16_rounding";
        case RopeReferenceCaseKind::wide_magnitude_boundaries:
            return "wide_magnitude_boundaries";
    }
    return "unknown";
}

enum class RopeReferenceTensorRole : std::uint8_t { query, key };

// One logical `[..., H, R, D]` input.  `leading_dimensions` describes the
// complete leading tuple of the logical view; `leading_coordinates` contains
// one coordinate tuple per independent plane in row-major plane order.  Raw
// element encodings are likewise row-major `(plane, head, row, feature)`.
// `input_leading_order` and `output_leading_order`, when nonempty, describe
// the order passed to TensorView::permute for the corresponding view.  They
// make the logical view independent of its owner/native plane layout.
struct RopeReferenceCase {
    RopeReferenceCaseKind kind = RopeReferenceCaseKind::position_zero;
    RopeReferenceTensorRole role = RopeReferenceTensorRole::query;
    iom::DataType data_type = iom::DataType::F32;
    iom::QuantizationFormat quantization = iom::QuantizationFormat::NONE;
    std::vector<std::size_t> leading_dimensions;
    std::vector<std::vector<std::size_t>> leading_coordinates;
    std::size_t H = 0;
    std::size_t R = 0;
    std::size_t D = 0;
    std::size_t a = 0;
    double theta = 10000.0;
    std::vector<std::uint64_t> input_bits;
    std::vector<std::size_t> input_leading_order;
    std::vector<std::size_t> output_leading_order;
    std::uint64_t padding_salt = 0;
    std::string label;
};

// One expected destination element.  `decoded` and `value_class` are obtained
// by decoding `bits` with this header's codec.  The source pair is retained so
// the fixed pair-norm slack is applied without reconstructing input metadata at
// measurement time.
struct RopeReferenceValue {
    std::uint64_t bits = 0;
    double decoded = 0.0;
    RopeReferenceClass value_class = RopeReferenceClass::finite;
    double source_first = 0.0;
    double source_second = 0.0;
    double pair_norm = 0.0;
};

enum class RopeDataTypeClassification : std::uint8_t {
    applicable,
    unsupported,
};

inline constexpr std::array<iom::DataType, 9> kRopeApplicableDataTypes = {
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

inline constexpr std::array<iom::DataType, 14> kRopeUnsupportedDataTypes = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F8_E8M0,
};

inline constexpr std::array<iom::DataType, 9> kRopeCpuExpectedSupported =
        kRopeApplicableDataTypes;
inline constexpr std::array<iom::DataType, 9> kRopeCudaExpectedSupported =
        kRopeApplicableDataTypes;
inline constexpr std::array<iom::DataType, 9> kRopeRocmExpectedSupported =
        kRopeApplicableDataTypes;
inline constexpr std::array<iom::DataType, 8> kRopeSyclExpectedSupported = {
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN,
        iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::BF16,
        iom::DataType::F32,
};
inline constexpr std::array<iom::DataType, 2> kRopeTtnnExpectedSupported = {
        iom::DataType::BF16,
        iom::DataType::F32,
};

static_assert(kRopeApplicableDataTypes.size() + kRopeUnsupportedDataTypes.size()
                      == 23,
              "RoPE classification must cover all 23 data-type leaves");

inline constexpr RopeDataTypeClassification rope_data_type_classification(
        iom::DataType type) noexcept {
    switch (type) {
        case iom::DataType::F4_E2M1:
        case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2:
        case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2:
        case iom::DataType::F16:
        case iom::DataType::BF16:
        case iom::DataType::F32:
        case iom::DataType::F64:
            return RopeDataTypeClassification::applicable;
        case iom::DataType::BOOL:
        case iom::DataType::I2:
        case iom::DataType::U2:
        case iom::DataType::I4:
        case iom::DataType::U4:
        case iom::DataType::I8:
        case iom::DataType::U8:
        case iom::DataType::I16:
        case iom::DataType::U16:
        case iom::DataType::I32:
        case iom::DataType::U32:
        case iom::DataType::I64:
        case iom::DataType::U64:
        case iom::DataType::F8_E8M0:
            return RopeDataTypeClassification::unsupported;
    }
    return RopeDataTypeClassification::unsupported;
}

namespace detail {


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

inline constexpr std::uint64_t exponent_mask(
        const FloatFormat& format) noexcept {
    return (std::uint64_t{1} << format.exponent_bits) - 1;
}

inline constexpr std::uint64_t mantissa_mask(
        const FloatFormat& format) noexcept {
    return (std::uint64_t{1} << format.mantissa_bits) - 1;
}

inline constexpr std::uint64_t width_mask(const FloatFormat& format) noexcept {
    return format.bits >= 64 ? ~std::uint64_t{0}
                             : (std::uint64_t{1} << format.bits) - 1;
}

inline constexpr std::uint64_t finite_exponent_max(
        const FloatFormat& format) noexcept {
    return format.finite_only && format.exponent_bits < 4
                   ? exponent_mask(format)
                   : exponent_mask(format) - 1;
}

inline constexpr bool has_nan_encoding(const FloatFormat& format) noexcept {
    // F4/F6 finite-only leaves use every exponent as a finite value.  E4M3FN
    // has four exponent bits and reserves all-ones for NaNs but no infinity.
    return !format.finite_only || format.exponent_bits >= 4;
}

template <typename Carrier>
inline Carrier decode_carrier(
        std::uint64_t raw, const FloatFormat& format) noexcept {
    if (format.bits == 0) {
        return Carrier{0};
    }
    raw &= width_mask(format);
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
            ? std::ldexp(static_cast<Carrier>(
                                 (std::uint64_t{1} << format.mantissa_bits)
                                         + mantissa),
                         static_cast<int>(exponent) - format.bias
                                 - static_cast<int>(format.mantissa_bits))
            : std::ldexp(static_cast<Carrier>(mantissa),
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
                   && (truncated & 1U)))
                      ? std::uint64_t{1}
                      : std::uint64_t{0});
}

template <typename Carrier>
inline std::uint64_t encode_carrier(
        Carrier value, const FloatFormat& format) noexcept {
    if (format.bits == 0) {
        return 0;
    }
    const std::uint64_t exponent_max = exponent_mask(format);
    const std::uint64_t fraction_max = mantissa_mask(format);
    const std::uint64_t finite_max = finite_exponent_max(format);
    if (std::isnan(value)) {
        const std::uint64_t exponent = has_nan_encoding(format)
                ? exponent_max
                : finite_max;
        const std::uint64_t mantissa = format.has_infinity
                ? std::uint64_t{1} << (format.mantissa_bits - 1)
                : fraction_max;
        return (exponent << format.mantissa_bits) | mantissa;
    }
    const std::uint64_t sign = std::signbit(value) ? std::uint64_t{1} : 0;
    value = std::fabs(value);
    const auto pack = [sign, &format](
                              std::uint64_t exponent,
                              std::uint64_t mantissa) noexcept {
        return (sign << (format.exponent_bits + format.mantissa_bits))
               | (exponent << format.mantissa_bits) | mantissa;
    };
    const std::uint64_t overflow = format.has_infinity
            ? pack(exponent_max, 0)
            : pack(finite_max, fraction_max);
    if (std::isinf(value)) {
        return overflow;
    }
    if (value == 0) {
        return sign << (format.exponent_bits + format.mantissa_bits);
    }
    int exponent = 0;
    (void)std::frexp(value, &exponent);
    --exponent;
    const int subnormal_exponent =
            1 - format.bias - static_cast<int>(format.mantissa_bits);
    const int maximum_exponent = static_cast<int>(finite_max) - format.bias;
    if (exponent < subnormal_exponent + static_cast<int>(format.mantissa_bits)) {
        const std::uint64_t quantized = round_to_nearest_even(
                std::ldexp(value, -subnormal_exponent));
        return quantized == (std::uint64_t{1} << format.mantissa_bits)
                ? pack(1, 0)
                : pack(0, quantized);
    }
    if (exponent > maximum_exponent) {
        return overflow;
    }
    std::uint64_t fraction = round_to_nearest_even(
            std::ldexp(value, static_cast<int>(format.mantissa_bits) - exponent)
            - static_cast<Carrier>(std::uint64_t{1} << format.mantissa_bits));
    if (fraction == (std::uint64_t{1} << format.mantissa_bits)) {
        ++exponent;
        fraction = 0;
    }
    if (exponent > maximum_exponent) {
        return overflow;
    }
    return pack(static_cast<std::uint64_t>(exponent + format.bias), fraction);
}

inline float decode_f32(iom::DataType type, std::uint64_t raw) noexcept {
    return decode_carrier<float>(raw, float_format(type));
}
inline double decode_f64(std::uint64_t raw) noexcept {
    return decode_carrier<double>(raw, float_format(iom::DataType::F64));
}
inline std::uint64_t encode_f32(iom::DataType type, float value) noexcept {
    return encode_carrier<float>(value, float_format(type));
}
inline std::uint64_t encode_f64(double value) noexcept {
    return encode_carrier<double>(value, float_format(iom::DataType::F64));
}

inline double decode_as_double(
        iom::DataType type, std::uint64_t raw) noexcept {
    return type == iom::DataType::F64
                   ? decode_f64(raw)
                   : static_cast<double>(decode_f32(type, raw));
}

inline RopeReferenceClass classify(float value) noexcept {
    if (std::isnan(value)) return RopeReferenceClass::quiet_nan;
    if (std::isinf(value)) {
        return std::signbit(value) ? RopeReferenceClass::negative_infinity
                                   : RopeReferenceClass::positive_infinity;
    }
    if (value == 0) {
        return std::signbit(value) ? RopeReferenceClass::negative_zero
                                   : RopeReferenceClass::positive_zero;
    }
    return RopeReferenceClass::finite;
}
inline RopeReferenceClass classify(double value) noexcept {
    if (std::isnan(value)) return RopeReferenceClass::quiet_nan;
    if (std::isinf(value)) {
        return std::signbit(value) ? RopeReferenceClass::negative_infinity
                                   : RopeReferenceClass::positive_infinity;
    }
    if (value == 0) {
        return std::signbit(value) ? RopeReferenceClass::negative_zero
                                   : RopeReferenceClass::positive_zero;
    }
    return RopeReferenceClass::finite;
}

inline bool represents_infinity(iom::DataType type) noexcept {
    return float_format(type).has_infinity;
}
inline bool represents_nan(iom::DataType type) noexcept {
    return has_nan_encoding(float_format(type));
}

inline std::uint64_t value_bits(iom::DataType type, double value) noexcept {
    return type == iom::DataType::F64
                   ? encode_f64(value)
                   : encode_f32(type, static_cast<float>(value));
}


inline double adjacent_destination_ulp(
        iom::DataType type, std::uint64_t bits, double decoded) noexcept {
    const FloatFormat format = float_format(type);
    const std::uint64_t mask = width_mask(format);
    std::uint64_t adjacent = bits & mask;
    const auto is_valid_neighbor = [&](std::uint64_t candidate) noexcept {
        if (candidate > mask) return false;
        const double value = decode_as_double(type, candidate);
        return std::isfinite(value)
               && std::signbit(value) == std::signbit(decoded);
    };
    if (decoded > 0) {
        if (adjacent < mask && is_valid_neighbor(adjacent + 1)) {
            ++adjacent;
        } else if (adjacent > 0 && is_valid_neighbor(adjacent - 1)) {
            --adjacent;
        }
    } else if (decoded < 0) {
        if (adjacent > 0 && is_valid_neighbor(adjacent - 1)) {
            --adjacent;
        } else if (adjacent < mask && is_valid_neighbor(adjacent + 1)) {
            ++adjacent;
        }
    } else {
        adjacent = 1;
    }
    return std::fabs(decode_as_double(type, adjacent) - decoded);
}

inline std::size_t checked_add(
        std::size_t lhs, std::size_t rhs, std::string_view what) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(std::string(what));
    }
    return lhs + rhs;
}
inline std::size_t checked_mul(
        std::size_t lhs, std::size_t rhs, std::string_view what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(what));
    }
    return lhs * rhs;
}

inline std::size_t leaf_bits(iom::DataType type) {
    const FloatFormat format = float_format(type);
    if (format.bits == 0) {
        throw std::invalid_argument("RoPE reference requires a floating leaf");
    }
    return format.bits;
}

inline std::size_t plane_count(const RopeReferenceCase& reference_case) {
    std::size_t count = 1;
    for (const std::size_t dimension : reference_case.leading_dimensions) {
        if (dimension == 0) {
            throw std::invalid_argument(
                    "RoPE reference leading dimensions must be nonzero");
        }
        count = checked_mul(count, dimension, "RoPE reference plane count overflows");
    }
    return count;
}

inline std::size_t element_count(const RopeReferenceCase& reference_case) {
    std::size_t count = plane_count(reference_case);
    count = checked_mul(count, reference_case.H,
                       "RoPE reference element count overflows");
    count = checked_mul(count, reference_case.R,
                       "RoPE reference element count overflows");
    return checked_mul(count, reference_case.D,
                      "RoPE reference element count overflows");
}

inline void validate_permutation(
        std::span<const std::size_t> order, std::size_t rank,
        std::string_view name) {
    if (order.empty()) return;
    if (order.size() != rank) {
        throw std::invalid_argument(std::string(name) + " has the wrong rank");
    }
    std::vector<bool> seen(rank, false);
    for (const std::size_t axis : order) {
        if (axis >= rank || seen[axis]) {
            throw std::invalid_argument(std::string(name) + " is not a permutation");
        }
        seen[axis] = true;
    }
}

inline void validate_case(const RopeReferenceCase& reference_case) {
    if (reference_case.quantization != iom::QuantizationFormat::NONE) {
        throw std::invalid_argument(
                "RoPE reference admits only QuantizationFormat::NONE");
    }
    if (reference_case.leading_dimensions.size() > 5) {
        throw std::invalid_argument("RoPE reference rank exceeds eight");
    }
    if (reference_case.H == 0 || reference_case.R == 0
        || reference_case.D == 0 || (reference_case.D & 1U) != 0) {
        throw std::invalid_argument(
                "RoPE reference requires nonzero H/R and positive even D");
    }
    const std::size_t planes = plane_count(reference_case);
    const std::size_t elements = element_count(reference_case);
    if (reference_case.input_bits.size() != elements) {
        throw std::invalid_argument(
                "RoPE reference input bits do not match the logical shape");
    }
    if (!reference_case.leading_coordinates.empty()) {
        if (reference_case.leading_coordinates.size() != planes) {
            throw std::invalid_argument(
                    "RoPE reference leading coordinates do not match planes");
        }
        for (const auto& coordinate : reference_case.leading_coordinates) {
            if (coordinate.size() != reference_case.leading_dimensions.size()) {
                throw std::invalid_argument(
                        "RoPE reference coordinate has the wrong rank");
            }
            for (std::size_t axis = 0;
                 axis < coordinate.size(); ++axis) {
                if (coordinate[axis] >= reference_case.leading_dimensions[axis]) {
                    throw std::invalid_argument(
                            "RoPE reference coordinate is out of range");
                }
            }
        }
    }
    validate_permutation(reference_case.input_leading_order,
                         reference_case.leading_dimensions.size(),
                         "RoPE input permutation");
    validate_permutation(reference_case.output_leading_order,
                         reference_case.leading_dimensions.size(),
                         "RoPE output permutation");
    if (!std::isfinite(reference_case.theta)
        || reference_case.theta < 1.0
        || reference_case.theta
                   > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument("RoPE reference theta is outside [1,float_max]");
    }
    const std::size_t last = checked_add(
            reference_case.a, reference_case.R - 1,
            "RoPE reference absolute position overflows");
    if (last > ((std::size_t{1} << 24) - 1)) {
        throw std::invalid_argument("RoPE reference position exceeds 2^24-1");
    }
    if (rope_data_type_classification(reference_case.data_type)
        == RopeDataTypeClassification::applicable) {
        (void)leaf_bits(reference_case.data_type);
    }
}


inline std::vector<std::byte> pack_bits(
        iom::DataType type, std::span<const std::uint64_t> values) {
    const std::size_t bits = leaf_bits(type);
    const std::size_t total_bits = checked_mul(
            bits, values.size(), "RoPE reference byte count overflows");
    const std::size_t bytes = checked_add(
            total_bits, 7, "RoPE reference byte count overflows")
            / 8;
    std::vector<std::byte> result(bytes, std::byte{0});
    auto* raw = reinterpret_cast<unsigned char*>(result.data());
    for (std::size_t element = 0; element < values.size(); ++element) {
        const std::uint64_t value = values[element];
        for (std::size_t bit = 0; bit < bits; ++bit) {
            const std::size_t offset = checked_add(
                    checked_mul(element, bits,
                                "RoPE reference bit offset overflows"),
                    bit, "RoPE reference bit offset overflows");
            const unsigned char mask =
                    static_cast<unsigned char>(1u << (offset % 8));
            if ((value >> bit) & 1U) {
                raw[offset / 8] |= mask;
            }
        }
    }
    return result;
}

inline std::vector<std::uint64_t> unpack_bits(
        iom::DataType type, std::span<const std::byte> bytes,
        std::size_t count) {
    const std::size_t bits = leaf_bits(type);
    const std::size_t expected_bytes =
            checked_add(checked_mul(bits, count,
                                    "RoPE reference byte count overflows"),
                        7, "RoPE reference byte count overflows")
            / 8;
    if (bytes.size() != expected_bytes) {
        throw std::invalid_argument("RoPE reference readback has the wrong size");
    }
    const auto* raw = reinterpret_cast<const unsigned char*>(bytes.data());
    std::vector<std::uint64_t> result(count, 0);
    for (std::size_t element = 0; element < count; ++element) {
        for (std::size_t bit = 0; bit < bits; ++bit) {
            const std::size_t offset = checked_add(
                    checked_mul(element, bits,
                                "RoPE reference bit offset overflows"),
                    bit, "RoPE reference bit offset overflows");
            if (raw[offset / 8] & static_cast<unsigned char>(1u << (offset % 8))) {
                result[element] |= std::uint64_t{1} << bit;
            }
        }
    }
    return result;
}
inline void poison_unused_tail_bits(
        iom::DataType type, std::size_t count, std::vector<std::byte>& bytes,
        std::uint64_t salt) {
    const std::size_t bits = leaf_bits(type);
    const std::size_t total_bits = checked_mul(
            bits, count, "RoPE reference tail-bit count overflows");
    if (total_bits % 8 == 0) return;
    auto* raw = reinterpret_cast<unsigned char*>(bytes.data());
    const std::size_t storage_bits = checked_mul(
            bytes.size(), std::size_t{8},
            "RoPE reference tail storage-bit count overflows");
    for (std::size_t bit = total_bits; bit < storage_bits; ++bit) {
        const unsigned char mask =
                static_cast<unsigned char>(1u << (bit % 8));
        if ((::iom_conformance::splitmix64(salt ^ bit) & 1u) != 0) {
            raw[bit / 8] |= mask;
        } else {
            raw[bit / 8] &= static_cast<unsigned char>(~mask);
        }
    }
}

inline std::vector<std::byte> encode_view_storage_with_poison(
        const iom::TensorView& view, const iom::TensorSpec& owner,
        std::span<const std::byte> logical, std::uint64_t salt) {
    if (logical.size() != view.spec().logical_nbytes()) {
        throw std::invalid_argument(
                "RoPE reference storage image has the wrong logical size");
    }
    std::vector<std::byte> storage =
            ::iom_conformance::encode_standard_tiled_storage(owner);
    ::iom_conformance::apply_standard_tiled_view(
            view, owner, logical, storage);
    const std::size_t bits = ::iom_conformance::bits_of(owner.data_type);
    std::vector<unsigned char> touched(storage.size(), 0);
    const std::size_t count = view.spec().shape.element_count();
    for (std::size_t linear = 0; linear < count; ++linear) {
        const std::size_t slot = ::iom_conformance::standard_layout_view_slot(
                view, owner, linear);
        const std::size_t bit_offset = checked_mul(
                slot, bits, "RoPE reference storage bit offset overflows");
        for (std::size_t bit = 0; bit < bits; ++bit) {
            const std::size_t absolute = checked_add(
                    bit_offset, bit,
                    "RoPE reference storage bit offset overflows");
            touched[absolute / 8] |=
                    static_cast<unsigned char>(1u << (absolute % 8));
        }
    }
    for (std::size_t byte = 0; byte < storage.size(); ++byte) {
        const unsigned char current =
                std::to_integer<unsigned char>(storage[byte]);
        const unsigned char salt_byte = static_cast<unsigned char>(
                ::iom_conformance::splitmix64(salt ^ byte));
        storage[byte] = std::byte{
                static_cast<unsigned char>(
                        (current & touched[byte])
                        | (salt_byte & static_cast<unsigned char>(
                                  ~touched[byte])))};
    }
    return storage;
}

inline std::vector<std::size_t> owner_dimensions(
        const RopeReferenceCase& reference_case,
        std::span<const std::size_t> order) {
    const std::size_t leading = reference_case.leading_dimensions.size();
    validate_permutation(order, leading, "RoPE owner permutation");
    std::vector<std::size_t> dimensions = reference_case.leading_dimensions;
    for (std::size_t view_axis = 0; view_axis < order.size(); ++view_axis) {
        dimensions[order[view_axis]] =
                reference_case.leading_dimensions[view_axis];
    }
    dimensions.push_back(reference_case.H);
    dimensions.push_back(reference_case.R);
    dimensions.push_back(reference_case.D);
    return dimensions;
}

inline iom::TensorView transformed_view(
        iom::Tensor& owner, std::span<const std::size_t> order) {
    const std::span<const std::size_t> dimensions =
            owner.view().spec().shape.dimensions();
    if (dimensions.size() < 3) {
        throw std::invalid_argument("RoPE owner rank is too small");
    }
    const std::size_t leading = dimensions.size() - 3;
    validate_permutation(order, leading, "RoPE transformed view permutation");
    if (order.empty()) {
        return owner.view();
    }
    std::vector<std::size_t> full_order(order.begin(), order.end());
    full_order.push_back(leading);
    return owner.view().permute(full_order);
}



// ---------------------------------------------------------------------------
// Deterministic fixture content and pinned analytic constants.
// ---------------------------------------------------------------------------

// Decimal constants were generated offline with mpmath 1.3.0 at 100 decimal
// digits and checked in here.  The C++ oracle still computes its own float or
// double trig values; these constants are only analytic/golden self-check pins.
struct PinnedCoefficient {
    std::size_t position;
    std::size_t j;
    long double angle;
    long double sine;
    long double cosine;
};

inline constexpr std::array<PinnedCoefficient, 8> kPinnedTheta10000D8 = {{
        {15, 0, 15.0L, 0.650287840157116820705983335299L,
         -0.759687912858821318991774529190L},
        {15, 1, 1.5L, 0.997494986604054430941723371142L,
         0.070737201667702910088189851434L},
        {15, 2, 0.15L, 0.149438132473599224466751208753L,
         0.988771077936042410144626239953L},
        {15, 3, 0.015L, 0.014999437506328090398760147649L,
         0.999887502109359173380991012719L},
        {16, 0, 16.0L, -0.287903316665065300850746619962L,
         -0.957659480323384675715387842464L},
        {16, 1, 1.6L, 0.999573603041505164550603816501L,
         -0.029199522301288815068657861933L},
        {16, 2, 0.16L, 0.159318206614245963311463159686L,
         0.987227283375626917333146566478L},
        {16, 3, 0.016L, 0.015999317342071413405852864079L,
         0.999872002730643365084299481132L},
}};

inline constexpr std::array<long double, 2> kPinnedThetaOneD2PositionOne = {
        0.540302305868139717400936607443L,
        0.841470984807896506652502321630L,
};

inline constexpr std::array<double, 12> kFixtureValues = {
        1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
        3.0, -3.0, 4.0, -4.0, 6.0, -6.0,
};

inline std::string leaf_name(iom::DataType type) {
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
        default: return "unsupported";
    }
}

inline std::size_t fixture_plane_count(
        std::span<const std::size_t> leading_dimensions) {
    std::size_t planes = 1;
    for (const std::size_t dimension : leading_dimensions) {
        planes = checked_mul(planes, dimension,
                             "RoPE fixture plane count overflows");
    }
    return planes;
}

inline std::vector<std::vector<std::size_t>> make_leading_coordinates(
        std::span<const std::size_t> dimensions) {
    const std::size_t planes = fixture_plane_count(dimensions);
    std::vector<std::vector<std::size_t>> coordinates;
    coordinates.reserve(planes);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        std::size_t remaining = plane;
        std::vector<std::size_t> coordinate(dimensions.size(), 0);
        for (std::size_t axis = dimensions.size(); axis-- > 0;) {
            coordinate[axis] = dimensions[axis] == 0
                    ? 0
                    : remaining % dimensions[axis];
            if (dimensions[axis] != 0) remaining /= dimensions[axis];
        }
        coordinates.push_back(std::move(coordinate));
    }
    return coordinates;
}

inline RopeReferenceCase make_pattern_case(
        RopeReferenceCaseKind kind, iom::DataType type,
        std::vector<std::size_t> leading_dimensions, std::size_t heads,
        std::size_t rows, std::size_t width, std::size_t position,
        double theta, RopeReferenceTensorRole role = RopeReferenceTensorRole::query,
        std::uint64_t salt = 0) {
    RopeReferenceCase reference_case;
    reference_case.kind = kind;
    reference_case.role = role;
    reference_case.data_type = type;
    reference_case.leading_dimensions = std::move(leading_dimensions);
    reference_case.leading_coordinates = make_leading_coordinates(
            reference_case.leading_dimensions);
    reference_case.H = heads;
    reference_case.R = rows;
    reference_case.D = width;
    reference_case.a = position;
    reference_case.theta = theta;
    reference_case.padding_salt = salt;
    reference_case.label = std::string(rope_reference_case_kind_name(kind));
    reference_case.label += "/" + leaf_name(type);
    const std::size_t planes = plane_count(reference_case);
    reference_case.input_bits.reserve(element_count(reference_case));
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < heads; ++head) {
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t feature = 0; feature < width; ++feature) {
                    const std::size_t pattern_index =
                            (feature * 5 + row * 3 + head * 7
                             + plane * 11 + static_cast<std::size_t>(salt))
                            % kFixtureValues.size();
                    reference_case.input_bits.push_back(
                            value_bits(type, kFixtureValues[pattern_index]));
                }
            }
        }
    }
    return reference_case;
}

inline RopeReferenceCase make_position_zero_case(iom::DataType type) {
    RopeReferenceCase reference_case = make_pattern_case(
            RopeReferenceCaseKind::position_zero, type, {2}, 2, 1, 8, 0, 1.0,
            RopeReferenceTensorRole::query, 1);
    const auto format = float_format(type);
    const std::uint64_t sign =
            std::uint64_t{1} << (format.exponent_bits + format.mantissa_bits);
    const std::uint64_t exponent = exponent_mask(format);
    const std::uint64_t infinity = exponent << format.mantissa_bits;
    const std::uint64_t nan_payload =
            infinity | (format.mantissa_bits == 0
                               ? std::uint64_t{1}
                               : std::uint64_t{1} << (format.mantissa_bits - 1));
    if (!reference_case.input_bits.empty()) reference_case.input_bits[0] = 0;
    if (reference_case.input_bits.size() > 1) reference_case.input_bits[1] = sign;
    if (reference_case.input_bits.size() > 2 && represents_infinity(type)) {
        reference_case.input_bits[2] = infinity;
        reference_case.input_bits[3] = infinity | sign;
    }
    if (reference_case.input_bits.size() > 4 && represents_nan(type)) {
        reference_case.input_bits[4] = nan_payload;
        reference_case.input_bits[5] = nan_payload | 1U;
    }
    return reference_case;
}

inline RopeReferenceCase make_adjacent_case(iom::DataType type) {
    RopeReferenceCase reference_case = make_pattern_case(
            RopeReferenceCaseKind::adjacent_pair, type, {}, 1, 1, 4, 1, 1.0,
            RopeReferenceTensorRole::query, 0);
    for (std::size_t index = 0; index < 4; ++index) {
        reference_case.input_bits[index] = value_bits(
                type, static_cast<double>(index + 1));
    }
    return reference_case;
}

inline RopeReferenceCase make_reset_case(iom::DataType type) {
    return make_pattern_case(RopeReferenceCaseKind::reset_position, type, {2},
                             2, 17, 4, 15, 1.0,
                             RopeReferenceTensorRole::query, 4);
}

inline RopeReferenceCase make_split_case(iom::DataType type) {
    return make_pattern_case(RopeReferenceCaseKind::split_prefill_decode, type,
                             {2, 3}, 3, 17, 8, 15, 10000.0,
                             RopeReferenceTensorRole::query, 5);
}

inline RopeReferenceCase make_bf16_rounding_case(iom::DataType type) {
    RopeReferenceCase reference_case = make_pattern_case(
            RopeReferenceCaseKind::intermediate_bf16_rounding, type, {}, 2, 3,
            8, 17, 10000.0, RopeReferenceTensorRole::key, 9);
    if (type == iom::DataType::BF16) {
        // Values adjacent to binary32/BF16 half-way boundaries make a double
        // intermediate path observably different from the specified FP32 path.
        const std::array<std::uint64_t, 16> raw = {
                0x3f81, 0xbf81, 0x3f83, 0xbf83,
                0x3f85, 0xbf85, 0x3f87, 0xbf87,
                0x3f89, 0xbf89, 0x3f8b, 0xbf8b,
                0x3f8d, 0xbf8d, 0x3f8f, 0xbf8f,
        };
        for (std::size_t index = 0; index < reference_case.input_bits.size();
             ++index) {
            reference_case.input_bits[index] = raw[index % raw.size()];
        }
        // At position 17, (0x3ff2, 0x3f4e) and j=0 straddle a BF16
        // destination halfway boundary only when the prescribed FP32
        // products are rounded separately.
        reference_case.input_bits[0] = 0x3ff2;
        reference_case.input_bits[4] = 0x3f4e;
    }
    return reference_case;
}

inline RopeReferenceCase make_wrong_rne_case(iom::DataType type) {
    RopeReferenceCase reference_case = make_pattern_case(
            RopeReferenceCaseKind::wrong_rne, type, {}, 1, 1, 2, 1, 1.0,
            RopeReferenceTensorRole::query, 0);
    reference_case.input_bits[0] = value_bits(type, 1.0);
    reference_case.input_bits[1] = value_bits(type, 0.0);
    return reference_case;
}

inline RopeReferenceCase make_wide_magnitude_case(iom::DataType type) {
    RopeReferenceCase reference_case = make_pattern_case(
            RopeReferenceCaseKind::wide_magnitude_boundaries, type, {}, 1, 2,
            4, 3, 10000.0, RopeReferenceTensorRole::key, 14);
    const detail::FloatFormat format = detail::float_format(type);
    const std::uint64_t mantissa_mask =
            (std::uint64_t{1} << format.mantissa_bits) - 1;
    const std::uint64_t max_exponent =
            (std::uint64_t{1} << format.exponent_bits) - 1;
    const std::uint64_t finite_exponent =
            format.has_infinity ? max_exponent - 1 : max_exponent;
    const std::uint64_t max_finite =
            (finite_exponent << format.mantissa_bits) | mantissa_mask;
    const std::uint64_t min_subnormal = 1;
    reference_case.input_bits[0] = max_finite;
    reference_case.input_bits[1] = min_subnormal;
    reference_case.input_bits[2] = max_finite;
    reference_case.input_bits[3] = min_subnormal;
    reference_case.input_bits[4] = max_finite | (std::uint64_t{1}
                                                  << (format.exponent_bits
                                                      + format.mantissa_bits));
    reference_case.input_bits[5] = min_subnormal;
    reference_case.input_bits[6] = reference_case.input_bits[4];
    reference_case.input_bits[7] = min_subnormal;
    return reference_case;
}

inline std::vector<RopeReferenceCase> rope_reference_cases(
        iom::DataType type) {
    std::vector<RopeReferenceCase> cases;
    if (rope_data_type_classification(type)
        != RopeDataTypeClassification::applicable) {
        return cases;
    }
    cases.push_back(make_position_zero_case(type));
    cases.push_back(make_pattern_case(RopeReferenceCaseKind::theta_one, type,
                                      {2}, 3, 3, 6, 15, 1.0,
                                      RopeReferenceTensorRole::query, 2));
    RopeReferenceCase theta = make_pattern_case(
            RopeReferenceCaseKind::theta_10000, type, {2, 3}, 2, 17, 8, 15,
            10000.0, RopeReferenceTensorRole::query, 3);
    theta.input_leading_order = {1, 0};
    theta.output_leading_order = {1, 0};
    cases.push_back(std::move(theta));
    cases.push_back(make_pattern_case(RopeReferenceCaseKind::run_1, type, {}, 2,
                                      1, 18, 1, 10000.0,
                                      RopeReferenceTensorRole::key, 5));
    cases.push_back(make_pattern_case(RopeReferenceCaseKind::run_15, type, {2},
                                      2, 15, 10, 15, 10000.0,
                                      RopeReferenceTensorRole::query, 6));
    cases.push_back(make_pattern_case(RopeReferenceCaseKind::run_16, type, {2},
                                      2, 16, 6, 16, 10000.0,
                                      RopeReferenceTensorRole::key, 7));
    cases.push_back(make_pattern_case(RopeReferenceCaseKind::run_17, type,
                                      {2, 2}, 3, 17, 18, 15, 10000.0,
                                      RopeReferenceTensorRole::query, 8));
    cases.push_back(make_adjacent_case(type));
    cases.push_back(make_reset_case(type));
    cases.push_back(make_split_case(type));
    RopeReferenceCase mapping = make_pattern_case(
            RopeReferenceCaseKind::leading_plane_head_mapping, type, {2, 3}, 3,
            2, 6, 16, 1.0, RopeReferenceTensorRole::query, 12);
    mapping.input_leading_order = {1, 0};
    mapping.output_leading_order = {1, 0};
    cases.push_back(std::move(mapping));
    RopeReferenceCase padding = make_pattern_case(
            RopeReferenceCaseKind::padding_isolation, type, {2, 3}, 2, 16, 18,
            17, 10000.0, RopeReferenceTensorRole::key, 13);
    padding.padding_salt = 0xA5A5A5A5u;
    cases.push_back(std::move(padding));
    cases.push_back(make_wide_magnitude_case(type));
    cases.push_back(make_bf16_rounding_case(type));
    cases.push_back(make_wrong_rne_case(type));
    return cases;
}

inline std::string rope_reference_case_label(
        const RopeReferenceCase& reference_case) {
    if (!reference_case.label.empty()) return reference_case.label;
    std::string result(
            rope_reference_case_kind_name(reference_case.kind));
    result += "/" + leaf_name(reference_case.data_type);
    return result;
}

// ---------------------------------------------------------------------------
// Independent split-half evaluation and fixed comparison policy.
// ---------------------------------------------------------------------------

inline std::vector<RopeReferenceValue> evaluate(
        const RopeReferenceCase& reference_case) {
    detail::validate_case(reference_case);
    if (rope_data_type_classification(reference_case.data_type)
        != RopeDataTypeClassification::applicable) {
        return {};
    }
    const detail::FloatFormat format =
            detail::float_format(reference_case.data_type);
    const std::uint64_t mask = detail::width_mask(format);
    const std::size_t count = detail::element_count(reference_case);
    std::vector<RopeReferenceValue> result(count);
    const std::size_t planes = detail::plane_count(reference_case);
    const std::size_t half = reference_case.D / 2;
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < reference_case.H; ++head) {
            for (std::size_t row = 0; row < reference_case.R; ++row) {
                const std::size_t position = detail::checked_add(
                        reference_case.a, row,
                        "RoPE reference absolute position overflows");
                std::size_t row_base = detail::checked_mul(
                        plane, reference_case.H,
                        "RoPE reference index overflows");
                row_base = detail::checked_add(
                        row_base, head, "RoPE reference index overflows");
                row_base = detail::checked_mul(
                        row_base, reference_case.R,
                        "RoPE reference index overflows");
                row_base = detail::checked_add(
                        row_base, row, "RoPE reference index overflows");
                row_base = detail::checked_mul(
                        row_base, reference_case.D,
                        "RoPE reference index overflows");
                for (std::size_t j = 0; j < half; ++j) {
                    const std::size_t first_index = detail::checked_add(
                            row_base, j, "RoPE reference index overflows");
                    const std::size_t second_index = detail::checked_add(
                            row_base, j + half,
                            "RoPE reference index overflows");
                    const std::uint64_t first_raw =
                            reference_case.input_bits[first_index] & mask;
                    const std::uint64_t second_raw =
                            reference_case.input_bits[second_index] & mask;
                    const double first_decoded = detail::decode_as_double(
                            reference_case.data_type, first_raw);
                    const double second_decoded = detail::decode_as_double(
                            reference_case.data_type, second_raw);
                    const double pair_norm =
                            std::fabs(first_decoded) + std::fabs(second_decoded);
                    const auto set_value = [&](std::size_t index,
                                               std::uint64_t bits) {
                        RopeReferenceValue& value = result[index];
                        value.bits = bits & mask;
                        value.decoded = detail::decode_as_double(
                                reference_case.data_type, value.bits);
                        value.value_class = reference_case.data_type
                                        == iom::DataType::F64
                                ? detail::classify(
                                          detail::decode_f64(value.bits))
                                : detail::classify(detail::decode_f32(
                                          reference_case.data_type, value.bits));
                        value.source_first = first_decoded;
                        value.source_second = second_decoded;
                        value.pair_norm = pair_norm;
                    };
                    // This branch intentionally copies raw payloads without
                    // decoding or canonicalizing them for the destination.
                    if (position == 0) {
                        set_value(first_index, first_raw);
                        set_value(second_index, second_raw);
                        continue;
                    }

                    if (reference_case.data_type == iom::DataType::F64) {
                        const double exponent =
                                (-2.0 * static_cast<double>(j))
                                / static_cast<double>(reference_case.D);
                        const double frequency =
                                std::pow(reference_case.theta, exponent);
                        const double angle =
                                static_cast<double>(position) * frequency;
                        const double sine = std::sin(angle);
                        const double cosine = std::cos(angle);
                        const volatile double first_product =
                                first_decoded * cosine;
                        const volatile double second_product =
                                second_decoded * sine;
                        const volatile double first_output =
                                first_product - second_product;
                        const volatile double second_cosine_product =
                                second_decoded * cosine;
                        const volatile double first_sine_product =
                                first_decoded * sine;
                        const volatile double second_output =
                                second_cosine_product + first_sine_product;
                        set_value(first_index,
                                  detail::encode_f64(first_output));
                        set_value(second_index,
                                  detail::encode_f64(second_output));
                    } else {
                        const float first = detail::decode_f32(
                                reference_case.data_type, first_raw);
                        const float second = detail::decode_f32(
                                reference_case.data_type, second_raw);
                        const float theta =
                                static_cast<float>(reference_case.theta);
                        const float exponent = static_cast<float>(
                                (-2.0 * static_cast<double>(j))
                                / static_cast<double>(reference_case.D));
                        const float frequency = std::pow(theta, exponent);
                        const float position_f = static_cast<float>(position);
                        const float angle = position_f * frequency;
                        const float sine = std::sin(angle);
                        const float cosine = std::cos(angle);
                        const volatile float first_product = first * cosine;
                        const volatile float second_product = second * sine;
                        const volatile float first_output =
                                first_product - second_product;
                        const volatile float second_cosine_product =
                                second * cosine;
                        const volatile float first_sine_product = first * sine;
                        const volatile float second_output =
                                second_cosine_product + first_sine_product;
                        set_value(first_index,
                                  detail::encode_f32(
                                          reference_case.data_type,
                                          first_output));
                        set_value(second_index,
                                  detail::encode_f32(
                                          reference_case.data_type,
                                          second_output));
                    }
                }
            }
        }
    }
    return result;
}

inline bool matches(
        iom::DataType type, std::uint64_t actual_bits,
        const RopeReferenceValue& expected) noexcept {
    if (rope_data_type_classification(type)
        != RopeDataTypeClassification::applicable) {
        return false;
    }
    const detail::FloatFormat format = detail::float_format(type);
    const std::uint64_t actual_pattern =
            actual_bits & detail::width_mask(format);
    const RopeReferenceClass actual_class = type == iom::DataType::F64
            ? detail::classify(detail::decode_f64(actual_pattern))
            : detail::classify(detail::decode_f32(type, actual_pattern));
    if (actual_class != expected.value_class) return false;
    if (expected.value_class != RopeReferenceClass::finite) return true;
    const double actual = detail::decode_as_double(type, actual_pattern);
    const double reference = detail::decode_as_double(type, expected.bits);
    if (!std::isfinite(actual) || !std::isfinite(reference)) return false;
    const double destination_ulp =
            detail::adjacent_destination_ulp(type, expected.bits, reference);
    const double pair_norm = expected.pair_norm;
    const double slack = type == iom::DataType::F64
            ? std::ldexp(pair_norm, -44)
            : std::ldexp(pair_norm, -9);
    return std::fabs(actual - reference) <= destination_ulp + slack;
}

// ---------------------------------------------------------------------------
// Callable success-path runner.  Admission/rejection/lifetime cases belong to
// the separate contract leaf; this function intentionally executes only the
// declared successful support span.
// ---------------------------------------------------------------------------

inline void run_rope_case(
        iom::Device& candidate, const RopeReferenceCase& reference_case,
        iom_conformance::AcceleratorStorageOracle* oracle) {
    const std::size_t count = detail::element_count(reference_case);
    std::vector<std::byte> source_bytes = detail::pack_bits(
            reference_case.data_type, reference_case.input_bits);
    detail::poison_unused_tail_bits(
            reference_case.data_type, count, source_bytes,
            reference_case.padding_salt);
    auto owner_dimensions = detail::owner_dimensions(
            reference_case, reference_case.input_leading_order);
    const iom::TensorSpec source_spec{
            iom::TensorShape{owner_dimensions}, reference_case.data_type,
            reference_case.quantization};
    owner_dimensions = detail::owner_dimensions(
            reference_case, reference_case.output_leading_order);
    const iom::TensorSpec destination_spec{
            iom::TensorShape{owner_dimensions}, reference_case.data_type,
            reference_case.quantization};
    auto source = candidate.create_tensor(source_spec);
    auto destination = candidate.create_tensor(destination_spec);
    iom::TensorView source_view = detail::transformed_view(
            *source, reference_case.input_leading_order);
    iom::TensorView destination_view = detail::transformed_view(
            *destination, reference_case.output_leading_order);
    if (oracle != nullptr) {
        oracle->set_owner_spec(source_spec);
        const std::vector<std::byte> source_storage =
                detail::encode_view_storage_with_poison(
                        source_view, source_spec, source_bytes,
                        reference_case.padding_salt);
        oracle->seed(source->view(), source_storage);
    } else {
        ::iom_conformance::copy_from_host(source_view, source_bytes);
    }
    std::vector<std::byte> poison(destination_view.spec().logical_nbytes(),
                                  std::byte{0xA5});
    detail::poison_unused_tail_bits(
            reference_case.data_type, count, poison,
            reference_case.padding_salt ^ 0xD1CEB00Cull);
    if (oracle != nullptr) {
        oracle->set_owner_spec(destination_spec);
        const std::vector<std::byte> destination_storage =
                detail::encode_view_storage_with_poison(
                        destination_view, destination_spec, poison,
                        reference_case.padding_salt ^ 0xD1CEB00Cull);
        oracle->seed(destination->view(), destination_storage);
    } else {
        ::iom_conformance::copy_from_host(destination_view, poison);
    }
    auto queue = candidate.create_ops();
    const iom::WorkspaceRequirements requirements =
            queue->rope_workspace_requirements(
                    source_view, destination_view, reference_case.a,
                    reference_case.theta);
    const iom::WorkspaceRequirements expected_requirements{0, 1};
    INFO(reference_case.label);
    REQUIRE_MESSAGE(requirements == expected_requirements,
                    "rope query must return {0,1}");
    const iom::oid token = queue->rope(
            source_view, destination_view, reference_case.a,
            reference_case.theta);
    REQUIRE_MESSAGE(iom::oid_is_token(token),
                    "rope submission did not return a token");
    queue->wait(token);
    std::vector<std::byte> observed_bytes(
            destination_view.spec().logical_nbytes(), std::byte{0xA5});
    ::iom_conformance::copy_to_host(destination_view, observed_bytes);
    const std::vector<std::uint64_t> observed = detail::unpack_bits(
            reference_case.data_type, observed_bytes, count);
    const std::vector<RopeReferenceValue> expected = evaluate(reference_case);
    REQUIRE_EQ(expected.size(), observed.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        INFO(reference_case.label);
        INFO(index);
        const std::size_t row =
                (index / reference_case.D) % reference_case.R;
        if (reference_case.a == 0 && row == 0) {
            CHECK_EQ(observed[index], expected[index].bits);
        } else {
            CHECK(matches(
                    reference_case.data_type, observed[index], expected[index]));
        }
    }
}

inline RopeReferenceCase slice_rows(
        const RopeReferenceCase& source, std::size_t first,
        std::size_t count, std::size_t position) {
    detail::validate_case(source);
    if (first > source.R || count > source.R - first || count == 0) {
        throw std::invalid_argument("RoPE split slice is outside the row range");
    }
    RopeReferenceCase result = source;
    result.R = count;
    result.a = position;
    result.label = source.label + "/split" + std::to_string(position);
    const std::size_t planes = detail::plane_count(source);
    const std::size_t source_head_width = detail::checked_mul(
            source.R, source.D, "RoPE split head width overflows");
    const std::size_t source_plane_width = detail::checked_mul(
            source.H, source_head_width, "RoPE split plane width overflows");
    const std::size_t slice_width = detail::checked_mul(
            count, source.D, "RoPE split slice width overflows");
    result.input_bits.clear();
    const std::size_t result_plane_width = detail::checked_mul(
            source.H, slice_width, "RoPE split result plane width overflows");
    result.input_bits.reserve(detail::checked_mul(
            planes, result_plane_width, "RoPE split input size overflows"));
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t head = 0; head < source.H; ++head) {
            const std::size_t begin = detail::checked_add(
                    detail::checked_mul(
                            plane, source_plane_width,
                            "RoPE split input offset overflows"),
                    detail::checked_add(
                            detail::checked_mul(
                                    head, source_head_width,
                                    "RoPE split input offset overflows"),
                            detail::checked_mul(
                                    first, source.D,
                                    "RoPE split input offset overflows"),
                            "RoPE split input offset overflows"),
                    "RoPE split input offset overflows");
            result.input_bits.insert(
                    result.input_bits.end(),
                    source.input_bits.begin()
                            + static_cast<std::ptrdiff_t>(begin),
                    source.input_bits.begin()
                            + static_cast<std::ptrdiff_t>(begin + slice_width));
        }
    }
    return result;
}

inline void run_rope_conformance(
        iom::Device& candidate,
        std::span<const iom::DataType> expected_supported,
        iom_conformance::AcceleratorStorageOracle* oracle) {
    for (const iom::DataType type : expected_supported) {
        REQUIRE_MESSAGE(
                rope_data_type_classification(type)
                        == RopeDataTypeClassification::applicable,
                "RoPE expected support contains an inapplicable data type");
        const std::vector<RopeReferenceCase> cases =
                rope_reference_cases(type);
        REQUIRE_MESSAGE(!cases.empty(),
                        "RoPE applicable data type has no reference cases");
        for (const RopeReferenceCase& reference_case : cases) {
            run_rope_case(candidate, reference_case, oracle);
            if (reference_case.kind
                == RopeReferenceCaseKind::split_prefill_decode) {
                // Compare one R=17 call with decode-style R=1 submissions at
                // positions 15, 16, and 17, then the remaining tail.  Every
                // call carries its own absolute position; nothing is reset or
                // inferred from queue/session state.
                run_rope_case(
                        candidate, slice_rows(reference_case, 0, 1, 15),
                        oracle);
                run_rope_case(
                        candidate, slice_rows(reference_case, 1, 1, 16),
                        oracle);
                run_rope_case(
                        candidate, slice_rows(reference_case, 2, 1, 17),
                        oracle);
                run_rope_case(
                        candidate,
                        slice_rows(reference_case, 3,
                                   reference_case.R - 3, 18),
                        oracle);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backend-free self-check report.  The focused target turns every failure into
// a doctest assertion; keeping the report in this header lets later leaves run
// the same oracle checks without constructing a Device.
// ---------------------------------------------------------------------------

struct RopeReferenceSelfCheckReport {
    std::size_t checks = 0;
    std::vector<std::string> failures;
    [[nodiscard]] bool ok() const noexcept { return failures.empty(); }
};

inline RopeReferenceSelfCheckReport rope_reference_self_check() {
    RopeReferenceSelfCheckReport report;
    const auto require = [&report](bool condition, std::string message) {
        ++report.checks;
        if (!condition) report.failures.push_back(std::move(message));
    };
    const auto occurrence = [](std::span<const iom::DataType> values,
                               iom::DataType type) {
        return static_cast<std::size_t>(std::count(values.begin(), values.end(), type));
    };

    require(kRopeApplicableDataTypes.size() == 9,
            "RoPE must have nine applicable leaves");
    require(kRopeUnsupportedDataTypes.size() == 14,
            "RoPE must have fourteen inapplicable leaves");
    for (const iom::DataType type : kRopeApplicableDataTypes) {
        require(rope_data_type_classification(type)
                        == RopeDataTypeClassification::applicable,
                leaf_name(type) + " classification is not applicable");
        require(occurrence(kRopeApplicableDataTypes, type) == 1
                        && occurrence(kRopeUnsupportedDataTypes, type) == 0,
                leaf_name(type) + " classification is duplicated");
        require(!rope_reference_cases(type).empty(),
                leaf_name(type) + " has no deterministic fixtures");
    }
    for (const iom::DataType type : kRopeUnsupportedDataTypes) {
        require(rope_data_type_classification(type)
                        == RopeDataTypeClassification::unsupported,
                "inapplicable leaf classified as applicable");
        require(occurrence(kRopeUnsupportedDataTypes, type) == 1
                        && occurrence(kRopeApplicableDataTypes, type) == 0,
                "inapplicable classification is duplicated");
        RopeReferenceCase unsupported;
        unsupported.data_type = type;
        unsupported.leading_dimensions = {};
        unsupported.leading_coordinates = { { } };
        unsupported.H = 1;
        unsupported.R = 1;
        unsupported.D = 2;
        unsupported.input_bits = {0, 0};
        require(evaluate(unsupported).empty(),
                "inapplicable leaf must not produce arithmetic");
    }

    // Position zero is a raw identity, including signed zeros and every
    // representable nonfinite payload present in each encoding.
    for (const iom::DataType type : kRopeApplicableDataTypes) {
        for (const RopeReferenceCase& reference_case : rope_reference_cases(type)) {
            const std::vector<RopeReferenceValue> expected = evaluate(reference_case);
            require(expected.size() == reference_case.input_bits.size(),
                    rope_reference_case_label(reference_case)
                            + " has one output per input element");
            for (std::size_t index = 0; index < expected.size(); ++index) {
                const std::size_t row =
                        (index / reference_case.D) % reference_case.R;
                if (reference_case.a == 0 && row == 0) {
                    require(expected[index].bits
                                    == reference_case.input_bits[index],
                            rope_reference_case_label(reference_case)
                                    + " rewrote a position-zero raw payload");
                }
            }
            for (const RopeReferenceValue& value : expected) {
                require(matches(type, value.bits, value),
                        rope_reference_case_label(reference_case)
                                + " does not match its own expectation");
            }
        }
    }

    // Pinned high-precision analytic coefficients and a golden unit vector.
    for (const PinnedCoefficient& coefficient : kPinnedTheta10000D8) {
        const float theta = 10000.0F;
        const float exponent = static_cast<float>(
                (-2.0 * static_cast<double>(coefficient.j)) / 8.0);
        const float angle = static_cast<float>(coefficient.position) 
                            * std::pow(theta, exponent);
        require(std::fabs(static_cast<long double>(std::sin(angle))
                          - coefficient.sine)
                            < 2.0e-6L,
                "pinned sine coefficient drifted");
        require(std::fabs(static_cast<long double>(std::cos(angle))
                          - coefficient.cosine)
                            < 2.0e-6L,
                "pinned cosine coefficient drifted");
    }
    {
        RopeReferenceCase golden = make_pattern_case(
                RopeReferenceCaseKind::theta_one, iom::DataType::F32, {}, 1, 1,
                2, 1, 1.0, RopeReferenceTensorRole::query, 0);
        golden.input_bits = {detail::value_bits(iom::DataType::F32, 1.0), 0};
        const auto values = evaluate(golden);
        require(std::fabs(values[0].decoded -
                          static_cast<double>(kPinnedThetaOneD2PositionOne[0]))
                            < 2.0e-6,
                "theta-one cosine golden value drifted");
        require(std::fabs(values[1].decoded -
                          static_cast<double>(kPinnedThetaOneD2PositionOne[1]))
                            < 2.0e-6,
                "theta-one sine golden value drifted");
    }

    // Adjacent-pair and reset-position substitutions must be observable.
    {
        const RopeReferenceCase adjacent = make_adjacent_case(iom::DataType::F32);
        const auto expected = evaluate(adjacent);
        RopeReferenceCase wrong = adjacent;
        wrong.input_bits = {detail::value_bits(iom::DataType::F32, 1.0),
                            detail::value_bits(iom::DataType::F32, 2.0),
                            detail::value_bits(iom::DataType::F32, 3.0),
                            detail::value_bits(iom::DataType::F32, 4.0)};
        const float sine = std::sin(1.0F);
        const float cosine = std::cos(1.0F);
        for (std::size_t pair = 0; pair < 2; ++pair) {
            const float first = static_cast<float>(pair * 2 + 1);
            const float second = static_cast<float>(pair * 2 + 2);
            wrong.input_bits[pair * 2] = detail::encode_f32(
                    iom::DataType::F32, first * cosine - second * sine);
            wrong.input_bits[pair * 2 + 1] = detail::encode_f32(
                    iom::DataType::F32, second * cosine + first * sine);
        }
        bool differs = false;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            differs = differs || expected[index].bits != wrong.input_bits[index];
        }
        require(differs, "adjacent-pair substitution was not falsified");
    }
    {
        const RopeReferenceCase reset = make_reset_case(iom::DataType::F32);
        const auto expected = evaluate(reset);
        RopeReferenceCase wrong = reset;
        wrong.a = 0;
        const auto reset_expected = evaluate(wrong);
        bool differs_after_first_row = false;
        const std::size_t row_width = reset.H * reset.D;
        for (std::size_t index = row_width; index < expected.size(); ++index) {
            differs_after_first_row = differs_after_first_row
                                      || expected[index].bits
                                                 != reset_expected[index].bits;
        }
        require(differs_after_first_row,
                "reset-position substitution was not falsified");
    }

    // Independent planes and head order are semantic, not cancellation-prone
    // metadata.  Swapping either mapping must reject at least one expected
    // element.
    {
        const RopeReferenceCase mapping = make_pattern_case(
                RopeReferenceCaseKind::leading_plane_head_mapping,
                iom::DataType::F32, {2, 3}, 3, 2, 6, 16, 1.0,
                RopeReferenceTensorRole::query, 12);
        const auto expected = evaluate(mapping);
        const std::size_t head_stride = mapping.R * mapping.D;
        const std::size_t plane_stride = mapping.H * head_stride;
        std::vector<RopeReferenceValue> wrong = expected;
        for (std::size_t plane = 0; plane < 2 * 3; ++plane) {
            for (std::size_t offset = 0; offset < head_stride; ++offset) {
                std::swap(wrong[plane * plane_stride + offset],
                          wrong[plane * plane_stride + head_stride + offset]);
            }
        }
        bool rejected_head_order = false;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            rejected_head_order = rejected_head_order
                                  || !matches(iom::DataType::F32,
                                               expected[index].bits, wrong[index]);
        }
        require(rejected_head_order,
                "head-order perturbation was not falsified");
        std::swap(wrong[0], wrong[plane_stride]);
        bool rejected_plane_order = false;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            rejected_plane_order = rejected_plane_order
                                   || !matches(iom::DataType::F32,
                                                expected[index].bits, wrong[index]);
        }
        require(rejected_plane_order,
                "leading-plane perturbation was not falsified");
    }

    // A continuous R=17 call and split prefill/decode slices must retain the
    // same absolute positions, including the 15/16/17 boundary.
    {
        const RopeReferenceCase full = make_split_case(iom::DataType::F32);
        const auto full_values = evaluate(full);
        const std::array<std::pair<std::size_t, std::size_t>, 2> slices = {{
                {0, 3},
                {3, 14},
        }};
        for (const auto& [first, count] : slices) {
            const RopeReferenceCase part = slice_rows(full, first, count,
                                                      full.a + first);
            const auto part_values = evaluate(part);
            const std::size_t part_width = part.H * part.R * part.D;
            const std::size_t full_row_width = full.H * full.R * full.D;
            const std::size_t planes = detail::plane_count(full);
            for (std::size_t plane = 0; plane < planes; ++plane) {
                const std::size_t full_plane = plane * full_row_width;
                const std::size_t part_plane = plane * part_width;
                for (std::size_t head = 0; head < full.H; ++head) {
                    for (std::size_t row = 0; row < count; ++row) {
                        for (std::size_t feature = 0; feature < full.D;
                             ++feature) {
                            const std::size_t full_index =
                                    full_plane + head * full.R * full.D
                                    + (first + row) * full.D + feature;
                            const std::size_t part_index =
                                    part_plane + head * part.R * part.D
                                    + row * part.D + feature;
                            require(full_values[full_index].bits
                                            == part_values[part_index].bits,
                                    "split prefill/decode changed a value");
                        }
                    }
                }
            }
        }
    }

    {
        const auto zero = evaluate(make_position_zero_case(iom::DataType::F32));
        bool saw_positive_infinity = false;
        bool saw_negative_infinity = false;
        for (const RopeReferenceValue& value : zero) {
            saw_positive_infinity = saw_positive_infinity
                                    || value.value_class
                                               == RopeReferenceClass::positive_infinity;
            saw_negative_infinity = saw_negative_infinity
                                    || value.value_class
                                               == RopeReferenceClass::negative_infinity;
        }
        require(saw_positive_infinity && saw_negative_infinity,
                "position-zero infinity classes were not retained");
    }

    // RNE: the exact halfway point above 1.0 in F16 rounds to the even 1.0
    // destination, while a half-up encoder would choose the next encoding.
    {
        const std::uint64_t tie = detail::encode_f32(
                iom::DataType::F16, 1.0F + 0x1p-11F);
        const std::uint64_t half_up = detail::encode_f32(
                iom::DataType::F16, 1.0F + 0x1p-10F);
        require(tie == 0x3c00 && half_up == 0x3c01,
                "round-to-nearest-even boundary is not pinned");
        require(tie != half_up, "wrong-RNE substitution was not falsified");
    }

    // Intermediate BF16 rounding: search only deterministic repository-owned
    // values and angles, then require one concrete mismatch against the wrong
    // double-intermediate variant.  This is a negative oracle check, never a
    // backend measurement.
    {
        const RopeReferenceCase fixture =
                make_bf16_rounding_case(iom::DataType::BF16);
        const auto expected = evaluate(fixture);
        bool differs = false;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const std::size_t feature = index % fixture.D;
            const std::size_t half = fixture.D / 2;
            const std::size_t pair_index = feature < half ? feature : feature - half;
            const std::size_t row = (index / fixture.D) % fixture.R;
            const std::size_t first_index =
                    (index / fixture.D) * fixture.D + pair_index;
            const std::size_t second_index = first_index + half;
            const float first = detail::decode_f32(
                    iom::DataType::BF16, fixture.input_bits[first_index]);
            const float second = detail::decode_f32(
                    iom::DataType::BF16, fixture.input_bits[second_index]);
            const double angle =
                    static_cast<double>(fixture.a + row)
                    * std::pow(fixture.theta,
                               (-2.0 * static_cast<double>(pair_index))
                                       / static_cast<double>(fixture.D));
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const double wrong = feature < half
                    ? static_cast<double>(first) * cosine
                              - static_cast<double>(second) * sine
                    : static_cast<double>(second) * cosine
                              + static_cast<double>(first) * sine;
            differs = differs
                      || detail::encode_f32(iom::DataType::BF16,
                                            static_cast<float>(wrong))
                                 != expected[index].bits;
        }
        require(differs,
                "intermediate-BF16-rounding substitution was not falsified");
    }

    // Fixed tolerance margins: exactly one destination ULP passes, two do not
    // when the pair norm is zero; the wide leaves use the same frozen rule.
    {
        RopeReferenceValue f32;
        f32.bits = 0x3f800000;
        f32.decoded = 1.0;
        f32.value_class = RopeReferenceClass::finite;
        require(matches(iom::DataType::F32, 0x3f800001, f32),
                "one F32 destination ULP must pass");
        require(!matches(iom::DataType::F32, 0x3f800002, f32),
                "two F32 destination ULPs must fail");
        RopeReferenceValue f64;
        f64.bits = 0x3ff0000000000000ull;
        f64.decoded = 1.0;
        f64.value_class = RopeReferenceClass::finite;
        require(matches(iom::DataType::F64, f64.bits + 1, f64),
                "one F64 destination ULP must pass");
        require(!matches(iom::DataType::F64, f64.bits + 2, f64),
                "two F64 destination ULPs must fail");
        RopeReferenceValue nan;
        nan.bits = detail::encode_f32(
                iom::DataType::F32, std::numeric_limits<float>::quiet_NaN());
        nan.value_class = RopeReferenceClass::quiet_nan;
        require(matches(iom::DataType::F32, nan.bits ^ 0x1234, nan),
                "NaN payloads must be ignored");
        RopeReferenceValue negative_zero;
        negative_zero.bits = 0x80000000;
        negative_zero.value_class = RopeReferenceClass::negative_zero;
        require(!matches(iom::DataType::F32, 0, negative_zero),
                "signed zero classes must not be conflated");
    }

    // Required semantic fixture categories remain visible in every applicable
    // leaf, and Q/K are represented as independent unequal-head invocations.
    for (const iom::DataType type : kRopeApplicableDataTypes) {
        const auto cases = rope_reference_cases(type);
        const auto has_kind = [&cases](RopeReferenceCaseKind kind) {
            return std::any_of(cases.begin(), cases.end(),
                               [kind](const RopeReferenceCase& value) {
                                   return value.kind == kind;
                               });
        };
        for (const RopeReferenceCaseKind kind : {
                     RopeReferenceCaseKind::position_zero,
                     RopeReferenceCaseKind::theta_one,
                     RopeReferenceCaseKind::theta_10000,
                     RopeReferenceCaseKind::run_1,
                     RopeReferenceCaseKind::run_15,
                     RopeReferenceCaseKind::run_16,
                     RopeReferenceCaseKind::run_17,
                     RopeReferenceCaseKind::adjacent_pair,
                     RopeReferenceCaseKind::reset_position,
                     RopeReferenceCaseKind::split_prefill_decode,
                     RopeReferenceCaseKind::leading_plane_head_mapping,
                     RopeReferenceCaseKind::padding_isolation,
                     RopeReferenceCaseKind::intermediate_bf16_rounding,
                     RopeReferenceCaseKind::wrong_rne,
                     RopeReferenceCaseKind::wide_magnitude_boundaries,
             }) {
            require(has_kind(kind), leaf_name(type) + " is missing a fixture");
        }
        bool query = false;
        bool key = false;
        bool unequal_heads = false;
        for (const RopeReferenceCase& reference_case : cases) {
            query = query || reference_case.role == RopeReferenceTensorRole::query;
            key = key || reference_case.role == RopeReferenceTensorRole::key;
            if (reference_case.role == RopeReferenceTensorRole::query
                && reference_case.H != 2) {
                unequal_heads = true;
            }
        }
        require(query && key && unequal_heads,
                leaf_name(type) + " lacks independent unequal-head Q/K fixtures");
    }

    return report;
}
}  // namespace detail
using detail::rope_reference_cases;
using detail::evaluate;
using detail::matches;

inline void run_rope_conformance(
        iom::Device& candidate,
        std::span<const iom::DataType> expected_supported) {
    detail::run_rope_conformance(candidate, expected_supported, nullptr);
}

inline void run_rope_conformance(
        iom::Device& candidate,
        std::span<const iom::DataType> expected_supported,
        iom_conformance::AcceleratorStorageOracle* oracle) {
    detail::run_rope_conformance(candidate, expected_supported, oracle);
}

inline void run_rope_conformance(
        iom::Device& candidate,
        std::span<const iom::DataType> expected_supported,
        iom_conformance::AcceleratorStorageOracle& oracle) {
    detail::run_rope_conformance(candidate, expected_supported, &oracle);
}

using RopeReferenceSelfCheckReport = detail::RopeReferenceSelfCheckReport;
using detail::rope_reference_self_check;

}  // namespace iom_conformance::rope_reference

#pragma once

// Backend-neutral SiLU conformance for the frozen DeviceOps::silu ABI.
//
// This header owns the independent named-format codec, stable scalar oracle,
// fixed comparison policy, deterministic fixtures, capability matrix, and the
// observable query/admission/lifetime scenarios.  Drivers provide only their
// device triple, the explicit leaves their port queues, an optional native
// storage observer, and (when the backend has one) an accepted-failure seam.
// The suite never switches on BackendKind and never uses a production codec,
// SiLU implementation, address helper, or capability report as its oracle.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "backend/backend_conformance_oracle.hpp"
#include "iom/device.hpp"
#include "iom/iom.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {

// ---------------------------------------------------------------------------
// Explicit semantic and backend capability matrices.
// ---------------------------------------------------------------------------

inline constexpr std::array<iom::DataType, 9> kSiluApplicableDataTypes = {
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

inline constexpr std::array<iom::DataType, 14> kSiluInapplicableDataTypes = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F8_E8M0,
};

inline constexpr std::array<iom::DataType, 9> kSiluCpuExpectedSupported =
        kSiluApplicableDataTypes;
inline constexpr std::array<iom::DataType, 9> kSiluCudaExpectedSupported =
        kSiluApplicableDataTypes;
inline constexpr std::array<iom::DataType, 9> kSiluRocmExpectedSupported =
        kSiluApplicableDataTypes;
inline constexpr std::array<iom::DataType, 8> kSiluSyclExpectedSupported = {
        iom::DataType::F4_E2M1,
        iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN,
        iom::DataType::F8_E5M2,
        iom::DataType::F16,
        iom::DataType::BF16,
        iom::DataType::F32,
};

// An empty span is the explicit declaration of a port that has not landed yet.
// The shared suite observes capability rejection only in that configuration;
// a later backend port changes only its driver's span to the corresponding
// expected matrix above.
inline constexpr std::span<const iom::DataType> kNoSiluSpan{};

static_assert(kSiluApplicableDataTypes.size()
                      + kSiluInapplicableDataTypes.size()
              == 23,
              "SiLU classification must cover all 23 data-type leaves");

enum class SiluDataTypeClassification : std::uint8_t {
    applicable,
    inapplicable,
};

inline constexpr SiluDataTypeClassification silu_data_type_classification(
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
            return SiluDataTypeClassification::applicable;
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
            return SiluDataTypeClassification::inapplicable;
    }
    return SiluDataTypeClassification::inapplicable;
}

// ---------------------------------------------------------------------------
// Independent scalar/raw reference.
// ---------------------------------------------------------------------------

namespace silu_oracle {

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

inline constexpr std::uint64_t width_mask(
        const FloatFormat& format) noexcept {
    return format.bits >= 64 ? ~std::uint64_t{0}
                             : (std::uint64_t{1} << format.bits) - 1;
}

inline constexpr std::uint64_t exponent_mask(
        const FloatFormat& format) noexcept {
    return (std::uint64_t{1} << format.exponent_bits) - 1;
}

inline constexpr std::uint64_t mantissa_mask(
        const FloatFormat& format) noexcept {
    return (std::uint64_t{1} << format.mantissa_bits) - 1;
}

inline constexpr std::uint64_t finite_exponent_max(
        const FloatFormat& format) noexcept {
    return format.finite_only && format.exponent_bits < 4
                   ? exponent_mask(format)
                   : exponent_mask(format) - 1;
}

inline constexpr bool has_nan_encoding(
        const FloatFormat& format) noexcept {
    // E4M3FN reserves the all-ones exponent for NaN; F4/F6 consume every
    // exponent as a finite value.
    return !format.finite_only || format.exponent_bits >= 4;
}

inline constexpr bool represents_nan(iom::DataType type) noexcept {
    return has_nan_encoding(float_format(type));
}

inline constexpr bool represents_infinity(iom::DataType type) noexcept {
    return float_format(type).has_infinity;
}

inline constexpr std::size_t leaf_bits(iom::DataType type) {
    const FloatFormat format = float_format(type);
    if (format.bits == 0) {
        throw std::invalid_argument("SiLU oracle requires an applicable float leaf");
    }
    return format.bits;
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
            ? std::ldexp(
                      static_cast<Carrier>(
                              (std::uint64_t{1} << format.mantissa_bits)
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
    const std::uint64_t truncated = static_cast<std::uint64_t>(integral);
    const Carrier remainder = value - integral;
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
        // NaN payload and sign are outside the contract. Canonicalize to a
        // positive quiet NaN where the named format has one, otherwise use its
        // finite saturation value.
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
    const int maximum_exponent =
            static_cast<int>(finite_max) - format.bias;
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
            std::ldexp(
                    value,
                    static_cast<int>(format.mantissa_bits) - exponent)
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

enum class SiluReferenceClass : std::uint8_t {
    finite,
    positive_zero,
    negative_zero,
    positive_infinity,
    negative_infinity,
    quiet_nan,
};

inline constexpr std::string_view class_name(
        SiluReferenceClass value_class) noexcept {
    switch (value_class) {
        case SiluReferenceClass::finite: return "finite";
        case SiluReferenceClass::positive_zero: return "+0";
        case SiluReferenceClass::negative_zero: return "-0";
        case SiluReferenceClass::positive_infinity: return "+inf";
        case SiluReferenceClass::negative_infinity: return "-inf";
        case SiluReferenceClass::quiet_nan: return "nan";
    }
    return "unknown";
}

template <typename Carrier>
inline SiluReferenceClass classify(Carrier value) noexcept {
    if (std::isnan(value)) return SiluReferenceClass::quiet_nan;
    if (std::isinf(value)) {
        return std::signbit(value) ? SiluReferenceClass::negative_infinity
                                   : SiluReferenceClass::positive_infinity;
    }
    if (value == 0) {
        return std::signbit(value) ? SiluReferenceClass::negative_zero
                                   : SiluReferenceClass::positive_zero;
    }
    return SiluReferenceClass::finite;
}

inline SiluReferenceClass class_of_bits(
        iom::DataType type, std::uint64_t bits) noexcept {
    return type == iom::DataType::F64
                   ? classify(decode_f64(bits))
                   : classify(decode_f32(type, bits));
}

// Stable finite SiLU in the required accumulator domain.  The assignments in
// the negative branch intentionally preserve the written left-associated
// ((x*t)*t)/(1+t*t) order, including F32 -104 and F64 -746 tails.
template <typename Carrier>
inline Carrier stable_silu(Carrier x) noexcept {
    if (std::isnan(x)) return std::numeric_limits<Carrier>::quiet_NaN();
    if (std::isinf(x)) {
        return std::signbit(x) ? static_cast<Carrier>(-0.0)
                               : std::numeric_limits<Carrier>::infinity();
    }
    if (x == 0) return x;
    if (x < 0) {
        const Carrier half = x / static_cast<Carrier>(2);
        const Carrier t = std::exp(half);
        const Carrier xt = x * t;
        const Carrier numerator = xt * t;
        const Carrier tt = t * t;
        const Carrier denominator = static_cast<Carrier>(1) + tt;
        return numerator / denominator;
    }
    const Carrier negated = -x;
    const Carrier exponential = std::exp(negated);
    const Carrier denominator = static_cast<Carrier>(1) + exponential;
    return x / denominator;
}

inline std::uint64_t value_bits(
        iom::DataType type, double value) noexcept {
    return type == iom::DataType::F64
                   ? encode_f64(value)
                   : encode_f32(type, static_cast<float>(value));
}

inline double leaf_max_finite(iom::DataType type) noexcept {
    const FloatFormat format = float_format(type);
    const std::uint64_t raw =
            (finite_exponent_max(format) << format.mantissa_bits)
            | mantissa_mask(format);
    return decode_as_double(type, raw);
}

inline double leaf_min_positive(iom::DataType type) noexcept {
    return decode_as_double(type, 1);
}

inline double silu_value(iom::DataType type, std::uint64_t input_bits) noexcept {
    if (type == iom::DataType::F64) {
        const double x = decode_f64(input_bits);
        return stable_silu(x);
    }
    const float x = decode_f32(type, input_bits);
    return static_cast<double>(stable_silu(x));
}

struct SiluReferenceValue {
    std::uint64_t bits = 0;
    SiluReferenceClass value_class = SiluReferenceClass::finite;
};

inline SiluReferenceValue evaluate_value(
        iom::DataType type, std::uint64_t input_bits) noexcept {
    std::uint64_t bits = type == iom::DataType::F64
            ? encode_f64(stable_silu(decode_f64(input_bits)))
            : encode_f32(
                      type,
                      stable_silu(decode_f32(type, input_bits)));
    const SiluReferenceClass value_class = class_of_bits(type, bits);
    if (value_class == SiluReferenceClass::positive_zero
            || value_class == SiluReferenceClass::negative_zero) {
        // A zero-class expectation is compared exactly, so it must not be an
        // artifact of the reference's own carrier. `silu(2^-24)` is
        // `2^-25 * (1 + 2^-25)`, which sits just above the F16 rounding tie at
        // `2^-25`; a binary32 carrier cannot represent that correction and
        // resolves the minimum F16 input to `+0`, while the widest carrier the
        // suite has rounds it to the minimum subnormal. Every `+0`/`-0` input,
        // and every leaf whose tie the widest carrier also cannot resolve,
        // keeps its carrier-independent zero, so only a carrier-dependent zero
        // is replaced here and the fixture is then compared by sign and the
        // declared ULP ceiling instead of by an exact zero class.
        const FloatFormat format = float_format(type);
        bits = encode_carrier<long double>(
                stable_silu(
                        decode_carrier<long double>(input_bits, format)),
                format);
    }
    return {bits, class_of_bits(type, bits)};
}

inline std::uint64_t ordered_distance(
        std::uint64_t left, std::uint64_t right,
        std::uint8_t bits) noexcept {
    const std::uint64_t mask = bits >= 64
            ? ~std::uint64_t{0}
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

inline std::uint64_t tolerance_ulp(iom::DataType type) noexcept {
    switch (type) {
        case iom::DataType::F32: return 4;
        case iom::DataType::F64: return 8;
        case iom::DataType::F4_E2M1:
        case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2:
        case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2:
        case iom::DataType::F16:
        case iom::DataType::BF16:
            return 1;
        default: return 0;
    }
}

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

}  // namespace silu_oracle

using SiluReferenceClass = silu_oracle::SiluReferenceClass;
using SiluReferenceValue = silu_oracle::SiluReferenceValue;

inline constexpr std::string_view silu_reference_class_name(
        SiluReferenceClass value_class) noexcept {
    return silu_oracle::class_name(value_class);
}

// ---------------------------------------------------------------------------
// Independent fixture descriptions and expected logical images.
// ---------------------------------------------------------------------------

enum class SiluReferenceCaseKind : std::uint8_t {
    ordinary,
    special_values,
    run_1,
    run_15,
    run_16,
    run_17,
    rounding,
    negative_tail,
    boundary_sizes,
    transformed_leading,
};

inline constexpr std::string_view silu_reference_case_kind_name(
        SiluReferenceCaseKind kind) noexcept {
    switch (kind) {
        case SiluReferenceCaseKind::ordinary: return "ordinary";
        case SiluReferenceCaseKind::special_values: return "special_values";
        case SiluReferenceCaseKind::run_1: return "run_1";
        case SiluReferenceCaseKind::run_15: return "run_15";
        case SiluReferenceCaseKind::run_16: return "run_16";
        case SiluReferenceCaseKind::run_17: return "run_17";
        case SiluReferenceCaseKind::rounding: return "rounding";
        case SiluReferenceCaseKind::negative_tail: return "negative_tail";
        case SiluReferenceCaseKind::boundary_sizes: return "boundary_sizes";
        case SiluReferenceCaseKind::transformed_leading:
            return "transformed_leading";
    }
    return "unknown";
}

struct SiluReferenceCase {
    SiluReferenceCaseKind kind = SiluReferenceCaseKind::ordinary;
    iom::DataType data_type = iom::DataType::F32;
    std::vector<std::size_t> leading_dimensions;
    std::size_t runs = 0;
    std::size_t features = 0;
    std::vector<std::uint64_t> input_bits;
    std::string label;
};

inline std::size_t silu_checked_mul(
        std::size_t lhs, std::size_t rhs, std::string_view what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(what));
    }
    return lhs * rhs;
}

inline std::size_t silu_plane_count(
        std::span<const std::size_t> leading_dimensions) {
    std::size_t result = 1;
    for (const std::size_t extent : leading_dimensions) {
        if (extent == 0) {
            throw std::invalid_argument("SiLU fixture leading extent is zero");
        }
        result = silu_checked_mul(
                result, extent, "SiLU fixture plane count overflows");
    }
    return result;
}

inline std::size_t silu_case_element_count(
        const SiluReferenceCase& reference_case) {
    std::size_t result = silu_plane_count(reference_case.leading_dimensions);
    result = silu_checked_mul(
            result, reference_case.runs,
            "SiLU fixture run count overflows");
    return silu_checked_mul(
            result, reference_case.features,
            "SiLU fixture feature count overflows");
}

inline void validate_silu_reference_case(
        const SiluReferenceCase& reference_case) {
    if (silu_data_type_classification(reference_case.data_type)
            != SiluDataTypeClassification::applicable) {
        throw std::invalid_argument("SiLU fixture uses an inapplicable leaf");
    }
    if (reference_case.runs == 0 || reference_case.features == 0) {
        throw std::invalid_argument("SiLU fixture has a zero extent");
    }
    if (reference_case.input_bits.size()
            != silu_case_element_count(reference_case)) {
        throw std::invalid_argument(
                "SiLU fixture bits do not match its logical shape");
    }
}

inline std::vector<std::byte> silu_pack_bits(
        iom::DataType type, std::span<const std::uint64_t> values) {
    const std::size_t bits = silu_oracle::leaf_bits(type);
    const std::size_t bit_count = silu_checked_mul(
            values.size(), bits, "SiLU logical bit count overflows");
    std::vector<std::byte> result((bit_count + 7) / 8, std::byte{0});
    auto* raw = reinterpret_cast<unsigned char*>(result.data());
    for (std::size_t index = 0; index < values.size(); ++index) {
        write_bits(raw, index * bits, bits, values[index]);
    }
    return result;
}

inline std::uint64_t silu_read_bits(
        std::span<const std::byte> bytes, iom::DataType type,
        std::size_t index) {
    const std::size_t bits = silu_oracle::leaf_bits(type);
    if (index >= bytes.size() * 8 / bits) {
        throw std::invalid_argument("SiLU logical image index is out of bounds");
    }
    return read_storage_bits(bytes.data(), index * bits, bits);
}

inline std::vector<SiluReferenceValue> silu_evaluate(
        const SiluReferenceCase& reference_case) {
    validate_silu_reference_case(reference_case);
    std::vector<SiluReferenceValue> result;
    result.reserve(reference_case.input_bits.size());
    for (const std::uint64_t input : reference_case.input_bits) {
        result.push_back(
                silu_oracle::evaluate_value(reference_case.data_type, input));
    }
    return result;
}

inline bool silu_matches(
        iom::DataType type, std::uint64_t actual_bits,
        const SiluReferenceValue& expected) noexcept {
    const SiluReferenceClass actual_class =
            silu_oracle::class_of_bits(type, actual_bits);
    if (expected.value_class != SiluReferenceClass::finite) {
        if (expected.value_class == SiluReferenceClass::quiet_nan) {
            return actual_class == SiluReferenceClass::quiet_nan;
        }
        // A zero expectation can be a target-format rounding tie rather than a
        // class of its own (see `evaluate_value`: `silu(2^-24)` sits just above
        // the F16 tie at `2^-25`), so a zero is compared by the same sign plus
        // declared ULP ceiling as any other value. Infinity expectations stay
        // exact, and a zero expectation still rejects a different sign.
        if (expected.value_class != SiluReferenceClass::positive_zero
                && expected.value_class != SiluReferenceClass::negative_zero) {
            return actual_class == expected.value_class;
        }
    }
    if (actual_class != SiluReferenceClass::finite
            && actual_class != SiluReferenceClass::positive_zero
            && actual_class != SiluReferenceClass::negative_zero) {
        return false;
    }
    const double actual = silu_oracle::decode_as_double(type, actual_bits);
    const double reference =
            silu_oracle::decode_as_double(type, expected.bits);
    if (std::signbit(actual) != std::signbit(reference)) {
        return false;
    }
    return silu_oracle::ordered_distance(
                   actual_bits, expected.bits,
                   static_cast<std::uint8_t>(silu_oracle::leaf_bits(type)))
           <= silu_oracle::tolerance_ulp(type);
}

inline std::string silu_value_text(
        iom::DataType type, std::uint64_t bits) {
    std::string result(silu_oracle::class_name(
            silu_oracle::class_of_bits(type, bits)));
    result += " bits=0x";
    constexpr char digits[] = "0123456789ABCDEF";
    const std::size_t width = silu_oracle::leaf_bits(type);
    for (std::size_t nibble = (width + 3) / 4; nibble-- > 0;) {
        result.push_back(digits[(bits >> (nibble * 4)) & 0xFu]);
    }
    return result;
}

inline std::string silu_compare(
        iom::DataType type, std::span<const std::byte> actual_image,
        std::span<const SiluReferenceValue> expected,
        std::string_view label) {
    std::string details;
    std::size_t mismatches = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::uint64_t actual = silu_read_bits(actual_image, type, index);
        if (silu_matches(type, actual, expected[index])) {
            continue;
        }
        if (mismatches < 4) {
            details += "\n  [" + std::to_string(index) + "] actual "
                    + silu_value_text(type, actual) + " expected "
                    + silu_value_text(type, expected[index].bits);
        }
        ++mismatches;
    }
    if (mismatches == 0) return {};
    return std::string(label) + ": " + std::to_string(mismatches)
            + " logical SiLU mismatches" + details;
}

inline std::vector<double> silu_pattern_values(
        std::size_t planes, std::size_t runs, std::size_t features) {
    constexpr std::array<double, 16> values = {
            -6.0, -3.0, -1.5, -0.5,
            -0.0,  0.0,  0.5,  1.5,
            3.0,   6.0,  0.25, -0.25,
            2.0,  -2.0,  4.0, -4.0,
    };
    std::vector<double> result(
            silu_checked_mul(
                    silu_checked_mul(planes, runs,
                                     "SiLU pattern count overflows"),
                    features, "SiLU pattern count overflows"));
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t run = 0; run < runs; ++run) {
            for (std::size_t feature = 0; feature < features; ++feature) {
                const std::size_t index =
                        (plane * runs + run) * features + feature;
                result[index] = values[(feature * 7 + run * 3 + plane * 5)
                                       % values.size()];
            }
        }
    }
    return result;
}

inline SiluReferenceCase silu_make_value_case(
        SiluReferenceCaseKind kind, iom::DataType type,
        std::vector<std::size_t> leading_dimensions, std::size_t runs,
        std::size_t features, std::span<const double> values,
        std::string label) {
    SiluReferenceCase result;
    result.kind = kind;
    result.data_type = type;
    result.leading_dimensions = std::move(leading_dimensions);
    result.runs = runs;
    result.features = features;
    result.label = std::move(label);
    const std::size_t expected_count = silu_case_element_count(result);
    if (values.size() != expected_count) {
        throw std::invalid_argument("SiLU value fixture has the wrong size");
    }
    result.input_bits.reserve(values.size());
    for (const double value : values) {
        result.input_bits.push_back(silu_oracle::value_bits(type, value));
    }
    return result;
}

inline SiluReferenceCase silu_make_pattern_case(
        SiluReferenceCaseKind kind, iom::DataType type,
        std::vector<std::size_t> leading_dimensions, std::size_t runs,
        std::size_t features, std::string label) {
    const std::size_t planes = silu_plane_count(leading_dimensions);
    const std::vector<double> values =
            silu_pattern_values(planes, runs, features);
    return silu_make_value_case(
            kind, type, std::move(leading_dimensions), runs, features,
            values, std::move(label));
}

inline std::vector<std::uint64_t> silu_special_codes(iom::DataType type) {
    const silu_oracle::FloatFormat format = silu_oracle::float_format(type);
    const std::uint64_t sign =
            std::uint64_t{1} << (format.exponent_bits + format.mantissa_bits);
    const std::uint64_t exponent_max = silu_oracle::exponent_mask(format);
    const std::uint64_t mantissa_max = silu_oracle::mantissa_mask(format);
    std::vector<std::uint64_t> codes = {
            0,
            sign,
            1,
            sign | 1,
            silu_oracle::value_bits(type, 0.5),
            silu_oracle::value_bits(type, -0.5),
            silu_oracle::value_bits(type, 1.0),
            silu_oracle::value_bits(type, -1.0),
            (silu_oracle::finite_exponent_max(format) << format.mantissa_bits)
                    | mantissa_max,
    };
    if (silu_oracle::represents_infinity(type)) {
        codes.push_back(exponent_max << format.mantissa_bits);
        codes.push_back(sign | (exponent_max << format.mantissa_bits));
    }
    if (silu_oracle::represents_nan(type)) {
        codes.push_back((exponent_max << format.mantissa_bits)
                        | (mantissa_max != 0 ? mantissa_max : 1));
        codes.push_back(sign | (exponent_max << format.mantissa_bits)
                        | (mantissa_max != 0 ? mantissa_max : 1));
    }
    return codes;
}

inline SiluReferenceCase silu_make_special_case(iom::DataType type) {
    constexpr std::size_t planes = 2;
    constexpr std::size_t runs = 2;
    constexpr std::size_t features = 17;
    const std::vector<std::uint64_t> codes = silu_special_codes(type);
    SiluReferenceCase result{
            SiluReferenceCaseKind::special_values,
            type,
            {planes},
            runs,
            features,
            {},
            "special_values/" + silu_oracle::leaf_name(type)};
    result.input_bits.resize(silu_case_element_count(result));
    for (std::size_t index = 0; index < result.input_bits.size(); ++index) {
        result.input_bits[index] = codes[index % codes.size()];
    }
    return result;
}

inline SiluReferenceCase silu_make_rounding_case(iom::DataType type) {
    constexpr std::size_t planes = 2;
    constexpr std::size_t runs = 2;
    constexpr std::size_t features = 17;
    const silu_oracle::FloatFormat format = silu_oracle::float_format(type);
    const std::uint64_t mask = silu_oracle::width_mask(format);
    const std::uint64_t finite_max =
            (silu_oracle::finite_exponent_max(format) << format.mantissa_bits)
            | silu_oracle::mantissa_mask(format);
    const std::array<std::uint64_t, 12> seeds = {
            0, 1, 2, 3, 4, 5, 6, 7,
            finite_max, finite_max > 0 ? finite_max - 1 : 0,
            mask ^ (std::uint64_t{1} << (format.exponent_bits
                                         + format.mantissa_bits)),
            silu_oracle::value_bits(type, 1.5),
    };
    SiluReferenceCase result{
            SiluReferenceCaseKind::rounding,
            type,
            {planes},
            runs,
            features,
            {},
            "rounding/" + silu_oracle::leaf_name(type)};
    result.input_bits.resize(silu_case_element_count(result));
    for (std::size_t index = 0; index < result.input_bits.size(); ++index) {
        result.input_bits[index] = seeds[index % seeds.size()] & mask;
    }
    return result;
}

inline SiluReferenceCase silu_make_tail_case(iom::DataType type) {
    const double input = type == iom::DataType::F64 ? -746.0 : -104.0;
    SiluReferenceCase result{
            SiluReferenceCaseKind::negative_tail,
            type,
            {2},
            2,
            17,
            {},
            "negative_tail/" + silu_oracle::leaf_name(type)};
    result.input_bits.resize(silu_case_element_count(result));
    const std::uint64_t raw = silu_oracle::value_bits(type, input);
    for (std::size_t index = 0; index < result.input_bits.size(); ++index) {
        result.input_bits[index] = index % 3 == 0
                ? raw
                : silu_oracle::value_bits(type, index % 2 == 0 ? -6.0 : 6.0);
    }
    return result;
}

inline std::vector<SiluReferenceCase> silu_reference_cases(
        iom::DataType type) {
    std::vector<SiluReferenceCase> result;
    result.push_back(silu_make_pattern_case(
            SiluReferenceCaseKind::ordinary, type, {2}, 2, 17,
            "ordinary/" + silu_oracle::leaf_name(type)));
    result.push_back(silu_make_special_case(type));
    const std::array<std::size_t, 4> boundaries = {1, 15, 16, 17};
    for (const std::size_t boundary : boundaries) {
        SiluReferenceCaseKind kind = SiluReferenceCaseKind::run_1;
        if (boundary == 15) kind = SiluReferenceCaseKind::run_15;
        if (boundary == 16) kind = SiluReferenceCaseKind::run_16;
        if (boundary == 17) kind = SiluReferenceCaseKind::run_17;
        result.push_back(silu_make_pattern_case(
                kind, type, {2}, boundary, boundary,
                std::string(silu_reference_case_kind_name(kind)) + "/"
                        + silu_oracle::leaf_name(type)));
    }
    result.push_back(silu_make_rounding_case(type));
    if (type == iom::DataType::F32 || type == iom::DataType::F64) {
        result.push_back(silu_make_tail_case(type));
    }
    return result;
}

struct SiluRankGeometry {
    std::vector<std::size_t> leading;
    std::size_t runs;
    std::size_t features;
};

inline std::array<SiluRankGeometry, 7> silu_rank_geometries() {
    return {{
            {{}, 1, 15},
            {{2}, 15, 16},
            {{2, 2}, 16, 17},
            {{2, 1, 2}, 17, 15},
            {{1, 2, 1, 2}, 15, 17},
            {{1, 1, 2, 1, 1}, 16, 16},
            {{1, 1, 1, 2, 1, 1}, 17, 17},
    }};
}

inline bool silu_reference_self_check() {
    for (const iom::DataType type : kSiluApplicableDataTypes) {
        const SiluReferenceCase reference_case =
                silu_make_pattern_case(
                        SiluReferenceCaseKind::ordinary, type, {1}, 1, 17,
                        "self-check");
        const std::vector<SiluReferenceValue> expected =
                silu_evaluate(reference_case);
        for (const SiluReferenceValue& value : expected) {
            std::uint64_t perturbed = value.bits;
            if (value.value_class == SiluReferenceClass::quiet_nan) {
                perturbed = silu_oracle::value_bits(type, 0.0);
            } else {
                perturbed ^= std::uint64_t{1}
                        << (silu_oracle::leaf_bits(type) - 1);
            }
            if (silu_matches(type, perturbed, value)) {
                return false;
            }
        }
        if (expected.size() >= 2) {
            // A swapped element must not be accepted as the expected mapping.
            const SiluReferenceValue& first = expected[0];
            const SiluReferenceValue& second = expected[1];
            if (silu_matches(type, first.bits, second)
                    && silu_matches(type, second.bits, first)) {
                return false;
            }
        }
    }
    const SiluReferenceCase f32_tail = silu_make_tail_case(iom::DataType::F32);
    const SiluReferenceValue tail = silu_evaluate(f32_tail).front();
    const float tail_value = silu_oracle::decode_f32(
            iom::DataType::F32, tail.bits);
    if (!(tail_value < 0.0F)
            || std::fpclassify(tail_value) != FP_SUBNORMAL) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shared contract configuration and helpers.
// ---------------------------------------------------------------------------

struct SiluNativeFailureSeam {
    std::function<void()> arm;
    std::function<void()> clear;
    std::string_view name;
    std::string_view unavailable_reason;

    [[nodiscard]] bool available() const noexcept {
        return static_cast<bool>(arm) && static_cast<bool>(clear);
    }
};

struct SiluConformanceConfig {
    ConformanceDevices devices;
    // Explicit driver declaration of the leaves this port queues. This is not
    // derived from Device::supported_data_types().
    std::span<const iom::DataType> supported_leaves;
    ConformanceObserver* observer = nullptr;
    AcceleratorStorageOracle* native_storage = nullptr;
    SiluNativeFailureSeam native_failure{};
};

static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::silu),
              iom::oid (iom::DeviceOps::*)(
                      const iom::TensorView&, iom::TensorView&,
                      iom::RawWorkspaceView) noexcept>);
static_assert(std::is_same_v<
              decltype(&iom::DeviceOps::silu_workspace_requirements),
              iom::WorkspaceRequirements (iom::DeviceOps::*)(
                      const iom::TensorView&, const iom::TensorView&)>);

inline bool silu_declares(
        std::span<const iom::DataType> leaves, iom::DataType leaf) {
    return std::find(leaves.begin(), leaves.end(), leaf) != leaves.end();
}

inline std::vector<iom::DataType> silu_rejected_leaves(
        const SiluConformanceConfig& config) {
    const std::span<const iom::DataType> storable =
            config.devices.candidate.supported_data_types();
    std::vector<iom::DataType> result;
    for (const iom::DataType leaf : kSiluInapplicableDataTypes) {
        REQUIRE_MESSAGE(
                silu_declares(storable, leaf),
                "candidate capability table must retain every SiLU leaf");
        result.push_back(leaf);
    }
    for (const iom::DataType leaf : kSiluApplicableDataTypes) {
        REQUIRE_MESSAGE(
                silu_declares(storable, leaf),
                "candidate capability table must retain every SiLU leaf");
        if (!silu_declares(config.supported_leaves, leaf)) {
            result.push_back(leaf);
        }
    }
    return result;
}

inline iom::DataType silu_first_storable(
        const SiluConformanceConfig& config) {
    const std::span<const iom::DataType> storable =
            config.devices.candidate.supported_data_types();
    for (const iom::DataType leaf : config.supported_leaves) {
        if (silu_declares(storable, leaf)) return leaf;
    }
    throw std::invalid_argument(
            "declared SiLU span names no storable candidate leaf");
}

inline iom::TensorSpec silu_make_spec(
        std::vector<std::size_t> dimensions, iom::DataType type) {
    return iom::TensorSpec{iom::TensorShape{std::move(dimensions)}, type};
}

inline std::vector<std::size_t> silu_full_dimensions(
        std::span<const std::size_t> leading, std::size_t runs,
        std::size_t features) {
    std::vector<std::size_t> dimensions(leading.begin(), leading.end());
    dimensions.push_back(runs);
    dimensions.push_back(features);
    return dimensions;
}

inline iom::TensorSpec silu_case_spec(
        const SiluReferenceCase& reference_case) {
    return silu_make_spec(
            silu_full_dimensions(
                    reference_case.leading_dimensions,
                    reference_case.runs, reference_case.features),
            reference_case.data_type);
}

class SiluCaseWindow final {
public:
    explicit SiluCaseWindow(ConformanceObserver* observer)
            : observer_(observer) {
        if (observer_ != nullptr) observer_->setup_complete();
    }
    ~SiluCaseWindow() {
        if (observer_ != nullptr) observer_->case_complete();
    }
    SiluCaseWindow(const SiluCaseWindow&) = delete;
    SiluCaseWindow& operator=(const SiluCaseWindow&) = delete;

private:
    ConformanceObserver* observer_;
};

class SiluTestWorkspace final : public iom::RawWorkspace {
public:
    SiluTestWorkspace(
            const iom::Device& device, void* address, std::size_t bytes)
            : iom::RawWorkspace(device, bytes), address_(address) {}

private:
    [[nodiscard]] void* workspace_address() const noexcept override {
        return address_;
    }
    void* address_;
};

inline iom::RawWorkspaceView silu_dead_workspace_view(
        const iom::Device& device, void* address, std::size_t bytes) {
    auto owner = std::make_unique<SiluTestWorkspace>(device, address, bytes);
    const iom::RawWorkspaceView view = owner->view();
    owner.reset();
    return view;
}

class SiluOwnerSpecRestore final {
public:
    SiluOwnerSpecRestore(iom::Tensor& owner, const iom::TensorSpec& replacement)
            : spec_(const_cast<iom::TensorSpec&>(owner.view().spec())),
              saved_(spec_) {
        spec_ = replacement;
    }
    ~SiluOwnerSpecRestore() { spec_ = saved_; }
    SiluOwnerSpecRestore(const SiluOwnerSpecRestore&) = delete;
    SiluOwnerSpecRestore& operator=(const SiluOwnerSpecRestore&) = delete;

private:
    iom::TensorSpec& spec_;
    iom::TensorSpec saved_;
};

enum class SiluViewKind : std::uint8_t {
    full,
    sliced_permuted,
    selected,
};

inline iom::TensorView silu_transformed_view(iom::Tensor& owner) {
    const iom::TensorView sliced = owner.view().slice(0, 1, 2, 1);
    const iom::TensorView stepped = sliced.slice(1, 0, 2, 2);
    const std::array<std::size_t, 2> order = {1, 0};
    return stepped.permute(order);
}

inline iom::TensorView silu_selected_view(iom::Tensor& owner) {
    const iom::TensorView selected = owner.view().select(0, 1);
    return selected.slice(0, 0, 2, 1);
}

inline std::vector<std::byte> silu_storage_image(
        const iom::TensorSpec& owner_spec, const iom::TensorView& view,
        std::span<const std::byte> logical, std::byte padding) {
    std::vector<std::byte> storage(owner_spec.tiled_storage_nbytes(), padding);
    apply_standard_tiled_view(view, owner_spec, logical, storage);
    return storage;
}

inline void silu_seed(
        SiluConformanceConfig& config, iom::Tensor& owner,
        iom::TensorView& view, std::span<const std::byte> logical,
        std::byte padding, std::vector<std::byte>* physical_before) {
    if (config.native_storage != nullptr) {
        const std::vector<std::byte> storage = silu_storage_image(
                owner.view().spec(), view, logical, padding);
        *physical_before = storage;
        config.native_storage->set_owner_spec(owner.view().spec());
        config.native_storage->seed(view, storage);
    } else {
        copy_from_host(view, logical);
    }
}

inline std::vector<std::byte> silu_observe_storage(
        SiluConformanceConfig& config, const iom::Tensor& owner,
        const iom::TensorView& view) {
    if (config.native_storage == nullptr) return {};
    config.native_storage->set_owner_spec(owner.view().spec());
    return config.native_storage->observe(view);
}

inline void silu_check_tail_exact(
        iom::DataType type, const SiluReferenceCase& reference_case,
        std::span<const std::byte> actual,
        std::span<const SiluReferenceValue> expected,
        std::string_view label) {
    if (reference_case.kind != SiluReferenceCaseKind::negative_tail) return;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (reference_case.input_bits[index]
                != silu_oracle::value_bits(
                        type, type == iom::DataType::F64 ? -746.0 : -104.0)) {
            continue;
        }
        const std::uint64_t observed = silu_read_bits(actual, type, index);
        // The requirement belongs to the destination leaf's own domain, so the
        // stored element is classified as the leaf value the port wrote, the
        // way `silu_reference_self_check` classifies its own F32 tail. A
        // binary32 subnormal is FP_NORMAL once promoted to binary64, so the
        // wider promotion would report every conforming F32 tail as erased,
        // while a genuinely erased tail still classifies as FP_ZERO here.
        bool negative_subnormal = false;
        if (type == iom::DataType::F64) {
            const double value = silu_oracle::decode_f64(observed);
            negative_subnormal =
                    value < 0.0 && std::fpclassify(value) == FP_SUBNORMAL;
        } else {
            const float value = silu_oracle::decode_f32(type, observed);
            negative_subnormal =
                    value < 0.0F && std::fpclassify(value) == FP_SUBNORMAL;
        }
        CHECK_MESSAGE(
                negative_subnormal,
                label << ": required negative subnormal tail was erased");
        CHECK_EQ(observed, expected[index].bits);
    }
}

inline void run_silu_reference_fixture(
        SiluConformanceConfig& config, iom::DeviceOps& queue,
        const SiluReferenceCase& reference_case,
        SiluViewKind view_kind = SiluViewKind::full) {
    const iom::TensorSpec logical_spec = silu_case_spec(reference_case);
    iom::TensorSpec owner_spec = logical_spec;
    if (view_kind != SiluViewKind::full) {
        owner_spec = silu_make_spec(
                {3, 4, reference_case.runs, reference_case.features},
                reference_case.data_type);
    }
    auto input = config.devices.candidate.create_tensor(owner_spec);
    auto output = config.devices.candidate.create_tensor(owner_spec);
    iom::TensorView input_view = view_kind == SiluViewKind::sliced_permuted
            ? silu_transformed_view(*input)
            : view_kind == SiluViewKind::selected
                    ? silu_selected_view(*input)
                    : input->view();
    iom::TensorView output_view = view_kind == SiluViewKind::sliced_permuted
            ? silu_transformed_view(*output)
            : view_kind == SiluViewKind::selected
                    ? silu_selected_view(*output)
                    : output->view();
    REQUIRE(input_view.spec() == logical_spec);
    REQUIRE(output_view.spec() == logical_spec);
    CHECK(input_view.owner_identity() != output_view.owner_identity());
    CHECK(input_view.native_handle() != output_view.native_handle());

    const std::vector<std::byte> input_logical = silu_pack_bits(
            reference_case.data_type, reference_case.input_bits);
    const std::vector<SiluReferenceValue> expected =
            silu_evaluate(reference_case);
    const std::vector<std::byte> output_logical(
            logical_spec.logical_nbytes(), kReadbackSentinel);
    std::vector<std::byte> input_physical;
    std::vector<std::byte> output_physical;
    silu_seed(
            config, *input, input_view, input_logical,
            std::byte{0x5A}, &input_physical);
    silu_seed(
            config, *output, output_view, output_logical,
            std::byte{0xA5}, &output_physical);

    const std::string label = reference_case.label
            + (view_kind == SiluViewKind::full
                       ? ""
                       : view_kind == SiluViewKind::selected
                               ? " selected-leading"
                               : " transformed-leading");
    {
        SiluCaseWindow window(config.observer);
        CHECK(queue.silu_workspace_requirements(
                      input_view, output_view)
              == iom::WorkspaceRequirements{0, 1});
        CHECK(queue.silu_workspace_requirements(
                      input_view, output_view)
              == iom::WorkspaceRequirements{0, 1});
        const iom::oid token = queue.silu(input_view, output_view);
        REQUIRE_MESSAGE(iom::oid_is_token(token), label);
        CHECK_NOTHROW(queue.wait(token));
        CHECK_NOTHROW(queue.wait(token));
    }

    const std::vector<std::byte> actual = read_logical(output_view);
    const std::string mismatch = silu_compare(
            reference_case.data_type, actual, expected, label);
    CHECK_MESSAGE(mismatch.empty(), mismatch);
    silu_check_tail_exact(
            reference_case.data_type, reference_case, actual, expected, label);
    CHECK(read_logical(input_view) == input_logical);

    if (config.native_storage != nullptr) {
        CHECK(silu_observe_storage(config, *input, input_view) == input_physical);
        // The physical image is rebuilt from the port's own logical readback
        // through the independent canonical slot map, so this comparison keeps
        // enforcing the tiled-layout invariants: every logical element must sit
        // in the slot the standard 16x16 map assigns it, and no physical slot
        // or packed sub-byte padding bit outside that set may change from the
        // caller's poison. Value accuracy deliberately stays with the
        // ULP-bounded comparison above instead of re-pinning host-libm bits: a
        // conforming device port's `exp` and division are permitted the
        // contract's 1-to-8-ULP ceilings, which the exact expected bits cannot
        // represent.
        std::vector<std::byte> expected_output_storage = output_physical;
        apply_standard_tiled_view(
                output_view, output->view().spec(), actual,
                expected_output_storage);
        CHECK(
                silu_observe_storage(config, *output, output_view)
                == expected_output_storage);
    }
}

inline void run_silu_reference_conformance(
        SiluConformanceConfig& config) {
    REQUIRE_MESSAGE(
            silu_reference_self_check(),
            "independent SiLU oracle self-check failed");
    if (config.supported_leaves.empty()) return;
    REQUIRE_MESSAGE(
            config.native_storage != nullptr,
            "positive SiLU conformance requires a native storage observer");
    for (const iom::DataType type : config.supported_leaves) {
        CAPTURE(static_cast<int>(type));
        auto queue = config.devices.candidate.create_ops();
        for (const SiluReferenceCase& reference_case :
             silu_reference_cases(type)) {
            run_silu_reference_fixture(config, *queue, reference_case);
        }
        for (const SiluRankGeometry& geometry : silu_rank_geometries()) {
            const std::size_t planes = silu_plane_count(geometry.leading);
            const SiluReferenceCase reference_case = silu_make_pattern_case(
                    SiluReferenceCaseKind::boundary_sizes, type,
                    geometry.leading, geometry.runs, geometry.features,
                    "rank" + std::to_string(geometry.leading.size() + 2)
                            + "/" + silu_oracle::leaf_name(type));
            REQUIRE_EQ(
                    planes,
                    silu_plane_count(reference_case.leading_dimensions));
            run_silu_reference_fixture(config, *queue, reference_case);
        }
        const SiluReferenceCase transformed = silu_make_pattern_case(
                SiluReferenceCaseKind::transformed_leading, type, {2, 2},
                17, 17, "transformed-leading/" + silu_oracle::leaf_name(type));
        run_silu_reference_fixture(
                config, *queue, transformed, SiluViewKind::sliced_permuted);
        const SiluReferenceCase selected = silu_make_pattern_case(
                SiluReferenceCaseKind::transformed_leading, type, {2},
                17, 17, "selected-leading/" + silu_oracle::leaf_name(type));
        run_silu_reference_fixture(
                config, *queue, selected, SiluViewKind::selected);
    }
}

// ---------------------------------------------------------------------------
// Query, workspace, capability, and admission scenarios.
// ---------------------------------------------------------------------------

inline void run_silu_query_and_workspace_conformance(
        SiluConformanceConfig& config) {
    if (config.supported_leaves.empty()) return;
    const iom::DataType type = silu_first_storable(config);
    const iom::TensorSpec spec = silu_make_spec({2, 17, 17}, type);
    auto input = config.devices.candidate.create_tensor(spec);
    auto output = config.devices.candidate.create_tensor(spec);
    copy_from_host(
            output->view(),
            std::vector<std::byte>(spec.logical_nbytes(), kReadbackSentinel));
    const std::vector<std::byte> before = read_logical(output->view());
    std::array<std::byte, 128> scratch{};
    SiluTestWorkspace local_workspace(
            config.devices.candidate, scratch.data(), scratch.size());
    SiluTestWorkspace zero_workspace(
            config.devices.candidate, scratch.data(), 0);
    SiluTestWorkspace foreign_workspace(
            config.devices.foreign, scratch.data(), scratch.size());
    SiluTestWorkspace overlapping_workspace(
            config.devices.candidate,
            const_cast<void*>(input->view().native_handle()), scratch.size());
    const iom::RawWorkspaceView dead_workspace = silu_dead_workspace_view(
            config.devices.candidate, scratch.data(), scratch.size());
    auto queue = config.devices.candidate.create_ops();
    {
        SiluCaseWindow window(config.observer);
        CHECK(queue->silu_workspace_requirements(input->view(), output->view())
              == iom::WorkspaceRequirements{0, 1});
        CHECK(queue->silu_workspace_requirements(input->view(), output->view())
              == iom::WorkspaceRequirements{0, 1});
        CHECK(read_logical(output->view()) == before);
        const iom::oid first = queue->silu(input->view(), output->view());
        REQUIRE(iom::oid_is_token(first));
        CHECK_EQ(token_sequence(first), std::uint64_t{1});
        CHECK_NOTHROW(queue->wait(first));
        CHECK_NOTHROW(queue->wait(first));
        for (const iom::RawWorkspaceView workspace : {
                     local_workspace.view(), zero_workspace.view(),
                     foreign_workspace.view(), overlapping_workspace.view(),
                     dead_workspace}) {
            const iom::oid token = queue->silu(
                    input->view(), output->view(), workspace);
            REQUIRE(iom::oid_is_token(token));
            CHECK_NOTHROW(queue->wait(token));
            CHECK_NOTHROW(queue->wait(token));
        }
    }
}

inline void run_silu_capability_conformance(
        SiluConformanceConfig& config) {
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const std::vector<iom::DataType> rejected = silu_rejected_leaves(config);
    auto queue = config.devices.candidate.create_ops();
    const iom::TensorSpec spec = silu_make_spec({2, 17, 17}, iom::DataType::F32);
    for (const iom::DataType type : rejected) {
        CAPTURE(static_cast<int>(type));
        const iom::TensorSpec leaf_spec = silu_make_spec({2, 17, 17}, type);
        auto input = config.devices.candidate.create_tensor(leaf_spec);
        auto output = config.devices.candidate.create_tensor(leaf_spec);
        const std::vector<std::byte> before = read_logical(output->view());
        {
            SiluCaseWindow window(config.observer);
            CHECK_EQ(queue->silu(input->view(), output->view()), unsupported);
        }
        CHECK(read_logical(output->view()) == before);
        CHECK_THROWS_AS(
                (void)queue->silu_workspace_requirements(
                        input->view(), output->view()),
                std::runtime_error);
    }
    (void)spec;
}

inline void run_silu_admission_conformance(
        SiluConformanceConfig& config) {
    const iom::oid invalid = iom::to_oid(iom::OidError::InvalidArgument);
    const iom::oid overflow = iom::to_oid(iom::OidError::Overflow);
    const iom::oid unsupported = iom::to_oid(iom::OidError::Unsupported);
    const std::span<const iom::DataType> storable =
            config.devices.candidate.supported_data_types();
    const iom::DataType type =
            silu_declares(storable, iom::DataType::F32)
                    ? iom::DataType::F32
                    : storable.front();
    const iom::TensorSpec spec = silu_make_spec({2, 17, 17}, type);
    auto input = config.devices.candidate.create_tensor(spec);
    auto output = config.devices.candidate.create_tensor(spec);
    auto short_output = config.devices.candidate.create_tensor(
            silu_make_spec({2, 16, 17}, type));
    auto mixed_output = config.devices.candidate.create_tensor(
            silu_make_spec({2, 17, 17}, iom::DataType::F16));
    auto foreign_input = config.devices.foreign.create_tensor(spec);
    auto foreign_output = config.devices.foreign.create_tensor(spec);
    auto leading_owner = config.devices.candidate.create_tensor(
            silu_make_spec({3, 2, 17, 17}, type));
    auto rank_output = config.devices.candidate.create_tensor(
            silu_make_spec({3, 2, 17, 17}, type));
    auto quantized_input = config.devices.candidate.create_tensor(spec);
    auto quantized_output = config.devices.candidate.create_tensor(spec);
    const std::vector<std::byte> output_before = read_logical(output->view());
    auto queue = config.devices.candidate.create_ops();

    const auto reject = [&](const iom::TensorView& x_view,
                            iom::TensorView& y_view, iom::oid expected,
                            std::string_view label) {
        const std::vector<std::byte> before = read_logical(y_view);
        const iom::oid observed = queue->silu(x_view, y_view);
        CHECK_MESSAGE(observed == expected, label);
        CHECK_MESSAGE(read_logical(y_view) == before, label);
    };
    reject(input->view(), short_output->view(), invalid, "shape mismatch");
    reject(input->view(), mixed_output->view(), invalid, "dtype mismatch");
    reject(foreign_input->view(), output->view(), invalid, "foreign input");
    reject(input->view(), foreign_output->view(), invalid, "foreign output");
    reject(input->view(), input->view(), invalid, "same-owner alias");
    const iom::TensorView first_plane = leading_owner->view().select(0, 0);
    iom::TensorView second_plane = leading_owner->view().select(0, 1);
    reject(first_plane, second_plane, invalid, "transformed same-owner alias");

    for (const std::size_t axis : {std::size_t{0}, std::size_t{1},
                                   std::size_t{2}}) {
        iom::TensorView zero_extent = input->view();
        const_cast<std::size_t*>(
                zero_extent.spec().shape.dimensions().data())[axis] = 0;
        reject(zero_extent, output->view(), invalid, "zero extent");
    }

    iom::TensorView zero_stride = leading_owner->view();
    std::size_t* stride = const_cast<std::size_t*>(
            zero_stride.plane_strides().data());
    const std::size_t saved_stride = stride[0];
    stride[0] = 0;
    reject(zero_stride, rank_output->view(), invalid, "zero plane stride");
    stride[0] = std::numeric_limits<std::size_t>::max();
    reject(zero_stride, rank_output->view(), overflow, "plane stride overflow");
    stride[0] = saved_stride;

    {
        iom::TensorSpec unknown = input->view().spec();
        unknown.data_type = static_cast<iom::DataType>(255);
        SiluOwnerSpecRestore restore_input(*input, unknown);
        SiluOwnerSpecRestore restore_output(*output, unknown);
        const iom::oid observed = queue->silu(input->view(), output->view());
        CHECK_EQ(observed, invalid);
        CHECK_THROWS_AS(
                (void)queue->silu_workspace_requirements(
                        input->view(), output->view()),
                std::invalid_argument);
    }
    {
        iom::TensorSpec unknown = input->view().spec();
        unknown.quantization = static_cast<iom::QuantizationFormat>(9999);
        SiluOwnerSpecRestore restore_input(*input, unknown);
        SiluOwnerSpecRestore restore_output(*output, unknown);
        CHECK_EQ(queue->silu(input->view(), output->view()), invalid);
        CHECK_THROWS_AS(
                (void)queue->silu_workspace_requirements(
                        input->view(), output->view()),
                std::invalid_argument);
    }
    {
        iom::TensorSpec grouped = input->view().spec();
        grouped.quantization = iom::QuantizationFormat::OCP_MXFP4;
        SiluOwnerSpecRestore restore_input(*quantized_input, grouped);
        SiluOwnerSpecRestore restore_output(*quantized_output, grouped);
        CHECK_EQ(
                queue->silu(
                        quantized_input->view(), quantized_output->view()),
                unsupported);
        CHECK_THROWS_AS(
                (void)queue->silu_workspace_requirements(
                        quantized_input->view(), quantized_output->view()),
                std::runtime_error);
    }
    {
        iom::TensorView malformed = input->view();
        const_cast<std::size_t*>(
                malformed.spec().shape.dimensions().data())[0] = 0;
        reject(malformed, output->view(), invalid, "malformed view");
    }
    CHECK(read_logical(output->view()) == output_before);

    // A rank-nine shape is rejected before a tensor owner can be created.
    std::vector<std::size_t> rank_nine(9, 1);
    rank_nine[7] = 17;
    rank_nine[8] = 17;
    CHECK_THROWS_AS((iom::TensorShape{rank_nine}), std::invalid_argument);

    // A valid request on a declared leaf is still accepted after all negative
    // probes, proving that none reserved a sequence. Unported declarations
    // intentionally remain rejection-only.
    if (silu_declares(config.supported_leaves, type)) {
        auto accepted_input = config.devices.candidate.create_tensor(spec);
        auto accepted_output = config.devices.candidate.create_tensor(spec);
        auto accepted_queue = config.devices.candidate.create_ops();
        const iom::oid token = accepted_queue->silu(
                accepted_input->view(), accepted_output->view());
        REQUIRE(iom::oid_is_token(token));
        CHECK_EQ(token_sequence(token), std::uint64_t{1});
        CHECK_NOTHROW(accepted_queue->wait(token));
    }
}

// ---------------------------------------------------------------------------
// Queue ordering, temporary-view ownership, accepted failures, and mul.
// ---------------------------------------------------------------------------

class SiluNativeFailureGuard final {
public:
    explicit SiluNativeFailureGuard(const SiluNativeFailureSeam& seam)
            : seam_(seam) {
        seam_.arm();
        armed_ = true;
    }
    ~SiluNativeFailureGuard() {
        if (armed_) seam_.clear();
    }
    SiluNativeFailureGuard(const SiluNativeFailureGuard&) = delete;
    SiluNativeFailureGuard& operator=(const SiluNativeFailureGuard&) = delete;

private:
    const SiluNativeFailureSeam& seam_;
    bool armed_ = false;
};

inline void run_silu_queue_and_lifetime_conformance(
        SiluConformanceConfig& config) {
    if (config.supported_leaves.empty()) return;
    const iom::DataType type = silu_first_storable(config);
    const iom::TensorSpec spec = silu_make_spec({2, 2, 17}, type);
    auto source = config.devices.candidate.create_tensor(spec);
    auto input = config.devices.candidate.create_tensor(spec);
    auto activated = config.devices.candidate.create_tensor(spec);
    auto consumed = config.devices.candidate.create_tensor(spec);
    const SiluReferenceCase fixture = silu_make_pattern_case(
            SiluReferenceCaseKind::boundary_sizes, type, {2}, 2, 17,
            "queue-order/" + silu_oracle::leaf_name(type));
    const std::vector<std::byte> source_bytes = silu_pack_bits(
            type, fixture.input_bits);
    const std::vector<std::byte> zeros(
            spec.logical_nbytes(), std::byte{0});
    copy_from_host(source->view(), source_bytes);
    copy_from_host(input->view(), zeros);
    copy_from_host(activated->view(), zeros);
    copy_from_host(consumed->view(), zeros);
    auto queue = config.devices.candidate.create_ops();
    const iom::TensorView source_view = source->view();
    {
        SiluCaseWindow window(config.observer);
        const iom::oid producer = queue->copy(source_view, input->view());
        REQUIRE(iom::oid_is_token(producer));
        source.reset();
        const iom::oid activation = queue->silu(
                input->view(), activated->view());
        REQUIRE(iom::oid_is_token(activation));
        const iom::oid consumer = queue->copy(
                activated->view(), consumed->view());
        REQUIRE(iom::oid_is_token(consumer));
        CHECK_NOTHROW(queue->wait(consumer));
        CHECK_NOTHROW(queue->wait(consumer));
        CHECK_NOTHROW(queue->wait(activation));
    }
    const std::vector<SiluReferenceValue> expected = silu_evaluate(fixture);
    const std::string activated_mismatch = silu_compare(
            type, read_logical(activated->view()), expected,
            "queue-order activated");
    CHECK_MESSAGE(activated_mismatch.empty(), activated_mismatch);
    CHECK(read_logical(consumed->view()) == read_logical(activated->view()));
}

inline void run_silu_native_failure_conformance(
        SiluConformanceConfig& config) {
    if (config.supported_leaves.empty()) return;
    const SiluNativeFailureSeam& seam = config.native_failure;
    if (!seam.available()) {
        REQUIRE_MESSAGE(
                !seam.unavailable_reason.empty(),
                "a backend without a SiLU failure seam must state why");
        std::printf(
                "silu-native-failure-record seam=%.*s coverage=unavailable "
                "reason=%.*s\n",
                static_cast<int>(seam.name.size()), seam.name.data(),
                static_cast<int>(seam.unavailable_reason.size()),
                seam.unavailable_reason.data());
        return;
    }
    const iom::DataType type = silu_declares(
                    config.supported_leaves, iom::DataType::F32)
            ? iom::DataType::F32
            : silu_first_storable(config);
    const SiluReferenceCase fixture = silu_make_pattern_case(
            SiluReferenceCaseKind::ordinary, type, {2}, 2, 17,
            "native-failure/" + silu_oracle::leaf_name(type));
    const iom::TensorSpec spec = silu_case_spec(fixture);
    auto input = config.devices.candidate.create_tensor(spec);
    auto output = config.devices.candidate.create_tensor(spec);
    const std::vector<std::byte> input_bytes = silu_pack_bits(
            type, fixture.input_bits);
    copy_from_host(input->view(), input_bytes);
    copy_from_host(
            output->view(),
            std::vector<std::byte>(spec.logical_nbytes(), kReadbackSentinel));
    const std::vector<SiluReferenceValue> expected = silu_evaluate(fixture);
    auto queue = config.devices.candidate.create_ops();
    std::printf(
            "silu-native-failure-record seam=%.*s coverage=native reason=\n",
            static_cast<int>(seam.name.size()), seam.name.data());
    const iom::oid healthy = queue->silu(input->view(), output->view());
    REQUIRE(iom::oid_is_token(healthy));
    CHECK_NOTHROW(queue->wait(healthy));
    {
        SiluNativeFailureGuard guard(seam);
        const iom::oid failed = queue->silu(input->view(), output->view());
        REQUIRE(iom::oid_is_token(failed));
        CHECK_EQ(token_sequence(failed), token_sequence(healthy) + 1);
    }
    const iom::oid recovered = queue->silu(input->view(), output->view());
    REQUIRE(iom::oid_is_token(recovered));
    CHECK_NOTHROW(queue->wait(recovered));
    // The successful recovery is compared; the accepted failed output itself
    // is intentionally unspecified by the contract.
    const std::string mismatch = silu_compare(
            type, read_logical(output->view()), expected,
            "recovered native SiLU");
    CHECK_MESSAGE(mismatch.empty(), mismatch);
}

inline std::uint64_t silu_mul_expected(
        iom::DataType type, std::uint64_t lhs, std::uint64_t rhs) noexcept {
    if (type == iom::DataType::F64) {
        return silu_oracle::encode_f64(
                silu_oracle::decode_f64(lhs) * silu_oracle::decode_f64(rhs));
    }
    const float value = silu_oracle::decode_f32(type, lhs)
            * silu_oracle::decode_f32(type, rhs);
    return silu_oracle::encode_f32(type, value);
}

inline void run_silu_stored_mul_conformance(
        SiluConformanceConfig& config) {
    if (!silu_declares(config.supported_leaves, iom::DataType::F32)) return;
    constexpr iom::DataType type = iom::DataType::F32;
    const SiluReferenceCase fixture = silu_make_pattern_case(
            SiluReferenceCaseKind::ordinary, type, {2}, 2, 17,
            "stored-mul/F32");
    const iom::TensorSpec spec = silu_case_spec(fixture);
    auto gate = config.devices.candidate.create_tensor(spec);
    auto up = config.devices.candidate.create_tensor(spec);
    auto activated = config.devices.candidate.create_tensor(spec);
    auto product = config.devices.candidate.create_tensor(spec);
    const std::vector<std::uint64_t> up_bits = [&] {
        std::vector<std::uint64_t> values(fixture.input_bits.size());
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = silu_oracle::value_bits(
                    type, index % 3 == 0 ? 0.75 : (index % 2 == 0 ? 2.0 : -1.25));
        }
        return values;
    }();
    copy_from_host(gate->view(), silu_pack_bits(type, fixture.input_bits));
    copy_from_host(up->view(), silu_pack_bits(type, up_bits));
    copy_from_host(
            activated->view(),
            std::vector<std::byte>(spec.logical_nbytes(), kReadbackSentinel));
    copy_from_host(
            product->view(),
            std::vector<std::byte>(spec.logical_nbytes(), kReadbackSentinel));
    auto queue = config.devices.candidate.create_ops();
    {
        SiluCaseWindow window(config.observer);
        const iom::oid activation = queue->silu(
                gate->view(), activated->view());
        REQUIRE(iom::oid_is_token(activation));
        const iom::WorkspaceRequirements requirements =
                queue->mul_workspace_requirements(
                        activated->view(), up->view(), product->view());
        std::unique_ptr<iom::RawWorkspace> workspace;
        if (requirements.bytes != 0) {
            workspace = config.devices.candidate.create_workspace(
                    requirements.bytes);
        }
        const iom::oid product_token = workspace
                ? queue->mul(
                          activated->view(), up->view(), product->view(),
                          workspace->view())
                : queue->mul(
                          activated->view(), up->view(), product->view());
        REQUIRE(iom::oid_is_token(product_token));
        CHECK_NOTHROW(queue->wait(product_token));
        CHECK_NOTHROW(queue->wait(product_token));
    }
    const std::vector<SiluReferenceValue> activated_expected =
            silu_evaluate(fixture);
    const std::string activated_mismatch = silu_compare(
            type, read_logical(activated->view()), activated_expected,
            "stored SiLU result");
    CHECK_MESSAGE(activated_mismatch.empty(), activated_mismatch);
    std::vector<std::uint64_t> product_expected;
    product_expected.reserve(up_bits.size());
    for (std::size_t index = 0; index < up_bits.size(); ++index) {
        product_expected.push_back(
                silu_mul_expected(type, activated_expected[index].bits,
                                  up_bits[index]));
    }
    const std::vector<SiluReferenceValue> product_values = [&] {
        std::vector<SiluReferenceValue> values;
        values.reserve(product_expected.size());
        for (const std::uint64_t bits : product_expected) {
            values.push_back({bits, silu_oracle::class_of_bits(type, bits)});
        }
        return values;
    }();
    const std::string product_mismatch = silu_compare(
            type, read_logical(product->view()), product_values,
            "stored SiLU result composed with mul");
    CHECK_MESSAGE(product_mismatch.empty(), product_mismatch);
}

inline void run_silu_conformance(SiluConformanceConfig config) {
    for (const iom::DataType leaf : config.supported_leaves) {
        REQUIRE_MESSAGE(
                silu_data_type_classification(leaf)
                        == SiluDataTypeClassification::applicable,
                "SiLU driver span contains an inapplicable leaf");
        REQUIRE_MESSAGE(
                silu_declares(config.devices.candidate.supported_data_types(), leaf),
                "SiLU driver span contains an unstorable leaf");
    }
    run_silu_capability_conformance(config);
    run_silu_admission_conformance(config);
    run_silu_query_and_workspace_conformance(config);
    run_silu_reference_conformance(config);
    run_silu_queue_and_lifetime_conformance(config);
    run_silu_stored_mul_conformance(config);
    run_silu_native_failure_conformance(config);
}

}  // namespace iom_conformance

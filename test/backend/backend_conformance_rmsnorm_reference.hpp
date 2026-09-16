#pragma once

// Independent RMSNorm reference oracle (change 006-tinyllama / 05-rms-normalization / 02).
//
// The oracle models the frozen `RMS normalization` contract of
// docs/BACKEND_CONTRACT.md while sharing no arithmetic with production code: it
// calls no production RMSNorm declaration, codec, tile mapping, or address
// helper, and it never switches on BackendKind. It owns
//
//   * an independent raw-bit decode of the nine applicable float leaves into
//     the accumulator domain (FP32 for F4/F6/F8/F16/BF16/F32, FP64 for F64),
//   * a scalar evaluation of the documented row equation
//         mean = (sum_j square(x_j)) / F
//         inv  = rsqrt(mean + eps)
//         y_j  = (x_j * inv) * scale_j
//     that materializes and rounds every FP32 square, reduction step, mean,
//     epsilon addition, square root, reciprocal, normalization product, and
//     scale product separately (F64 keeps every step in double),
//   * exactly one round-to-nearest-even encode per output element with the
//     established named-format special and saturation rules, storing a NaN
//     result as the canonical positive quiet NaN because NaN payloads and NaN
//     signs are outside the contract and pinning them keeps expected bits
//     host-independent,
//   * the fixed expected-value comparison policy (exact classes and signed
//     zeros with NaN payloads ignored, two adjacent destination encodings
//     below F32, and the pinned F32/F64 error bounds),
//   * deterministic repository-owned fixture cases for every applicable leaf
//     and the explicit applicable-versus-unsupported classification of all 23
//     data-type leaves, and
//   * analytic self-checks that run without a backend or production operation.
//
// Shared conformance owns everything observable around the arithmetic: tensor
// allocation, backend capability spans, logical and native reads, admission
// shape/alias/workspace rejection, owner registration, queue ordering, and
// repeated-wait failure semantics. It consumes this header's cases, expected
// values, and comparisons instead of duplicating arithmetic.
//
// Row model: `x` and `out` are identical `[..., R, F]` and `scale` is exactly
// rank-two `[1, F]`. Leading planes and rows are independent, and only the
// logical `F` features participate in the reduction, so padding, other rows,
// and other planes never enter the equation. Case bits follow the host encoding
// convention of backend_conformance_common.hpp: least-significant bit first at
// bit offset `element_index * leaf_width` of a contiguous row-major buffer, so
// shared callers pack and unpack them with the harness `bits_of`/`write_bits`
// helpers. This header deliberately depends on `iom/tensor.hpp` alone, because
// the oracle needs no device, doctest, or harness type of its own.

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
// Expected values.
// ---------------------------------------------------------------------------

// Class of one expected, and of one measured, leaf value. Zeros and infinities
// keep their sign; NaN payloads and NaN signs are deliberately not modelled
// because the contract compares NaN-ness only.
enum class RmsNormReferenceClass : std::uint8_t {
    finite,
    positive_zero,
    negative_zero,
    positive_infinity,
    negative_infinity,
    quiet_nan,
};

inline constexpr std::string_view rmsnorm_reference_class_name(
        RmsNormReferenceClass value_class) noexcept {
    switch (value_class) {
        case RmsNormReferenceClass::finite: return "finite";
        case RmsNormReferenceClass::positive_zero: return "+0";
        case RmsNormReferenceClass::negative_zero: return "-0";
        case RmsNormReferenceClass::positive_infinity: return "+inf";
        case RmsNormReferenceClass::negative_infinity: return "-inf";
        case RmsNormReferenceClass::quiet_nan: return "nan";
    }
    return "unknown";
}
// One expected logical output element: the raw bits of its single destination
// encoding plus the class of the value those bits decode to.
struct RmsNormReferenceValue {
    std::uint64_t bits = 0;
    RmsNormReferenceClass value_class = RmsNormReferenceClass::finite;
};

// Class of one logical row's accumulator-domain reduction, independent of the
// destination leaf. A sum that underflowed to zero stays `finite_sum`; only a
// square that overflowed to infinity or a NaN feature changes it. The class
// carries no NaN payload promise.
enum class RmsNormAccumulatorClass : std::uint8_t {
    finite_sum,
    infinite_sum,
    quiet_nan_sum,
};

// ---------------------------------------------------------------------------
// Fixture cases.
// ---------------------------------------------------------------------------

// Fixture categories. Every applicable leaf provides `ones`, `signed_zeros`,
// `zero_row_zero_eps`, `mixed`, `scaled`, `destination_overflow`,
// `destination_underflow`, and the complete `boundary_sizes` sweep. The
// nonfinite categories appear only where the leaf can encode NaN or infinity,
// and the accumulator boundary categories only where the accumulator domain
// can actually overflow or underflow (F32, BF16, F64).
enum class RmsNormReferenceCaseKind : std::uint8_t {
    ones,                  // constant ones with unit scale: closed-form identity
    signed_zeros,          // positive and negative zeros from input and scale
    zero_row_zero_eps,     // all-zero row with eps == 0: quiet NaN with no payload promise
    mixed,                 // mixed sign and magnitude, unit scale
    scaled,                // mixed sign and magnitude with non-unit per-feature scale
    nan_row,               // representable NaN feature poisons its whole row
    infinity_row,          // representable infinity: signed-zero and NaN features
    nonfinite_scale,       // nonfinite scale feature affects only its own feature
    destination_overflow,  // finite accumulator that saturates the destination leaf
    destination_underflow, // finite accumulator that underflows the destination leaf to zero
    accumulator_overflow,  // squares overflow the accumulator: infinite row sum
    accumulator_underflow, // squares underflow to zero: infinite reciprocal square root
    boundary_sizes,        // independent planes/rows with non-tile feature widths
};

inline constexpr std::string_view rmsnorm_reference_case_kind_name(
        RmsNormReferenceCaseKind kind) noexcept {
    switch (kind) {
        case RmsNormReferenceCaseKind::ones: return "ones";
        case RmsNormReferenceCaseKind::signed_zeros: return "signed_zeros";
        case RmsNormReferenceCaseKind::zero_row_zero_eps: return "zero_row_zero_eps";
        case RmsNormReferenceCaseKind::mixed: return "mixed";
        case RmsNormReferenceCaseKind::scaled: return "scaled";
        case RmsNormReferenceCaseKind::nan_row: return "nan_row";
        case RmsNormReferenceCaseKind::infinity_row: return "infinity_row";
        case RmsNormReferenceCaseKind::nonfinite_scale: return "nonfinite_scale";
        case RmsNormReferenceCaseKind::destination_overflow: return "destination_overflow";
        case RmsNormReferenceCaseKind::destination_underflow: return "destination_underflow";
        case RmsNormReferenceCaseKind::accumulator_overflow: return "accumulator_overflow";
        case RmsNormReferenceCaseKind::accumulator_underflow: return "accumulator_underflow";
        case RmsNormReferenceCaseKind::boundary_sizes: return "boundary_sizes";
    }
    return "unknown";
}

// One deterministic fixture: identical `x`/`out` extents `[planes, R, F]`, a
// shared rank-two `[1, F]` scale, and one raw leaf encoding per element.
struct RmsNormReferenceCase {
    RmsNormReferenceCaseKind kind = RmsNormReferenceCaseKind::ones;
    iom::DataType data_type = iom::DataType::F32;
    std::size_t planes = 0;
    std::size_t rows = 0;
    std::size_t features = 0;
    float eps = 0.0f;
    // Row-major raw encodings indexed by `(plane * rows + row) * features +
    // feature`; only the low leaf-width bits are meaningful.
    std::vector<std::uint64_t> x_bits;
    // One raw encoding per logical feature of the shared scale.
    std::vector<std::uint64_t> scale_bits;
};

// Diagnostic label: fixture category, extents, and epsilon.
inline std::string rmsnorm_reference_case_label(
        const RmsNormReferenceCase& reference_case) {
    std::string label(rmsnorm_reference_case_kind_name(reference_case.kind));
    label += " P=" + std::to_string(reference_case.planes);
    label += " R=" + std::to_string(reference_case.rows);
    label += " F=" + std::to_string(reference_case.features);
    label += " eps=" + std::to_string(reference_case.eps);
    return label;
}

// ---------------------------------------------------------------------------
// Data-type classification.
// ---------------------------------------------------------------------------

// Iterable literal classification matrix for RMSNorm. Keeping the matrix as
// two explicit lists makes the classification observable and lets shared
// conformance iterate the applicable leaves and the rejection probes without
// duplicating them.
inline constexpr std::array<iom::DataType, 9> kRmsNormApplicableDataTypes = {
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

inline constexpr std::array<iom::DataType, 14> kRmsNormUnsupportedDataTypes = {
        iom::DataType::BOOL,
        iom::DataType::I2, iom::DataType::U2,
        iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8,
        iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32,
        iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F8_E8M0,
};

static_assert(kRmsNormApplicableDataTypes.size()
                      + kRmsNormUnsupportedDataTypes.size()
              == 23,
              "every RMSNorm data type leaf must be classified exactly once");

enum class RmsNormDataTypeClass : std::uint8_t {
    // Ordinary signed float: numeric conformance applies wherever the owning
    // backend advertises the leaf. CPU, CUDA, and ROCm support all nine;
    // SYCL supports the eight non-F64 leaves and F64 only with
    // `aspect::fp64`; TTNN supports only BF16 and F32 and rejects the seven
    // encoded-carrier leaves. Shared conformance combines this classification
    // with the backend capability span, never this header.
    applicable,
    // Unsupported everywhere: BOOL, the twelve integer leaves, and F8_E8M0
    // (an exponent-only scale carrier) carry an explicit rejection
    // expectation and never coerced arithmetic.
    unsupported,
};

inline constexpr RmsNormDataTypeClass rmsnorm_data_type_classification(
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
            return RmsNormDataTypeClass::applicable;
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
            return RmsNormDataTypeClass::unsupported;
    }
    return RmsNormDataTypeClass::unsupported;
}

// ---------------------------------------------------------------------------
// Independent codec, accumulator, and fixture content. The `rmsnorm_oracle`
// namespace holds the internals shared with the analytic self-checks; the
// entry points above and below it are the consumable API.
// ---------------------------------------------------------------------------
namespace rmsnorm_oracle {

// Metadata of one named leaf. `finite_only` marks encodings whose all-ones
// exponent field is finite unless the format is wide enough to keep its NaN
// encoding (F8_E4M3FN); `has_infinity` marks the leaves with a distinct
// infinity encoding.
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

// Largest exponent field that still encodes a finite value: an all-ones
// exponent is a finite encoding for the narrow finite-only leaves.
inline constexpr std::uint64_t finite_exponent_max(const FloatFormat& format) noexcept {
    return format.finite_only && format.exponent_bits < 4
                   ? exponent_mask(format)
                   : exponent_mask(format) - 1;
}

// A leaf keeps a NaN encoding unless it is a narrow finite-only format, whose
// NaN results therefore saturate to the maximum finite value instead.
inline constexpr bool has_nan_encoding(const FloatFormat& format) noexcept {
    return !format.finite_only || format.exponent_bits >= 4;
}

inline constexpr bool represents_nan(iom::DataType type) noexcept {
    return has_nan_encoding(float_format(type));
}

inline constexpr bool represents_infinity(iom::DataType type) noexcept {
    return float_format(type).has_infinity;
}

// Only these leaves carry an exponent range wide enough for the FP32 or FP64
// accumulator square to overflow or to underflow to zero.
inline constexpr bool accumulator_can_overflow(iom::DataType type) noexcept {
    return type == iom::DataType::F32 || type == iom::DataType::BF16
           || type == iom::DataType::F64;
}

inline constexpr bool accumulator_can_underflow(iom::DataType type) noexcept {
    return accumulator_can_overflow(type);
}

// Short leaf name for diagnostics.
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
        default: return "unsupported leaf";
    }
}

// Raw-bit decode into the accumulator domain. Signed zeros keep their sign and
// every finite encoding of the six narrower leaves is exactly representable in
// the FP32 accumulator.
template <typename Carrier>
inline Carrier decode_carrier(
        std::uint64_t raw, const FloatFormat& format) noexcept {
    if (format.bits == 0) {
        return Carrier{0};
    }
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
            ? std::ldexp(static_cast<Carrier>((std::uint64_t{1} << format.mantissa_bits) + mantissa),
                         static_cast<int>(exponent) - format.bias
                                 - static_cast<int>(format.mantissa_bits))
            : std::ldexp(static_cast<Carrier>(mantissa),
                         1 - format.bias - static_cast<int>(format.mantissa_bits));
    return negative ? -magnitude : magnitude;
}

// Round to nearest, ties to even.
template <typename Carrier>
inline std::uint64_t round_to_nearest_even(Carrier value) noexcept {
    const Carrier integral = std::floor(value);
    const Carrier remainder = value - integral;
    const std::uint64_t truncated = static_cast<std::uint64_t>(integral);
    return truncated
           + ((remainder > static_cast<Carrier>(0.5)
               || (remainder == static_cast<Carrier>(0.5) && (truncated & 1)))
                      ? std::uint64_t{1}
                      : std::uint64_t{0});
}

// Independent round-to-nearest-even encode with the established named-format
// special and saturation rules: a NaN saturates to the maximum finite value in
// a leaf without a NaN encoding and stores the canonical quiet NaN otherwise,
// an infinity saturates where the leaf has no infinity encoding, and a finite
// value that cannot be represented saturates the same way. Signed zeros are
// preserved, and underflow below the smallest subnormal rounds to signed zero.
template <typename Carrier>
inline std::uint64_t encode_carrier(
        Carrier value, const FloatFormat& format) noexcept {
    if (format.bits == 0) {
        return 0;
    }
    const std::uint64_t exponent_max = exponent_mask(format);
    const std::uint64_t fraction_max = mantissa_mask(format);
    const std::uint64_t finite_max = finite_exponent_max(format);
    // A NaN is stored as the canonical positive quiet NaN: NaN payloads and NaN
    // signs are outside the contract (the comparison ignores both), and pinning
    // them keeps the expected bits independent of the host's invalid-operation
    // result. A leaf without a NaN encoding stores its maximum finite value.
    if (std::isnan(value)) {
        const std::uint64_t exponent = has_nan_encoding(format) || format.has_infinity
                                               ? exponent_max
                                               : finite_max;
        const std::uint64_t mantissa = format.has_infinity
                ? std::uint64_t{1} << (format.mantissa_bits - 1)
                : fraction_max;
        return (exponent << format.mantissa_bits) | mantissa;
    }
    const std::uint64_t sign = std::signbit(value) ? std::uint64_t{1} : 0;
    value = std::fabs(value);
    const auto pack = [sign, &format](std::uint64_t exponent, std::uint64_t mantissa) {
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
        const std::uint64_t quantized =
                round_to_nearest_even(std::ldexp(value, -subnormal_exponent));
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

inline std::uint64_t encode_f32(iom::DataType type, float value) noexcept {
    return encode_carrier<float>(value, float_format(type));
}

inline double decode_f64(std::uint64_t raw) noexcept {
    return decode_carrier<double>(raw, float_format(iom::DataType::F64));
}

inline std::uint64_t encode_f64(double value) noexcept {
    return encode_carrier<double>(value, float_format(iom::DataType::F64));
}

// Class of one accumulator-domain value.
inline RmsNormReferenceClass classify(float value) noexcept {
    if (std::isnan(value)) {
        return RmsNormReferenceClass::quiet_nan;
    }
    if (std::isinf(value)) {
        return std::signbit(value) ? RmsNormReferenceClass::negative_infinity
                                   : RmsNormReferenceClass::positive_infinity;
    }
    if (value == 0) {
        return std::signbit(value) ? RmsNormReferenceClass::negative_zero
                                   : RmsNormReferenceClass::positive_zero;
    }
    return RmsNormReferenceClass::finite;
}

inline RmsNormReferenceClass classify(double value) noexcept {
    if (std::isnan(value)) {
        return RmsNormReferenceClass::quiet_nan;
    }
    if (std::isinf(value)) {
        return std::signbit(value) ? RmsNormReferenceClass::negative_infinity
                                   : RmsNormReferenceClass::positive_infinity;
    }
    if (value == 0) {
        return std::signbit(value) ? RmsNormReferenceClass::negative_zero
                                   : RmsNormReferenceClass::positive_zero;
    }
    return RmsNormReferenceClass::finite;
}

// Class of a row's accumulator-domain sum. Squares are never negative, so the
// running sum is NaN only when a feature is NaN and infinite only when a square
// overflowed; an underflowed zero sum stays finite.
inline RmsNormAccumulatorClass accumulator_class_of(float sum) noexcept {
    if (std::isnan(sum)) {
        return RmsNormAccumulatorClass::quiet_nan_sum;
    }
    return std::isinf(sum) ? RmsNormAccumulatorClass::infinite_sum
                           : RmsNormAccumulatorClass::finite_sum;
}

inline RmsNormAccumulatorClass accumulator_class_of(double sum) noexcept {
    if (std::isnan(sum)) {
        return RmsNormAccumulatorClass::quiet_nan_sum;
    }
    return std::isinf(sum) ? RmsNormAccumulatorClass::infinite_sum
                           : RmsNormAccumulatorClass::finite_sum;
}

// Distance between two raw encodings of one signed-magnitude leaf in adjacent
// destination encodings, using the monotone mapping that keeps -0 next to +0.
inline std::uint64_t ordered_distance(
        std::uint64_t left, std::uint64_t right, std::uint8_t bits) noexcept {
    const std::uint64_t mask =
            bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    const auto ordered = [mask, sign](std::uint64_t pattern) {
        const std::uint64_t value = pattern & mask;
        return (value & sign) ? (~value & mask) : (value | sign);
    };
    const std::uint64_t first = ordered(left);
    const std::uint64_t second = ordered(right);
    return first > second ? first - second : second - first;
}

// Largest finite magnitude and smallest positive magnitude of one leaf, derived
// from the same format rules as the encoder instead of hardcoded numbers.
inline float leaf_max_finite_f32(iom::DataType type) noexcept {
    const FloatFormat format = float_format(type);
    return decode_carrier<float>(
            (finite_exponent_max(format) << format.mantissa_bits)
                    | mantissa_mask(format),
            format);
}

inline float leaf_min_positive_f32(iom::DataType type) noexcept {
    return decode_carrier<float>(1, float_format(type));
}

inline double leaf_max_finite_f64() noexcept {
    const FloatFormat format = float_format(iom::DataType::F64);
    return decode_carrier<double>(
            (finite_exponent_max(format) << format.mantissa_bits)
                    | mantissa_mask(format),
            format);
}

inline double leaf_min_positive_f64() noexcept {
    return decode_carrier<double>(1, float_format(iom::DataType::F64));
}

inline double leaf_max_finite(iom::DataType type) noexcept {
    return type == iom::DataType::F64
                   ? leaf_max_finite_f64()
                   : static_cast<double>(leaf_max_finite_f32(type));
}

inline double leaf_min_positive(iom::DataType type) noexcept {
    return type == iom::DataType::F64
                   ? leaf_min_positive_f64()
                   : static_cast<double>(leaf_min_positive_f32(type));
}

// One accumulator-domain value encoded into a leaf's raw bits; every fixture
// and every analytic identity enters through this single encoder.
inline std::uint64_t value_bits(iom::DataType type, double value) noexcept {
    return type == iom::DataType::F64
                   ? encode_f64(value)
                   : encode_f32(type, static_cast<float>(value));
}

// ---------------------------------------------------------------------------
// Row accumulation and row encoding.
// ---------------------------------------------------------------------------

// One logical row's reduction: the rounded running sum and the rounded
// reciprocal square root of `sum / F + eps`.
template <typename Carrier>
struct RowAccumulator {
    Carrier sum = static_cast<Carrier>(0);
    Carrier inverse = static_cast<Carrier>(0);
};

// Every FP32 square, reduction step, mean, epsilon addition, square root, and
// reciprocal is materialized separately, so no expression offers a contraction
// point and no extended-precision or reassociated path exists.
inline RowAccumulator<float> accumulate_row_f32(
        iom::DataType type, std::span<const std::uint64_t> x_bits,
        float eps) noexcept {
    const float divisor = static_cast<float>(x_bits.size());
    float sum = 0.0f;
    for (const std::uint64_t raw : x_bits) {
        const float feature = decode_f32(type, raw);
        const float square = feature * feature;
        sum = sum + square;
    }
    const float mean = sum / divisor;
    const float shifted = mean + eps;
    const float root = std::sqrt(shifted);
    return {sum, 1.0f / root};
}

// The F64 path keeps every step, including the epsilon addition, in double.
inline RowAccumulator<double> accumulate_row_f64(
        std::span<const std::uint64_t> x_bits, float eps) noexcept {
    const double divisor = static_cast<double>(x_bits.size());
    double sum = 0.0;
    for (const std::uint64_t raw : x_bits) {
        const double feature = decode_f64(raw);
        const double square = feature * feature;
        sum = sum + square;
    }
    const double mean = sum / divisor;
    const double shifted = mean + static_cast<double>(eps);
    const double root = std::sqrt(shifted);
    return {sum, 1.0 / root};
}

// One encoded destination value per logical feature: every normalization and
// scale product is rounded in the accumulator domain first, then encoded once.
inline void encode_row_f32(
        iom::DataType type, std::span<const std::uint64_t> x_bits,
        std::span<const std::uint64_t> scale_bits, float inverse,
        std::span<RmsNormReferenceValue> output) noexcept {
    for (std::size_t feature = 0; feature < output.size(); ++feature) {
        const float normalized = decode_f32(type, x_bits[feature]) * inverse;
        const float scaled = normalized * decode_f32(type, scale_bits[feature]);
        const std::uint64_t bits = encode_f32(type, scaled);
        output[feature] = {bits, classify(decode_f32(type, bits))};
    }
}

inline void encode_row_f64(
        std::span<const std::uint64_t> x_bits,
        std::span<const std::uint64_t> scale_bits, double inverse,
        std::span<RmsNormReferenceValue> output) noexcept {
    for (std::size_t feature = 0; feature < output.size(); ++feature) {
        const double normalized = decode_f64(x_bits[feature]) * inverse;
        const double scaled = normalized * decode_f64(scale_bits[feature]);
        const std::uint64_t bits = encode_f64(scaled);
        output[feature] = {bits, classify(decode_f64(bits))};
    }
}

// Validated row-major bit offset of one logical (plane, row) inside a case.
inline std::size_t row_offset(
        const RmsNormReferenceCase& reference_case, std::size_t plane,
        std::size_t row) {
    if (reference_case.planes == 0 || reference_case.rows == 0
        || reference_case.features == 0) {
        throw std::invalid_argument("RMSNorm reference case has a zero extent");
    }
    if (reference_case.scale_bits.size() != reference_case.features) {
        throw std::invalid_argument(
                "RMSNorm reference case scale bits do not match its features");
    }
    if (reference_case.x_bits.size()
        != reference_case.planes * reference_case.rows
                   * reference_case.features) {
        throw std::invalid_argument(
                "RMSNorm reference case input bits do not match its extents");
    }
    if (plane >= reference_case.planes || row >= reference_case.rows) {
        throw std::invalid_argument(
                "RMSNorm reference case coordinates are out of range");
    }
    return (plane * reference_case.rows + row) * reference_case.features;
}

// Bits of one logical row of the shared input tensor.
inline std::span<const std::uint64_t> row_bits(
        const RmsNormReferenceCase& reference_case, std::size_t plane,
        std::size_t row) {
    const std::span<const std::uint64_t> all(reference_case.x_bits);
    return all.subspan(
            row_offset(reference_case, plane, row), reference_case.features);
}

// ---------------------------------------------------------------------------
// Fixture content.
// ---------------------------------------------------------------------------

// Deterministic repository-owned content. Every pattern value is an FP4 E2M1
// value and is therefore exactly representable in all nine applicable leaves,
// so the pinned cases exercise the arithmetic rather than leaf rounding.
inline constexpr std::array<double, 12> kPatternValues = {
        1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0};

// Per-feature scale pattern: unit and non-unit values with both signs.
inline constexpr std::array<double, 6> kScaleValues = {
        1.0, 2.0, 0.5, -1.0, -2.0, 3.0};

inline double pattern_value(
        std::size_t plane, std::size_t row, std::size_t feature) noexcept {
    return kPatternValues[(feature * 5 + row * 3 + plane * 7)
                          % kPatternValues.size()];
}

inline double scale_value(std::size_t feature) noexcept {
    return kScaleValues[feature % kScaleValues.size()];
}

inline std::vector<double> pattern_content(
        std::size_t planes, std::size_t rows, std::size_t features) {
    std::vector<double> values(planes * rows * features);
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t feature = 0; feature < features; ++feature) {
                values[(plane * rows + row) * features + feature] =
                        pattern_value(plane, row, feature);
            }
        }
    }
    return values;
}

inline std::vector<double> scale_content(std::size_t features) {
    std::vector<double> values(features);
    for (std::size_t feature = 0; feature < features; ++feature) {
        values[feature] = scale_value(feature);
    }
    return values;
}

// Assembles one case; the value tables pass through the independent encoder, so
// a case pins exactly the bits the oracle decodes.
inline RmsNormReferenceCase make_case(
        RmsNormReferenceCaseKind kind, iom::DataType type, std::size_t planes,
        std::size_t rows, std::size_t features, float eps,
        const std::vector<double>& x_values,
        const std::vector<double>& scale_values) {
    RmsNormReferenceCase reference_case;
    reference_case.kind = kind;
    reference_case.data_type = type;
    reference_case.planes = planes;
    reference_case.rows = rows;
    reference_case.features = features;
    reference_case.eps = eps;
    reference_case.x_bits.reserve(x_values.size());
    for (const double value : x_values) {
        reference_case.x_bits.push_back(value_bits(type, value));
    }
    reference_case.scale_bits.reserve(scale_values.size());
    for (const double value : scale_values) {
        reference_case.scale_bits.push_back(value_bits(type, value));
    }
    return reference_case;
}

// Constant ones with unit scale and a non-tile feature width, so the closed
// form also proves the divisor is the logical `F`.
inline RmsNormReferenceCase make_ones_case(iom::DataType type) {
    constexpr std::size_t planes = 2;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 17;
    return make_case(RmsNormReferenceCaseKind::ones, type, planes, rows, features,
                     1.0e-5f, std::vector<double>(planes * rows * features, 1.0),
                     std::vector<double>(features, 1.0));
}

// Positive and negative signed zeros: an all-zero row, zero inputs beside a
// zero scale, and finite features multiplied by positive and negative zeros.
inline RmsNormReferenceCase make_signed_zero_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 6;
    const std::vector<double> x = {
            +0.0, +0.0, -0.0, -0.0, +1.5, -1.5,
            +1.5, -1.5, +2.0, -2.0, +0.5, -0.5};
    const std::vector<double> scale = {+1.0, -1.0, +2.0, -2.0, +0.0, -0.0};
    return make_case(RmsNormReferenceCaseKind::signed_zeros, type, planes, rows,
                     features, 1.0e-5f, x, scale);
}

// All-zero rows with zero epsilon: the reciprocal square root is infinite, so
// every output is a quiet NaN, in a leaf with a NaN encoding or the maximum
// finite value where the leaf has none.
inline RmsNormReferenceCase make_zero_row_zero_eps_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 4;
    const std::vector<double> x = {
            +0.0, -0.0, +0.0, -0.0,
            -0.0, +0.0, -0.0, +0.0};
    const std::vector<double> scale = {+1.0, -1.0, +0.0, +2.0};
    return make_case(RmsNormReferenceCaseKind::zero_row_zero_eps, type, planes,
                     rows, features, 0.0f, x, scale);
}

// Mixed sign and magnitude with unit scale.
inline RmsNormReferenceCase make_mixed_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 8;
    return make_case(RmsNormReferenceCaseKind::mixed, type, planes, rows,
                     features, 1.0e-5f,
                     pattern_content(planes, rows, features),
                     std::vector<double>(features, 1.0));
}

// Mixed sign and magnitude with a non-unit per-feature scale and a smaller
// finite epsilon.
inline RmsNormReferenceCase make_scaled_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 8;
    return make_case(RmsNormReferenceCaseKind::scaled, type, planes, rows,
                     features, 1.0e-6f,
                     pattern_content(planes, rows, features),
                     scale_content(features));
}

// A representable NaN feature poisons its whole row, while the sibling row
// stays finite. Non-tile `F` keeps padded features out of the reduction.
inline RmsNormReferenceCase make_nan_row_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 17;
    std::vector<double> x = pattern_content(planes, rows, features);
    x[2] = std::numeric_limits<double>::quiet_NaN();
    return make_case(RmsNormReferenceCaseKind::nan_row, type, planes, rows,
                     features, 1.0e-5f, x, scale_content(features));
}

// Representable infinities: the row sum is infinite, the finite features
// normalize to signed zeros, and the infinite features become NaN before the
// scale is applied.
inline RmsNormReferenceCase make_infinity_row_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 17;
    std::vector<double> x = pattern_content(planes, rows, features);
    x[2] = std::numeric_limits<double>::infinity();
    x[5] = -std::numeric_limits<double>::infinity();
    return make_case(RmsNormReferenceCaseKind::infinity_row, type, planes, rows,
                     features, 1.0e-5f, x, scale_content(features));
}

// The strongest nonfinite scale the leaf can represent: infinity where the leaf
// has an infinity encoding, otherwise a NaN. Only the nonfinite features may
// change their own output.
inline RmsNormReferenceCase make_nonfinite_scale_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 17;
    std::vector<double> scale = scale_content(features);
    scale[2] = represents_infinity(type)
                       ? std::numeric_limits<double>::infinity()
                       : std::numeric_limits<double>::quiet_NaN();
    if (represents_infinity(type)) {
        scale[5] = std::numeric_limits<double>::quiet_NaN();
    }
    return make_case(RmsNormReferenceCaseKind::nonfinite_scale, type, planes,
                     rows, features, 1.0e-5f,
                     pattern_content(planes, rows, features), scale);
}

// Finite accumulator, destination overflow: the magnified features saturate
// into signed infinities, or into the maximum finite value where the leaf has
// no infinity encoding.
inline RmsNormReferenceCase make_destination_overflow_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 1;
    constexpr std::size_t features = 4;
    const double maximum = leaf_max_finite(type);
    const std::vector<double> x = {+1.0, -1.0, +0.0, -0.0};
    const std::vector<double> scale = {maximum, maximum, +1.0, +1.0};
    return make_case(RmsNormReferenceCaseKind::destination_overflow, type, planes,
                     rows, features, 1.0e-5f, x, scale);
}

// Finite accumulator, destination underflow: one feature's exact result lies
// far below the smallest subnormal of the leaf and therefore stores signed zero
// while its neighbours stay finite.
inline RmsNormReferenceCase make_destination_underflow_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 1;
    constexpr std::size_t features = 4;
    const double minimum = leaf_min_positive(type);
    // A large but finite feature keeps the accumulator finite: the narrow
    // encoded leaves cannot hold 1e6 and would otherwise saturate to infinity.
    const double large = std::min(1.0e6, leaf_max_finite(type));
    const std::vector<double> x = {minimum, large, +0.0, -0.0};
    const std::vector<double> scale = {minimum, +1.0, +1.0, +1.0};
    return make_case(RmsNormReferenceCaseKind::destination_underflow, type, planes,
                     rows, features, 1.0e-5f, x, scale);
}

// Squares that overflow the accumulator: the row sum is infinite, the
// reciprocal square root is positive zero, and every output is a signed zero.
inline RmsNormReferenceCase make_accumulator_overflow_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 4;
    const double overflowing =
            type == iom::DataType::F64 ? 1.0e200 : 3.0e19;
    const std::vector<double> x = {
            +overflowing, -overflowing, +overflowing, -overflowing,
            -overflowing, +overflowing, -overflowing, +overflowing};
    const std::vector<double> scale = {+1.0, -1.0, +2.0, -2.0};
    return make_case(RmsNormReferenceCaseKind::accumulator_overflow, type, planes,
                     rows, features, 1.0e-5f, x, scale);
}

// Squares that underflow to zero with zero epsilon: the row sum stays finite,
// the reciprocal square root is infinite, and every nonzero feature stores a
// signed infinity.
inline RmsNormReferenceCase make_accumulator_underflow_case(iom::DataType type) {
    constexpr std::size_t planes = 1;
    constexpr std::size_t rows = 2;
    constexpr std::size_t features = 4;
    const double tiny = type == iom::DataType::F64 ? std::ldexp(1.0, -600)
                                                   : std::ldexp(1.0, -80);
    const std::vector<double> x = {
            +tiny, -tiny, +tiny, -tiny,
            -tiny, +tiny, -tiny, +tiny};
    const std::vector<double> scale = {+1.0, -1.0, +2.0, -2.0};
    return make_case(RmsNormReferenceCaseKind::accumulator_underflow, type, planes,
                     rows, features, 0.0f, x, scale);
}

// Independent planes and rows across the boundary sizes of the tiled layout.
inline RmsNormReferenceCase make_boundary_sizes_case(
        iom::DataType type, std::size_t rows, std::size_t features) {
    constexpr std::size_t planes = 3;
    return make_case(RmsNormReferenceCaseKind::boundary_sizes, type, planes, rows,
                     features, 1.0e-5f,
                     pattern_content(planes, rows, features),
                     scale_content(features));
}

}  // namespace rmsnorm_oracle

// Non-tile feature widths and boundary row counts of the pinned size sweep,
// chosen around the fixed 16x16 tile so padded features and rows cannot enter
// the equation.
inline constexpr std::array<std::size_t, 6> kRmsNormBoundaryFeatures = {
        1, 15, 16, 17, 31, 33};
inline constexpr std::array<std::size_t, 4> kRmsNormBoundaryRows = {
        1, 15, 16, 17};

// ---------------------------------------------------------------------------
// Consumable API.
// ---------------------------------------------------------------------------

// One expected value per logical output element of the case's `[planes, R, F]`
// result, in row-major order. An unsupported leaf carries a rejection
// expectation instead of arithmetic and therefore returns no values. A case
// whose declared extents and bit counts disagree is a fixture defect and throws
// std::invalid_argument.
inline std::vector<RmsNormReferenceValue> evaluate(
        const RmsNormReferenceCase& reference_case) {
    std::vector<RmsNormReferenceValue> values;
    if (rmsnorm_data_type_classification(reference_case.data_type)
        != RmsNormDataTypeClass::applicable) {
        return values;
    }
    values.resize(reference_case.planes * reference_case.rows
                  * reference_case.features);
    const std::span<const std::uint64_t> scale_bits(reference_case.scale_bits);
    for (std::size_t plane = 0; plane < reference_case.planes; ++plane) {
        for (std::size_t row = 0; row < reference_case.rows; ++row) {
            const std::span<const std::uint64_t> x_bits =
                    rmsnorm_oracle::row_bits(reference_case, plane, row);
            const std::span<RmsNormReferenceValue> output(
                    values.data()
                            + (plane * reference_case.rows + row)
                                      * reference_case.features,
                    reference_case.features);
            if (reference_case.data_type == iom::DataType::F64) {
                const rmsnorm_oracle::RowAccumulator<double> accumulator =
                        rmsnorm_oracle::accumulate_row_f64(
                                x_bits, reference_case.eps);
                rmsnorm_oracle::encode_row_f64(
                        x_bits, scale_bits, accumulator.inverse, output);
            } else {
                const rmsnorm_oracle::RowAccumulator<float> accumulator =
                        rmsnorm_oracle::accumulate_row_f32(
                                reference_case.data_type, x_bits,
                                reference_case.eps);
                rmsnorm_oracle::encode_row_f32(
                        reference_case.data_type, x_bits, scale_bits,
                        accumulator.inverse, output);
            }
        }
    }
    return values;
}

// Accumulator-domain class of one logical row, independent of the destination
// leaf. Requires an applicable leaf and in-range coordinates, and otherwise
// throws std::invalid_argument.
inline RmsNormAccumulatorClass rmsnorm_accumulator_class(
        const RmsNormReferenceCase& reference_case, std::size_t plane,
        std::size_t row) {
    if (rmsnorm_data_type_classification(reference_case.data_type)
        != RmsNormDataTypeClass::applicable) {
        throw std::invalid_argument(
                "RMSNorm accumulator class requires an applicable leaf");
    }
    const std::span<const std::uint64_t> x_bits =
            rmsnorm_oracle::row_bits(reference_case, plane, row);
    if (reference_case.data_type == iom::DataType::F64) {
        return rmsnorm_oracle::accumulator_class_of(
                rmsnorm_oracle::accumulate_row_f64(x_bits, reference_case.eps)
                        .sum);
    }
    return rmsnorm_oracle::accumulator_class_of(
            rmsnorm_oracle::accumulate_row_f32(
                    reference_case.data_type, x_bits, reference_case.eps)
                    .sum);
}

// A deliberately perturbed expectation: the values of one logical row replaced
// by its neighbouring row's values, which must fail the comparison. The caller
// must name a case with at least two rows and rows whose expectations differ;
// anything else throws std::invalid_argument.
inline std::vector<RmsNormReferenceValue> rmsnorm_wrong_row_expected(
        const RmsNormReferenceCase& reference_case, std::size_t plane,
        std::size_t row) {
    if (reference_case.rows < 2) {
        throw std::invalid_argument(
                "a wrong-row expectation requires at least two rows");
    }
    const std::size_t offset =
            rmsnorm_oracle::row_offset(reference_case, plane, row);
    const std::size_t neighbour = rmsnorm_oracle::row_offset(
            reference_case, plane, (row + 1) % reference_case.rows);
    std::vector<RmsNormReferenceValue> values = evaluate(reference_case);
    if (values.empty()) {
        throw std::invalid_argument(
                "a wrong-row expectation requires an applicable leaf");
    }
    std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(neighbour),
                static_cast<std::ptrdiff_t>(reference_case.features),
                values.begin() + static_cast<std::ptrdiff_t>(offset));
    return values;
}

// Fixed comparison policy, implemented before any backend measurement:
//   * the class of the measured leaf value must equal the expected class, which
//     compares finite against nonfinite, infinities with their sign, and
//     positive against negative zero exactly while ignoring NaN payloads;
//   * finite F4/F6/F8/F16/BF16 must stay within two adjacent destination
//     encodings of the expected value;
//   * F32 accepts `abs_err <= 1e-6 + 2e-5 * abs(reference)`;
//   * F64 accepts `abs_err <= 1e-15 + 1e-12 * abs(reference)`.
// Actual bits are decoded independently here; production conversions are never
// used. An unsupported leaf never matches.
inline bool matches(
        iom::DataType type, std::uint64_t actual_bits,
        const RmsNormReferenceValue& expected) noexcept {
    if (type == iom::DataType::F64) {
        const double actual = rmsnorm_oracle::decode_f64(actual_bits);
        if (rmsnorm_oracle::classify(actual) != expected.value_class) {
            return false;
        }
        if (expected.value_class != RmsNormReferenceClass::finite) {
            return true;
        }
        const double reference = rmsnorm_oracle::decode_f64(expected.bits);
        const double error = std::fabs(actual - reference);
        return error <= 1.0e-15 + 1.0e-12 * std::fabs(reference);
    }
    const rmsnorm_oracle::FloatFormat format =
            rmsnorm_oracle::float_format(type);
    if (format.bits == 0) {
        return false;
    }
    const std::uint64_t width_mask =
            format.bits >= 64 ? ~std::uint64_t{0}
                              : (std::uint64_t{1} << format.bits) - 1;
    const std::uint64_t actual_pattern = actual_bits & width_mask;
    const std::uint64_t expected_pattern = expected.bits & width_mask;
    const float actual = rmsnorm_oracle::decode_f32(type, actual_pattern);
    if (rmsnorm_oracle::classify(actual) != expected.value_class) {
        return false;
    }
    if (expected.value_class != RmsNormReferenceClass::finite) {
        return true;
    }
    if (type == iom::DataType::F32) {
        const double reference = static_cast<double>(
                rmsnorm_oracle::decode_f32(type, expected_pattern));
        const double error =
                std::fabs(static_cast<double>(actual) - reference);
        return error <= 1.0e-6 + 2.0e-5 * std::fabs(reference);
    }
    return rmsnorm_oracle::ordered_distance(
                   actual_pattern, expected_pattern, format.bits)
           <= 2;
}

// Deterministic fixtures for one leaf. Applicable leaves cover ones, signed
// zeros, zero-epsilon zero rows, mixed sign and magnitude, non-unit scale,
// representable NaN and infinity, destination saturation and underflow,
// accumulator overflow and underflow where the accumulator can reach them, and
// the complete boundary-size sweep. Unsupported leaves return no cases: their
// expectation is rejection, never coerced arithmetic.
inline std::vector<RmsNormReferenceCase> rmsnorm_reference_cases(
        iom::DataType type) {
    std::vector<RmsNormReferenceCase> cases;
    if (rmsnorm_data_type_classification(type)
        != RmsNormDataTypeClass::applicable) {
        return cases;
    }
    cases.push_back(rmsnorm_oracle::make_ones_case(type));
    cases.push_back(rmsnorm_oracle::make_signed_zero_case(type));
    cases.push_back(rmsnorm_oracle::make_zero_row_zero_eps_case(type));
    cases.push_back(rmsnorm_oracle::make_mixed_case(type));
    cases.push_back(rmsnorm_oracle::make_scaled_case(type));
    if (rmsnorm_oracle::represents_nan(type)) {
        cases.push_back(rmsnorm_oracle::make_nan_row_case(type));
        cases.push_back(rmsnorm_oracle::make_nonfinite_scale_case(type));
    }
    if (rmsnorm_oracle::represents_infinity(type)) {
        cases.push_back(rmsnorm_oracle::make_infinity_row_case(type));
    }
    cases.push_back(rmsnorm_oracle::make_destination_overflow_case(type));
    cases.push_back(rmsnorm_oracle::make_destination_underflow_case(type));
    if (rmsnorm_oracle::accumulator_can_overflow(type)) {
        cases.push_back(rmsnorm_oracle::make_accumulator_overflow_case(type));
    }
    if (rmsnorm_oracle::accumulator_can_underflow(type)) {
        cases.push_back(rmsnorm_oracle::make_accumulator_underflow_case(type));
    }
    for (const std::size_t rows : kRmsNormBoundaryRows) {
        for (const std::size_t features : kRmsNormBoundaryFeatures) {
            cases.push_back(
                    rmsnorm_oracle::make_boundary_sizes_case(type, rows, features));
        }
    }
    return cases;
}

// ---------------------------------------------------------------------------
// Analytic self-checks.
// ---------------------------------------------------------------------------

// Result of the backend-free analytic self-checks: the number of executed
// checks and one message per failed check.
struct RmsNormReferenceSelfCheckReport {
    std::size_t checks = 0;
    std::vector<std::string> failures;

    [[nodiscard]] bool ok() const noexcept { return failures.empty(); }
};

// Verifies, without a backend or production operation, that the classification
// partitions all 23 leaves, that every applicable leaf pins the required
// fixtures, that every expectation is a canonical encoding whose class matches
// its bits and matches its own comparison, that the all-ones rows follow
// `1/sqrt(1+eps) * scale`, that signed zeros keep their analytic sign, that a
// zero row with zero epsilon is a quiet NaN decision, that a nonfinite scale
// changes only its own feature, that planes and rows are independent, that the
// accumulator overflow and underflow decisions hold, that a deliberately
// wrong-row vector fails comparison, and that the fixed tolerances accept and
// reject exactly their stated margins.
inline RmsNormReferenceSelfCheckReport rmsnorm_reference_self_check() {
    RmsNormReferenceSelfCheckReport report;
    const auto require = [&report](bool condition, const std::string& message) {
        ++report.checks;
        if (!condition) {
            report.failures.push_back(message);
        }
    };
    const auto element_value = [](iom::DataType type, std::uint64_t bits) {
        return type == iom::DataType::F64
                       ? rmsnorm_oracle::decode_f64(bits)
                       : static_cast<double>(
                                 rmsnorm_oracle::decode_f32(type, bits));
    };
    const auto canonical_bits = [](iom::DataType type, std::uint64_t bits) {
        return type == iom::DataType::F64
                       ? rmsnorm_oracle::encode_f64(
                                 rmsnorm_oracle::decode_f64(bits))
                       : rmsnorm_oracle::encode_f32(
                                 type, rmsnorm_oracle::decode_f32(type, bits));
    };
    const auto occurrences = [](const auto& leaves, iom::DataType type) {
        std::size_t count = 0;
        for (const iom::DataType leaf : leaves) {
            if (leaf == type) {
                ++count;
            }
        }
        return count;
    };
    const auto is_zero_class = [](RmsNormReferenceClass value_class) {
        return value_class == RmsNormReferenceClass::positive_zero
               || value_class == RmsNormReferenceClass::negative_zero;
    };
    const auto is_infinite_class = [](RmsNormReferenceClass value_class) {
        return value_class == RmsNormReferenceClass::positive_infinity
               || value_class == RmsNormReferenceClass::negative_infinity;
    };

    // 1. Classification: the literals and the explicit classification agree and
    //    partition all 23 leaves, and unsupported leaves stay arithmetic-free.
    require(kRmsNormApplicableDataTypes.size()
                            + kRmsNormUnsupportedDataTypes.size()
                    == 23,
            "classification matrix must cover all 23 leaves");
    for (const iom::DataType type : kRmsNormApplicableDataTypes) {
        const std::string leaf(rmsnorm_oracle::leaf_name(type));
        require(rmsnorm_data_type_classification(type)
                        == RmsNormDataTypeClass::applicable,
                leaf + ": applicable leaf must classify as applicable");
        require(occurrences(kRmsNormApplicableDataTypes, type) == 1
                        && occurrences(kRmsNormUnsupportedDataTypes, type) == 0,
                leaf + ": applicable leaf must appear exactly once");
        require(!rmsnorm_reference_cases(type).empty(),
                leaf + ": applicable leaf must provide reference cases");
    }
    for (const iom::DataType type : kRmsNormUnsupportedDataTypes) {
        const std::string leaf(rmsnorm_oracle::leaf_name(type));
        require(rmsnorm_data_type_classification(type)
                        == RmsNormDataTypeClass::unsupported,
                leaf + ": unsupported leaf must classify as unsupported");
        require(occurrences(kRmsNormUnsupportedDataTypes, type) == 1
                        && occurrences(kRmsNormApplicableDataTypes, type) == 0,
                leaf + ": unsupported leaf must appear exactly once");
        require(rmsnorm_reference_cases(type).empty(),
                leaf + ": unsupported leaf must carry no numeric case");
    }
    {
        RmsNormReferenceCase integral_case;
        integral_case.data_type = iom::DataType::I8;
        integral_case.planes = 1;
        integral_case.rows = 1;
        integral_case.features = 1;
        integral_case.eps = 1.0e-5f;
        integral_case.x_bits = {0};
        integral_case.scale_bits = {0};
        require(evaluate(integral_case).empty(),
                "unsupported leaves must never produce arithmetic");
    }

    // 2. Fixtures, expected values, and comparisons for every applicable leaf.
    for (const iom::DataType type : kRmsNormApplicableDataTypes) {
        const std::string leaf(rmsnorm_oracle::leaf_name(type));
        const std::vector<RmsNormReferenceCase> cases =
                rmsnorm_reference_cases(type);
        for (const RmsNormReferenceCaseKind required :
             {RmsNormReferenceCaseKind::ones,
              RmsNormReferenceCaseKind::signed_zeros,
              RmsNormReferenceCaseKind::zero_row_zero_eps,
              RmsNormReferenceCaseKind::mixed,
              RmsNormReferenceCaseKind::scaled,
              RmsNormReferenceCaseKind::destination_overflow,
              RmsNormReferenceCaseKind::destination_underflow}) {
            std::size_t count = 0;
            for (const RmsNormReferenceCase& reference_case : cases) {
                if (reference_case.kind == required) {
                    ++count;
                }
            }
            require(count == 1,
                    leaf + ": exactly one "
                            + std::string(
                                      rmsnorm_reference_case_kind_name(required))
                            + " fixture is required");
        }
        std::size_t boundary_cases = 0;
        for (const RmsNormReferenceCase& reference_case : cases) {
            if (reference_case.kind == RmsNormReferenceCaseKind::boundary_sizes) {
                ++boundary_cases;
            }
        }
        require(boundary_cases
                        == kRmsNormBoundaryFeatures.size()
                                   * kRmsNormBoundaryRows.size(),
                leaf + ": the boundary sweep must cover every F and R pairing");

        std::size_t perturbed_cases = 0;
        for (const RmsNormReferenceCase& reference_case : cases) {
            const std::string label =
                    leaf + " " + rmsnorm_reference_case_label(reference_case);
            const std::vector<RmsNormReferenceValue> expected =
                    evaluate(reference_case);
            const std::size_t element_count = reference_case.planes
                                              * reference_case.rows
                                              * reference_case.features;
            require(expected.size() == element_count,
                    label + ": one expected value per logical element");
            const std::span<const std::uint64_t> scale_bits(
                    reference_case.scale_bits);
            const std::uint64_t sign_pattern = std::uint64_t{1}
                    << (rmsnorm_oracle::float_format(type).bits - 1);
            std::size_t positive_zeros = 0;
            std::size_t negative_zeros = 0;
            for (std::size_t index = 0; index < expected.size(); ++index) {
                const RmsNormReferenceValue& value = expected[index];
                const RmsNormReferenceClass stored =
                        rmsnorm_oracle::classify(
                                element_value(type, value.bits));
                require(canonical_bits(type, value.bits) == value.bits,
                        label + ": element " + std::to_string(index)
                                + " is not a canonical leaf encoding");
                require(stored == value.value_class,
                        label + ": element " + std::to_string(index) + " stores "
                                + std::string(
                                          rmsnorm_reference_class_name(stored))
                                + " but claims "
                                + std::string(rmsnorm_reference_class_name(
                                          value.value_class)));
                require(matches(type, value.bits, value),
                        label + ": element " + std::to_string(index)
                                + " must match its own expectation");
                positive_zeros += value.value_class
                                          == RmsNormReferenceClass::positive_zero
                                  ? 1
                                  : 0;
                negative_zeros += value.value_class
                                          == RmsNormReferenceClass::negative_zero
                                  ? 1
                                  : 0;
            }

            if (reference_case.kind == RmsNormReferenceCaseKind::ones) {
                const double closed_form = 1.0
                        / std::sqrt(1.0 + static_cast<double>(reference_case.eps));
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    const std::size_t feature =
                            index % reference_case.features;
                    const double reference =
                            closed_form
                            * element_value(type, scale_bits[feature]);
                    require(expected[index].value_class
                                    == RmsNormReferenceClass::finite,
                            label + ": an all-ones row must stay finite at element "
                                    + std::to_string(index));
                    require(matches(type,
                                    rmsnorm_oracle::value_bits(type, reference),
                                    expected[index]),
                            label + ": ones identity 1/sqrt(1+eps)*scale fails "
                                    "at element "
                                    + std::to_string(index));
                }
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::signed_zeros) {
                require(positive_zeros > 0 && negative_zeros > 0,
                        label + ": both signed zeros must be pinned");
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    const RmsNormReferenceValue& value = expected[index];
                    if (!is_zero_class(value.value_class)) {
                        continue;
                    }
                    const bool negative =
                            value.value_class
                            == RmsNormReferenceClass::negative_zero;
                    require(value.bits == (negative ? sign_pattern : 0),
                            label + ": a signed zero must be the pure sign "
                                    "encoding at element "
                                    + std::to_string(index));
                    const bool input_negative = std::signbit(
                            element_value(type, reference_case.x_bits[index]));
                    const bool scale_negative = std::signbit(element_value(
                            type, scale_bits[index % reference_case.features]));
                    require((input_negative != scale_negative) == negative,
                            label + ": a signed zero must follow the input and "
                                    "scale signs at element "
                                    + std::to_string(index));
                }
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::zero_row_zero_eps) {
                const bool nan_encoding = rmsnorm_oracle::has_nan_encoding(
                        rmsnorm_oracle::float_format(type));
                for (std::size_t plane = 0; plane < reference_case.planes;
                     ++plane) {
                    for (std::size_t row = 0; row < reference_case.rows; ++row) {
                        require(rmsnorm_accumulator_class(reference_case, plane, row)
                                        == RmsNormAccumulatorClass::finite_sum,
                                label + ": an all-zero row keeps a finite "
                                        "accumulator sum");
                    }
                }
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    if (nan_encoding) {
                        require(expected[index].value_class
                                        == RmsNormReferenceClass::quiet_nan,
                                label + ": zero epsilon with an all-zero row must "
                                        "produce a quiet NaN at element "
                                        + std::to_string(index));
                    } else {
                        require(expected[index].value_class
                                                == RmsNormReferenceClass::finite
                                        && expected[index].bits
                                                   == rmsnorm_oracle::encode_f32(
                                                           type,
                                                           std::numeric_limits<float>::max()),
                                label + ": a NaN-less leaf must store its maximum "
                                        "finite value at element "
                                        + std::to_string(index));
                    }
                }
            }

            if (reference_case.kind == RmsNormReferenceCaseKind::nan_row) {
                // A NaN feature poisons its whole row, and its siblings stay
                // finite: the reduction is row-local.
                std::size_t poisoned_rows = 0;
                for (std::size_t plane = 0; plane < reference_case.planes;
                     ++plane) {
                    for (std::size_t row = 0; row < reference_case.rows; ++row) {
                        const std::size_t offset = rmsnorm_oracle::row_offset(
                                reference_case, plane, row);
                        bool row_has_nan = false;
                        for (std::size_t feature = 0;
                             feature < reference_case.features; ++feature) {
                            row_has_nan =
                                    row_has_nan
                                    || std::isnan(element_value(
                                            type,
                                            reference_case.x_bits[offset
                                                                  + feature]));
                        }
                        const RmsNormAccumulatorClass sum_class =
                                rmsnorm_accumulator_class(reference_case, plane,
                                                          row);
                        if (!row_has_nan) {
                            require(sum_class
                                            == RmsNormAccumulatorClass::finite_sum,
                                    label + ": an unpoisoned row keeps a finite "
                                            "accumulator sum");
                            continue;
                        }
                        ++poisoned_rows;
                        require(sum_class
                                        == RmsNormAccumulatorClass::quiet_nan_sum,
                                label + ": a NaN feature must poison the row sum");
                        for (std::size_t feature = 0;
                             feature < reference_case.features; ++feature) {
                            require(expected[offset + feature].value_class
                                            == RmsNormReferenceClass::quiet_nan,
                                    label + ": a NaN feature must poison every "
                                            "output of its row");
                        }
                    }
                }
                require(poisoned_rows > 0,
                        label + ": a NaN fixture must pin a poisoned row");
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::infinity_row) {
                // An infinite feature makes the row sum infinite, so the
                // reciprocal square root is positive zero: finite features
                // normalize to signed zeros and infinite features become NaN.
                std::size_t infinite_rows = 0;
                for (std::size_t plane = 0; plane < reference_case.planes;
                     ++plane) {
                    for (std::size_t row = 0; row < reference_case.rows; ++row) {
                        const std::size_t offset = rmsnorm_oracle::row_offset(
                                reference_case, plane, row);
                        bool row_has_infinity = false;
                        for (std::size_t feature = 0;
                             feature < reference_case.features; ++feature) {
                            row_has_infinity =
                                    row_has_infinity
                                    || std::isinf(element_value(
                                            type,
                                            reference_case.x_bits[offset
                                                                  + feature]));
                        }
                        if (!row_has_infinity) {
                            require(rmsnorm_accumulator_class(reference_case,
                                                              plane, row)
                                            == RmsNormAccumulatorClass::finite_sum,
                                    label + ": a finite row keeps a finite "
                                            "accumulator sum");
                            continue;
                        }
                        ++infinite_rows;
                        require(rmsnorm_accumulator_class(reference_case, plane,
                                                          row)
                                        == RmsNormAccumulatorClass::infinite_sum,
                                label + ": an infinite feature makes the row sum "
                                        "infinite");
                        for (std::size_t feature = 0;
                             feature < reference_case.features; ++feature) {
                            const double feature_value = element_value(
                                    type,
                                    reference_case.x_bits[offset + feature]);
                            const double weight = element_value(
                                    type, scale_bits[feature]);
                            if (std::isinf(feature_value)) {
                                require(expected[offset + feature].value_class
                                                == RmsNormReferenceClass::quiet_nan,
                                        label + ": an infinite feature must become "
                                                "NaN before the scale at element "
                                                + std::to_string(offset
                                                                 + feature));
                                continue;
                            }
                            const bool negative =
                                    std::signbit(feature_value)
                                    != std::signbit(weight);
                            require(is_zero_class(
                                            expected[offset + feature]
                                                    .value_class)
                                            && expected[offset + feature].bits
                                                       == (negative ? sign_pattern
                                                                    : 0),
                                    label + ": a finite feature beside an "
                                            "infinity must normalize to a signed "
                                            "zero at element "
                                            + std::to_string(offset + feature));
                        }
                    }
                }
                require(infinite_rows > 0,
                        label + ": an infinity fixture must pin an infinite row");
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::nonfinite_scale) {
                RmsNormReferenceCase finite_scale = reference_case;
                std::size_t nonfinite_features = 0;
                for (std::size_t feature = 0; feature < reference_case.features;
                     ++feature) {
                    if (std::isfinite(element_value(type, scale_bits[feature]))) {
                        continue;
                    }
                    ++nonfinite_features;
                    finite_scale.scale_bits[feature] =
                            rmsnorm_oracle::value_bits(type, 1.0);
                }
                require(nonfinite_features > 0,
                        label + ": a nonfinite scale fixture must pin a "
                                "nonfinite feature");
                const std::vector<RmsNormReferenceValue> finite_expected =
                        evaluate(finite_scale);
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    const std::size_t feature =
                            index % reference_case.features;
                    if (!std::isfinite(
                                element_value(type, scale_bits[feature]))) {
                        require(is_infinite_class(expected[index].value_class)
                                        || expected[index].value_class
                                                   == RmsNormReferenceClass::quiet_nan,
                                label + ": a nonfinite scale feature must produce "
                                        "a nonfinite value at element "
                                        + std::to_string(index));
                        continue;
                    }
                    require(expected[index].bits == finite_expected[index].bits
                                    && expected[index].value_class
                                               == finite_expected[index].value_class,
                            label + ": a nonfinite scale feature changed another "
                                    "feature at element "
                                    + std::to_string(index));
                }
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::destination_overflow) {
                for (std::size_t row = 0; row < reference_case.rows; ++row) {
                    require(rmsnorm_accumulator_class(reference_case, 0, row)
                                    == RmsNormAccumulatorClass::finite_sum,
                            label + ": destination saturation keeps a finite "
                                    "accumulator");
                }
                if (rmsnorm_oracle::represents_infinity(type)) {
                    require(expected[0].value_class
                                            == RmsNormReferenceClass::positive_infinity
                                    && expected[1].value_class
                                               == RmsNormReferenceClass::negative_infinity,
                            label + ": an infinity-capable leaf must overflow to "
                                    "signed infinities");
                } else {
                    require(expected[0].bits
                                            == rmsnorm_oracle::encode_f32(
                                                    type,
                                                    std::numeric_limits<float>::max())
                                    && expected[1].bits
                                               == rmsnorm_oracle::encode_f32(
                                                       type,
                                                       -std::numeric_limits<float>::max()),
                            label + ": a finite-only leaf must saturate to its "
                                    "maximum finite value");
                }
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::destination_underflow) {
                for (std::size_t row = 0; row < reference_case.rows; ++row) {
                    require(rmsnorm_accumulator_class(reference_case, 0, row)
                                    == RmsNormAccumulatorClass::finite_sum,
                            label + ": destination underflow keeps a finite "
                                    "accumulator");
                }
                require(expected[0].value_class
                                        == RmsNormReferenceClass::positive_zero
                                && expected[0].bits == 0,
                        label + ": a destination underflow must round to signed "
                                "zero");
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::accumulator_overflow) {
                for (std::size_t row = 0; row < reference_case.rows; ++row) {
                    require(rmsnorm_accumulator_class(reference_case, 0, row)
                                    == RmsNormAccumulatorClass::infinite_sum,
                            label + ": an overflowing row sum must be infinite");
                }
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    const bool negative = expected[index].value_class
                                          == RmsNormReferenceClass::negative_zero;
                    require(is_zero_class(expected[index].value_class)
                                    && expected[index].bits
                                               == (negative ? sign_pattern : 0),
                            label + ": an infinite row sum must normalize to "
                                    "signed zeros at element "
                                    + std::to_string(index));
                    const bool input_negative = std::signbit(
                            element_value(type, reference_case.x_bits[index]));
                    const bool scale_negative = std::signbit(element_value(
                            type, scale_bits[index % reference_case.features]));
                    require((input_negative != scale_negative) == negative,
                            label + ": an infinite row sum must keep the input "
                                    "and scale signs at element "
                                    + std::to_string(index));
                }
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::accumulator_underflow) {
                for (std::size_t row = 0; row < reference_case.rows; ++row) {
                    require(rmsnorm_accumulator_class(reference_case, 0, row)
                                    == RmsNormAccumulatorClass::finite_sum,
                            label + ": an underflowing row sum stays finite");
                }
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    const double feature =
                            element_value(type, reference_case.x_bits[index]);
                    require(std::isfinite(feature) && feature != 0.0,
                            label + ": an accumulator underflow fixture needs "
                                    "nonzero finite features");
                    const bool negative = expected[index].value_class
                                          == RmsNormReferenceClass::negative_infinity;
                    require(is_infinite_class(expected[index].value_class),
                            label + ": an infinite reciprocal square root must "
                                    "produce a signed infinity at element "
                                    + std::to_string(index));
                    const bool input_negative = std::signbit(feature);
                    const bool scale_negative = std::signbit(element_value(
                            type, scale_bits[index % reference_case.features]));
                    require((input_negative != scale_negative) == negative,
                            label + ": an infinite reciprocal square root must "
                                    "keep the input and scale signs at element "
                                    + std::to_string(index));
                }
            }

            if (reference_case.kind
                == RmsNormReferenceCaseKind::boundary_sizes) {
                // Plane and row independence: a row evaluated alone must
                // reproduce exactly its slice of the shared-scale case.
                const std::array<std::pair<std::size_t, std::size_t>, 4>
                        coordinates = {{
                                {0, 0},
                                {reference_case.planes > 1 ? 1 : 0, 0},
                                {0, reference_case.rows > 1 ? 1 : 0},
                                {reference_case.planes - 1,
                                 reference_case.rows - 1},
                        }};
                for (const auto& coordinate : coordinates) {
                    const std::size_t offset = rmsnorm_oracle::row_offset(
                            reference_case, coordinate.first, coordinate.second);
                    RmsNormReferenceCase single_row = reference_case;
                    single_row.planes = 1;
                    single_row.rows = 1;
                    single_row.x_bits.assign(
                            reference_case.x_bits.begin()
                                    + static_cast<std::ptrdiff_t>(offset),
                            reference_case.x_bits.begin()
                                    + static_cast<std::ptrdiff_t>(
                                            offset + reference_case.features));
                    const std::vector<RmsNormReferenceValue> single_expected =
                            evaluate(single_row);
                    for (std::size_t feature = 0;
                         feature < reference_case.features; ++feature) {
                        require(single_expected[feature].bits
                                                == expected[offset + feature].bits
                                        && single_expected[feature].value_class
                                                   == expected[offset + feature]
                                                              .value_class,
                                label + ": plane/row independence fails at plane "
                                        + std::to_string(coordinate.first)
                                        + " row "
                                        + std::to_string(coordinate.second)
                                        + " feature "
                                        + std::to_string(feature));
                    }
                }
            }

            // A deliberately perturbed wrong-row vector must fail comparison
            // wherever the fixture pins differing rows.
            if (reference_case.rows >= 2) {
                const std::size_t first =
                        rmsnorm_oracle::row_offset(reference_case, 0, 0);
                const std::size_t second =
                        rmsnorm_oracle::row_offset(reference_case, 0, 1);
                bool rows_differ = false;
                for (std::size_t feature = 0;
                     feature < reference_case.features; ++feature) {
                    rows_differ = rows_differ
                                  || expected[first + feature].bits
                                             != expected[second + feature].bits;
                }
                if (rows_differ) {
                    ++perturbed_cases;
                    const std::vector<RmsNormReferenceValue> wrong =
                            rmsnorm_wrong_row_expected(reference_case, 0, 0);
                    std::size_t rejected = 0;
                    for (std::size_t feature = 0;
                         feature < reference_case.features; ++feature) {
                        if (!matches(type, expected[first + feature].bits,
                                     wrong[first + feature])) {
                            ++rejected;
                        }
                    }
                    require(rejected > 0,
                            label + ": a deliberately wrong row must fail "
                                    "comparison");
                }
            }
        }
        require(perturbed_cases > 0,
                leaf + ": at least one fixture must pin differing rows");
    }

    // 3. Fixed tolerances and class rejection, pinned before any measurement.
    {
        RmsNormReferenceValue one;
        one.bits = rmsnorm_oracle::encode_f32(iom::DataType::F32, 1.0f);
        one.value_class = RmsNormReferenceClass::finite;
        require(!matches(iom::DataType::F32,
                         rmsnorm_oracle::encode_f32(iom::DataType::F32,
                                                    1.0f + 1.0e-2f),
                         one),
                "F32: an error above 1e-6 + 2e-5*abs(reference) must be rejected");
        require(matches(iom::DataType::F32,
                        rmsnorm_oracle::encode_f32(iom::DataType::F32,
                                                   1.0f + 1.0e-6f),
                        one),
                "F32: an error inside 1e-6 + 2e-5*abs(reference) must pass");

        RmsNormReferenceValue double_one;
        double_one.bits = rmsnorm_oracle::encode_f64(1.0);
        double_one.value_class = RmsNormReferenceClass::finite;
        require(!matches(iom::DataType::F64,
                         rmsnorm_oracle::encode_f64(1.0 + 1.0e-9), double_one),
                "F64: an error above 1e-15 + 1e-12*abs(reference) must be rejected");
        require(matches(iom::DataType::F64,
                        rmsnorm_oracle::encode_f64(1.0 + 1.0e-14), double_one),
                "F64: an error inside 1e-15 + 1e-12*abs(reference) must pass");

        RmsNormReferenceValue half_one;
        half_one.bits = rmsnorm_oracle::encode_f32(iom::DataType::F16, 1.0f);
        half_one.value_class = RmsNormReferenceClass::finite;
        require(matches(iom::DataType::F16,
                        rmsnorm_oracle::encode_f32(
                                iom::DataType::F16, 1.0f + 2.0f * 0x1p-10f),
                        half_one),
                "F16: two adjacent destination encodings must pass");
        require(!matches(iom::DataType::F16,
                         rmsnorm_oracle::encode_f32(
                                 iom::DataType::F16, 1.0f + 3.0f * 0x1p-10f),
                         half_one),
                "F16: three adjacent destination encodings must be rejected");

        require(!matches(iom::DataType::F32,
                         rmsnorm_oracle::encode_f32(
                                 iom::DataType::F32,
                                 std::numeric_limits<float>::infinity()),
                         one),
                "a nonfinite measurement must not match a finite expectation");
        RmsNormReferenceValue nan_value;
        nan_value.bits = rmsnorm_oracle::encode_f32(
                iom::DataType::F32, std::numeric_limits<float>::quiet_NaN());
        nan_value.value_class = RmsNormReferenceClass::quiet_nan;
        require(!matches(iom::DataType::F32,
                         rmsnorm_oracle::encode_f32(iom::DataType::F32, 1.0f),
                         nan_value),
                "a finite measurement must not match a NaN expectation");
        require(matches(iom::DataType::F32,
                        rmsnorm_oracle::encode_f32(
                                iom::DataType::F32,
                                std::numeric_limits<float>::quiet_NaN()),
                        nan_value),
                "NaN payloads must be ignored");
        RmsNormReferenceValue negative_zero;
        negative_zero.bits =
                rmsnorm_oracle::encode_f32(iom::DataType::F32, -0.0f);
        negative_zero.value_class = RmsNormReferenceClass::negative_zero;
        require(matches(iom::DataType::F32, negative_zero.bits, negative_zero),
                "a signed zero must match itself");
        require(!matches(iom::DataType::F32, negative_zero.bits ^ 1u,
                         negative_zero),
                "a nonzero measurement must not match a signed zero expectation");
    }

    // 4. Codec round trip and exact step arithmetic.
    {
        for (const iom::DataType type : kRmsNormApplicableDataTypes) {
            const std::string leaf(rmsnorm_oracle::leaf_name(type));
            const std::uint64_t canonical_nan = rmsnorm_oracle::value_bits(
                    type, std::numeric_limits<double>::quiet_NaN());
            const std::vector<double> probes = {
                    0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 6.0, 240.0, 65504.0, 1.0e30,
                    std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN(),
                    rmsnorm_oracle::leaf_min_positive(type),
                    rmsnorm_oracle::leaf_max_finite(type)};
            for (const double probe : probes) {
                const std::uint64_t bits = rmsnorm_oracle::value_bits(type, probe);
                const std::uint64_t reencoded = rmsnorm_oracle::value_bits(
                        type, element_value(type, bits));
                require(reencoded == bits || reencoded == canonical_nan,
                        leaf + ": codec round trip fails for a pinned probe");
            }

            // F == 4 with power-of-two features: the reduction, the mean, the
            // root, the reciprocal, and the scale product are all exact, so the
            // stored values must equal the exact results.
            const RmsNormReferenceCase exact = rmsnorm_oracle::make_case(
                    RmsNormReferenceCaseKind::mixed, type, 1, 1, 4, 0.0f,
                    std::vector<double>{2.0, 2.0, 2.0, 2.0},
                    std::vector<double>{1.0, -1.0, 2.0, -2.0});
            const std::vector<RmsNormReferenceValue> exact_values =
                    evaluate(exact);
            const std::array<double, 4> exact_results = {1.0, -1.0, 2.0, -2.0};
            for (std::size_t feature = 0; feature < exact_results.size();
                 ++feature) {
                require(exact_values[feature].bits
                                == rmsnorm_oracle::value_bits(
                                        type, exact_results[feature]),
                        leaf + ": exact step arithmetic must store exactly "
                                + std::to_string(exact_results[feature]));
            }
        }
    }

    return report;
}

}  // namespace iom_conformance
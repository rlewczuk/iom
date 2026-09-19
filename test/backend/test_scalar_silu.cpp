#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "iom/tensor.hpp"
#include "src/shared/scalar_silu.hpp"

namespace {

struct Format {
    iom::DataType type;
    unsigned bits;
    unsigned ebits;
    unsigned fbits;
    int bias;
    bool finite_only;
    bool infinities;
};

constexpr Format kFormats[] = {
        {iom::DataType::F4_E2M1, 4, 2, 1, 1, true, false},
        {iom::DataType::F6_E2M3, 6, 2, 3, 1, true, false},
        {iom::DataType::F6_E3M2, 6, 3, 2, 3, true, false},
        {iom::DataType::F8_E4M3FN, 8, 4, 3, 7, true, false},
        {iom::DataType::F8_E5M2, 8, 5, 2, 15, false, true},
        {iom::DataType::F16, 16, 5, 10, 15, false, true},
        {iom::DataType::BF16, 16, 8, 7, 127, false, true},
        {iom::DataType::F32, 32, 8, 23, 127, false, true},
        {iom::DataType::F64, 64, 11, 52, 1023, false, true},
};

std::uint64_t exponent_mask(const Format& format) noexcept {
    return (std::uint64_t{1} << format.ebits) - 1;
}

std::uint64_t fraction_mask(const Format& format) noexcept {
    return (std::uint64_t{1} << format.fbits) - 1;
}

std::uint64_t sign_mask(const Format& format) noexcept {
    return std::uint64_t{1} << (format.ebits + format.fbits);
}

bool is_special_encoding(std::uint64_t raw, const Format& format) noexcept {
    const auto exponent = (raw >> format.fbits) & exponent_mask(format);
    return exponent == exponent_mask(format)
            && (!format.finite_only || format.ebits >= 4);
}

long double decode(std::uint64_t raw, const Format& format) noexcept {
    const auto exponent = (raw >> format.fbits) & exponent_mask(format);
    const auto fraction = raw & fraction_mask(format);
    if (is_special_encoding(raw, format)) {
        if (format.infinities && fraction == 0)
            return raw & sign_mask(format)
                    ? -std::numeric_limits<long double>::infinity()
                    : std::numeric_limits<long double>::infinity();
        return std::numeric_limits<long double>::quiet_NaN();
    }
    const long double value = exponent
            ? std::ldexp(
                      static_cast<long double>((std::uint64_t{1} << format.fbits)
                                               + fraction),
                      static_cast<int>(exponent) - format.bias
                              - static_cast<int>(format.fbits))
            : std::ldexp(
                      static_cast<long double>(fraction),
                      1 - format.bias - static_cast<int>(format.fbits));
    return raw & sign_mask(format) ? -value : value;
}

std::uint64_t round_even(long double value) noexcept {
    const auto integral = std::floor(value);
    const auto remainder = value - integral;
    return static_cast<std::uint64_t>(integral)
            + (remainder > 0.5L
                    || (remainder == 0.5L
                            && (static_cast<std::uint64_t>(integral) & 1)));
}

std::uint64_t encode(long double value, const Format& format) noexcept {
    const auto sign = std::signbit(value) ? sign_mask(format) : 0;
    value = std::fabs(value);
    const auto exponent_max = exponent_mask(format);
    const auto finite_exponent_max =
            format.finite_only && format.ebits < 4
            ? exponent_max
            : exponent_max - 1;
    const auto pack = [=](std::uint64_t exponent, std::uint64_t fraction) {
        return sign | (exponent << format.fbits) | fraction;
    };
    const auto overflow = format.infinities
            ? pack(exponent_max, 0)
            : pack(finite_exponent_max, fraction_mask(format));
    if (std::isnan(value)) {
        return format.finite_only && format.ebits < 4
                ? pack(finite_exponent_max, fraction_mask(format))
                : pack(exponent_max, format.finite_only
                                              ? fraction_mask(format)
                                              : std::uint64_t{1}
                                                        << (format.fbits - 1));
    }
    if (std::isinf(value)) return overflow;
    if (value == 0) return sign;

    const int minimum_subnormal_exponent =
            1 - format.bias - static_cast<int>(format.fbits);
    const int maximum_exponent =
            static_cast<int>(finite_exponent_max) - format.bias;
    int exponent = 0;
    (void)std::frexp(value, &exponent);
    --exponent;
    if (exponent < minimum_subnormal_exponent
                             + static_cast<int>(format.fbits)) {
        const auto rounded = round_even(
                std::ldexp(value, -minimum_subnormal_exponent));
        return rounded == (std::uint64_t{1} << format.fbits)
                ? pack(1, 0)
                : pack(0, rounded);
    }
    if (exponent > maximum_exponent) return overflow;
    auto fraction = round_even(
            std::ldexp(value, static_cast<int>(format.fbits) - exponent)
            - static_cast<long double>(std::uint64_t{1} << format.fbits));
    if (fraction == (std::uint64_t{1} << format.fbits)) {
        ++exponent;
        fraction = 0;
    }
    if (exponent > maximum_exponent) return overflow;
    return pack(static_cast<std::uint64_t>(exponent + format.bias), fraction);
}

long double reference_silu(long double x) noexcept {
    if (std::isnan(x)) return std::numeric_limits<long double>::quiet_NaN();
    if (std::isinf(x))
        return std::signbit(x) ? -0.0L
                               : std::numeric_limits<long double>::infinity();
    if (x == 0) return x;
    return x / (1.0L + std::exp(-x));
}

bool is_nan(std::uint64_t raw, const Format& format) noexcept {
    return std::isnan(decode(raw, format));
}

bool is_infinity(std::uint64_t raw, const Format& format) noexcept {
    return std::isinf(decode(raw, format));
}

bool is_zero(std::uint64_t raw, const Format& format) noexcept {
    return decode(raw, format) == 0;
}

std::uint64_t ordered(std::uint64_t raw, const Format& format) noexcept {
    return raw & sign_mask(format) ? ~raw : raw | sign_mask(format);
}

unsigned allowed_ulp(const Format& format) noexcept {
    if (format.type == iom::DataType::F32) return 4;
    if (format.type == iom::DataType::F64) return 8;
    return 1;
}

void check_value(const Format& format, std::uint64_t input) {
    const auto actual = iom::detail::scalar_silu(format.type, input);
    const auto expected = encode(reference_silu(decode(input, format)), format);
    const auto expected_value = decode(expected, format);
    const auto actual_value = decode(actual, format);
    CAPTURE(static_cast<int>(format.type));
    CAPTURE(input);
    CAPTURE(actual);
    CAPTURE(expected);
    if (std::isnan(expected_value)) {
        CHECK(is_nan(actual, format));
        return;
    }
    if (std::isinf(expected_value)) {
        CHECK(is_infinity(actual, format));
        CHECK(std::signbit(actual_value) == std::signbit(expected_value));
        return;
    }
    if (expected_value == 0) {
        CHECK(is_zero(actual, format));
        CHECK(std::signbit(actual_value) == std::signbit(expected_value));
        return;
    }
    REQUIRE(std::isfinite(actual_value));
    CHECK(std::signbit(actual_value) == std::signbit(expected_value));
    const auto distance = ordered(actual, format) > ordered(expected, format)
            ? ordered(actual, format) - ordered(expected, format)
            : ordered(expected, format) - ordered(actual, format);
    CHECK(distance <= allowed_ulp(format));
}

std::vector<std::uint64_t> finite_inputs(const Format& format) {
    if (format.bits <= 8) {
        std::vector<std::uint64_t> values;
        const auto count = std::uint64_t{1} << format.bits;
        for (std::uint64_t raw = 0; raw < count; ++raw)
            if (!is_special_encoding(raw, format)) values.push_back(raw);
        return values;
    }
    if (format.type == iom::DataType::F16) {
        return {0x0001, 0x8001, 0x0400, 0x8400, 0x3555, 0xb555,
                0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000,
                0x7bff, 0xfbff};
    }
    if (format.type == iom::DataType::BF16) {
        return {0x0001, 0x8001, 0x0080, 0x8080, 0x3f00, 0xbf00,
                0x3f80, 0xbf80, 0x4000, 0xc000, 0x7f7f, 0xff7f};
    }
    if (format.type == iom::DataType::F32) {
        return {0x00000001u, 0x80000001u, 0x00800000u, 0x80800000u,
                0x3f000000u, 0xbf000000u, 0x3f800000u, 0xbf800000u,
                0x40000000u, 0xc0000000u, 0x42d00000u, 0xc2d00000u,
                0x7f7fffffu, 0xff7fffffu};
    }
    return {0x0000000000000001ull, 0x8000000000000001ull,
            0x0010000000000000ull, 0x8010000000000000ull,
            0x3fe0000000000000ull, 0xbfe0000000000000ull,
            0x3ff0000000000000ull, 0xbff0000000000000ull,
            0x4000000000000000ull, 0xc000000000000000ull,
            0x4080000000000000ull, 0xc080000000000000ull,
            0x7fefffffffffffffull, 0xffefffffffffffff};
}

}  // namespace

TEST_CASE("SiLU covers every finite named floating format") {
    for (const auto& format : kFormats)
        for (const auto input : finite_inputs(format)) check_value(format, input);
}

TEST_CASE("SiLU preserves signed zeros and handles representable special classes") {
    for (const auto& format : kFormats) {
        const auto sign = sign_mask(format);
        CHECK(iom::detail::scalar_silu(format.type, 0) == 0);
        CHECK(iom::detail::scalar_silu(format.type, sign) == sign);

        const auto exponent = exponent_mask(format);
        const auto positive_special = exponent << format.fbits;
        const auto negative_special = positive_special | sign;
        if (format.infinities) {
            const auto positive = iom::detail::scalar_silu(
                    format.type, positive_special);
            CHECK(is_infinity(positive, format));
            CHECK(!std::signbit(decode(positive, format)));
            const auto negative = iom::detail::scalar_silu(
                    format.type, negative_special);
            CHECK(is_zero(negative, format));
            CHECK(std::signbit(decode(negative, format)));
        } else {
            CHECK(!is_infinity(
                    iom::detail::scalar_silu(format.type, positive_special),
                    format));
        }

        const auto nan_input = positive_special | fraction_mask(format);
        if (format.finite_only && format.ebits < 4)
            check_value(format, nan_input);
        else
            CHECK(is_nan(iom::detail::scalar_silu(format.type, nan_input), format));
    }
}

TEST_CASE("SiLU destination encoding pins RNE ties and finite overflow") {
    for (const auto& format : kFormats) {
        const auto one = encode(1.0L, format);
        const auto one_upper = one + 1;
        const auto spacing = decode(one_upper, format) - decode(one, format);
        const auto midpoint = decode(one, format) + spacing / 2;
        CHECK(encode(midpoint - spacing / 4, format) == one);
        CHECK(encode(midpoint, format) == one);
        CHECK(encode(midpoint + spacing / 4, format) == one_upper);

        const auto negative_one = encode(-1.0L, format);
        const auto negative_more = negative_one + 1;
        const auto negative_spacing =
                decode(negative_more, format) - decode(negative_one, format);
        const auto negative_midpoint =
                decode(negative_one, format) + negative_spacing / 2;
        CHECK(encode(negative_midpoint, format) == negative_one);
        CHECK(encode(negative_midpoint - negative_spacing / 4, format)
              == negative_one);
        CHECK(encode(negative_midpoint + negative_spacing / 4, format)
              == negative_more);

        const auto finite_exponent_max =
                exponent_mask(format) - (format.finite_only && format.ebits < 4
                                                  ? 0
                                                  : 1);
        const auto maximum = (finite_exponent_max << format.fbits)
                | fraction_mask(format);
        const auto maximum_value = decode(maximum, format);
        const auto maximum_exponent =
                static_cast<int>(finite_exponent_max) - format.bias;
        const auto half_ulp = std::ldexp(
                1.0L,
                maximum_exponent - static_cast<int>(format.fbits) - 1);
        const auto overflow = format.infinities
                ? exponent_mask(format) << format.fbits
                : maximum;
        CHECK(encode(maximum_value + half_ulp, format) == overflow);
        CHECK(iom::detail::scalar_silu(format.type, maximum) == maximum);
    }
}

TEST_CASE("SiLU retains the negative half-exponential tails") {
    const Format f32 = kFormats[7];
    const auto f32_input = std::bit_cast<std::uint32_t>(-104.0F);
    const auto f32_output = iom::detail::scalar_silu(f32.type, f32_input);
    CHECK(decode(f32_output, f32) < 0);
    CHECK(!is_zero(f32_output, f32));
    check_value(f32, f32_input);

    const Format f64 = kFormats[8];
    const auto f64_input = std::bit_cast<std::uint64_t>(-746.0);
    const auto f64_output = iom::detail::scalar_silu(f64.type, f64_input);
    CHECK(decode(f64_output, f64) < 0);
    CHECK(!is_zero(f64_output, f64));
    check_value(f64, f64_input);
}

TEST_CASE("SiLU rejects non-floating named formats without coercion") {
    constexpr iom::DataType unsupported[] = {
            iom::DataType::BOOL, iom::DataType::I2, iom::DataType::U2,
            iom::DataType::I4, iom::DataType::U4, iom::DataType::I8,
            iom::DataType::U8, iom::DataType::I16, iom::DataType::U16,
            iom::DataType::I32, iom::DataType::U32, iom::DataType::I64,
            iom::DataType::U64, iom::DataType::F8_E8M0};
    for (const auto type : unsupported)
        CHECK(iom::detail::scalar_silu(type, 0x123456789abcdef0ull) == 0);
}

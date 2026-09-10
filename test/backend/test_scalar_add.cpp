#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <cstdint>
#include <cmath>
#include "backend_conformance_add.hpp"
#include "src/shared/scalar_add.hpp"

using Op = iom_conformance::add_oracle::operation;

TEST_CASE("compact integer multiply and subtract wrap modulo width") {
    const iom::DataType types[] = {iom::DataType::I2, iom::DataType::U2, iom::DataType::I4, iom::DataType::U4};
    for (const auto type : types) {
        const unsigned n = type == iom::DataType::I2 || type == iom::DataType::U2 ? 4 : 16;
        for (unsigned a = 0; a < n; ++a)
            for (unsigned b = 0; b < n; ++b) {
                CHECK(iom::detail::scalar_mul(type, a, b) ==
                      iom_conformance::add_oracle::binary(type, a, b, Op::mul));
                CHECK(iom::detail::scalar_sub(type, a, b) ==
                      iom_conformance::add_oracle::binary(type, a, b, Op::sub));
            }
    }
}

TEST_CASE("compact floating operations preserve ordered raw encodings") {
    const iom::DataType types[] = {iom::DataType::F4_E2M1, iom::DataType::F6_E2M3,
        iom::DataType::F6_E3M2, iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2};
    const unsigned widths[] = {4, 6, 6, 8, 8};
    for (unsigned k = 0; k < 5; ++k)
        for (unsigned a = 0; a < (1u << widths[k]); ++a)
            for (unsigned b = 0; b < (1u << widths[k]); ++b) {
                CHECK(iom::detail::scalar_mul(types[k], a, b) ==
                      iom_conformance::add_oracle::binary(types[k], a, b, Op::mul));
                CHECK(iom::detail::scalar_sub(types[k], a, b) ==
                      iom_conformance::add_oracle::binary(types[k], a, b, Op::sub));
                CHECK(iom::detail::scalar_div(types[k], a, b) ==
                      iom_conformance::add_oracle::binary(types[k], a, b, Op::div));
            }
}

TEST_CASE("wide integers multiply and subtract retain low bits") {
    CHECK(iom::detail::scalar_mul(iom::DataType::U64, ~0ull, 3) == ~2ull);
    CHECK(iom::detail::scalar_mul(iom::DataType::I32, 0x80000000u, 3) == 0x80000000u);
    CHECK(iom::detail::scalar_sub(iom::DataType::U16, 0, 1) == 0xffffu);
    CHECK(iom::detail::scalar_sub(iom::DataType::I64, 0, 1) == ~0ull);
}

TEST_CASE("wide floating operations preserve special values and signed zero") {
    CHECK(iom::detail::scalar_mul(iom::DataType::F32, 0x7f800000, 0) == 0x7fc00000);
    CHECK(iom::detail::scalar_sub(iom::DataType::F32, 0x7f800000, 0x7f800000) == 0x7fc00000);
    CHECK(iom::detail::scalar_div(iom::DataType::F32, 0, 0) == 0x7fc00000);
    CHECK(iom::detail::scalar_div(iom::DataType::F32, 0x80000000, 1) == 0x80000000);
    CHECK(iom::detail::scalar_div(iom::DataType::F32, 0x80000000, 0x80000000) == 0x7fc00000);
    CHECK(iom::detail::scalar_div(iom::DataType::F32, 0x3f800000, 0x7f800000) == 0);
}

TEST_CASE("add retains established scalar encoding") {
    CHECK(iom::detail::scalar_add(iom::DataType::F32, 0x3f800000, 0x3f800000) ==
          iom_conformance::add_oracle::add(iom::DataType::F32, 0x3f800000, 0x3f800000));
    CHECK(iom::detail::scalar_add(iom::DataType::F16, 0x7c00, 0xfc00) == 0x7e00);
    CHECK(iom::detail::scalar_add(iom::DataType::F32, 0x80000000, 0) == 0);
}

TEST_CASE("floating overflow uses infinity only for infinity-capable formats") {
    struct OverflowCase {
        iom::DataType type;
        std::uint64_t max;
        std::uint64_t two;
        std::uint64_t half;
        std::uint64_t infinity;
    };
    constexpr OverflowCase cases[] = {
        {iom::DataType::F8_E5M2, 0x7b, 0x40, 0x38, 0x7c},
        {iom::DataType::F16, 0x7bff, 0x4000, 0x3800, 0x7c00},
        {iom::DataType::BF16, 0x7f7f, 0x4000, 0x3f00, 0x7f80},
        {iom::DataType::F32, 0x7f7fffff, 0x40000000, 0x3f000000, 0x7f800000},
        {iom::DataType::F64, 0x7fefffffffffffff, 0x4000000000000000,
         0x3fe0000000000000, 0x7ff0000000000000},
    };
    for (const auto& c : cases) {
        const auto format = iom::detail::scalar_add_detail::format(c.type);
        const std::uint64_t sign = std::uint64_t{1}
                << (format.ebits + format.fbits);
        const std::uint64_t negative_max = c.max | sign;
        const std::uint64_t negative_infinity = c.infinity | sign;
        const auto check = [=](auto production, Op op, std::uint64_t lhs,
                               std::uint64_t rhs, std::uint64_t expected) {
            CHECK(production(c.type, lhs, rhs) == expected);
            CHECK(iom_conformance::add_oracle::binary(
                          c.type, lhs, rhs, op) == expected);
        };
        check(iom::detail::scalar_add, Op::add, c.max, c.max, c.infinity);
        check(iom::detail::scalar_mul, Op::mul, c.max, c.two, c.infinity);
        check(iom::detail::scalar_sub, Op::sub, c.max, negative_max, c.infinity);
        check(iom::detail::scalar_div, Op::div, c.max, c.half, c.infinity);
        check(iom::detail::scalar_add, Op::add, negative_max, negative_max,
              negative_infinity);
        check(iom::detail::scalar_mul, Op::mul, negative_max, c.two,
              negative_infinity);
        check(iom::detail::scalar_sub, Op::sub, negative_max, c.max,
              negative_infinity);
        check(iom::detail::scalar_div, Op::div, negative_max, c.half,
              negative_infinity);

        const int max_exp =
                static_cast<int>((std::uint64_t{1} << format.ebits) - 2)
                - format.bias;
        const long double max_value =
                iom::detail::scalar_add_detail::decode_small(c.max, format);
        const long double half_ulp =
                std::ldexp(1.0L, max_exp - static_cast<int>(format.fbits) - 1);
        CHECK(iom::detail::scalar_add_detail::encode_small(
                      max_value, format) == c.max);
        CHECK(iom::detail::scalar_add_detail::encode_small(
                      max_value + half_ulp / 2, format) == c.max);
        CHECK(iom::detail::scalar_add_detail::encode_small(
                      max_value + half_ulp, format) == c.infinity);
        CHECK(iom::detail::scalar_add_detail::encode_small(
                      -(max_value + half_ulp), format) == negative_infinity);
        const auto oracle_format =
                iom_conformance::add_oracle::spec(c.type);
        const long double oracle_max_value =
                iom_conformance::add_oracle::value(c.max, oracle_format);
        CHECK(iom_conformance::add_oracle::round_encode(
                      oracle_max_value + half_ulp / 2, oracle_format)
              == c.max);
        CHECK(iom_conformance::add_oracle::round_encode(
                      oracle_max_value + half_ulp, oracle_format)
              == c.infinity);
        CHECK(iom_conformance::add_oracle::round_encode(
                      -(oracle_max_value + half_ulp), oracle_format)
              == negative_infinity);
    }

    struct FiniteOnlyCase {
        iom::DataType type;
        std::uint64_t max;
        std::uint64_t two;
    };
    constexpr FiniteOnlyCase finite_only[] = {
        {iom::DataType::F4_E2M1, 0x7, 0x4},
        {iom::DataType::F6_E2M3, 0x1f, 0x10},
        {iom::DataType::F6_E3M2, 0x1f, 0x10},
        {iom::DataType::F8_E4M3FN, 0x77, 0x40},
    };
    for (const auto& c : finite_only) {
        const auto format = iom::detail::scalar_add_detail::format(c.type);
        const std::uint64_t sign = std::uint64_t{1}
                << (format.ebits + format.fbits);
        CHECK(iom::detail::scalar_mul(c.type, c.max, c.two) == c.max);
        CHECK(iom::detail::scalar_mul(c.type, c.max | sign, c.two)
              == (c.max | sign));
    }
}

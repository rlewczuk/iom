#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <cstdint>
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

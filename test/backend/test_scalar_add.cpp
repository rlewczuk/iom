#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <cstdint>
#include "backend_conformance_add.hpp"
#include "src/shared/scalar_add.hpp"

TEST_CASE("small integer ADD wraps modulo width") {
    const iom::DataType types[] = {iom::DataType::I2,iom::DataType::U2,iom::DataType::I4,iom::DataType::U4};
    for (auto t : types) { unsigned n = t==iom::DataType::I2||t==iom::DataType::U2?4:16; for(unsigned a=0;a<n;++a) for(unsigned b=0;b<n;++b) CHECK(iom::detail::scalar_add(t,a,b)==iom_conformance::add_oracle::add(t,a,b)); }
}
TEST_CASE("all compact floating encodings match independent oracle") {
    const iom::DataType types[] = {iom::DataType::F4_E2M1,iom::DataType::F6_E2M3,iom::DataType::F6_E3M2,iom::DataType::F8_E4M3FN,iom::DataType::F8_E5M2};
    const unsigned widths[] = {4,6,6,8,8};
    for(unsigned k=0;k<5;++k) for(unsigned a=0;a<(1u<<widths[k]);++a) for(unsigned b=0;b<(1u<<widths[k]);++b) CHECK(iom::detail::scalar_add(types[k],a,b)==iom_conformance::add_oracle::add(types[k],a,b));
}
TEST_CASE("wide floating formats preserve canonical specials and boundaries") {
    CHECK(iom::detail::scalar_add(iom::DataType::F16,0x7c00,0)==0x7c00);
    CHECK(iom::detail::scalar_add(iom::DataType::BF16,0x7f80,0)==0x7f80);
    CHECK(iom::detail::scalar_add(iom::DataType::F32,0x7f800000,0)==0x7f800000);
    CHECK(iom::detail::scalar_add(iom::DataType::F64,0x7ff0000000000000ull,0)==0x7ff0000000000000ull);
    CHECK(iom::detail::scalar_add(iom::DataType::F16,0x7c00,0xfc00)==0x7e00);
    CHECK(iom::detail::scalar_add(iom::DataType::BF16,0x7f80,0xff80)==0x7fc0);
    CHECK(iom::detail::scalar_add(iom::DataType::F32,0x7f800000,0xff800000)==0x7fc00000);
    CHECK(iom::detail::scalar_add(iom::DataType::F64,0x7ff0000000000000ull,0xfff0000000000000ull)==0x7ff8000000000000ull);
    CHECK(iom::detail::scalar_add(iom::DataType::F16,0x8000,0x8000)==0x8000);
    CHECK(iom::detail::scalar_add(iom::DataType::F32,0x80000000,0)==0);
}

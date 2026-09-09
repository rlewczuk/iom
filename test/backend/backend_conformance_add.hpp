#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "backend/backend_conformance_common.hpp"
#include "iom/tensor.hpp"

namespace iom_conformance {
namespace add_oracle {
struct Spec { unsigned e, f; int bias; bool inf; bool finite; };
inline Spec spec(iom::DataType t) noexcept {
    switch (t) {
        case iom::DataType::F4_E2M1: return {2, 1, 1, 0, 1};
        case iom::DataType::F6_E2M3: return {2, 3, 1, 0, 1};
        case iom::DataType::F6_E3M2: return {3, 2, 3, 0, 1};
        case iom::DataType::F8_E4M3FN: return {4, 3, 7, 0, 1};
        case iom::DataType::F8_E5M2: return {5, 2, 15, 1, 0};
        case iom::DataType::F16: return {5, 10, 15, 1, 0};
        case iom::DataType::BF16: return {8, 7, 127, 1, 0};
        case iom::DataType::F32: return {8, 23, 127, 1, 0};
        default: return {11, 52, 1023, 1, 0};
    }
}
inline long double value(std::uint64_t r, Spec s) noexcept {
    const auto em = (std::uint64_t{1} << s.e) - 1, fm = (std::uint64_t{1} << s.f) - 1;
    const auto ex = (r >> s.f) & em, fr = r & fm;
    if (ex == em && (!s.finite || s.e == 4)) {
        if (s.inf && fr == 0) return (r >> (s.e + s.f)) ? -INFINITY : INFINITY;
        return std::numeric_limits<long double>::quiet_NaN();
    }
    const long double v = ex
            ? std::ldexp(static_cast<long double>((std::uint64_t{1} << s.f) + fr),
                         static_cast<int>(ex) - s.bias - static_cast<int>(s.f))
            : std::ldexp(static_cast<long double>(fr), 1 - s.bias - static_cast<int>(s.f));
    return (r >> (s.e + s.f)) ? -v : v;
}
inline std::uint64_t round_encode(long double x, Spec s) noexcept {
    const auto em = (std::uint64_t{1} << s.e) - 1, fm = (std::uint64_t{1} << s.f) - 1;
    const auto sg = std::signbit(x) ? std::uint64_t{1} : 0; x = std::fabs(x);
    if (std::isnan(x))
        return (sg << (s.e + s.f)) | (em << s.f) |
               (s.finite ? fm : std::uint64_t{1} << (s.f - 1));
    if (std::isinf(x)) return (sg << (s.e + s.f)) | (s.inf ? em << s.f : ((em - 1) << s.f) | fm);
    if (x == 0) return sg << (s.e + s.f);
    int e = 0; std::frexp(x, &e); --e;
    const int minsub = 1 - s.bias - static_cast<int>(s.f);
    const int maxe = static_cast<int>((s.finite && s.e < 4 ? em : em - 1) - s.bias);
    auto rn = [](long double q) {
        const auto n = std::floor(q), r = q - n;
        return static_cast<std::uint64_t>(n + (r > .5L || (r == .5L && (static_cast<std::uint64_t>(n) & 1))));
    };
    if (e < minsub + static_cast<int>(s.f)) {
        const auto q = rn(std::ldexp(x, -minsub));
        return (sg << (s.e + s.f)) | (q == (std::uint64_t{1} << s.f) ? std::uint64_t{1} << s.f : q);
    }
    if (e > maxe) return (sg << (s.e + s.f)) | ((s.finite && s.e < 4 ? em : em - 1) << s.f) | fm;
    auto q = rn(std::ldexp(x, static_cast<int>(s.f) - e) - (std::uint64_t{1} << s.f));
    if (q == (std::uint64_t{1} << s.f)) { ++e; q = 0; }
    if (e > maxe) return (sg << (s.e + s.f)) | ((s.finite && s.e < 4 ? em : em - 1) << s.f) | fm;
    return (sg << (s.e + s.f)) | (static_cast<std::uint64_t>(e + s.bias) << s.f) | q;
}
inline std::uint64_t add(iom::DataType t, std::uint64_t a, std::uint64_t b) noexcept {
    unsigned w = 0;
    switch (t) {
        case iom::DataType::I2: case iom::DataType::U2: w = 2; break;
        case iom::DataType::I4: case iom::DataType::U4: w = 4; break;
        case iom::DataType::I8: case iom::DataType::U8: w = 8; break;
        case iom::DataType::I16: case iom::DataType::U16: w = 16; break;
        case iom::DataType::I32: case iom::DataType::U32: w = 32; break;
        case iom::DataType::I64: case iom::DataType::U64: w = 64; break;
        default: {
            const auto s = spec(t);
            const auto x = value(a, s), y = value(b, s);
            if (std::isnan(x) || std::isnan(y) ||
                (std::isinf(x) && std::isinf(y) && std::signbit(x) != std::signbit(y)))
                return round_encode(std::numeric_limits<long double>::quiet_NaN(), s);
            const long double sum = x + y;
            // The device's F64 path overflows to infinity; narrower formats
            // retain the established saturation contract.
            if (s.e == 11 && s.inf && std::isfinite(sum)) {
                const auto em = (std::uint64_t{1} << s.e) - 1;
                const auto fm = (std::uint64_t{1} << s.f) - 1;
                const auto max_raw = ((em - 1) << s.f) | fm;
                const long double max_value = value(max_raw, s);
                if (std::fabs(sum) > max_value)
                    return round_encode(
                            std::copysign(std::numeric_limits<long double>::infinity(), sum), s);
            }
            return round_encode(sum, s);
        }
    }
    return (a + b) & (w == 64 ? ~std::uint64_t{} : (std::uint64_t{1} << w) - 1);
}
}  // namespace add_oracle

inline std::uint64_t add_read_bits(const std::vector<std::byte>& bytes, std::size_t index, std::size_t bits) {
    std::uint64_t value = 0;
    for (std::size_t bit = 0; bit < bits; ++bit)
        if ((static_cast<unsigned char>(bytes[(index * bits + bit) / 8]) >> ((index * bits + bit) % 8)) & 1)
            value |= std::uint64_t{1} << bit;
    return value;
}
inline bool add_f32_within_ulp(std::uint32_t actual, std::uint32_t expected) {
    const float a = std::bit_cast<float>(actual), e = std::bit_cast<float>(expected);
    if (std::isnan(e)) return std::isnan(a);
    if (std::isinf(e) || e == 0.0f) return actual == expected;
    if (!std::isfinite(a)) return false;
    const auto ordered = [](std::uint32_t bits) { return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u; };
    const auto oa = ordered(actual), oe = ordered(expected);
    return (oa > oe ? oa - oe : oe - oa) <= 1;
}
inline bool add_is_float(iom::DataType type) {
    switch (type) {
        case iom::DataType::F4_E2M1: case iom::DataType::F6_E2M3:
        case iom::DataType::F6_E3M2: case iom::DataType::F8_E4M3FN:
        case iom::DataType::F8_E5M2: case iom::DataType::F16:
        case iom::DataType::BF16: case iom::DataType::F32:
        case iom::DataType::F64: return true;
        default: return false;
    }
}
inline bool add_non_f32_float_classes(iom::DataType type, std::uint64_t actual, std::uint64_t expected) {
    const auto a = add_oracle::value(actual, add_oracle::spec(type));
    const auto e = add_oracle::value(expected, add_oracle::spec(type));
    if (std::isnan(e)) return std::isnan(a);
    if (std::isinf(e)) return std::isinf(a) && std::signbit(a) == std::signbit(e);
    if (e == 0) return a == 0 && std::signbit(a) == std::signbit(e);
    return std::isfinite(a) && actual == expected;
}
inline void run_add_value_conformance(iom::Device& candidate) {
    const iom::DataType types[] = {
        iom::DataType::I2, iom::DataType::U2, iom::DataType::I4, iom::DataType::U4,
        iom::DataType::I8, iom::DataType::U8, iom::DataType::I16, iom::DataType::U16,
        iom::DataType::I32, iom::DataType::U32, iom::DataType::I64, iom::DataType::U64,
        iom::DataType::F4_E2M1, iom::DataType::F6_E2M3, iom::DataType::F6_E3M2,
        iom::DataType::F8_E4M3FN, iom::DataType::F8_E5M2, iom::DataType::F16,
        iom::DataType::BF16, iom::DataType::F32, iom::DataType::F64};
    auto queue = candidate.create_ops();
    for (const auto type : types) {
        const iom::TensorSpec spec{iom::TensorShape{{type == iom::DataType::F32 ? 1u : 2u,
            type == iom::DataType::F32 ? 33u : 17u, type == iom::DataType::F32 ? 17u : 33u}}, type};
        const auto count = spec.shape.element_count(), bits = iom::detail::leaf_bits(type);
        std::vector<std::byte> lhs(spec.logical_nbytes()), rhs(spec.logical_nbytes()), expected(spec.logical_nbytes());
        for (std::size_t i = 0; i < count; ++i) {
            auto a = element_pattern(type, i, 0x1234), b = element_pattern(type, i, 0x9876);
            if (type == iom::DataType::F32) {
                static constexpr std::uint32_t values[] = {0x00000000, 0x80000000, 0x00000001, 0x3f800000,
                    0x3f800001, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00001};
                a = values[i % 9]; b = values[(i * 5 + 2) % 9];
            }
            if (type != iom::DataType::F32 && type != iom::DataType::U8) {
                a = 1;
                b = 2;
            }
            write_bits(reinterpret_cast<unsigned char*>(lhs.data()), i * bits, bits, a);
            write_bits(reinterpret_cast<unsigned char*>(rhs.data()), i * bits, bits, b);
            write_bits(reinterpret_cast<unsigned char*>(expected.data()), i * bits, bits, add_oracle::add(type, a, b));
        }
        auto lhs_tensor = candidate.create_tensor(spec), rhs_tensor = candidate.create_tensor(spec), out_tensor = candidate.create_tensor(spec);
        lhs_tensor->view().copy_from_host(lhs); rhs_tensor->view().copy_from_host(rhs); out_tensor->view().copy_from_host(expected);
        const auto token = queue->add(lhs_tensor->view(), rhs_tensor->view(), out_tensor->view());
        REQUIRE(iom::oid_is_token(token)); CHECK_NOTHROW(queue->wait(token));
        const auto observed = read_logical(out_tensor->view());
        for (std::size_t i = 0; i < count; ++i) {
            const auto actual = add_read_bits(observed, i, bits), want = add_read_bits(expected, i, bits);
            if (type == iom::DataType::F32)
                CHECK(add_f32_within_ulp(static_cast<std::uint32_t>(actual), static_cast<std::uint32_t>(want)));
            else if (add_is_float(type))
                CHECK(add_non_f32_float_classes(type, actual, want));
            else
                CHECK_EQ(actual, want);
        }
    }
    const iom::TensorSpec lhs_spec{iom::TensorShape{{2, 1, 33}}, iom::DataType::U8};
    const iom::TensorSpec rhs_spec{iom::TensorShape{{1, 17, 1}}, iom::DataType::U8};
    const iom::TensorSpec out_spec{iom::TensorShape{{2, 17, 33}}, iom::DataType::U8};
    auto lhs = candidate.create_tensor(lhs_spec), rhs = candidate.create_tensor(rhs_spec), out = candidate.create_tensor(out_spec);
    auto lhs_bytes = encode_logical(lhs_spec, 0x1111), rhs_bytes = encode_logical(rhs_spec, 0x2222);
    std::vector<std::byte> expected(out_spec.logical_nbytes());
    for (std::size_t plane = 0; plane < 2; ++plane)
        for (std::size_t row = 0; row < 17; ++row)
            for (std::size_t column = 0; column < 33; ++column) {
                const auto a = add_read_bits(lhs_bytes, plane * 33 + column, 8);
                const auto b = add_read_bits(rhs_bytes, row, 8);
                write_bits(reinterpret_cast<unsigned char*>(expected.data()),
                            (plane * 17 * 33 + row * 33 + column) * 8, 8,
                            add_oracle::add(iom::DataType::U8, a, b));
            }
    lhs->view().copy_from_host(lhs_bytes); rhs->view().copy_from_host(rhs_bytes);
    out->view().copy_from_host(expected);
    const auto broadcast_token = queue->add(lhs->view(), rhs->view(), out->view());
    REQUIRE(iom::oid_is_token(broadcast_token)); CHECK_NOTHROW(queue->wait(broadcast_token));
    const auto observed = read_logical(out->view());
    for (std::size_t i = 0; i < expected.size(); ++i) CHECK_EQ(observed[i], expected[i]);
}
}  // namespace iom_conformance

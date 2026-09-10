#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include "iom/tensor.hpp"

namespace iom::detail {
namespace scalar_add_detail {

struct Format { unsigned bits, ebits, fbits; int bias; bool finite_only, infs; };

inline constexpr Format format(DataType t) noexcept {
    switch (t) {
        case DataType::F4_E2M1: return {4, 2, 1, 1, true, false};
        case DataType::F6_E2M3: return {6, 2, 3, 1, true, false};
        case DataType::F6_E3M2: return {6, 3, 2, 3, true, false};
        case DataType::F8_E4M3FN: return {8, 4, 3, 7, true, false};
        case DataType::F8_E5M2: return {8, 5, 2, 15, false, true};
        case DataType::F16: return {16, 5, 10, 15, false, true};
        case DataType::BF16: return {16, 8, 7, 127, false, true};
        case DataType::F32: return {32, 8, 23, 127, false, true};
        case DataType::F64: return {64, 11, 52, 1023, false, true};
        default: return {};
    }
}

inline long double decode_small(std::uint64_t raw, Format f) noexcept {
    const auto sign = raw >> (f.ebits + f.fbits);
    const auto emask = (std::uint64_t{1} << f.ebits) - 1;
    const auto frac = raw & ((std::uint64_t{1} << f.fbits) - 1);
    const auto exp = (raw >> f.fbits) & emask;
    if (exp == emask && (!f.finite_only || f.ebits >= 4)) {
        if (f.infs && frac == 0) return sign ? -INFINITY : INFINITY;
        return std::numeric_limits<long double>::quiet_NaN();
    }
    const long double v = exp
        ? std::ldexp(static_cast<long double>((std::uint64_t{1} << f.fbits) + frac),
                     static_cast<int>(exp) - f.bias - static_cast<int>(f.fbits))
        : std::ldexp(static_cast<long double>(frac), 1 - f.bias - static_cast<int>(f.fbits));
    return sign ? -v : v;
}

inline std::uint64_t encode_small(long double x, Format f) noexcept {
    const auto sign = std::signbit(x) ? std::uint64_t{1} : 0;
    x = std::fabs(x);
    const auto emask = (std::uint64_t{1} << f.ebits) - 1;
    const auto fmask = (std::uint64_t{1} << f.fbits) - 1;
    const auto finite_emax = f.finite_only && f.ebits < 4 ? emask : emask - 1;
    const auto pack = [&](std::uint64_t e, std::uint64_t m) {
        return (sign << (f.ebits + f.fbits)) | (e << f.fbits) | m;
    };
    if (std::isnan(x)) {
        if (f.finite_only && f.ebits < 4) x = std::numeric_limits<long double>::max();
        else return pack(emask, f.finite_only ? fmask : std::uint64_t{1} << (f.fbits - 1));
    }
    if (std::isinf(x)) return f.infs ? pack(emask, 0) : pack(finite_emax, fmask);
    if (x == 0) return sign << (f.ebits + f.fbits);
    const int min_sub = 1 - f.bias - static_cast<int>(f.fbits);
    const int max_exp = static_cast<int>(finite_emax) - f.bias;
    int e = 0;
    (void)std::frexp(x, &e);
    --e;
    const auto round_even = [](long double y) noexcept {
        const auto q = std::floor(y);
        const auto r = y - q;
        return static_cast<std::uint64_t>(q) +
               (r > 0.5L || (r == 0.5L && (static_cast<std::uint64_t>(q) & 1)));
    };
    if (e < min_sub + static_cast<int>(f.fbits)) {
        const auto q = round_even(std::ldexp(x, -min_sub));
        return q == (std::uint64_t{1} << f.fbits) ? pack(1, 0) : pack(0, q);
    }
    if (e > max_exp) return pack(finite_emax, fmask);
    auto frac = round_even(std::ldexp(x, f.fbits - e) - static_cast<long double>(std::uint64_t{1} << f.fbits));
    if (frac == (std::uint64_t{1} << f.fbits)) { ++e; frac = 0; }
    if (e > max_exp) return pack(finite_emax, fmask);
    return pack(static_cast<std::uint64_t>(e + f.bias), frac);
}

enum class BinaryOp { add, mul, sub, div };

template <BinaryOp Op>
inline long double arithmetic(long double x, long double y) noexcept {
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<long double>::quiet_NaN();
    if constexpr (Op == BinaryOp::add) {
        if (std::isinf(x) && std::isinf(y) && std::signbit(x) != std::signbit(y))
            return std::numeric_limits<long double>::quiet_NaN();
        if (x == 0 && y == 0)
            return (std::signbit(x) && std::signbit(y)) ? -0.0L : 0.0L;
        const auto z = x + y;
        return z == 0 ? 0.0L : z;
    } else if constexpr (Op == BinaryOp::mul) {
        if ((std::isinf(x) && y == 0) || (std::isinf(y) && x == 0))
            return std::numeric_limits<long double>::quiet_NaN();
        return x * y;
    } else if constexpr (Op == BinaryOp::sub) {
        if (std::isinf(x) && std::isinf(y) && std::signbit(x) == std::signbit(y))
            return std::numeric_limits<long double>::quiet_NaN();
        return x - y;
    } else {
        if ((x == 0 && y == 0) || (std::isinf(x) && std::isinf(y)))
            return std::numeric_limits<long double>::quiet_NaN();
        return x / y;
    }
}

template <BinaryOp Op>
[[nodiscard]] inline std::uint64_t scalar_binary(DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    unsigned w = 0;
    switch (type) {
        case DataType::I2: case DataType::U2: w = 2; break;
        case DataType::I4: case DataType::U4: w = 4; break;
        case DataType::I8: case DataType::U8: w = 8; break;
        case DataType::I16: case DataType::U16: w = 16; break;
        case DataType::I32: case DataType::U32: w = 32; break;
        case DataType::I64: case DataType::U64: w = 64; break;
        default: break;
    }
    if (w) {
        if constexpr (Op == BinaryOp::div) return 0;
        const auto mask = w == 64 ? ~std::uint64_t{} : (std::uint64_t{1} << w) - 1;
        if constexpr (Op == BinaryOp::add) return (a + b) & mask;
        if constexpr (Op == BinaryOp::mul) return (a * b) & mask;
        return (a - b) & mask;
    }
    const auto f = format(type);
    if (!f.bits) return 0;
    return encode_small(arithmetic<Op>(decode_small(a, f), decode_small(b, f)), f);
}

}  // namespace scalar_add_detail

template <scalar_add_detail::BinaryOp Op>
[[nodiscard]] inline std::uint64_t scalar_binary(DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_add_detail::scalar_binary<Op>(type, a, b);
}
inline std::uint64_t scalar_add(DataType t, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::add>(t, a, b);
}
inline std::uint64_t scalar_mul(DataType t, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::mul>(t, a, b);
}
inline std::uint64_t scalar_sub(DataType t, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::sub>(t, a, b);
}
inline std::uint64_t scalar_div(DataType t, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::div>(t, a, b);
}
}  // namespace iom::detail

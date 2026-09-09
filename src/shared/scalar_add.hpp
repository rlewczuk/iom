#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include "iom/tensor.hpp"

namespace iom::detail {

namespace scalar_add_detail {
struct Format { unsigned bits, ebits, fbits; int bias; bool finite_only; bool infs; };

inline constexpr Format format(DataType t) noexcept {
    switch (t) {
        case DataType::F4_E2M1: return {4, 2, 1, 1, true, false};
        case DataType::F6_E2M3: return {6, 2, 3, 1, true, false};
        case DataType::F6_E3M2: return {6, 3, 2, 3, true, false};
        case DataType::F8_E4M3FN: return {8, 4, 3, 7, true, false};
        case DataType::F8_E5M2: return {8, 5, 2, 15, false, true};
        default: return {0, 0, 0, 0, false, false};
    }
}
inline long double decode_small(std::uint64_t raw, Format f) noexcept {
    const std::uint64_t sign = raw >> (f.ebits + f.fbits);
    const std::uint64_t emask = (std::uint64_t{1} << f.ebits) - 1;
    const std::uint64_t frac = raw & ((std::uint64_t{1} << f.fbits) - 1);
    const std::uint64_t exp = (raw >> f.fbits) & emask;
    const std::uint64_t emax = emask;
    if (exp == emax && (!f.finite_only || f.ebits >= 4)) {
        if (f.infs && frac == 0) return sign ? -INFINITY : INFINITY;
        return std::numeric_limits<long double>::quiet_NaN();
    }
    long double v;
    if (exp == 0) v = std::ldexp(static_cast<long double>(frac), 1 - f.bias - static_cast<int>(f.fbits));
    else v = std::ldexp(static_cast<long double>((std::uint64_t{1} << f.fbits) + frac), static_cast<int>(exp) - f.bias - static_cast<int>(f.fbits));
    return sign ? -v : v;
}
inline std::uint64_t encode_small(long double x, Format f) noexcept {
    const std::uint64_t sign = std::signbit(x) ? 1 : 0;
    x = std::fabs(x);
    const std::uint64_t emask = (std::uint64_t{1} << f.ebits) - 1;
    const std::uint64_t fmask = (std::uint64_t{1} << f.fbits) - 1;
    if (std::isnan(x)) {
        if (f.finite_only && f.ebits < 4) x = std::numeric_limits<long double>::max();
        else return (sign << (f.ebits + f.fbits)) | (f.finite_only ? (emask << f.fbits) | fmask : (emask << f.fbits) | (std::uint64_t{1} << (f.fbits - 1)));
    }
    if (std::isinf(x)) {
        if (f.infs) return (sign << (f.ebits + f.fbits)) | (emask << f.fbits);
        return (sign << (f.ebits + f.fbits)) | ((f.finite_only && f.ebits < 4 ? emask : emask - 1) << f.fbits) | fmask;
    }
    if (x == 0) return sign << (f.ebits + f.fbits);
    const int min_sub = 1 - f.bias - static_cast<int>(f.fbits);
    const int max_exp = static_cast<int>(f.finite_only && f.ebits < 4 ? emask : emask - 1) - f.bias;
    int e = 0; (void)std::frexp(x, &e); --e;
    auto round_even = [](long double y) noexcept -> std::uint64_t {
        const long double q = std::floor(y); const long double r = y - q;
        if (r > 0.5L || (r == 0.5L && (static_cast<std::uint64_t>(q) & 1))) return static_cast<std::uint64_t>(q) + 1;
        return static_cast<std::uint64_t>(q);
    };
    if (e < min_sub + static_cast<int>(f.fbits)) {
        const auto q = round_even(std::ldexp(x, -min_sub));
        if (q == (std::uint64_t{1} << f.fbits)) return sign << (f.ebits + f.fbits) | (std::uint64_t{1} << f.fbits);
        return (sign << (f.ebits + f.fbits)) | q;
    }
    if (e > max_exp) return (sign << (f.ebits + f.fbits)) | ((f.finite_only && f.ebits < 4 ? emask : emask - 1) << f.fbits) | fmask;
    auto frac = round_even(std::ldexp(x, f.fbits - e) - static_cast<long double>(std::uint64_t{1} << f.fbits));
    if (frac == (std::uint64_t{1} << f.fbits)) { ++e; frac = 0; }
    if (e > max_exp) return (sign << (f.ebits + f.fbits)) | ((f.finite_only && f.ebits < 4 ? emask : emask - 1) << f.fbits) | fmask;
    return (sign << (f.ebits + f.fbits)) | (static_cast<std::uint64_t>(e + f.bias) << f.fbits) | frac;
}

template <typename T> inline T bits(std::uint64_t x) noexcept { return std::bit_cast<T>(x); }
}

[[nodiscard]] inline std::uint64_t scalar_add(DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    using namespace scalar_add_detail;
    switch (type) {
        case DataType::I2: case DataType::U2: case DataType::I4: case DataType::U4:
        case DataType::I8: case DataType::U8: case DataType::I16: case DataType::U16:
        case DataType::I32: case DataType::U32: case DataType::I64: case DataType::U64: {
            const unsigned w = (type == DataType::I2 || type == DataType::U2) ? 2 : (type == DataType::I4 || type == DataType::U4) ? 4 : (type == DataType::I8 || type == DataType::U8) ? 8 : (type == DataType::I16 || type == DataType::U16) ? 16 : (type == DataType::I32 || type == DataType::U32) ? 32 : 64;
            const std::uint64_t mask = w == 64 ? ~std::uint64_t{} : (std::uint64_t{1} << w) - 1;
            return (a + b) & mask;
        }
        case DataType::F4_E2M1: case DataType::F6_E2M3: case DataType::F6_E3M2: case DataType::F8_E4M3FN: case DataType::F8_E5M2: {
            const Format f = format(type); const long double x = decode_small(a, f), y = decode_small(b, f);
            long double z;
            if (std::isnan(x) || std::isnan(y) || (std::isinf(x) && std::isinf(y) && std::signbit(x) != std::signbit(y))) z = std::numeric_limits<long double>::quiet_NaN();
            else z = x + y;
            return encode_small(z, f);
        }
        case DataType::F16: case DataType::BF16: case DataType::F32: case DataType::F64: {
            const Format f = type == DataType::F16 ? Format{16, 5, 10, 15, false, true}
                              : type == DataType::BF16 ? Format{16, 8, 7, 127, false, true}
                              : type == DataType::F32 ? Format{32, 8, 23, 127, false, true}
                              : Format{64, 11, 52, 1023, false, true};
            const long double x = decode_small(a, f), y = decode_small(b, f);
            long double z;
            if (std::isnan(x) || std::isnan(y)
                || (std::isinf(x) && std::isinf(y)
                    && std::signbit(x) != std::signbit(y))) {
                z = std::numeric_limits<long double>::quiet_NaN();
            } else if (x == 0 && y == 0) {
                z = (std::signbit(x) && std::signbit(y)) ? -0.0L : 0.0L;
            } else {
                z = x + y;
                if (z == 0) z = 0.0L;
            }
            return encode_small(z, f);
        }
        default: return 0;
    }
}
}

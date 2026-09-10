#pragma once

#include <cstdint>

#include "iom/tensor.hpp"

namespace iom::detail::scalar_binary_codec_detail {

enum class BinaryOp { add, mul, sub, div };

struct Format {
    unsigned bits, ebits, fbits;
    int bias;
    bool finite_only, infs;
};

#ifndef IOM_SCALAR_CODEC_QUALIFIER
#ifdef IOM_GPU_DEVICE
#define IOM_SCALAR_CODEC_QUALIFIER IOM_GPU_DEVICE inline
#else
#define IOM_SCALAR_CODEC_QUALIFIER inline
#endif
#define IOM_SCALAR_CODEC_QUALIFIER_LOCAL
#endif

template <typename Traits>
struct Codec {
    using carrier_type = typename Traits::carrier_type;

    IOM_SCALAR_CODEC_QUALIFIER static Format format(DataType type) noexcept {
        switch (type) {
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

    IOM_SCALAR_CODEC_QUALIFIER static carrier_type decode(
            std::uint64_t raw, Format f) noexcept {
        const auto sign = raw >> (f.ebits + f.fbits);
        const auto emask = (std::uint64_t{1} << f.ebits) - 1;
        const auto frac = raw & ((std::uint64_t{1} << f.fbits) - 1);
        const auto exp = (raw >> f.fbits) & emask;
        if (exp == emask && (!f.finite_only || f.ebits >= 4)) {
            if (f.infs && frac == 0)
                return sign ? -Traits::positive_infinity()
                            : Traits::positive_infinity();
            return Traits::quiet_nan();
        }
        const carrier_type value = exp
                ? Traits::ldexp(
                          static_cast<carrier_type>((std::uint64_t{1} << f.fbits) + frac),
                          static_cast<int>(exp) - f.bias - static_cast<int>(f.fbits))
                : Traits::ldexp(
                          static_cast<carrier_type>(frac),
                          1 - f.bias - static_cast<int>(f.fbits));
        return sign ? -value : value;
    }

    IOM_SCALAR_CODEC_QUALIFIER static std::uint64_t encode(
            carrier_type value, Format f) noexcept {
        const auto sign = Traits::signbit(value) ? std::uint64_t{1} : 0;
        value = Traits::fabs(value);
        const auto emask = (std::uint64_t{1} << f.ebits) - 1;
        const auto fmask = (std::uint64_t{1} << f.fbits) - 1;
        const auto finite_emax = f.finite_only && f.ebits < 4 ? emask : emask - 1;
        const auto pack = [=](std::uint64_t exponent, std::uint64_t mantissa) {
            return (sign << (f.ebits + f.fbits))
                    | (exponent << f.fbits) | mantissa;
        };
        const auto overflow = f.infs ? pack(emask, 0) : pack(finite_emax, fmask);
        if (Traits::isnan(value)) {
            if (f.finite_only && f.ebits < 4) value = Traits::max_finite();
            else {
                return pack(emask, f.finite_only
                        ? fmask : std::uint64_t{1} << (f.fbits - 1));
            }
        }
        if (Traits::isinf(value)) return overflow;
        if (value == 0) return sign << (f.ebits + f.fbits);
        const int min_sub = 1 - f.bias - static_cast<int>(f.fbits);
        const int max_exp = static_cast<int>(finite_emax) - f.bias;
        int exponent = 0;
        (void)Traits::frexp(value, &exponent);
        --exponent;
        if (exponent < min_sub + static_cast<int>(f.fbits)) {
            const auto q = round_even(Traits::ldexp(value, -min_sub));
            return q == (std::uint64_t{1} << f.fbits)
                    ? pack(1, 0) : pack(0, q);
        }
        if (exponent > max_exp) return overflow;
        auto fraction = round_even(
                Traits::ldexp(value, f.fbits - exponent)
                - static_cast<carrier_type>(std::uint64_t{1} << f.fbits));
        if (fraction == (std::uint64_t{1} << f.fbits)) {
            ++exponent;
            fraction = 0;
        }
        if (exponent > max_exp) return overflow;
        return pack(static_cast<std::uint64_t>(exponent + f.bias), fraction);
    }

    IOM_SCALAR_CODEC_QUALIFIER static std::uint64_t round_even(
            carrier_type value) noexcept {
        const auto integral = Traits::floor(value);
        const auto remainder = value - integral;
        return static_cast<std::uint64_t>(integral)
                + (remainder > static_cast<carrier_type>(0.5)
                   || (remainder == static_cast<carrier_type>(0.5)
                       && (static_cast<std::uint64_t>(integral) & 1)));
    }

    template <BinaryOp Op>
    IOM_SCALAR_CODEC_QUALIFIER static carrier_type arithmetic(
            carrier_type x, carrier_type y) noexcept {
        if (Traits::isnan(x) || Traits::isnan(y)) return Traits::quiet_nan();
        if constexpr (Op == BinaryOp::add) {
            if (Traits::isinf(x) && Traits::isinf(y)
                    && Traits::signbit(x) != Traits::signbit(y))
                return Traits::quiet_nan();
            if (x == 0 && y == 0)
                return (Traits::signbit(x) && Traits::signbit(y))
                        ? -static_cast<carrier_type>(0) : static_cast<carrier_type>(0);
            const auto result = x + y;
            return result == 0 ? static_cast<carrier_type>(0) : result;
        } else if constexpr (Op == BinaryOp::mul) {
            if ((Traits::isinf(x) && y == 0) || (Traits::isinf(y) && x == 0))
                return Traits::quiet_nan();
            return x * y;
        } else if constexpr (Op == BinaryOp::sub) {
            if (Traits::isinf(x) && Traits::isinf(y)
                    && Traits::signbit(x) == Traits::signbit(y))
                return Traits::quiet_nan();
            return x - y;
        } else {
            if ((x == 0 && y == 0)
                    || (Traits::isinf(x) && Traits::isinf(y)))
                return Traits::quiet_nan();
            return x / y;
        }
    }

    IOM_SCALAR_CODEC_QUALIFIER static unsigned integer_width(
            DataType type) noexcept {
        switch (type) {
            case DataType::I2: case DataType::U2: return 2;
            case DataType::I4: case DataType::U4: return 4;
            case DataType::I8: case DataType::U8: return 8;
            case DataType::I16: case DataType::U16: return 16;
            case DataType::I32: case DataType::U32: return 32;
            case DataType::I64: case DataType::U64: return 64;
            default: return 0;
        }
    }

    template <BinaryOp Op>
    IOM_SCALAR_CODEC_QUALIFIER static std::uint64_t binary(
            DataType type, std::uint64_t a, std::uint64_t b) noexcept {
        const unsigned width = integer_width(type);
        if (width) {
            if constexpr (Op == BinaryOp::div) return 0;
            const auto mask = width == 64
                    ? ~std::uint64_t{} : (std::uint64_t{1} << width) - 1;
            if constexpr (Op == BinaryOp::add) return (a + b) & mask;
            if constexpr (Op == BinaryOp::mul) return (a * b) & mask;
            return (a - b) & mask;
        }
        const auto f = format(type);
        if (!f.bits) return 0;
        return encode(arithmetic<Op>(decode(a, f), decode(b, f)), f);
    }
};

#ifdef IOM_SCALAR_CODEC_QUALIFIER_LOCAL
#undef IOM_SCALAR_CODEC_QUALIFIER
#undef IOM_SCALAR_CODEC_QUALIFIER_LOCAL
#endif

}  // namespace iom::detail::scalar_binary_codec_detail

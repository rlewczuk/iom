#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "scalar_binary_codec.hpp"

namespace iom::detail {
namespace scalar_add_detail {

using BinaryOp = scalar_binary_codec_detail::BinaryOp;
using Format = scalar_binary_codec_detail::Format;

struct HostTraits {
    using carrier_type = long double;

    static inline carrier_type positive_infinity() noexcept {
        return std::numeric_limits<carrier_type>::infinity();
    }
    static inline carrier_type quiet_nan() noexcept {
        return std::numeric_limits<carrier_type>::quiet_NaN();
    }
    static inline carrier_type max_finite() noexcept {
        return std::numeric_limits<carrier_type>::max();
    }
    static inline bool isnan(carrier_type value) noexcept { return std::isnan(value); }
    static inline bool isinf(carrier_type value) noexcept { return std::isinf(value); }
    static inline bool signbit(carrier_type value) noexcept { return std::signbit(value); }
    static inline carrier_type fabs(carrier_type value) noexcept { return std::fabs(value); }
    static inline carrier_type floor(carrier_type value) noexcept { return std::floor(value); }
    static inline carrier_type ldexp(carrier_type value, int exponent) noexcept {
        return std::ldexp(value, exponent);
    }
    static inline carrier_type frexp(carrier_type value, int* exponent) noexcept {
        return std::frexp(value, exponent);
    }
};

using HostCodec = scalar_binary_codec_detail::Codec<HostTraits>;

inline Format format(DataType type) noexcept {
    return HostCodec::format(type);
}
inline long double decode_small(std::uint64_t raw, Format f) noexcept {
    return HostCodec::decode(raw, f);
}
inline std::uint64_t encode_small(long double value, Format f) noexcept {
    return HostCodec::encode(value, f);
}

template <BinaryOp Op>
[[nodiscard]] inline std::uint64_t scalar_binary(
        DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return HostCodec::binary<Op>(type, a, b);
}

}  // namespace scalar_add_detail

template <scalar_add_detail::BinaryOp Op>
[[nodiscard]] inline std::uint64_t scalar_binary(
        DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_add_detail::scalar_binary<Op>(type, a, b);
}
inline std::uint64_t scalar_add(
        DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::add>(type, a, b);
}
inline std::uint64_t scalar_mul(
        DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::mul>(type, a, b);
}
inline std::uint64_t scalar_sub(
        DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::sub>(type, a, b);
}
inline std::uint64_t scalar_div(
        DataType type, std::uint64_t a, std::uint64_t b) noexcept {
    return scalar_binary<scalar_add_detail::BinaryOp::div>(type, a, b);
}
}  // namespace iom::detail

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "scalar_binary_codec.hpp"

namespace iom::detail {
namespace scalar_silu_detail {

using Format = scalar_binary_codec_detail::Format;

#ifndef IOM_SCALAR_SILU_QUALIFIER
#ifdef IOM_GPU_DEVICE
#define IOM_SCALAR_SILU_QUALIFIER IOM_GPU_DEVICE inline
#else
#define IOM_SCALAR_SILU_QUALIFIER inline
#endif
#define IOM_SCALAR_SILU_QUALIFIER_LOCAL
#endif

// Traits are deliberately supplied by the caller for device instantiations.
// Their carrier_type is the evaluation domain and must provide the operations
// required by Codec plus exp().
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
    static inline carrier_type exp(carrier_type value) noexcept {
        return std::exp(value);
    }
    static inline bool isnan(carrier_type value) noexcept {
        return std::isnan(value);
    }
    static inline bool isinf(carrier_type value) noexcept {
        return std::isinf(value);
    }
    static inline bool signbit(carrier_type value) noexcept {
        return std::signbit(value);
    }
    static inline carrier_type fabs(carrier_type value) noexcept {
        return std::fabs(value);
    }
    static inline carrier_type floor(carrier_type value) noexcept {
        return std::floor(value);
    }
    static inline carrier_type ldexp(carrier_type value, int exponent) noexcept {
        return std::ldexp(value, exponent);
    }
    static inline carrier_type frexp(carrier_type value, int* exponent) noexcept {
        return std::frexp(value, exponent);
    }
};

using HostCodec = scalar_binary_codec_detail::Codec<HostTraits>;

// Stable wide-domain SiLU evaluation. The half-exponential form for negative
// finite inputs keeps exp() in range and intentionally preserves the
// left-associated numerator order required for representable negative tails.
template <typename Traits>
IOM_SCALAR_SILU_QUALIFIER typename Traits::carrier_type evaluate(
        typename Traits::carrier_type x) noexcept {
    using Carrier = typename Traits::carrier_type;
    if (Traits::isnan(x)) return Traits::quiet_nan();
    if (Traits::isinf(x)) {
        return Traits::signbit(x)
                ? -static_cast<Carrier>(0)
                : Traits::positive_infinity();
    }
    if (x == static_cast<Carrier>(0)) return x;
    if (x >= static_cast<Carrier>(0)) {
        return x / (static_cast<Carrier>(1) + Traits::exp(-x));
    }

    const Carrier t = Traits::exp(x * static_cast<Carrier>(0.5));
    const Carrier numerator = (x * t) * t;
    return numerator / (static_cast<Carrier>(1) + t * t);
}

template <typename Traits>
IOM_SCALAR_SILU_QUALIFIER std::uint64_t scalar_silu(
        DataType type, std::uint64_t raw) noexcept {
    using Codec = scalar_binary_codec_detail::Codec<Traits>;
    const Format format = Codec::format(type);
    if (!format.bits) return 0;
    const auto value = Codec::decode(raw, format);
    return Codec::encode(evaluate<Traits>(value), format);
}

#ifdef IOM_SCALAR_SILU_QUALIFIER_LOCAL
#undef IOM_SCALAR_SILU_QUALIFIER
#undef IOM_SCALAR_SILU_QUALIFIER_LOCAL
#endif

}  // namespace scalar_silu_detail

[[nodiscard]] inline std::uint64_t scalar_silu(
        DataType type, std::uint64_t raw) noexcept {
    return scalar_silu_detail::scalar_silu<scalar_silu_detail::HostTraits>(
            type, raw);
}

}  // namespace iom::detail

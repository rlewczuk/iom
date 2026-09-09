#pragma once

#include <cstdint>

namespace iom {

    using oid = std::int64_t;

    enum class OidError : oid {
        InvalidArgument = -1,
        Unsupported = -2,
        Overflow = -3,
        ResourceExhausted = -4,
        DeviceError = -5,
        InternalError = -6,
    };

    [[nodiscard]] constexpr oid to_oid(OidError error) noexcept {
        return static_cast<oid>(error);
    }

    [[nodiscard]] constexpr bool oid_is_error(oid value) noexcept {
        return value < 0;
    }

    [[nodiscard]] constexpr bool oid_is_token(oid value) noexcept {
        return value > 0;
    }

}  // namespace iom

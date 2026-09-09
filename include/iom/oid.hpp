#pragma once

#include <cstdint>

namespace iom {

    /**
     * Signed operation identifier returned by every public DeviceOps
     * submission facade.  Negative values are terminal OidError results;
     * positive values are accepted asynchronous tokens; zero is invalid and
     * is neither an error nor a token.
     */
    using oid = std::int64_t;

    /** Stable negative results returned synchronously by OID facades. */
    enum class OidError : oid {
        InvalidArgument = -1,
        Unsupported = -2,
        Overflow = -3,
        ResourceExhausted = -4,
        DeviceError = -5,
        InternalError = -6,
    };

    /**
     * Convert an OidError to its negative signed representation.  This is a
     * sign-preserving, non-throwing classification helper.
     */
    [[nodiscard]] constexpr oid to_oid(OidError error) noexcept {
        return static_cast<oid>(error);
    }

    /** Return true exactly for negative OID error results. */
    [[nodiscard]] constexpr bool oid_is_error(oid value) noexcept {
        return value < 0;
    }

    /** Return true exactly for positive, accepted asynchronous OID tokens. */
    [[nodiscard]] constexpr bool oid_is_token(oid value) noexcept {
        return value > 0;
    }

}  // namespace iom

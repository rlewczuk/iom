#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace iom::detail {

// Full tensor shapes span rank two through rank eight inclusive.
// Implementation-private: no public rank constant or query API exists, and a
// leading-dimension helper span is never a full shape.
inline constexpr std::size_t kMaxTensorRank = 8;

inline std::size_t checked_add(
        std::size_t lhs, std::size_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::overflow_error(what);
    }
    return lhs + rhs;
}

inline std::size_t checked_mul(
        std::size_t lhs, std::size_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(what);
    }
    return lhs * rhs;
}

inline std::size_t bits_to_bytes(std::size_t bits, const char* what) {
    return checked_add(bits, 7, what) / 8;
}

struct UnsupportedOperation final : std::runtime_error {
    UnsupportedOperation()
            : std::runtime_error("operation is unsupported") {}
};

}  // namespace iom::detail

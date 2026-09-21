#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace iom::detail {

// A backing key is stable and canonical across wrappers. A non-null base
// denotes addressable storage; a null base denotes opaque storage. The key
// is only compared, never used as an address, offset origin, or descriptor.
struct StorageIdentity {
    void* key = nullptr;
    void* base = nullptr;

    friend bool operator==(const StorageIdentity&, const StorageIdentity&) = default;
};

// Checked byte extent for addressable storage, logical extent for opaque
// storage. Opaque offsets have meaning only within one exact owner.
struct StorageRange {
    StorageIdentity backing;
    std::size_t offset = 0;
    std::size_t bytes = 0;

    [[nodiscard]] void* address() const noexcept {
        if (backing.base == nullptr) return nullptr;
        return reinterpret_cast<void*>(
                reinterpret_cast<std::uintptr_t>(backing.base) + offset);
    }

    [[nodiscard]] void* registry_key() const noexcept {
        return backing.base == nullptr ? backing.key : address();
    }
};

[[nodiscard]] inline StorageRange checked_storage_range(
        StorageIdentity backing, std::size_t offset, std::size_t bytes,
        const char* overflow) {
    if (backing.key == nullptr) {
        throw std::invalid_argument("storage backing identity is null");
    }
    if (bytes > std::numeric_limits<std::size_t>::max() - offset) {
        throw std::overflow_error(overflow);
    }
    if (backing.base != nullptr) {
        const auto base = reinterpret_cast<std::uintptr_t>(backing.base);
        const auto limit = std::numeric_limits<std::uintptr_t>::max();
        if (offset > limit - base || bytes > limit - base - offset) {
            throw std::overflow_error(overflow);
        }
    }
    return {backing, offset, bytes};
}

// Operation layers retain their own alias policies and validation order.
// This helper compares facts only, never execution handles. Opaque backing
// cannot overlap a host/addressable range merely because its key is nearby.
[[nodiscard]] inline bool storage_ranges_overlap(
        const StorageRange& lhs, const StorageRange& rhs) noexcept {
    if (lhs.bytes == 0 || rhs.bytes == 0) return false;
    if (lhs.backing.base == nullptr || rhs.backing.base == nullptr) {
        return lhs.backing.base == nullptr && rhs.backing.base == nullptr
                && lhs.backing.key == rhs.backing.key;
    }
    const auto left = reinterpret_cast<std::uintptr_t>(lhs.address());
    const auto right = reinterpret_cast<std::uintptr_t>(rhs.address());
    return left < right + rhs.bytes && right < left + lhs.bytes;
}

}  // namespace iom::detail

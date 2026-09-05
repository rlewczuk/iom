#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

#include "iom/alloc.hpp"

namespace iom::detail {

inline constexpr std::size_t kStorageAlignment = 32;
struct no_op {
    void operator()() const noexcept {}
};

inline void* allocate_aligned_storage(
        Allocator& allocator, std::size_t nbytes,
        std::invocable<> auto pre_allocate,
        std::string_view misalignment_message) {
    pre_allocate();
    void* address = allocator.alloc(nbytes);
    if (address == nullptr) {
        throw std::bad_alloc();
    }
    if (reinterpret_cast<std::uintptr_t>(address) % kStorageAlignment != 0) {
        allocator.free(address);
        throw std::runtime_error(std::string(misalignment_message));
    }
    return address;
}

template <typename PreRelease = no_op>
    requires std::invocable<PreRelease>
inline void release_aligned_storage(
        Allocator& allocator, void*& address,
        PreRelease pre_release = {}) {
    if (address == nullptr) {
        return;
    }
    try {
        pre_release();
    } catch (...) {
    }
    try {
        allocator.free(address);
    } catch (...) {
    }
    address = nullptr;
}

}  // namespace iom::detail

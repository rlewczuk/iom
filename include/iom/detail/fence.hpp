#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace iom::detail {

inline constexpr std::size_t kFenceStorageBytes = 32;
inline constexpr std::size_t kFenceStorageAlign = alignof(std::max_align_t);

using EntryId = std::uint64_t;
using QueueId = std::uint64_t;
enum class EntryState { Live, Invalidated };

struct FenceResult {
    bool succeeded = true;
    std::exception_ptr failure;

    [[nodiscard]] static FenceResult success() noexcept {
        return {true, nullptr};
    }

    // A submission that has not reached a terminal state yet. Never a
    // success: callers that observe it must hold the storage back.
    [[nodiscard]] static FenceResult pending() noexcept {
        return {false, nullptr};
    }

    [[nodiscard]] static FenceResult failed(
            std::exception_ptr error) noexcept {
        return {false, std::move(error)};
    }
};

struct Fence {
    alignas(kFenceStorageAlign) unsigned char storage[kFenceStorageBytes]{};
    FenceResult (*invoke)(const Fence&) noexcept = nullptr;
    void (*copy_construct)(Fence* dst, const Fence& src) noexcept = nullptr;
    void (*move_construct)(Fence* dst, Fence* src) noexcept = nullptr;
    void (*destroy)(Fence*) noexcept = nullptr;

    Fence() noexcept = default;

    ~Fence() noexcept {
        if (destroy != nullptr) {
            destroy(this);
        }
    }

    Fence(const Fence& other) noexcept {
        if (other.copy_construct != nullptr) {
            other.copy_construct(this, other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
    }

    Fence(Fence&& other) noexcept {
        if (other.move_construct != nullptr) {
            other.move_construct(this, &other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
        other.invoke = nullptr;
        other.copy_construct = nullptr;
        other.move_construct = nullptr;
        other.destroy = nullptr;
    }

    Fence& operator=(const Fence& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (destroy != nullptr) {
            destroy(this);
        }
        if (other.copy_construct != nullptr) {
            other.copy_construct(this, other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
        return *this;
    }

    Fence& operator=(Fence&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (destroy != nullptr) {
            destroy(this);
        }
        if (other.move_construct != nullptr) {
            other.move_construct(this, &other);
        } else {
            std::memcpy(storage, other.storage, kFenceStorageBytes);
        }
        invoke = other.invoke;
        copy_construct = other.copy_construct;
        move_construct = other.move_construct;
        destroy = other.destroy;
        other.invoke = nullptr;
        other.copy_construct = nullptr;
        other.move_construct = nullptr;
        other.destroy = nullptr;
        return *this;
    }

    FenceResult operator()() const { return invoke(*this); }
    explicit operator bool() const noexcept { return invoke != nullptr; }
};

static_assert(std::is_nothrow_default_constructible_v<Fence>);
static_assert(std::is_nothrow_copy_constructible_v<Fence>);
static_assert(std::is_nothrow_move_constructible_v<Fence>);
static_assert(std::is_nothrow_destructible_v<Fence>);
static_assert(std::is_nothrow_copy_assignable_v<Fence>);
static_assert(std::is_nothrow_move_assignable_v<Fence>);

template <typename Capture>
struct FenceCaptureOps {
    static_assert(std::is_nothrow_copy_constructible_v<Capture>
            && std::is_nothrow_move_constructible_v<Capture>
            && std::is_nothrow_destructible_v<Capture>);
    static_assert(sizeof(Capture) <= kFenceStorageBytes);
    static_assert(alignof(Capture) <= kFenceStorageAlign);

    static void copy_construct(
            Fence* destination, const Fence& source) noexcept {
        ::new (destination->storage) Capture{
                *std::launder(reinterpret_cast<const Capture*>(
                        source.storage))};
    }

    static void move_construct(Fence* destination, Fence* source) noexcept {
        ::new (destination->storage) Capture{
                std::move(*std::launder(reinterpret_cast<Capture*>(
                        source->storage)))};
        std::destroy_at(std::launder(reinterpret_cast<Capture*>(
                source->storage)));
    }

    static void destroy(Fence* fence) noexcept {
        std::destroy_at(
                std::launder(reinterpret_cast<Capture*>(fence->storage)));
    }
};

inline FenceResult failed_invalidated_fence_invoke(
        const Fence&) noexcept {
    try {
        return FenceResult::failed(std::make_exception_ptr(
                std::runtime_error(
                        "outstanding-work entry was invalidated")));
    } catch (...) {
        return FenceResult::failed(std::current_exception());
    }
}

inline Fence make_invalidated_fence() noexcept {
    Fence f;
    f.invoke = &failed_invalidated_fence_invoke;
    return f;
}

}  // namespace iom::detail

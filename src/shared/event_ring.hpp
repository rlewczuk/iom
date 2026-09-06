#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "iom/detail/outstanding_work_registry.hpp"
#include "metadata_slot_pool.hpp"

namespace iom::detail {

template <typename Policy>
class EventRingState final {
public:
    using context_type = typename Policy::context_type;
    using event_type = typename Policy::event_type;

    static constexpr std::size_t kEventRingCount = 16;
    static constexpr std::size_t kNoAttachedSlot =
            std::numeric_limits<std::size_t>::max();

    struct Slot {
        event_type event = nullptr;
        bool in_use = false;
        std::size_t attached_metadata_slot = kNoAttachedSlot;
        FenceResult cached_result = FenceResult::success();
    };

    EventRingState(
            context_type context, MetadataSlotPool<Policy>& metadata_pool)
            : context_(context), metadata_pool_(&metadata_pool) {}

    ~EventRingState() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < created_count_; ++index) {
            Slot& slot = slots_[index];
            if (slot.in_use) {
                if (slot.attached_metadata_slot != kNoAttachedSlot) {
                    metadata_pool_->release(slot.attached_metadata_slot);
                    slot.attached_metadata_slot = kNoAttachedSlot;
                }
                slot.in_use = false;
            }
        }
        try {
            Policy::activate(context_);
        } catch (...) {
        }
        for (std::size_t index = 0; index < created_count_; ++index) {
            Slot& slot = slots_[index];
            Policy::synchronize_event_noexcept(slot.event);
            Policy::destroy_event_noexcept(slot.event);
            slot.event = nullptr;
        }
    }

    EventRingState(const EventRingState&) = delete;
    EventRingState& operator=(const EventRingState&) = delete;
    EventRingState(EventRingState&&) = delete;
    EventRingState& operator=(EventRingState&&) = delete;

    [[nodiscard]] Slot& acquire() {
        Policy::check_acquire_event_fault();
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            for (std::size_t index = 0; index < created_count_; ++index) {
                Slot& slot = slots_[index];
                if (!slot.in_use) {
                    slot.in_use = true;
                    slot.attached_metadata_slot = kNoAttachedSlot;
                    return slot;
                }
            }
            if (created_count_ < kEventRingCount) {
                Slot& slot = slots_[created_count_];
                Policy::create_event(&slot.event);
                ++created_count_;
                slot.in_use = true;
                slot.attached_metadata_slot = kNoAttachedSlot;
                return slot;
            }
            completion_.wait(lock, [this] {
                for (std::size_t index = 0; index < created_count_; ++index) {
                    if (!slots_[index].in_use) {
                        return true;
                    }
                }
                return false;
            });
        }
    }

    void release(Slot& slot) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        release_locked(slot);
    }

    void on_worker_complete(Slot& slot) {
        std::unique_lock<std::mutex> lock(mutex_);
        FenceResult result;
        try {
            Policy::activate(context_);
            Policy::synchronize_event(slot.event);
            result = FenceResult::success();
        } catch (...) {
            result = FenceResult::failed(std::current_exception());
        }
        slot.cached_result = result;
        lock.unlock();
        if (result.failure) {
            std::rethrow_exception(result.failure);
        }
    }

    void on_worker_destroy(Slot& slot) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            Policy::activate(context_);
        } catch (...) {
        }
        Policy::synchronize_event_noexcept(slot.event);
        if (slot.attached_metadata_slot != kNoAttachedSlot) {
            metadata_pool_->release(slot.attached_metadata_slot);
        }
        release_locked(slot);
    }

    // Returns the result recorded by the worker without touching the event.
    [[nodiscard]] FenceResult invoke_result(
            std::size_t slot_index) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return slots_[slot_index].cached_result;
    }


    [[nodiscard]] std::size_t slot_index(const Slot& slot) const noexcept {
        return static_cast<std::size_t>(&slot - slots_.data());
    }

#ifdef IOM_ENABLE_TESTING
    [[nodiscard]] std::size_t created_event_count_for_testing()
            const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return created_count_;
    }

    [[nodiscard]] std::size_t in_use_count_for_testing() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (std::size_t index = 0; index < created_count_; ++index) {
            count += slots_[index].in_use ? 1 : 0;
        }
        return count;
    }
#endif

private:
    void release_locked(Slot& slot) noexcept {
        if (!slot.in_use) {
            return;
        }
        slot.in_use = false;
        slot.attached_metadata_slot = kNoAttachedSlot;
        completion_.notify_one();
    }

    context_type context_;
    MetadataSlotPool<Policy>* metadata_pool_;
    std::array<Slot, kEventRingCount> slots_;
    std::size_t created_count_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable completion_;
};

}  // namespace iom::detail

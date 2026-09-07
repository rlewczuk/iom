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
class EventRingState final
        : public std::enable_shared_from_this<EventRingState<Policy>> {
public:
    using context_type = typename Policy::context_type;
    using event_type = typename Policy::event_type;

    static constexpr std::size_t kEventRingCount = 16;
    static constexpr std::size_t kNoAttachedSlot =
            std::numeric_limits<std::size_t>::max();

    // One pooled event. Owned by the ring for its whole lifetime and reused
    // for later submissions once every record referencing it is gone.
    struct Slot {
        event_type event = nullptr;
        bool in_use = false;
    };

    // One submission's completion record. The registry fence captures a
    // shared_ptr to this record, so the record — and its pooled event — stays
    // reserved until every registry entry (and destructor snapshot) that
    // references it has disappeared. Re-acquiring the pool entry for a later
    // submission therefore cannot change the result observed by an earlier
    // fence. The result starts pending (non-success) and only becomes
    // success after the worker has synchronized this submission's event.
    class Submission final {
    public:
        ~Submission() noexcept {
            ring_->return_pool_entry(pool_index_);
        }

        // Returns the result recorded by this submission's worker, or a
        // pending (non-success) result until on_worker_complete has
        // synchronized the event. Never touches the event.
        [[nodiscard]] FenceResult invoke_result() const noexcept {
            std::lock_guard<std::mutex> lock(ring_->mutex_);
            return result_;
        }

        void attach_metadata_slot(std::size_t slot) noexcept {
            metadata_slot_ = slot;
        }

        // Public so std::make_shared can allocate the record in a single
        // allocation: access to a nested class's private constructor is not
        // granted inside <memory>'s make_shared instantiation. Only
        // EventRingState::acquire constructs it, with a valid pool index
        // and while holding the ring's own shared reference.
        Submission(
                std::shared_ptr<EventRingState> ring,
                std::size_t pool_index) noexcept
                : ring_(std::move(ring)), pool_index_(pool_index) {}

    private:
        friend class EventRingState;

        std::shared_ptr<EventRingState> ring_;
        std::size_t pool_index_ = kNoAttachedSlot;
        FenceResult result_ = FenceResult::pending();
        std::size_t metadata_slot_ = kNoAttachedSlot;
        // True once this submission's on_worker_complete successfully
        // synchronized its pooled event. Fresh records start false; because
        // acquire() constructs a new record for every reuse of a pool entry,
        // the marker is implicitly reset on slot acquire. It stays false for
        // queue-drain and failed/partially-observed paths, which must still
        // perform the noexcept cleanup wait on on_worker_destroy.
        bool synchronized_on_complete_ = false;
    };

    EventRingState(
            context_type context, MetadataSlotPool<Policy>& metadata_pool)
            : context_(context), metadata_pool_(&metadata_pool) {}

    ~EventRingState() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // No submission record can outlive the ring (every record holds
            // a shared reference), so no pool entry is reserved here.
            for (std::size_t index = 0; index < created_count_; ++index) {
                slots_[index].in_use = false;
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

    [[nodiscard]] std::shared_ptr<Submission> acquire() {
        Policy::check_acquire_event_fault();
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            for (std::size_t index = 0; index < created_count_; ++index) {
                Slot& slot = slots_[index];
                if (!slot.in_use) {
                    slot.in_use = true;
                    return std::make_shared<Submission>(
                            this->shared_from_this(), index);
                }
            }
            if (created_count_ < kEventRingCount) {
                Slot& slot = slots_[created_count_];
                Policy::create_event(&slot.event);
                ++created_count_;
                slot.in_use = true;
                return std::make_shared<Submission>(
                        this->shared_from_this(), created_count_ - 1);
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

    void on_worker_complete(Submission& submission) {
        std::unique_lock<std::mutex> lock(mutex_);
        FenceResult result;
        try {
            Policy::activate(context_);
            Policy::synchronize_event(
                    slots_[submission.pool_index_].event);
            result = FenceResult::success();
            submission.synchronized_on_complete_ = true;
        } catch (...) {
            result = FenceResult::failed(std::current_exception());
        }
        submission.result_ = result;
        lock.unlock();
        if (result.failure) {
            std::rethrow_exception(result.failure);
        }
    }

    // Releases the recorded event's resources (metadata) for reuse. The
    // pooled event itself is only returned to the pool once the submission
    // record is destroyed (its last fence reference gone), so an earlier
    // fence can never read a later submission's state. Normal completion is
    // waited exactly once: on_worker_destroy skips the second synchronization
    // when the worker's on_worker_complete already waited and observed this
    // event successfully. Queue-drain and failed/partially-observed paths
    // (marker unset) retain the required cleanup wait.
    void on_worker_destroy(Submission& submission) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!submission.synchronized_on_complete_) {
            try {
                Policy::activate(context_);
            } catch (...) {
            }
            Policy::synchronize_event_noexcept(
                    slots_[submission.pool_index_].event);
        }
        if (submission.metadata_slot_ != kNoAttachedSlot) {
            metadata_pool_->release(submission.metadata_slot_);
            submission.metadata_slot_ = kNoAttachedSlot;
        }
    }

    // The pooled event recorded for this submission. The event does not
    // change while the record lives.
    [[nodiscard]] event_type event_of(
            const Submission& submission) const noexcept {
        return slots_[submission.pool_index_].event;
    }

    [[nodiscard]] std::size_t slot_index(
            const Submission& submission) const noexcept {
        return submission.pool_index_;
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
    void return_pool_entry(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < created_count_ && slots_[index].in_use) {
            slots_[index].in_use = false;
            completion_.notify_one();
        }
    }

    context_type context_;
    MetadataSlotPool<Policy>* metadata_pool_;
    std::array<Slot, kEventRingCount> slots_;
    std::size_t created_count_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable completion_;
};

}  // namespace iom::detail
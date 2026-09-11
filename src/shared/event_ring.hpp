#pragma once

// Each queue owns exactly C completion resources, eagerly created at queue
// construction before the queue is published, and reuses them proof-by-proof.
// A pooled event is only a completion proof after the current submission
// records it successfully. A covering queue drain is an equivalent proof
// for the exceptional path where both event-record attempts fail. There is
// no lazy creation, replacement, or array resizing after setup: acquiring a
// submission waits for one of the fixed C resources to become free.

#include <stdexcept>

#include <array>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

#include "iom/detail/outstanding_work_registry.hpp"
#include "metadata_slot_pool.hpp"

namespace iom::detail {
enum class CompletionDisposition {
    Pending,
    Complete,
    RetireUnknown,
};

template <typename Policy>
class EventRingState final
        : public std::enable_shared_from_this<EventRingState<Policy>> {
public:
    using context_type = typename Policy::context_type;
    using event_type = typename Policy::event_type;

    static constexpr std::size_t kNoAttachedSlot =
            std::numeric_limits<std::size_t>::max();

    // One fixed completion resource. Created eagerly at queue setup and
    // reused for later submissions only after every record referencing it
    // is gone and its last use is proved.
    struct Slot {
        event_type event = nullptr;
        bool in_use = false;
    };

    // One submission's completion record. The registry fence captures a
    // shared_ptr to this record, so the record — and its fixed completion
    // resource — stays reserved until every registry entry (and destructor
    // snapshot) that references it has disappeared. Re-acquiring the
    // resource for a later submission therefore cannot change the result
    // observed by an earlier fence. The result starts pending (non-success)
    // and only becomes success after the worker has synchronized this
    // submission's event.
    class Submission final {
    public:
        ~Submission() noexcept {
            ring_->return_resource(pool_index_);
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
        // EventRingState::acquire constructs it, with a valid resource
        // index and while holding the ring's own shared reference.
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
        bool event_recorded_ = false;
        bool stream_drained_ = false;
        bool synchronized_on_complete_ = false;
        CompletionDisposition disposition_ =
                CompletionDisposition::Pending;
    };

    // Eagerly creates exactly event_count completion resources through the
    // backend policy. Any failure destroys only the resources already
    // created; the caller's rollback then returns the partition and
    // queue-count reservation while the native context remains valid.
    EventRingState(
            context_type context, MetadataSlotPool& metadata_pool,
            std::size_t event_count)
            : context_(context),
              metadata_pool_(&metadata_pool),
              event_count_(event_count),
              event_slots_(std::make_unique<Slot[]>(event_count)) {
        if (event_count_ == 0) {
            throw std::invalid_argument(
                    "event ring requires at least one completion resource");
        }
        Policy::activate(context_);
        try {
            for (std::size_t index = 0; index < event_count_; ++index) {
                Policy::check_queue_event_fault();
                Policy::create_event(&event_slots_[index].event);
            }
        } catch (...) {
            // Destroy only what was already created; nothing is published.
            for (std::size_t index = 0; index < event_count_; ++index) {
                Policy::destroy_event_noexcept(event_slots_[index].event);
                event_slots_[index].event = nullptr;
            }
            throw;
        }
    }

    ~EventRingState() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // No submission record can outlive the ring (every record holds
            // a shared reference), so no resource is reserved here.
            for (std::size_t index = 0; index < event_count_; ++index) {
                event_slots_[index].in_use = false;
            }
        }
        try {
            Policy::activate(context_);
        } catch (...) {
        }
        for (std::size_t index = 0; index < event_count_; ++index) {
            Slot& slot = event_slots_[index];
            Policy::synchronize_event_noexcept(slot.event);
            Policy::destroy_event_noexcept(slot.event);
            slot.event = nullptr;
        }
    }

    EventRingState(const EventRingState&) = delete;
    EventRingState& operator=(const EventRingState&) = delete;
    EventRingState(EventRingState&&) = delete;
    EventRingState& operator=(EventRingState&&) = delete;

    // Waits for one of the fixed C completion resources to become free.
    // Never creates, grows, or replaces a resource.
    [[nodiscard]] std::shared_ptr<Submission> acquire() {
        Policy::check_acquire_event_fault();
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (std::size_t index = 0; index < event_count_; ++index) {
                if (!event_slots_[index].in_use) {
                    return true;
                }
            }
            return false;
        });
        for (std::size_t index = 0; index < event_count_; ++index) {
            Slot& slot = event_slots_[index];
            if (!slot.in_use) {
                slot.in_use = true;
                return std::make_shared<Submission>(
                        this->shared_from_this(), index);
            }
        }
        throw std::logic_error(
                "event ring acquisition lost a free completion resource");
    }

    void mark_event_recorded(Submission& submission) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        submission.event_recorded_ = true;
    }

    void mark_stream_drained(Submission& submission) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        submission.stream_drained_ = true;
    }
    void mark_retire_unknown(Submission& submission) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        set_retire_unknown_locked(submission);
    }

    void on_worker_complete(Submission& submission) {
        std::unique_lock<std::mutex> lock(mutex_);
        FenceResult result;
        try {
            if (submission.stream_drained_) {
                result = FenceResult::success();
            } else if (submission.event_recorded_) {
                Policy::activate(context_);
                Policy::synchronize_event(
                        event_slots_[submission.pool_index_].event);
                result = FenceResult::success();
            } else {
                throw std::runtime_error(
                        "GPU completion event was never recorded");
            }
            submission.synchronized_on_complete_ = true;
            submission.disposition_ = CompletionDisposition::Complete;
        } catch (...) {
            result = FenceResult::failed(std::current_exception());
            set_retire_unknown_locked(submission);
        }
        submission.result_ = result;
        lock.unlock();
        if (result.failure) {
            std::rethrow_exception(result.failure);
        }
    }

    void on_worker_destroy(Submission& submission) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!submission.synchronized_on_complete_
                && submission.event_recorded_) {
            try {
                Policy::activate(context_);
                Policy::synchronize_event(
                        event_slots_[submission.pool_index_].event);
                submission.disposition_ = CompletionDisposition::Complete;
            } catch (...) {
                set_retire_unknown_locked(submission);
            }
        }
        if (submission.metadata_slot_ != kNoAttachedSlot) {
            if (submission.disposition_
                    == CompletionDisposition::RetireUnknown) {
                // Unknown completion: the whole lease stays reserved; this
                // slot is never reassigned until a covering proof.
                metadata_pool_->protect(submission.metadata_slot_);
            } else {
                metadata_pool_->release_after_proof(
                        submission.metadata_slot_);
            }
            submission.metadata_slot_ = kNoAttachedSlot;
        }
    }

    // A queue drain covers every retired submission on its stream: all
    // unknown completions are proved and every protected slot is released.
    void on_queue_drain(bool successful) noexcept {
        if (!successful) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        unproved_unknowns_ = 0;
        metadata_pool_->release_all_protected();
    }

    // True while any submission of this queue retired with an unknown
    // completion and no covering drain has proved it yet.
    [[nodiscard]] bool has_unproven_completion() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return unproved_unknowns_ != 0;
    }

    // The pooled event recorded for this submission. The event does not
    // change while the record lives.
    [[nodiscard]] event_type event_of(
            const Submission& submission) const noexcept {
        return event_slots_[submission.pool_index_].event;
    }

    [[nodiscard]] std::size_t slot_index(
            const Submission& submission) const noexcept {
        return submission.pool_index_;
    }

#ifdef IOM_ENABLE_TESTING
    // Exactly the C resources created at queue setup; never larger.
    [[nodiscard]] std::size_t event_count_for_testing() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return event_count_;
    }

    [[nodiscard]] std::size_t in_use_count_for_testing() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (std::size_t index = 0; index < event_count_; ++index) {
            count += event_slots_[index].in_use ? 1 : 0;
        }
        return count;
    }
#endif

private:
    void set_retire_unknown_locked(Submission& submission) noexcept {
        if (submission.disposition_ != CompletionDisposition::RetireUnknown) {
            submission.disposition_ = CompletionDisposition::RetireUnknown;
            ++unproved_unknowns_;
        }
    }

    void return_resource(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < event_count_ && event_slots_[index].in_use) {
            event_slots_[index].in_use = false;
            completion_.notify_one();
        }
    }

    context_type context_;
    MetadataSlotPool* metadata_pool_;
    std::size_t event_count_ = 0;
    std::unique_ptr<Slot[]> event_slots_;
    std::size_t unproved_unknowns_ = 0;
    mutable std::mutex mutex_;
    std::condition_variable completion_;
};

}  // namespace iom::detail

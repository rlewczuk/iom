#pragma once

// Each queue owns exactly C completion resources, eagerly created at queue
// construction before the queue is published, and reuses them proof-by-proof.
// A pooled event is only a completion proof after the current submission
// records it successfully. A covering queue drain is an equivalent proof
// for the exceptional path where both event-record attempts fail. There is
// no lazy creation, replacement, or array resizing after setup: acquiring a
// submission waits for one of the fixed C resources to become free.

#include <stdexcept>
#include <cstdint>
#include <optional>

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
    // reused for later submissions only after the current event use is
    // proved. The Submission retains only the immutable terminal result.
    struct Slot {
        event_type event = nullptr;
        void* owner = nullptr;
        bool in_use = false;
        bool quarantined = false;
    };

    // One submission's completion record. The registry fence captures a
    // shared_ptr to this record, so the terminal result remains available
    // after a proven event generation is returned to the fixed pool. An
    // unknown generation remains quarantined by the ring until a covering
    // queue drain proves that it is safe to reuse.
    class Submission final {
    public:
        ~Submission() noexcept {
            ring_->return_resource(*this);
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

        // Attached only after the status transfer has been enqueued. Its
        // metadata lease retains the cell until completion proof and capture.
        void attach_status_cell(const std::uint32_t* cell) noexcept {
            status_cell_ = cell;
        }

        [[nodiscard]] std::optional<std::uint32_t> status_word() const noexcept {
            std::lock_guard<std::mutex> lock(ring_->mutex_);
            return status_word_;
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

        [[nodiscard]] bool completion_proven() const noexcept {
            std::lock_guard<std::mutex> lock(ring_->mutex_);
            return disposition_ == CompletionDisposition::Complete;
        }
    private:
        friend class EventRingState;

        std::shared_ptr<EventRingState> ring_;
        std::size_t pool_index_ = kNoAttachedSlot;
        FenceResult result_ = FenceResult::pending();
        std::size_t metadata_slot_ = kNoAttachedSlot;
        const std::uint32_t* status_cell_ = nullptr;
        std::optional<std::uint32_t> status_word_;
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
            // Submission records hold the ring alive until their cached
            // result and any quarantine state are no longer referenced.
            for (std::size_t index = 0; index < event_count_; ++index) {
                event_slots_[index].in_use = false;
                event_slots_[index].owner = nullptr;
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
                auto submission = std::make_shared<Submission>(
                        this->shared_from_this(), index);
                slot.in_use = true;
                slot.owner = submission.get();
                return submission;
            }
        }
        throw std::logic_error(
                "event ring acquisition lost a free completion resource");
    }

    // Dispatch callbacks use this nonblocking form so a quarantined event
    // cannot turn common FIFO admission into a worker or submitter deadlock.
    [[nodiscard]] std::shared_ptr<Submission> try_acquire() {
        Policy::check_acquire_event_fault();
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < event_count_; ++index) {
            Slot& slot = event_slots_[index];
            if (!slot.in_use) {
                auto submission = std::make_shared<Submission>(
                        this->shared_from_this(), index);
                slot.in_use = true;
                slot.owner = submission.get();
                return submission;
            }
        }
        return nullptr;
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
            snapshot_status_locked(submission);
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
                && submission.stream_drained_) {
            submission.result_ = FenceResult::success();
            submission.synchronized_on_complete_ = true;
            submission.disposition_ = CompletionDisposition::Complete;
        } else if (!submission.synchronized_on_complete_
                && submission.event_recorded_) {
            try {
                Policy::activate(context_);
                Policy::synchronize_event(
                        event_slots_[submission.pool_index_].event);
                // Preserve a failure already observed by the worker callback;
                // a later covering proof makes reuse safe but cannot erase
                // the token's terminal failure.
                if (submission.result_.failure == nullptr) {
                    submission.result_ = FenceResult::success();
                }
                submission.synchronized_on_complete_ = true;
                if (submission.disposition_
                        == CompletionDisposition::RetireUnknown) {
                    if (unproved_unknowns_ != 0) {
                        --unproved_unknowns_;
                    }
                    event_slots_[submission.pool_index_].quarantined = false;
                }
                submission.disposition_ = CompletionDisposition::Complete;
            } catch (...) {
                const std::exception_ptr failure =
                        std::current_exception();
                set_retire_unknown_locked(submission);
                submission.result_ = FenceResult::failed(failure);
            }
        }
        // Shutdown may skip on_worker_complete; its covering proof above
        // must also capture the status before returning the slot to the pool.
        if (submission.disposition_ == CompletionDisposition::Complete) {
            snapshot_status_locked(submission);
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
        if (submission.disposition_ == CompletionDisposition::Complete) {
            // Completion callbacks run before the worker destroys its Task.
            // Return the event now so common FIFO pumping cannot wait on the
            // current Task's raw completion pointer.
            release_resource_locked(submission);
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
        for (std::size_t index = 0; index < event_count_; ++index) {
            if (event_slots_[index].quarantined) {
                event_slots_[index].quarantined = false;
                event_slots_[index].in_use = false;
                event_slots_[index].owner = nullptr;
                completion_.notify_one();
            }
        }
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
    [[nodiscard]] bool completion_proven(
            const Submission& submission) const noexcept {
        return submission.completion_proven();
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
    void snapshot_status_locked(Submission& submission) noexcept {
        if (submission.status_cell_ != nullptr) {
            submission.status_word_ = *submission.status_cell_;
            submission.status_cell_ = nullptr;
        }
    }

    void set_retire_unknown_locked(Submission& submission) noexcept {
        if (submission.disposition_ != CompletionDisposition::RetireUnknown) {
            submission.disposition_ = CompletionDisposition::RetireUnknown;
            if (submission.pool_index_ < event_count_) {
                event_slots_[submission.pool_index_].quarantined = true;
            }
            ++unproved_unknowns_;
        }
    }

    void return_resource(Submission& submission) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        release_resource_locked(submission);
    }

    void release_resource_locked(Submission& submission) noexcept {
        const std::size_t index = submission.pool_index_;
        if (index < event_count_ && event_slots_[index].in_use
                && event_slots_[index].owner == &submission
                && !event_slots_[index].quarantined) {
            event_slots_[index].in_use = false;
            event_slots_[index].owner = nullptr;
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

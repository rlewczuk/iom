#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "copy.hpp"

namespace iom::sycl_detail {

// Submission-fault consumption remains owned by copy.cpp while queue.cpp
// consumes the same one-shot state at queue construction and dispatch points.
[[nodiscard]] bool consume_submission_fault(
        SubmissionFault point) noexcept;
extern std::atomic<std::size_t> g_fence_wait_count;

class SyclCompletionPool final {
public:
    explicit SyclCompletionPool(std::size_t count)
            : slots_(std::make_unique<Slot[]>(count)), count_(count) {
        if (count_ == 0) {
            throw std::invalid_argument(
                    "SYCL completion pool requires at least one slot");
        }
    }

    SyclCompletionPool(const SyclCompletionPool&) = delete;
    SyclCompletionPool& operator=(const SyclCompletionPool&) = delete;

    [[nodiscard]] std::size_t count() const noexcept { return count_; }

    // Dispatch runs on the submitting thread or the completion worker, so
    // resource exhaustion must be reported without waiting for a slot.
    [[nodiscard]] std::optional<std::size_t> try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < count_; ++index) {
            if (!slots_[index].in_use) {
                slots_[index].in_use = true;
                return index;
            }
        }
        return std::nullopt;
    }

    void set_event(std::size_t index, sycl::event event) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < count_ && slots_[index].in_use) {
            slots_[index].event = std::move(event);
            slots_[index].has_event = true;
        }
    }

    void wait(std::size_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= count_ || !slots_[index].in_use
                || !slots_[index].has_event) {
            return;
        }
        slots_[index].event.wait_and_throw();
    }

    // Reviewer-one refined closure: the caller's result() must
    // distinguish three terminal states:
    //   (a) slot has no event bound yet — no-op, leave unproven and
    //       uncached so the later publisher binding is re-observed;
    //   (b) slot has an event bound and the wait succeeds — caller
    //       proves/caches the success;
    //   (c) slot has an event bound and the wait throws — caller
    //       folds FenceResult::failed(native), proves (native failure
    //       precedence), and caches the failure so repeated waits do
    //       not re-block.
    // `*observed` is set true at the moment the wait is about to be
    // called, BEFORE the throw can happen, so the caller's cache and
    // proof decisions can see it even when wait_and_throw throws.
    // `wait_observed` itself returns true iff the wait ran (cases
    // (b) and (c)) and false iff it was a no-op (case (a)). The
    // function is NOT noexcept: the throw on case (c) propagates to
    // the caller's catch, which converts it into FenceResult::failed
    // (round-5 reviewer-1 NEW HIGH) and must NOT terminate.
    [[nodiscard]] bool wait_observed(std::size_t index, bool& observed) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= count_ || !slots_[index].in_use
                || !slots_[index].has_event) {
            observed = false;
            return false;
        }
        observed = true;
        slots_[index].event.wait_and_throw();
        return true;
    }

    void protect(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < count_) {
            slots_[index].protected_ = true;
            slots_[index].in_use = true;
        }
    }

    void release_after_proof(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < count_ && slots_[index].in_use) {
            slots_[index].event = sycl::event{};
            slots_[index].has_event = false;
            slots_[index].protected_ = false;
            slots_[index].in_use = false;
        }
    }
    void release_all_protected() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < count_; ++index) {
            if (slots_[index].protected_) {
                slots_[index].event = sycl::event{};
                slots_[index].has_event = false;
                slots_[index].protected_ = false;
                slots_[index].in_use = false;
            }
        }
    }

    [[nodiscard]] bool has_unproven_leases() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < count_; ++index) {
            if (slots_[index].in_use) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::size_t in_use_count() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t result = 0;
        for (std::size_t index = 0; index < count_; ++index) {
            result += slots_[index].in_use ? 1u : 0u;
        }
        return result;
    }

    [[nodiscard]] std::size_t protected_count() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t result = 0;
        for (std::size_t index = 0; index < count_; ++index) {
            result += slots_[index].protected_ ? 1u : 0u;
        }
        return result;
    }

private:
    struct Slot {
        sycl::event event{};
        bool in_use = false;
        bool protected_ = false;
        bool has_event = false;
    };

    std::unique_ptr<Slot[]> slots_;
    std::size_t count_ = 0;
    std::mutex mutex_;
};

class SyclQueue;

class SyclFenceState final {
public:
    static_assert(
            std::is_nothrow_move_constructible_v<sycl::event>
            && std::is_nothrow_move_assignable_v<sycl::event>);

    ~SyclFenceState() noexcept {
        release_metadata_slot();
        release_completion_slot();
    }

    void set_event(sycl::event incoming) noexcept {
        // H1 closure (round-6, reviewer two): the previous shape
        // published `event_bound_` AFTER setting the pool event, so a
        // concurrent result() landing in the lag window saw
        // `wait_observed` true (real native wait just completed) while
        // `event_bound_` was still false — the completion action was
        // skipped and the OOV read was permanently masked. Publish
        // `event_bound_` FIRST under mu_ (so any subsequent
        // result() that observes `event_bound_ == true` is also
        // guaranteed to see the corresponding `has_event` after the
        // pool event set completes), then set the pool event under the
        // pool's own mutex. A reader observing `event_bound_ == true`
        // before the pool event set runs sees `has_event == false`
        // (premature window — wait_observed returns false and the slot
        // is left unproven/uncached, the later publisher binding is
        // re-observed). The null-pool branch does the same: mu_-guarded
        // `event_` and `event_bound_` are set under the SAME lock so no
        // observer can see a stored `event_` without the matching
        // `event_bound_`.
        bool needs_pool_publish = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            event_bound_ = true;
            if (completion_pool_ != nullptr) {
                needs_pool_publish = true;
            } else {
                event_ = std::move(incoming);
            }
        }
        if (needs_pool_publish) {
            // Pool event set under the pool's own mutex. Lock order
            // preserved: result() takes mu_ then pool.mutex_; here we
            // take pool.mutex_ only after releasing mu_.
            completion_pool_->set_event(
                    completion_slot_, std::move(incoming));
        }
    }

    void set_failure(std::exception_ptr incoming) noexcept {
        // F2 closure: a fence invoke racing result() can otherwise cache
        // a success result over a just-written retained post-launch
        // fault. The lock matches the read side under mu_ in result().
        std::lock_guard<std::mutex> lock(mu_);
        retained_failure_ = std::move(incoming);
    }

    void set_cleanup(std::function<void()> cleanup) {
        std::lock_guard<std::mutex> lock(mu_);
        cleanup_ = std::move(cleanup);
    }

    void set_completion_action(std::function<void()> action) {
        std::lock_guard<std::mutex> lock(mu_);
        completion_action_ = std::move(action);
    }

    void cleanup_now() noexcept {
        std::function<void()> cleanup;
        {
            std::lock_guard<std::mutex> lock(mu_);
            cleanup = std::move(cleanup_);
        }
        if (cleanup) {
            try {
                cleanup();
            } catch (...) {
            }
        }
    }

    void mark_completion_proven() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        proved_ = true;
    }

    [[nodiscard]] bool completion_proven() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return proved_;
    }

    [[nodiscard]] detail::FenceResult result() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        if (cached_.has_value()) {
            return *cached_;
        }

        // Stage 1: native completion wait. Capture failure but do not
        // yet touch completion action state. Reviewer-one refined F1
        // closure: a fence invoke landing between set_completion_slot
        // and set_event would see a pool-bound slot whose pool::wait
        // is a no-op return. Real submissions all take the pool branch
        // after slot acquisition — the no-pool path never executes in
        // production. We use `wait_observed` to distinguish a real
        // native completion from a no-op return: a no-op return means
        // the slot has no event bound yet and the result MUST stay
        // uncached and unproven so the subsequent publisher binding is
        // observed. The captured `native_event_bound` (under mu_) is
        // also reused so the completion action only runs when the
        // publisher has actually bound a real event.
        bool native_event_bound = event_bound_;
        detail::FenceResult result = detail::FenceResult::success();
        // `native_wait_observed` requires the wait returned without
        // throwing (used for the proof decision). The cache decision
        // uses `wait_observed` so a thrown wait is still a TERMINAL
        // observation that must be cached — the LOW contract — while
        // the false return from `wait_observed` (no event bound) leaves
        // the slot uncached.
        bool native_wait_observed = false;
        bool wait_observed = false;
        if (completion_pool_ != nullptr) {
            ++g_fence_wait_count;
            try {
                if (completion_pool_->wait_observed(
                            completion_slot_, wait_observed)) {
                    native_wait_observed = true;
                }
                // wait_observed returning false is a no-op (slot has no
                // event bound yet): the slot is left unproven and the
                // result is left uncached so a later publisher binding
                // is re-observed.
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        } else if (event_.has_value()) {
            ++g_fence_wait_count;
            try {
                event_->wait_and_throw();
                native_wait_observed = true;
                wait_observed = true;
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
                wait_observed = true;
            }
        }
        // H1 closure (round-6, reviewer two): an actually-observed
        // native wait is sufficient to run the deferred OOV read and
        // prove, even when `event_bound_` is not yet visible. The
        // completion action reads the host USM status cell that the
        // event-ordered queue::memcpy wrote immediately after the
        // gather kernel, so once the native wait completed that cell
        // holds the terminal observation. The premature-publication
        // window still leaves the slot unproven/uncached when
        // `wait_observed` returns false; once the wait has run, the
        // OOV read MUST run and the slot MUST prove/cache.
        const bool embedding_with_completion_action =
                static_cast<bool>(completion_action_)
                && (native_event_bound || wait_observed);
        // Stage 2: retained-failure fold. Unconditional: a post-launch
        // retained fault MUST surface regardless of the eventual
        // completion action. Native failure takes precedence.
        if (!result.failure && retained_failure_) {
            result = detail::FenceResult::failed(retained_failure_);
        }
        // Stage 3: completion action — embedding only, guarded by a
        // successful native wait. After a native failure the deferred
        // OOV read MUST NOT run: it would replace the native error with
        // a stale, possibly uninitialized status-cell value (no zeroing
        // in allocate_host_status; sticky 1 from a prior OOV call on the
        // same metadata slot is possible). Restoring the guard.
        if (!result.failure && embedding_with_completion_action) {
            std::function<void()> action = std::move(completion_action_);
            try {
                action();
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        }
        // Stage 4: cleanup. Falls outside the proof decision.
        if (cleanup_) {
            try {
                cleanup_();
            } catch (...) {
                if (!result.failure) {
                    result = detail::FenceResult::failed(
                            std::current_exception());
                }
            }
            cleanup_ = {};
        }
        // Stage 5: proof decision. Embedding proves unconditionally when
        // its completion action is bound — the OOV throw above only folds
        // into the cached failure, it does NOT clear proof. Copy /
        // binary / RMSNorm keep the strict semantic-success path so a
        // post-launch fault keeps the metadata slot protected.
        if (embedding_with_completion_action) {
            proved_ = true;
        } else if (native_wait_observed) {
            proved_ = proved_ || (result.succeeded && result.failure == nullptr);
        }
        // LOW closure: a real native completion (success OR exception) is
        // a TERMINAL observation. Cache the result so repeated waits do
        // not re-wait on a permanent native failure. The premature-fence
        // window (the pool path was attempted but no event was bound
        // yet, so wait_observed stays false) keeps cached_ unset and a
        // later publisher binding is re-observed.
        if (wait_observed || embedding_with_completion_action) {
            cached_ = result;
        }
        return result;
    }

    void clear_event() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        event_ = std::nullopt;
    }

    void release_metadata_slot() noexcept {
        detail::MetadataSlotPool* pool = nullptr;
        std::size_t slot = 0;
        bool proved = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            pool = pool_;
            slot = slot_;
            proved = proved_;
            pool_ = nullptr;
            slot_ = 0;
        }
        if (pool != nullptr) {
            if (proved) {
                pool->release_after_proof(slot);
            } else {
                // Unknown completion: the whole lease stays reserved; this
                // slot is never reassigned until a covering proof.
                pool->protect(slot);
            }
        }
    }

    void release_completion_slot() noexcept {
        SyclCompletionPool* pool = nullptr;
        std::size_t slot = 0;
        bool proved = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            pool = completion_pool_;
            slot = completion_slot_;
            proved = proved_;
            completion_pool_ = nullptr;
            completion_slot_ = 0;
        }
        if (pool != nullptr) {
            if (proved) {
                pool->release_after_proof(slot);
            } else {
                pool->protect(slot);
            }
        }
    }

private:
    void set_metadata_slot(
            detail::MetadataSlotPool& pool, std::size_t slot) noexcept {
        // F2 closure: serializes against the read in result() under mu_
        // and against the matching set_completion_slot call site.
        std::lock_guard<std::mutex> lock(mu_);
        pool_ = &pool;
        slot_ = slot;
    }

    void set_completion_slot(
            SyclCompletionPool& pool, std::size_t slot) noexcept {
        // F2 closure: serializes against the read in result() so a
        // fence invoke landing before this call returns does not see
        // a partially published pool pointer; result() always sees
        // either both pointer+slot set or both unset under mu_.
        std::lock_guard<std::mutex> lock(mu_);
        completion_pool_ = &pool;
        completion_slot_ = slot;
    }

    mutable std::mutex mu_;
    bool event_bound_ = false;
    bool proved_ = false;
    std::optional<sycl::event> event_;
    std::exception_ptr retained_failure_;
    std::optional<detail::FenceResult> cached_;
    std::function<void()> completion_action_;
    std::function<void()> cleanup_;
    std::size_t slot_ = 0;
    detail::MetadataSlotPool* pool_ = nullptr;
    std::size_t completion_slot_ = 0;
    SyclCompletionPool* completion_pool_ = nullptr;

    friend class SyclQueue;
};

// Fence construction and the no-op transfer fence stay implemented in
// copy.cpp, which remains the owner of SYCL fence support.
detail::Fence build_sycl_fence(
        const std::shared_ptr<SyclFenceState>& state) noexcept;
detail::Fence transfer_fence() noexcept;

}  // namespace iom::sycl_detail

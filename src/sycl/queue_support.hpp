#pragma once

#include <atomic>
#include <condition_variable>
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

    [[nodiscard]] std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (std::size_t index = 0; index < count_; ++index) {
                if (!slots_[index].in_use) {
                    return true;
                }
            }
            return false;
        });
        for (std::size_t index = 0; index < count_; ++index) {
            if (!slots_[index].in_use) {
                slots_[index].in_use = true;
                return index;
            }
        }
        throw std::logic_error(
                "SYCL completion pool lost a free slot");
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
            completion_.notify_one();
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
                completion_.notify_one();
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
    std::condition_variable completion_;
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
        if (completion_pool_ != nullptr) {
            completion_pool_->set_event(completion_slot_, std::move(incoming));
        } else {
            event_ = std::move(incoming);
        }
    }

    void set_failure(std::exception_ptr incoming) noexcept {
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

        detail::FenceResult result = detail::FenceResult::success();
        if (completion_pool_ != nullptr) {
            ++g_fence_wait_count;
            try {
                completion_pool_->wait(completion_slot_);
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        } else if (event_.has_value()) {
            ++g_fence_wait_count;
            try {
                event_->wait_and_throw();
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        }
        if (!result.failure && retained_failure_) {
            result = detail::FenceResult::failed(retained_failure_);
        }
        if (!result.failure && completion_action_) {
            try {
                completion_action_();
            } catch (...) {
                result = detail::FenceResult::failed(
                        std::current_exception());
            }
        }
        completion_action_ = {};
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
        // The fixed slot's host mirror stays immutable until this result is
        // known. An operation failure can be retained independently from the
        // completion proof established by the explicit queue drain.
        proved_ = proved_ || (result.succeeded && result.failure == nullptr);
        cached_ = result;
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
        pool_ = &pool;
        slot_ = slot;
    }

    void set_completion_slot(
            SyclCompletionPool& pool, std::size_t slot) noexcept {
        completion_pool_ = &pool;
        completion_slot_ = slot;
    }

    mutable std::mutex mu_;
    std::optional<sycl::event> event_;
    std::exception_ptr retained_failure_;
    std::optional<detail::FenceResult> cached_;
    std::function<void()> completion_action_;
    std::function<void()> cleanup_;
    bool proved_ = false;
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

#pragma once

#include <cstddef>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

namespace iom::detail {

template <typename Policy>
class StagingSlotPool final {
public:
    static constexpr std::size_t kMaxStagingBytes =
            8ull * 1024ull * 1024ull * 1024ull;
    static constexpr std::size_t kMaxSlotCount = 16;
    using device_pointer = typename Policy::device_pointer;

    class Lease final {
    public:
        ~Lease() noexcept {
            if (pool_ != nullptr) {
                pool_->release(index_, poisoned_);
            }
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;

        [[nodiscard]] device_pointer staging() const noexcept {
            return staging_;
        }
        [[nodiscard]] std::size_t capacity() const noexcept {
            return capacity_;
        }
        void poison() noexcept { poisoned_ = true; }

    private:
        Lease(
                StagingSlotPool& pool, std::size_t index,
                device_pointer staging, std::size_t capacity)
                : pool_(&pool), index_(index), staging_(staging),
                  capacity_(capacity) {}

        StagingSlotPool* pool_;
        std::size_t index_;
        device_pointer staging_;
        std::size_t capacity_;
        bool poisoned_ = false;

        friend class StagingSlotPool;
    };

    explicit StagingSlotPool(typename Policy::context_type context)
            : context_(context) {
        slots_.reserve(kMaxSlotCount);
        free_.reserve(kMaxSlotCount);
        vacant_.reserve(kMaxSlotCount);
    }
    ~StagingSlotPool() noexcept { destroy(); }

    StagingSlotPool(const StagingSlotPool&) = delete;
    StagingSlotPool& operator=(const StagingSlotPool&) = delete;

    [[nodiscard]] Lease acquire(std::size_t required_bytes) {
        if (required_bytes > kMaxStagingBytes) {
            throw std::invalid_argument(
                    "GPU staging request exceeds the 8 GiB staging limit");
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (closing_) {
            throw Policy::staging_pool_closing_error();
        }
        available_.wait(lock, [this] {
            return closing_ || !free_.empty() || !vacant_.empty()
                    || slots_.size() < kMaxSlotCount;
        });
        if (closing_) {
            throw Policy::staging_pool_closing_error();
        }

        if (free_.empty()) {
            if (!vacant_.empty()) {
                const std::size_t index = vacant_.back();
                vacant_.pop_back();
                const device_pointer staging = [&] {
                    try {
                        return allocate_locked(required_bytes);
                    } catch (...) {
                        vacant_.push_back(index);
                        throw;
                    }
                }();
                Slot& slot = slots_.at(index);
                slot = Slot{staging, required_bytes, true};
                ++active_count_;
                if (staging != device_pointer{}) {
                    ++allocation_count_;
                }
                return Lease{*this, index, staging, required_bytes};
            }

            const std::size_t index = slots_.size();
            const device_pointer staging = allocate_locked(required_bytes);
            try {
                slots_.push_back(Slot{staging, required_bytes, true});
            } catch (...) {
                Policy::staging_free_noexcept(staging);
                throw;
            }
            ++active_count_;
            if (staging != device_pointer{}) {
                ++allocation_count_;
            }
            return Lease{*this, index, staging, required_bytes};
        }

        const std::size_t index = free_.back();
        free_.pop_back();
        Slot& slot = slots_.at(index);
        if (slot.capacity < required_bytes) {
            const std::size_t replacement_capacity =
                    grown_capacity(slot.capacity, required_bytes);
            device_pointer replacement{};
            try {
                replacement = allocate_locked(replacement_capacity);
            } catch (...) {
                free_.push_back(index);
                throw;
            }
            const device_pointer old_staging = slot.staging;
            slot.staging = replacement;
            slot.capacity = replacement_capacity;
            if (old_staging != device_pointer{}) {
                Policy::staging_free_noexcept(old_staging);
            } else if (replacement != device_pointer{}) {
                ++allocation_count_;
            }
        }
        slot.active = true;
        ++active_count_;
        return Lease{*this, index, slot.staging, slot.capacity};
    }

    void destroy() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        closing_ = true;
        available_.wait(lock, [this] { return active_count_ == 0; });
        for (Slot& slot : slots_) {
            if (slot.staging != device_pointer{}) {
                Policy::staging_free_noexcept(slot.staging);
                slot.staging = device_pointer{};
            }
            slot.capacity = 0;
            slot.active = false;
        }
        slots_.clear();
        free_.clear();
        vacant_.clear();
        allocation_count_ = 0;
    }

    [[nodiscard]] std::size_t idle_count_for_testing() const noexcept {
#ifdef IOM_ENABLE_TESTING
        std::lock_guard<std::mutex> lock(mutex_);
        return free_.size();
#else
        return 0;
#endif
    }

    [[nodiscard]] std::size_t allocation_count_for_testing() const noexcept {
#ifdef IOM_ENABLE_TESTING
        std::lock_guard<std::mutex> lock(mutex_);
        return allocation_count_;
#else
        return 0;
#endif
    }

    void fail_next_allocation_for_testing() noexcept {
#ifdef IOM_ENABLE_TESTING
        std::lock_guard<std::mutex> lock(mutex_);
        fail_next_allocation_ = true;
#else
        (void)this;
#endif
    }

private:
    struct Slot {
        device_pointer staging{};
        std::size_t capacity = 0;
        bool active = false;
    };

    [[nodiscard]] device_pointer allocate_locked(std::size_t bytes) {
        if (bytes == 0) {
            return device_pointer{};
        }
#ifdef IOM_ENABLE_TESTING
        if (fail_next_allocation_) {
            fail_next_allocation_ = false;
            throw std::bad_alloc();
        }
#else
        (void)fail_next_allocation_;
#endif
        return Policy::staging_allocate(bytes);
    }

    [[nodiscard]] std::size_t grown_capacity(
            std::size_t current, std::size_t required) const noexcept {
        std::size_t capacity = current == 0 ? required : current;
        while (capacity < required) {
            if (capacity > kMaxStagingBytes / 2) {
                return required;
            }
            capacity *= 2;
        }
        return capacity;
    }

    void release(std::size_t index, bool poisoned) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= slots_.size() || !slots_[index].active) {
            return;
        }
        Slot& slot = slots_[index];
        slot.active = false;
        --active_count_;
        if (poisoned) {
            if (slot.staging != device_pointer{}) {
                Policy::staging_free_noexcept(slot.staging);
                slot.staging = device_pointer{};
                if (allocation_count_ != 0) {
                    --allocation_count_;
                }
            }
            slot.capacity = 0;
            vacant_.push_back(index);
        } else {
            free_.push_back(index);
        }
        available_.notify_all();
    }

    typename Policy::context_type context_;
    std::vector<Slot> slots_;
    std::vector<std::size_t> free_;
    std::vector<std::size_t> vacant_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::size_t active_count_ = 0;
    std::size_t allocation_count_ = 0;
    bool closing_ = false;
    bool fail_next_allocation_ = false;

    friend class Lease;
};

}  // namespace iom::detail

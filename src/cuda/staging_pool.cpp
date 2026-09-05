#include "staging_pool.hpp"

#include <limits>
#include <new>
#include <stdexcept>

#include "driver.hpp"

namespace iom::cuda_detail {

StagingSlotPool::Lease::~Lease() noexcept {
    if (pool_ != nullptr) {
        pool_->release(index_, poisoned_);
    }
}

StagingSlotPool::~StagingSlotPool() noexcept {
    destroy();
}

CUdeviceptr StagingSlotPool::allocate_locked(std::size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
#ifdef IOM_ENABLE_TESTING
    if (fail_next_allocation_) {
        fail_next_allocation_ = false;
        throw std::bad_alloc();
    }
#else
    (void)fail_next_allocation_;
#endif
    CUdeviceptr staging = 0;
    check_cuda("cuMemAlloc", cuMemAlloc(&staging, bytes));
    return staging;
}

std::size_t StagingSlotPool::grown_capacity(
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

StagingSlotPool::Lease StagingSlotPool::acquire(std::size_t required_bytes) {
    if (required_bytes > kMaxStagingBytes) {
        throw std::invalid_argument(
                "GPU staging request exceeds the 8 GiB staging limit");
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (closing_) {
        throw std::runtime_error("CUDA staging pool is closing");
    }
    available_.wait(lock, [this] {
        return closing_ || !free_.empty() || slots_.size() < kMaxSlotCount;
    });
    if (closing_) {
        throw std::runtime_error("CUDA staging pool is closing");
    }

    if (free_.empty()) {
        const std::size_t index = slots_.size();
        const CUdeviceptr staging = allocate_locked(required_bytes);
        try {
            slots_.push_back(Slot{staging, required_bytes, true});
        } catch (...) {
            if (staging != 0) {
                (void)cuMemFree(staging);
            }
            throw;
        }
        ++active_count_;
        if (staging != 0) {
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
        CUdeviceptr replacement = 0;
        try {
            replacement = allocate_locked(replacement_capacity);
        } catch (...) {
            free_.push_back(index);
            throw;
        }
        const CUdeviceptr old_staging = slot.staging;
        slot.staging = replacement;
        slot.capacity = replacement_capacity;
        if (old_staging != 0) {
            (void)cuMemFree(old_staging);
        } else if (replacement != 0) {
            ++allocation_count_;
        }
    }
    slot.active = true;
    ++active_count_;
    return Lease{*this, index, slot.staging, slot.capacity};
}

void StagingSlotPool::release(std::size_t index, bool poisoned) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= slots_.size() || !slots_[index].active) {
        return;
    }
    Slot& slot = slots_[index];
    slot.active = false;
    --active_count_;
    if (poisoned) {
        if (slot.staging != 0) {
            (void)cuMemFree(slot.staging);
            slot.staging = 0;
            if (allocation_count_ != 0) {
                --allocation_count_;
            }
        }
        slot.capacity = 0;
    }
    free_.push_back(index);
    available_.notify_all();
}

void StagingSlotPool::destroy() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    available_.wait(lock, [this] { return active_count_ == 0; });
    for (Slot& slot : slots_) {
        if (slot.staging != 0) {
            (void)cuMemFree(slot.staging);
            slot.staging = 0;
        }
        slot.capacity = 0;
        slot.active = false;
    }
    free_.clear();
    allocation_count_ = 0;
}

std::size_t StagingSlotPool::idle_count_for_testing() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_.size();
}

std::size_t StagingSlotPool::allocation_count_for_testing() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_count_;
}

void StagingSlotPool::fail_next_allocation_for_testing() noexcept {
#ifdef IOM_ENABLE_TESTING
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_allocation_ = true;
#else
    (void)this;
#endif
}

}  // namespace iom::cuda_detail

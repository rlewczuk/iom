#include "staging_pool.hpp"

#include <new>
#include <stdexcept>
#include <string>

namespace iom::rocm_detail {
namespace {

[[nodiscard]] std::runtime_error hip_failure(
        const char* operation, hipError_t status) {
    return std::runtime_error(
            std::string(operation) + " failed with "
            + hipGetErrorName(status) + ": " + hipGetErrorString(status));
}

void check_hip(const char* operation, hipError_t status) {
    if (status != hipSuccess) {
        throw hip_failure(operation, status);
    }
}

}  // namespace

StagingSlotPool::Lease::~Lease() noexcept {
    if (pool_ != nullptr) {
        pool_->release(index_, poisoned_);
    }
}

StagingSlotPool::~StagingSlotPool() noexcept {
    destroy();
}

hipDeviceptr_t StagingSlotPool::allocate_locked(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
#ifdef IOM_ENABLE_TESTING
    if (fail_next_allocation_) {
        fail_next_allocation_ = false;
        throw std::bad_alloc();
    }
#else
    (void)fail_next_allocation_;
#endif
    void* staging = nullptr;
    check_hip("hipMalloc", hipMalloc(&staging, bytes));
    return static_cast<hipDeviceptr_t>(staging);
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
        throw std::runtime_error("ROCm staging pool is closing");
    }
    available_.wait(lock, [this] {
        return closing_ || !free_.empty() || slots_.size() < kMaxSlotCount;
    });
    if (closing_) {
        throw std::runtime_error("ROCm staging pool is closing");
    }

    if (free_.empty()) {
        const std::size_t index = slots_.size();
        const hipDeviceptr_t staging = allocate_locked(required_bytes);
        try {
            slots_.push_back(Slot{staging, required_bytes, true});
        } catch (...) {
            if (staging != nullptr) {
                (void)hipFree(staging);
            }
            throw;
        }
        ++active_count_;
        if (staging != nullptr) {
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
        hipDeviceptr_t replacement = nullptr;
        try {
            replacement = allocate_locked(replacement_capacity);
        } catch (...) {
            free_.push_back(index);
            throw;
        }
        const hipDeviceptr_t old_staging = slot.staging;
        slot.staging = replacement;
        slot.capacity = replacement_capacity;
        if (old_staging != nullptr) {
            (void)hipFree(old_staging);
        } else if (replacement != nullptr) {
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
        if (slot.staging != nullptr) {
            (void)hipFree(slot.staging);
            slot.staging = nullptr;
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
        if (slot.staging != nullptr) {
            (void)hipFree(slot.staging);
            slot.staging = nullptr;
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

}  // namespace iom::rocm_detail

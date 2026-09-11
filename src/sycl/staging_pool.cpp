#include "staging_pool.hpp"

#include <new>
#include <stdexcept>

#include "runtime.hpp"

namespace iom::sycl_detail {

StagingSlotPool::Lease::~Lease() noexcept {
    if (pool_ != nullptr) {
        pool_->release(index_, poisoned_);
    }
}

StagingSlotPool::~StagingSlotPool() noexcept {
    destroy();
}

StagingSlotPool::Slot StagingSlotPool::allocate_locked(std::size_t bytes) {
    if (bytes == 0) {
        return {};
    }
#ifdef IOM_ENABLE_TESTING
    if (fail_next_allocation_) {
        fail_next_allocation_ = false;
        throw std::bad_alloc();
    }
#else
    (void)fail_next_allocation_;
#endif

    void* device_staging = alloc_attempt_device(
            bytes, device_, context_, AllocationClass::staging,
            AllocationPhase::post_publication);
    if (device_staging == nullptr) {
        throw std::bad_alloc();
    }
    void* host_mirror = nullptr;
    try {
        host_mirror = sycl::malloc_host(bytes, context_);
        if (host_mirror == nullptr) {
            throw std::bad_alloc();
        }
    } catch (...) {
        // Rollback of the successful device allocation stays observable as
        // a staging free paired with the failed attempt's completion.
        free_attempt_device(
                device_staging, context_, AllocationClass::staging,
                AllocationPhase::post_publication);
        throw;
    }
    return Slot{device_staging, host_mirror, bytes, true};
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

void StagingSlotPool::free_slot_noexcept(Slot& slot) noexcept {
    if (slot.device_staging != nullptr) {
        free_attempt_device(
                slot.device_staging, context_, AllocationClass::staging,
                AllocationPhase::post_publication);
        slot.device_staging = nullptr;
    }
    if (slot.host_mirror != nullptr) {
        try {
            sycl::free(slot.host_mirror, context_);
        } catch (...) {
        }
        slot.host_mirror = nullptr;
    }
    slot.capacity = 0;
    slot.active = false;
}

StagingSlotPool::Lease StagingSlotPool::acquire(
        std::size_t required_bytes) {
    if (required_bytes > kMaxStagingBytes) {
        throw std::invalid_argument(
                "GPU staging request exceeds the 8 GiB staging limit");
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (closing_) {
        throw std::runtime_error("SYCL staging pool is closing");
    }
    available_.wait(lock, [this] {
        return closing_ || !free_.empty() || !vacant_.empty()
                || slots_.size() < kMaxSlotCount;
    });
    if (closing_) {
        throw std::runtime_error("SYCL staging pool is closing");
    }

    if (free_.empty()) {
        if (!vacant_.empty()) {
            const std::size_t index = vacant_.back();
            vacant_.pop_back();
            Slot slot;
            try {
                slot = allocate_locked(required_bytes);
            } catch (...) {
                vacant_.push_back(index);
                throw;
            }
            slots_.at(index) = slot;
            ++active_count_;
            if (slot.device_staging != nullptr) {
                ++allocation_count_;
            }
            return Lease{
                    *this, index, slot.device_staging, slot.host_mirror,
                    slot.capacity};
        }

        const std::size_t index = slots_.size();
        Slot slot = allocate_locked(required_bytes);
        try {
            slots_.push_back(slot);
        } catch (...) {
            free_slot_noexcept(slot);
            throw;
        }
        ++active_count_;
        if (slot.device_staging != nullptr) {
            ++allocation_count_;
        }
        return Lease{
                *this, index, slot.device_staging, slot.host_mirror,
                slot.capacity};
    }

    const std::size_t index = free_.back();
    free_.pop_back();
    Slot& slot = slots_.at(index);
    if (slot.capacity < required_bytes) {
        const std::size_t replacement_capacity =
                grown_capacity(slot.capacity, required_bytes);
        Slot replacement;
        try {
            replacement = allocate_locked(replacement_capacity);
        } catch (...) {
            free_.push_back(index);
            throw;
        }
        Slot old = slot;
        slot = replacement;
        free_slot_noexcept(old);
    }
    slot.active = true;
    ++active_count_;
    return Lease{
            *this, index, slot.device_staging, slot.host_mirror,
            slot.capacity};
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
        free_slot_noexcept(slot);
        if (allocation_count_ != 0) {
            --allocation_count_;
        }
        vacant_.push_back(index);
    } else {
        free_.push_back(index);
    }
    available_.notify_all();
}

void StagingSlotPool::destroy() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    available_.wait(lock, [this] { return active_count_ == 0; });
    for (Slot& slot : slots_) {
        free_slot_noexcept(slot);
    }
    slots_.clear();
    free_.clear();
    vacant_.clear();
    allocation_count_ = 0;
}

std::size_t StagingSlotPool::idle_count_for_testing() const noexcept {
#ifdef IOM_ENABLE_TESTING
    std::lock_guard<std::mutex> lock(mutex_);
    return free_.size();
#else
    return 0;
#endif
}

std::size_t StagingSlotPool::allocation_count_for_testing() const noexcept {
#ifdef IOM_ENABLE_TESTING
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_count_;
#else
    return 0;
#endif
}

void StagingSlotPool::fail_next_allocation_for_testing() noexcept {
#ifdef IOM_ENABLE_TESTING
    std::lock_guard<std::mutex> lock(mutex_);
    fail_next_allocation_ = true;
#else
    (void)this;
#endif
}

}  // namespace iom::sycl_detail

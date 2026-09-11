#pragma once

// Fixed per-queue metadata slot bookkeeping over one reserved partition of
// the Device-wide metadata arena. The pool owns no native storage at all:
// device slots address the partition's fixed 512-byte blocks inside the
// Device metadata backing, and host mirrors are the lease's fixed C
// 512-byte buffers. Nothing here allocates, frees, resizes, or replaces
// storage after construction; only in-use/protected bookkeeping changes.
// A protected slot (unknown completion) is never reassigned and is released
// only by a covering proof.

#include <array>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "queue_resources.hpp"

namespace iom::detail {

class MetadataSlotPool final {
public:
    // Takes ownership of one queue's reserved partition and its fixed host
    // mirrors. Construction performs no native work.
    explicit MetadataSlotPool(QueueResourceLease resources)
            : resources_(std::move(resources)),
              slot_count_(resources_.slot_count()),
              in_use_(std::make_unique<bool[]>(slot_count_)),
              protected_(std::make_unique<bool[]>(slot_count_)) {
        if (slot_count_ == 0) {
            throw std::invalid_argument(
                    "metadata slot pool requires a reserved partition");
        }
        for (std::size_t index = 0; index < slot_count_; ++index) {
            in_use_[index] = false;
            protected_[index] = false;
        }
    }

    MetadataSlotPool(const MetadataSlotPool&) = delete;
    MetadataSlotPool& operator=(const MetadataSlotPool&) = delete;

    ~MetadataSlotPool() noexcept = default;

    [[nodiscard]] std::size_t slot_count() const noexcept {
        return slot_count_;
    }

    [[nodiscard]] std::size_t slot_stride() const noexcept {
        return kMetadataSlotBytes;
    }

    // Device address of the partition's first slot inside the Device
    // metadata backing; slot i lives at base + i * 512 with 32-byte
    // alignment.
    [[nodiscard]] void* device_base() const noexcept {
        return resources_.device_base();
    }

    // Base of the fixed host mirrors backing every slot.
    [[nodiscard]] std::byte* host_mirrors() const noexcept {
        return resources_.host_mirrors();
    }

    [[nodiscard]] std::size_t in_use_count() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (std::size_t index = 0; index < slot_count_; ++index) {
            count += in_use_[index] ? 1u : 0u;
        }
        return count;
    }

    [[nodiscard]] std::size_t protected_count() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (std::size_t index = 0; index < slot_count_; ++index) {
            count += protected_[index] ? 1u : 0u;
        }
        return count;
    }

    // True when any slot is reserved by live work or retained by an
    // unproven completion; such a partition must never be released.
    [[nodiscard]] bool has_unproven_leases() noexcept {
        return in_use_count() != 0;
    }

    // Wait for a free fixed slot of this queue's partition. Exactly C slots
    // exist; acquisition never grows, borrows, or replaces storage.
    [[nodiscard]] std::size_t acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_.wait(lock, [this] {
            for (std::size_t index = 0; index < slot_count_; ++index) {
                if (!in_use_[index]) {
                    return true;
                }
            }
            return false;
        });
        for (std::size_t index = 0; index < slot_count_; ++index) {
            if (!in_use_[index]) {
                in_use_[index] = true;
                return index;
            }
        }
        throw std::logic_error("metadata slot acquisition lost a free slot");
    }

    // Retain a slot whose completion could not be proven: it stays reserved
    // forever until a covering proof releases it. Protected slots are never
    // handed out again.
    void protect(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < slot_count_) {
            protected_[index] = true;
            in_use_[index] = true;
        }
    }

    void release_after_proof(std::size_t index) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        release_locked(index);
    }

    void release(std::size_t index) noexcept {
        release_after_proof(index);
    }

    // Covering proof: every protected slot's asynchronous use has ended.
    void release_all_protected() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t index = 0; index < slot_count_; ++index) {
            if (protected_[index]) {
                release_locked(index);
            }
        }
    }

    // Out-of-range slot indices are programming errors; these accessors
    // report them with std::out_of_range rather than noexcept termination.
    [[nodiscard]] std::byte* host_data(std::size_t index) {
        if (index >= slot_count_) {
            throw std::out_of_range("metadata slot index is out of range");
        }
        return resources_.host_data(index);
    }

    [[nodiscard]] void* device_data(std::size_t index) {
        if (index >= slot_count_) {
            throw std::out_of_range("metadata slot index is out of range");
        }
        return resources_.device_data(index);
    }

private:
    void release_locked(std::size_t index) noexcept {
        if (index < slot_count_ && in_use_[index]) {
            protected_[index] = false;
            in_use_[index] = false;
            completion_.notify_one();
        }
    }

    QueueResourceLease resources_;
    std::size_t slot_count_ = 0;
    std::unique_ptr<bool[]> in_use_;
    std::unique_ptr<bool[]> protected_;
    std::mutex mutex_;
    std::condition_variable completion_;
};

}  // namespace iom::detail

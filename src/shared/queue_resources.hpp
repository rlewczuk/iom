#pragma once

// Fixed GPU queue-resource geometry shared by the CUDA, ROCm, and SYCL
// backends. Each backend Device owns exactly one device-wide metadata
// FixedSizeAllocator spanning 4 * C blocks of 512 bytes (alignment 32) and
// leases one disjoint C-block partition plus one queue-count credit to each
// live DeviceOps queue. The lease is RAII: destruction returns the partition
// and the credit, so a safely drained queue's geometry is reusable. The
// reservation happens at the owning Device boundary before any native
// stream, worker, or completion-resource creation, and a fifth live queue
// reports std::bad_alloc from that first step.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "iom/alloc.hpp"

namespace iom::detail {

inline constexpr std::size_t kMaxLiveGpuQueues = 4;

// Fixed metadata slot payload/stride shared with gpu_arena_config.hpp's
// checked 4 * C * 512 backing sizing. Slots are 32-byte aligned inside the
// metadata arena backing.
inline constexpr std::size_t kMetadataSlotBytes = 512;

struct QueueResourceReservation final {
    std::size_t first_slot;
    std::unique_ptr<std::byte[]> host_mirrors;
};

// Reserve one aligned, contiguous C-slot partition and its fixed host mirrors.
// The caller owns allocator synchronization and the returned slots.
[[nodiscard]] inline QueueResourceReservation reserve_queue_partition(
        FixedSizeAllocator& allocator, std::size_t slot_count) {
    std::vector<std::size_t> indices;
    indices.reserve(slot_count);
    try {
        for (std::size_t index = 0; index < slot_count; ++index) {
            void* block = allocator.alloc(kMetadataSlotBytes);
            indices.push_back(allocator.index_of(block));
        }
    } catch (...) {
        for (const std::size_t index : indices) {
            allocator.free(allocator.ptr_from_index(index));
        }
        throw;
    }

    std::sort(indices.begin(), indices.end());
    const std::size_t base = indices.front();
    for (std::size_t index = 0; index < slot_count; ++index) {
        if (indices[index] != base + index || base % slot_count != 0) {
            for (const std::size_t reserved : indices) {
                allocator.free(allocator.ptr_from_index(reserved));
            }
            throw std::logic_error(
                    "metadata partition reservation lost the device's C-slot "
                    "geometry");
        }
    }

    std::unique_ptr<std::byte[]> host_mirrors;
    try {
        host_mirrors = std::make_unique<std::byte[]>(
                slot_count * kMetadataSlotBytes);
    } catch (...) {
        for (const std::size_t index : indices) {
            allocator.free(allocator.ptr_from_index(index));
        }
        throw;
    }
    return {base, std::move(host_mirrors)};
}

class QueueResourceProvider;


// One queue's fixed resource lease: the queue-count credit, one disjoint
// C-slot partition of the Device-wide metadata FixedSizeAllocator, the
// queue's exactly C fixed host metadata mirrors, and optional C host status
// cells for accelerator embedding. No lease field ever changes after
// reservation; reuse happens only through release and a later reservation of
// the same geometry.
class QueueResourceLease final {
public:
    using StatusCellsDeleter = void (*)(void*) noexcept;

    QueueResourceLease() noexcept = default;

    QueueResourceLease(QueueResourceLease&& other) noexcept {
        *this = std::move(other);
    }

    QueueResourceLease& operator=(QueueResourceLease&& other) noexcept {
        if (this != &other) {
            impl_release();
            provider_ = std::exchange(other.provider_, nullptr);
            first_slot_ = std::exchange(other.first_slot_, 0);
            slot_count_ = std::exchange(other.slot_count_, 0);
            device_base_ = std::exchange(other.device_base_, nullptr);
            host_mirrors_ = std::move(other.host_mirrors_);
            status_cells_ = std::exchange(other.status_cells_, nullptr);
            status_cells_deleter_ = std::exchange(
                    other.status_cells_deleter_, nullptr);
        }
        return *this;
    }

    QueueResourceLease(const QueueResourceLease&) = delete;
    QueueResourceLease& operator=(const QueueResourceLease&) = delete;

    ~QueueResourceLease() { impl_release(); }

    explicit operator bool() const noexcept { return provider_ != nullptr; }

    // Number of slots in this partition; exactly the Device's immutable C.
    [[nodiscard]] std::size_t slot_count() const noexcept {
        return slot_count_;
    }

    // Device address of the partition's first 512-byte slot. The address
    // lies inside the Device metadata backing and every partition-local
    // slot keeps the allocator's 32-byte alignment.
    [[nodiscard]] void* device_base() const noexcept {
        return device_base_;
    }

    // Base of the queue's fixed host mirrors: slot_count * 512 bytes, one
    // immutable mirror per slot, owned by this lease for the queue's whole
    // lifetime.
    [[nodiscard]] std::byte* host_mirrors() const noexcept {
        return host_mirrors_.get();
    }

    [[nodiscard]] std::byte* host_data(std::size_t slot) const noexcept {
        return host_mirrors_.get() + slot * kMetadataSlotBytes;
    }

    [[nodiscard]] void* device_data(std::size_t slot) const noexcept {
        return static_cast<std::byte*>(device_base_)
                + slot * kMetadataSlotBytes;
    }

    // Queue-owned status cells used by accelerator embedding operations.
    // Providers may leave this optional backing null while the operation is
    // disabled for that backend.
    [[nodiscard]] std::uint32_t* status_cells() const noexcept {
        return static_cast<std::uint32_t*>(status_cells_);
    }
    [[nodiscard]] std::uint32_t* status_data(std::size_t slot) const noexcept {
        return status_cells() == nullptr ? nullptr : status_cells() + slot;
    }

private:
    friend class QueueResourceProvider;

    void impl_release() noexcept;

    QueueResourceProvider* provider_ = nullptr;
    // Index of the partition's first block inside the Device-wide
    // FixedSizeAllocator; a whole partition is always one contiguous
    // C-block run, so release frees exactly this range.
    std::size_t first_slot_ = 0;
    std::size_t slot_count_ = 0;
    void* device_base_ = nullptr;
    std::unique_ptr<std::byte[]> host_mirrors_;
    void* status_cells_ = nullptr;
    StatusCellsDeleter status_cells_deleter_ = nullptr;
};


// Device-boundary owner of the fixed queue-resource geometry. Concrete
// standard-GPU Devices implement it; queue construction reserves through it
// and never touches the allocator, backings, or reservation table directly.
class QueueResourceProvider {
public:
    virtual ~QueueResourceProvider() = default;

    QueueResourceProvider(const QueueResourceProvider&) = delete;
    QueueResourceProvider& operator=(const QueueResourceProvider&) = delete;

    // Immutable per-Device C selected at construction; every reservation on
    // this Device carries exactly this many slots.
    [[nodiscard]] virtual std::size_t queue_slot_count() const noexcept = 0;

    // Atomically claim one queue-count credit and one disjoint C-slot
    // partition from the Device-wide FixedSizeAllocator. Throws
    // std::bad_alloc when every credit is live (fifth queue) and rolls back
    // completely when any later step of the reservation fails. The Device's
    // allocator bookkeeping lock is held only across host bookkeeping.
    [[nodiscard]] virtual QueueResourceLease reserve_queue_resources() = 0;

    // Return one partition's C blocks to the Device-wide allocator and free
    // its queue-count credit. Noexcept host bookkeeping only.
    virtual void release_queue_resources(
            std::size_t first_slot, std::size_t slot_count) noexcept = 0;

    // Quarantine an entire unresolved queue lease at the Device boundary so
    // queue destruction cannot drop its final lifetime protection: the
    // partition, the queue-count credit, the host mirrors and optional status
    // cells, the completion resources, and the queue's own covering-proof
    // handle stay retained until `reclaim` proves the retained queue safe
    // (a drain of another queue is never sufficient). `reclaim` returns true
    // when it proved completion and released everything; `discard` is a
    // best-effort teardown used when the Device itself is being destroyed.
    virtual void retain_unknown_lease(
            std::function<bool()> reclaim, std::function<void()> discard) = 0;

    // Attempt a covering proof for every retained lease. Reclaimed bundles
    // return their partition and credit; unproven bundles stay retained
    // with their original token outcomes untouched.
    virtual void reclaim_retained_leases() = 0;

protected:
    QueueResourceProvider() = default;

    [[nodiscard]] static QueueResourceLease make_lease(
            QueueResourceProvider& provider, std::size_t first_slot,
            std::size_t slot_count, void* device_base,
            std::unique_ptr<std::byte[]> host_mirrors,
            void* status_cells = nullptr,
            QueueResourceLease::StatusCellsDeleter status_cells_deleter =
                    nullptr) noexcept {
        QueueResourceLease lease;
        lease.provider_ = &provider;
        lease.first_slot_ = first_slot;
        lease.slot_count_ = slot_count;
        lease.device_base_ = device_base;
        lease.host_mirrors_ = std::move(host_mirrors);
        lease.status_cells_ = status_cells;
        lease.status_cells_deleter_ = status_cells_deleter;
        return lease;
    }
};

// Defined after QueueResourceProvider is complete: release routes through
// the owning provider's virtual.
inline void QueueResourceLease::impl_release() noexcept {
    if (status_cells_ != nullptr && status_cells_deleter_ != nullptr) {
        status_cells_deleter_(status_cells_);
    }
    status_cells_ = nullptr;
    status_cells_deleter_ = nullptr;
    if (provider_ != nullptr) {
        QueueResourceProvider* owner = std::exchange(provider_, nullptr);
        owner->release_queue_resources(first_slot_, slot_count_);
    }
    first_slot_ = 0;
    slot_count_ = 0;
    device_base_ = nullptr;
    host_mirrors_.reset();
}

}  // namespace iom::detail

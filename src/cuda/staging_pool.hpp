#pragma once

#include <cuda.h>

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace iom::cuda_detail {

class StagingSlotPool final {
public:
    static constexpr std::size_t kMaxStagingBytes = 8ull * 1024ull * 1024ull * 1024ull;
    static constexpr std::size_t kMaxSlotCount = 16;

    class Lease final {
    public:
        ~Lease() noexcept;

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;

        [[nodiscard]] CUdeviceptr staging() const noexcept { return staging_; }
        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
        void poison() noexcept { poisoned_ = true; }

    private:
        Lease(
                StagingSlotPool& pool, std::size_t index,
                CUdeviceptr staging, std::size_t capacity)
                : pool_(&pool), index_(index), staging_(staging),
                  capacity_(capacity) {}

        StagingSlotPool* pool_;
        std::size_t index_;
        CUdeviceptr staging_;
        std::size_t capacity_;
        bool poisoned_ = false;

        friend class StagingSlotPool;
    };

    explicit StagingSlotPool(CUcontext context) : context_(context) {}
    ~StagingSlotPool() noexcept;

    StagingSlotPool(const StagingSlotPool&) = delete;
    StagingSlotPool& operator=(const StagingSlotPool&) = delete;

    [[nodiscard]] Lease acquire(std::size_t required_bytes);
    void destroy() noexcept;

    [[nodiscard]] std::size_t idle_count_for_testing() const noexcept;
    [[nodiscard]] std::size_t allocation_count_for_testing() const noexcept;
    void fail_next_allocation_for_testing() noexcept;

private:
    struct Slot {
        CUdeviceptr staging = 0;
        std::size_t capacity = 0;
        bool active = false;
    };

    [[nodiscard]] CUdeviceptr allocate_locked(std::size_t bytes);
    [[nodiscard]] std::size_t grown_capacity(
            std::size_t current, std::size_t required) const noexcept;
    void release(std::size_t index, bool poisoned) noexcept;

    CUcontext context_;
    std::vector<Slot> slots_;
    std::vector<std::size_t> free_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::size_t active_count_ = 0;
    std::size_t allocation_count_ = 0;
    bool closing_ = false;
    bool fail_next_allocation_ = false;

    friend class Lease;
};

}  // namespace iom::cuda_detail

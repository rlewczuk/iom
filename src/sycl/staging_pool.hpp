#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace iom::sycl_detail {

class StagingSlotPool final {
public:
    static constexpr std::size_t kMaxStagingBytes =
            8ull * 1024ull * 1024ull * 1024ull;
    static constexpr std::size_t kMaxSlotCount = 16;

    class Lease final {
    public:
        ~Lease() noexcept;

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&&) = delete;
        Lease& operator=(Lease&&) = delete;

        [[nodiscard]] void* device_staging() const noexcept {
            return device_staging_;
        }
        [[nodiscard]] void* staging() const noexcept {
            return device_staging_;
        }
        [[nodiscard]] void* host_mirror() const noexcept {
            return host_mirror_;
        }
        [[nodiscard]] std::size_t capacity() const noexcept {
            return capacity_;
        }
        void poison() noexcept { poisoned_ = true; }

    private:
        Lease(
                StagingSlotPool& pool, std::size_t index,
                void* device_staging, void* host_mirror,
                std::size_t capacity)
                : pool_(&pool), index_(index),
                  device_staging_(device_staging), host_mirror_(host_mirror),
                  capacity_(capacity) {}

        StagingSlotPool* pool_;
        std::size_t index_;
        void* device_staging_;
        void* host_mirror_;
        std::size_t capacity_;
        bool poisoned_ = false;

        friend class StagingSlotPool;
    };

    StagingSlotPool(const sycl::context& context, const sycl::device& device)
            : context_(context), device_(device) {
        slots_.reserve(kMaxSlotCount);
        free_.reserve(kMaxSlotCount);
        vacant_.reserve(kMaxSlotCount);
    }
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
        void* device_staging = nullptr;
        void* host_mirror = nullptr;
        std::size_t capacity = 0;
        bool active = false;
    };

    [[nodiscard]] Slot allocate_locked(std::size_t bytes);
    [[nodiscard]] std::size_t grown_capacity(
            std::size_t current, std::size_t required) const noexcept;
    void free_slot_noexcept(Slot& slot) noexcept;
    void release(std::size_t index, bool poisoned) noexcept;

    sycl::context context_;
    sycl::device device_;
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

}  // namespace iom::sycl_detail

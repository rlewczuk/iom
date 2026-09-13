#include "registry_state.hpp"
#include "staging.hpp"
#include "testing_internal.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <stdexcept>

#ifdef IOM_ENABLE_TESTING
namespace iom::ttnn_detail {
static std::atomic<bool> g_fail_next_quarantine_action{false};
static bool g_quarantine_action_fault_consumed = false;

static std::atomic<bool> g_fail_next_copy_registration{false};
static std::atomic<bool> g_fail_next_copy_outcome_insertion{false};
static std::atomic<bool> g_fail_next_binary_outcome_insertion{false};
static std::atomic<bool> g_copy_registration_fault_consumed{false};
static std::atomic<bool> g_copy_outcome_insertion_fault_consumed{false};
static std::atomic<bool> g_binary_outcome_insertion_fault_consumed{false};

static std::atomic<std::size_t> g_fail_copy_mesh_finishes{0};
static std::atomic<std::uint64_t> g_copy_mesh_finish_count{0};

static std::atomic<bool> g_copy_planes_fault_armed{false};
static std::atomic<std::size_t> g_copy_planes_fault_at{0};
static std::atomic<bool> g_copy_planes_fault_consumed{false};

static std::atomic<bool> g_host_transfer_fault_armed{false};
static std::atomic<std::size_t> g_host_transfer_fault_at{0};
static std::atomic<bool> g_host_transfer_fault_consumed{false};

static std::mutex g_binary_execution_mutex;
static std::condition_variable g_binary_execution_cv;
static bool g_binary_execution_armed = false;
static bool g_binary_execution_reached = false;

void consume_quarantine_action_fault_locked() noexcept(false) {
    if (g_fail_next_quarantine_action.exchange(
                false, std::memory_order_acquire)
            && !g_quarantine_action_fault_consumed) {
        g_quarantine_action_fault_consumed = true;
        throw std::bad_alloc();
    }
}

void consume_copy_registration_fault() noexcept(false) {
    if (g_fail_next_copy_registration.exchange(
                false, std::memory_order_acquire)) {
        g_copy_registration_fault_consumed.store(
                true, std::memory_order_release);
        throw std::bad_alloc();
    }
}

void consume_copy_outcome_insertion_fault() noexcept(false) {
    if (g_fail_next_copy_outcome_insertion.exchange(
                false, std::memory_order_acquire)) {
        g_copy_outcome_insertion_fault_consumed.store(
                true, std::memory_order_release);
        throw std::bad_alloc();
    }
}

void consume_binary_outcome_insertion_fault() noexcept(false) {
    if (g_fail_next_binary_outcome_insertion.exchange(
                false, std::memory_order_acquire)) {
        g_binary_outcome_insertion_fault_consumed.store(
                true, std::memory_order_release);
        throw std::bad_alloc();
    }
}

bool consume_copy_finish_fault() noexcept {
    g_copy_mesh_finish_count.fetch_add(1, std::memory_order_relaxed);
    std::size_t remaining = g_fail_copy_mesh_finishes.load(
            std::memory_order_acquire);
    for (;;) {
        if (remaining == 0) {
            return false;
        }
        if (g_fail_copy_mesh_finishes.compare_exchange_weak(
                    remaining, remaining - 1,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
            return true;
        }
    }
}

void fail_copy_planes_submission_at(std::size_t index) noexcept(false) {
    if (g_copy_planes_fault_armed.load(std::memory_order_acquire)
            && index
                    == g_copy_planes_fault_at.load(
                            std::memory_order_acquire)) {
        g_copy_planes_fault_armed.store(
                false, std::memory_order_release);
        g_copy_planes_fault_consumed.store(
                true, std::memory_order_release);
        throw std::runtime_error(
                "injected TTNN copy-plane submission failure");
    }
}

void fail_host_transfer_submission_at(std::size_t index) noexcept(false) {
    if (g_host_transfer_fault_armed.load(std::memory_order_acquire)
            && index
                    == g_host_transfer_fault_at.load(
                            std::memory_order_acquire)) {
        g_host_transfer_fault_armed.store(
                false, std::memory_order_release);
        g_host_transfer_fault_consumed.store(
                true, std::memory_order_release);
        throw std::runtime_error(
                "injected TTNN host-transfer submission failure");
    }
}

void wait_binary_execution_barrier() noexcept(false) {
    std::unique_lock<std::mutex> lock(g_binary_execution_mutex);
    if (!g_binary_execution_armed) {
        return;
    }
    g_binary_execution_reached = true;
    g_binary_execution_cv.notify_all();
    g_binary_execution_cv.wait(lock, [] {
        return !g_binary_execution_armed;
    });
}

}  // namespace iom::ttnn_detail

namespace iom::ttnn_test {
void hold_binary_execution_barrier_for_testing() noexcept {
    std::lock_guard<std::mutex> lock(
            ttnn_detail::g_binary_execution_mutex);
    ttnn_detail::g_binary_execution_reached = false;
    ttnn_detail::g_binary_execution_armed = true;
}

void wait_binary_execution_barrier_for_testing() noexcept {
    std::unique_lock<std::mutex> lock(
            ttnn_detail::g_binary_execution_mutex);
    ttnn_detail::g_binary_execution_cv.wait(lock, [] {
        return ttnn_detail::g_binary_execution_reached;
    });
}

void release_binary_execution_barrier_for_testing() noexcept {
    {
        std::lock_guard<std::mutex> lock(
                ttnn_detail::g_binary_execution_mutex);
        ttnn_detail::g_binary_execution_armed = false;
    }
    ttnn_detail::g_binary_execution_cv.notify_all();
}


void fail_next_quarantine_action_for_testing() noexcept {
    ttnn_detail::g_fail_next_quarantine_action.store(
            true, std::memory_order_release);
}

bool quarantine_action_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_quarantine_action_fault_consumed;
}

void fail_next_copy_planes_submission_for_testing(
        std::size_t plane_index) noexcept {
    ttnn_detail::g_copy_planes_fault_at.store(
            plane_index, std::memory_order_release);
    ttnn_detail::g_copy_planes_fault_consumed.store(
            false, std::memory_order_release);
    ttnn_detail::g_copy_planes_fault_armed.store(
            true, std::memory_order_release);
}

bool copy_planes_submission_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_copy_planes_fault_consumed.load(
            std::memory_order_acquire);
}

void fail_next_binary_outcome_insertion_for_testing() noexcept {
    ttnn_detail::g_fail_next_binary_outcome_insertion.store(
            true, std::memory_order_release);
}

bool binary_outcome_insertion_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_binary_outcome_insertion_fault_consumed.load(
            std::memory_order_acquire);
}

void fail_next_copy_registration_for_testing() noexcept {
    ttnn_detail::g_fail_next_copy_registration.store(
            true, std::memory_order_release);
}

bool copy_registration_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_copy_registration_fault_consumed.load(
            std::memory_order_acquire);
}

void fail_next_copy_outcome_insertion_for_testing() noexcept {
    ttnn_detail::g_fail_next_copy_outcome_insertion.store(
            true, std::memory_order_release);
}

bool copy_outcome_insertion_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_copy_outcome_insertion_fault_consumed.load(
            std::memory_order_acquire);
}

void fail_next_copy_finishes_for_testing(std::size_t count) noexcept {
    ttnn_detail::g_fail_copy_mesh_finishes.store(
            count, std::memory_order_release);
}

bool copy_finish_fault_pending_for_testing() noexcept {
    return ttnn_detail::g_fail_copy_mesh_finishes.load(
                   std::memory_order_acquire)
           != 0;
}

void reset_copy_finish_count_for_testing() noexcept {
    ttnn_detail::g_copy_mesh_finish_count.store(
            0, std::memory_order_release);
}

std::uint64_t copy_finish_count_for_testing() noexcept {
    return ttnn_detail::g_copy_mesh_finish_count.load(
            std::memory_order_acquire);
}

void fail_next_host_transfer_submission_for_testing(
        std::size_t plane_index) noexcept {
    ttnn_detail::g_host_transfer_fault_at.store(
            plane_index, std::memory_order_release);
    ttnn_detail::g_host_transfer_fault_consumed.store(
            false, std::memory_order_release);
    ttnn_detail::g_host_transfer_fault_armed.store(
            true, std::memory_order_release);
}

bool host_transfer_submission_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_host_transfer_fault_consumed.load(
            std::memory_order_acquire);
}

void fail_next_host_transfer_staging_allocation_for_testing() noexcept {
    ttnn_detail::g_fail_next_host_staging_allocation.store(
            true, std::memory_order_release);
    ttnn_detail::g_host_staging_allocation_fault_consumed.store(
            false, std::memory_order_release);
}

bool host_transfer_staging_allocation_fault_consumed_for_testing() noexcept {
    return ttnn_detail::g_host_staging_allocation_fault_consumed.load(
            std::memory_order_acquire);
}

std::size_t host_transfer_staging_allocation_count_for_testing() noexcept {
    return ttnn_detail::g_host_staging_allocations.load(
            std::memory_order_acquire);
}

}  // namespace iom::ttnn_test
#endif

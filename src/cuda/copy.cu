#include "copy.hpp"

#include "iom/device.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <limits>
#include <type_traits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include "driver.hpp"

namespace iom::cuda_detail {
namespace {

std::atomic<SubmissionFault> g_submission_fault{SubmissionFault::none};
std::atomic<int> g_event_record_failures{0};
std::atomic<int> g_stream_sync_failures{0};

}  // namespace

// External linkage: the policy's inline member functions in copy.hpp
// reference these from every including translation unit.
#ifdef IOM_ENABLE_TESTING
std::atomic<std::size_t> event_create_count_for_testing{0};
std::atomic<std::size_t> event_destroy_count_for_testing{0};
std::atomic<std::size_t> stream_create_count_for_testing{0};
std::atomic<std::size_t> stream_destroy_count_for_testing{0};
#endif  // IOM_ENABLE_TESTING

bool consume_submission_fault(
        SubmissionFault point) noexcept {
    if (point == SubmissionFault::event_record
            || point == SubmissionFault::stream_synchronize) {
        auto& failures = point == SubmissionFault::event_record
                ? g_event_record_failures : g_stream_sync_failures;
        int remaining = failures.load(std::memory_order_acquire);
        while (remaining > 0
                && !failures.compare_exchange_weak(
                        remaining, remaining - 1,
                        std::memory_order_acq_rel)) {
        }
        if (remaining > 0) {
            return true;
        }
    }
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}

}  // namespace iom::cuda_detail

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(__VA_ARGS__)
#include "../shared/standard_tiled_copy.inl"
#include "../shared/standard_tiled_add.inl"

#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

#include "../shared/gpu_queue.hpp"

namespace iom::cuda_detail {
void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_event_record_failures.store(
            fault == SubmissionFault::event_record ? 2 : 0,
            std::memory_order_release);
    g_stream_sync_failures.store(
            fault == SubmissionFault::event_record ? 1 : 0,
            std::memory_order_release);
    g_submission_fault.store(fault, std::memory_order_release);
}

void region_from_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        CUcontext context, const TensorView& destination,
        std::span<const std::byte> source) {
    detail::synchronous_transfer<gpu_policy>(
            transfer_pool, staging_pool, context, destination, source, {},
            true);
}

void region_to_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        CUcontext context, const TensorView& source,
        std::span<std::byte> destination) {
    detail::synchronous_transfer<gpu_policy>(
            transfer_pool, staging_pool, context, source, {}, destination,
            false);
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device,
        detail::QueueResourceProvider& resource_provider, CUcontext context,
        detail::RegistryState& registry_state) {
    return std::make_unique<detail::GpuQueue<gpu_policy>>(
            device, resource_provider, context, registry_state);
}

#ifdef IOM_ENABLE_TESTING
void queue_resource_snapshot_for_testing(
        DeviceOps& queue, QueueResourceSnapshot& snapshot) {
    auto* gpu_queue = dynamic_cast<detail::GpuQueue<gpu_policy>*>(&queue);
    if (gpu_queue == nullptr) {
        throw std::logic_error("queue is not a live CUDA GpuQueue");
    }
    detail::MetadataSlotPool& pool = gpu_queue->metadata_pool_for_testing();
    EventRingState& ring = gpu_queue->event_ring_for_testing();
    snapshot.slot_count = pool.slot_count();
    snapshot.device_base = pool.device_base();
    snapshot.slot_stride = pool.slot_stride();
    snapshot.events_total = ring.event_count_for_testing();
    snapshot.events_in_use = ring.in_use_count_for_testing();
    snapshot.slots_in_use = pool.in_use_count();
    snapshot.slots_protected = pool.protected_count();
}

void reclaim_retained_queue_leases_for_testing(Device& device) {
    auto* provider =
            dynamic_cast<detail::QueueResourceProvider*>(&device);
    if (provider == nullptr) {
        throw std::logic_error("device does not own fixed queue resources");
    }
    provider->reclaim_retained_leases();
}
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::cuda_detail
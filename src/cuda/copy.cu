#include "copy.hpp"

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

}  // namespace

bool consume_submission_fault(
        SubmissionFault point) noexcept {
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
#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

#include "../shared/gpu_queue.hpp"

namespace iom::cuda_detail {

void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
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
        const Device& device, CUcontext context,
        detail::RegistryState& registry_state) {
    return std::make_unique<detail::GpuQueue<gpu_policy>>(
            device, context, registry_state);
}

}  // namespace iom::cuda_detail
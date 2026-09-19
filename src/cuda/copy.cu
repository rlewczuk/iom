#include <array>
#include "copy.hpp"

#include "iom/device.hpp"
#include "../iom_internal.hpp"

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
#define IOM_GPU_SHARED __shared__
#define IOM_GPU_BARRIER __syncthreads()
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(__VA_ARGS__)
#include "../shared/standard_tiled_copy.inl"
#include "../shared/standard_tiled_embedding.inl"
#include "../shared/standard_tiled_add.inl"
#include "../shared/standard_tiled_rmsnorm.inl"
#include "../shared/standard_tiled_linear.inl"
#include "../shared/standard_tiled_rope.inl"

#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_BARRIER
#undef IOM_GPU_SHARED
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

#include "../shared/gpu_queue.hpp"

namespace iom::cuda_detail {

// CUDA's RMSNorm leaf delegates to the shared logical-row kernel. The queue
// supplies its already-created nonblocking stream and immutable descriptor;
// this wrapper adds no stream, allocation, staging, or synchronization.
void gpu_policy::launch_rmsnorm(
        cudaStream_t stream, const detail::RmsnormMetadata& metadata) {
    detail::launch_standard_tiled_rmsnorm<gpu_policy>(stream, metadata);
}

// The SiLU queue descriptor is present so a future CUDA wrapper can bind the
// common producer without changing admission or completion ownership. This
// leaf deliberately keeps the policy unported and reports Unsupported.
void gpu_policy::launch_silu(
        cudaStream_t, const detail::SiluMetadata& metadata) {
    static_cast<void>(metadata);
    throw detail::UnsupportedOperation();
}

// The shared Rope facility is compiled into this translation unit, but the
// queue policy remains an explicit Unsupported stub until the independent
// CUDA wrapper leaf enables its launch.
void gpu_policy::launch_rope(
        cudaStream_t, const detail::RopeMetadata& metadata) {
    static_cast<void>(metadata);
    throw detail::UnsupportedOperation();
}

// CUDA's linear projection dispatches the native BF16 specialization on its
// own immutable descriptor leaf and delegates every other leaf to the shared
// tiled projection kernel under the same boundary rules: the queue's own
// stream, one uploaded fixed descriptor, no staging, no allocation, and no
// synchronization. `BF16` reaches this dispatch only on a queue whose device
// resolved the BF16 WMMA facility, so the specialization is never entered on a
// device or build without it.
void gpu_policy::launch_linear(
        cudaStream_t stream, const detail::LinearMetadata& metadata) {
    if (metadata.type == static_cast<std::uint32_t>(DataType::BF16)) {
        launch_linear_bf16(stream, metadata);
        return;
    }
    detail::launch_standard_tiled_linear<gpu_policy>(stream, metadata);
}

namespace {

// The twenty non-BF16 applicable linear leaves this CUDA translation unit
// carries: the twelve integer leaves and the eight ordinary signed floating
// leaves. `BOOL`, `F8_E8M0`, and non-`NONE` quantization never reach a
// capability predicate because common admission classifies them first.
[[nodiscard]] bool scalar_linear_supported(DataType data_type) noexcept {
    switch (data_type) {
        case DataType::I2:
        case DataType::U2:
        case DataType::I4:
        case DataType::U4:
        case DataType::I8:
        case DataType::U8:
        case DataType::I16:
        case DataType::U16:
        case DataType::I32:
        case DataType::U32:
        case DataType::I64:
        case DataType::U64:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F16:
        case DataType::F32:
        case DataType::F64:
            return true;
        case DataType::BOOL:
        case DataType::F8_E8M0:
        case DataType::BF16:
            return false;
    }
    return false;
}

}  // namespace

// The queue policy of a device with the resolved BF16 WMMA facility carries
// all twenty-one applicable leaves: the twenty above plus `BF16` on the native
// specialization. `BF16` is applicable and implemented, so it is never a
// missing-implementation rejection here; a device without the facility uses
// `gpu_policy_scalar_linear`, whose predicate keeps it `Unsupported`.
bool gpu_policy::linear_supported(DataType data_type) noexcept {
    return scalar_linear_supported(data_type) || data_type == DataType::BF16;
}

// The scalar-only variant: the same twenty leaves, and `BF16` explicitly
// unsatisfied because this queue's device has no BF16 WMMA facility.
bool gpu_policy_scalar_linear::linear_supported(
        DataType data_type) noexcept {
    return scalar_linear_supported(data_type);
}

// The runtime BF16 WMMA facility of one exact device. Both facts are required:
// the device attribute that documents the BF16 tensor-core route (compute
// capability 8.0 or newer) and a loadable image of the native specialization
// that was actually compiled with its BF16 WMMA statements. Compile success or
// the presence of `mma.h` is never consulted, and a failed query is a
// conservative absence rather than a claim.
bool linear_bf16_wmma_facility(CUdevice device) noexcept {
#ifdef IOM_ENABLE_TESTING
    if (consume_submission_fault(SubmissionFault::bf16_wmma_facility)) {
        return false;
    }
#else
    static_cast<void>(device);
#endif
    int major = 0;
    int minor = 0;
    if (driver_calls.device_get_attribute(
                &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
                != CUDA_SUCCESS
            || driver_calls.device_get_attribute(
                       &minor,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
                    != CUDA_SUCCESS
            || major < 8) {
        return false;
    }
    return linear_bf16_wmma_image_arch() >= 800;
}

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

namespace {

detail::Fence transfer_fence() noexcept {
    detail::Fence fence;
    fence.invoke = [](const detail::Fence&) noexcept {
        return detail::FenceResult::success();
    };
    return fence;
}

struct WorkspaceAdmission {
    detail::WorkspaceLease lease;
    void* address = nullptr;
};

WorkspaceAdmission begin_workspace_transfer(
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& view, RawWorkspaceView workspace,
        bool& resource_poisoned) {
    const WorkspaceRequirements requirements =
            view.copy_from_host_workspace_requirements();
    const std::array<TensorView, 1> operands{view};
    const RawWorkspaceView checked =
            detail::WorkspaceValidation::validated(
                    device, workspace, requirements.bytes,
                    requirements.alignment, operands);
    if (resource_poisoned) {
        throw std::bad_alloc();
    }
    const detail::QueueId queue_id =
            detail::allocate_queue_id(registry_state);
    return {
            detail::acquire_workspace_lease(
                    registry_state, workspace.owner_identity(),
                    detail::WorkspaceValidation::address(checked),
                    checked.byte_size(), queue_id, queue_id,
                    transfer_fence()),
            detail::WorkspaceValidation::address(checked)};
}
void finish_workspace_transfer(
        detail::RegistryState& registry_state,
        const detail::WorkspaceLease& lease, bool proof) noexcept {
    detail::complete_workspace_lease(registry_state, lease, proof);
}

}  // namespace

void region_from_host(
        cudaStream_t transfer_stream, CUcontext context,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& destination, RawWorkspaceView workspace,
        std::span<const std::byte> source, bool& resource_poisoned) {
    WorkspaceAdmission admission = begin_workspace_transfer(
            device, registry_state, destination, workspace, resource_poisoned);
    try {
        detail::synchronous_transfer<gpu_policy>(
                transfer_stream, context, destination, admission.address,
                source, {}, true);
        finish_workspace_transfer(registry_state, admission.lease, true);
    } catch (...) {
        resource_poisoned = true;
        finish_workspace_transfer(registry_state, admission.lease, false);
        throw;
    }
}

void region_to_host(
        cudaStream_t transfer_stream, CUcontext context,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& source, RawWorkspaceView workspace,
        std::span<std::byte> destination, bool& resource_poisoned) {
    WorkspaceAdmission admission = begin_workspace_transfer(
            device, registry_state, source, workspace, resource_poisoned);
    try {
        detail::synchronous_transfer<gpu_policy>(
                transfer_stream, context, source, admission.address,
                {}, destination, false);
        finish_workspace_transfer(registry_state, admission.lease, true);
    } catch (...) {
        resource_poisoned = true;
        finish_workspace_transfer(registry_state, admission.lease, false);
        throw;
    }
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device,
        detail::QueueResourceProvider& resource_provider, CUcontext context,
        detail::RegistryState& registry_state, bool bf16_wmma_facility) {
    // The queue's policy is chosen once, from the runtime device fact, and
    // never re-read by a submission: a device with the BF16 WMMA facility
    // queues the native specialization, and a device without it keeps every
    // other leaf on the established scalar path and reports `BF16`
    // `Unsupported`.
    if (bf16_wmma_facility) {
        return std::make_unique<detail::GpuQueue<gpu_policy>>(
                device, resource_provider, context, registry_state);
    }
    return std::make_unique<detail::GpuQueue<gpu_policy_scalar_linear>>(
            device, resource_provider, context, registry_state);
}

#ifdef IOM_ENABLE_TESTING
namespace {

// The observed fixed resource geometry of one live queue, for either
// capability variant. Each variant is a distinct instantiation of the shared
// queue template, so both are tried before the queue is rejected as foreign.
template <typename Policy>
[[nodiscard]] bool queue_resource_snapshot_of(
        DeviceOps& queue, QueueResourceSnapshot& snapshot) {
    auto* gpu_queue = dynamic_cast<detail::GpuQueue<Policy>*>(&queue);
    if (gpu_queue == nullptr) {
        return false;
    }
    detail::MetadataSlotPool& pool = gpu_queue->metadata_pool_for_testing();
    typename detail::EventRingState<Policy>& ring =
            gpu_queue->event_ring_for_testing();
    snapshot.slot_count = pool.slot_count();
    snapshot.device_base = pool.device_base();
    snapshot.slot_stride = pool.slot_stride();
    snapshot.events_total = ring.event_count_for_testing();
    snapshot.events_in_use = ring.in_use_count_for_testing();
    snapshot.slots_in_use = pool.in_use_count();
    snapshot.slots_protected = pool.protected_count();
    return true;
}

}  // namespace

void queue_resource_snapshot_for_testing(
        DeviceOps& queue, QueueResourceSnapshot& snapshot) {
    if (queue_resource_snapshot_of<gpu_policy>(queue, snapshot)
            || queue_resource_snapshot_of<gpu_policy_scalar_linear>(
                    queue, snapshot)) {
        return;
    }
    throw std::logic_error("queue is not a live CUDA GpuQueue");
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

namespace iom::detail {

// Compile-time coverage of the otherwise deferred launch boundary: the shared
// launcher and its device operation are instantiated for this backend policy
// in every build of this translation unit, so both toolchains compile the
// complete shared implementation before a wrapper calls it. An explicit
// instantiation is only well-formed in an enclosing namespace of the
// template's own namespace, so this one lives here rather than in
// `iom::cuda_detail`.
template void launch_standard_tiled_rmsnorm<iom::cuda_detail::gpu_policy>(
        iom::cuda_detail::gpu_policy::stream_type stream,
        const RmsnormMetadata& metadata);
template void launch_standard_tiled_linear<iom::cuda_detail::gpu_policy>(
        iom::cuda_detail::gpu_policy::stream_type stream,
        const LinearMetadata& metadata);
template void launch_standard_tiled_rope<iom::cuda_detail::gpu_policy>(
        iom::cuda_detail::gpu_policy::stream_type stream,
        const RopeMetadata& metadata);

}  // namespace iom::detail
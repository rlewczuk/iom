#include "copy.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <array>
#include <cstdint>
#include <exception>
#include <mutex>
#include <condition_variable>
#include <limits>
#include <type_traits>
#include <memory>
#include <new>
#include <optional>
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

namespace {

void synchronize_and_destroy_stream(cudaStream_t stream) noexcept {
    if (stream == nullptr) {
        return;
    }
    (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
}


}  // namespace
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


namespace iom::cuda_detail {
namespace {
struct CudaFenceResource {
    cudaEvent_t event = nullptr;
    std::size_t metadata_slot = 0;
    detail::MetadataSlotPool<gpu_policy>* pool = nullptr;
    CUcontext context = nullptr;
    std::exception_ptr retained_failure;
    std::atomic<std::size_t> refcount{1};
    std::optional<detail::FenceResult> cached_result;
    std::mutex cached_result_mu;
};

struct CudaFenceLease {
    CudaFenceResource* resource = nullptr;

    CudaFenceLease() noexcept = default;

    static CudaFenceLease acquire_lease(CudaFenceResource* resource) noexcept {
        CudaFenceLease lease;
        lease.resource = resource;
        if (resource != nullptr) {
            resource->refcount.fetch_add(1, std::memory_order_relaxed);
        }
        return lease;
    }

    CudaFenceLease(const CudaFenceLease& other) noexcept
            : resource(other.resource) {
        if (resource != nullptr) {
            resource->refcount.fetch_add(1, std::memory_order_relaxed);
        }
    }

    CudaFenceLease(CudaFenceLease&& other) noexcept
            : resource(other.resource) {
        other.resource = nullptr;
    }

    CudaFenceLease& operator=(const CudaFenceLease&) = delete;
    CudaFenceLease& operator=(CudaFenceLease&&) = delete;

    ~CudaFenceLease() noexcept {
        if (resource != nullptr
                && resource->refcount.fetch_sub(
                           1, std::memory_order_acq_rel)
                        == 1) {
            try {
                gpu_policy::activate(resource->context);
            } catch (...) {
            }
            gpu_policy::destroy_event_noexcept(resource->event);
            delete resource;
        }
    }
};

static_assert(sizeof(CudaFenceLease) <= detail::kFenceStorageBytes);
static_assert(alignof(CudaFenceLease) <= detail::kFenceStorageAlign);
static_assert(noexcept(
        CudaFenceLease(std::declval<const CudaFenceLease&>())));
static_assert(noexcept(
        CudaFenceLease(std::declval<CudaFenceLease&&>())));
static_assert(noexcept(std::declval<CudaFenceLease&>().~CudaFenceLease()));
static_assert(!std::is_copy_assignable_v<CudaFenceLease>);
static_assert(!std::is_move_assignable_v<CudaFenceLease>);

detail::FenceResult cuda_synchronized_fence_event(
        CudaFenceResource& resource) noexcept {
    std::lock_guard<std::mutex> lock(resource.cached_result_mu);
    if (resource.cached_result.has_value()) {
        return *resource.cached_result;
    }
    try {
        gpu_policy::activate(resource.context);
        gpu_policy::synchronize_event(resource.event);
        if (resource.retained_failure) {
            resource.cached_result.emplace(
                    detail::FenceResult::failed(
                            std::move(resource.retained_failure)));
        } else {
            resource.cached_result.emplace(detail::FenceResult::success());
        }
    } catch (...) {
        resource.cached_result.emplace(
                detail::FenceResult::failed(std::current_exception()));
    }
    return *resource.cached_result;
}

static_assert(noexcept(cuda_synchronized_fence_event(
        std::declval<CudaFenceResource&>())));

void cuda_fence_complete(void* opaque) {
    if (opaque == nullptr) {
        return;
    }
    const detail::FenceResult result =
            cuda_synchronized_fence_event(
                    *static_cast<CudaFenceResource*>(opaque));
    if (result.failure) {
        std::rethrow_exception(result.failure);
    }
}

void cuda_fence_destroy(void* opaque) noexcept {
    if (opaque == nullptr) {
        return;
    }
    auto* resource = static_cast<CudaFenceResource*>(opaque);
    (void)cuda_synchronized_fence_event(*resource);
    detail::MetadataSlotPool<gpu_policy>* pool = resource->pool;
    resource->pool = nullptr;
    if (pool != nullptr) {
        pool->release(resource->metadata_slot);
    }
    if (resource->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        try {
            gpu_policy::activate(resource->context);
        } catch (...) {
        }
        gpu_policy::destroy_event_noexcept(resource->event);
        delete resource;
    }
}

void cuda_fence_copy_construct(
        detail::Fence* destination, const detail::Fence& source) noexcept {
    ::new (destination->storage) CudaFenceLease{
            *std::launder(reinterpret_cast<const CudaFenceLease*>(
                    source.storage))};
}

static_assert(noexcept(cuda_fence_copy_construct(
        std::declval<detail::Fence*>(),
        std::declval<const detail::Fence&>())));

void cuda_fence_move_construct(
        detail::Fence* destination, detail::Fence* source) noexcept {
    ::new (destination->storage) CudaFenceLease{
            std::move(*std::launder(reinterpret_cast<CudaFenceLease*>(
                    source->storage)))};
    std::destroy_at(std::launder(reinterpret_cast<CudaFenceLease*>(
            source->storage)));
}

static_assert(noexcept(cuda_fence_move_construct(
        std::declval<detail::Fence*>(),
        std::declval<detail::Fence*>())));

void cuda_fence_storage_destroy(detail::Fence* fence) noexcept {
    std::destroy_at(std::launder(reinterpret_cast<CudaFenceLease*>(
            fence->storage)));
}

static_assert(noexcept(cuda_fence_storage_destroy(
        std::declval<detail::Fence*>())));

detail::FenceResult cuda_fence_invoke(
        const detail::Fence& fence) noexcept {
    const auto& lease = *std::launder(reinterpret_cast<const CudaFenceLease*>(
            fence.storage));
    if (lease.resource == nullptr) {
        return detail::FenceResult::success();
    }
    return cuda_synchronized_fence_event(*lease.resource);
}

static_assert(noexcept(cuda_fence_invoke(
        std::declval<const detail::Fence&>())));

detail::Fence build_cuda_fence(CudaFenceResource* resource) noexcept {
    detail::Fence fence;
    ::new (fence.storage) CudaFenceLease{
            CudaFenceLease::acquire_lease(resource)};
    fence.invoke = &cuda_fence_invoke;
    fence.copy_construct = &cuda_fence_copy_construct;
    fence.move_construct = &cuda_fence_move_construct;
    fence.destroy = &cuda_fence_storage_destroy;
    return fence;
}

}  // namespace
namespace {

class CudaQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        cudaEvent_t event = nullptr;
        bool no_op;
        void* fence = nullptr;
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
    };


public:
    CudaQueue(
            const Device& device, CUcontext context,
            detail::RegistryState& registry_state)
            : device_(&device),
              state_(&registry_state),
              registry_queue_id_(detail::allocate_queue_id(*state_)),
              context_(context),
              metadata_pool_(context_),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [](void* fence) {
                                  cuda_fence_complete(fence);
                              },
                              [](void* fence) {
                                  cuda_fence_destroy(fence);
                              },
                              [this](
                                      std::uint64_t sequence,
                                      std::exception_ptr failure) {
                                  complete_task(
                                          sequence, std::move(failure));
                              }},
                      detail::StagedWorker<Task>::PublishPolicy::Splice) {
        gpu_policy::activate(context_);
        try {
            stream_ = gpu_policy::create_queue_stream();
            worker_.start();
        } catch (...) {
            gpu_policy::destroy_queue_stream_noexcept(stream_);
            stream_ = gpu_policy::null_stream();
            throw;
        }
    }

    ~CudaQueue() override {
        state_->registry.invalidate_entries_for_queue(registry_queue_id_);
        worker_.shutdown_and_drain();
        try {
            gpu_policy::activate(context_);
            synchronize_and_destroy_stream(stream_);
        } catch (...) {
        }
        stream_ = gpu_policy::null_stream();
    }

    oid copy(const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(
                            Task{sequence, &source, &destination, nullptr,
                                 no_op});
                });
    }

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "CUDA";
    }

private:
    void execute(Task& task) {
        const TensorView& source = *task.source;
        TensorView& destination = *task.destination;
        task.source = nullptr;
        task.destination = nullptr;
        if (task.no_op) {
            task.event = nullptr;
            task.fence = nullptr;
            return;
        }

        gpu_policy::activate(context_);
        const std::size_t metadata_slot = metadata_pool_.acquire();
        cudaEvent_t event = nullptr;
        std::unique_ptr<CudaFenceResource> resource;
        bool kernel_enqueued = false;
        bool event_recorded = false;
        std::exception_ptr retained_failure;
        try {
            const detail::CopyMetadataLayout layout =
                    detail::copy_metadata_layout(
                            source, destination);
            metadata_pool_.ensure_slot_capacity(
                    metadata_slot, layout.bytes, stream_);
            detail::write_copy_metadata(
                    metadata_pool_.host_data(metadata_slot),
                    source, destination);
            gpu_policy::create_event(&event);
            gpu_policy::copy_from_host(
                    stream_, metadata_pool_.device_data(metadata_slot),
                    metadata_pool_.host_data(metadata_slot), layout.bytes);
            detail::launch_grid_stride_copy<gpu_policy>(
                    stream_,
                    static_cast<const unsigned char*>(
                            source.native_handle()),
                    static_cast<unsigned char*>(
                            destination.native_handle()),
                    static_cast<const detail::CopyMetadataHeader*>(
                            metadata_pool_.device_data(metadata_slot)),
                    layout.total_words);
            kernel_enqueued = true;
            gpu_policy::check_kernel(gpu_policy::copy_kernel_operation());
            gpu_policy::after_grid_stride_launch();
            gpu_policy::record_event(event, stream_);
            event_recorded = true;
            resource = std::make_unique<CudaFenceResource>();
            resource->event = event;
            resource->metadata_slot = metadata_slot;
            resource->pool = &metadata_pool_;
            resource->context = context_;
            task.event = event;
            task.fence = resource.release();
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (kernel_enqueued) {
                if (!event_recorded) {
                    gpu_policy::record_event_no_fault(event, stream_);
                    event_recorded = true;
                }
                if (!resource) {
                    try {
                        resource = std::make_unique<CudaFenceResource>();
                    } catch (...) {
                        gpu_policy::synchronize_event_noexcept(event);
                        gpu_policy::destroy_event_noexcept(event);
                        metadata_pool_.release(metadata_slot);
                        throw;
                    }
                    resource->event = event;
                    resource->metadata_slot = metadata_slot;
                    resource->pool = &metadata_pool_;
                    resource->context = context_;
                }
                resource->retained_failure = failure;
                task.event = event;
                task.fence = resource.release();
                retained_failure = failure;
            } else {
                gpu_policy::destroy_event_noexcept(event);
                metadata_pool_.release(metadata_slot);
                throw;
            }
        }

        const detail::Fence fence = build_cuda_fence(
                static_cast<CudaFenceResource*>(task.fence));
        detail::EntryRegistration entries;
        try {
            entries = detail::register_copy_entries(
                    *state_, registry_queue_id_, task.sequence,
                    const_cast<void*>(source.native_handle()),
                    destination.native_handle(), fence);
            task.source_entry_id = entries.source;
            task.destination_entry_id = entries.destination;
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    detail::SequenceOutcome{
                            entries.source, entries.destination,
                            retained_failure});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate CUDA outstanding-work sequence");
            }
        } catch (...) {
            if (entries.source != 0) {
                state_->registry.remove_entry_if_present(
                        entries.source,
                        const_cast<void*>(source.native_handle()));
            }
            if (entries.destination != 0) {
                state_->registry.remove_entry_if_present(
                        entries.destination,
                        destination.native_handle());
            }
            if (task.fence != nullptr) {
                cuda_fence_destroy(task.fence);
                task.fence = nullptr;
                task.event = nullptr;
            } else {
                metadata_pool_.release(metadata_slot);
            }
            throw;
        }

        if (retained_failure) {
            commit_failure(task.sequence, retained_failure);
        }
    }

    void complete_task(
            std::uint64_t sequence, std::exception_ptr failure) {
        detail::SequenceOutcome outcome;
        bool has_outcome = false;
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto it = outcomes_.find(sequence);
            if (it != outcomes_.end()) {
                outcome = std::move(it->second);
                outcomes_.erase(it);
                has_outcome = true;
            }
        }
        if (has_outcome) {
            const bool fence_succeeded = !failure && !outcome.retained_failure;
            (void)detail::release_or_invalidate_entries(
                    state_->registry, outcome, static_cast<bool>(failure),
                    fence_succeeded);
        }
        complete(sequence, std::move(failure));
    }

    const Device* device_;
    detail::RegistryState* state_;
    detail::QueueId registry_queue_id_;
    CUcontext context_;
    cudaStream_t stream_ = gpu_policy::null_stream();
    detail::MetadataSlotPool<gpu_policy> metadata_pool_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, detail::SequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};
}  // namespace


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
    return std::make_unique<CudaQueue>(
            device, context, registry_state);
}

}  // namespace iom::cuda_detail

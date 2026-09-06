#include "copy.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <array>
#include <cstdint>
#include <exception>
#include <mutex>
#include <array>
#include <condition_variable>
#include <limits>
#include <type_traits>
#include <memory>
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


detail::FenceResult fence_event(
        CUcontext context, cudaEvent_t event,
        std::exception_ptr retained_failure) noexcept {
    if (event == nullptr && !retained_failure) {
        return detail::FenceResult::success();
    }
    try {
        const CUresult context_status = driver_calls.ctx_set_current(context);
        if (context_status != CUDA_SUCCESS) {
            throw cuda_error("cuCtxSetCurrent", context_status);
        }
        if (event != nullptr) {
            const cudaError_t status = cudaEventSynchronize(event);
            if (status != cudaSuccess) {
                throw std::runtime_error(
                        std::string("cudaEventSynchronize failed with ")
                        + cudaGetErrorName(status) + ": "
                        + cudaGetErrorString(status));
            }
        }
        if (retained_failure) {
            return detail::FenceResult::failed(std::move(retained_failure));
        }
        return detail::FenceResult::success();
    } catch (...) {
        return detail::FenceResult::failed(std::current_exception());
    }
}

detail::Fence make_fence(
        CUcontext context, cudaEvent_t event,
        std::exception_ptr retained_failure = nullptr) {
    return [context, event,
            retained_failure = std::move(retained_failure)]() mutable noexcept {
        return fence_event(context, event, retained_failure);
    };
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
};

void destroy_cuda_resource_noexcept(CudaFenceResource* resource) noexcept {
    if (resource == nullptr) {
        return;
    }
    gpu_policy::destroy_event_noexcept(resource->event);
    resource->pool->release(resource->metadata_slot);
    delete resource;
}

void cuda_fence_complete(void* opaque) {
    auto* resource = static_cast<CudaFenceResource*>(opaque);
    gpu_policy::activate(resource->context);
    gpu_policy::synchronize_event(resource->event);
}

void cuda_fence_destroy(void* opaque) noexcept {
    auto* resource = static_cast<CudaFenceResource*>(opaque);
    if (resource == nullptr) {
        return;
    }
    try {
        gpu_policy::activate(resource->context);
    } catch (...) {
    }
    gpu_policy::synchronize_event_noexcept(resource->event);
    destroy_cuda_resource_noexcept(resource);
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

    struct SequenceOutcome {
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
        bool fence_succeeded = false;
        std::exception_ptr retained_failure;
    };

public:
    CudaQueue(
            const Device& device, CUcontext context,
            CudaRegistryState& registry_state)
            : device_(&device),
              state_(&registry_state),
              registry_queue_id_(allocate_queue_id(*state_)),
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

    oid add(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "add");
    }
    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "mul");
    }
    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "silu");
    }
    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "linear");
    }
    oid rmsnorm(const TensorView&, TensorView&, const TensorView&, float,
                size_t) override {
        throw unsupported("CUDA", "rmsnorm");
    }
    oid sdpa(const TensorView&, const TensorView&, const TensorView&,
             size_t, size_t, size_t, TensorView&) override {
        throw unsupported("CUDA", "sdpa");
    }

private:
    void execute(Task& task) {
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
                            *task.source, *task.destination);
            metadata_pool_.ensure_slot_capacity(
                    metadata_slot, layout.bytes, stream_);
            detail::write_copy_metadata(
                    metadata_pool_.host_data(metadata_slot),
                    *task.source, *task.destination);
            gpu_policy::create_event(&event);
            resource = std::make_unique<CudaFenceResource>(
                    CudaFenceResource{
                            event, metadata_slot, &metadata_pool_, context_});
            gpu_policy::copy_from_host(
                    stream_, metadata_pool_.device_data(metadata_slot),
                    metadata_pool_.host_data(metadata_slot), layout.bytes);
            detail::launch_grid_stride_copy<gpu_policy>(
                    stream_,
                    static_cast<const unsigned char*>(
                            task.source->native_handle()),
                    static_cast<unsigned char*>(
                            task.destination->native_handle()),
                    static_cast<const detail::CopyMetadataHeader*>(
                            metadata_pool_.device_data(metadata_slot)),
                    layout.total_words);
            kernel_enqueued = true;
            gpu_policy::check_kernel(gpu_policy::copy_kernel_operation());
            gpu_policy::after_grid_stride_launch();
            gpu_policy::record_event(event, stream_);
            event_recorded = true;
            task.event = event;
            task.fence = resource.release();
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (kernel_enqueued) {
                if (!event_recorded && resource) {
                    gpu_policy::record_event_no_fault(event, stream_);
                    event_recorded = true;
                }
                if (event_recorded && resource) {
                    task.event = event;
                    task.fence = resource.release();
                }
                retained_failure = failure;
            } else {
                if (resource) {
                    resource->event = event;
                    destroy_cuda_resource_noexcept(resource.release());
                } else {
                    gpu_policy::destroy_event_noexcept(event);
                    metadata_pool_.release(metadata_slot);
                }
                throw;
            }
        }

        const detail::Fence fence = make_fence(
                context_, task.event, retained_failure);
        detail::EntryRegistration entries;
        try {
            entries = register_copy_entries(
                    *state_, registry_queue_id_, task.sequence,
                    const_cast<void*>(task.source->native_handle()),
                    task.destination->native_handle(), fence);
            task.source_entry_id = entries.source;
            task.destination_entry_id = entries.destination;
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    SequenceOutcome{
                            entries.source, entries.destination,
                            retained_failure == nullptr, retained_failure});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate CUDA outstanding-work sequence");
            }
        } catch (...) {
            if (entries.source != 0) {
                state_->registry.remove_entry_if_present(
                        entries.source,
                        const_cast<void*>(task.source->native_handle()));
            }
            if (entries.destination != 0) {
                state_->registry.remove_entry_if_present(
                        entries.destination,
                        task.destination->native_handle());
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
        SequenceOutcome outcome;
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
            outcome.fence_succeeded = !failure && !outcome.retained_failure;
            const std::array<detail::EntryId, 2> entries{
                    outcome.source_entry_id,
                    outcome.destination_entry_id};
            if (failure || outcome.retained_failure) {
                state_->registry.invalidate_entries(entries);
            } else {
                (void)state_->registry.try_release_entry(entries[0]);
                (void)state_->registry.try_release_entry(entries[1]);
            }
        }
        complete(sequence, std::move(failure));
    }

    const Device* device_;
    CudaRegistryState* state_;
    detail::QueueId registry_queue_id_;
    CUcontext context_;
    cudaStream_t stream_ = gpu_policy::null_stream();
    detail::MetadataSlotPool<gpu_policy> metadata_pool_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SequenceOutcome> outcomes_;
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
        CudaRegistryState& registry_state) {
    return std::make_unique<CudaQueue>(
            device, context, registry_state);
}

}  // namespace iom::cuda_detail

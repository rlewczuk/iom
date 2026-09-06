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
detail::FenceResult cuda_fence_invoke(
        const detail::Fence& fence) noexcept {
    const auto& capture =
            *std::launder(reinterpret_cast<const EventLeaseWithFailure*>(
                    fence.storage));
    if (capture.state == nullptr) {
        return detail::FenceResult::success();
    }
    return capture.state->invoke_result(capture.slot_index);
}

static_assert(noexcept(cuda_fence_invoke(
        std::declval<const detail::Fence&>())));

detail::Fence build_cuda_fence(
        const std::shared_ptr<EventRingState>& state,
        std::size_t slot_index, std::exception_ptr retained_failure) noexcept {
    detail::Fence fence;
    ::new (fence.storage) EventLeaseWithFailure{
            state, slot_index, std::move(retained_failure)};
    fence.invoke = &cuda_fence_invoke;
    fence.copy_construct =
            &detail::FenceCaptureOps<EventLeaseWithFailure>::copy_construct;
    fence.move_construct =
            &detail::FenceCaptureOps<EventLeaseWithFailure>::move_construct;
    fence.destroy = &detail::FenceCaptureOps<EventLeaseWithFailure>::destroy;
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
        EventRingState::Slot* fence = nullptr;
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
    };


public:
    CudaQueue(
            const Device& device, CUcontext context,
            detail::RegistryState& registry_state)
            : device_(&device),
              registry_state_(&registry_state),
              registry_queue_id_(detail::allocate_queue_id(*registry_state_)),
              context_(context),
              metadata_pool_(context_),
              state_(
                      std::make_shared<EventRingState>(
                              context_, metadata_pool_)),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [state = state_](void* fence) {
                                  state->on_worker_complete(
                                          *static_cast<EventRingState::Slot*>(
                                                  fence));
                              },
                              [state = state_](void* fence) {
                                  state->on_worker_destroy(
                                          *static_cast<EventRingState::Slot*>(
                                                  fence));
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
        registry_state_->registry.invalidate_entries_for_queue(registry_queue_id_);
        worker_.shutdown_and_drain();
        try {
            gpu_policy::activate(context_);
            synchronize_and_destroy_stream(stream_);
        } catch (...) {
        }
        stream_ = gpu_policy::null_stream();
        state_.reset();
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
        const detail::CopyMetadataLayout layout =
                detail::copy_metadata_layout(source, destination);
        EventRingState::Slot* slot = nullptr;
        std::size_t slot_index = 0;
        std::size_t metadata_slot = EventRingState::kNoAttachedSlot;
        bool metadata_acquired = false;
        bool kernel_enqueued = false;
        bool event_recorded = false;
        std::exception_ptr retained_failure;
        detail::InlineCopyMetadata inline_metadata{};
        try {
            slot = &state_->acquire();
            slot_index = state_->slot_index(*slot);
            if (layout.bytes <= sizeof(detail::InlineCopyMetadata)) {
                detail::write_copy_metadata(
                        inline_metadata, source, destination);
                slot->attached_metadata_slot =
                        EventRingState::kNoAttachedSlot;
                detail::launch_grid_stride_copy<gpu_policy>(
                        stream_,
                        static_cast<const unsigned char*>(
                                source.native_handle()),
                        static_cast<unsigned char*>(
                                destination.native_handle()),
                        inline_metadata, layout.total_words);
            } else {
                metadata_slot = metadata_pool_.acquire();
                metadata_acquired = true;
                metadata_pool_.ensure_slot_capacity(
                        metadata_slot, layout.bytes, stream_);
                detail::write_copy_metadata(
                        metadata_pool_.host_data(metadata_slot),
                        source, destination);
                slot->attached_metadata_slot = metadata_slot;
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
            }
            kernel_enqueued = true;
            gpu_policy::check_kernel(gpu_policy::copy_kernel_operation());
            gpu_policy::after_grid_stride_launch();
            gpu_policy::record_event(slot->event, stream_);
            event_recorded = true;
            task.event = slot->event;
            task.fence = slot;
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (slot == nullptr) {
                throw;
            }
            if (kernel_enqueued) {
                if (!event_recorded) {
                    gpu_policy::record_event_no_fault(slot->event, stream_);
                    event_recorded = true;
                }
                task.event = slot->event;
                task.fence = slot;
                retained_failure = failure;
            } else {
                if (metadata_acquired) {
                    metadata_pool_.release(metadata_slot);
                }
                state_->release(*slot);
                throw;
            }
        }

        const std::exception_ptr outcome_failure = retained_failure;
        const detail::Fence fence = build_cuda_fence(
                state_, slot_index, std::move(retained_failure));
        detail::EntryRegistration entries;
        try {
            entries = detail::register_copy_entries(
                    *registry_state_, registry_queue_id_, task.sequence,
                    const_cast<void*>(source.native_handle()),
                    destination.native_handle(), fence);
            task.source_entry_id = entries.source;
            task.destination_entry_id = entries.destination;
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    detail::SequenceOutcome{
                            entries.source, entries.destination,
                            outcome_failure});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate CUDA outstanding-work sequence");
            }
        } catch (...) {
            if (entries.source != 0) {
                registry_state_->registry.remove_entry_if_present(
                        entries.source,
                        const_cast<void*>(source.native_handle()));
            }
            if (entries.destination != 0) {
                registry_state_->registry.remove_entry_if_present(
                        entries.destination,
                        destination.native_handle());
            }
            if (task.fence != nullptr) {
                state_->on_worker_destroy(*task.fence);
                task.fence = nullptr;
                task.event = nullptr;
            } else {
                if (metadata_acquired) {
                    metadata_pool_.release(metadata_slot);
                }
                state_->release(*slot);
            }
            throw;
        }

        if (outcome_failure) {
            commit_failure(task.sequence, outcome_failure);
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
                    registry_state_->registry, outcome, static_cast<bool>(failure),
                    fence_succeeded);
        }
        complete(sequence, std::move(failure));
    }

    const Device* device_;
    detail::RegistryState* registry_state_;
    detail::QueueId registry_queue_id_;
    CUcontext context_;
    cudaStream_t stream_ = gpu_policy::null_stream();
    detail::MetadataSlotPool<gpu_policy> metadata_pool_;
    std::shared_ptr<EventRingState> state_;
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

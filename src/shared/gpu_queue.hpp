#pragma once

// The one policy-templated asynchronous queue implementation shared by the
// CUDA and ROCm backends. Every accelerator copy preserves the same
// submission, ownership, ordering, deferred-error, fence, registry,
// quarantine, and queue-destruction protocol; only the runtime primitives
// and diagnostics differ, and those stay behind the backend-local
// `gpu_policy` type. The template composes the shared EventRingState<Policy>,
// MetadataSlotPool<Policy>, StagedWorker<Task>, register_copy_entries, and
// release_or_invalidate_entries primitives.
//
// This header must be included by a backend translation unit only after the
// backend-specific expansion of standard_tiled_copy.inl, because the queue
// composes that file's metadata helpers (CopyMetadataLayout,
// InlineCopyMetadata, copy_metadata_layout, write_copy_metadata) by
// non-dependent names.

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "event_ring.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/iom.hpp"

namespace iom::detail {

template <typename Policy>
class GpuQueue final : public DeviceOps {
    using EventRing = iom::detail::EventRingState<Policy>;

    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        bool no_op;
        // Owns the completion record while the task is queued; keeps the
        // record alive for the worker callbacks referenced by fence.
        std::shared_ptr<typename EventRing::Submission> submission;
        void* fence = nullptr;
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
    };

    // One submission's completion lease paired with a post-launch retained
    // failure. Captured by the registry fence so the completion record —
    // and its pooled event — stays reserved until every registry entry and
    // destructor snapshot referencing it has disappeared.
    struct EventLeaseWithFailure {
        std::shared_ptr<typename EventRing::Submission> submission;
        std::exception_ptr retained_failure;
    };
    static_assert(
            sizeof(EventLeaseWithFailure)
            <= iom::detail::kFenceStorageBytes);
    static_assert(
            alignof(EventLeaseWithFailure)
            <= iom::detail::kFenceStorageAlign);

    [[nodiscard]] static detail::FenceResult fence_invoke(
            const detail::Fence& fence) noexcept {
        const auto& capture =
                *std::launder(reinterpret_cast<const EventLeaseWithFailure*>(
                        fence.storage));
        if (capture.submission == nullptr) {
            return detail::FenceResult::success();
        }
        detail::FenceResult result = capture.submission->invoke_result();
        if (result.succeeded && result.failure == nullptr
                && capture.retained_failure) {
            result = detail::FenceResult::failed(capture.retained_failure);
        }
        return result;
    }

    static_assert(noexcept(fence_invoke(
            std::declval<const detail::Fence&>())));

    [[nodiscard]] static detail::Fence build_fence(
            std::shared_ptr<typename EventRing::Submission> submission,
            std::exception_ptr retained_failure) noexcept {
        detail::Fence fence;
        ::new (fence.storage)
                EventLeaseWithFailure{std::move(submission),
                        std::move(retained_failure)};
        fence.invoke = &GpuQueue::fence_invoke;
        fence.copy_construct =
                &detail::FenceCaptureOps<EventLeaseWithFailure>::copy_construct;
        fence.move_construct =
                &detail::FenceCaptureOps<EventLeaseWithFailure>::move_construct;
        fence.destroy =
                &detail::FenceCaptureOps<EventLeaseWithFailure>::destroy;
        return fence;
    }

    // The worker's four callbacks. Built in a helper so the dependent
    // Callbacks type is named with `typename` outside a member initializer,
    // where a braced-init-list of a qualified dependent name would misparse.
    [[nodiscard]] static typename detail::StagedWorker<Task>::Callbacks
    make_worker_callbacks(GpuQueue* self, std::shared_ptr<EventRing> state) {
        typename detail::StagedWorker<Task>::Callbacks callbacks{
                [self](Task& task) {
                    self->execute(task);
                },
                [state](void* fence) {
                    state->on_worker_complete(
                            *static_cast<typename EventRing::Submission*>(
                                    fence));
                },
                [state](void* fence) {
                    state->on_worker_destroy(
                            *static_cast<typename EventRing::Submission*>(
                                    fence));
                },
                [self](
                        std::uint64_t sequence,
                        std::exception_ptr failure) {
                    self->complete_task(sequence, std::move(failure));
                }};
        return callbacks;
    }

public:
    GpuQueue(
            const Device& device, typename Policy::context_type context,
            detail::RegistryState& registry_state)
            : device_(&device),
              registry_state_(&registry_state),
              registry_queue_id_(
                      detail::allocate_queue_id(*registry_state_)),
              context_(context),
              metadata_pool_(context_),
              state_(std::make_shared<EventRing>(context_, metadata_pool_)),
              worker_(
                      make_worker_callbacks(this, state_),
                      detail::StagedWorker<Task>::PublishPolicy::Splice) {
        Policy::activate(context_);
        try {
            stream_ = Policy::create_queue_stream();
            worker_.start();
        } catch (...) {
            Policy::destroy_queue_stream_noexcept(stream_);
            stream_ = Policy::null_stream();
            throw;
        }
    }

    ~GpuQueue() override {
        registry_state_->registry.invalidate_entries_for_queue(
                registry_queue_id_);
        worker_.shutdown_and_drain();
        try {
            Policy::activate(context_);
            (void)Policy::synchronize_stream_noexcept(stream_);
            Policy::destroy_queue_stream_noexcept(stream_);
        } catch (...) {
        }
        stream_ = Policy::null_stream();
        state_.reset();
    }

    iom::oid copy(
            const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(
                            Task{sequence, &source, &destination, no_op});
                });
    }

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return Policy::backend_label();
    }

private:
    void execute(Task& task) {
        const TensorView& source = *task.source;
        TensorView& destination = *task.destination;
        task.source = nullptr;
        task.destination = nullptr;
        if (task.no_op) {
            task.submission.reset();
            task.fence = nullptr;
            return;
        }

        Policy::activate(context_);
        const detail::CopyMetadataLayout layout =
                detail::copy_metadata_layout(source, destination);
        std::shared_ptr<typename EventRing::Submission> submission;
        std::size_t metadata_slot = EventRing::kNoAttachedSlot;
        bool metadata_acquired = false;
        bool kernel_enqueued = false;
        bool event_recorded = false;
        std::exception_ptr retained_failure;
        detail::InlineCopyMetadata inline_metadata{};
        try {
            submission = state_->acquire();
            if (layout.bytes <= sizeof(detail::InlineCopyMetadata)) {
                detail::write_copy_metadata(
                        inline_metadata, source, destination);
                detail::launch_grid_stride_copy<Policy>(
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
                        metadata_slot, layout.bytes);
                detail::write_copy_metadata(
                        metadata_pool_.host_data(metadata_slot),
                        source, destination);
                submission->attach_metadata_slot(metadata_slot);
                Policy::copy_from_host(
                        stream_, metadata_pool_.device_data(metadata_slot),
                        metadata_pool_.host_data(metadata_slot), layout.bytes);
                detail::launch_grid_stride_copy<Policy>(
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
            Policy::check_kernel(Policy::copy_kernel_operation());
            Policy::after_grid_stride_launch();
            Policy::record_event(
                    state_->event_of(*submission), stream_);
            event_recorded = true;
            state_->mark_event_recorded(*submission);
            task.submission = submission;
            task.fence = submission.get();
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if (submission == nullptr) {
                throw;
            }
            if (kernel_enqueued) {
                if (!event_recorded) {
                    event_recorded = Policy::record_event_no_fault(
                            state_->event_of(*submission), stream_);
                    if (event_recorded) {
                        state_->mark_event_recorded(*submission);
                    } else if (Policy::synchronize_stream_noexcept(stream_)) {
                        state_->mark_stream_drained(*submission);
                    }
                }
                task.submission = submission;
                task.fence = submission.get();
                retained_failure = failure;
            } else {
                if (metadata_acquired) {
                    metadata_pool_.release(metadata_slot);
                }
                submission.reset();
                throw;
            }
        }

        const std::exception_ptr outcome_failure = retained_failure;
        const detail::Fence fence = build_fence(
                submission, std::move(retained_failure));
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
                        std::string("duplicate ")
                        + Policy::backend_label()
                        + " outstanding-work sequence");
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
                state_->on_worker_destroy(
                        *static_cast<typename EventRing::Submission*>(
                                task.fence));
                task.submission.reset();
                task.fence = nullptr;
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
                    registry_state_->registry, outcome,
                    static_cast<bool>(failure), fence_succeeded);
        }
        complete(sequence, std::move(failure));
    }

    const Device* device_;
    detail::RegistryState* registry_state_;
    detail::QueueId registry_queue_id_;
    typename Policy::context_type context_;
    typename Policy::stream_type stream_ = Policy::null_stream();
    detail::MetadataSlotPool<Policy> metadata_pool_;
    std::shared_ptr<EventRing> state_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, detail::SequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace iom::detail
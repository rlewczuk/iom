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
#include <optional>
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
    struct MetadataLease {
        MetadataSlotPool<Policy>* pool = nullptr;
        std::size_t slot = EventRing::kNoAttachedSlot;

        ~MetadataLease() {
            if (pool != nullptr) {
                pool->release(slot);
            }
        }

        MetadataLease() = default;
        MetadataLease(
                MetadataSlotPool<Policy>* pool_value,
                std::size_t slot_value)
                : pool(pool_value), slot(slot_value) {}
        MetadataLease(const MetadataLease&) = delete;
        MetadataLease& operator=(const MetadataLease&) = delete;
        MetadataLease(MetadataLease&& other) noexcept
                : pool(std::exchange(other.pool, nullptr)),
                  slot(std::exchange(
                          other.slot, EventRing::kNoAttachedSlot)) {}
        MetadataLease& operator=(MetadataLease&& other) noexcept {
            if (this != &other) {
                if (pool != nullptr) {
                    pool->release(slot);
                }
                pool = std::exchange(other.pool, nullptr);
                slot = std::exchange(
                        other.slot, EventRing::kNoAttachedSlot);
            }
            return *this;
        }

        void handoff() noexcept {
            pool = nullptr;
            slot = EventRing::kNoAttachedSlot;
        }
    };


    struct Task {
        std::uint64_t sequence;
        const TensorView* source = nullptr;
        TensorView* destination = nullptr;
        bool no_op = false;
        bool is_add = false;
        std::optional<AddRequest> add_request;
        detail::AddEntryRegistration add_entries{};
        std::shared_ptr<typename EventRing::Submission> submission;
        void* fence = nullptr;
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
        MetadataLease metadata_lease;
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

    struct GpuOutcome {
        detail::SequenceOutcome common{};
        detail::AddEntryRegistration add_entries{};
        std::exception_ptr retained_failure;
        bool is_add = false;
    };

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
            : DeviceOps(device),
              device_(&device),
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

    iom::oid copy_impl(
            const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(
                            Task{sequence, &source, &destination, no_op});
                });
    }

    oid add_impl(const AddRequest& request) override {
        // Validate rank-dependent arithmetic and reserve the complete
        // metadata representation before accepting the request. The slot is
        // retained by the completion submission once the worker receives it.
        const std::size_t metadata_bytes =
                detail::add_metadata_storage_bytes(
                        request.result_shape.dimensions().size());
        (void)detail::make_add_metadata(request);
        Policy::activate(context_);
        const std::size_t metadata_slot = metadata_pool_.acquire();
        try {
            metadata_pool_.ensure_slot_capacity(
                    metadata_slot, metadata_bytes);
            auto submission = state_->acquire();
            const detail::Fence fence = build_fence(submission, nullptr);
            return submit_add(
                    request, *registry_state_, registry_queue_id_, fence,
                    [this, submission = std::move(submission),
                     metadata_slot](
                            std::uint64_t sequence,
                            const AddRequest& captured,
                            detail::AddEntryRegistration entries) mutable {
                        Task task;
                        task.sequence = sequence;
                        task.is_add = true;
                        task.add_request.emplace(captured);
                        task.add_entries = entries;
                        task.submission = std::move(submission);
                        task.metadata_lease = MetadataLease{
                                &metadata_pool_, metadata_slot};
                        worker_.submit_copy(std::move(task));
                    });
        } catch (...) {
            metadata_pool_.release(metadata_slot);
            throw;
        }
    }

private:
    void execute(Task& task) {
        if (task.is_add) {
            Policy::activate(context_);
            // Reserve the outcome — and its registration ownership — before
            // any device effect. A failure here is pre-acceptance: the
            // caller's submit_add rolls the sequence back with no live work.
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                const auto [it, inserted] =
                        outcomes_.try_emplace(task.sequence);
                if (!inserted) {
                    throw std::logic_error("duplicate ADD sequence");
                }
                it->second.add_entries = task.add_entries;
                it->second.is_add = true;
            }
            std::exception_ptr failure;
            try {
                const std::size_t metadata_bytes =
                        detail::add_metadata_storage_bytes(
                                task.add_request->result_shape
                                        .dimensions()
                                        .size());
                const std::size_t metadata_slot =
                        task.metadata_lease.slot;
                task.submission->attach_metadata_slot(metadata_slot);
                task.metadata_lease.handoff();
                detail::write_add_metadata(
                        metadata_pool_.host_data(metadata_slot),
                        metadata_pool_.device_data(metadata_slot),
                        *task.add_request);
                Policy::copy_from_host(
                        stream_,
                        metadata_pool_.device_data(metadata_slot),
                        metadata_pool_.host_data(metadata_slot),
                        metadata_bytes);
                const detail::AddMetadata metadata =
                        *reinterpret_cast<const detail::AddMetadata*>(
                                metadata_pool_.host_data(metadata_slot));
                detail::launch_grid_stride_add<Policy>(
                        stream_,
                        static_cast<const unsigned char*>(
                                task.add_request->lhs.native_handle),
                        static_cast<const unsigned char*>(
                                task.add_request->rhs.native_handle),
                        static_cast<unsigned char*>(
                                task.add_request->out.native_handle),
                        metadata);
                Policy::check_kernel(Policy::copy_kernel_operation());
                Policy::after_grid_stride_launch();
                Policy::record_event(
                        state_->event_of(*task.submission), stream_);
                state_->mark_event_recorded(*task.submission);
            } catch (...) {
                // Post-launch: never throw through submit_copy. Retain the
                // failure; completion releases entries and reports it.
                failure = std::current_exception();
                try {
                    const bool recorded = Policy::record_event_no_fault(
                            state_->event_of(*task.submission), stream_);
                    if (recorded) {
                        state_->mark_event_recorded(*task.submission);
                    } else if (Policy::synchronize_stream_noexcept(stream_)) {
                        state_->mark_stream_drained(*task.submission);
                    }
                } catch (...) {
                }
            }
            if (failure) {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_[task.sequence].retained_failure = failure;
            }
            task.fence = task.submission.get();
            return;
        }
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
            const auto [it, inserted] =
                    outcomes_.try_emplace(task.sequence);
            if (!inserted) {
                throw std::logic_error(
                        std::string("duplicate ")
                        + Policy::backend_label()
                        + " outstanding-work sequence");
            }
            it->second.common.source_entry_id = entries.source;
            it->second.common.destination_entry_id = entries.destination;
            it->second.common.retained_failure = outcome_failure;
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
        GpuOutcome outcome;
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
            std::exception_ptr combined = failure;
            if (!combined) {
                combined = outcome.retained_failure
                        ? outcome.retained_failure
                        : outcome.common.retained_failure;
            }
            const bool failed = static_cast<bool>(combined);
            if (outcome.is_add) {
                (void)detail::release_or_invalidate_add_entries(
                        registry_state_->registry, outcome.add_entries,
                        failed, !failed);
            } else {
                (void)detail::release_or_invalidate_entries(
                        registry_state_->registry, outcome.common,
                        failed, !failed);
            }
            complete(sequence, std::move(combined));
            return;
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
    std::map<std::uint64_t, GpuOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};
}  // namespace iom::detail
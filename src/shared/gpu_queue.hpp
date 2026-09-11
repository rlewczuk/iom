#pragma once

// The one policy-templated asynchronous queue implementation shared by the
// CUDA and ROCm backends. Every accelerator copy preserves the same
// submission, ownership, ordering, deferred-error, fence, registry,
// quarantine, and queue-destruction protocol; only the runtime primitives
// and diagnostics differ, and those stay behind the backend-local
// `gpu_policy` type. The template composes the shared EventRingState<Policy>,
// MetadataSlotPool, StagedWorker<Task>, register_copy_entries, and
// release_or_invalidate_entries primitives over one fixed queue-resource
// lease reserved at the Device boundary.
//
// Queue construction is transactional and ordered: the queue-count credit
// and one disjoint C-slot metadata partition are reserved first (a fifth
// live queue throws std::bad_alloc there), then the exactly C completion
// resources are created eagerly, then the native stream and the worker
// start. Any failure destroys only what was already created and returns
// every reservation. After publication, submission, dispatch, retirement,
// waits, and queue operations never allocate, free, resize, or replace
// native resources.
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
#include "queue_resources.hpp"

namespace iom::detail {

template <typename Policy>
class GpuQueue final : public DeviceOps {
    using EventRing = iom::detail::EventRingState<Policy>;

    struct CompletionState {
        mutable std::mutex mutex;
        std::shared_ptr<typename EventRing::Submission> submission;
        std::exception_ptr retained_failure;
        bool finalized = false;
        detail::FenceResult final_result = detail::FenceResult::pending();
        bool final_completion_proven = false;

        void bind(
                std::shared_ptr<typename EventRing::Submission> value) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            submission = std::move(value);
        }

        void clear() noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            submission.reset();
        }

        void set_failure(std::exception_ptr failure) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (retained_failure == nullptr) {
                retained_failure = std::move(failure);
            }
        }

        [[nodiscard]] detail::FenceResult finalize(
                std::exception_ptr callback_failure) noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (finalized) {
                return final_result;
            }
            detail::FenceResult result = detail::FenceResult::pending();
            if (submission != nullptr) {
                result = submission->invoke_result();
                final_completion_proven = submission->completion_proven();
            }
            if (callback_failure != nullptr) {
                result = detail::FenceResult::failed(
                        std::move(callback_failure));
            } else if (retained_failure != nullptr) {
                result = detail::FenceResult::failed(retained_failure);
            }
            final_result = result;
            finalized = true;
            // The terminal result is now independent of the reusable event
            // generation. Unknown generations remain quarantined by the ring
            // even after this shared_ptr is released.
            submission.reset();
            return final_result;
        }

        [[nodiscard]] detail::FenceResult result() const noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (finalized) {
                return final_result;
            }
            if (submission == nullptr) {
                return detail::FenceResult::pending();
            }
            detail::FenceResult result = submission->invoke_result();
            if (retained_failure != nullptr) {
                result = detail::FenceResult::failed(retained_failure);
            }
            return result;
        }

        [[nodiscard]] bool completion_proven() const noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            if (finalized) {
                return final_completion_proven;
            }
            return submission != nullptr && submission->completion_proven();
        }
    };

    struct GpuFenceCapture {
        std::shared_ptr<CompletionState> completion;
    };
    static_assert(sizeof(GpuFenceCapture) <= detail::kFenceStorageBytes);
    static_assert(alignof(GpuFenceCapture) <= detail::kFenceStorageAlign);

    struct MetadataLease {
        MetadataSlotPool* pool = nullptr;
        std::size_t slot = EventRing::kNoAttachedSlot;

        ~MetadataLease() {
            if (pool != nullptr) {
                pool->release(slot);
            }
        }

        MetadataLease() = default;
        MetadataLease(
                MetadataSlotPool* pool_value, std::size_t slot_value)
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
        std::uint64_t sequence = 0;
        std::optional<CopyRequest> copy_request;
        bool is_binary = false;
        std::optional<BinaryRequest> binary_request;
        detail::BinaryEntryRegistration binary_entries{};
        detail::EntryRegistration entries{};
        typename EventRing::Submission* submission = nullptr;
        std::shared_ptr<CompletionState> completion;
        void* fence = nullptr;
        MetadataLease metadata_lease;
    };

    struct SnapshotView {
        const CopyViewSnapshot& value;

        [[nodiscard]] const TensorSpec& spec() const noexcept {
            return value.spec;
        }
        [[nodiscard]] std::size_t plane_offset() const noexcept {
            return value.plane_offset;
        }
        [[nodiscard]] std::span<const std::size_t> plane_strides()
                const noexcept {
            return value.plane_strides;
        }
    };

    struct GpuOutcome {
        detail::SequenceOutcome common{};
        detail::BinaryEntryRegistration binary_entries{};
        detail::WorkspaceLease workspace_lease{};
        std::shared_ptr<CompletionState> completion;
        bool is_binary = false;
    };

    [[nodiscard]] static detail::FenceResult fence_invoke(
            const detail::Fence& fence) noexcept {
        const auto& capture =
                *std::launder(reinterpret_cast<const GpuFenceCapture*>(
                        fence.storage));
        if (capture.completion == nullptr) {
            return detail::FenceResult::pending();
        }
        return capture.completion->result();
    }

    static_assert(noexcept(fence_invoke(
            std::declval<const detail::Fence&>())));

    [[nodiscard]] static detail::Fence build_fence(
            std::shared_ptr<CompletionState> completion) noexcept {
        detail::Fence fence;
        ::new (fence.storage) GpuFenceCapture{std::move(completion)};
        fence.invoke = &GpuQueue::fence_invoke;
        fence.copy_construct =
                &detail::FenceCaptureOps<GpuFenceCapture>::copy_construct;
        fence.move_construct =
                &detail::FenceCaptureOps<GpuFenceCapture>::move_construct;
        fence.destroy =
                &detail::FenceCaptureOps<GpuFenceCapture>::destroy;
        return fence;
    }

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
    // Reservation order is fixed by member declaration order: the queue
    // resource lease (credit plus disjoint C-slot partition) is reserved
    // first, then the exactly C completion resources are created eagerly,
    // and only then does the constructor body create the native stream and
    // start the worker. A fifth live queue therefore throws std::bad_alloc
    // before any stream, worker, or completion resource exists.
    GpuQueue(
            const Device& device,
            QueueResourceProvider& resource_provider,
            typename Policy::context_type context,
            detail::RegistryState& registry_state)
            : DeviceOps(device),
              device_(&device),
              registry_state_(&registry_state),
              registry_queue_id_(
                      detail::allocate_queue_id(*registry_state_)),
              resource_provider_(&resource_provider),
              context_(context),
              metadata_pool_(std::make_shared<MetadataSlotPool>(
                      resource_provider.reserve_queue_resources())),
              state_(std::make_shared<EventRing>(
                      context_,
                      *metadata_pool_,
                      metadata_pool_->slot_count())),
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
        // Invalidate owner fences before draining the worker. Accepted work
        // that outlives this queue must quarantine its tensor payloads when
        // owners are destroyed after queue teardown; the device retains the
        // backing until a covering proof at its boundary.
        registry_state_->registry.invalidate_entries_for_queue(
                registry_queue_id_);
        // Close common admission before worker drain so no new operation
        // can race teardown; parked prepared leases are finished by the
        // second close after executing work retires.
        close_and_drain();
        worker_.shutdown_and_drain();
        // The first close leaves executing nodes to the backend drain. Once
        // the worker has retired them, finish the parked tail in FIFO order.
        close_and_drain();
        bool drained = false;
        try {
            Policy::activate(context_);
            drained = Policy::synchronize_stream_noexcept(stream_);
        } catch (...) {
            drained = false;
        }
        state_->on_queue_drain(drained);
        if (drained || !state_->has_unproven_completion()) {
            // Every lease is proved: release the native stream and return
            // the partition, queue-count reservation, and completion
            // resources for reuse.
            Policy::destroy_queue_stream_noexcept(stream_);
            stream_ = Policy::null_stream();
            state_.reset();
            metadata_pool_.reset();
            return;
        }
        // Unknown completion: quarantine the entire unresolved lease at the
        // Device boundary. The partition and queue-count reservation stay
        // retained with the completion resources, host mirrors, and the
        // queue's own covering-proof handle until this queue's own drain is
        // proved; a drain of another queue is never sufficient. Capture
        // order matters: the pool must be destroyed after the ring that
        // borrows it, so it is declared first (lambda captures are
        // destroyed in reverse declaration order).
        const typename Policy::context_type retained_context = context_;
        const typename Policy::stream_type retained_stream = stream_;
        stream_ = Policy::null_stream();
        std::shared_ptr<MetadataSlotPool> retained_pool =
                std::move(metadata_pool_);
        std::shared_ptr<EventRing> retained_state = std::move(state_);
        resource_provider_->retain_unknown_lease(
                [retained_context, retained_stream, retained_pool,
                 retained_state]() mutable -> bool {
                    try {
                        Policy::activate(retained_context);
                    } catch (...) {
                        return false;
                    }
                    if (!Policy::synchronize_stream_noexcept(
                                retained_stream)) {
                        return false;
                    }
                    // Covering proof: release protected slots and drop the
                    // lease. Dropping the captures destroys the completion
                    // resources and returns the partition and credit.
                    retained_state->on_queue_drain(true);
                    Policy::destroy_queue_stream_noexcept(retained_stream);
                    retained_state.reset();
                    retained_pool.reset();
                    return true;
                },
                [retained_context, retained_stream, retained_pool,
                 retained_state]() mutable {
                    // Best-effort teardown while the native context is still
                    // valid; unproven leases keep their outcome untouched.
                    try {
                        Policy::activate(retained_context);
                    } catch (...) {
                    }
                    Policy::destroy_queue_stream_noexcept(retained_stream);
                    retained_state.reset();
                    retained_pool.reset();
                });
    }
    iom::oid copy_impl(
            const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        // These injected failures model host registration/preparation faults,
        // so they are consumed before common acceptance reserves a sequence.
        if (Policy::consume_copy_registration_fault()
                || Policy::consume_copy_outcome_insertion_fault()) {
            throw std::bad_alloc();
        }
        auto completion = std::make_shared<CompletionState>();
        const detail::Fence fence = build_fence(completion);
        return submit_copy(
                source, destination, *registry_state_, registry_queue_id_,
                fence,
                [this, completion](
                        std::uint64_t sequence,
                        const CopyRequest& captured,
                        detail::EntryRegistration entries) {
                    auto submission = state_->try_acquire();
                    if (submission == nullptr) {
                        throw detail::AdmissionResourceUnavailable{};
                    }
                    completion->bind(submission);
                    try {
                        {
                            std::lock_guard<std::mutex> lock(outcome_mutex_);
                            const auto [it, inserted] =
                                    outcomes_.try_emplace(sequence);
                            if (!inserted) {
                                throw std::logic_error(
                                        std::string("duplicate ")
                                        + Policy::backend_label()
                                        + " outstanding-work sequence");
                            }
                            it->second.common.source_entry_id =
                                    entries.source;
                            it->second.common.destination_entry_id =
                                    entries.destination;
                            it->second.completion = completion;
                        }
                        Task task;
                        task.sequence = sequence;
                        task.copy_request.emplace(captured);
                        task.entries = entries;
                        task.submission = submission.get();
                        task.completion = completion;
                        task.fence = task.submission;
                        worker_.submit_copy(std::move(task));
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> lock(outcome_mutex_);
                            outcomes_.erase(sequence);
                        }
                        completion->clear();
                        throw;
                    }
                });
    }

    oid binary_impl(const BinaryRequest& request) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        // Validate descriptor arithmetic before common acceptance. The
        // metadata slot itself is acquired only by the dispatching head.
        (void)detail::make_binary_metadata(request);
        auto completion = std::make_shared<CompletionState>();
        const detail::Fence fence = build_fence(completion);
        return submit_binary(
                request, *registry_state_, registry_queue_id_, fence,
                [this, completion](
                        std::uint64_t sequence,
                        const BinaryRequest& captured,
                        detail::BinaryEntryRegistration entries) {
                    auto submission = state_->try_acquire();
                    if (submission == nullptr) {
                        throw detail::AdmissionResourceUnavailable{};
                    }
                    const auto metadata_slot = metadata_pool_->try_acquire();
                    if (!metadata_slot.has_value()) {
                        throw detail::AdmissionResourceUnavailable{};
                    }
                    MetadataLease metadata_lease{
                            metadata_pool_.get(), *metadata_slot};
                    completion->bind(submission);
                    try {
                        {
                            std::lock_guard<std::mutex> lock(outcome_mutex_);
                            const auto [it, inserted] =
                                    outcomes_.try_emplace(sequence);
                            if (!inserted) {
                                throw std::logic_error(
                                        "duplicate GPU binary sequence");
                            }
                            it->second.binary_entries = entries;
                            it->second.workspace_lease =
                                    captured.workspace_lease;
                            it->second.completion = completion;
                            it->second.is_binary = true;
                        }
                        Task task;
                        task.sequence = sequence;
                        task.is_binary = true;
                        task.binary_request.emplace(captured);
                        task.binary_entries = entries;
                        task.submission = submission.get();
                        task.completion = completion;
                        task.fence = task.submission;
                        task.metadata_lease = std::move(metadata_lease);
                        worker_.submit_copy(std::move(task));
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> lock(outcome_mutex_);
                            outcomes_.erase(sequence);
                        }
                        completion->clear();
                        throw;
                    }
                });
    }

private:
    void execute(Task& task) {
        task.fence = task.submission;
        std::exception_ptr retained_failure;
        bool native_work_submitted = false;
        bool event_recorded = false;

        try {
            Policy::activate(context_);
            if (task.is_binary) {
                const BinaryRequest& request = *task.binary_request;
                const std::size_t metadata_bytes =
                        detail::binary_metadata_storage_bytes(
                                request.result_shape.dimensions().size());
                const std::size_t metadata_slot =
                        task.metadata_lease.slot;
                task.submission->attach_metadata_slot(metadata_slot);
                task.metadata_lease.handoff();
                detail::write_binary_metadata(
                        metadata_pool_->host_data(metadata_slot),
                        metadata_pool_->device_data(metadata_slot), request);
                native_work_submitted = true;
                Policy::copy_from_host(
                        stream_, metadata_pool_->device_data(metadata_slot),
                        metadata_pool_->host_data(metadata_slot), metadata_bytes);
                const detail::BinaryMetadata metadata =
                        *reinterpret_cast<const detail::BinaryMetadata*>(
                                metadata_pool_->host_data(metadata_slot));
                const auto launch = [&]<DeviceBinaryOp Op>() {
                    detail::launch_grid_stride_binary<Policy, Op>(
                            stream_,
                            static_cast<const unsigned char*>(
                                    request.lhs.native_handle),
                            static_cast<const unsigned char*>(
                                    request.rhs.native_handle),
                            static_cast<unsigned char*>(
                                    request.out.native_handle), metadata);
                };
                switch (request.operation) {
                    case DeviceOps::BinaryOperation::Add:
                        launch.template operator()<DeviceBinaryOp::add>();
                        break;
                    case DeviceOps::BinaryOperation::Mul:
                        launch.template operator()<DeviceBinaryOp::mul>();
                        break;
                    case DeviceOps::BinaryOperation::Sub:
                        launch.template operator()<DeviceBinaryOp::sub>();
                        break;
                    case DeviceOps::BinaryOperation::Div:
                        launch.template operator()<DeviceBinaryOp::div>();
                        break;
                }
                Policy::check_kernel(Policy::copy_kernel_operation());
                Policy::after_grid_stride_launch();
                Policy::record_event(
                        state_->event_of(*task.submission), stream_);
                event_recorded = true;
                state_->mark_event_recorded(*task.submission);
            } else {
                const CopyRequest& request = *task.copy_request;
                if (request.no_op) {
                    state_->mark_stream_drained(*task.submission);
                    return;
                }
                const detail::CopyMetadataLayout layout =
                        detail::copy_metadata_layout(
                                SnapshotView{request.source},
                                SnapshotView{request.destination});
                detail::InlineCopyMetadata metadata{};
                detail::write_copy_metadata(
                        metadata, SnapshotView{request.source},
                        SnapshotView{request.destination});
                native_work_submitted = true;
                detail::launch_grid_stride_copy<Policy>(
                        stream_,
                        static_cast<const unsigned char*>(
                                request.source.native_handle),
                        static_cast<unsigned char*>(
                                request.destination.native_handle),
                        metadata, layout.total_words);
                Policy::check_kernel(Policy::copy_kernel_operation());
                Policy::after_grid_stride_launch();
                Policy::record_event(
                        state_->event_of(*task.submission), stream_);
                event_recorded = true;
                state_->mark_event_recorded(*task.submission);
            }
        } catch (...) {
            retained_failure = std::current_exception();
            if (native_work_submitted && !event_recorded) {
                try {
                    event_recorded = Policy::record_event_no_fault(
                            state_->event_of(*task.submission), stream_);
                    if (event_recorded) {
                        state_->mark_event_recorded(*task.submission);
                    } else if (Policy::synchronize_stream_noexcept(stream_)) {
                        state_->mark_stream_drained(*task.submission);
                    } else {
                        state_->mark_retire_unknown(*task.submission);
                    }
                } catch (...) {
                    state_->mark_retire_unknown(*task.submission);
                }
            } else if (!native_work_submitted) {
                // No native operation was attempted, so the fixed event
                // resource is safe even though the accepted token fails.
                state_->mark_stream_drained(*task.submission);
            }
        }

        if (retained_failure != nullptr) {
            task.completion->set_failure(retained_failure);
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
        if (!has_outcome) {
            complete(sequence, std::move(failure));
            return;
        }

        if (outcome.completion != nullptr) {
            const detail::FenceResult result =
                    outcome.completion->finalize(std::move(failure));
            failure = result.failure;
        }
        const bool failed = static_cast<bool>(failure);
        const bool completion_proven =
                outcome.completion == nullptr
                || outcome.completion->completion_proven();
        if (outcome.is_binary) {
            (void)detail::release_or_invalidate_binary_entries(
                    registry_state_->registry, outcome.binary_entries,
                    failed, completion_proven);
            detail::complete_workspace_lease(
                    *registry_state_, outcome.workspace_lease,
                    completion_proven);
        } else {
            (void)detail::release_or_invalidate_entries(
                    registry_state_->registry, outcome.common, failed,
                    completion_proven);
        }
        complete(sequence, std::move(failure));
    }

#ifdef IOM_ENABLE_TESTING
public:
    [[nodiscard]] MetadataSlotPool& metadata_pool_for_testing() noexcept {
        return *metadata_pool_;
    }

    [[nodiscard]] EventRing& event_ring_for_testing() noexcept {
        return *state_;
    }
#endif

private:
    const Device* device_;
    detail::RegistryState* registry_state_;
    detail::QueueId registry_queue_id_;
    QueueResourceProvider* resource_provider_;
    typename Policy::context_type context_;
    typename Policy::stream_type stream_ = Policy::null_stream();
    std::shared_ptr<MetadataSlotPool> metadata_pool_;
    std::shared_ptr<EventRing> state_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, GpuOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};
}  // namespace iom::detail

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
// backend-specific expansion of standard_tiled_copy.inl and
// standard_tiled_rmsnorm.inl, because the queue composes those files'
// metadata helpers (CopyMetadataLayout, InlineCopyMetadata,
// copy_metadata_layout, write_copy_metadata, RmsnormMetadata,
// make_rmsnorm_metadata, write_rmsnorm_metadata) by non-dependent names.

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

#include "../iom_internal.hpp"

#include "event_ring.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/iom.hpp"
#include "queue_resources.hpp"
#include "standard_tiled_embedding.hpp"

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

        [[nodiscard]] std::optional<std::uint32_t> status_word() const noexcept {
            std::lock_guard<std::mutex> lock(mutex);
            return submission != nullptr ? submission->status_word()
                                         : std::nullopt;
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
        bool is_embedding = false;
        std::optional<EmbeddingRequest> embedding_request;
        std::size_t status_slot = EventRing::kNoAttachedSlot;
        detail::BinaryEntryRegistration binary_entries{};
        bool is_rmsnorm = false;
        std::optional<RmsnormRequest> rmsnorm_request;
        detail::BinaryEntryRegistration rmsnorm_entries{};
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
        detail::BinaryEntryRegistration rmsnorm_entries{};
        detail::WorkspaceLease workspace_lease{};
        std::shared_ptr<CompletionState> completion;
        bool is_binary = false;
        bool is_rmsnorm = false;
        bool is_embedding = false;
    };

    [[nodiscard]] static detail::FenceResult fence_invoke(
            const detail::Fence& fence) noexcept;

    [[nodiscard]] static detail::Fence build_fence(
            std::shared_ptr<CompletionState> completion) noexcept;

    [[nodiscard]] static typename detail::StagedWorker<Task>::Callbacks
    make_worker_callbacks(GpuQueue* self, std::shared_ptr<EventRing> state);

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
            detail::RegistryState& registry_state);
    ~GpuQueue() override;
    oid copy_impl(
            const TensorView& source, TensorView& destination) override;
    oid binary_impl(const BinaryRequest& request) override;

    // RMSNorm consumes the immutable common request and the common
    // owner-registration output; the shared branch adds no admission of its
    // own and reuses the fixed metadata, event, worker, and metadata-lease
    // resources above. The capability stays the backend policy's until its
    // RMSNorm wrapper lands.
    oid rmsnorm_impl(const RmsnormRequest& request) override;

    [[nodiscard]] bool rmsnorm_supported(
            DataType data_type) const override;

    WorkspaceRequirements embedding_workspace_requirements_impl(
            const TensorView& table, const TensorView& indices,
            const TensorView& out) override;
    oid embedding_impl(const EmbeddingRequest& request) override;
private:
    void execute(Task& task);

    void complete_task(
            std::uint64_t sequence, std::exception_ptr failure);

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

#include "gpu_queue_lifecycle.inl"
#include "gpu_queue_operations.inl"

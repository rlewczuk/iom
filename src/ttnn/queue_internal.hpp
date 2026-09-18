#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

#include "iom/iom.hpp"
#include "device_internal.hpp"
#include "copy.hpp"
#include "embedding.hpp"
#include "linear.hpp"
namespace iom::ttnn_detail {

// Builds the per-submission TTNN fence used by every queued submission
// (copy, binary, embedding, rmsnorm). The capture carries the device
// pointer, the queue's domain-specific execution marker pointer, and the
// assigned submission sequence so `ttnn_fence_invoke` can gate the mesh
// finish on the parked-seam marker (a submission whose native work has
// not yet been fully enqueued must report `FenceResult::pending()` so
// `release_or_quarantine` retains the storage until the mesh actually
// runs).
//
// `executed_seq` must point at the marker for the submission's
// execution domain: `caller_executed_seq_` for copy/embedding, whose
// `execute()` runs synchronously on the submitter thread via
// `worker_.submit_copy`; `worker_executed_seq_` for binary/rmsnorm, whose
// `execute()` runs asynchronously on the staged worker's thread via
// `worker_.submit_after_publish`. Keeping the markers distinct stops one
// thread from over-advancing the marker for the other thread's
// submissions and letting `ttnn_fence_invoke` return success for a
// parked binary/rmsnorm whose native work has not yet been enqueued.
[[nodiscard]] detail::Fence build_ttnn_fence(
        TtnnDevice& device,
        std::atomic<std::uint64_t>& executed_seq,
        std::uint64_t sequence) noexcept;
void finish_locked_mesh(
        tt::tt_metal::distributed::MeshDevice& device);
void finish_locked(TtnnDevice& device);
[[nodiscard]] detail::FenceResult finish_native(TtnnDevice& device) noexcept;

}  // namespace iom::ttnn_detail

namespace iom {

class TtnnQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        std::optional<ttnn_detail::CopySnapshot> source;
        std::optional<ttnn_detail::CopySnapshot> destination;
        std::optional<ttnn_detail::EmbeddingNativeRequest>
                embedding_request;
        bool no_op = false;
        bool is_binary = false;
        bool is_rmsnorm = false;
        bool is_linear = false;
        bool is_embedding = false;
        DeviceOps::BinaryOperation operation =
                DeviceOps::BinaryOperation::Add;
        ttnn_detail::BinaryRequest binary_request{
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                TensorShape{{1, 1}}};
        std::optional<RmsnormRequest> rmsnorm_request;
        std::optional<ttnn_detail::LinearNativeRequest> linear_request;
        void* fence = nullptr;
        detail::BinaryEntryRegistration binary_entries{};
        detail::BinaryEntryRegistration rmsnorm_entries{};
        detail::WorkspaceLease workspace_lease{};
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;
        std::size_t embedding_slot = 0;

        Task(
                std::uint64_t sequence_, const CopyRequest& request,
                detail::EntryRegistration entries)
            : sequence(sequence_),
              source(ttnn_detail::CopySnapshot{
                      request.source.spec, request.source.native_handle,
                      request.source.plane_offset,
                      request.source.plane_strides}),
              destination(ttnn_detail::CopySnapshot{
                      request.destination.spec,
                      request.destination.native_handle,
                      request.destination.plane_offset,
                      request.destination.plane_strides}),
              no_op(request.no_op), source_entry_id(entries.source),
              destination_entry_id(entries.destination) {}

        Task(
                std::uint64_t sequence_,
                DeviceOps::BinaryOperation operation_,
                const ttnn_detail::BinaryRequest& request,
                detail::BinaryEntryRegistration entries,
                detail::WorkspaceLease workspace_lease_)
            : sequence(sequence_), is_binary(true), operation(operation_),
              binary_request(request), binary_entries(entries),
              workspace_lease(workspace_lease_) {}

        Task(
                std::uint64_t sequence_, const RmsnormRequest& request,
                detail::BinaryEntryRegistration entries)
            : sequence(sequence_), is_rmsnorm(true),
              rmsnorm_request(request), rmsnorm_entries(entries) {}

        Task(
                std::uint64_t sequence_,
                ttnn_detail::LinearNativeRequest request_,
                detail::BinaryEntryRegistration entries)
            : sequence(sequence_), is_linear(true),
              linear_request(std::move(request_)), binary_entries(entries) {}

        Task(
                std::uint64_t sequence_,
                ttnn_detail::EmbeddingNativeRequest request_,
                detail::BinaryEntryRegistration entries,
                std::size_t embedding_slot_)
            : sequence(sequence_), embedding_request(std::move(request_)),
              is_embedding(true), binary_entries(entries),
              workspace_lease(embedding_request->workspace_lease),
              embedding_slot(embedding_slot_) {}
    };

    struct BinaryOutcome {
        detail::BinaryEntryRegistration entries;
        detail::WorkspaceLease workspace_lease;
        bool native_work_submitted = false;
        bool native_completion_proven = false;
        std::exception_ptr retained_failure;
    };

    struct RmsnormOutcome {
        detail::BinaryEntryRegistration entries;
        bool native_work_submitted = false;
        bool native_completion_proven = false;
        std::exception_ptr retained_failure;
    };

    struct LinearOutcome {
        detail::BinaryEntryRegistration entries;
        detail::WorkspaceLease workspace_lease;
        bool native_work_submitted = false;
        bool native_completion_proven = false;
        std::exception_ptr retained_failure;
    };

    struct EmbeddingOutcome {
        detail::BinaryEntryRegistration entries;
        detail::WorkspaceLease workspace_lease;
        std::size_t status_slot = 0;
        bool native_work_submitted = false;
        std::exception_ptr retained_failure;
    };

public:
    explicit TtnnQueue(TtnnDevice& device);
    ~TtnnQueue() override;
    oid copy_impl(
            const TensorView& source,
            TensorView& destination) override;
    oid binary_impl(const BinaryRequest& request) override;
    oid rmsnorm_impl(const RmsnormRequest& request) override;
    [[nodiscard]] bool rmsnorm_supported(
            DataType data_type) const override;
    oid linear_impl(const LinearRequest& request) override;
    [[nodiscard]] WorkspaceRequirements linear_workspace_requirements_impl(
            const TensorView& x, const TensorView& w,
            const TensorView& out, std::size_t s, std::size_t R,
            LinearOutputLayout layout, std::size_t H,
            std::size_t D) override;
    oid embedding_impl(const EmbeddingRequest& request) override;
    [[nodiscard]] WorkspaceRequirements
            embedding_workspace_requirements_impl(
                    const TensorView& table, const TensorView& indices,
                    const TensorView& out) override;

private:
    void execute(Task& task);
    void execute_copy(Task& task);
    void execute_rmsnorm(Task& task);
    void execute_linear(Task& task);
    void execute_embedding(Task& task);
    void publish_native_completion(std::uint64_t sequence) noexcept;
    void complete_task(
            std::uint64_t sequence, std::exception_ptr failure);
    void complete_embedding_task(
            std::uint64_t sequence, std::exception_ptr failure);
    void fence_through_sequence(
            std::uint64_t sequence) noexcept override;

    TtnnDevice* device_;
    std::map<std::uint64_t, BinaryOutcome> binary_outcomes_;
    std::map<std::uint64_t, RmsnormOutcome> rmsnorm_outcomes_;
    std::map<std::uint64_t, LinearOutcome> linear_outcomes_;
    std::map<std::uint64_t, EmbeddingOutcome> embedding_outcomes_;
    detail::RegistryState* state_;
    detail::QueueId registry_queue_id_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, detail::SequenceOutcome> outcomes_;
    ttnn_detail::EmbeddingStatusResources embedding_status_;
    ttnn_detail::EmbeddingProgram embedding_program_;
    ttnn_detail::LinearProgram linear_program_;
    detail::StagedWorker<Task> worker_;
    std::mutex fence_mutex_;
    std::atomic<std::uint64_t> last_finished_seq_{0};
    // Two per-domain execution markers (`caller_executed_seq_` and
    // `worker_executed_seq_`). Each carries the highest sequence whose
    // task finished executing on that domain's thread:
    //   * caller_executed_seq_  - copy and embedding submissions, whose
    //                              `execute()` runs synchronously on the
    //                              submitter thread inside
    //                              `worker_.submit_copy`. The mesh drain
    //                              has already completed by the time the
    //                              marker is advanced, so the corresponding
    //                              outcome can be released on the same
    //                              thread.
    //   * worker_executed_seq_  - binary and rmsnorm submissions, whose
    //                              `execute()` runs asynchronously on the
    //                              staged worker's thread inside
    //                              `worker_.submit_after_publish`. The
    //                              marker only advances after the worker
    //                              thread returns from the SDK launch.
    // The markers are domain-local because the two threads can run
    // interleaved: a higher caller-domain sequence can finish executing
    // before a lower worker-domain sequence that was parked before it.
    // A single shared marker would let `ttnn_fence_invoke` return
    // success for the lower worker-domain sequence whose native work
    // has not actually reached the mesh; per-domain markers stop the
    // cross-domain overtake by isolating each fence's check to its
    // own domain's progress.
    //
    // Each marker is advanced via `monotonic_max_store`, a CAS-loop
    // replacement for `std::atomic::fetch_max` (C++26) that keeps the
    // marker strictly non-decreasing. Today each domain runs on a
    // single thread so plain stores would also be monotonic, but the
    // repair requirement enforces the invariant defensively in case the
    // queue is later parallelized.
    std::atomic<std::uint64_t> caller_executed_seq_{0};
    std::atomic<std::uint64_t> worker_executed_seq_{0};

    // CAS-loop monotonic-max store. Replaces `std::atomic::fetch_max`
    // (C++26) and keeps the marker strictly non-decreasing. Used for
    // both per-domain execution markers so reordering on each domain
    // can never regress the value.
    static void monotonic_max_store(
            std::atomic<std::uint64_t>& marker, std::uint64_t value,
            std::memory_order order) noexcept {
        std::uint64_t current =
                marker.load(std::memory_order_relaxed);
        while (current < value &&
               !marker.compare_exchange_weak(
                       current, value,
                       order, std::memory_order_relaxed)) {
            // CAS failed: `current` is refreshed by
            // compare_exchange_weak; try again until either `value`
            // is no longer greater than the marker or this thread
            // wins the race.
        }
    }
};

}  // namespace iom

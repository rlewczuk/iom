#pragma once

#include <atomic>
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

namespace iom::ttnn_detail {

[[nodiscard]] detail::Fence build_ttnn_fence(TtnnDevice& device) noexcept;
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
        bool no_op = false;
        bool is_binary = false;
        DeviceOps::BinaryOperation operation =
                DeviceOps::BinaryOperation::Add;
        ttnn_detail::BinaryRequest binary_request{
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                {TensorSpec{TensorShape{{1, 1}}, DataType::I32}, nullptr, 0, {}},
                TensorShape{{1, 1}}};
        void* fence = nullptr;
        detail::BinaryEntryRegistration binary_entries{};
        detail::WorkspaceLease workspace_lease{};
        detail::EntryId source_entry_id = 0;
        detail::EntryId destination_entry_id = 0;

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
    };

    struct BinaryOutcome {
        detail::BinaryEntryRegistration entries;
        detail::WorkspaceLease workspace_lease;
        bool native_work_submitted = false;
        bool native_completion_proven = false;
        std::exception_ptr retained_failure;
    };

public:
    explicit TtnnQueue(TtnnDevice& device);
    ~TtnnQueue() override;

    oid copy_impl(
            const TensorView& source,
            TensorView& destination) override;
    oid binary_impl(const BinaryRequest& request) override;

private:
    void execute(Task& task);
    void execute_copy(Task& task);
    void publish_native_completion(std::uint64_t sequence) noexcept;
    void complete_task(
            std::uint64_t sequence, std::exception_ptr failure);
    void fence_through_sequence(
            std::uint64_t sequence) noexcept override;

    TtnnDevice* device_;
    std::map<std::uint64_t, BinaryOutcome> binary_outcomes_;
    detail::RegistryState* state_;
    detail::QueueId registry_queue_id_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, detail::SequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
    std::mutex fence_mutex_;
    std::atomic<std::uint64_t> last_finished_seq_{0};
    // Highest sequence whose task finished executing with its outcome still
    // registered (every native plane enqueued, or a determined no-op).
    // Registration precedes native submission, so an outcome alone is not
    // proof its work reached the mesh; complete_task's batch collection
    // reads this marker, under outcome_mutex_, to stop the batch before
    // any not-yet-executed sequence.
    std::atomic<std::uint64_t> executed_seq_{0};
};

}  // namespace iom

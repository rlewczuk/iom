#include "queue_internal.hpp"

#include <tt-metalium/host_api.hpp>
#include <ttnn/device.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>

#include <exception>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "rmsnorm_device_operation.hpp"

namespace iom::ttnn_detail {
namespace {

struct TtnnFenceCapture {
    TtnnDevice* device = nullptr;
};

static_assert(sizeof(TtnnFenceCapture) <= detail::kFenceStorageBytes);
static_assert(alignof(TtnnFenceCapture) <= detail::kFenceStorageAlign);
static_assert(std::is_trivially_copyable_v<TtnnFenceCapture>);
static_assert(std::is_trivially_destructible_v<TtnnFenceCapture>);

}  // namespace

detail::FenceResult ttnn_fence_invoke(
        const detail::Fence& fence) noexcept {
    const auto& capture =
            *std::launder(reinterpret_cast<const TtnnFenceCapture*>(
                    fence.storage));
    try {
        std::lock_guard<std::mutex> lock(capture.device->api_mutex());
        capture.device->mesh().mesh_command_queue(0).finish();
        return detail::FenceResult::success();
    } catch (...) {
        return detail::FenceResult::failed(std::current_exception());
    }
}

detail::Fence build_ttnn_fence(TtnnDevice& device) noexcept {
    detail::Fence fence;
    ::new (fence.storage) TtnnFenceCapture{&device};
    fence.invoke = &ttnn_fence_invoke;
    return fence;
}

// Finishes the mesh command queue. The caller must already hold the
// device API mutex: the synchronous drain of a failed submission
// runs under execute's api lock, and complete_task's retry reaches
// the same drain through finish_native's own lock. Every copy-drain
// path routes through here so an armed finish-failure seam faults
// exactly those attempts; host transfers, fence invokes, and
// quarantine drains keep their own finish calls unaffected.
void finish_locked_mesh(tt::tt_metal::distributed::MeshDevice& device) {
#ifdef IOM_ENABLE_TESTING
    if (consume_copy_finish_fault()) {
        throw std::runtime_error("injected TTNN copy finish failure");
    }
#endif
    device.mesh_command_queue(0).finish();
}

void finish_locked(TtnnDevice& device) {
    finish_locked_mesh(device.mesh());
}

detail::FenceResult finish_native(TtnnDevice& device) noexcept {
    try {
        std::lock_guard<std::mutex> lock(device.api_mutex());
        finish_locked(device);
        return detail::FenceResult::success();
    } catch (...) {
        return detail::FenceResult::failed(std::current_exception());
    }
}

}  // namespace iom::ttnn_detail

namespace iom {

TtnnQueue::TtnnQueue(TtnnDevice& device)
        : DeviceOps(device),
          device_(&device),
          state_(&device.registry_state()),
          registry_queue_id_(detail::allocate_queue_id(*state_)),
          embedding_status_(device.queue_config().max_in_flight_per_queue),
          embedding_program_(device.mesh()),
          worker_(
                  detail::StagedWorker<Task>::Callbacks{
                          [this](Task& task) { execute(task); },
                          [](void*) {},
                          [](void*) {},
                          [this](
                                  std::uint64_t sequence,
                                  std::exception_ptr failure) {
                              complete_task(sequence, std::move(failure));
                          }},
                  detail::StagedWorker<Task>::PublishPolicy::CompleteOnThrow) {
    worker_.start();
}

TtnnQueue::~TtnnQueue() {
    // Invalidate owner fences before draining the worker. Accepted work
    // that outlives this queue must quarantine its tensor payloads when
    // owners are destroyed after queue teardown; the device retains the
    // backing until a covering proof at its boundary.
    state_->registry.invalidate_entries_for_queue(registry_queue_id_);
    // Close admission before worker drain so parked submissions cannot
    // race teardown; the second close completes any parked tail after
    // executing work retires.
    close_and_drain();
    worker_.shutdown_and_drain();
    close_and_drain();
}

oid TtnnQueue::copy_impl(
        const TensorView& source, TensorView& destination) {
    std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
#ifdef IOM_ENABLE_TESTING
    ttnn_detail::consume_copy_registration_fault();
#endif
    detail::Fence fence = ttnn_detail::build_ttnn_fence(*device_);
    return submit_copy(
            source, destination, *state_, registry_queue_id_, fence,
            [this](std::uint64_t sequence,
                   const CopyRequest& captured,
                   detail::EntryRegistration entries) {
                try {
                    worker_.submit_copy(Task(sequence, captured, entries));
                } catch (...) {
                    state_->registry.remove_entry_if_present(
                            entries.source,
                            captured.source.native_handle);
                    state_->registry.remove_entry_if_present(
                            entries.destination,
                            captured.destination.native_handle);
                    std::rethrow_exception(std::current_exception());
                }
            });
}

oid TtnnQueue::binary_impl(const BinaryRequest& request) {
    std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
    detail::Fence fence = ttnn_detail::build_ttnn_fence(*device_);
    return submit_binary(
            request, *state_, registry_queue_id_, fence,
            [this](std::uint64_t sequence, const BinaryRequest& captured,
                   detail::BinaryEntryRegistration entries) {
                try {
                    ttnn_detail::BinaryRequest internal{
                            {captured.lhs.spec, captured.lhs.native_handle,
                             captured.lhs.plane_offset,
                             captured.lhs.logical_plane_strides},
                            {captured.rhs.spec, captured.rhs.native_handle,
                             captured.rhs.plane_offset,
                             captured.rhs.logical_plane_strides},
                            {captured.out.spec, captured.out.native_handle,
                             captured.out.plane_offset,
                             captured.out.logical_plane_strides},
                            captured.result_shape};
                    Task task(
                            sequence, captured.operation, internal, entries,
                            captured.workspace_lease);
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
#ifdef IOM_ENABLE_TESTING
                        ttnn_detail::consume_binary_outcome_insertion_fault();
#endif
                        const auto [it, inserted] = binary_outcomes_.emplace(
                                sequence,
                                BinaryOutcome{
                                        task.binary_entries,
                                        task.workspace_lease});
                        if (!inserted) {
                            throw std::logic_error(
                                    "duplicate TTNN binary sequence");
                        }
                    }
                    worker_.submit_after_publish(std::move(task));
                } catch (...) {
                    bool rollback_outcome = false;
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        rollback_outcome =
                                binary_outcomes_.erase(sequence) != 0;
                    }
                    if (rollback_outcome) {
                        state_->registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(),
                                        entries.count));
                        detail::complete_workspace_lease(
                                *state_, captured.workspace_lease, true);
                    }
                    std::rethrow_exception(std::current_exception());
                }
            });
}
oid TtnnQueue::rmsnorm_impl(const RmsnormRequest& request) {
    std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
    const detail::Fence fence = ttnn_detail::build_ttnn_fence(*device_);
    return submit_rmsnorm(
            request, *state_, registry_queue_id_, fence,
            [this](std::uint64_t sequence, const RmsnormRequest& captured,
                   detail::BinaryEntryRegistration entries) {
                Task task(sequence, captured, entries);
                try {
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] = rmsnorm_outcomes_.emplace(
                                sequence,
                                RmsnormOutcome{task.rmsnorm_entries});
                        if (!inserted) {
                            throw std::logic_error(
                                    "duplicate TTNN RMSNorm sequence");
                        }
                    }
                    worker_.submit_after_publish(std::move(task));
                } catch (...) {
                    bool rollback_outcome = false;
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        rollback_outcome =
                                rmsnorm_outcomes_.erase(sequence) != 0;
                    }
                    if (rollback_outcome) {
                        state_->registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(),
                                        entries.count));
                    }
                    std::rethrow_exception(std::current_exception());
                }
            });
}
bool TtnnQueue::rmsnorm_supported(DataType data_type) const {
    return ttnn_detail::rmsnorm_supported(data_type);
}

void TtnnQueue::execute_rmsnorm(Task& task) {
    bool submitted = false;
    std::exception_ptr submission_failure;
    try {
        std::lock_guard<std::mutex> api_lock(device_->api_mutex());
        const RmsnormRequest& request = *task.rmsnorm_request;
        const auto* const x_planes =
                static_cast<const ttnn::Tensor*>(request.x.native_handle);
        const auto* const scale_planes =
                static_cast<const ttnn::Tensor*>(
                        request.scale.native_handle);
        auto* const out_planes =
                static_cast<ttnn::Tensor*>(request.out.native_handle);
        const std::size_t plane_count =
                ttnn_detail::snapshot_plane_count(request.x.spec);
        for (std::size_t index = 0; index < plane_count; ++index) {
            const std::size_t x_plane =
                    ttnn_detail::snapshot_owner_plane_at(
                            request.x.spec, request.x.plane_offset,
                            request.x.plane_strides, index);
            const std::size_t out_plane =
                    ttnn_detail::snapshot_owner_plane_at(
                            request.out.spec, request.out.plane_offset,
                            request.out.plane_strides, index);
            // Mark the sequence as having potentially reached native code
            // before invoking the adapter. If the SDK throws after accepting
            // a launch, completion must take the conservative drain path.
            submitted = true;
            static_cast<void>(ttnn_detail::launch_preallocated_rmsnorm(
                    device_->mesh(), static_cast<ttnn::QueueId>(0),
                    x_planes[x_plane], scale_planes[request.scale.plane_offset],
                    out_planes[out_plane], request.epsilon));
        }
    } catch (...) {
        submission_failure = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto it = rmsnorm_outcomes_.find(task.sequence);
        if (it != rmsnorm_outcomes_.end()) {
            it->second.retained_failure = submission_failure;
            it->second.native_work_submitted = submitted;
        }
    }
    executed_seq_.store(task.sequence, std::memory_order_release);
}

void TtnnQueue::execute(Task& task) {
    if (task.is_rmsnorm) {
        execute_rmsnorm(task);
        return;
    }
    if (task.is_embedding) {
        execute_embedding(task);
        return;
    }
    if (task.is_binary) {
        bool submitted = false;
        bool completion_proven = false;
        try {
            {
                std::lock_guard<std::mutex> api_lock(device_->api_mutex());
#ifdef IOM_ENABLE_TESTING
                ttnn_detail::wait_binary_execution_barrier();
#endif
                auto* lhs = static_cast<ttnn::Tensor*>(
                        task.binary_request.lhs.native_handle);
                auto* rhs = static_cast<ttnn::Tensor*>(
                        task.binary_request.rhs.native_handle);
                auto* out = static_cast<ttnn::Tensor*>(
                        task.binary_request.out.native_handle);
                const auto binary =
                        [&]<detail::scalar_add_detail::BinaryOp Op>() {
                            ttnn_detail::binary_planes<Op>(
                                    device_->mesh(), device_->host_staging(),
                                    task.binary_request, lhs, rhs, out,
                                    submitted, completion_proven,
                                    ttnn_detail::finish_locked_mesh);
                        };
                switch (task.operation) {
                    case DeviceOps::BinaryOperation::Add:
                        binary.template operator()<
                                detail::scalar_add_detail::BinaryOp::add>();
                        break;
                    case DeviceOps::BinaryOperation::Mul:
                        binary.template operator()<
                                detail::scalar_add_detail::BinaryOp::mul>();
                        break;
                    case DeviceOps::BinaryOperation::Sub:
                        binary.template operator()<
                                detail::scalar_add_detail::BinaryOp::sub>();
                        break;
                    case DeviceOps::BinaryOperation::Div:
                        binary.template operator()<
                                detail::scalar_add_detail::BinaryOp::div>();
                        break;
                }
            }
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                binary_outcomes_.at(task.sequence).native_work_submitted =
                        submitted;
                binary_outcomes_.at(task.sequence).native_completion_proven =
                        completion_proven;
            }
            if (completion_proven) {
                publish_native_completion(task.sequence);
            }
            executed_seq_.store(task.sequence, std::memory_order_release);
            return;
        } catch (...) {
            const std::exception_ptr submission_failure =
                    std::current_exception();
            // Native work may have been accepted, or the worker may have
            // failed before its first upload. In both cases the published
            // outcome retains the failure for complete_task and repeatable
            // waits; the accepted registry and workspace leases stay owned
            // until that completion path runs.
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                auto it = binary_outcomes_.find(task.sequence);
                if (it != binary_outcomes_.end()) {
                    it->second.retained_failure = submission_failure;
                    it->second.native_work_submitted = submitted;
                    it->second.native_completion_proven = completion_proven;
                }
            }
            if (completion_proven) {
                publish_native_completion(task.sequence);
            }
            executed_seq_.store(task.sequence, std::memory_order_release);
            return;
        }
    }
    execute_copy(task);
}


void TtnnQueue::publish_native_completion(std::uint64_t sequence) noexcept {
    std::lock_guard<std::mutex> fence_lock(fence_mutex_);
    const std::uint64_t now =
            last_finished_seq_.load(std::memory_order_acquire);
    if (sequence > now) {
        last_finished_seq_.store(sequence, std::memory_order_release);
    }
}

void TtnnQueue::complete_task(
        std::uint64_t sequence, std::exception_ptr failure) {
    bool is_embedding = false;
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        is_embedding =
                embedding_outcomes_.find(sequence) != embedding_outcomes_.end();
    }
    if (is_embedding) {
        complete_embedding_task(sequence, std::move(failure));
        return;
    }
    RmsnormOutcome rmsnorm_outcome;
    bool is_rmsnorm = false;
    BinaryOutcome binary_outcome;
    bool is_binary = false;
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto rmsnorm_it = rmsnorm_outcomes_.find(sequence);
        if (rmsnorm_it != rmsnorm_outcomes_.end()) {
            rmsnorm_outcome = std::move(rmsnorm_it->second);
            rmsnorm_outcomes_.erase(rmsnorm_it);
            is_rmsnorm = true;
        }
        const auto binary_it = binary_outcomes_.find(sequence);
        if (!is_rmsnorm && binary_it != binary_outcomes_.end()) {
            binary_outcome = std::move(binary_it->second);
            binary_outcomes_.erase(binary_it);
            is_binary = true;
        }
    }
    if (is_rmsnorm) {
        detail::FenceResult fence_result = detail::FenceResult::success();
        bool completion_proven = rmsnorm_outcome.native_completion_proven;
        if (!rmsnorm_outcome.native_work_submitted) {
            completion_proven = true;
        }
        if (rmsnorm_outcome.native_work_submitted && !completion_proven) {
            fence_result = ttnn_detail::finish_native(*device_);
            completion_proven =
                    fence_result.succeeded && !fence_result.failure;
            if (completion_proven) {
                publish_native_completion(sequence);
            }
        }
        const std::exception_ptr operation_failure =
                failure ? failure : rmsnorm_outcome.retained_failure;
        const bool fence_succeeded =
                fence_result.succeeded && !fence_result.failure;
        (void)detail::release_or_invalidate_binary_entries(
                state_->registry, rmsnorm_outcome.entries,
                static_cast<bool>(operation_failure) && !completion_proven,
                fence_succeeded);
        complete(sequence, operation_failure
                ? operation_failure : fence_result.failure);
        return;
    }
    if (is_binary) {
        detail::FenceResult fence_result = detail::FenceResult::success();
        bool completion_proven = binary_outcome.native_completion_proven;
        // A post-publication failure before the first native upload has no
        // outstanding device work to fence. Treat that empty path as proven
        // so accepted owners and workspace leases remain reusable.
        if (!binary_outcome.native_work_submitted) {
            completion_proven = true;
        }
        if (binary_outcome.native_work_submitted && !completion_proven) {
            fence_result = ttnn_detail::finish_native(*device_);
            completion_proven =
                    fence_result.succeeded && !fence_result.failure;
            if (completion_proven) {
                publish_native_completion(sequence);
            }
        }
        const std::exception_ptr operation_failure =
                failure ? failure : binary_outcome.retained_failure;
        const bool fence_succeeded =
                fence_result.succeeded && !fence_result.failure;
        const bool released = detail::release_or_invalidate_binary_entries(
                state_->registry, binary_outcome.entries,
                static_cast<bool>(operation_failure) && !completion_proven,
                fence_succeeded);
        (void)released;
        detail::complete_workspace_lease(
                *state_, binary_outcome.workspace_lease, completion_proven);
        complete(sequence, operation_failure
                ? operation_failure : fence_result.failure);
        return;
    }
    std::vector<std::pair<std::uint64_t, detail::SequenceOutcome>> batch;
    bool has_outcome = false;
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto first = outcomes_.find(sequence);
        if (first != outcomes_.end()) {
            batch.emplace_back(sequence, std::move(first->second));
            outcomes_.erase(first);
            has_outcome = true;
            const std::uint64_t executed =
                    executed_seq_.load(std::memory_order_acquire);
            std::uint64_t next = sequence + 1;
            for (auto it = outcomes_.upper_bound(sequence);
                 it != outcomes_.end() && it->first == next
                         && next <= executed;
                 it = outcomes_.erase(it), ++next) {
                batch.emplace_back(next, std::move(it->second));
            }
        }
    }
    if (!has_outcome) {
        complete(sequence, std::move(failure));
        return;
    }

    bool has_native_work = false;
    for (const auto& member : batch) {
        has_native_work =
                has_native_work || member.second.native_work_submitted;
    }
    // A no-op-only batch has no mesh work to drain. Native work in any
    // member requires one finish for the whole contiguous batch.
    const detail::FenceResult fence_result =
            has_native_work
                    ? ttnn_detail::finish_native(*device_)
                    : detail::FenceResult::success();
    const bool fence_succeeded =
            fence_result.succeeded && !fence_result.failure;
    for (std::size_t index = 0; index < batch.size(); ++index) {
        const std::uint64_t seq = batch[index].first;
        const detail::SequenceOutcome& outcome = batch[index].second;
        // The worker-supplied failure belongs only to the head task;
        // later members carry at most their retained submission failure.
        const std::exception_ptr operation_failure =
                index == 0 && failure ? failure : outcome.retained_failure;
        const bool released = detail::release_or_invalidate_entries(
                state_->registry, outcome,
                static_cast<bool>(operation_failure), fence_succeeded);
        if (released) {
            last_finished_seq_.store(seq, std::memory_order_release);
        }
        // The worker-supplied failure wins, then the retained submission
        // failure, then the native finish failure, in submission order
        // within the batch.
        std::exception_ptr completion_failure;
        if (index == 0 && failure) {
            completion_failure = failure;
        } else if (operation_failure) {
            completion_failure = operation_failure;
        } else {
            completion_failure = fence_result.failure;
        }
        if (index == 0) {
            complete(seq, std::move(completion_failure));
        } else if (completion_failure) {
            commit_failure(seq, std::move(completion_failure));
        }
    }
}

void TtnnQueue::fence_through_sequence(std::uint64_t sequence) noexcept {
    std::lock_guard<std::mutex> fence_lock(fence_mutex_);
    const std::uint64_t now =
            last_finished_seq_.load(std::memory_order_acquire);
    if (sequence <= now) {
        return;
    }
    const detail::FenceResult result = ttnn_detail::finish_native(*device_);
    if (result.succeeded && !result.failure) {
        last_finished_seq_.store(sequence, std::memory_order_release);
    } else {
        record_post_completion_failure(sequence, result.failure);
    }
}

std::unique_ptr<DeviceOps> TtnnDevice::create_ops() {
    return std::make_unique<TtnnQueue>(*this);
}

}  // namespace iom

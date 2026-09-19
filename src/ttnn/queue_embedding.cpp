#include "queue_internal.hpp"
#include "../iom_internal.hpp"

#include <ttnn/tensor/tensor.hpp>

#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <span>

namespace iom {
namespace {

[[nodiscard]] bool embedding_payload_supported(DataType type) noexcept {
    switch (type) {
        case DataType::BOOL:
        case DataType::I2:
        case DataType::U2:
        case DataType::I4:
        case DataType::U4:
        case DataType::I8:
        case DataType::U8:
        case DataType::I16:
        case DataType::U16:
        case DataType::I32:
        case DataType::U32:
        case DataType::I64:
        case DataType::U64:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F16:
        case DataType::BF16:
        case DataType::F32:
        case DataType::F64:
            return true;
        case DataType::F8_E8M0:
            return false;
    }
    return false;
}

[[nodiscard]] bool embedding_id_supported(DataType type) noexcept {
    switch (type) {
        case DataType::I2:
        case DataType::U2:
        case DataType::I4:
        case DataType::U4:
        case DataType::I8:
        case DataType::U8:
        case DataType::I16:
        case DataType::U16:
        case DataType::I32:
        case DataType::U32:
        case DataType::I64:
        case DataType::U64:
            return true;
        case DataType::BOOL:
        case DataType::F4_E2M1:
        case DataType::F6_E2M3:
        case DataType::F6_E3M2:
        case DataType::F8_E4M3FN:
        case DataType::F8_E5M2:
        case DataType::F8_E8M0:
        case DataType::F16:
        case DataType::BF16:
        case DataType::F32:
        case DataType::F64:
            return false;
    }
    return false;
}


}  // namespace

WorkspaceRequirements TtnnQueue::embedding_workspace_requirements_impl(
        const TensorView& table, const TensorView& indices,
        const TensorView&) {
    if (!embedding_payload_supported(table.spec().data_type)
            || !embedding_id_supported(indices.spec().data_type)) {
        throw detail::UnsupportedOperation();
    }
    return WorkspaceRequirements{32, 32};
}

oid TtnnQueue::embedding_impl(const EmbeddingRequest& request) {
    std::lock_guard<std::mutex> submission_lock(submission_order_mutex_);
    return submit_embedding(
            request, *state_, registry_queue_id_,
            [this](std::uint64_t sequence) {
                return ttnn_detail::build_ttnn_fence(
                        // Embedding's `execute()` runs synchronously on
                        // the submitter thread inside `worker_.submit_copy`,
                        // so its fence is gated on the caller-domain
                        // marker; this matches copy_impl's domain choice.
                        *device_, caller_executed_seq_, sequence);
            },
            [this](std::uint64_t sequence, const EmbeddingRequest& captured,
                   detail::BinaryEntryRegistration entries) {
                std::size_t status_slot = 0;
                bool status_acquired = false;
                try {
                    status_slot = embedding_status_.acquire();
                    status_acquired = true;
                    ttnn_detail::NativeWorkspace native_workspace;
                    {
                        std::lock_guard<std::mutex> api_lock(
                                device_->api_mutex());
                        native_workspace =
                                ttnn_detail::checked_native_workspace(
                                        *device_, captured.workspace);
                    }
                    ttnn_detail::EmbeddingNativeRequest internal{
                            {captured.table.spec,
                             captured.table.native_handle,
                             captured.table.plane_offset,
                             captured.table.plane_strides},
                            {captured.indices.spec,
                             captured.indices.native_handle,
                             captured.indices.plane_offset,
                             captured.indices.plane_strides},
                            {captured.out.spec,
                             captured.out.native_handle,
                             captured.out.plane_offset,
                             captured.out.plane_strides},
                            native_workspace,
                            captured.workspace_lease};
                    Task task(
                            sequence, std::move(internal), entries,
                            status_slot);
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        const auto [it, inserted] = embedding_outcomes_.emplace(
                                sequence,
                                EmbeddingOutcome{
                                        task.binary_entries,
                                        task.workspace_lease,
                                        status_slot});
                        if (!inserted) {
                            throw std::logic_error(
                                    "duplicate TTNN embedding sequence");
                        }
                    }
                    // submit_copy runs execute() before token publication so
                    // the mesh-fence registered at prepare time covers the
                    // already-dispatched native work for any embedding whose
                    // admission credit was free at submit time. Embedding
                    // submissions parked behind max_in_flight_per_queue are
                    // registered before they reach the worker, but the
                    // caller-domain executed-marker gate (see
                    // src/ttnn/queue_internal.hpp's per-domain marker
                    // rationale and src/ttnn/queue.cpp's ttnn_fence_invoke)
                    // causes the parked submission's fence to return
                    // FenceResult::pending() rather than success(); the
                    // subsequent release_or_quarantine call therefore
                    // takes the quarantine branch and retains the
                    // ttnn::Tensor planes in TtnnNativeCleanupAction until
                    // the later credit-driven dispatch consumes them. No
                    // residual UAF survives the gate.
                    worker_.submit_copy(std::move(task));
                    status_acquired = false;
                } catch (...) {
                    if (status_acquired) {
                        embedding_status_.release(status_slot, true);
                    }
                    bool outcome_erased = false;
                    {
                        std::lock_guard<std::mutex> lock(outcome_mutex_);
                        outcome_erased = embedding_outcomes_.erase(sequence) != 0;
                    }
                    if (outcome_erased) {
                        state_->registry.remove_entries(
                                std::span<const detail::EntryId>(
                                        entries.entries.data(), entries.count));
                        detail::complete_workspace_lease(
                                *state_, captured.workspace_lease, true);
                    }
                    throw;
                }
            });
}

void TtnnQueue::execute_embedding(Task& task) {
    bool submitted = false;
    try {
        std::lock_guard<std::mutex> api_lock(device_->api_mutex());
        ttnn_detail::embedding_planes(
                device_->mesh(), embedding_program_,
                embedding_status_.get(task.embedding_slot),
                *task.embedding_request,
                task.embedding_request->native_workspace, submitted);
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            auto& outcome = embedding_outcomes_.at(task.sequence);
            outcome.native_work_submitted = submitted;
        }
        monotonic_max_store(caller_executed_seq_, task.sequence, std::memory_order_release);
        return;
    } catch (...) {
        const std::exception_ptr submission_failure =
                std::current_exception();
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            auto it = embedding_outcomes_.find(task.sequence);
            if (it != embedding_outcomes_.end()) {
                it->second.retained_failure = submission_failure;
                it->second.native_work_submitted = submitted;
            }
        }
        monotonic_max_store(caller_executed_seq_, task.sequence, std::memory_order_release);
    }
}

void TtnnQueue::complete_embedding_task(
        std::uint64_t sequence, std::exception_ptr failure) {
    EmbeddingOutcome outcome;
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto it = embedding_outcomes_.find(sequence);
        if (it == embedding_outcomes_.end()) {
            complete(sequence, std::move(failure));
            return;
        }
        outcome = std::move(it->second);
        embedding_outcomes_.erase(it);
    }

    detail::FenceResult fence_result = detail::FenceResult::success();
    bool completion_proven = !outcome.native_work_submitted;
    if (outcome.native_work_submitted) {
        fence_result = ttnn_detail::finish_native(*device_);
        completion_proven = fence_result.succeeded && !fence_result.failure;
        if (completion_proven) {
            publish_native_completion(sequence);
        }
    }

    std::exception_ptr operation_failure =
            failure ? failure : outcome.retained_failure;
    if (completion_proven && !operation_failure) {
        std::uint32_t status = 0;
        const auto& slot = embedding_status_.get(outcome.status_slot);
        std::memcpy(&status, slot.packet.bytes.data(), sizeof(status));
        if (status == 1) {
            operation_failure = std::make_exception_ptr(std::invalid_argument(
                    "TTNN embedding index is negative or out of bounds"));
        } else if (status != 0) {
            operation_failure = std::make_exception_ptr(std::runtime_error(
                    "TTNN embedding status packet is invalid"));
        }
    }

    const bool fence_succeeded =
            fence_result.succeeded && !fence_result.failure;
    (void)detail::release_or_invalidate_binary_entries(
            state_->registry, outcome.entries,
            static_cast<bool>(operation_failure) && !completion_proven,
            fence_succeeded);
    detail::complete_workspace_lease(
            *state_, outcome.workspace_lease, completion_proven);
    embedding_status_.release(outcome.status_slot, completion_proven);
    complete(sequence, operation_failure
            ? operation_failure : fence_result.failure);
}

}  // namespace iom

#include "queue_internal.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <utility>

#include "embedding.hpp"
#include "../shared/standard_tiled_embedding.hpp"
#include "../iom_internal.hpp"

namespace iom::sycl_detail {

namespace {

[[nodiscard]] std::uint32_t* embedding_status_cell(
        detail::MetadataSlotPool& pool, std::size_t slot) {
    auto* cell = pool.status_data(slot);
    if (cell == nullptr) {
        // Queue construction requires fixed host-USM status cells. Reaching
        // this branch means that resource contract was not established.
        throw std::bad_alloc();
    }
    return cell;
}

}  // namespace

oid SyclQueue::embedding_impl(const EmbeddingRequest& request) {
    std::lock_guard<std::mutex> submission_lock(
            submission_order_mutex_);
    if (consume_submission_fault(SubmissionFault::state_allocation)) {
        throw std::bad_alloc();
    }
    auto state = std::make_shared<SyclFenceState>();
    if (consume_submission_fault(SubmissionFault::fence_construction)) {
        throw std::bad_alloc();
    }
    detail::Fence fence = build_sycl_fence(state);
    return submit_embedding(
            request, *state_, registry_queue_id_, fence,
            [this, state](
                    std::uint64_t sequence, const EmbeddingRequest& captured,
                    detail::BinaryEntryRegistration entries) {
                Task task;
                task.sequence = sequence;
                task.state = state;
                task.fence = state.get();
                task.embedding_request.emplace(captured);
                task.binary_entries = entries;
                worker_.submit_copy(std::move(task));
            });
}

void SyclQueue::execute_embedding(Task& task) {
    if (consume_submission_fault(SubmissionFault::outcome_insertion)) {
        throw std::bad_alloc();
    }
    const EmbeddingRequest captured = *task.embedding_request;
    {
        const auto [it, inserted] = outcomes_.emplace(
                task.sequence,
                SyclSequenceOutcome{
                        detail::SequenceOutcome{}, task.state,
                        task.binary_entries, captured.workspace_lease,
                        std::nullopt, true});
        if (!inserted) {
            throw std::logic_error(
                    "duplicate SYCL outstanding-work sequence");
        }
    }

    bool native_attempted = false;
    try {
        const auto completion_slot = completion_pool_->try_acquire();
        if (!completion_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_completion_slot(
                *completion_pool_, *completion_slot);
        const auto metadata_slot = metadata_pool_->try_acquire();
        if (!metadata_slot.has_value()) {
            throw detail::AdmissionResourceUnavailable{};
        }
        task.state->set_metadata_slot(*metadata_pool_, *metadata_slot);
        const std::size_t metadata_bytes =
                detail::embedding_metadata_storage_bytes(
                        captured.indices.spec.shape.rank());
        const auto metadata = detail::write_embedding_metadata(
                metadata_pool_->host_data(*metadata_slot),
                metadata_pool_->device_data(*metadata_slot), captured);
        auto* status_cell = embedding_status_cell(
                *metadata_pool_, *metadata_slot);
        void* status_address = detail::WorkspaceValidation::address(
                captured.workspace);
        if (status_address == nullptr) {
            throw std::invalid_argument(
                    "SYCL embedding status workspace is unavailable");
        }
        if (consume_submission_fault(SubmissionFault::first_submit)) {
            throw std::runtime_error(
                    "injected SYCL first-submit failure");
        }
        native_attempted = true;
        const sycl::event metadata_event = queue_.memcpy(
                metadata_pool_->device_data(*metadata_slot),
                metadata_pool_->host_data(*metadata_slot), metadata_bytes);
        const sycl::event reset_event = queue_.submit(
                [&](sycl::handler& handler) {
                    handler.depends_on(metadata_event);
                    handler.memset(
                            static_cast<std::uint32_t*>(status_address),
                            0, sizeof(std::uint32_t));
                });
        const sycl::event gather_event = launch_embedding_words(
                queue_, reset_event,
                static_cast<const unsigned char*>(
                        captured.table.native_handle),
                static_cast<const unsigned char*>(
                        captured.indices.native_handle),
                static_cast<unsigned char*>(captured.out.native_handle),
                static_cast<std::uint32_t*>(status_address), metadata);
        const sycl::event status_event = queue_.submit(
                [&](sycl::handler& handler) {
                    handler.depends_on(gather_event);
                    handler.memcpy(
                            status_cell, status_address,
                            sizeof(std::uint32_t));
                });
        task.state->set_completion_action([status_cell] {
            if (*status_cell != 0) {
                throw std::invalid_argument(
                        "embedding index is out of range");
            }
        });
        task.state->set_event(std::move(status_event));
        if (consume_submission_fault(SubmissionFault::post_launch)) {
            throw std::runtime_error(
                    "injected SYCL post-launch failure");
        }
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (!native_attempted) {
            {
                std::lock_guard<std::mutex> lock(outcome_mutex_);
                outcomes_.erase(task.sequence);
            }
            task.state->mark_completion_proven();
            task.state.reset();
            task.fence = nullptr;
            throw;
        }
        task.state->set_failure(failure);
    }
}

}  // namespace iom::sycl_detail

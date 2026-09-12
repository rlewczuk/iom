#include "queue_internal.hpp"

#include <ttnn/tensor/tensor.hpp>

#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "copy.hpp"
#include "testing_internal.hpp"

namespace iom {

void TtnnQueue::execute_copy(Task& task) {
    const detail::EntryRegistration entries{
            task.source_entry_id, task.destination_entry_id};
    try {
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
#ifdef IOM_ENABLE_TESTING
            ttnn_detail::consume_copy_outcome_insertion_fault();
#endif
            const auto [it, inserted] = outcomes_.emplace(
                    task.sequence,
                    detail::SequenceOutcome{
                            entries.source, entries.destination,
                            nullptr});
            if (!inserted) {
                throw std::logic_error(
                        "duplicate TTNN outstanding-work sequence");
            }
        }
    } catch (...) {
        state_->registry.remove_entry_if_present(
                entries.source,
                task.source->native_handle);
        state_->registry.remove_entry_if_present(
                entries.destination,
                task.destination->native_handle);
        throw;
    }

    if (task.no_op) {
        // An identical-window copy submits no native work. Its
        // registered entries are released by complete_task without a
        // device-wide finish.
        executed_seq_.store(task.sequence, std::memory_order_release);
        return;
    }

    std::lock_guard<std::mutex> api_lock(device_->api_mutex());
    bool any_submitted = false;
    std::exception_ptr submission_failure;
    try {
        const ttnn::Tensor* source_planes =
                static_cast<const ttnn::Tensor*>(
                        task.source->native_handle);
        ttnn::Tensor* destination_planes =
                static_cast<ttnn::Tensor*>(
                        task.destination->native_handle);
        ttnn_detail::copy_planes(
                *task.source, source_planes,
                *task.destination, destination_planes,
                any_submitted);
    } catch (...) {
        submission_failure = std::current_exception();
    }
    if (!submission_failure) {
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            outcomes_.at(task.sequence).native_work_submitted = true;
        }
        // Every plane is submitted and owned; complete_task performs one
        // mesh finish per ready batch of contiguous executed tasks.
        executed_seq_.store(task.sequence, std::memory_order_release);
        return;
    }

    if (!any_submitted) {
        // The failure struck before the first plane reached the mesh:
        // nothing is pending, so ownership rolls back and the exception
        // propagates synchronously with no device work outstanding.
        state_->registry.remove_entry_if_present(
                entries.source,
                task.source->native_handle);
        state_->registry.remove_entry_if_present(
                entries.destination,
                task.destination->native_handle);
        {
            std::lock_guard<std::mutex> lock(outcome_mutex_);
            outcomes_.erase(task.sequence);
        }
        std::rethrow_exception(submission_failure);
    }
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        outcomes_.at(task.sequence).native_work_submitted = true;
    }

    // One or more planes reached the mesh. Drain them synchronously
    // under the API mutex before ownership is removed, so a thrown call
    // has established the terminal result of every submitted command
    // before the owners can be destroyed or reused.
    try {
        ttnn_detail::finish_locked(*device_);
    } catch (...) {
        // The mesh cannot be drained. Retain the failed fenced sequence:
        // the task is published normally, so the caller receives a
        // waitable token whose waits report the retained submission
        // failure, and the registered entries keep protecting the planes
        // until complete_task's finish retry (or the quarantine drain
        // when the owners are destroyed) establishes the terminal result.
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        const auto it = outcomes_.find(task.sequence);
        if (it != outcomes_.end()) {
            it->second.retained_failure =
                    std::move(submission_failure);
        }
        executed_seq_.store(task.sequence, std::memory_order_release);
        return;
    }
    state_->registry.remove_entry_if_present(
            entries.source,
            task.source->native_handle);
    state_->registry.remove_entry_if_present(
            entries.destination,
            task.destination->native_handle);
    {
        std::lock_guard<std::mutex> lock(outcome_mutex_);
        outcomes_.erase(task.sequence);
    }
    std::rethrow_exception(submission_failure);
}

}  // namespace iom

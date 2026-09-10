#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <ttnn/tensor/tensor.hpp>

#include "iom/detail/outstanding_work_registry.hpp"

namespace iom::ttnn_test {
    void fail_next_quarantine_action_for_testing() noexcept;
    bool quarantine_action_fault_consumed_for_testing() noexcept;

    // Fails the next ttnn_detail::copy_planes call just before the chosen
    // plane index is enqueued; planes below it have already reached the
    // mesh. A fault at index zero fails before any plane is submitted.
    void fail_next_copy_planes_submission_for_testing(
            std::size_t plane_index) noexcept;
    bool copy_planes_submission_fault_consumed_for_testing() noexcept;

    // Fails the binary backend's provisional completion-outcome insertion
    // before any native output upload; the common transaction must then
    // remove all registered owners and roll back the sequence.
    void fail_next_binary_outcome_insertion_for_testing() noexcept;
    bool binary_outcome_insertion_fault_consumed_for_testing() noexcept;
    // Fails the next copy's registration phase with std::bad_alloc before
    // any native plane is submitted: either the entry registration itself
    // or the outcome insertion after both entries were registered.
    void fail_next_copy_registration_for_testing() noexcept;
    bool copy_registration_fault_consumed_for_testing() noexcept;
    void fail_next_copy_outcome_insertion_for_testing() noexcept;
    bool copy_outcome_insertion_fault_consumed_for_testing() noexcept;

    // Fails the next `count` mesh-finish attempts made while draining a
    // copy operation (the synchronous drain in the failed submission and
    // the completion retry). Every other mesh finish is unaffected.
    void fail_next_copy_finishes_for_testing(std::size_t count) noexcept;
    bool copy_finish_fault_pending_for_testing() noexcept;

    // Counts mesh-finish attempts routed through the copy-drain path (the
    // same attempts the failure seam above can fault). Batch-completion
    // tests assert one native finish per ready batch through this counter.
    void reset_copy_finish_count_for_testing() noexcept;
    std::uint64_t copy_finish_count_for_testing() noexcept;
    // Fails the next host-transfer (region_from_host or region_to_host)
    // plane submission just before the chosen plane index reaches the mesh;
    // planes below it have already been submitted. A fault at index zero
    // fails before any submission. The fault fires exactly once at the
    // armed index.
    void fail_next_host_transfer_submission_for_testing(
            std::size_t plane_index) noexcept;
    bool host_transfer_submission_fault_consumed_for_testing() noexcept;

    // Fails the next retained host-staging allocation (an upload or
    // download slot creation or growth) with std::bad_alloc, leaving the
    // slot untouched; the armed fault is consumed only when an allocation
    // is actually attempted.
    void fail_next_host_transfer_staging_allocation_for_testing() noexcept;
    bool host_transfer_staging_allocation_fault_consumed_for_testing() noexcept;

    // Number of fresh retained host-staging allocations made so far (slot
    // creations and growths, including replacements after a discarded
    // slot).
    std::size_t host_transfer_staging_allocation_count_for_testing() noexcept;
}
namespace iom::ttnn_detail {


class TtnnNativeCleanupAction final : public detail::CleanupAction {
public:
    TtnnNativeCleanupAction(
            std::vector<ttnn::Tensor> planes, std::function<void()> finish)
            : planes_(std::move(planes)), finish_(std::move(finish)) {}

    void run() noexcept override {
        if (completed_) {
            return;
        }
        try {
            if (finish_) {
                finish_();
            }
            planes_.clear();
            completed_ = true;
        } catch (...) {
            if (failure_ == nullptr) {
                failure_ = std::current_exception();
            }
        }
    }

    [[nodiscard]] bool completed() const noexcept override {
        return completed_;
    }

    [[nodiscard]] bool failed() const noexcept override {
        return static_cast<bool>(failure_);
    }

    [[nodiscard]] std::exception_ptr failure() const noexcept override {
        return failure_;
    }

private:
    std::vector<ttnn::Tensor> planes_;
    std::function<void()> finish_;
    std::exception_ptr failure_;
    bool completed_ = false;
};

}  // namespace iom::ttnn_detail

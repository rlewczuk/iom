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
}
namespace iom::ttnn_detail {


class TtnnNativeCleanupAction final : public detail::CleanupAction {
public:
    TtnnNativeCleanupAction(
            std::vector<ttnn::Tensor> planes, std::function<void()> finish)
            : planes_(std::move(planes)), finish_(std::move(finish)) {}

    void run() noexcept override {
        if (attempted_) {
            return;
        }
        attempted_ = true;
        try {
            if (finish_) {
                finish_();
            }
        } catch (...) {
            failure_ = std::current_exception();
        }
        planes_.clear();
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
    bool attempted_ = false;
};

}  // namespace iom::ttnn_detail

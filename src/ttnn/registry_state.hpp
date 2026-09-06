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

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

struct TtnnRegistryState {
    detail::OutstandingWorkRegistry registry;
    detail::Quarantine quarantine;
    detail::EntryId next_entry_id = 1;
    detail::QueueId next_queue_id = 1;
};

inline detail::QueueId allocate_queue_id(TtnnRegistryState& state) {
    return detail::allocate_registry_queue_id(state.next_queue_id);
}

inline detail::EntryRegistration register_copy_entries(
        TtnnRegistryState& state, detail::QueueId queue_id,
        std::uint64_t sequence, void* source, void* destination,
        const detail::Fence& fence) {
    return detail::register_registry_entries(
            state.registry, state.next_entry_id, queue_id, sequence, source,
            destination, fence);
}

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

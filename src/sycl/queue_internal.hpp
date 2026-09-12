#pragma once

#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

#include "device_internal.hpp"
#include "queue_support.hpp"
#include "scalar_add.hpp"

namespace iom::sycl_detail {

class SyclQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence = 0;
        bool no_op = false;
        std::optional<CopyRequest> copy_request;
        std::shared_ptr<SyclFenceState> state;
        void* fence = nullptr;
        detail::EntryRegistration copy_entries;
        std::optional<BinaryRequest> binary_request;
        detail::BinaryEntryRegistration binary_entries;
    };

    struct SyclSequenceOutcome {
        detail::SequenceOutcome common;
        std::shared_ptr<SyclFenceState> state;
        std::optional<detail::BinaryEntryRegistration> binary_entries;
        detail::WorkspaceLease workspace_lease;
    };

public:
#ifdef IOM_ENABLE_TESTING
    [[nodiscard]] detail::MetadataSlotPool&
            metadata_pool_for_testing() noexcept {
        return *metadata_pool_;
    }
    [[nodiscard]] SyclCompletionPool&
            completion_pool_for_testing() noexcept {
        return *completion_pool_;
    }
#endif

    SyclQueue(
            const Device& device,
            detail::QueueResourceProvider& resource_provider,
            const sycl::context& context, const sycl::device& native_device,
            detail::RegistryState& state);
    ~SyclQueue() override;

    oid copy_impl(
            const TensorView& source,
            TensorView& destination) override;
    oid binary_impl(const BinaryRequest& request) override;

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "SYCL";
    }

    [[nodiscard]] WorkspaceRequirements binary_workspace_requirements(
            const BinaryRequest& request) override;

private:
    void execute(Task& task);
    void execute_binary(Task& task);
    void complete_task(
            std::uint64_t sequence, std::exception_ptr callback_failure);

    static std::size_t binary_view_staging_bytes(
            const BinaryRequest& captured, const BinaryViewSnapshot& view);
    static std::size_t checked_add_local(
            std::size_t lhs, std::size_t rhs, const char* what);
    static std::size_t checked_mul_local(
            std::size_t lhs, std::size_t rhs, const char* what);
    static std::size_t align_up_checked(
            std::size_t value, std::size_t alignment);
    static std::size_t checked_binary_view_staging_bytes(
            const BinaryRequest& captured, const BinaryViewSnapshot& view);

    template <detail::scalar_add_detail::BinaryOp Op>
    static void binary_elements_impl(
            const BinaryRequest& captured, const unsigned char* lhs_storage,
            const unsigned char* rhs_storage, unsigned char* out_storage);

    static void binary_elements(
            const BinaryRequest& captured, const unsigned char* lhs_storage,
            const unsigned char* rhs_storage, unsigned char* out_storage);
    static void free_binary_host_staging(
            void* staging, const sycl::context& context) noexcept;

    static sycl::queue make_queue_with_fault_check(
            const sycl::context& context, const sycl::device& native_device);

    const Device* device_;
    detail::RegistryState* state_;
    detail::QueueResourceProvider* resource_provider_;
    detail::QueueId registry_queue_id_;
    // The fixed partition lease is reserved before the native queue is
    // constructed (member declaration order).
    std::shared_ptr<detail::MetadataSlotPool> metadata_pool_;
    std::shared_ptr<SyclCompletionPool> completion_pool_;
    sycl::queue queue_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SyclSequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace iom::sycl_detail

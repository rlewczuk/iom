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

// Immutable device capability of the native `BF16` linear specialization,
// computed once by the queue from its exact native device and published as
// `SyclQueue::bf16_linear_supported_`. It is defined next to the BF16 kernels
// in `queue_linear.cpp`; a device that cannot prove the queried subgroup-16
// BF16/BF16/FP32 facility is `Unsupported` rather than emulated.
[[nodiscard]] bool bf16_linear_device_capable(
        const sycl::device& device) noexcept;

class SyclQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence = 0;
        bool no_op = false;
        std::optional<CopyRequest> copy_request;
        std::optional<BinaryRequest> binary_request;
        std::optional<EmbeddingRequest> embedding_request;
        std::optional<CacheAppendRequest> cache_append_request;
        std::shared_ptr<SyclFenceState> state;
        void* fence = nullptr;
        detail::EntryRegistration copy_entries;
        detail::BinaryEntryRegistration binary_entries;
        detail::BinaryEntryRegistration cache_append_entries;
        // Immutable RMS normalization request captured by admission, plus the
        // common owner registrations retained until proven completion. RMS
        // normalization consumes no `RawWorkspace`, so no lease is carried.
        std::optional<RmsnormRequest> rmsnorm_request;
        detail::BinaryEntryRegistration rmsnorm_entries;
        // Immutable linear projection request captured by admission, plus the
        // common owner registrations retained until proven completion. The
        // twenty scalar leaves consume no `RawWorkspace`, while the native
        // `BF16` specialization carries the positive product-scratch lease
        // captured with the request.
        std::optional<LinearRequest> linear_request;
        detail::BinaryEntryRegistration linear_entries;
        // Immutable split-half RoPE request captured by admission, plus the
        // common owner registrations retained until proven completion.
        std::optional<RopeRequest> rope_request;
        detail::BinaryEntryRegistration rope_entries;
    };

    struct SyclSequenceOutcome {
        detail::SequenceOutcome common;
        std::shared_ptr<SyclFenceState> state;
        std::optional<detail::BinaryEntryRegistration> binary_entries;
        detail::WorkspaceLease workspace_lease;
        std::optional<detail::BinaryEntryRegistration> rmsnorm_entries;
        bool is_embedding = false;
        // Trailing so every existing aggregate initialization of this outcome
        // keeps its own field order. Linear and RoPE register their
        // deduplicated owner sets; both currently consume no raw workspace.
        std::optional<detail::BinaryEntryRegistration> linear_entries;
        std::optional<detail::BinaryEntryRegistration> rope_entries;
        std::optional<detail::BinaryEntryRegistration> cache_append_entries;
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
    // Immutable capability the native `BF16` linear specialization was
    // constructed with, so a driver can assert that the port's own decision
    // agrees with the device facts it queries independently.
    [[nodiscard]] bool bf16_linear_supported_for_testing() const noexcept {
        return bf16_linear_supported_;
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
    [[nodiscard]] WorkspaceRequirements
            cache_append_workspace_requirements(
                    const CacheAppendRequest& request) override;
    oid cache_append_impl(const CacheAppendRequest& request) override;
    oid embedding_impl(const EmbeddingRequest& request) override;
    oid rmsnorm_impl(const RmsnormRequest& request) override;
    oid linear_impl(const LinearRequest& request) override;
    oid rope_impl(const RopeRequest& request) override;
    [[nodiscard]] WorkspaceRequirements
            rope_workspace_requirements(const RopeRequest& request) override;


    /**
     * Pure capability decision and exact raw-workspace requirement of the
     * implemented linear leaf set: the twenty non-`BF16` scalar leaves report
     * the `{0, 1}` zero-scratch path, `F64` is queueable only when the
     * selected device reports `sycl::aspect::fp64`, and `BF16` is queueable
     * only when the selected device proves the queried subgroup-16
     * BF16/BF16/FP32 `joint_matrix` facility, in which case it reports the
     * checked alignment-32 `A32(P*pad16(R)*pad16(O)*4)` product scratch. The
     * common facade consults this hook for both the call and the pure
     * requirement query, so an unimplemented leaf, an absent aspect, and an
     * absent device matrix facility are `Unsupported` before any queue effect.
     */
    [[nodiscard]] WorkspaceRequirements linear_workspace_requirements_impl(
            const TensorView& x, const TensorView& w, const TensorView& out,
            std::size_t s, std::size_t R, LinearOutputLayout layout,
            std::size_t H, std::size_t D) override;

    // Immutable device capability of the implemented RMS normalization leaf
    // set: the eight non-`F64` applicable float leaves are always queueable,
    // and `F64` is queueable exactly when the selected device reports
    // `sycl::aspect::fp64`. The common facade consults this predicate for
    // both the call and the pure requirement query, so an absent aspect is
    // `Unsupported` before any queue effect.
    [[nodiscard]] bool rmsnorm_supported(
            DataType data_type) const override;

    [[nodiscard]] std::string_view backend_label() const noexcept override {
        return "SYCL";
    }

    [[nodiscard]] WorkspaceRequirements binary_workspace_requirements(
            const BinaryRequest& request) override;

    // Pure `{32, 32}` workspace requirement the SYCL gather reports: the
    // caller-supplied range owns one uint32 status word plus reserved
    // padding. Common validation precedes this hook so capability-stage
    // `Unsupported` results never reach the host-side query.
    [[nodiscard]] WorkspaceRequirements embedding_workspace_requirements_impl(
            const TensorView& table, const TensorView& indices,
            const TensorView& out) override {
        (void)table; (void)indices; (void)out;
        return {32, 32};
    }

private:
    void execute(Task& task);
    void execute_binary(Task& task);
    void execute_cache_append(Task& task);
    void execute_embedding(Task& task);
    void execute_rmsnorm(Task& task);
    void execute_linear(Task& task);
    void execute_rope(Task& task);
    // Immutable capability predicate of the linear scalar leaf set, shared by
    // the pure requirement query and the admitted execution hook so the query
    // and the call can never disagree about `F64`.
    [[nodiscard]] bool linear_leaf_supported(
            DataType data_type) const noexcept;
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
    // Immutable capability of the selected native device, published to the
    // common RMS normalization facade through `rmsnorm_supported`.
    const bool fp64_supported_ = false;
    // Immutable capability of the same exact device for the native `BF16`
    // linear specialization: the Intel matrix aspect, subgroup 16, and the
    // BF16/BF16/FP32 combination family the row decomposition queues. It is
    // computed once from device facts, is never an echo of a compilation or a
    // bounded sample, and gates both the pure requirement query and the
    // submission before any queue effect.
    const bool bf16_linear_supported_ = false;
    sycl::queue queue_;
    std::mutex submission_order_mutex_;
    std::mutex outcome_mutex_;
    std::map<std::uint64_t, SyclSequenceOutcome> outcomes_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace iom::sycl_detail

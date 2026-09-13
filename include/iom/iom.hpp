#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <initializer_list>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <span>

#include "detail/outstanding_work_registry.hpp"
#include "detail/staged_worker.hpp"
#include "oid.hpp"
#include "tensor.hpp"

namespace iom {

    namespace detail {
        // A backend dispatch may defer an accepted FIFO head when a fixed
        // completion or metadata resource is quarantined. This is not a
        // terminal operation failure and must leave the node parked.
        class AdmissionResourceUnavailable final : public std::exception {
        public:
            [[nodiscard]] const char* what() const noexcept override {
                return "fixed admission resource is unavailable";
            }
        };
    }  // namespace detail

    namespace detail {
        /**
         * Shared reusable raw-workspace validation (leaf 05). Centralizes
         * the required-capacity, base/range alignment, checked-offset,
         * exact-Device-identity, exact-live-owner-identity, and
         * workspace-versus-operand/output overlap rules that tasks 07 and
         * 11 enforce on borrowed workspace arguments. Every rejection is
         * invalid input: `std::invalid_argument`; range-end overflow
         * reports `std::overflow_error`. Pure: no allocation, registry,
         * lease, token/queue resource, or backend effect.
         */
        class WorkspaceValidation {
        public:
            /**
             * Validates `workspace` as the scratch range for one operation
             * on `device` needing `required_capacity` bytes at
             * `required_alignment`. A zero capacity accepts any view (the
             * facades default to the empty view exactly for this case); a
             * positive capacity additionally requires a non-empty view of
             * a live owner created by this exact device, sufficient
             * capacity, a 32-byte-aligned base meeting the required
             * alignment, and a range disjoint from every operand/output
             * storage range. Returns the unchanged view.
             */
            [[nodiscard]] static RawWorkspaceView validated(
                    const Device& device,
                    const RawWorkspaceView& workspace,
                    std::size_t required_capacity,
                    std::size_t required_alignment,
                    std::span<const TensorView> operands);
            [[nodiscard]] static void* address(
                    const RawWorkspaceView& workspace) noexcept;
        };

    }  // namespace detail
    /**
     * One in-order asynchronous operation queue over caller-created tensor
     * views. Every OID-returning operation is a common `noexcept` facade:
     * it validates, maps failures, encodes tokens, and registers lifetimes
     * before backend effects or token acceptance. Backend hooks cannot bypass
     * this protocol.
     *
     * `oid` is signed `int64_t`: -1 InvalidArgument, -2 Unsupported, -3
     * Overflow, -4 ResourceExhausted, -5 DeviceError, and -6 InternalError.
     * Negative results are synchronous errors, positive values are accepted
     * tokens, and zero is invalid. `add`, `mul`, `sub`, and `div` each have
     * exactly three views and are the binary operation support signals.
     * Every full tensor, view, and binary result shape must have rank two
     * through eight; a rank outside that interval is an invalid argument
     * and maps through OidError::InvalidArgument (OID -1) before any
     * sequence, token, registry entry, metadata upload, or backend effect.
     * Synchronous failures never cross the facade; `wait` throws for invalid
     * tokens and retained post-acceptance failures.
     */
    class DeviceOps {
    public:
        DeviceOps(const DeviceOps&) = delete;
        DeviceOps& operator=(const DeviceOps&) = delete;
        DeviceOps(DeviceOps&&) = delete;
        DeviceOps& operator=(DeviceOps&&) = delete;
        virtual ~DeviceOps();
        /**
         * Observe accepted work in queue order. A negative, zero, foreign,
         * future, skipped/reserved-but-never-submitted, or otherwise
         * unsubmitted value throws std::invalid_argument immediately; a
         * skipped value remains invalid after later completion. Accepted
         * tokens are repeat-waitable, and retained asynchronous failures are
         * rethrown by every wait.
         */
        void wait(oid token);
        /**
         * Common `noexcept` OID facades for three-view binary operations.
         * Validation and lifetime ownership are shared by all operations.
         * All three operand/output full specs and the computed broadcast
         * result must have rank two through eight; invalid rank returns
         * OidError::InvalidArgument with no sequence, token, registry,
         * metadata, or backend effect.
         */
        oid copy(const TensorView& source, TensorView& destination) noexcept;
        oid add(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        oid mul(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        oid sub(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        oid div(const TensorView& lhs, const TensorView& rhs,
                TensorView& out,
                RawWorkspaceView workspace = {}) noexcept;
        /**
         * Pure deterministic raw-workspace requirement queries for the
         * three-view binary operations (leaf 05). Each runs the exact
         * validation of its binary operation — operation, all three
         * views, exact device identity, specifications, rank and view
         * bounds, broadcasting, alias rules, and backend capability —
         * and then reports, with no allocation, registration, lease,
         * token/queue resource, metadata upload, submission, and no
         * dependence on free data-arena capacity, fragmentation, queue
         * occupancy, or completion state:
         * CPU, TTNN, CUDA, and ROCm report `{0, 1}`; SYCL reports its
         * checked whole-plane staging sum at alignment 32. Validation
         * failures surface as the corresponding exception
         * (`std::invalid_argument`, `std::overflow_error`, unsupported
         * operation), never through the OID error mapping.
         */
        [[nodiscard]] WorkspaceRequirements add_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] WorkspaceRequirements mul_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] WorkspaceRequirements sub_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] WorkspaceRequirements div_workspace_requirements(
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        oid silu(const TensorView& x, TensorView& y) noexcept;
        oid linear(const TensorView& x, const TensorView& w,
                   TensorView& y) noexcept;
        oid rmsnorm(const TensorView& x, TensorView& y, const TensorView& w,
                    float eps, size_t dim) noexcept;
        oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
                 size_t n_heads, size_t n_kv_heads, size_t head_dim,
                 TensorView& attn_out) noexcept;

    protected:
        /*
         * Backend hooks are implementation extension points only. Public
         * facades retain common validation, error mapping, token encoding,
         * queue ordering, and lifetime registration.
         */
        enum class BinaryOperation { Add, Mul, Sub, Div };
        struct BinaryViewSnapshot {
            TensorSpec spec;
            const Device* device_identity;
            const Tensor* owner_identity;
            void* native_handle;
            std::size_t plane_offset;
            std::vector<std::size_t> plane_strides;
            std::vector<std::size_t> logical_plane_strides;
            bool broadcast_rows;
            bool broadcast_columns;
            bool broadcasts;
        };
        struct CopyViewSnapshot {
            TensorSpec spec;
            const Device* device_identity;
            const Tensor* owner_identity;
            void* native_handle;
            std::size_t plane_offset;
            std::vector<std::size_t> plane_strides;
        };
        [[nodiscard]] static CopyViewSnapshot snapshot_copy_view(
                const TensorView& view);
        struct BinaryRequest {
            BinaryOperation operation;
            BinaryViewSnapshot lhs;
            BinaryViewSnapshot rhs;
            BinaryViewSnapshot out;
            TensorShape result_shape;
            RawWorkspaceView workspace;
            WorkspaceRequirements workspace_requirements;
            detail::WorkspaceLease workspace_lease;
            BinaryRequest(
                    BinaryOperation operation_,
                    BinaryViewSnapshot lhs_, BinaryViewSnapshot rhs_,
                    BinaryViewSnapshot out_, TensorShape result_shape_,
                    RawWorkspaceView workspace_ = {},
                    WorkspaceRequirements workspace_requirements_ = {},
                    detail::WorkspaceLease workspace_lease_ = {})
                : operation(operation_), lhs(std::move(lhs_)),
                  rhs(std::move(rhs_)), out(std::move(out_)),
                  result_shape(std::move(result_shape_)),
                  workspace(workspace_),
                  workspace_requirements(workspace_requirements_),
                  workspace_lease(workspace_lease_) {}
            BinaryRequest(const BinaryRequest&) = default;
            BinaryRequest& operator=(const BinaryRequest&) = delete;
            BinaryRequest(BinaryRequest&& other) noexcept
                : operation(other.operation), lhs(std::move(other.lhs)),
                  rhs(std::move(other.rhs)), out(std::move(other.out)),
                  result_shape(std::move(other.result_shape)),
                  workspace(other.workspace),
                  workspace_requirements(other.workspace_requirements),
                  workspace_lease(other.workspace_lease) {}
        };

        // Immutable host-side copy descriptor retained by admission. It
        // contains value-copied view metadata and stable owner identities; no
        // callback may retain the caller's borrowed view object.
        struct CopyRequest {
            CopyViewSnapshot source;
            CopyViewSnapshot destination;
            bool no_op = false;
            CopyRequest(
                    const TensorView& source_, const TensorView& destination_,
                    bool no_op_)
                : source(snapshot_copy_view(source_)),
                  destination(snapshot_copy_view(destination_)),
                  no_op(no_op_) {}
        };

        DeviceOps();
        explicit DeviceOps(const Device& device);
        virtual oid copy_impl(
                const TensorView& source, TensorView& destination);
        virtual oid binary_impl(const BinaryRequest& request);
        virtual oid silu_impl(const TensorView& x, TensorView& y);
        virtual oid linear_impl(const TensorView& x, const TensorView& w,
                                TensorView& y);
        virtual oid rmsnorm_impl(const TensorView& x, TensorView& y,
                                 const TensorView& w, float eps, size_t dim);
        virtual oid sdpa_impl(const TensorView& q, const TensorView& k,
                              const TensorView& v, size_t n_heads,
                              size_t n_kv_heads, size_t head_dim,
                              TensorView& attn_out);
        /**
         * Backend hook behind the four *_workspace_requirements queries.
         * Receives an already fully validated request snapshot and must
         * stay pure: no allocation beyond the returned value, no
         * registration, lease, token/queue resource, or backend effect.
         */
        [[nodiscard]] virtual WorkspaceRequirements
                binary_workspace_requirements(const BinaryRequest& request);
        [[nodiscard]] const Device& queue_device() const;
        virtual void fence_through_sequence(
                std::uint64_t sequence) noexcept;
        void record_post_completion_failure(
                std::uint64_t sequence, std::exception_ptr failure);
        [[nodiscard]] static BinaryRequest validate_binary(
                const Device& device, BinaryOperation operation,
                const TensorView& lhs, const TensorView& rhs,
                const TensorView& out);
        [[nodiscard]] static BinaryViewSnapshot snapshot_binary_view(
                const TensorView& view,
                std::span<const std::size_t> result_dimensions);
        static void validate_copy(
                const Device& device, const TensorView& source,
                const TensorView& destination);
        static void validate_views(const Device& device,
                                   std::initializer_list<const TensorView*> views);
        [[nodiscard]] static bool identical_window(
                const TensorView& source, const TensorView& destination);
        [[nodiscard]] virtual std::string_view backend_label() const noexcept {
            return "unknown";
        }
        [[nodiscard]] static std::runtime_error unsupported(
                std::string_view backend, std::string_view operation);

        // Admission callbacks are prepared synchronously, then retained in
        // an immutable FIFO node until a native credit is available.
        oid submit_prepared(
                std::function<void(std::uint64_t)> prepare,
                std::function<void(std::uint64_t)> dispatch,
                std::function<void()> rollback = {});
        template <typename QueueWork>
        oid submit(QueueWork queue_work) {
            auto work = std::make_shared<QueueWork>(std::move(queue_work));
            return submit_prepared(
                    {}, [work](std::uint64_t sequence) {
                        (*work)(sequence);
                    });
        }
        template <typename QueueWork>
        oid submit_copy(
                const TensorView& source, const TensorView& destination,
                detail::RegistryState& state, detail::QueueId queue_id,
                const detail::Fence& fence, QueueWork queue_work) {
            struct Prepared {
                CopyRequest request;
                detail::EntryRegistration entries;
                std::function<void(
                        std::uint64_t, const CopyRequest&,
                        detail::EntryRegistration)> work;
                Prepared(
                        const TensorView& source_,
                        const TensorView& destination_, bool no_op_,
                        QueueWork work_)
                    : request(source_, destination_, no_op_),
                      work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    source, destination, identical_window(source, destination),
                    std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, fence](
                            std::uint64_t sequence) {
                        prepared->entries = detail::register_copy_entries(
                                state, queue_id, sequence,
                                prepared->request.source.native_handle,
                                prepared->request.destination.native_handle,
                                fence);
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.source != 0) {
                            state.registry.remove_entry_if_present(
                                    prepared->entries.source,
                                    prepared->request.source.native_handle);
                        }
                        if (prepared->entries.destination != 0) {
                            state.registry.remove_entry_if_present(
                                    prepared->entries.destination,
                                    prepared->request.destination.native_handle);
                        }
                    });
        }

        template <typename QueueWork>
        oid submit_binary(
                const BinaryRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            struct Prepared {
                BinaryRequest request;
                detail::WorkspaceLease workspace_lease;
                detail::BinaryEntryRegistration entries;
                std::function<void(
                        std::uint64_t, const BinaryRequest&,
                        detail::BinaryEntryRegistration)> work;
                Prepared(
                        const BinaryRequest& request_, QueueWork work_)
                    : request(request_), work(std::move(work_)) {}
            };
            auto prepared = std::make_shared<Prepared>(
                    request, std::move(queue_work));
            return submit_prepared(
                    [prepared, &state, queue_id, fence](
                            std::uint64_t sequence) {
                        try {
                            if (prepared->request.workspace_requirements.bytes
                                    != 0) {
                                prepared->workspace_lease =
                                        detail::acquire_workspace_lease(
                                                state,
                                                prepared->request.workspace
                                                        .owner_identity(),
                                                prepared->request.workspace
                                                        .range_address(),
                                                prepared->request.workspace
                                                        .byte_size(),
                                                sequence, queue_id, fence);
                                prepared->request.workspace_lease =
                                        prepared->workspace_lease;
                            }
                            const std::array<detail::BinaryOwnerRegistration, 3>
                                    owners{{
                                            {prepared->request.lhs
                                                     .owner_identity,
                                             prepared->request.lhs.native_handle},
                                            {prepared->request.rhs
                                                     .owner_identity,
                                             prepared->request.rhs.native_handle},
                                            {prepared->request.out
                                                     .owner_identity,
                                             prepared->request.out.native_handle},
                                    }};
                            prepared->entries = detail::register_binary_entries(
                                    state, queue_id, sequence, owners, fence);
                        } catch (...) {
                            if (prepared->entries.count != 0) {
                                state.registry.remove_entries(
                                        std::span<const detail::EntryId>(
                                                prepared->entries.entries.data(),
                                                prepared->entries.count));
                                prepared->entries = {};
                            }
                            if (prepared->workspace_lease.entry_id != 0) {
                                detail::complete_workspace_lease(
                                        state, prepared->workspace_lease, true);
                                prepared->workspace_lease = {};
                                prepared->request.workspace_lease = {};
                            }
                            throw;
                        }
                    },
                    [prepared](std::uint64_t sequence) {
                        prepared->work(
                                sequence, prepared->request,
                                prepared->entries);
                    },
                    [prepared, &state] {
                        if (prepared->entries.count != 0) {
                            state.registry.remove_entries(
                                    std::span<const detail::EntryId>(
                                            prepared->entries.entries.data(),
                                            prepared->entries.count));
                        }
                        if (prepared->workspace_lease.entry_id != 0) {
                            detail::complete_workspace_lease(
                                    state, prepared->workspace_lease, true);
                        }
                    });
        }

        void complete(std::uint64_t sequence,
                      std::exception_ptr failure = nullptr);
        void commit_failure(std::uint64_t sequence,
                            std::exception_ptr failure);
        void seek_next_sequence(std::uint64_t next_sequence);
        // Backend destructors call this before their worker/native members
        // disappear. It closes acceptance and drains accepted FIFO nodes.
        void close_and_drain() noexcept;

    private:
        [[nodiscard]] oid submit_binary_operation(
                BinaryOperation operation, const TensorView& lhs,
                const TensorView& rhs, TensorView& out,
                RawWorkspaceView workspace) noexcept;
        [[nodiscard]] oid invoke(oid result) noexcept;
        [[nodiscard]] oid invoke_failure(std::exception_ptr failure) noexcept;
        [[nodiscard]] static oid map_failure(
                std::exception_ptr failure) noexcept;
        [[nodiscard]] static std::uint8_t lease_queue_id();
        static void release_queue_id(std::uint8_t queue_id) noexcept;
        [[nodiscard]] oid encode_token(std::uint64_t sequence) const noexcept;
        const Device* device_ = nullptr;
        std::uint8_t queue_id_ = 0;
        static constexpr std::uint64_t kSequenceBits = 55;
        static constexpr std::uint64_t kSequenceMask =
                (std::uint64_t{1} << kSequenceBits) - 1;
        static constexpr std::uint64_t kMaxSequence = kSequenceMask;
        std::map<std::uint64_t, std::uint64_t> skipped_sequences_;
        std::uint64_t next_sequence_ = 1;
        std::uint64_t completed_ = 0;
        std::mutex completion_mutex_;
        std::condition_variable completion_cv_;
        std::map<std::uint64_t, std::exception_ptr> failures_;
        std::map<std::uint64_t, std::exception_ptr> pending_failures_;
        struct AdmissionNode {
            std::function<void(std::uint64_t)> dispatch;
            // Resources prepared before admission are released if queue
            // teardown drains this parked node before dispatch.
            std::function<void()> rollback;
            bool executing = false;
        };
        void pump_admission(
                std::exception_ptr* synchronous_failure = nullptr,
                std::uint64_t synchronous_sequence = 0) noexcept;
        void abandon_sequence(std::uint64_t sequence) noexcept;
        std::map<std::uint64_t, AdmissionNode> admission_nodes_;
        std::deque<std::uint64_t> admission_fifo_;
        std::size_t admission_credits_ = 0;
        std::size_t admission_capacity_ = 16;
        bool admission_pumping_ = false;
        bool admission_closing_ = false;
        bool queue_slot_reserved_ = false;
    };

    class Block {
    public:
        virtual ~Block() = default;
        virtual void forward(const Tensor& x, Tensor& y) = 0;
    };
}

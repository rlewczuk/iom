#pragma once

#include <array>
#include <initializer_list>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <span>

#include "detail/outstanding_work_registry.hpp"


#include "oid.hpp"

#include "tensor.hpp"

namespace iom {

    namespace detail {

        /**
         * One backend-neutral staged submission worker. Tasks are executed
         * after staging and completed in submission order.
         */
        template <typename Task>
        class StagedWorker {
        public:
            using Execute = std::function<void(Task&)>;
            using FenceComplete = std::function<void(void*)>;
            using FenceDestroy = std::function<void(void*)>;
            using Complete =
                    std::function<void(std::uint64_t, std::exception_ptr)>;

            enum class PublishPolicy { Splice, CompleteOnThrow };

            struct Callbacks {
                Execute execute;
                FenceComplete fence_complete;
                FenceDestroy fence_destroy;
                Complete complete;
            };

            explicit StagedWorker(
                    Callbacks callbacks, PublishPolicy publish_policy)
                    : callbacks_(std::move(callbacks)),
                      publish_policy_(publish_policy) {}

            ~StagedWorker() { shutdown_and_drain(); }

            StagedWorker(const StagedWorker&) = delete;
            StagedWorker& operator=(const StagedWorker&) = delete;
            StagedWorker(StagedWorker&&) = delete;
            StagedWorker& operator=(StagedWorker&&) = delete;

            void start() {
                if (worker_.joinable()) {
                    throw std::logic_error(
                            "staged worker has already started");
                }
                worker_ = std::thread([this] { run(); });
            }

            void submit_copy(Task task) {
                typename std::list<Task>::iterator staged;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (shutdown_) {
                        throw std::logic_error(
                                "staged worker is shut down");
                    }
                    staged_.push_back(std::move(task));
                    staged = staged_.end();
                    --staged;
                }


                try {
                    callbacks_.execute(*staged);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        staged_.erase(staged);
                    }
                    throw;
                }

                std::exception_ptr publish_failure;
                std::uint64_t publish_failure_sequence = 0;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (publish_policy_ == PublishPolicy::Splice) {
                        tasks_.splice(tasks_.end(), staged_, staged);
                    } else {
                        publish_failure_sequence = staged->sequence;
                        try {
                            tasks_.push_back(std::move(*staged));
                            staged_.erase(staged);
                        } catch (...) {
                            staged_.erase(staged);
                            publish_failure = std::current_exception();
                        }
                    }
                }
                if (publish_failure) {
                    callbacks_.complete(
                            publish_failure_sequence, publish_failure);
                    std::rethrow_exception(publish_failure);
                }
                completion_.notify_one();
            }

            void shutdown_and_drain() {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    shutdown_ = true;
                }
                completion_.notify_all();
                if (worker_.joinable()) {
                    worker_.join();
                }
                std::list<Task> staged;
                std::list<Task> tasks;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    staged.splice(staged.end(), staged_);
                    tasks.splice(tasks.end(), tasks_);
                }
                drain_list(staged);
                drain_list(tasks);
            }

        private:
            void process(Task&& task, bool wait_for_fence) {
                std::exception_ptr failure;
                if (wait_for_fence && task.fence != nullptr) {
                    try {
                        callbacks_.fence_complete(task.fence);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                }
                if (task.fence != nullptr) {
                    try {
                        callbacks_.fence_destroy(task.fence);
                    } catch (...) {
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                }
                callbacks_.complete(task.sequence, std::move(failure));
            }

            void drain_list(std::list<Task>& list) {
                while (!list.empty()) {
                    Task current(std::move(list.front()));
                    list.pop_front();
                    process(std::move(current), false);
                }
            }

            void run() {
                for (;;) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    completion_.wait(lock, [this] {
                        return shutdown_ || !tasks_.empty();
                    });
                    if (shutdown_) {
                        while (!tasks_.empty()) {
                            Task current(std::move(tasks_.front()));
                            tasks_.pop_front();
                            lock.unlock();
                            process(std::move(current), false);
                            lock.lock();
                        }
                        return;
                    }
                    Task current(std::move(tasks_.front()));
                    tasks_.pop_front();
                    lock.unlock();
                    process(std::move(current), true);
                }
            }

            Callbacks callbacks_;
            PublishPolicy publish_policy_;
            std::mutex mutex_;
            std::condition_variable completion_;
            std::list<Task> staged_;
            std::list<Task> tasks_;
            bool shutdown_ = false;
            std::thread worker_;
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
                TensorView& out) noexcept;
        oid mul(const TensorView& lhs, const TensorView& rhs,
                TensorView& out) noexcept;
        oid sub(const TensorView& lhs, const TensorView& rhs,
                TensorView& out) noexcept;
        oid div(const TensorView& lhs, const TensorView& rhs,
                TensorView& out) noexcept;

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

        struct BinaryRequest {
            BinaryOperation operation;
            BinaryViewSnapshot lhs;
            BinaryViewSnapshot rhs;
            BinaryViewSnapshot out;
            TensorShape result_shape;
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
         * The base reports `{0, 1}`; SYCL overrides with its checked
         * whole-plane staging sum.
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

        template <typename QueueWork>
        oid submit(QueueWork queue_work) {
            std::uint64_t sequence = 0;
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (next_sequence_ > kMaxSequence) {
                    throw std::overflow_error(
                        "DeviceOps 55-bit submission sequence is exhausted");
                }
                sequence = next_sequence_++;
            }
            try {
                queue_work(sequence);
            } catch (...) {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                if (next_sequence_ == sequence + 1 && completed_ < sequence) {
                    pending_failures_.erase(sequence);
                    next_sequence_ = sequence;
                }
                throw;
            }
            return encode_token(sequence);
        }

        template <typename QueueWork>
        oid submit_binary(
                const BinaryRequest& request, detail::RegistryState& state,
                detail::QueueId queue_id, const detail::Fence& fence,
                QueueWork queue_work) {
            return submit([&](std::uint64_t sequence) {
                const std::array<detail::BinaryOwnerRegistration, 3> owners{{
                        {request.lhs.owner_identity, request.lhs.native_handle},
                        {request.rhs.owner_identity, request.rhs.native_handle},
                        {request.out.owner_identity, request.out.native_handle},
                }};
                detail::BinaryEntryRegistration entries =
                        detail::register_binary_entries(
                                state, queue_id, sequence, owners, fence);
                try {
                    queue_work(sequence, request, entries);
                } catch (...) {
                    state.registry.remove_entries(
                            std::span<const detail::EntryId>(
                                    entries.entries.data(), entries.count));
                    throw;
                }
            });
        }

        void complete(std::uint64_t sequence,
                      std::exception_ptr failure = nullptr);
        void commit_failure(std::uint64_t sequence,
                            std::exception_ptr failure);
        void seek_next_sequence(std::uint64_t next_sequence);

    private:
        [[nodiscard]] oid invoke(oid result) noexcept;
        [[nodiscard]] oid invoke_failure(std::exception_ptr failure) noexcept;
        [[nodiscard]] static oid map_failure(
                std::exception_ptr failure) noexcept;
        [[nodiscard]] static std::uint8_t lease_queue_id();
        static void release_queue_id(std::uint8_t queue_id) noexcept;
        [[nodiscard]] oid encode_token(std::uint64_t sequence) const noexcept;

        const Device* device_ = nullptr;
        std::uint8_t queue_id_;
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
    };

    class Block {
    public:
        virtual ~Block() = default;
        virtual void forward(const Tensor& x, Tensor& y) = 0;
    };
}

#pragma once

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


    /**
     * One in-order asynchronous operation queue over caller-created tensor
     * views. Public operations are non-virtual exception-free facades; backend
     * implementations override only the protected hooks below.
     */
    class DeviceOps {
    public:
        DeviceOps(const DeviceOps&) = delete;
        DeviceOps& operator=(const DeviceOps&) = delete;
        DeviceOps(DeviceOps&&) = delete;
        DeviceOps& operator=(DeviceOps&&) = delete;

        virtual ~DeviceOps();

        void wait(oid token);

        oid copy(const TensorView& source, TensorView& destination) noexcept;
        oid add(const TensorView& a, const TensorView& b,
                TensorView& c) noexcept;
        oid mul(const TensorView& a, const TensorView& b,
                TensorView& c) noexcept;
        oid silu(const TensorView& x, TensorView& y) noexcept;
        oid linear(const TensorView& x, const TensorView& w,
                   TensorView& y) noexcept;
        oid rmsnorm(const TensorView& x, TensorView& y, const TensorView& w,
                    float eps, size_t dim) noexcept;
        oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
                 size_t n_heads, size_t n_kv_heads, size_t head_dim,
                 TensorView& attn_out) noexcept;

    protected:
        DeviceOps();
        explicit DeviceOps(const Device& device);

        virtual oid copy_impl(
                const TensorView& source, TensorView& destination);
        virtual oid add_impl(const TensorView& a, const TensorView& b,
                             TensorView& c);
        virtual oid mul_impl(const TensorView& a, const TensorView& b,
                             TensorView& c);
        virtual oid silu_impl(const TensorView& x, TensorView& y);
        virtual oid linear_impl(const TensorView& x, const TensorView& w,
                                TensorView& y);
        virtual oid rmsnorm_impl(const TensorView& x, TensorView& y,
                                 const TensorView& w, float eps, size_t dim);
        virtual oid sdpa_impl(const TensorView& q, const TensorView& k,
                              const TensorView& v, size_t n_heads,
                              size_t n_kv_heads, size_t head_dim,
                              TensorView& attn_out);

        [[nodiscard]] const Device& queue_device() const;
        virtual void fence_through_sequence(
                std::uint64_t sequence) noexcept;
        void record_post_completion_failure(
                std::uint64_t sequence, std::exception_ptr failure);
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

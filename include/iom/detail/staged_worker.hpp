#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

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

}  // namespace iom

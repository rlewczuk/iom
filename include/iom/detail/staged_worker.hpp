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
                submit(std::move(task), false);
            }

            // Backend-specific callers may publish a task before its execute
            // callback runs. Existing submit_copy users retain their
            // synchronous callback and publication behavior.
            void submit_after_publish(Task task) {
                submit(std::move(task), true);
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
                std::list<QueuedTask> staged;
                std::list<QueuedTask> tasks;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    staged.splice(staged.end(), staged_);
                    tasks.splice(tasks.end(), tasks_);
                }
                drain_list(staged);
                drain_list(tasks);
            }

        private:
            struct QueuedTask {
                Task task;
                bool execute_after_publish = false;

                QueuedTask(Task task_, bool execute_after_publish_)
                    : task(std::move(task_)),
                      execute_after_publish(execute_after_publish_) {}
            };

            void submit(Task task, bool execute_after_publish) {
                typename std::list<QueuedTask>::iterator staged;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (shutdown_) {
                        throw std::logic_error(
                                "staged worker is shut down");
                    }
                    staged_.emplace_back(
                            std::move(task), execute_after_publish);
                    staged = staged_.end();
                    --staged;
                }

                try {
                    if (!execute_after_publish) {
                        callbacks_.execute(staged->task);
                    }
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
                        publish_failure_sequence = staged->task.sequence;
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

            void process(QueuedTask&& queued, bool wait_for_fence) {
                std::exception_ptr failure;
                if (queued.execute_after_publish) {
                    try {
                        callbacks_.execute(queued.task);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                }
                if (wait_for_fence && queued.task.fence != nullptr) {
                    try {
                        callbacks_.fence_complete(queued.task.fence);
                    } catch (...) {
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                }
                if (queued.task.fence != nullptr) {
                    try {
                        callbacks_.fence_destroy(queued.task.fence);
                    } catch (...) {
                        if (!failure) {
                            failure = std::current_exception();
                        }
                    }
                }
                callbacks_.complete(queued.task.sequence, std::move(failure));
            }

            void drain_list(std::list<QueuedTask>& list) {
                while (!list.empty()) {
                    QueuedTask current(std::move(list.front()));
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
                            QueuedTask current(std::move(tasks_.front()));
                            tasks_.pop_front();
                            lock.unlock();
                            process(std::move(current), false);
                            lock.lock();
                        }
                        return;
                    }
                    QueuedTask current(std::move(tasks_.front()));
                    tasks_.pop_front();
                    lock.unlock();
                    process(std::move(current), true);
                }
            }


            Callbacks callbacks_;
            PublishPolicy publish_policy_;
            std::mutex mutex_;
            std::condition_variable completion_;
            std::list<QueuedTask> staged_;
            std::list<QueuedTask> tasks_;
            bool shutdown_ = false;
            std::thread worker_;
        };
    }  // namespace detail

}  // namespace iom

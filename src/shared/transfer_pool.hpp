#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace iom::detail {

template <typename Policy>
struct TransferStreamPool final {
    class Scope final {
    public:
        Scope(TransferStreamPool& pool, typename Policy::stream_type stream)
                : pool_(&pool), stream_(stream), poisoned_(false) {}
        ~Scope() noexcept {
            const bool synchronized =
                    Policy::synchronize_stream_noexcept(stream_);
            const bool drop_stream = poisoned_ || !synchronized;
            if (drop_stream) {
                std::lock_guard<std::mutex> lock(pool_->mutex_);
                const auto it = pool_->in_use_.find(stream_);
                if (it != pool_->in_use_.end()) {
                    pool_->in_use_.erase(it);
                    pool_->cv_.notify_all();
                }
                Policy::destroy_queue_stream_noexcept(stream_);
            } else {
                pool_->release(stream_);
            }
        }

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;

        [[nodiscard]] typename Policy::stream_type stream() const noexcept {
            return stream_;
        }
        void poison() noexcept { poisoned_ = true; }

    private:
        TransferStreamPool* pool_;
        typename Policy::stream_type stream_;
        bool poisoned_;
    };

    [[nodiscard]] Scope acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (closing_) {
            throw Policy::transfer_pool_closing_error();
        }
        if (idle_.empty()) {
            const typename Policy::stream_type stream =
                    Policy::create_queue_stream();
            try {
                idle_.push_back(stream);
            } catch (...) {
                Policy::destroy_queue_stream_noexcept(stream);
                throw;
            }
        }
        const typename Policy::stream_type stream = idle_.back();
        idle_.pop_back();
        try {
            in_use_.insert(stream);
        } catch (...) {
            idle_.push_back(stream);
            throw;
        }
        return Scope{*this, stream};
    }

    void release(typename Policy::stream_type stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = in_use_.find(stream);
        if (it != in_use_.end()) {
            in_use_.erase(it);
            try {
                idle_.push_back(stream);
            } catch (...) {
                Policy::destroy_queue_stream_noexcept(stream);
            }
            cv_.notify_all();
        }
    }

    void destroy() {
        std::unique_lock<std::mutex> lock(mutex_);
        closing_ = true;
        cv_.wait(lock, [this] { return in_use_.empty(); });
        for (const typename Policy::stream_type stream : idle_) {
            Policy::destroy_queue_stream_noexcept(stream);
        }
        idle_.clear();
    }

    [[nodiscard]] std::size_t idle_count_for_testing() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return idle_.size();
    }

private:
    std::vector<typename Policy::stream_type> idle_;
    std::unordered_set<typename Policy::stream_type> in_use_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool closing_ = false;
};

}  // namespace iom::detail

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace iom::cuda_detail {

struct TransferStreamPool {
    class Scope {
    public:
        Scope(TransferStreamPool& pool, cudaStream_t stream)
                : pool_(&pool), stream_(stream), poisoned_(false) {}
        ~Scope() noexcept;

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;

        [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
        void poison() noexcept;

    private:
        TransferStreamPool* pool_;
        cudaStream_t stream_;
        bool poisoned_;
    };

    [[nodiscard]] Scope acquire();
    void release(cudaStream_t stream);
    void destroy();
    [[nodiscard]] std::size_t idle_count_for_testing() const noexcept;

private:
    std::vector<cudaStream_t> idle_{};
    std::unordered_set<cudaStream_t> in_use_{};
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool closing_ = false;
};

}  // namespace iom::cuda_detail

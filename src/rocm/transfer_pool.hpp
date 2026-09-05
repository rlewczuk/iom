#pragma once

#include <hip/hip_runtime_api.h>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace iom::rocm_detail {

struct TransferStreamPool final {
    class Scope final {
    public:
        Scope(TransferStreamPool& pool, hipStream_t stream)
                : pool_(&pool), stream_(stream), poisoned_(false) {}
        ~Scope() noexcept;

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&&) = delete;
        Scope& operator=(Scope&&) = delete;

        [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
        void poison() noexcept { poisoned_ = true; }

    private:
        TransferStreamPool* pool_;
        hipStream_t stream_;
        bool poisoned_;
    };

    [[nodiscard]] Scope acquire();
    void release(hipStream_t stream);
    void destroy();
    [[nodiscard]] std::size_t idle_count_for_testing() const noexcept;

private:
    std::vector<hipStream_t> idle_;
    std::unordered_set<hipStream_t> in_use_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool closing_ = false;
};

}  // namespace iom::rocm_detail

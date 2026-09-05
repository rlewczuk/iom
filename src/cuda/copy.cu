#include "copy.hpp"
#include "transfer_pool.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include "driver.hpp"

namespace iom::cuda_detail {
namespace {

std::atomic<SubmissionFault> g_submission_fault{SubmissionFault::none};

[[nodiscard]] bool consume_submission_fault(
        SubmissionFault point) noexcept {
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}


void check_cuda_kernel(const char* operation, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + cudaGetErrorName(status) + ": "
                + cudaGetErrorString(status));
    }
}

struct gpu_policy {
    using context_type = CUcontext;
    using stream_type = cudaStream_t;
    using event_type = cudaEvent_t;

    [[nodiscard]] static constexpr stream_type null_stream() noexcept {
        return nullptr;
    }
    [[nodiscard]] static constexpr event_type null_event() noexcept {
        return nullptr;
    }

    static void activate(context_type context) {
        check_cuda(
                "cuCtxSetCurrent",
                driver_calls.ctx_set_current(context));
    }

    [[nodiscard]] static stream_type create_queue_stream() {
        stream_type stream = nullptr;
        check_cuda_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return stream;
    }
    static void destroy_queue_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
    }

    static void synchronize_stream(stream_type stream) {
        check_cuda_kernel("cudaStreamSynchronize", cudaStreamSynchronize(stream));
    }

    static void create_event(event_type* event) {
        if (consume_submission_fault(SubmissionFault::event_create)) {
            check_cuda_kernel(
                    "cudaEventCreateWithFlags", cudaErrorInvalidValue);
        }
        check_cuda_kernel(
                "cudaEventCreateWithFlags",
                cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
    [[nodiscard]] static bool event_is_valid(event_type event) noexcept {
        return event != nullptr;
    }
    static void destroy_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
        }
    }
    static void synchronize_event(event_type event) {
        check_cuda_kernel("cudaEventSynchronize", cudaEventSynchronize(event));
    }
    static constexpr bool discard_failure_event_on_error = false;
    static void record_event(event_type event, stream_type stream) {
        cudaError_t status = cudaEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = cudaErrorInvalidValue;
        }
        check_cuda_kernel("cudaEventRecord", status);
    }
    static void record_event_no_fault(
            event_type event, stream_type stream) noexcept {
        (void)cudaEventRecord(event, stream);
    }

    [[nodiscard]] static void* allocate(std::size_t bytes) {
        CUdeviceptr address = 0;
        check_cuda("cuMemAlloc", cuMemAlloc(&address, bytes));
        return reinterpret_cast<void*>(address);
    }
    static void free(void* address) {
        check_cuda("cuMemFree", cuMemFree(reinterpret_cast<CUdeviceptr>(address)));
    }
    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)cuMemFree(reinterpret_cast<CUdeviceptr>(address));
        }
    }

    static void copy_from_host(
            stream_type stream, void* destination, const void* source,
            std::size_t bytes) {
        check_cuda_kernel(
                "cudaMemcpyAsync HtoD",
                cudaMemcpyAsync(
                        destination, source, bytes,
                        cudaMemcpyHostToDevice, stream));
    }
    static void copy_to_host(
            stream_type, void* destination, const void* source,
            std::size_t bytes) {
        check_cuda_kernel(
                "cudaMemcpy DtoH",
                cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
    }
    static void memset(
            stream_type stream, void* destination, std::size_t bytes) {
        check_cuda_kernel(
                "cudaMemsetAsync",
                cudaMemsetAsync(destination, 0, bytes, stream));
    }

    static void check_kernel(const char* operation) {
        ::iom::cuda_detail::check_cuda_kernel(operation, cudaGetLastError());
    }
    static void after_copy_plane_launch(std::size_t plane_index) {
        if (plane_index == 2
                && consume_submission_fault(
                        SubmissionFault::third_plane_launch)) {
            ::iom::cuda_detail::check_cuda_kernel(
                    copy_kernel_operation(), cudaErrorInvalidValue);
        }
    }
    [[nodiscard]] static constexpr const char* scatter_kernel_operation() noexcept {
        return "CUDA scatter kernel launch";
    }
    [[nodiscard]] static constexpr const char* gather_kernel_operation() noexcept {
        return "CUDA gather kernel launch";
    }
    [[nodiscard]] static constexpr const char* copy_kernel_operation() noexcept {
        return "CUDA copy kernel launch";
    }

};
void fence_and_destroy(cudaEvent_t event) noexcept {
    if (event == nullptr) {
        return;
    }
    (void)cudaEventSynchronize(event);
    (void)cudaEventDestroy(event);
}

void synchronize_and_destroy_stream(cudaStream_t stream) noexcept {
    if (stream == nullptr) {
        return;
    }
    (void)cudaStreamSynchronize(stream);
    (void)cudaStreamDestroy(stream);
}


}  // namespace
}  // namespace iom::cuda_detail

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_GPU_ATOMIC_OR atomicOr
#define IOM_GPU_ATOMIC_AND atomicAnd
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(__VA_ARGS__)
#include "../shared/standard_tiled_copy.inl"
#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_ATOMIC_AND
#undef IOM_GPU_ATOMIC_OR
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

namespace iom::cuda_detail {
void TransferStreamPool::Scope::poison() noexcept {
    poisoned_ = true;
}

std::size_t TransferStreamPool::idle_count_for_testing() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return idle_.size();
}

TransferStreamPool::Scope::~Scope() noexcept {
    const cudaError_t sync_status = cudaStreamSynchronize(stream_);
    const bool drop_stream = poisoned_ || (sync_status != cudaSuccess);
    if (drop_stream) {
        std::lock_guard<std::mutex> lock(pool_->mutex_);
        const auto it = pool_->in_use_.find(stream_);
        if (it != pool_->in_use_.end()) {
            pool_->in_use_.erase(it);
            pool_->cv_.notify_all();
        }
        (void)cudaStreamDestroy(stream_);
    } else {
        pool_->release(stream_);
    }
}

TransferStreamPool::Scope TransferStreamPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (closing_) {
        throw std::runtime_error(
                "cudaStreamCreateWithFlags failed with "
                "cudaErrorStreamDestroyed: TransferStreamPool is closing");
    }
    if (idle_.empty()) {
        cudaStream_t stream = nullptr;
        check_cuda_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        idle_.push_back(stream);
    }
    cudaStream_t stream = idle_.back();
    idle_.pop_back();
    try {
        in_use_.insert(stream);
    } catch (...) {
        idle_.push_back(stream);
        throw;
    }
    return Scope{*this, stream};
}

void TransferStreamPool::release(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = in_use_.find(stream);
    if (it != in_use_.end()) {
        in_use_.erase(it);
        idle_.push_back(stream);
        cv_.notify_all();
    }
}

void TransferStreamPool::destroy() {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    cv_.wait(lock, [this] { return in_use_.empty(); });
    for (const cudaStream_t stream : idle_) {
        (void)cudaStreamDestroy(stream);
    }
    idle_.clear();
}

namespace {

void synchronous_transfer(
        TransferStreamPool& pool, CUcontext context, const TensorView& view,
        std::span<const std::byte> source, std::span<std::byte> destination,
        bool from_host) {
    const std::size_t logical_nbytes = view.spec().logical_nbytes();
    const std::size_t staging_nbytes =
            gpu_algorithm::compute_staging_size(logical_nbytes);
    gpu_policy::activate(context);

    void* staging = nullptr;
    std::exception_ptr failure;
    try {
        staging = gpu_policy::allocate(staging_nbytes);
        {
            auto scope = pool.acquire();
            const cudaStream_t stream = scope.stream();
            try {
                if (from_host) {
                    gpu_policy::copy_from_host(
                            stream, staging, source.data(), source.size());
                    detail::launch_view_transfer<gpu_policy>(
                            stream, view, staging,
                            const_cast<void*>(view.native_handle()), true);
                    gpu_policy::after_copy_plane_launch(2);
                } else {
                    gpu_policy::memset(stream, staging, staging_nbytes);
                    detail::launch_view_transfer<gpu_policy>(
                            stream, view, view.native_handle(), staging, false);
                    gpu_policy::after_copy_plane_launch(2);
                }
                gpu_policy::synchronize_stream(stream);
                if (!from_host) {
                    gpu_policy::copy_to_host(
                            stream, destination.data(), staging,
                            destination.size());
                }
            } catch (...) {
                scope.poison();
                throw;
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    if (failure) {
        if (staging != nullptr) {
            gpu_policy::free_noexcept(staging);
            staging = nullptr;
        }
        std::rethrow_exception(failure);
    }
    gpu_policy::free(staging);
}

}  // namespace

namespace {

class CudaQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        cudaEvent_t event = nullptr;
        bool no_op;
        void* fence = nullptr;
    };

public:
    CudaQueue(const Device& device, CUcontext context)
            : device_(&device),
              context_(context),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [this](void* fence) {
                                  gpu_policy::activate(context_);
                                  gpu_policy::synchronize_event(
                                          static_cast<cudaEvent_t>(fence));
                              },
                              [this](void* fence) {
                                  try {
                                      gpu_policy::activate(context_);
                                      fence_and_destroy(
                                              static_cast<cudaEvent_t>(fence));
                                  } catch (...) {
                                  }
                              },
                              [this](
                                      std::uint64_t sequence,
                                      std::exception_ptr failure) {
                                  complete(sequence, std::move(failure));
                              }},
                      detail::StagedWorker<Task>::PublishPolicy::Splice) {
        gpu_policy::activate(context_);
        try {
            stream_ = gpu_policy::create_queue_stream();
            worker_.start();
        } catch (...) {
            gpu_policy::destroy_queue_stream_noexcept(stream_);
            stream_ = gpu_policy::null_stream();
            throw;
        }
    }

    ~CudaQueue() override {
        worker_.shutdown_and_drain();
        try {
            gpu_policy::activate(context_);
            synchronize_and_destroy_stream(stream_);
        } catch (...) {
        }
        stream_ = gpu_policy::null_stream();
    }

    oid copy(const TensorView& source, TensorView& destination) override {
        std::lock_guard<std::mutex> submission_lock(
                submission_order_mutex_);
        validate_copy(*device_, source, destination);
        const bool no_op = identical_window(source, destination);
        return submit(
                [this, &source, &destination, no_op](
                        std::uint64_t sequence) {
                    worker_.submit_copy(
                            Task{sequence, &source, &destination, nullptr,
                                 no_op});
                });
    }

    oid add(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "add");
    }
    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "mul");
    }
    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "silu");
    }
    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("CUDA", "linear");
    }
    oid rmsnorm(const TensorView&, TensorView&, const TensorView&, float,
                size_t) override {
        throw unsupported("CUDA", "rmsnorm");
    }
    oid sdpa(const TensorView&, const TensorView&, const TensorView&,
             size_t, size_t, size_t, TensorView&) override {
        throw unsupported("CUDA", "sdpa");
    }

private:
    void execute(Task& task) {
        gpu_policy::activate(context_);
        detail::PendingEvent<gpu_policy> event_guard;
        gpu_policy::create_event(&event_guard.event);
        task.event = event_guard.event;
        task.fence = static_cast<void*>(event_guard.event);
        const std::vector<detail::PlanePair> pairs =
                task.no_op ? std::vector<detail::PlanePair>{}
                           : detail::plane_pairs(*task.source, *task.destination);
        const std::span<const std::size_t> dimensions =
                task.source->spec().shape.dimensions();
        const std::size_t rows = dimensions[dimensions.size() - 2];
        const std::size_t columns = dimensions[dimensions.size() - 1];
        const unsigned int bits = static_cast<unsigned int>(
                detail::leaf_bits(task.source->spec().data_type));

        try {
            std::size_t plane_index = 0;
            for (const detail::PlanePair pair : pairs) {
                detail::launch_copy_plane<gpu_policy>(
                        stream_, pair, plane_index++, rows, columns, bits,
                        task.source->native_handle(),
                        task.destination->native_handle());
            }
            gpu_policy::record_event(task.event, stream_);
            event_guard.linked = true;
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            if constexpr (gpu_policy::discard_failure_event_on_error) {
                task.fence = nullptr;
                event_guard.linked = false;
            } else {
                gpu_policy::record_event_no_fault(task.event, stream_);
                event_guard.linked = true;
            }
            commit_failure(task.sequence, failure);
        }
    }

    const Device* device_;
    CUcontext context_;
    cudaStream_t stream_ = gpu_policy::null_stream();
    std::mutex submission_order_mutex_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace


void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_submission_fault.store(fault, std::memory_order_release);
}

void region_from_host(
        TransferStreamPool& pool, CUcontext context,
        const TensorView& destination, std::span<const std::byte> source) {
    synchronous_transfer(pool, context, destination, source, {}, true);
}

void region_to_host(
        TransferStreamPool& pool, CUcontext context, const TensorView& source,
        std::span<std::byte> destination) {
    synchronous_transfer(pool, context, source, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(const Device& device, CUcontext context) {
    return std::make_unique<CudaQueue>(device, context);
}

}  // namespace iom::cuda_detail

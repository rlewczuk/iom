#include "copy.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace iom::rocm_detail {
namespace {

std::atomic<SubmissionFault> g_submission_fault{SubmissionFault::none};

[[nodiscard]] bool consume_submission_fault(
        SubmissionFault point) noexcept {
    SubmissionFault expected = point;
    return g_submission_fault.compare_exchange_strong(
            expected, SubmissionFault::none, std::memory_order_acq_rel);
}

[[nodiscard]] std::runtime_error hip_error(
        const char* operation, hipError_t status) {
    return std::runtime_error(
            std::string(operation) + " failed with "
            + hipGetErrorName(status) + ": " + hipGetErrorString(status));
}

void check_hip(const char* operation, hipError_t status) {
    if (status != hipSuccess) {
        throw hip_error(operation, status);
    }
}

struct gpu_policy {
    using context_type = int;
    using stream_type = hipStream_t;
    using event_type = hipEvent_t;

    [[nodiscard]] static constexpr stream_type null_stream() noexcept {
        return nullptr;
    }
    [[nodiscard]] static constexpr event_type null_event() noexcept {
        return nullptr;
    }
    [[nodiscard]] static bool stream_is_valid(stream_type stream) noexcept {
        return stream != nullptr;
    }
    [[nodiscard]] static bool event_is_valid(event_type event) noexcept {
        return event != nullptr;
    }

    static void activate(context_type device_ordinal) {
        check_hip("hipSetDevice", hipSetDevice(device_ordinal));
    }

    [[nodiscard]] static stream_type create_queue_stream() {
        stream_type stream = nullptr;
        check_hip(
                "hipStreamCreateWithFlags",
                hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        return stream;
    }
    static void destroy_queue_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
        }
    }

    [[nodiscard]] static stream_type create_transfer_stream() {
        stream_type stream = nullptr;
        check_hip(
                "hipStreamCreateWithFlags",
                hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        return stream;
    }
    static void destroy_transfer_stream(stream_type stream) {
        check_hip("hipStreamDestroy", hipStreamDestroy(stream));
    }
    static void destroy_transfer_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
        }
    }

    static void create_event(event_type* event) {
        if (consume_submission_fault(SubmissionFault::event_create)) {
            check_hip("hipEventCreateWithFlags", hipErrorInvalidValue);
        }
        check_hip(
                "hipEventCreateWithFlags",
                hipEventCreateWithFlags(event, hipEventDisableTiming));
    }
    static void destroy_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)hipEventDestroy(event);
        }
    }
    static void synchronize_event(event_type event) {
        check_hip("hipEventSynchronize", hipEventSynchronize(event));
    }
    static constexpr bool discard_failure_event_on_error = true;
    static void record_event(event_type event, stream_type stream) {
        hipError_t status = hipEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = hipErrorInvalidValue;
        }
        check_hip("hipEventRecord", status);
    }

    [[nodiscard]] static void* allocate(std::size_t bytes) {
        void* address = nullptr;
        check_hip("hipMalloc", hipMalloc(&address, bytes));
        return address;
    }
    static void free(void* address) {
        check_hip("hipFree", hipFree(address));
    }
    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)hipFree(address);
        }
    }

    static void copy_from_host(
            stream_type stream, void* destination, const void* source,
            std::size_t bytes) {
        check_hip(
                "hipMemcpyAsync HtoD",
                hipMemcpyAsync(
                        destination, source, bytes,
                        hipMemcpyHostToDevice, stream));
    }
    static void copy_to_host(
            stream_type, void* destination, const void* source,
            std::size_t bytes) {
        check_hip(
                "hipMemcpy DtoH",
                hipMemcpy(destination, source, bytes, hipMemcpyDeviceToHost));
    }
    static void memset(
            stream_type stream, void* destination, std::size_t bytes) {
        check_hip(
                "hipMemset",
                hipMemsetAsync(destination, 0, bytes, stream));
    }

    static void synchronize_stream(stream_type stream) {
        check_hip("hipStreamSynchronize", hipStreamSynchronize(stream));
    }
    static void synchronize_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)hipStreamSynchronize(stream);
        }
    }

    static void check_kernel(const char* operation) {
        check_hip(operation, hipGetLastError());
    }
    static void after_copy_plane_launch(std::size_t plane_index) {
        if (plane_index == 2
                && consume_submission_fault(
                        SubmissionFault::third_plane_launch)) {
            check_hip("HIP copy kernel launch", hipErrorInvalidValue);
        }
    }
    [[nodiscard]] static constexpr const char* scatter_kernel_operation() noexcept {
        return "HIP kernel launch";
    }
    [[nodiscard]] static constexpr const char* gather_kernel_operation() noexcept {
        return "HIP kernel launch";
    }
    [[nodiscard]] static constexpr const char* copy_kernel_operation() noexcept {
        return "HIP kernel launch";
    }

};
void fence_and_destroy(hipEvent_t event) noexcept {
    if (event == nullptr) {
        return;
    }
    (void)hipEventSynchronize(event);
    (void)hipEventDestroy(event);
}

void synchronize_and_destroy_stream(hipStream_t stream) noexcept {
    if (stream == nullptr) {
        return;
    }
    (void)hipStreamSynchronize(stream);
    (void)hipStreamDestroy(stream);
}


}  // namespace
}  // namespace iom::rocm_detail

#define IOM_GPU_DEVICE __device__
#define IOM_GPU_GLOBAL __global__
#define IOM_GPU_GLOBAL_INDEX (blockIdx.x * blockDim.x + threadIdx.x)
#define IOM_GPU_ATOMIC_OR atomicOr
#define IOM_GPU_ATOMIC_AND atomicAnd
#define IOM_LAUNCH_KERNEL(kernel, blocks, threads, stream, ...) \
    hipLaunchKernelGGL( \
            kernel, dim3(blocks), dim3(threads), 0, stream, __VA_ARGS__)
#include "../shared/standard_tiled_copy.inl"
#undef IOM_LAUNCH_KERNEL
#undef IOM_GPU_ATOMIC_AND
#undef IOM_GPU_ATOMIC_OR
#undef IOM_GPU_GLOBAL_INDEX
#undef IOM_GPU_GLOBAL
#undef IOM_GPU_DEVICE

namespace iom::rocm_detail {

namespace {

class RocmQueue final : public DeviceOps {
    struct Task {
        std::uint64_t sequence;
        const TensorView* source;
        TensorView* destination;
        hipEvent_t event = nullptr;
        bool no_op;
        void* fence = nullptr;
    };

public:
    RocmQueue(const Device& device, int device_ordinal)
            : device_(&device),
              context_(device_ordinal),
              worker_(
                      detail::StagedWorker<Task>::Callbacks{
                              [this](Task& task) {
                                  execute(task);
                              },
                              [this](void* fence) {
                                  gpu_policy::activate(context_);
                                  gpu_policy::synchronize_event(
                                          static_cast<hipEvent_t>(fence));
                              },
                              [this](void* fence) {
                                  try {
                                      gpu_policy::activate(context_);
                                      fence_and_destroy(
                                              static_cast<hipEvent_t>(fence));
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

    ~RocmQueue() override {
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
        throw unsupported("ROCm", "add");
    }
    oid mul(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("ROCm", "mul");
    }
    oid silu(const TensorView&, TensorView&) override {
        throw unsupported("ROCm", "silu");
    }
    oid linear(const TensorView&, const TensorView&, TensorView&) override {
        throw unsupported("ROCm", "linear");
    }
    oid rmsnorm(const TensorView&, TensorView&, const TensorView&, float,
                size_t) override {
        throw unsupported("ROCm", "rmsnorm");
    }
    oid sdpa(const TensorView&, const TensorView&, const TensorView&,
             size_t, size_t, size_t, TensorView&) override {
        throw unsupported("ROCm", "sdpa");
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
            task.fence = nullptr;
            event_guard.linked = false;
            commit_failure(task.sequence, failure);
        }
    }

    const Device* device_;
    int context_;
    hipStream_t stream_ = gpu_policy::null_stream();
    std::mutex submission_order_mutex_;
    detail::StagedWorker<Task> worker_;
};

}  // namespace
void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_submission_fault.store(fault, std::memory_order_release);
}

void region_from_host(
        int device_ordinal, const TensorView& destination,
        std::span<const std::byte> source) {
    detail::synchronous_transfer<gpu_policy>(
            device_ordinal, destination, source, {}, true);
}

void region_to_host(
        int device_ordinal, const TensorView& source,
        std::span<std::byte> destination) {
    detail::synchronous_transfer<gpu_policy>(
            device_ordinal, source, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(
        const Device& device, int device_ordinal) {
    return std::make_unique<RocmQueue>(device, device_ordinal);
}


}  // namespace iom::rocm_detail

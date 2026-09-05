#include "copy.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
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

[[nodiscard]] std::runtime_error cuda_error(
        const char* operation, CUresult status) {
    const char* name = nullptr;
    const char* description = nullptr;
    (void)cuGetErrorName(status, &name);
    (void)cuGetErrorString(status, &description);
    return std::runtime_error(
            std::string(operation) + " failed with "
            + (name != nullptr ? name : "unknown CUDA error") + ": "
            + (description != nullptr ? description : "unknown error"));
}

void check_cuda(const char* operation, CUresult status) {
    if (status != CUDA_SUCCESS) {
        throw cuda_error(operation, status);
    }
}

void check_kernel(const char* operation, cudaError_t status) {
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
        check_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return stream;
    }
    static void destroy_queue_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
    }

    [[nodiscard]] static stream_type create_transfer_stream() noexcept {
        return nullptr;
    }
    [[nodiscard]] static bool stream_is_valid(stream_type stream) noexcept {
        return stream != nullptr;
    }
    static void destroy_transfer_stream(stream_type) noexcept {}
    static void destroy_transfer_stream_noexcept(stream_type) noexcept {}
    static void synchronize_stream_noexcept(stream_type) noexcept {}
    static void synchronize_stream(stream_type stream) {
        check_kernel("cudaStreamSynchronize", cudaStreamSynchronize(stream));
    }

    static void create_event(event_type* event) {
        if (consume_submission_fault(SubmissionFault::event_create)) {
            check_kernel(
                    "cudaEventCreateWithFlags", cudaErrorInvalidValue);
        }
        check_kernel(
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
        check_kernel("cudaEventSynchronize", cudaEventSynchronize(event));
    }
    static void record_event(event_type event, stream_type stream) {
        cudaError_t status = cudaEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = cudaErrorInvalidValue;
        }
        check_kernel("cudaEventRecord", status);
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
            stream_type, void* destination, const void* source,
            std::size_t bytes) {
        check_kernel(
                "cudaMemcpy HtoD",
                cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
    }
    static void copy_to_host(
            stream_type, void* destination, const void* source,
            std::size_t bytes) {
        check_kernel(
                "cudaMemcpy DtoH",
                cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
    }
    static void memset(
            stream_type, void* destination, std::size_t bytes) {
        check_kernel("cudaMemset", cudaMemset(destination, 0, bytes));
    }

    static void check_kernel(const char* operation) {
        ::iom::cuda_detail::check_kernel(operation, cudaGetLastError());
    }
    static void after_copy_plane_launch(std::size_t plane_index) {
        if (plane_index == 2
                && consume_submission_fault(
                        SubmissionFault::third_plane_launch)) {
            ::iom::cuda_detail::check_kernel(
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

    [[nodiscard]] static std::runtime_error unsupported(const char* operation) {
        return std::runtime_error(
                std::string("CUDA backend does not implement ") + operation);
    }
};

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

void inject_submission_fault_for_testing(
        SubmissionFault fault) noexcept {
    g_submission_fault.store(fault, std::memory_order_release);
}

void region_from_host(
        CUcontext context, const TensorView& destination,
        std::span<const std::byte> source) {
    detail::synchronous_transfer<gpu_policy>(context, destination, source, {}, true);
}

void region_to_host(
        CUcontext context, const TensorView& source,
        std::span<std::byte> destination) {
    detail::synchronous_transfer<gpu_policy>(context, source, {}, destination, false);
}

std::unique_ptr<DeviceOps> make_queue(const Device& device, CUcontext context) {
    return std::make_unique<detail::GpuQueue<gpu_policy>>(device, context);
}

}  // namespace iom::cuda_detail

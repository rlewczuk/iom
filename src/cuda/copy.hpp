#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "driver.hpp"
#include "../shared/event_ring.hpp"
#include "../shared/metadata_slot_pool.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "../shared/staging_pool.hpp"
#include "../shared/transfer_pool.hpp"

#include "iom/iom.hpp"

namespace iom::cuda_detail {
enum class SubmissionFault {
    none,
    event_create,
    third_plane_launch,
    event_record,
};

void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;
[[nodiscard]] bool consume_submission_fault(SubmissionFault fault) noexcept;

inline void check_cuda_kernel(const char* operation, cudaError_t status) {
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
    using device_pointer = CUdeviceptr;

    [[nodiscard]] static constexpr stream_type null_stream() noexcept {
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
        check_cuda_kernel(
                "cudaStreamSynchronize", cudaStreamSynchronize(stream));
    }

    [[nodiscard]] static bool synchronize_stream_noexcept(
            stream_type stream) noexcept {
        return stream == nullptr
                || cudaStreamSynchronize(stream) == cudaSuccess;
    }

    static void check_acquire_event_fault() {
        if (consume_submission_fault(SubmissionFault::event_create)) {
            check_cuda_kernel(
                    "cudaEventCreateWithFlags", cudaErrorInvalidValue);
        }
    }

    static void create_event(event_type* event) {
        check_cuda_kernel(
                "cudaEventCreateWithFlags",
                cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }

    static void destroy_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
        }
    }

    static void synchronize_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)cudaEventSynchronize(event);
        }
    }

    static void synchronize_event(event_type event) {
        check_cuda_kernel(
                "cudaEventSynchronize", cudaEventSynchronize(event));
    }

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
        device_pointer address = 0;
        check_cuda("cuMemAlloc", cuMemAlloc(&address, bytes));
        return reinterpret_cast<void*>(address);
    }

    [[nodiscard]] static void* staging_address(
            device_pointer address) noexcept {
        return reinterpret_cast<void*>(address);
    }

    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)cuMemFree(reinterpret_cast<device_pointer>(address));
        }
    }

    [[nodiscard]] static device_pointer staging_allocate(
            std::size_t bytes) {
        device_pointer address = 0;
        check_cuda("cuMemAlloc", cuMemAlloc(&address, bytes));
        return address;
    }

    static void staging_free_noexcept(device_pointer address) noexcept {
        if (address != device_pointer{}) {
            (void)cuMemFree(address);
        }
    }

    [[nodiscard]] static std::runtime_error staging_pool_closing_error() {
        return std::runtime_error("CUDA staging pool is closing");
    }

    [[nodiscard]] static std::runtime_error transfer_pool_closing_error() {
        return std::runtime_error(
                "cudaStreamCreateWithFlags failed with "
                "cudaErrorStreamDestroyed: TransferStreamPool is closing");
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
        check_cuda_kernel(operation, cudaGetLastError());
    }

    static void after_copy_plane_launch(std::size_t plane_index) {
        if (plane_index == 2
                && consume_submission_fault(
                        SubmissionFault::third_plane_launch)) {
            check_cuda_kernel(copy_kernel_operation(), cudaErrorInvalidValue);
        }
    }

    static void after_grid_stride_launch() {
        if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
            check_cuda_kernel(copy_kernel_operation(), cudaErrorInvalidValue);
        }
    }

    [[nodiscard]] static constexpr const char* scatter_kernel_operation()
            noexcept {
        return "CUDA scatter kernel launch";
    }

    [[nodiscard]] static constexpr const char* gather_kernel_operation()
            noexcept {
        return "CUDA gather kernel launch";
    }

    [[nodiscard]] static constexpr const char* copy_kernel_operation()
            noexcept {
        return "CUDA copy kernel launch";
    }

    [[nodiscard]] static constexpr const char* backend_label() noexcept {
        return "CUDA";
    }
};
using EventRingState = iom::detail::EventRingState<gpu_policy>;

using StagingSlotPool = iom::detail::StagingSlotPool<gpu_policy>;
using TransferStreamPool = iom::detail::TransferStreamPool<gpu_policy>;


void region_from_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        CUcontext context, const TensorView& destination,
        std::span<const std::byte> source);

void region_to_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        CUcontext context, const TensorView& source,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, CUcontext context,
        detail::RegistryState& registry_state);

}  // namespace iom::cuda_detail

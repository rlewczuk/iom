#pragma once

#include <hip/hip_runtime_api.h>

#include <atomic>

#include <exception>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "driver.hpp"
#include "../shared/event_ring.hpp"
#include "../shared/metadata_slot_pool.hpp"
#include "../shared/queue_resources.hpp"
#include "../shared/staging_pool.hpp"
#include "../shared/transfer_pool.hpp"
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/iom.hpp"

namespace iom::rocm_detail {
enum class SubmissionFault {
    none,
    event_create,
    queue_event_create,
    queue_stream_create,
    third_plane_launch,
    event_record,
    stream_synchronize,
    registration,
    outcome_insertion,
};

void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;
[[nodiscard]] bool consume_submission_fault(SubmissionFault fault) noexcept;

#ifdef IOM_ENABLE_TESTING
// Counters over the policy's native queue-resource lifecycle. Queue setup
// creates every stream and completion resource eagerly; tests assert that a
// rejected fifth queue creates none of them.
extern std::atomic<std::size_t> event_create_count_for_testing;
extern std::atomic<std::size_t> event_destroy_count_for_testing;
extern std::atomic<std::size_t> stream_create_count_for_testing;
extern std::atomic<std::size_t> stream_destroy_count_for_testing;
#endif  // IOM_ENABLE_TESTING

[[nodiscard]] inline std::runtime_error hip_error(
        const char* operation, hipError_t status) {
    return std::runtime_error(
            std::string(operation) + " failed with "
            + hipGetErrorName(status) + ": " + hipGetErrorString(status));
}

inline void check_hip(const char* operation, hipError_t status) {
    if (status != hipSuccess) {
        throw hip_error(operation, status);
    }
}

struct gpu_policy {
    using context_type = int;
    using stream_type = hipStream_t;
    using event_type = hipEvent_t;
    using device_pointer = hipDeviceptr_t;

    [[nodiscard]] static constexpr stream_type null_stream() noexcept {
        return nullptr;
    }

    [[nodiscard]] static bool stream_is_valid(stream_type stream) noexcept {
        return stream != nullptr;
    }

    static void activate(context_type device_ordinal) {
        check_hip("hipSetDevice", hipSetDevice(device_ordinal));
    }
    [[nodiscard]] static bool consume_copy_registration_fault() noexcept {
        return consume_submission_fault(SubmissionFault::registration);
    }

    [[nodiscard]] static bool consume_copy_outcome_insertion_fault()
            noexcept {
        return consume_submission_fault(SubmissionFault::outcome_insertion);
    }

    // Queue setup fault points: a queue's C completion resources are
    // created eagerly at construction, so queue_event_create surfaces a
    // completion-resource failure during construction rollback, while
    // event_create remains the submission-time acquisition fault.
    static void check_queue_event_fault() {
        if (consume_submission_fault(SubmissionFault::queue_event_create)) {
            check_hip("hipEventCreateWithFlags", hipErrorInvalidValue);
        }
    }

    static void check_create_queue_stream_fault() {
        if (consume_submission_fault(
                    SubmissionFault::queue_stream_create)) {
            check_hip(
                    "hipStreamCreateWithFlags", hipErrorInvalidValue);
        }
    }

    [[nodiscard]] static stream_type create_queue_stream() {
        check_create_queue_stream_fault();
        stream_type stream = nullptr;
        check_hip(
                "hipStreamCreateWithFlags",
                hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
#ifdef IOM_ENABLE_TESTING
        ++stream_create_count_for_testing;
#endif
        return stream;
    }

    static void destroy_queue_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
#ifdef IOM_ENABLE_TESTING
            ++stream_destroy_count_for_testing;
#endif
        }
    }

    static void check_acquire_event_fault() {
        if (consume_submission_fault(SubmissionFault::event_create)) {
            check_hip("hipEventCreateWithFlags", hipErrorInvalidValue);
        }
    }

    static void create_event(event_type* event) {
        check_hip(
                "hipEventCreateWithFlags",
                hipEventCreateWithFlags(event, hipEventDisableTiming));
#ifdef IOM_ENABLE_TESTING
        ++event_create_count_for_testing;
#endif
    }

    static void destroy_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)hipEventDestroy(event);
#ifdef IOM_ENABLE_TESTING
            ++event_destroy_count_for_testing;
#endif
        }
    }

    static void synchronize_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)hipEventSynchronize(event);
        }
    }

    static void synchronize_event(event_type event) {
        check_hip("hipEventSynchronize", hipEventSynchronize(event));
    }

    static void record_event(event_type event, stream_type stream) {
        hipError_t status = hipEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = hipErrorInvalidValue;
        }
        check_hip("hipEventRecord", status);
    }

    [[nodiscard]] static bool record_event_no_fault(
            event_type event, stream_type stream) noexcept {
        hipError_t status = hipEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = hipErrorInvalidValue;
        }
        return status == hipSuccess;
    }

    [[nodiscard]] static void* allocate(std::size_t bytes) {
        void* address = nullptr;
        // Device-side copy metadata slots: operation metadata backing.
        check_hip(
                "hipMalloc",
                allocation_attempt(
                        &address, bytes,
                        AllocationClass::operation_metadata,
                        AllocationPhase::post_publication));
        return address;
    }

    [[nodiscard]] static void* staging_address(
            device_pointer address) noexcept {
        return static_cast<void*>(address);
    }

    static void free(void* address) {
        check_hip(
                "hipFree",
                free_attempt(
                        address, AllocationClass::operation_metadata,
                        AllocationPhase::post_publication));
    }

    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)free_attempt(
                    address, AllocationClass::operation_metadata,
                    AllocationPhase::post_publication);
        }
    }

    [[nodiscard]] static device_pointer staging_allocate(
            std::size_t bytes) {
        void* address = nullptr;
        check_hip(
                "hipMalloc",
                allocation_attempt(
                        &address, bytes, AllocationClass::staging,
                        AllocationPhase::post_publication));
        return static_cast<device_pointer>(address);
    }

    static void staging_free_noexcept(device_pointer address) noexcept {
        if (address != device_pointer{}) {
            (void)free_attempt(
                    static_cast<void*>(address), AllocationClass::staging,
                    AllocationPhase::post_publication);
        }
    }

    [[nodiscard]] static std::runtime_error staging_pool_closing_error() {
        return std::runtime_error("ROCm staging pool is closing");
    }

    [[nodiscard]] static std::runtime_error transfer_pool_closing_error() {
        return std::runtime_error(
                "hipStreamCreateWithFlags failed with "
                "hipErrorContextIsDestroyed: TransferStreamPool is closing");
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
    [[nodiscard]] static bool synchronize_stream_noexcept(
            stream_type stream) noexcept {
        if (consume_submission_fault(SubmissionFault::stream_synchronize)) {
            return false;
        }
        return stream == nullptr
                || hipStreamSynchronize(stream) == hipSuccess;
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

    static void after_grid_stride_launch() {
        if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
            check_hip("HIP copy kernel launch", hipErrorInvalidValue);
        }
    }

    [[nodiscard]] static constexpr const char* scatter_kernel_operation()
            noexcept {
        return "HIP kernel launch";
    }

    [[nodiscard]] static constexpr const char* gather_kernel_operation()
            noexcept {
        return "HIP kernel launch";
    }

    [[nodiscard]] static constexpr const char* copy_kernel_operation()
            noexcept {
        return "HIP kernel launch";
    }

    [[nodiscard]] static constexpr const char* backend_label() noexcept {
        return "ROCm";
    }
};
using EventRingState = iom::detail::EventRingState<gpu_policy>;

using StagingSlotPool = iom::detail::StagingSlotPool<gpu_policy>;
using TransferStreamPool = iom::detail::TransferStreamPool<gpu_policy>;


void region_from_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        int device_ordinal, const TensorView& destination,
        std::span<const std::byte> source);

void region_to_host(
        TransferStreamPool& transfer_pool, StagingSlotPool& staging_pool,
        int device_ordinal, const TensorView& source,
        std::span<std::byte> destination);

[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, detail::QueueResourceProvider& resource_provider,
        int device_ordinal, detail::RegistryState& registry_state);

#ifdef IOM_ENABLE_TESTING
// Observed fixed resource geometry of one live queue: the reserved
// partition (device address of its first 512-byte slot inside the Device
// metadata backing), the immutable per-queue C, the eagerly created C
// completion resources, and the current fixed-slot bookkeeping state.
struct QueueResourceSnapshot {
    std::size_t slot_count = 0;
    void* device_base = nullptr;
    std::size_t slot_stride = 0;
    std::size_t events_total = 0;
    std::size_t events_in_use = 0;
    std::size_t slots_in_use = 0;
    std::size_t slots_protected = 0;
};

void queue_resource_snapshot_for_testing(
        DeviceOps& queue, QueueResourceSnapshot& snapshot);

// Attempts a covering proof for every queue lease retained by unknown
// completion on this Device's boundary; reclaimed leases return their
// partition and queue-count reservation.
void reclaim_retained_queue_leases_for_testing(Device& device);
#endif  // IOM_ENABLE_TESTING

}  // namespace iom::rocm_detail

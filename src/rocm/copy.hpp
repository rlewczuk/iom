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
#include "iom/detail/outstanding_work_registry.hpp"
#include "iom/iom.hpp"

namespace iom::detail {
// Device descriptor of the shared 16x16 tiled RMSNorm operation
// (src/shared/standard_tiled_rmsnorm.inl). The ROCm translation unit is the
// only place where its definition is needed.
struct RmsnormMetadata;
// Device descriptor of the shared tiled scalar linear projection
// (src/shared/standard_tiled_linear.inl). Like the RMSNorm descriptor it is
// only needed inside the ROCm translation unit.
struct LinearMetadata;
// Device descriptor of the shared SiLU queue dispatch metadata. The
// operation is intentionally unported here; the descriptor preserves the
// callback seam without retaining a borrowed TensorView.
struct SiluMetadata;
// Device descriptor of the shared tiled RoPE operation
// (src/shared/standard_tiled_rope.inl).
struct RopeMetadata;
}  // namespace iom::detail

namespace iom::rocm_detail {
enum class SubmissionFault {
    none,
    event_create,
    queue_event_create,
    queue_stream_create,
    third_plane_launch,
    embedding_status_copy,
    event_record,
    stream_synchronize,
    registration,
    outcome_insertion,
    // SiLU's own accepted-failure seam: consumed by `launch_silu` before the
    // native launch, so an armed failure never starts the device operation
    // and never counts as a native launch.
    silu_launch,
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
// Counters over the dispatched linear path. A BF16 submission that reached
// the queue must increment the native counter exactly once and never the
// scalar counter, so the conformance evidence connects an accepted OID to
// the executed native kernel instead of trusting the source text.
extern std::atomic<std::size_t> linear_native_bf16_launch_count_for_testing;
extern std::atomic<std::size_t> linear_scalar_launch_count_for_testing;
// Counter over the dispatched SiLU path. An accepted submission must reach
// the HIP kernel exactly once, and an armed `silu_launch` failure must not
// reach it at all, so the conformance evidence connects an accepted OID to
// the executed device kernel instead of trusting the source text.
extern std::atomic<std::size_t> silu_launch_count_for_testing;
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

    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)free_attempt(
                    address, AllocationClass::operation_metadata,
                    AllocationPhase::post_publication);
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
    static void copy_status_to_host(
            stream_type stream, void* destination, const void* source,
            std::size_t bytes) {
        hipError_t status = hipMemcpyAsync(
                destination, source, bytes, hipMemcpyDeviceToHost, stream);
        if (consume_submission_fault(SubmissionFault::embedding_status_copy)) {
            status = hipErrorInvalidValue;
        }
        check_hip("hipMemcpyAsync embedding status DtoH", status);
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

    static void after_embedding_launch() {
        if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
            check_hip(gather_kernel_operation(), hipErrorInvalidValue);
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

    // Cache-row append is a direct raw-word update over the queue's existing
    // nonblocking stream.  The request is an immutable common snapshot; the
    // ROCm wrapper derives a compact device descriptor and launches no staging
    // or workspace allocation.
    template <typename Request>
    static void launch_cache_append(
            stream_type stream, const Request& request);
    // RMSNorm seams. The complete backend-parameterized device operation
    // lives in src/shared/standard_tiled_rmsnorm.inl; the ROCm RMSNorm
    // wrapper leaf launches it on the queue's existing nonblocking stream.
    // Common admission has already restricted this immutable capability
    // predicate to the nine applicable floating leaves.
    [[nodiscard]] static constexpr const char* rmsnorm_kernel_operation()
            noexcept {
        return "HIP RMSNorm kernel launch";
    }

    [[nodiscard]] static constexpr bool rmsnorm_supported() noexcept {
        return true;
    }

    static void launch_rmsnorm(
            stream_type stream, const detail::RmsnormMetadata& metadata);

    // SiLU is ported: the HIP kernel in copy.hip evaluates every admitted
    // logical element through the shared stable evaluator and named-format
    // codec on the queue's existing nonblocking stream. The predicate is a
    // compile-time leaf statement only, exactly as the RMSNorm and RoPE
    // predicates are; the common facade has already restricted it to the nine
    // applicable floating leaves before the queue is reached.
    [[nodiscard]] static constexpr bool silu_supported() noexcept {
        return true;
    }

    static void launch_silu(
            stream_type stream, const detail::SiluMetadata& metadata);

    [[nodiscard]] static constexpr const char* silu_kernel_operation()
            noexcept {
        return "HIP SiLU kernel launch";
    }

    // The shared standard-tiled Rope producer is launched by the ROCm wrapper
    // on the queue's existing nonblocking stream. Common admission restricts
    // this immutable capability to the nine applicable floating leaves.
    [[nodiscard]] static constexpr bool rope_supported() noexcept {
        return true;
    }

    static void launch_rope(
            stream_type stream, const detail::RopeMetadata& metadata);

    [[nodiscard]] static constexpr const char* rope_kernel_operation()
            noexcept {
        return "HIP RoPE kernel launch";
    }

    // Linear projection seams. The complete backend-parameterized device
    // operation lives in src/shared/standard_tiled_linear.inl; the ROCm
    // wrapper below launches it on the queue's existing nonblocking stream.
    // Common admission has already restricted this immutable capability
    // predicate to the twenty-one applicable leaves, and this port carries
    // every one of them: the shared scalar path covers the twenty non-BF16
    // leaves and the native BF16 specialization in this translation unit
    // covers `BF16`. `linear_supported` is therefore a compile-time leaf
    // statement only; the device fact behind the native leaf (the proven
    // GFX12 wave32 WMMA facility) is the separate runtime gate below.
    [[nodiscard]] static constexpr const char* linear_kernel_operation()
            noexcept {
        return "HIP linear kernel launch";
    }

    [[nodiscard]] static constexpr bool linear_supported(
            DataType type) noexcept {
        switch (type) {
            case DataType::I2: case DataType::U2:
            case DataType::I4: case DataType::U4:
            case DataType::I8: case DataType::U8:
            case DataType::I16: case DataType::U16:
            case DataType::I32: case DataType::U32:
            case DataType::I64: case DataType::U64:
            case DataType::F4_E2M1:
            case DataType::F6_E2M3: case DataType::F6_E3M2:
            case DataType::F8_E4M3FN: case DataType::F8_E5M2:
            case DataType::F16:
            case DataType::BF16:
            case DataType::F32:
            case DataType::F64:
                return true;
            case DataType::BOOL:
            case DataType::F8_E8M0:
                return false;
        }
        return false;
    }

    // Native BF16 capability of one exact device ordinal: the linear
    // projection's native path is direct GFX12 wave32 WMMA, so a device
    // without that proven facility is `Unsupported` rather than a queued
    // failure. The predicate reads the ordinal's immutable device
    // properties and allocates, registers, leases, submits, and
    // synchronizes nothing, so the pure requirement query stays pure.
    [[nodiscard]] static bool linear_native_bf16_available(
            context_type device_ordinal) noexcept;

    // Exact `{bytes, alignment}` of one admitted native-BF16 linear request:
    // alignment 32 over the checked path-specific sum
    // `A32(P*pad16(R)*pad16(I)*2) + A32(P*pad16(R)*pad16(O)*2)`, whose two
    // terms are the packed input and packed product regions. Every product,
    // `pad16`, byte conversion, alignment round-up, and addition is checked
    // and rejects with the established overflow category.
    [[nodiscard]] static WorkspaceRequirements linear_native_bf16_requirements(
            const TensorView& x, const TensorView& w, std::size_t rows);

    static void launch_linear(
            stream_type stream, const detail::LinearMetadata& metadata);

    [[nodiscard]] static constexpr const char* backend_label() noexcept {

        return "ROCm";
    }

};
using EventRingState = iom::detail::EventRingState<gpu_policy>;

void region_from_host(
        hipStream_t transfer_stream, int device_ordinal,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& destination, RawWorkspaceView workspace,
        std::span<const std::byte> source, bool& resource_poisoned);

void region_to_host(
        hipStream_t transfer_stream, int device_ordinal,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& source, RawWorkspaceView workspace,
        std::span<std::byte> destination, bool& resource_poisoned);

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

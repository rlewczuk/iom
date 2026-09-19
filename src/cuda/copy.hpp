#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <atomic>

#include "driver.hpp"
#include "../shared/event_ring.hpp"
#include "../shared/metadata_slot_pool.hpp"
#include "../shared/queue_resources.hpp"
#include "iom/detail/outstanding_work_registry.hpp"

#include "iom/iom.hpp"

namespace iom::detail {
// Device descriptor of the shared 16x16 tiled RMSNorm operation
// (src/shared/standard_tiled_rmsnorm.inl). The CUDA translation unit is the
// only place where its definition is needed.
struct RmsnormMetadata;
// Device descriptor of the shared tiled scalar linear projection
// (src/shared/standard_tiled_linear.inl), declared for the same reason.
struct LinearMetadata;
// Device descriptor of the shared SiLU queue dispatch metadata. The
// operation is intentionally unported here; the descriptor preserves the
// callback seam without retaining a borrowed TensorView.
struct SiluMetadata;
// Device descriptor of the shared tiled RoPE operation
// (src/shared/standard_tiled_rope.inl).
struct RopeMetadata;
}  // namespace iom::detail

namespace iom::cuda_detail {
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
    // Capability fault point: the runtime BF16 WMMA facility fact is a
    // property of the exact device and of the loaded image, so an injected
    // absence is the only way the unsupported-device capability path is
    // observable on a device that has the facility.
    bf16_wmma_facility,
};
void inject_submission_fault_for_testing(SubmissionFault fault) noexcept;
[[nodiscard]] bool consume_submission_fault(
        SubmissionFault fault) noexcept;

// ---------------------------------------------------------------------------
// Runtime BF16 WMMA facility of one exact CUDA device. The native BF16 linear
// specialization of src/cuda/linear.cu is a direct `<mma.h>`
// BF16-input/FP32-accumulate route, so its capability is a runtime fact and
// never a claim read off the compiled source or a disassembly: it requires
// both the device's BF16 tensor-core facility (compute capability 8.0 or
// newer) and a loaded image of that specialization that was compiled with the
// BF16 WMMA statements. A device or a build without either reports the
// established `Unsupported` before any launch, and no fallback of any kind is
// substituted.
// ---------------------------------------------------------------------------

// The architecture the loadable image of the native BF16 specialization was
// compiled for, or zero when the current device has no loadable image at all.
// The value comes from the loaded image itself (the runtime reports the
// arch-dependent device constant of the module it loaded), so a `compute_75`
// JIT image reports a pre-Ampere arch and can never be mistaken for the
// specialization.
[[nodiscard]] unsigned int linear_bf16_wmma_image_arch() noexcept;

// True exactly when this device provides the facility the native
// specialization requires. The query performs no allocation, submission, or
// queue-state change, and its result is captured once per queue.
[[nodiscard]] bool linear_bf16_wmma_facility(CUdevice device) noexcept;

// The native BF16 specialization of the shared linear descriptor
// (src/cuda/linear.cu). Only that translation unit includes `<mma.h>`; this
// seam stays vendor-neutral, exactly like `launch_linear`.
void launch_linear_bf16(
        cudaStream_t stream, const detail::LinearMetadata& metadata);

#ifdef IOM_ENABLE_TESTING
// Counters over the policy's native queue-resource lifecycle. Queue setup
// creates every stream and completion resource eagerly; tests assert that a
// rejected fifth queue creates none of them.
extern std::atomic<std::size_t> event_create_count_for_testing;
extern std::atomic<std::size_t> event_destroy_count_for_testing;
extern std::atomic<std::size_t> stream_create_count_for_testing;
extern std::atomic<std::size_t> stream_destroy_count_for_testing;
#endif  // IOM_ENABLE_TESTING

inline void check_cuda_kernel(const char* operation, cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
                std::string(operation) + " failed with "
                + cudaGetErrorName(status) + ": "
                + cudaGetErrorString(status));
    }
}
struct CacheAppendLaunchView {
    std::size_t rank = 0;
    std::array<std::size_t, 8> dimensions{};
    std::array<std::size_t, 6> plane_strides{};
    std::size_t plane_offset = 0;
    DataType data_type = DataType::BOOL;
    void* native_handle = nullptr;
};

struct CacheAppendLaunchRequest {
    CacheAppendLaunchView source;
    CacheAppendLaunchView destination;
    std::size_t a = 0;
};

void launch_cache_append_native(
        cudaStream_t stream, const CacheAppendLaunchRequest& request);


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
            check_cuda_kernel(
                    "cudaEventCreateWithFlags", cudaErrorInvalidValue);
        }
    }

    static void check_create_queue_stream_fault() {
        if (consume_submission_fault(
                    SubmissionFault::queue_stream_create)) {
            check_cuda_kernel(
                    "cudaStreamCreateWithFlags", cudaErrorInvalidValue);
        }
    }

    [[nodiscard]] static stream_type create_queue_stream() {
        check_create_queue_stream_fault();
        stream_type stream = nullptr;
        check_cuda_kernel(
                "cudaStreamCreateWithFlags",
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#ifdef IOM_ENABLE_TESTING
        ++stream_create_count_for_testing;
#endif
        return stream;
    }

    static void destroy_queue_stream_noexcept(stream_type stream) noexcept {
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
#ifdef IOM_ENABLE_TESTING
            ++stream_destroy_count_for_testing;
#endif
        }
    }
    static void synchronize_stream(stream_type stream) {
        check_cuda_kernel(
                "cudaStreamSynchronize", cudaStreamSynchronize(stream));
    }
    [[nodiscard]] static bool synchronize_stream_noexcept(
            stream_type stream) noexcept {
        if (consume_submission_fault(SubmissionFault::stream_synchronize)) {
            return false;
        }
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
#ifdef IOM_ENABLE_TESTING
        ++event_create_count_for_testing;
#endif
    }

    static void destroy_event_noexcept(event_type event) noexcept {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
#ifdef IOM_ENABLE_TESTING
            ++event_destroy_count_for_testing;
#endif
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

    [[nodiscard]] static bool record_event_no_fault(
            event_type event, stream_type stream) noexcept {
        cudaError_t status = cudaEventRecord(event, stream);
        if (consume_submission_fault(SubmissionFault::event_record)) {
            status = cudaErrorInvalidValue;
        }
        return status == cudaSuccess;
    }

    [[nodiscard]] static void* allocate(std::size_t bytes) {
        device_pointer address = 0;
        // Device-side copy metadata slots: operation metadata backing.
        check_cuda(
                "cuMemAlloc",
                allocation_attempt(
                        &address, bytes,
                        AllocationClass::operation_metadata,
                        AllocationPhase::post_publication));
        return reinterpret_cast<void*>(address);
    }

    static void free_noexcept(void* address) noexcept {
        if (address != nullptr) {
            (void)free_attempt(
                    reinterpret_cast<device_pointer>(address),
                    AllocationClass::operation_metadata,
                    AllocationPhase::post_publication);
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

    static void copy_status_to_host(
            stream_type stream, void* destination, const void* source,
            std::size_t bytes) {
        cudaError_t status = cudaMemcpyAsync(
                destination, source, bytes, cudaMemcpyDeviceToHost, stream);
        if (consume_submission_fault(SubmissionFault::embedding_status_copy)) {
            status = cudaErrorInvalidValue;
        }
        check_cuda_kernel("cudaMemcpyAsync embedding status DtoH", status);
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

    static void after_embedding_launch() {
        if (consume_submission_fault(SubmissionFault::third_plane_launch)) {
            check_cuda_kernel(gather_kernel_operation(), cudaErrorInvalidValue);
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

    // RMSNorm uses the shared logical-row kernel and the queue's existing
    // nonblocking stream. Capability is limited to all nine applicable signed
    // floating leaves; common admission keeps inapplicable and quantized
    // leaves explicitly Unsupported.
    [[nodiscard]] static constexpr const char* rmsnorm_kernel_operation()
            noexcept {
        return "CUDA RMSNorm kernel launch";
    }

    [[nodiscard]] static constexpr bool rmsnorm_supported() noexcept {
        return true;
    }

    static void launch_rmsnorm(
            stream_type stream, const detail::RmsnormMetadata& metadata);
    // Cache append uses one direct packed-word CUDA launch. The queue keeps
    // the immutable request and owner/workspace lifetimes; this policy adds
    // only its already-created nonblocking stream launch.
    template <typename Request>
    static void launch_cache_append(
            stream_type stream, const Request& request) {
        CacheAppendLaunchRequest native;
        native.source.rank = request.source.rank;
        native.source.dimensions = request.source.dimensions;
        native.source.plane_strides = request.source.plane_strides;
        native.source.plane_offset = request.source.plane_offset;
        native.source.data_type = request.source.data_type;
        native.source.native_handle = request.source.native_handle;
        native.destination.rank = request.destination.rank;
        native.destination.dimensions = request.destination.dimensions;
        native.destination.plane_strides = request.destination.plane_strides;
        native.destination.plane_offset =
                request.destination.plane_offset;
        native.destination.data_type = request.destination.data_type;
        native.destination.native_handle =
                request.destination.native_handle;
        native.a = request.a;
        launch_cache_append_native(stream, native);
    }


    [[nodiscard]] static constexpr const char* cache_append_kernel_operation()
            noexcept {
        return "CUDA cache append kernel launch";
    }


    // SiLU remains an explicit Unsupported policy stub until the independent
    // CUDA wrapper supplies a real device launch. The common queue therefore
    // rejects it before owner registration, sequence consumption, metadata,
    // event, or stream effects.
    [[nodiscard]] static constexpr bool silu_supported() noexcept {
        return false;
    }

    static void launch_silu(
            stream_type stream, const detail::SiluMetadata& metadata);

    [[nodiscard]] static constexpr const char* silu_kernel_operation()
            noexcept {
        return "CUDA SiLU kernel launch";
    }

    // CUDA uses the shared standard-tiled RoPE kernel on the queue's
    // already-created nonblocking stream.  Common admission and the queue
    // retain the immutable request, metadata slot, owner leases, and
    // completion proof; this policy contributes only the launch boundary.
    [[nodiscard]] static constexpr bool rope_supported() noexcept {
        return true;
    }

    static void launch_rope(
            stream_type stream, const detail::RopeMetadata& metadata);

    [[nodiscard]] static constexpr const char* rope_kernel_operation()
            noexcept {
        return "CUDA RoPE kernel launch";
    }

    // Linear projection capability: exactly the twenty non-BF16 applicable
    // leaves on the shared tiled scalar path plus `BF16` on the native
    // specialization of src/cuda/linear.cu. The inapplicable `BOOL`/`F8_E8M0`
    // leaves and non-`NONE` quantization never reach this predicate, and `F64`
    // uses the device's native double-precision arithmetic, which every CUDA
    // device of this toolchain provides. `BF16` is the one leaf whose
    // availability is a runtime device fact rather than a property of this
    // build: this policy is the queue policy of a device that has the BF16
    // WMMA facility, and `gpu_policy_scalar_linear` below is the exact queue
    // policy of a device that does not.
    [[nodiscard]] static bool linear_supported(DataType data_type) noexcept;

    // The CUDA wrapper of the shared tiled projection
    // (src/shared/standard_tiled_linear.inl). It supplies the queue's
    // already-created nonblocking stream, dispatches the `BF16` leaf to the
    // native WMMA specialization, and adds no stream, allocation, staging, or
    // synchronization of its own.
    static void launch_linear(
            stream_type stream, const detail::LinearMetadata& metadata);

    [[nodiscard]] static constexpr const char* linear_kernel_operation()
            noexcept {
        return "CUDA linear kernel launch";
    }

    [[nodiscard]] static constexpr const char* backend_label() noexcept {
        return "CUDA";
    }
};

// The queue policy of a CUDA device whose resolved runtime BF16 WMMA facility
// is absent, for example a device below compute capability 8.0 or a loaded
// image without the specialization. The shared queue template is instantiated
// once per capability variant, so such a device reports the established
// `Unsupported` for `BF16` before registration, credit, metadata, or launch —
// never a scalar, emulated, or host substitute — while every other leaf keeps
// the established scalar path unchanged. The fact is resolved once, at queue
// creation, and no submission re-reads it.
struct gpu_policy_scalar_linear final : gpu_policy {
    [[nodiscard]] static bool linear_supported(DataType data_type) noexcept;
};
using EventRingState = iom::detail::EventRingState<gpu_policy>;


void region_from_host(
        cudaStream_t transfer_stream, CUcontext context,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& destination, RawWorkspaceView workspace,
        std::span<const std::byte> source, bool& resource_poisoned);

void region_to_host(
        cudaStream_t transfer_stream, CUcontext context,
        const Device& device, detail::RegistryState& registry_state,
        const TensorView& source, RawWorkspaceView workspace,
        std::span<std::byte> destination, bool& resource_poisoned);
[[nodiscard]] std::unique_ptr<DeviceOps> make_queue(
        const Device& device, detail::QueueResourceProvider& resource_provider,
        CUcontext context, detail::RegistryState& registry_state,
        bool bf16_wmma_facility);

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

}  // namespace iom::cuda_detail

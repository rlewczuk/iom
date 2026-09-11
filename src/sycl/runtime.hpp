#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

namespace iom::sycl_detail {

    // Test-only observation points. They do not select a device or retain
    // runtime state; the backend owns the context independently of this seam.
    struct ContextCalls {
        void (*context_created)() = nullptr;
        void (*context_destroyed)() = nullptr;
        void (*context_ready)(const sycl::context&) = nullptr;
    };
    struct LaunchCalls {
        void (*kernel_launched)() = nullptr;
    };

    extern LaunchCalls launch_calls;

    extern ContextCalls context_calls;

    // -------------------------------------------------------------------
    // Planned backend-local native-allocation seam, shaped like the CUDA and
    // ROCm backend seams. IOM call sites that know the purpose of a
    // device-USM allocation route sycl::malloc_device/sycl::free (device
    // pointers only) through the wrappers below. Host USM (malloc_host,
    // malloc_shared) and vendor-internal allocations never cross the seam
    // and stay outside the IOM-native record. In non-testing builds the
    // wrappers degrade to the direct SYCL call and emit nothing; testing
    // builds (IOM_ENABLE_TESTING) emit an attempt record completed with
    // success/failure, byte count, operation class, lifecycle phase, and
    // free/allocation kind, and may replace the underlying calls to inject
    // boundary failure. Classification is always routed at the call site and
    // never inferred from pointers or byte totals. Future factory arena
    // reservations classify their setup calls as data_backing/
    // metadata_backing; current internal sites classify device-side copy
    // metadata slots (operation_metadata), host-transfer staging, and the
    // binary-fallback temporary device buffers (staging). No
    // synchronization, lock, ordering change, or production dependency is
    // introduced; the record callbacks run on the calling thread.
    // -------------------------------------------------------------------
    enum class AllocationClass {
        data_backing,
        metadata_backing,
        operation_metadata,
        staging,
        other_iom_setup,
    };

    enum class AllocationKind { allocate, free };

    enum class AllocationPhase { setup, post_publication };

#ifdef IOM_ENABLE_TESTING

    struct AllocationRecord {
        AllocationClass classification = AllocationClass::other_iom_setup;
        AllocationKind kind = AllocationKind::allocate;
        AllocationPhase phase = AllocationPhase::post_publication;
        // Allocation byte count; frees carry 0 (the free call sites do not
        // know the size, so allocate/free association is anchored on
        // address).
        std::size_t bytes = 0;
        // Final outcome: a non-null allocation result, or a free that did
        // not throw. A failed allocation keeps address null; an erroring
        // cleanup free keeps its address and records failure instead of
        // disappearing.
        bool succeeded = false;
        void* address = nullptr;
    };

    struct AllocationObserver {
        // Invoked before the native call with the planned attempt.
        void (*attempt)(const AllocationRecord&) = nullptr;
        // Invoked after the native call; record.succeeded and
        // record.address are final.
        void (*complete)(const AllocationRecord&) = nullptr;
    };

    struct AllocationCalls {
        void* (*device_alloc)(
                std::size_t bytes, const sycl::device& device,
                const sycl::context& context) = nullptr;
        void (*device_free)(
                void* pointer, const sycl::context& context) = nullptr;
    };

    extern AllocationObserver allocation_observer;
    extern AllocationCalls allocation_calls;

#endif  // IOM_ENABLE_TESTING

    [[nodiscard]] inline void* alloc_attempt_device(
            std::size_t bytes, const sycl::device& device,
            const sycl::context& context, AllocationClass classification,
            AllocationPhase phase) {
#ifdef IOM_ENABLE_TESTING
        AllocationRecord record{};
        record.classification = classification;
        record.kind = AllocationKind::allocate;
        record.phase = phase;
        record.bytes = bytes;
        if (allocation_observer.attempt != nullptr) {
            allocation_observer.attempt(record);
        }
        void* result = nullptr;
        try {
            result = allocation_calls.device_alloc != nullptr
                    ? allocation_calls.device_alloc(bytes, device, context)
                    : sycl::malloc_device(bytes, device, context);
        } catch (...) {
            record.succeeded = false;
            if (allocation_observer.complete != nullptr) {
                allocation_observer.complete(record);
            }
            throw;
        }
        record.address = result;
        record.succeeded = result != nullptr;
        if (allocation_observer.complete != nullptr) {
            allocation_observer.complete(record);
        }
        return result;
#else
        (void)classification;
        (void)phase;
        return sycl::malloc_device(bytes, device, context);
#endif
    }

    inline void free_attempt_device(
            void* pointer, const sycl::context& context,
            AllocationClass classification, AllocationPhase phase) noexcept {
        if (pointer == nullptr) {
            return;
        }
#ifdef IOM_ENABLE_TESTING
        AllocationRecord record{};
        record.classification = classification;
        record.kind = AllocationKind::free;
        record.phase = phase;
        record.address = pointer;
        if (allocation_observer.attempt != nullptr) {
            allocation_observer.attempt(record);
        }
        try {
            if (allocation_calls.device_free != nullptr) {
                allocation_calls.device_free(pointer, context);
            } else {
                sycl::free(pointer, context);
            }
            record.succeeded = true;
        } catch (...) {
            record.succeeded = false;
        }
        if (allocation_observer.complete != nullptr) {
            allocation_observer.complete(record);
        }
#else
        (void)classification;
        (void)phase;
        try {
            sycl::free(pointer, context);
        } catch (...) {
        }
#endif
    }

    [[nodiscard]] std::size_t eligible_device_count();

}  // namespace iom::sycl_detail

#pragma once

#include <cuda.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace iom::cuda_detail {

struct DriverCalls {
    CUresult (*primary_ctx_retain)(CUcontext*, CUdevice)
            = &cuDevicePrimaryCtxRetain;
    CUresult (*ctx_set_current)(CUcontext) = &cuCtxSetCurrent;
    CUresult (*primary_ctx_release)(CUdevice) = &cuDevicePrimaryCtxRelease;
    CUresult (*init)(unsigned int) = &cuInit;
    CUresult (*device_get_count)(int*) = &cuDeviceGetCount;
    CUresult (*device_get)(CUdevice*, int) = &cuDeviceGet;
    // The runtime capability query behind the native BF16 WMMA facility fact:
    // no context needs to be current for it, so the fact belongs to the exact
    // device and not to whatever context a calling thread happens to hold.
    CUresult (*device_get_attribute)(int*, CUdevice_attribute, CUdevice)
            = &cuDeviceGetAttribute;
};

extern DriverCalls driver_calls;

// ---------------------------------------------------------------------------
// Backend-local native-allocation seam. IOM call sites that know the
// purpose of a device allocation route their cuMemAlloc/cuMemFree boundary
// through the wrappers below. In production builds the wrappers degrade to
// the direct driver call and emit nothing; testing builds (IOM_ENABLE_TESTING)
// emit an attempt record and complete it with success/failure, byte count,
// operation class, lifecycle phase, and free/allocation kind, and may replace
// the underlying driver calls to inject boundary failure. Classification is
// always routed at the call site and never inferred from pointers or byte
// totals. Factory arena reservations classify their setup calls as
// data_backing/metadata_backing; current internal sites classify device-side
// copy metadata slots (operation_metadata) and host-transfer staging
// (staging). No synchronization, lock, ordering change, or production
// dependency is introduced; the record callbacks run on the calling thread.
// ---------------------------------------------------------------------------

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
    // Allocation byte count; frees carry 0 (the free call sites do not know
    // the size, so allocate/free association is anchored on address).
    std::size_t bytes = 0;
    // Final driver status: true for CUDA_SUCCESS. A failed allocation keeps
    // address null; an erroring cleanup free keeps its address and records
    // failure instead of disappearing.
    bool succeeded = false;
    void* address = nullptr;
};

struct AllocationObserver {
    // Invoked before the native driver call with the planned attempt.
    void (*attempt)(const AllocationRecord&) = nullptr;
    // Invoked after the native driver call; record.succeeded and
    // record.address are final.
    void (*complete)(const AllocationRecord&) = nullptr;
};

struct AllocationCalls {
    CUresult (*mem_alloc)(CUdeviceptr*, std::size_t) = &cuMemAlloc;
    CUresult (*mem_free)(CUdeviceptr) = &cuMemFree;
};

extern AllocationObserver allocation_observer;
extern AllocationCalls allocation_calls;

#endif  // IOM_ENABLE_TESTING

[[nodiscard]] inline CUresult allocation_attempt(
        CUdeviceptr* address, std::size_t bytes,
        AllocationClass classification, AllocationPhase phase) {
#ifdef IOM_ENABLE_TESTING
    AllocationRecord record{};
    record.classification = classification;
    record.kind = AllocationKind::allocate;
    record.phase = phase;
    record.bytes = bytes;
    if (allocation_observer.attempt != nullptr) {
        allocation_observer.attempt(record);
    }
    const CUresult status = allocation_calls.mem_alloc(address, bytes);
    record.succeeded = status == CUDA_SUCCESS;
    if (record.succeeded) {
        record.address = reinterpret_cast<void*>(*address);
    }
    if (allocation_observer.complete != nullptr) {
        allocation_observer.complete(record);
    }
    return status;
#else
    (void)classification;
    (void)phase;
    return cuMemAlloc(address, bytes);
#endif
}

inline CUresult free_attempt(
        CUdeviceptr address, AllocationClass classification,
        AllocationPhase phase) {
#ifdef IOM_ENABLE_TESTING
    AllocationRecord record{};
    record.classification = classification;
    record.kind = AllocationKind::free;
    record.phase = phase;
    record.address = reinterpret_cast<void*>(address);
    if (allocation_observer.attempt != nullptr) {
        allocation_observer.attempt(record);
    }
    const CUresult status = allocation_calls.mem_free(address);
    record.succeeded = status == CUDA_SUCCESS;
    if (allocation_observer.complete != nullptr) {
        allocation_observer.complete(record);
    }
    return status;
#else
    (void)classification;
    (void)phase;
    return cuMemFree(address);
#endif
}

}  // namespace iom::cuda_detail

namespace iom {

[[nodiscard]] std::runtime_error cuda_error(
        const char* operation, CUresult status);
void check_cuda(const char* operation, CUresult status);

inline std::runtime_error cuda_error(
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

inline void check_cuda(const char* operation, CUresult status) {
    if (status != CUDA_SUCCESS) {
        throw cuda_error(operation, status);
    }
}

}  // namespace iom

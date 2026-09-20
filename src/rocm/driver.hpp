#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>

namespace iom::rocm_detail {

// ---------------------------------------------------------------------------
// Backend-local native-allocation seam, shaped like the CUDA backend
// seam in src/cuda/driver.hpp. IOM call sites that know the purpose of a
// device allocation route their hipMalloc/hipFree boundary through the
// wrappers below. In production builds the wrappers degrade to the direct
// runtime call and emit nothing; testing builds (IOM_ENABLE_TESTING) emit an
// attempt record and complete it with success/failure, byte count, operation
// class, lifecycle phase, and free/allocation kind, and may replace the
// underlying runtime calls to inject boundary failure. Classification is
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

// Exact native causal grouped-query SDPA capability of one HIP device
// ordinal, shaped like the CUDA backend's `linear_bf16_wmma_facility`
// predicate: the operation's checked route is the GFX12 wave32 BF16 WMMA
// image with its completed native QK/PV and nonmatrix stages, so the
// installed `gfx1036`, a wave64 device, and any target whose WMMA execution
// evidence is absent report false and keep the established `Unsupported`
// behaviour instead of a broken or emulated route. The predicate reads
// immutable device properties only: it allocates, registers, leases, submits,
// and synchronizes nothing.
[[nodiscard]] bool sdpa_native_capability(int device_ordinal) noexcept;

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
    // Final runtime status: true for hipSuccess. A failed allocation keeps
    // address null; an erroring cleanup free keeps its address and records
    // failure instead of disappearing.
    bool succeeded = false;
    void* address = nullptr;
};

struct AllocationObserver {
    // Invoked before the native runtime call with the planned attempt.
    void (*attempt)(const AllocationRecord&) = nullptr;
    // Invoked after the native runtime call; record.succeeded and
    // record.address are final.
    void (*complete)(const AllocationRecord&) = nullptr;
};

struct AllocationCalls {
    hipError_t (*mem_alloc)(void**, std::size_t) = &hipMalloc;
    hipError_t (*mem_free)(void*) = &hipFree;
};

extern AllocationObserver allocation_observer;
extern AllocationCalls allocation_calls;

#endif  // IOM_ENABLE_TESTING

[[nodiscard]] inline hipError_t allocation_attempt(
        void** address, std::size_t bytes, AllocationClass classification,
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
    const hipError_t status = allocation_calls.mem_alloc(address, bytes);
    record.succeeded = status == hipSuccess;
    if (record.succeeded) {
        record.address = *address;
    }
    if (allocation_observer.complete != nullptr) {
        allocation_observer.complete(record);
    }
    return status;
#else
    (void)classification;
    (void)phase;
    return hipMalloc(address, bytes);
#endif
}

inline hipError_t free_attempt(
        void* address, AllocationClass classification,
        AllocationPhase phase) {
#ifdef IOM_ENABLE_TESTING
    AllocationRecord record{};
    record.classification = classification;
    record.kind = AllocationKind::free;
    record.phase = phase;
    record.address = address;
    if (allocation_observer.attempt != nullptr) {
        allocation_observer.attempt(record);
    }
    const hipError_t status = allocation_calls.mem_free(address);
    record.succeeded = status == hipSuccess;
    if (allocation_observer.complete != nullptr) {
        allocation_observer.complete(record);
    }
    return status;
#else
    (void)classification;
    (void)phase;
    return hipFree(address);
#endif
}

}  // namespace iom::rocm_detail
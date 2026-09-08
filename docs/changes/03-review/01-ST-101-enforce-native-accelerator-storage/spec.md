# Enforce native accelerator storage for CUDA and ROCm tensors

**Order:** 01
**Priority:** P0 — prevent invalid allocator storage from reaching accelerator kernels
**Blocked by:** None
**Review source:** `cpp-inference-gpu-stability` — `whole-codebase reviewed state: branch main, clean HEAD 86533aef347935405cb86d465cc5489f5c3d530a (Remove review files.)`
**Finding:** `ST-101`
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 90
**Review scope:** whole-codebase
**Backend scope:** CUDA + ROCm
**Location:** `src/cuda/device.cpp` — `CudaTensor::CudaTensor`; `src/rocm/device.cpp` — `RocmTensor::RocmTensor`

## Outcome

Every tensor successfully constructed by the CUDA or ROCm allocator-taking factory owns unmanaged native device memory for the exact owning context/device or ROCm ordinal, with the allocator-returned address equal to the queried device pointer. Host, pinned, mapped, managed, foreign-device, foreign-context, and pointer-query-failure cases fail synchronously during construction; any rejected non-null allocation is returned exactly once through the same allocator under owning activation, while existing native allocators continue to work.

## Current problem

`allocate_aligned_storage` currently establishes only non-null and 32-byte alignment. `CudaTensor::CudaTensor` and `RocmTensor::RocmTensor` then publish the raw allocator address without checking memory type, unmanaged status, owning device/context, or address identity. `GpuQueue::execute` later submits those raw handles to tiled-copy kernels after checking only device/spec compatibility. Root probes accepted ordinary aligned host storage for both CUDA and ROCm, so a valid-looking tensor can reach asynchronous device work with storage that is not proven usable by its owner; cleanup also lacks a context-qualified guarantee for arbitrary foreign storage. Existing tests use native `cudaMalloc`/`hipMalloc` allocators and do not cover invalid provenance.

## Scope

- Document and enforce the allocator precondition at the CUDA and ROCm public factories: successful tensor allocation must be unmanaged native device storage belonging to the exact owning CUDA context/device or ROCm ordinal, and the allocator's `free` must remain valid under the backend activation protocol.
- After allocation and under owning activation, query CUDA/HIP pointer attributes for memory type, managed status, context, device ordinal, and device-pointer identity. Reject host, pinned/mapped host, managed, peer/foreign-device, foreign-context, address-alias, and query-failure cases before publishing a native handle or allowing queue submission.
- Preserve existing alignment, borrowed allocator lifetime, accepted-native destruction/quarantine, and backend-specific activation semantics. On construction rejection, roll back the original pointer exactly once through that allocator; null allocation still requires no free.

## Implementation references

- **Modify:** `src/cuda/device.cpp` — `CudaTensor::CudaTensor`; this is the CUDA tensor ownership boundary and already activates the owned context for allocation/free.
- **Modify:** `src/rocm/device.cpp` — `RocmTensor::RocmTensor`; this is the ROCm tensor ownership boundary and already activates the owned ordinal for allocation/free.
- **Modify:** `include/iom/cuda/device.hpp` and `include/iom/rocm/device.hpp` — `make_cuda_device`/`make_rocm_device` documentation; state the native-storage and allocator-free preconditions at both public factories, not in the backend-neutral allocator contract.
- **Read:** `include/iom/detail/aligned_storage.hpp` — `allocate_aligned_storage`/`release_aligned_storage`; reuse alignment and rollback conventions.
- **Read:** `src/sycl/device.cpp` — `SyclTensor` allocation validation and constructor rollback; reuse its ownership-safe rejection pattern without adding a common API.
- **Read:** `src/shared/gpu_queue.hpp` — `GpuQueue::execute`; retain its queue behavior and do not move provenance checks into submission.
- **Tests:** `test/cuda/test_cuda_smoke.cpp`, `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_smoke.cpp`, and `test/rocm/test_rocm_conformance.cpp` — existing native allocator and construction/copy patterns; add focused negative construction/rollback coverage beside these tests.

## Requirements

- CUDA must query pointer attributes under the owning context and accept only device memory with unmanaged status, exact owning `CUcontext`, exact device ordinal, and `CU_POINTER_ATTRIBUTE_DEVICE_POINTER` equal to the allocator-returned address. HIP must perform the analogous query under the owning activation and accept only unmanaged device memory, exact owning ordinal/context, and `devicePointer` equal to the returned address.
- Treat every CUDA/HIP attribute-query failure or mismatch as a synchronous incompatible-storage construction error. Do not infer compatibility from a query failure, translate an alias, substitute an allocation, add queue-time fallback, or add a common allocator capability API.
- For every non-null rejection, clear the tensor's stored address and call `release_aligned_storage` exactly once with the original allocator pointer and existing owning activation callback; never call native free directly, free an attribute-derived alias, reset the allocator, or register/quarantine an incompletely constructed tensor. Preserve normal accepted-tensor cleanup.
- Keep existing native CUDA and HIP allocators valid and retain null/misalignment checks. The check establishes domain/context compatibility only; existing allocator lifetime and allocation-extent obligations remain unchanged.

## Non-goals

- Do not change `include/iom/alloc.hpp` into a backend-capability interface or add a shared pointer-provenance registry.
- Do not alter `GpuQueue`, `validate_copy`, tiled-copy kernels, CPU/SYCL/TTNN allocation behavior, peer-access support, cross-device copies, or allocator performance.
- Do not support managed, mapped, pinned, host, or foreign storage through pointer translation or a new handle representation; this task chooses strict unmanaged native storage.

## Acceptance criteria

- [ ] Matching native `cudaMalloc` and `hipMalloc` allocators construct tensors and complete the existing CUDA/ROCm copy and lifetime conformance paths; the returned address is proven to equal the queried device pointer for the exact owner.
- [ ] Aligned ordinary host, pinned/mapped host, managed, foreign-device, foreign-context, address-alias, and attribute-query-failure cases fail synchronously during tensor construction, before queue submission or sequence consumption, and expose no usable native handle.
- [ ] Each rejected non-null allocation is returned exactly once through the caller allocator's `free(original_pointer)` while the owning CUDA context or ROCm ordinal is active; null results are not freed and existing accepted-tensor cleanup remains unchanged.
- [ ] The focused CUDA and ROCm tests observe a stable incompatible-storage error and no registry/quarantine entry or kernel launch for rejected construction.

## Verification

- `cmake --build build-cuda --target iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-cuda -R 'iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests' --output-on-failure`
- `cmake --build build-rocm --target iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-rocm -R 'iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests' --output-on-failure`
- On real CUDA and ROCm hardware, exercise native, aligned-host, managed/mapped (where exposed), foreign-device, foreign-context, pointer-query-failure, null, and misaligned allocators. Assert construction-time failure/success, exact-once rollback under owner activation, no queue submission for rejection, and successful native copies. The root probes did reproduce acceptance of ordinary aligned host storage before remediation; no post-remediation ST-101 negative allocator gate has been run.
- Baseline validation ledger only (not ST-101 coverage): local GNU 15.2 CPU build/tests/bench/conformance passed 4/4; remote CUDA 13.2.78 smoke+conformance, ROCm HIP Clang 23 smoke+conformance, SYCL IntelLLVM 2026.1 smoke+conformance on two enumerated Arc Pro B60 Level Zero GPUs, and TTNN smoke+conformance each passed 2/2. No ST-101 negative allocator, sanitizer, static-analysis, or profiler gate has been run.

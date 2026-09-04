# Pool GPU host-transfer streams and staging buffers

**Order:** 43
**Priority:** P1 — medium, verified performance defect. Synchronous CUDA/ROCm host transfers allocate/free device staging on every call; ROCm also creates/destroys a stream per call and downloads zero the full staging allocation.
**Blocked by:** `29-AR-001-share-cuda-rocm-copy-queue`, `42-PF-002-grid-stride-gpu-copy`, `26-ST-005-dedicate-cuda-transfer-stream`
**Source:** `docs/changes/0001-tensor-view/review.md` — `PF-003`
**Review severity:** medium
**Review verification:** verified (mechanism code-deterministic; magnitude needs GPU timing), confidence 85

## Outcome

CUDA and ROCm `synchronous_transfer` reuse device-owned pools instead of allocating a staging buffer and, on ROCm, a transfer stream for every call. CUDA consumes the CUDA `TransferStreamPool` from ST-005; PF-003 supplies the symmetric ROCm transfer-stream pool and one staging-slot pool per backend. Slots grow geometrically, are reused only after the synchronous call returns, and are discarded after a poisoned transfer. Downloads zero only the partial tail word required by the staging-word contract.

The existing synchronous-at-return behavior is unchanged: every upload or download has completed and every failure has been observed before the public tensor method returns. Pools are owned by the device, not a queue or tensor, and are destroyed before the backend context/device teardown. No public API or allocator contract changes.

## Current failure

CUDA `synchronous_transfer` at `src/cuda/copy.cu:354-405` performs `cuMemAlloc` and `cuMemFree` for each call, uses the legacy default stream, and zeros the full staging allocation for downloads. ROCm `src/rocm/copy.hip:294-343` performs `hipMalloc`/`hipFree` and creates/destroys a non-blocking stream per call. These synchronizing lifecycle calls defeat repeated weight/expert streaming and serialize concurrent host transfers on shared resources.

The review evidence is the cited per-call lifecycle and full-size memset. The fix must preserve the separate caller allocator boundary: backend-private staging is not tensor storage and must not be routed through `iom::Allocator`.

## Scope

- **Consume:** `29-AR-001-share-cuda-rocm-copy-queue`'s backend-neutral `iom::gpu_algorithm::compute_staging_size(std::size_t)` for checked 32-bit-word rounding on both backends.
- **Consume:** `26-ST-005-dedicate-cuda-transfer-stream`'s CUDA `iom::cuda_detail::TransferStreamPool` API and device ownership.
- **Create:** `src/rocm/transfer_pool.hpp` and the matching PF-003 ROCm pool implementation with the same acquire/poison/destroy semantics as the CUDA pool.
- **Create:** `src/cuda/staging_pool.hpp/.cpp` and `src/rocm/staging_pool.hpp/.cpp`, each with a device-side staging-slot pool, RAII lease, geometric growth, cap, poison path, and test-only accounting seam.
- **Modify:** `src/cuda/device.cpp` and `src/rocm/device.cpp` to own stream and staging pools and destroy them while the backend context/device remains usable.
- **Modify:** `src/cuda/copy.hpp/.cu` and `src/rocm/copy.hpp/.hip` to pass typed pool references through `region_from_host`/`region_to_host` into `synchronous_transfer`.
- **Modify:** only the synchronous host-transfer path. PF-002's queued device-to-device copy and PF-006's kernels remain separate consumers of the shared copy boundary.
- **Tests:** add pool accounting/failure tests and concurrent host-transfer coverage to the CUDA/ROCm smoke or conformance targets following existing backend conventions.

## Implementation references

- **Read/consume:** `docs/changes/0001-tensor-view/29-AR-001-share-cuda-rocm-copy-queue/spec.md` — `<iom/gpu_algorithm.hpp>` and exact `compute_staging_size` overflow diagnostic.
- **Read/consume:** `docs/changes/0001-tensor-view/26-ST-005-dedicate-cuda-transfer-stream/spec.md` — CUDA `iom::cuda_detail::TransferStreamPool`, `Scope`, `acquire`, `poison`, `destroy`, and device ownership. Do not duplicate or rename that surface.
- **Read:** `docs/changes/0001-tensor-view/42-PF-002-grid-stride-gpu-copy/spec.md` — queued copy path and per-queue metadata pool; PF-003 must not alter it.
- **Create:** `src/cuda/staging_pool.hpp/.cpp` — CUDA `iom::cuda_detail::StagingSlotPool` with `Lease::staging()`, `Lease::capacity()`, `Lease::poison()`, `acquire(required_bytes)`, `destroy() noexcept`, and `idle_count_for_testing()`.
- **Create:** `src/rocm/staging_pool.hpp/.cpp` — HIP mirror with `hipDeviceptr_t` and HIP allocation/free operations.
- **Create:** `src/rocm/transfer_pool.hpp`; **modify:** `src/rocm/copy.hip` — `iom::rocm_detail::TransferStreamPool` and its member bodies in the matching namespace.
- **Modify:** `src/cuda/device.cpp`, `src/rocm/device.cpp`, `src/cuda/copy.hpp/.cu`, and `src/rocm/copy.hpp/.hip` — typed ownership and transfer call sites.
- **Modify:** `CMakeLists.txt` — add only the new private pool headers/sources to their backend libraries; no common target gets CUDA/HIP dependencies.
- **Tests:** `test/cuda/test_cuda_smoke.cpp`, `test/rocm/test_rocm_smoke.cpp`, and existing backend conformance tests. Preserve full logical byte comparisons and error categories.

### Pool contract

Each staging pool has a bounded slot count, free-slot collection, mutex, and outstanding allocation count. `acquire(required_bytes)` rejects values above `kMaxStagingBytes = 8 GiB` before touching state. If no slot exists it allocates the initial capacity. If a free slot is too small, it allocates the replacement before freeing the old allocation, then installs the replacement. A replacement allocation failure restores the original slot and leaves the outstanding count unchanged. A first allocation failure restores the pre-call count and inserts no slot.

A healthy `Lease` returns its slot to the free collection. `poison()` makes its destructor free and discard the slot, decrementing the outstanding count. `destroy()` frees all free slots, clears the free collection, and runs only after no active lease remains. A test-only failure hook may fail the next backend allocation and a test-only read-only counter may expose accounting; both are private, disabled in production, and do not change the public API.

Each transfer acquires one stream scope and one staging lease. The work body runs upload, scatter/gather, stream synchronization, and any synchronous D2H copy on those resources. A catch block poisons both scopes before rethrowing the original exception. The stream and slot are returned only after all work has completed. On download, zero exactly the bytes from the logical end to the rounded staging-word end; copy only the requested logical destination bytes to the host.

## Requirements

1. `compute_staging_size(view.spec().logical_nbytes())` is called exactly once per synchronous transfer on each backend; its exact `std::overflow_error("GPU transfer staging size overflows")` propagates before pool interaction.
2. CUDA uses ST-005's `iom::cuda_detail::TransferStreamPool`; ROCm uses PF-003's `iom::rocm_detail::TransferStreamPool`. No per-call stream create/destroy remains in `synchronous_transfer`.
3. Both backends use their PF-003 `StagingSlotPool`; no per-call `cuMemAlloc`/`cuMemFree` or `hipMalloc`/`hipFree` remains in `synchronous_transfer`.
4. Pools are device-owned, not queue-owned, and are destroyed before primary-context/device teardown. Multiple independent transfers can hold distinct stream/slot pairs.
5. Growth is geometric and checked. Allocation failure preserves the original slot, free collection, and outstanding count. The first-allocation and grow-allocation failure paths are separately tested.
6. A poisoned stream or slot is destroyed rather than returned to the free pool. Healthy RAII scopes return resources exactly once. Destructors are no-throw effective.
7. Downloads zero only the partial tail word after the logical byte range; full-size staging memset is removed.
8. The `copy_from_host` and `copy_to_host` public methods remain synchronous at return and preserve exact logical bytes, padding, sub-byte packing, and exception categories.
9. Staging remains outside the caller-supplied `iom::Allocator`; public headers and factory signatures are unchanged.
10. The queued PF-002 path, TTNN/CPU/SYCL paths, and ST-005 CUDA pool ownership are not duplicated or altered.
11. Narrow test-only allocation-failure hooks and accounting accessors are private and compiled out or inert in production builds; they are not prohibited instrumentation in the implementation, but they must not become a runtime API.

## Non-goals

- Replacing the tensor-storage allocator or changing `Allocator::alloc/free`.
- Changing queued device-to-device copy, metadata slots, events, queue workers, or PF-006 bit kernels.
- Changing TTNN, CPU, or SYCL transfer paths.
- Pinning the SafeTensors mmap region or redesigning host memory ownership.
- Adding public stream/staging accessors, a global pool, or a global active-backend registry.
- Performance claims without hardware timing; API traces and accounting prove reuse, while timing measures magnitude.

## Acceptance criteria

- [ ] After warmup, repeated CUDA and ROCm host transfers perform no per-call device allocation/free and no per-call stream create/destroy.
- [ ] Concurrent host transfers on one device acquire distinct stream/slot pairs and complete without invalid-handle or data-race diagnostics.
- [ ] CUDA consumes the exact ST-005 pool surface and ROCm uses the exact `iom::rocm_detail::TransferStreamPool` name and ownership.
- [ ] CUDA and ROCm staging-pool tests prove first-allocation failure decrements/restores accounting and grow-allocation failure restores the original slot without decrementing the already-accounted allocation.
- [ ] Healthy leases return to their pool; poisoned leases destroy and discard their resources; device destruction frees every idle resource before backend context/device teardown.
- [ ] `{1,17}` odd sub-byte downloads produce exact logical bytes and zero tail bits with no out-of-bounds access.
- [ ] Existing CUDA/ROCm smoke and conformance tests pass, including asynchronous copy and failure behavior.
- [ ] Source inspection finds zero lifecycle calls in the transfer function bodies and confirms the helper call occurs once per backend.
- [ ] Production builds contain no public pool API, no caller-allocator use for staging, and no common-backend runtime leakage.

## Verification

Follow `.agents/skills/remote-development` for CUDA and ROCm build and execution. Run pool failure/accounting, repeated-transfer, concurrent-transfer, odd-tail, and full conformance cases on each backend under exclusive GPU access. Use API tracing or the private test seam to show allocation/stream lifecycle counts drop after warmup; report wall-time changes separately. Run CUDA and ROCm memory diagnostics on failure and concurrent cases. Build a core/CPU configuration with optional accelerators disabled to prove no dependency leakage. Clean remote mirrors after verification.

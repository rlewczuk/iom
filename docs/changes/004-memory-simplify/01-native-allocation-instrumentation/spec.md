# Instrument standard-GPU native allocations

**Order:** 01
**Priority:** P0 — later two-backing/no-churn work needs a truthful native-boundary oracle.
**Blocked by:** None
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

Add a test-only, backend-driver instrumentation seam that observes every IOM-initiated CUDA `cuMemAlloc`/`cuMemFree`, ROCm `hipMalloc`/`hipFree`, and SYCL `sycl::malloc_device`/`sycl::free` boundary in current internal metadata, staging, and SYCL binary paths and in the future factory-arena paths. Tests can classify each attempt and prove setup-only native allocation without confusing current caller `Allocator` tensor traffic, net live bytes, or vendor-internal SDK allocations with IOM calls.

## Scope

- Instrument the CUDA, ROCm, and SYCL native allocation/free call sites currently used internally by metadata storage, host-transfer staging, SYCL binary temporary device buffers, and their cleanup or rollback paths. Instrument the future factory data/metadata backing calls when task 04 adds them; current tensor storage remains caller-`Allocator` traffic and is deliberately excluded from the IOM-native baseline.
- Provide backend-local test hooks and records for allocation attempts, successful allocations, frees, failed attempts, and allocate/free pairs that constitute transient churn.
- Record enough context to distinguish data backing from metadata backing; operation, metadata, or staging calls; setup from post-publication calls; allocation from free; and the owning backend/device/context where the backend exposes that identity.
- Add coverage for the future factory arena reservations so later tests can assert exactly two standard-GPU arena backing allocations: one data backing and one metadata backing.
- Keep the seam test-only and backend-local. It must not add a production global registry or alter allocator policy, arena ownership, queue scheduling, ownership/lifetime rules, or public APIs.

## Implementation references

- CUDA device setup and tensor ownership: `src/cuda/device.cpp`.
- CUDA metadata/staging/transfer policy boundary: `src/cuda/copy.hpp`.
- ROCm device setup and tensor ownership: `src/rocm/device.cpp`.
- ROCm metadata/staging/transfer policy boundary: `src/rocm/copy.hpp`.
- SYCL device setup and tensor ownership: `src/sycl/device.cpp`.
- SYCL metadata, binary temporary device buffers, and cleanup: `src/sycl/copy.cpp`.
- SYCL staging allocation and cleanup: `src/sycl/staging_pool.cpp`.
- Existing CUDA analogue: `iom::cuda_detail::DriverCalls` in `src/cuda/driver.hpp`, overridden by `DriverCallsRestore` and counting callbacks in `test/cuda/test_cuda_smoke.cpp:280-380`. That seam currently covers context calls only; extend the same backend-driver replacement pattern rather than treating it as already sufficient allocation coverage.
- Backend smoke target construction and exact test names: `test/CMakeLists.txt:83-176`, with CUDA/ROCm/SYCL registrations at lines 179-220.

## Requirements

1. Define planned backend-local driver-call seams or equivalent planned symbols for the CUDA, ROCm, and SYCL allocation/free boundaries. Production builds must retain direct backend behavior without a process-wide observer or registry; testing builds may replace the backend function pointers/callbacks and restore them reliably after each test.
2. Instrument the actual native boundary, not `iom::Allocator` helpers. CUDA records calls reaching `cuMemAlloc` and `cuMemFree`; ROCm records calls reaching `hipMalloc` and `hipFree`; SYCL records calls reaching `sycl::malloc_device` and `sycl::free`. Host allocations, `Allocator::allocate/deallocate`, and vendor-internal SDK/runtime allocations are outside the IOM-native record.
3. Emit an attempt record before or at each call and complete it with success/failure, byte count, operation class, lifecycle phase, and free/allocation kind. A failed allocation attempt must remain observable even when no pointer is returned; a cleanup free must remain observable even when the backend free reports an error or is invoked from a `noexcept` rollback path.
4. Define planned classification values privately within each backend test seam (or an equivalent backend-local record representation) for at least: data backing, metadata backing, operation metadata, staging, and other explicitly named IOM setup resources. Include setup versus post-publication phase and preserve a stable association for an allocation and its later free so transient allocate/free churn is visible instead of disappearing into a net-byte counter. Do not add public operation or direction enums.
5. Route classification at the IOM call sites that know the purpose of the allocation. Do not infer purpose from pointer address, current live-byte totals, allocator traffic, or vendor allocation statistics. The future factory reservation sites must classify their two setup calls as data backing and metadata backing before publication.
6. Preserve backend-local context/device activation and error categories. Instrumentation must not introduce a synchronization requirement, hidden lock, changed allocation ordering, changed cleanup ownership, or a production dependency on test code. Test hooks must be safe to restore on success, assertion failure, and exceptions.
7. Add focused backend smoke coverage that first establishes a baseline for current behavior: ordinary host transfers exercise internal staging boundaries; CUDA/ROCm metadata paths exercise hidden lazy metadata calls; and the SYCL binary fallback observes its three temporary device allocations and corresponding frees in `src/sycl/copy.cpp`. Separately prove current caller-`Allocator` tensor traffic is not misclassified as an IOM-native call. The assertions must use native-boundary records, not only caller `Allocator` counters or net live bytes.
8. Add failure-path coverage for at least one failed native allocation and one cleanup/rollback path per applicable backend, proving the failed attempt and any transient successful allocation/free pair are retained in the record with the correct phase/classification.
9. Make the seam support later arena tests without changing this task's allocation policy: after successful standard-GPU setup, tests can assert exactly two live IOM native backing allocations (data and metadata), and after device publication, tensor/workspace creation, queue creation, operations, metadata/staging use, waits, retirement, and teardown-safe frees can be separated from post-setup calls. The seam must make a nonzero post-setup allocation or free fail an explicit no-churn assertion while allowing the two setup frees at safe teardown.
10. Keep the instrumentation and its assertions in backend smoke drivers or backend-local test support. Shared backend conformance may consume normalized observations only through an existing test-facing path; do not expose instrumentation through the public IOM API or introduce a production global allocation tracker.

## Non-goals

- Do not implement the two arena allocation policy, `DeviceMemoryConfig`, queue/resource limits, workspace API, metadata-slot redesign, or removal of lazy allocations; those are specified elsewhere.
- Do not redirect, pool, coalesce, or otherwise change native allocations; this task observes boundaries only.
- Do not count or intercept vendor-internal CUDA, ROCm, or SYCL SDK allocations, kernel/runtime allocations, event internals, or host-USM allocations as IOM native device calls.
- Do not replace backend-local hooks with a process-wide registry, alter public APIs, add public operation/direction enums, or make standalone allocators secretly synchronized.
- Do not use caller `Allocator` allocation counts, net live bytes, or vendor memory telemetry as the sole oracle.

## Acceptance criteria

- CUDA smoke tests can install and restore a backend-local allocation seam and observe every IOM `cuMemAlloc`/`cuMemFree` call in the covered paths, including success, failure, cleanup, setup/post-publication phase, purpose, and transient churn. The existing `DriverCalls` context-call analogue remains intact and is not misrepresented as allocation coverage.
- ROCm smoke tests provide equivalent observations for every current internal IOM `hipMalloc`/`hipFree` metadata/staging boundary and the future factory data/metadata setup and rollback paths; caller-allocator tensor calls remain excluded.
- SYCL smoke tests provide equivalent observations for every IOM `sycl::malloc_device`/`sycl::free` boundary, including staging-pool growth/replacement and the three current binary temporary device buffers; host-USM and SDK-internal allocations are distinguishable and excluded.
- Baseline tests fail if instrumentation sees only caller `Allocator` traffic or only net live bytes: they explicitly observe current hidden lazy metadata/staging allocations and SYCL binary allocations at the native boundary, and verify matching frees or failed attempts where applicable.
- The normalized records distinguish data backing, metadata backing, operation/metadata/staging calls, setup versus post-publication calls, frees, failed attempts, and transient allocate/free churn. A failed call is not silently dropped, and an allocate/free pair is not collapsed into zero activity.
- With the later standard-GPU arena implementation, the same seam and assertions can prove exactly two setup backing allocations and zero post-setup IOM native allocation/free calls; this task does not weaken that assertion by counting only live allocations or by excluding frees.
- All hooks are backend-local, test-only, restorable, and absent from the public API and production global state. Existing smoke tests retain their behavior apart from deliberately added observations.

## Verification

Do not run gates while writing this mini-spec. After implementation, run each backend smoke target through the `remote-development` workflow on matching enabled hardware/configuration:

- CUDA: `ctest --test-dir <build-dir> --output-on-failure -R '^iom_cuda_smoke_tests$'`
- ROCm: `ctest --test-dir <build-dir> --output-on-failure -R '^iom_rocm_smoke_tests$'`
- SYCL: `ctest --test-dir <build-dir> --output-on-failure -R '^iom_sycl_smoke_tests$'`

The targeted smoke runs must cover successful and failed native attempts, hidden metadata/staging and SYCL binary baselines, classification/phase/free records, and restoration of each backend-local hook. Accelerator builds and tests must be executed through `remote-development`; no local accelerator gate is implied.

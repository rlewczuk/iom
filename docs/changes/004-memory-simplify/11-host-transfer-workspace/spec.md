# Cut host transfers over to explicit workspace

**Order:** 11
**Priority:** P1 — synchronous standard-GPU transfers must stop owning growing device staging and elastic streams; callers must budget and provide the transfer workspace explicitly.
**Blocked by:** `01-native-allocation-instrumentation`, `04-device-memory-arenas`, `05-raw-workspace-contract`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

`TensorView::copy_from_host` and `TensorView::copy_to_host` use the task-05 `RawWorkspaceView` contract and workspace-requirements query. CUDA, ROCm, and SYCL transfers use caller-provided device staging, with one eagerly provisioned host-transfer stream/queue and one active public transfer per `Device`. CPU and TTNN remain zero-workspace paths. Growing staging pools, elastic transfer-stream pools, and their compatibility callers are deleted.

## Scope

This task covers the atomic public API and backend migration for synchronous host transfers, including all direct callers and protected tensor hooks. It covers validation, workspace leasing, transfer-resource lifetime, same-device serialization, and focused regression coverage. It does not change queued `DeviceOps` operation APIs except where their existing caller waits are required before a host transfer.

## Implementation references

- Public and protected contracts: `include/iom/tensor.hpp` (`TensorView::copy_from_host`, `TensorView::copy_to_host`, `Tensor::region_from_host`, and `Tensor::region_to_host`); `src/iom.cpp` forwarding and host-size/BOOL validation.
- Workspace contract and query: task-05 planned `RawWorkspaceView` and `WorkspaceRequirements`; add planned `TensorView::copy_from_host_workspace_requirements()` and `TensorView::copy_to_host_workspace_requirements()`.
- Standard-GPU transfer implementation: `src/shared/standard_tiled_copy.hpp`, `src/shared/standard_tiled_copy.inl`, `src/cuda/copy.hpp`, `src/cuda/copy.cu`, `src/cuda/device.cpp`, `src/rocm/copy.hpp`, `src/rocm/copy.hip`, and `src/rocm/device.cpp`.
- CPU and TTNN hook implementations: `src/cpu/device.cpp`, `src/ttnn/copy.hpp`, `src/ttnn/copy.cpp`, and `src/ttnn/device.cpp`; preserve their direct-loop/native-storage boundaries.
- SYCL transfer implementation and setup: `src/sycl/copy.hpp`, `src/sycl/copy.cpp`, and `src/sycl/device.cpp`; retain the existing eager `transfer_queue`.
- Removal targets: `src/shared/staging_pool.hpp`, `src/shared/transfer_pool.hpp`, and `src/sycl/staging_pool.hpp` plus `src/sycl/staging_pool.cpp`.
- Caller and regression migration: `test/test_iom.cpp`, `test/backend/backend_conformance_common.hpp`, `test/backend/backend_conformance_copy_storage.hpp`, `test/backend/test_backend_coexistence.cpp`, `test/cuda/test_cuda_smoke.cpp`, `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_smoke.cpp`, `test/rocm/test_rocm_conformance.cpp`, `test/sycl/test_sycl_smoke.cpp`, `test/sycl/test_sycl_conformance.cpp`, `test/ttnn/test_ttnn_smoke.cpp`, `test/ttnn/test_ttnn_conformance.cpp`, `test/cpu/test_cpu.cpp`, `test/cpu/test_cpu_conformance.cpp`, and `test/cpu/test_cpu_bench.cpp`.

## Requirements

1. **Exact API cutover.** Extend the two public methods to accept the borrowed argument `RawWorkspaceView workspace = {}` consistently:
   ```cpp
   void copy_from_host(std::span<const std::byte> source,
                       RawWorkspaceView workspace = {});
   void copy_to_host(std::span<std::byte> destination,
                     RawWorkspaceView workspace = {}) const;
   ```
   Extend the protected virtual `Tensor::region_from_host` and `Tensor::region_to_host` hooks with the same workspace argument and forward the exact view. Add the two operation-specific `TensorView` requirements queries named above; they are pure, deterministic, and do not allocate, reserve, submit, or lease.

2. **Exact task-05 validation.** For CUDA, ROCm, and SYCL, each query returns `WorkspaceRequirements{bytes, alignment}` with `bytes == gpu_algorithm::compute_staging_size(view.spec().logical_nbytes())` and `alignment == 32`. The query and execution must agree exactly. Preserve checked size/offset arithmetic and existing host span-size, shape, and BOOL validation. Before the first native effect, execution must reject a missing workspace for a positive requirement, undersized capacity, misaligned base, invalid range, dead owner, non-exact `Device` identity, or workspace overlap with the tensor's addressed storage as `std::invalid_argument`. A range already exclusively leased by an accepted transfer rejects as `std::bad_alloc`; failed validation has no device write, host mutation, lease, or transfer-resource acquisition.

3. **Workspace lifetime and ownership.** Lease the supplied range exclusively from acceptance through synchronous completion proof. A disjoint range from the same live workspace owner may be used concurrently; successful reuse after completion is required. Keep the workspace, tensor owner, and host payload/mirror alive until the runtime proves all reads and writes complete. If completion is unknown or a transfer fails without a covering proof, quarantine the workspace range, payload/mirror, and transfer resource together; never return or reuse any of them prematurely. Preserve checked overflow behavior and never silently allocate a replacement workspace.

4. **Standard-GPU staging behavior.** CUDA, ROCm, and SYCL must use the supplied workspace as device staging and perform no internal device staging allocation, growth, replacement, or free. Preserve `compute_staging_size`, logical-to-tiled conversion, BOOL/tail-word initialization, untouched padding, and existing sub-byte/tiled semantics. Keep explicit caller waits for conflicting operation-queue reads or writes; host transfers must not introduce hidden operation-queue synchronization.

5. **CPU and TTNN boundary.** CPU continues its direct host loops and reports zero workspace requirements; its default empty view remains valid. TTNN reports zero requirements and keeps host-only staging/native storage behavior, with the default empty view valid. Neither backend may manufacture a raw device workspace or add device backing/growth for these transfers.

6. **Fixed transfer resource and concurrency.** CUDA and ROCm create exactly one dedicated host-transfer stream per standard-GPU `Device` during setup, with setup rollback and teardown/failure safety. SYCL retains its existing eagerly created dedicated `transfer_queue`. Serialize exactly one active public host transfer per `Device` at the owning-device boundary. Do not serialize compute queues, `DeviceOps` on the same device, or transfers belonging to another `Device`; distinct devices remain independent. A transfer resource is held through synchronous completion proof and quarantined with unresolved work.

7. **Atomic caller migration and removal.** Migrate every backend/direct caller and test to the exact query/argument contract. Callers must query, prepare a sufficient explicit workspace for positive requirements, pass its view, and retain it through the call. Keep source-simple default calls only on proven zero-workspace CPU/TTNN paths. Delete production `src/shared/staging_pool.hpp`, `src/shared/transfer_pool.hpp`, and `src/sycl/staging_pool.hpp`/`.cpp` after migration; remove their includes, symbols, allocation-failure hooks, and tests. Do not leave an overload, compatibility fallback, hidden allocator, elastic stream pool, or growth path.

8. **Focused regression coverage.** Replace the old CUDA/ROCm multi-stream, pool-growth, and staging-pool accounting tests with focused explicit-workspace tests. Coverage must include first-use and repeated transfers with zero IOM device alloc/free after setup; missing, undersized, misaligned, foreign, invalid-range, tensor-overlap, and conflicting-lease rejection before effects; disjoint workspace reuse; same-device serialization; independent cross-device transfers; teardown and injected-failure quarantine; logical/tiled/tail/padding preservation; and CPU/TTNN zero-query defaults. Migrate shared conformance and coexistence callers rather than weakening their data checks.

## Non-goals

- Changing tensor codecs, arithmetic, tile representation, view addressing, host span sizes, or supported data types.
- Adding asynchronous public host-transfer APIs, public operation/direction enums, duplicate owner query APIs, or an alternate legacy mode.
- Serializing compute operations or transfers across independent `Device` objects.
- Replacing TTNN native storage or eliminating its host-only staging allocations.
- Removing host allocations that have a proven asynchronous lifetime; only device staging ownership and growth are removed.
- Changing allocator synchronization outside the owning `Device` boundary or adding hidden locks to standalone allocators.

## Acceptance criteria

- Public and protected signatures, defaults, and the two query names/types match exactly; positive requirements cannot execute with `{}`, while CPU and TTNN zero requirements can.
- Every enabled backend and direct caller uses the same query result and workspace argument; no old pool symbol, include, source file, test, compatibility overload, or implicit fallback remains.
- CUDA, ROCm, and SYCL pass first-use and repeated-transfer allocation instrumentation with zero IOM device allocation/free after setup, and each standard-GPU device has exactly one setup-time host-transfer resource (SYCL's existing eager queue included).
- Invalid workspace capacity, alignment, range, exact-device/live-owner identity, tensor overlap, and accepted-range conflict are rejected before any effect with the specified exception category. Disjoint ranges reuse successfully and an active range remains leased through synchronous completion.
- Same-device public transfers are serialized, while compute and transfers on other devices proceed independently; conflicting operation-queue work still requires explicit caller waits.
- Full logical data, tiled conversion, BOOL/tail words, untouched padding, sub-byte cases, teardown, and injected runtime-failure behavior remain correct; unknown completion quarantines all affected workspace/payload/resource state.
- Focused tests replace the former CUDA/ROCm stream/staging-growth tests and prove no native churn, no growth, no hidden serialization beyond one device's transfer lane, and no compatibility path.

## Verification

Proposed gates (not run by this specification writer):

- Local: `ctest --test-dir <build> -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'`.
- Accelerator smoke and conformance, each enabled backend, through the `remote-development` workflow: `ctest --test-dir <build> -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_ttnn_smoke_tests|iom_ttnn_conformance_tests)$'`.
- Cross-backend coexistence, through the `remote-development` workflow when any accelerator backend is enabled: `ctest --test-dir <build> -R '^iom_backend_coexistence_tests$'`.
- Inspect allocation instrumentation and focused transfer tests for zero post-setup IOM device alloc/free, exact query agreement, validation-before-effects, lease/quarantine safety, data/padding preservation, same-device serialization, and cross-device independence; no build, test, formatter, linter, benchmark, or gate is run as part of writing this specification.

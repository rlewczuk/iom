# Share CUDA and ROCm standard-tiled copy and queue code behind typed policies

**Order:** 29
**Priority:** P1 — medium, verified maintenance defect. CUDA and ROCm contain near-duplicate standard-tiled copy/queue implementations that have already drifted in staging arithmetic and runtime policy.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-001`
**Review severity:** medium
**Review verification:** verified, confidence 95

## Outcome

`src/cuda/copy.cu` and `src/rocm/copy.hip` become thin policy/front-end translation units. A single private `src/shared/standard_tiled_copy.inl` owns the common standard-tiled kernels, plane mapping, host-transfer dispatcher, and a policy-parameterized `iom::GpuQueue<Policy>` used by both backends. CUDA and HIP supply typed policy operations and launch syntax; common code contains no SDK include, runtime type, backend switch, or global active-backend registry.

The shared queue template retains the pre-AR-002 task shape and existing event/failure semantics while consolidating the two GPU copies. AR-002 subsequently migrates all four backends to `iom::detail::StagedWorker<Task>`; it consumes this task/algorithm boundary rather than recreating the CUDA/ROCm implementation. PF-002, PF-003, and PF-006 consume the shared source and replace their respective launch, staging, and bit helpers in one location.

## Current failure

`src/cuda/copy.cu:19-624` and `src/rocm/copy.hip:19-590` are approximately 80% identical after normalizing CUDA/HIP spelling, but each has its own kernels, plane enumeration, host-transfer path, queue state machine, event handling, and unsupported-operation stubs. They already differ in staging rounding, staging allocation API, stream selection, and placement of fixes such as the word-rounded ROCm staging change.

The review evidence is the normalized source diff and the cited staging sites. Every future fix to common tiled-copy semantics must currently be applied twice; a missed port can pass one backend's tests while leaving the other backend wrong. This violates the intended invariant that CUDA and ROCm implement identical logical copy behavior and differ only where their runtimes require different policy operations.

## Scope

- **Create:** `src/shared/standard_tiled_copy.hpp` and `src/shared/standard_tiled_copy.inl`. The header declares the private policy surface; the inclusion unit defines common helpers, kernels, dispatch, and `GpuQueue<Policy>`.
- **Create:** `include/iom/gpu_algorithm.hpp` with the backend-neutral checked `iom::gpu_algorithm::compute_staging_size(std::size_t)` helper.
- **Modify:** `src/cuda/copy.cu` and `src/rocm/copy.hip` to define their policy types/macros, retain backend fault-injection entry points, include the shared body exactly once, and provide existing backend wrappers/factories.
- **Modify:** `CMakeLists.txt` only to list the private shared header and backend source entries according to existing target conventions.
- Move common `plane_slot`, plane enumeration, `scatter_plane_kernel`, `gather_plane_kernel`, `copy_plane_kernel`, `view_planes`, `plane_pairs`, `identical_window`, `launch_view_transfer`, `launch_copy_plane`, `synchronous_transfer`, unsupported-operation formatting, and GPU queue lifecycle into the shared body. Keep `SubmissionFault` storage and API names backend-local.
- Keep all current public signatures and error behavior. The pre-AR-002 `Task` contains `sequence`, source/destination view pointers, native event, and `no_op`; AR-001 does not invent the later opaque fence field.

CPU, TTNN, SYCL, public headers, capability modeling, allocator centralization, driver seam, and the separate performance/stability fixes are out of scope.

## Implementation references

- **Create:** `include/iom/gpu_algorithm.hpp` — backend-neutral checked `compute_staging_size`, rounding a logical byte count to the next 32-bit staging-word boundary and throwing `std::overflow_error("GPU transfer staging size overflows")` before overflow or allocation.
- **Create:** `src/shared/standard_tiled_copy.hpp` — private declarations for `GpuQueue<Policy>` and the policy operations. It includes no CUDA or HIP header and names no native handle.
- **Create:** `src/shared/standard_tiled_copy.inl` — sole source of common pre-AR-002 standard-tiled kernels, plane/stride mapping, host-transfer dispatch, `GpuQueue<Policy>`, task staging, event lifecycle, and unsupported-operation formatting.
- **Modify:** `src/cuda/copy.cu` — CUDA `gpu_policy`, `IOM_LAUNCH_KERNEL` expansion to CUDA triple-chevron syntax, CUDA error bridge, `SubmissionFault` seam, one shared inclusion, and existing CUDA wrappers.
- **Modify:** `src/rocm/copy.hip` — HIP `gpu_policy`, `IOM_LAUNCH_KERNEL` expansion to `hipLaunchKernelGGL`, HIP error bridge, `SubmissionFault` seam, one shared inclusion, and existing ROCm wrappers.
- **Read:** `src/cuda/copy.hpp`, `src/rocm/copy.hpp`, `src/cuda/device.cpp`, and `src/rocm/device.cpp` — existing factory and synchronous-region signatures and every caller to migrate.
- **Read:** `src/cuda/driver.hpp` and the existing backend fault seams — AR-005 owns the CUDA driver-call seam; AR-001 must consume it rather than duplicate it.
- **Modify:** `CMakeLists.txt` — list `src/shared/standard_tiled_copy.hpp` with both backend targets; do not list the `.inl` as an independent translation unit.
- **Read/modify tests:** `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, and shared backend conformance only for private symbol migration. Existing case names, fault names, messages, and byte assertions remain.

### Policy surface

The private policy supplies typed aliases and operations for stream/event handles, context or device activation, event create/record/synchronize/destroy, stream create/synchronize/destroy, device allocation/free, asynchronous and synchronous copies, memset, kernel-launch status, error name/string, and unsupported-operation messages. The shared body calls only `Policy::...` operations and `IOM_LAUNCH_KERNEL`. The two policy definitions may include their SDK headers; the shared `.hpp` and `.inl` may not.

### Shared queue boundary

`GpuQueue<Policy>` contains the one pre-AR-002 GPU queue state machine: validation and identical-window handling, sequence submission, staged/task lists, worker thread, event fence, completion, and destructor drain. It uses one `Task` layout for CUDA and ROCm. All backend-specific behavior enters through `Policy` and the backend-local retained-failure seam. The template is private and instantiated only once per backend.

AR-002's later `StagedWorker<Task>` is the planned successor for this queue state machine. AR-002 must remove `GpuQueue<Policy>`'s duplicated worker mechanics rather than create a third implementation; the shared kernel/transfer functions and `compute_staging_size` remain consumable by the later helper.

## Requirements

1. One shared `.inl` contains every common CUDA/ROCm algorithm and pre-AR-002 GPU queue body. No common kernel, plane traversal, transfer dispatcher, or queue state-machine body remains in either backend file.
2. The shared files contain no CUDA/HIP SDK include, native runtime type, backend-kind switch, global active-backend registry, or vendor token in source text.
3. CUDA and ROCm policies isolate launch syntax, stream/event handles, context/device activation, allocation, copies, memset, kernel status, and error strings.
4. `compute_staging_size` is the only staging-size arithmetic helper. Both backend synchronous-transfer paths call it and preserve its exact overflow category/message.
5. Standard 16x16 tiled layout, plane offsets/strides, high-rank leading traversal, LSB-first sub-byte packing, exact logical host byte counts, and identical-window no-op semantics remain unchanged.
6. The pre-AR-002 shared `Task` has exactly `sequence`, source view pointer, destination view pointer, native event, and `no_op`. It has no `void* fence` or failure field; later tasks own those changes.
7. Existing `SubmissionFault` names and retained-failure behavior remain backend-local and observable. Pre-link failures still roll back; post-enqueue failures still return valid tokens and surface through repeated `wait`.
8. Existing CUDA/ROCm factory and region-transfer callers continue to compile and link; private signature changes are migrated everywhere.
9. CPU, TTNN, SYCL, public APIs, allocator ownership, and tensor metadata remain unchanged.
10. The shared queue template is private, has no virtual runtime dispatch, and does not expose backend native types through installed headers.

## Non-goals

- AR-002's all-backend `StagedWorker<Task>` extraction. AR-002 consumes and supersedes the shared GPU queue state machine.
- PF-002's one-launch grid-stride copy and metadata-slot pool.
- PF-003's cached host-transfer streams and staging slots.
- PF-006's word-oriented field extraction and non-atomic stores.
- ST-001/ST-002/ST-003/ST-006/ST-007 changes except preserving their pre-existing contracts at this shared boundary.
- CPU, TTNN, SYCL, capability queries, allocator helpers, CUDA driver seam completion, TTNN type mapping, public ABI, or new runtime backend selection.

## Acceptance criteria

- [ ] `src/shared/standard_tiled_copy.inl` contains the sole definitions of the common pre-AR-002 kernels, plane mapping, host-transfer dispatch, and `GpuQueue<Policy>`; each backend includes it exactly once.
- [ ] CUDA and ROCm copy translation units contain only policy/macros, local fault-seam glue, shared inclusion, and required wrappers; no duplicate kernel or queue state-machine body remains.
- [ ] `include/iom/gpu_algorithm.hpp` contains the sole checked staging-size helper and both backends use it with the exact overflow diagnostic.
- [ ] CUDA and ROCm conformance suites pass with identical logical bytes for all existing supported leaf types, transformed/high-rank views, sub-byte values, and no-op windows.
- [ ] Existing event-create, third-launch, event-record, invalid-input, and repeated-wait tests preserve their exception categories and token behavior.
- [ ] Source audits find no SDK token/include in shared files, exactly one shared inclusion per backend, no second queue class, and no stale duplicated algorithm definitions.
- [ ] Core-only configuration remains free of CUDA/HIP dependencies; CUDA-only and ROCm-only targets each configure and build.
- [ ] AR-002 can replace the shared `GpuQueue<Policy>` worker mechanics with `StagedWorker<Task>` without restoring backend copies; PF-002/PF-003/PF-006 can replace their shared algorithm portions in one source location.

## Verification

Follow `.agents/skills/remote-development` for CUDA and ROCm configuration, build, and execution. Run both backend conformance suites and existing transactional-failure cases on their respective hardware. Run CPU/common tests with optional accelerators disabled. Use source audits to count shared inclusions and confirm the shared files have no SDK tokens or duplicate queue/kernel definitions. Compare the shared body with both former implementations and verify only policy-level differences remain. Clean remote mirrors after verification.

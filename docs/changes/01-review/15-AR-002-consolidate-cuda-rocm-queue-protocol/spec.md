# CUDA and ROCm duplicate the complete policy-independent asynchronous queue protocol

**Order:** 15
**Priority:** P2
**Blocked by:** 02-ST-001-fence-cuda-rocm-copies-before-operand-release, 10-PF-002-remove-duplicate-event-sync-on-completion
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** AR-002
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** strongly-supported, confidence 94
**Review scope:** whole-codebase
**Backend scope:** cuda, rocm
**Location:** `src/cuda/copy.cu:93-353 (CudaQueue); src/rocm/copy.hip:92-352 (RocmQueue); duplicated fence helpers src/cuda/copy.cu:75-87 (build_cuda_fence) and src/rocm/copy.hip:74-86 (build_hip_fence)`

## Outcome

A single compile-time `detail::GpuQueue<Policy>` template (plus one generic fence-capture helper) is the sole owner of the CUDA/ROCm asynchronous queue state machine. `cuda_detail::make_queue` and `rocm_detail::make_queue` instantiate it with their existing `gpu_policy` types, so all policy calls remain compile-time. No runtime behavior, backend diagnostic, allocator/event counter, or public interface changes; the two independent implementations of the same ~260-line ownership protocol become one.

## Current problem

Every accelerator copy must preserve the same submission, ownership, ordering, deferred-error, fence, registry, quarantine, and queue-destruction protocol; only runtime primitives and backend diagnostics should vary between CUDA and ROCm. Today the two backends independently encode that entire protocol:

- `CudaQueue` (`src/cuda/copy.cu:93-353`) and `RocmQueue` (`src/rocm/copy.hip:92-352`) have the same task fields (sequence, source/destination view pointers, native event, no-op flag, `EventRingSlot`, source/destination entry IDs), construct `StagedWorker<Task>` with the same four callbacks and `PublishPolicy::Splice`, invalidate the registry before `shutdown_and_drain`, perform identical validation/no-op submission, and execute the same sequence: acquire an `EventRingSlot` → choose inline or pooled metadata → launch `launch_grid_stride_copy<gpu_policy>` → check/record the event → register the two outstanding entries → insert the `SequenceOutcome` → on completion call `release_or_invalidate_entries` and `complete`, including post-launch retained-failure handling and registry rollback.
- `build_cuda_fence` (`src/cuda/copy.cu:75-87`) and `build_hip_fence` (`src/rocm/copy.hip:74-86`) are the same `EventLeaseWithFailure` capture and `FenceCaptureOps` construction with only the function name changed; the `EventLeaseWithFailure` struct is duplicated as well.
- The only primitive differences in this flow are already isolated in the backend-local `gpu_policy` definitions (`src/cuda/copy.hpp:42-223`, `src/rocm/copy.hpp:43-227`): context/stream/event/pointer types, activate/create/destroy/synchronize calls, allocation/copies, fault seams, kernel checks, and error text. Both policies satisfy the shared `EventRingState` (`src/shared/event_ring.hpp:17-178`) and `MetadataSlotPool` (`src/shared/metadata_slot_pool.hpp:14-106`) requirements and drive the same standard tiled-copy implementation (`src/shared/standard_tiled_copy.inl:370-718`).

The duplication is semantic, not merely syntactic: a future ownership/error/lifetime fix can land in one backend and silently diverge in the other. No production or test caller names `CudaQueue` or `RocmQueue` directly — `CudaDevice::create_ops` (`src/cuda/device.cpp:106-110`) and `RocmDevice::create_ops` (`src/rocm/device.cpp:91-95`) are the only factories, each constructing its private queue (`src/cuda/copy.cu:381-386`, `src/rocm/copy.hip:378-383`); public tests call `Device::create_ops`.

## Scope

- Delete the duplicate `CudaQueue`/`RocmQueue` class bodies, `build_cuda_fence`/`build_hip_fence`, and the duplicate `EventLeaseWithFailure` structs; add one policy-templated `detail::GpuQueue<Policy>` in a shared header/inl composing `EventRingState<Policy>`, `MetadataSlotPool<Policy>`, the existing `StagedWorker<Task>`, `register_copy_entries`, and `release_or_invalidate_entries` with the current `PublishPolicy::Splice` flow.
- Have `cuda_detail::make_queue` and `rocm_detail::make_queue` instantiate the template with their existing backend `gpu_policy`; keep policies, smoke-test aliases, native kernels, fault enums, and backend error labels unchanged. Use policy-provided static labels for duplicate-sequence diagnostics.
- Preserve every current branch ordering and rollback path: no-op handling, inline vs pooled metadata, post-launch retained failures, event-record fallback, registry rollback, queue invalidation before worker drain, and repeated `DeviceOps` failure waits.
- The template MUST introduce no virtual backend dispatch, extra allocation, copy, synchronization, or queue state.

## Implementation references

- **Modify:** `src/cuda/copy.cu` — replace `CudaQueue` (`:93-353`) and `build_cuda_fence` (`:75-87`) with the shared template instantiation; `make_queue` (`:381-386`) becomes `std::make_unique<detail::GpuQueue<cuda_detail::gpu_policy>>(...)`.
- **Modify:** `src/rocm/copy.hip` — same replacement for `RocmQueue` (`:92-352`) and `build_hip_fence` (`:74-86`); `make_queue` (`:378-383`) becomes `std::make_unique<detail::GpuQueue<rocm_detail::gpu_policy>>(...)`.
- **Read:** `src/cuda/copy.hpp:42-223` and `src/rocm/copy.hpp:43-227` — the `gpu_policy` surface the template requires; all backend primitives stay behind it.
- **Read:** `src/shared/event_ring.hpp:17-178`, `src/shared/metadata_slot_pool.hpp:14-106`, `src/shared/standard_tiled_copy.inl:370-718`, and `include/iom/iom.hpp:18-185` (`StagedWorker`, `PublishPolicy`) — the shared primitives the template composes.
- **Read (counterparts, explicitly not merged):** `src/sycl/copy.cpp:263-654` and `src/ttnn/device.cpp:370-553` — materially different event/fence/queue models (SYCL `sycl::event` waits; TTNN synchronous native execution with `CompleteOnThrow` and native-finish fencing); neither fits this consolidation.
- **Tests:** `test/cuda/test_cuda_conformance.cpp:341-518` and `test/rocm/test_rocm_conformance.cpp:492-684` — mirrored transactional post-enqueue faults, retained repeated failures, allocator quarantine/address non-reuse, queue destruction; event-ring smokes at `test/cuda/test_cuda_smoke.cpp:645-770` and `test/rocm/test_rocm_smoke.cpp:237-362` — event reuse, capacity, cached results, fault seams, metadata rank 8/9 boundaries.

## Requirements

- Land on top of the blocked-on fixes: 02-ST-001 (fenced CUDA/ROCm copies before operand release) and 10-PF-002 (single event wait on normal completion). The consolidated template MUST be written against the post-fix protocol (one event synchronization per normally completed task; releases fenced behind copy completion), not against the current duplicated callback flow.
- After the change, exactly one common queue implementation exists: no concrete `CudaQueue`/`RocmQueue` class and no duplicate fence-capture flow may remain; `grep` for `class CudaQueue`, `class RocmQueue`, `build_cuda_fence`, `build_hip_fence` in `src/` must return nothing.
- Observable behavior MUST match the current backend-specific tests exactly: tokens, no-op sequence consumption, copy results, event/metadata exhaustion behavior, post-enqueue failure retention, allocator non-reuse after quarantine, repeated waits, queue destruction with draining, foreign-device scenarios, and backend diagnostic strings.
- Policy calls MUST remain compile-time with no runtime indirection or added synchronization; queue object layout semantics and factory signatures are unchanged.

## Non-goals

- Do not merge CUDA and ROCm API policies, allocator/error strings, fault-injection enums, kernel implementations, device factories, or test seams.
- Do not merge SYCL or TTNN queues; do not add runtime backend registries or virtual dispatch.
- Do not change EventRing count, metadata limits, stream synchronization scope, registry semantics, or public interfaces.

## Acceptance criteria

- [ ] Both `make_queue` functions return queues passing the full CUDA and ROCm smoke + conformance suites with unchanged diagnostics: transactional post-enqueue fault, event-ring capacity, metadata rank 8/9, queue destruction, allocator quarantine/address non-reuse, repeated-wait, and foreign-device scenarios (per 02-ST-001/10-PF-002 post-fix expectations).
- [ ] A single common queue implementation owns the protocol: no `CudaQueue`/`RocmQueue`/`build_*_fence`/duplicate `EventLeaseWithFailure` remains in `src/`; the remaining CUDA/ROCm delta is policy-only (primitive types, fault seams, error text).
- [ ] No runtime indirection or added synchronization: policy calls instantiate at compile time per backend; allocator/event counters and sync-call counts match the pre-consolidation (post-blocker) baselines.

## Verification

- `actual validation: none (read-only review)`; proposed gates below (remote CUDA and ROCm Linux hosts with accelerators, per `remote-development`).
- `cmake --build <build> --target iom_cuda_smoke_tests iom_cuda_conformance_tests iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests` and `ctest --test-dir <build> -R 'iom_(cuda|rocm)_(smoke|conformance)_tests|iom_backend_coexistence_tests' --output-on-failure` on the respective hardware/toolchains.
- Compare backend diagnostics and allocator/event counters before and after consolidation, including the event-wait counters from 10-PF-002 and the copy/operand-release fencing from 02-ST-001, plus event-ring capacity, metadata rank 8/9, queue destruction, and foreign-device scenarios.
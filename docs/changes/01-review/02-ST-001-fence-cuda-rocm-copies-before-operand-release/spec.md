# CUDA/ROCm registry fences report success before stream completion

**Order:** 02
**Priority:** P0 — memory-safety/lifetime
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** ST-001
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 92
**Review scope:** whole-codebase
**Backend scope:** cuda, rocm
**Location:** `src/shared/event_ring.hpp:103-138` (`EventRingState::on_worker_complete`, `invoke_result`); `src/cuda/copy.cu:68-69,174-318`; `src/rocm/copy.hip:68-69,174-317`; `src/cuda/device.cpp:141-178`; `src/rocm/device.cpp:120-157`

## Outcome

A live CUDA/ROCm registry fence reports success only after its own recorded stream event has reached a terminal completion. Destroying a source or destination operand before an explicit wait never frees or recycles the allocation while the referencing stream operation is still in flight, and reusing an event-ring slot never changes the result observed by an earlier registry entry.

## Current problem

A live owner allocation can be released or reused before every asynchronous operation referencing it has reached terminal completion on its owning CUDA/ROCm stream, because the registry fence reads a success cache instead of the operation's event. Timeline: (1) `queue->copy(source.view(), destination.view())` reserves a sequence, launches the grid-stride copy on the queue stream, records an event, registers source/destination entries, and publishes the `StagedWorker` task; (2) `StagedWorker::submit_copy` returns after publication, but the worker thread has not necessarily run `on_worker_complete`; (3) each event-ring `Slot::cached_result` starts as `FenceResult::success()` and `invoke_result()` returns that cached value without touching the event (`src/shared/event_ring.hpp:133-138`); (4) if an owner is destroyed in this window, `release_or_quarantine()` sees success from the CUDA/HIP fence, removes the entries, and calls `Allocator::free` (`include/iom/detail/outstanding_work_registry.hpp:519-548`); (5) the stream worker later reads the freed/reused source or destination allocation. A second timeline exists: `on_worker_destroy` can release an old slot before `complete_task` removes registry entries; a concurrent submission reuses the slot, so a stale registry fence reads the new operation's cached result. Evidence: `on_worker_complete` synchronizes the event before writing `cached_result` (`src/shared/event_ring.hpp:103-113`), while `invoke_result` is documented as "without touching the event"; CUDA/HIP fences capture only `state` plus `slot_index` and call `invoke_result` (`src/cuda/copy.cu:59-69`, `src/rocm/copy.hip:59-69`); the queue executes launch/event/registration before publication (`src/cuda/copy.cu:174-318`, `src/rocm/copy.hip:174-317`) and `StagedWorker::submit_copy` runs before the worker is notified (`include/iom/iom.hpp:66-104`); the shared `EventRingState` has reusable 16-slot storage that is not reset to a pending state on acquire (`src/shared/event_ring.hpp:18-31,67-101`). Impact: use-after-free/device memory corruption, writes into a newly allocated tensor at a recycled address, or nondeterministic failed inference; the stale-success mechanism can also release storage while a newer operation using a reused event slot is still in flight.

## Scope

- Make the registry completion primitive operation-specific and terminal-state aware: a live CUDA/ROCm registry fence must either wait/synchronize its own recorded event before returning success, or return a non-success/pending result that forces quarantine; it must never use the default success cache for an in-use slot.
- Preserve the event-ring pool but prevent slot reuse from changing an earlier fence's identity: either retain a per-submission completion object/event state until all registry entries are removed, or defer slot release until completion/registry cleanup has finished.
- Update CUDA and ROCm queue callback ordering and failure cleanup consistently, removing the duplicated stale slot-index result responsibility rather than adding an independent lifetime flag.

## Implementation references

- **Modify:** `src/shared/event_ring.hpp` — `EventRingState::{Slot,acquire,on_worker_complete,on_worker_destroy,invoke_result,release_locked}`; the fence primitive is the single shared mechanism both backends use.
- **Modify:** `src/cuda/copy.cu:59-69,174-318` and `src/rocm/copy.hip:59-69,174-317` — `cuda_fence_invoke`/`build_cuda_fence`/`CudaQueue::{execute,complete_task,~CudaQueue}` and `hip_fence_invoke`/`build_hip_fence`/`RocmQueue::{execute,complete_task,~RocmQueue}` retain the operation-specific fence identity paired with registration/cleanup ordering.
- **Read:** `include/iom/detail/outstanding_work_registry.hpp:519-548` — `release_or_quarantine` success branch that removes entries and releases storage; `src/cuda/device.cpp:141-178` and `src/rocm/device.cpp:120-157` — `CudaTensor::~CudaTensor`/`RocmTensor::~RocmTensor` call it.
- **Tests:** `test/cuda/test_cuda_conformance.cpp:451-485`, `test/rocm/test_rocm_conformance.cpp:607-640` — existing wait/destruction patterns to extend; add backend-specific pre-wait destruction and address-recycling cases.

## Requirements

- A fence for a pending operation must never return `succeeded=true`; at minimum, an in-use slot must not report the cached default success.
- The event-ring pool may remain, but a later operation's acquisition of a previously used slot must not alter the result observed by an earlier registry entry (retain per-submission completion state until all registry entries referencing it are removed, or defer slot release until completion/registry cleanup finishes).
- Source/destination owner allocations must not be freed or recycled until the recorded event has completed and its result is known — including when an owner is destroyed before the worker thread has run `on_worker_complete`.
- Both CUDA and ROCm must get the identical correction with consistent callback ordering and failure cleanup; do not fix one backend only.
- Existing successful/failure waits must remain repeatable, and queue destruction must still drain without leaks or double release.
- Removing the duplicated stale slot-index result responsibility must not introduce a second, independent lifetime flag that duplicates state already owned by the event-ring slots.

## Non-goals

- Changing public tensor layouts, allocator APIs, queue token encoding, copy kernel arithmetic, or the intentional CUDA-vs-ROCm runtime policy differences.
- Replacing event pooling with device-wide synchronization.

## Acceptance criteria

- [ ] After a nonblocking CUDA/ROCm copy, destroying either operand before an explicit wait does not free or recycle its allocation until the recorded event has completed; waiting the token still yields correct destination bytes.
- [ ] A fence for a pending operation never returns `succeeded=true`; when an event-ring slot is reused by a later submission, the earlier registry entry still observes its own result, not the new operation's.
- [ ] Back-to-back copies sharing an operand (forcing event-slot reuse before the first registry entry is removed) release storage only after both operations complete; existing successful/failure waits remain repeatable and queue destruction drains without leaks or double release.

## Verification

Actual validation: none (read-only review); proposed gates below.

Proposed gates (accelerator hardware on a remote host per remote-development; sync the workspace, then run via `remote-exec`):

- Add/run a delayed-copy scenario: submit a sufficiently large copy, destroy source and/or destination immediately before any explicit `wait`, allocate replacement tensors from an address-reusing allocator, then wait the original token and assert allocator non-reuse and correct destination bytes.
- Repeat with two back-to-back copies sharing an operand to force event-slot reuse before the first registry entry is removed; assert the earlier fence's result is unchanged and storage is released only after both complete.
- Run the existing CUDA/ROCm conformance and queue-destruction tests: `ctest --test-dir <build> -R 'iom_(cuda|rocm)_(smoke|conformance)_tests' --output-on-failure`.
- Run Compute Sanitizer (CUDA) or an equivalent ROCm memory/race check over the delayed-copy and reuse scenarios.

- `ctest --test-dir <build> -R 'iom_(cuda|rocm)_(smoke|conformance)_tests' --output-on-failure`
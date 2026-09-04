# Synchronize queued GPU work before destroying events and streams

**Order:** 47
**Priority:** P0 — high, strongly-supported teardown UB. CUDA/HIP events and streams can still reference queued kernels when the current queue destructors destroy them.
**Blocked by:** `27-ST-006-bound-rocm-failure-wait`, `30-AR-002-share-backend-queue-scaffold`
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-001`
**Review severity:** high
**Review verification:** strongly-supported, confidence 85

## Outcome

CUDA and ROCm queue teardown fences every pending event before destroying it and synchronizes the queue stream before destroying the stream. The cleanup applies to both staged and published tasks, including the helper-owned shutdown drain. One failed event synchronization does not skip later entries or the stream finalizer. Null-fence retained-failure entries skip event calls. All cleanup paths are no-throw effective and preserve the existing completion callbacks.

The fix consumes AR-002's exact `StagedWorker<Task>` and four callbacks. The backend `fence_destroy(void*)` callback performs a no-throw synchronize-then-destroy for non-null event resources, so helper shutdown drains also fence before destroying. The queue destructor invokes `shutdown_and_drain()`, then performs context/device-guarded stream synchronize and stream destroy with errors swallowed after the full cleanup attempt. No second worker or queue state machine is added.

## Current failure

`~CudaQueue` at `src/cuda/copy.cu:424-448` and `~RocmQueue` at `src/rocm/copy.hip:361-387` currently destroy events and streams without synchronizing. Their worker exception drains also destroy pending events without first fencing them. AR-002's current helper drain makes this boundary explicit: shutdown calls `fence_destroy` without waiting, so a backend destroy callback that only destroys its native event would retain the same bug.

The review evidence is the absence of `cudaEventSynchronize`/`hipEventSynchronize` before the destructor `cudaEventDestroy`/`hipEventDestroy` calls and the absence of `cudaStreamSynchronize`/`hipStreamSynchronize` before stream destruction. A queue destroyed while copies are pending can invoke undefined behavior, leak remaining events after one exception, or destroy a busy stream.

## Scope

- **Modify:** `src/cuda/copy.cu` and `src/rocm/copy.hip` event callback implementations consumed by AR-002. For a non-null event resource, `fence_destroy` calls event synchronize, records no throw, then destroys the event exactly once. Null fences are a no-op.
- **Modify:** CUDA and ROCm queue destructors after AR-002 integration. Call `worker_.shutdown_and_drain()` first, then synchronize and destroy the queue stream under the correct context/device guard. Continue cleanup after an individual runtime error.
- **Modify:** any backend-specific normal worker completion callback only as needed to avoid direct event destroy outside the callback. AR-002 remains the sole worker owner; no backend run-loop copy is retained.
- **Tests:** add CUDA and ROCm queue-destruction cases with pending copies, multiple staged/tasks entries, and null-fence retained failures where the backend can inject them.

CPU and TTNN have no CUDA/HIP event or stream in this finding and are out of scope. Tensor destructor fencing is ST-003; bounded ROCm failure waits are ST-006.

## Implementation references

- **Read/modify:** `docs/changes/0001-tensor-view/30-AR-002-share-backend-queue-scaffold/spec.md` — `StagedWorker<Task>`, `shutdown_and_drain`, task publication, and exact callbacks. Its helper is the only worker and drains both lists.
- **Modify:** `src/cuda/copy.cu` — CUDA `fence_complete(void*)`, `fence_destroy(void*)`, `~CudaQueue`, context guard, stream finalization, and existing native error helpers. The callback obtains the event from the post-AR-002 opaque fence/resource.
- **Modify:** `src/rocm/copy.hip` — HIP mirror using `DeviceGuard`, `hipEventSynchronize`, `hipEventDestroy`, `hipStreamSynchronize`, and `hipStreamDestroy`.
- **Read/consume:** `docs/changes/0001-tensor-view/27-ST-006-bound-rocm-failure-wait/spec.md` — null-fence retained-failure behavior. A null fence must not be synchronized or destroyed.
- **Read:** `src/cuda/driver.hpp`, `src/cuda/device.cpp`, existing CUDA/ROCm queue constructors, and current event/stream creation sites to preserve the construction context and error category.
- **Modify tests:** `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp` for pending-copy teardown and multi-entry cleanup regressions.

### Event cleanup primitive

Define one TU-private no-throw helper per backend:

```cpp
void fence_and_destroy(cudaEvent_t event) noexcept;
void fence_and_destroy(hipEvent_t event) noexcept;
```

For null, return. Otherwise call the backend event synchronize, capture or discard the status without throwing, then call event destroy unconditionally and discard its status. The helper must not call throwing `check_cuda`/`check_hip`. Normal worker completion may already have called `fence_complete`; a second synchronize in `fence_destroy` is intentional because the same callback is also the helper's shutdown-drain cleanup boundary, where no `fence_complete` is invoked. The callback must not destroy an event twice.

### Stream finalization

After `shutdown_and_drain()` joins the worker, the queue destructor calls a TU-private `synchronize_and_destroy_stream(stream)` helper under `ContextGuard(context_)` or `DeviceGuard(device_ordinal_)`. It calls stream synchronize, records only the first diagnostic if construction succeeds, then calls stream destroy unconditionally. The helper is `noexcept`; the outer destructor catch remains only as a last-resort guard. The stream helper runs after all event callbacks and exactly once.

## Requirements

1. Every non-null pending CUDA/HIP event is synchronized before exactly one destroy call, whether it is in AR-002's staged list, task list, normal worker path, or shutdown drain.
2. Null-fence entries skip both event synchronization and destruction and still receive the established completion callback.
3. One event-sync or event-destroy failure never prevents later entries, the stream synchronization, or stream destruction.
4. Queue stream synchronization occurs after helper shutdown drain and before exactly one stream destroy call.
5. CUDA cleanup runs with the creation `CUcontext` active; ROCm cleanup runs with the creation device ordinal selected.
6. `fence_destroy` is no-throw effective and performs all cleanup needed by AR-002 shutdown drain without adding a fifth callback.
7. The existing `fence_complete`/`complete` behavior and ST-006 retained-failure wait semantics remain observable while the queue is alive.
8. No backend-specific worker loop, second mutex/list/thread, public API, or queue token contract is introduced.
9. Destructors do not call a whole-device synchronization beyond the owning queue's stream; tensor storage safety remains ST-003.
10. Every pending entry is removed from the worker lists and receives at most one completion callback.

## Non-goals

- CPU or TTNN queue changes.
- Tensor destructor fencing/quarantine (ST-003).
- ROCm bounded-wait or repeated-failure behavior (ST-006) beyond preserving its null-fence entry shape.
- PF-002 metadata slots, PF-003 host-transfer pools, PF-006 kernels, allocator changes, or public interfaces.
- A new runtime failure-injection seam for event synchronization.
- Making a destroyed queue waitable or retaining its private failure map.

## Acceptance criteria

- [ ] CUDA and ROCm queue destructors synchronize every pending non-null event before destroying it, including both staged and published tasks.
- [ ] CUDA and ROCm queue destructors synchronize their stream after worker drain and before stream destruction, even when an earlier event operation reports failure.
- [ ] Null-fence retained-failure tasks invoke no event runtime call and still complete through the existing callback.
- [ ] A pending-copy teardown regression passes under CUDA Compute Sanitizer and the repository's ROCm memory diagnostics without busy-event or busy-stream destruction diagnostics.
- [ ] A multi-entry teardown regression proves one bad event does not prevent later event cleanup or stream finalization.
- [ ] Existing transactional-failure and repeated-wait tests remain green.
- [ ] Source inspection finds no backend queue run-loop duplicate after AR-002 and exactly one event cleanup boundary per native event.
- [ ] CPU common conformance remains unchanged.

## Verification

Follow `.agents/skills/remote-development` for accelerator build and execution. Build and run CUDA and ROCm conformance targets on their respective hosts, exercising pending-copy destruction without explicit waits, multiple entries, and retained-failure null fences. Run CUDA Compute Sanitizer memcheck/racecheck and the established ROCm diagnostics for the focused teardown cases. Inspect event/stream call ordering and verify no direct destructor event destroy bypasses the callback. Run the CPU common suite locally or on the configured build without changing its expected cases, then clean remote mirrors.

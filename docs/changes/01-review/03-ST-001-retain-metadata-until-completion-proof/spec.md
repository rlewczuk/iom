# Retain pooled metadata until asynchronous completion is proven

**Order:** 03
**Priority:** P0 — async memory-safety/lifetime
**Blocked by:** None
**Review source:** `cpp-inference-gpu-stability` — whole-codebase review of checked-out main HEAD `ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb` (`Update remote hosts`), clean tree at review start
**Finding:** ST-001
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 90
**Review scope:** whole-codebase
**Backend scope:** multi-backend (CUDA, ROCm/HIP, SYCL; shared queue/pool machinery)
**Location:** `src/shared/gpu_queue.hpp:329-346` and `src/shared/event_ring.hpp:164-215` (`GpuQueue::execute`, `EventRingState::on_worker_complete/on_worker_destroy`); `src/shared/staging_pool.hpp:228-250`; `src/shared/transfer_pool.hpp:18-36`; analogous SYCL paths `src/sycl/copy.cpp:431-448,739-755,868-898` and `src/sycl/staging_pool.cpp:123-146`

## Outcome

A failed event/stream/queue completion attempt produces an explicit `Retire/Unknown` disposition rather than permission to free or recycle storage. Shared CUDA/ROCm metadata slots remain unavailable until a successful covering stream drain or queue teardown drain, while normal proven completion paths retain their current fast reuse. The same disposition protects the in-scope pooled asynchronous resources without using a fresh event as a fence.

## Current problem

The backend contract requires temporary storage and metadata to remain live until successful completion proves no submitted operation can reference them. In `GpuQueue::execute`, a pooled metadata submission can enqueue metadata and a kernel, fail both event-record attempts, and fail `synchronize_stream_noexcept`; `on_worker_complete` retains the token failure, but `on_worker_destroy` synchronizes only the fresh/unrecorded event, ignores the result, and then releases the metadata slot. Queue stream synchronization occurs later in the destructor, after the slot is reusable. A subsequent submission can therefore overwrite or free metadata while the original stream may still read it. The same poison/immediate-free pattern appears in CUDA/ROCm transfer staging and in SYCL copy/ADD metadata and temporary cleanup. Root-run validation established the state-machine contradiction against `docs/BACKEND_CONTRACT.md:44-128`, but the simultaneous double-fault (both record attempts and drain fail) was not hardware-reproduced; no build/test/sanitizer gate was run for that fault seam. The impact is use-after-free/reuse, corruption, device faults, and nondeterministic later failures.

## Scope

- Narrowly remediate shared CUDA/ROCm `EventRing` to `MetadataSlotPool` release disposition: a failed-proof submission's metadata slot stays unavailable for reuse/free until a successful covering stream drain or queue teardown drain.
- Preserve current fast behavior when a recorded event synchronizes successfully or a covering stream drain is proven; retain the original token failure.
- Keep storage protected if teardown/drain itself fails; only release after a successful proof.
- Include the required double-fault fault-seam test that fails both event-record attempts and the stream drain, then proves no metadata reuse.
- Exclude transfer staging and all SYCL paths from this task; they remain explicit non-goals rather than additional remediation in this spec.

## Implementation references

- **Modify:** `src/shared/event_ring.hpp` — `EventRingState::on_worker_complete/on_worker_destroy` and state fields; own the explicit `Complete` versus `Retire/Unknown` disposition and delayed release.
- **Modify:** `src/shared/gpu_queue.hpp` — `GpuQueue::execute` and queue teardown; mark a submission retired when event recording and covering stream drain cannot prove completion, and drain before final retirement cleanup.
- **Read:** `src/shared/staging_pool.hpp` and `src/shared/transfer_pool.hpp` — existing poison/release and stream-scope conventions; do not change them in this task.
- **Read:** `src/cuda` and `src/rocm` policy/owner destruction paths — preserve context/stream activation and retained token errors while connecting shared metadata retirement.
- **Tests:** `test/cuda/test_cuda_smoke.cpp` and `test/rocm/test_rocm_smoke.cpp` — add the named double-fault seam that fails both fallback record and drain, then proves metadata is not reused.

## Requirements

- Represent completion disposition explicitly for each pooled metadata submission. `Complete` is permitted only after a successful recorded-event synchronization or a successful covering stream drain; `Retire/Unknown` is mandatory when submitted work has no such proof.
- Never synchronize a fresh/unrecorded event as proof for work that may already be enqueued. If both event-record attempts fail and stream synchronization returns false, retain the failed submission's metadata slot and device/host metadata storage outside all reusable pools.
- A retired slot must not be released, freed, or handed to a later operation until a successful covering stream drain or queue/device teardown drain proves completion. A failed drain keeps the storage protected and the queue must not reuse it.
- Preserve successful-path event-ring reuse and existing positive token/retained-error semantics. `on_worker_complete` must continue reporting the original failure; retirement is a lifetime disposition, not error replacement.
- Make queue teardown drain before final retirement cleanup. Cleanup after an unsuccessful teardown must retain protected storage rather than free it speculatively.
- Add a deterministic fault seam that fails the primary event record, the fallback `record_event_no_fault`, and `synchronize_stream_noexcept`; submit delayed pooled metadata over the inline 240-byte form, wait the failed token, submit another operation, and assert the old slot/address is not reused until a later successful covering drain.

## Non-goals

- Do not modify transfer staging (`StagingSlotPool`, `TransferStreamPool`, synchronous transfer) or any SYCL copy, ADD, metadata, staging, or queue path; those are explicitly excluded from this narrow ST-001 task.
- Do not add broad synchronization to successful operations, change public token/error semantics, alter tensor allocator ownership, or optimize staging/launch count.
- Do not collapse CUDA/ROCm event semantics into SYCL events or treat a fresh event as a completion fence.

## Acceptance criteria

- [ ] Under the double-fault seam, the failed token remains repeatably failed, its metadata slot/address is unavailable to later operations, and no free/reuse occurs while the original stream may still be pending.
- [ ] A later successful covering stream drain or queue/device teardown releases every retired metadata resource exactly once; a failed teardown leaves it protected rather than freeing it.
- [ ] Successful recorded-event and successful stream-drain paths retain current fast reuse and do not incur an additional synchronization.
- [ ] The queue remains usable for a later valid operation without metadata corruption, and source/owner destruction does not release the retired slot early.
- [ ] The fault test fails if either fallback event record or drain failure is omitted, preventing a one-fault seam from falsely proving safety.

## Verification

- `(remote-development: CUDA host)` sync, build the CUDA smoke/conformance targets with the README CUDA commands, and run `ctest --test-dir build --output-on-failure -R '^iom_cuda_.*_tests$'` with the double-fault seam — failed metadata remains retired, later work cannot reuse it, and teardown frees only after a successful covering drain; run the same scenario under `compute-sanitizer --tool memcheck` and `--tool racecheck`.
- `(remote-development: ROCm host)` sync, build the ROCm smoke/conformance targets with the README ROCm commands, and run `ctest --test-dir build --output-on-failure -R '^iom_rocm_.*_tests$'` with the corresponding HIP fault seam — failed-proof metadata remains protected and the configured ROCm memory/race instrumentation reports no reuse race.
- Run the focused shared state-machine/fault-seam test twice: once with both fallback record and drain failing (must retain), then after an injected successful covering drain (must release exactly once). Expected token error remains the original retained failure in both cases.

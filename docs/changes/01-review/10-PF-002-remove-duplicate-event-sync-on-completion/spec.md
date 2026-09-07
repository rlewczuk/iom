# CUDA/ROCm event completion synchronizes each event twice

**Order:** 10
**Priority:** P1
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** PF-002
**Review area:** Performance
**Review severity:** medium
**Review verification:** strongly-supported, confidence 95
**Review scope:** whole-codebase
**Backend scope:** cuda, rocm
**Location:** `src/shared/event_ring.hpp:99-130` (`EventRingState::on_worker_complete`, `EventRingState::on_worker_destroy`); `include/iom/iom.hpp:120-138` (`StagedWorker::process`)

## Outcome

Each CUDA/ROCm event is waited exactly once on normal task completion: `on_worker_destroy` skips the second synchronization when the preceding successful `on_worker_complete` wait already observed and cached the event result, while queue-drain and failed/partially-observed paths retain the required cleanup wait.

## Current problem

A completed event must be waited before metadata/registry resources are released, but normal completion must not repeat an already successful event wait.

For every non-no-op CUDA/ROCm queue task, `StagedWorker::process` (`include/iom/iom.hpp:120-138`) invokes `fence_complete` and then `fence_destroy`. The queue callbacks route these to `EventRingState::on_worker_complete` and `EventRingState::on_worker_destroy` (`src/cuda/copy.cu:115-138`, `src/rocm/copy.hip:114-137`). `on_worker_complete` (`src/shared/event_ring.hpp:99-113`) activates the context, calls `Policy::synchronize_event(slot.event)`, and caches the result; `on_worker_destroy` (`:115-130`) activates again and calls `Policy::synchronize_event_noexcept(slot.event)` before releasing metadata and the event-ring slot. CUDA and ROCm `synchronize_event`/`synchronize_event_noexcept` both issue runtime event synchronization calls (`src/cuda/copy.hpp:102-110`, `src/rocm/copy.hpp:93-100`). Normal queue completion therefore performs two event waits per task on the same slot; the destroy callback's wait remains necessary while draining uncompleted tasks at queue destruction. Impact: an extra `cudaEventSynchronize`/`hipEventSynchronize` plus context/mutex work per queued accelerator copy, likely material for small-copy/decode-like workloads; the wall-time effect was not measured in the review.

## Scope

- Track whether each `EventRingState::Slot`'s worker-complete callback successfully synchronized its event; reset the marker on slot acquire.
- Have `on_worker_destroy` skip the second wait only for that successful normal-completion state, retaining the noexcept wait for queue-drain and failed/partially-observed paths.
- Keep cached failure propagation, metadata release, and slot release ordering unchanged.

## Implementation references

- **Modify:** `src/shared/event_ring.hpp` — `EventRingState::Slot` plus `on_worker_complete`/`on_worker_destroy` (`:99-130`); owns the slot lifecycle and both synchronization sites.
- **Read:** `include/iom/iom.hpp` — `StagedWorker::process` (`:120-138`); the caller contract that both callbacks serve.
- **Read:** `src/cuda/copy.cu:115-138`, `src/rocm/copy.hip:114-137` — the queue callbacks that route `fence_complete`/`fence_destroy`.
- **Tests:** `test/cuda/test_cuda_smoke.cpp:645-770`, `test/rocm/test_rocm_smoke.cpp:237-362` — event reuse, capacity, cached results, and fault seams to rerun unchanged.

## Requirements

- Normal successful tasks invoke exactly one event synchronization per event; the event-ring slot still releases its metadata at the same ordering as today.
- A task drained during queue destruction, or whose first wait failed or was partially observed, must still perform the required cleanup wait and preserve repeatable failure behavior.
- Do not change event-ring capacity, registry fences, queue scheduling, or the cached-result contract.

## Non-goals

- No removal of event waits required for destruction/failure cleanup; no event-ring redesign; SYCL's cached fence path is out of scope.

## Acceptance criteria

- [ ] An event-synchronization counter shows one wait per normal successful queued CUDA/ROCm copy (down from two), including 16-slot reuse across many tasks.
- [ ] Queue destruction and injected launch/event-failure scenarios still wait (or handle the event) exactly as required, and cached failure propagation and metadata release remain unchanged.

## Verification

`actual validation: none (read-only review); proposed gates below`.

- CUDA and ROCm (remote-host work per remote-development; requires the corresponding GPU hosts): instrument the policy event-wait hooks, configure with testing enabled, build, and run `ctest --test-dir <build> -R 'iom_(cuda|rocm)_(smoke|conformance)_tests' --output-on-failure`.
- Scenario coverage: normal copies, injected launch/event failures, 16-slot event reuse, and queue destruction; expected: one event wait per normal task and no regression in cached-result or metadata behavior.

- `ctest --test-dir <build> -R 'iom_(cuda|rocm)_(smoke|conformance)_tests' --output-on-failure`
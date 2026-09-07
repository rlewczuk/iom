# Make GPU event-record fallback completion truthful

**Order:** 03
**Priority:** P0 — prevent false-success fences from releasing in-flight GPU work
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `ST-005`
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 85
**Review scope:** whole-codebase
**Backend scope:** cuda; rocm; common GPU queue/event ring
**Location:** `src/shared/gpu_queue.hpp:251-264` (event-record catch path), `src/cuda/copy.hpp:121-124`, `src/rocm/copy.hpp:111-114`, `src/shared/event_ring.hpp:147-164`

## Outcome

A GPU event is exposed as a completion fence only after recording succeeds. If both event-record attempts fail, the queue performs a successful stream drain before exposing completion; otherwise the work and associated operands remain retained/quarantined, and an unrecorded event can never be interpreted as successful synchronization.

## Current problem

The invariant is that event-ring retirement may release metadata, staging, and operands only after a fence known to represent the submitted GPU work has completed. In `src/shared/gpu_queue.hpp:251-264`, the catch path calls `record_event_no_fault` and unconditionally sets `event_recorded=true`. The CUDA and ROCm policy wrappers at `src/cuda/copy.hpp:121-124` and `src/rocm/copy.hpp:111-114` discard the record status. `src/shared/event_ring.hpp:147-164` then treats a successful synchronize as completion.

Runtime probes on CUDA 13.2.78 and HIP gfx1036/gfx1201 show that synchronizing a fresh never-recorded event returns Success. Therefore a failed fallback record can leave a fresh or stale event that synchronizes successfully, falsely releasing entries while kernel work remains in flight. The root is not repaired by the normal event wait result because the event was never confirmed to represent this submission.

## Scope

- Return and propagate event-record status from CUDA/ROCm `record_event_no_fault` and the shared queue fallback.
- Set `event_recorded` only after a confirmed successful record of the event for the current work.
- If both record attempts fail, require a successful stream synchronize/drain before exposing completion. If drain fails, retain/quarantine the event-ring entries and operands with a failed fence.
- Make `EventRing` reject or retain any unrecorded event; never interpret a successful synchronize on an unrecorded event as completion.

## Implementation references

- **Modify:** `src/shared/gpu_queue.hpp:251-264` — event-record catch/fallback; preserve record status and choose drain-versus-retain disposition.
- **Modify:** `src/cuda/copy.hpp:121-124` and `src/rocm/copy.hpp:111-114` — return failure from no-fault record wrappers instead of discarding it.
- **Modify:** `src/shared/event_ring.hpp:147-164` — require `event_recorded` before treating synchronization as successful and retain failed entries.
- **Read:** existing event-ring cleanup and stream synchronization policy methods in `src/shared/event_ring.hpp` and backend copy policy headers; reuse no-throw error/retention conventions.
- **Tests:** shared GPU queue/event fault seam and CUDA/ROCm smoke/conformance tests; extend the seam to fail both record calls after enqueue.

## Requirements

- The record operation must report a definitive success/failure status through every backend policy layer.
- `event_recorded` must be false unless the current event record call returned success; never set it optimistically in a catch path.
- After two failed record attempts, release entries only following a successful stream synchronize/drain. A failed drain must preserve entries and operands for later completion or quarantine.
- `EventRing` must distinguish an unrecorded event from a recorded event even when vendor synchronize returns Success for the former.
- Preserve the original submission/record error where applicable, and keep cleanup paths non-throwing and ownership-safe.

## Non-goals

- Do not change kernel copy semantics, event-ring capacity policy, allocator API, or backend stream ownership.
- Do not assume vendor event synchronization semantics are a completion proof for never-recorded events.
- Do not add unrelated synchronization to successful event-record paths.

## Acceptance criteria

- [ ] A successful first or fallback record marks the event recorded and normal event-ring completion releases entries only after that event completes.
- [ ] A fault seam that fails both record calls after enqueue leaves entries and operands retained until a successful stream drain; no failed-fence path releases or reuses them.
- [ ] A fresh never-recorded CUDA/HIP event returning Success from vendor synchronize is still rejected as completion by `EventRing`.
- [ ] Repeated retained failures are safe and non-throwing, and subsequent valid work preserves correct data and ownership ordering.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync cuda 03-ST-005-gpu-queue-event-record-fallback`
- `.agents/skills/remote-development/scripts/remote-exec cuda 03-ST-005-gpu-queue-event-record-fallback 'cmake -S . -B build-cuda -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-cuda --target iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-cuda --output-on-failure -R "iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — CUDA smoke, conformance, and coexistence pass.
- `.agents/skills/remote-development/scripts/remote-sync rocm 03-ST-005-gpu-queue-event-record-fallback`
- `.agents/skills/remote-development/scripts/remote-exec rocm 03-ST-005-gpu-queue-event-record-fallback 'cmake -S . -B build-rocm -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-rocm --target iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-rocm --output-on-failure -R "iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — ROCm smoke, conformance, and coexistence pass.
- Run the extended fault seam on CUDA and ROCm, failing both record calls after enqueue; expect retained entries on failed drain, release only after successful drain, and no operand reuse while work may remain in flight. The existing probes remain a negative check: never-recorded-event synchronize Success must not satisfy the completion assertion.

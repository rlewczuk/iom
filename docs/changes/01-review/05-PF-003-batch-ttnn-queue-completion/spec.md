# TTNN queue finishes shared mesh command queue once per task

**Order:** 05
**Priority:** P1
**Blocked by:** 03-ST-002-fence-ttnn-native-submission-failures
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** PF-003
**Review area:** Performance
**Review severity:** high
**Review verification:** strongly-supported, confidence 91
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/device.cpp:438-450` (`TtnnQueue::execute`); `src/ttnn/device.cpp:489-530` (`TtnnQueue::complete_task`); `include/iom/iom.hpp:120-138` (`StagedWorker::process`)

## Outcome

Contiguous non-no-op TTNN tasks that form one ready batch on the shared mesh command queue are completed with a single `mesh_command_queue(0).finish()` instead of one finish per task, while per-token ordering, repeatable wait/failure behavior, and tensor lifetime protection are preserved and other queues can enqueue without a finish-held `api_mutex` blocking between every operation.

## Current problem

`TtnnQueue::execute` enqueues all planes through the one device `mesh_command_queue(0)` under `api_mutex` (`src/ttnn/device.cpp:438-450`; `copy_planes` issues native copies per plane at `src/ttnn/copy.cpp:284-292`). `StagedWorker::process` completes one task before processing the next (`include/iom/iom.hpp:120-138`); for each outcome `TtnnQueue::complete_task` calls `finish_native`, which reacquires the same device mutex and invokes `mesh_command_queue(0).finish()` (`src/ttnn/device.cpp:489-503`, `:336-345`). So a caller submitting multiple copies without waits encounters a finish barrier between tasks, redundant finishes when several tasks were already enqueued, and other TTNN queues cannot enqueue while the finish holds the global API mutex. TTNN's internal contract explicitly describes `copy_planes` as returning after submission with the caller synchronizing once per operation (`src/ttnn/copy.hpp:41-45`), so a per-task finish is not required for correctness of an already-synchronized batch. CUDA/ROCm use recorded events and SYCL uses `sycl::event` waits as counterparts; their normal completion paths do not call a device-wide queue finish per task. Impact: at least one `MeshCommandQueue::finish()` per submitted TTNN copy with lost batching and cross-queue overlap — expected throughput degradation for sequences of no-wait copies; exact magnitude is unmeasured.

## Scope

- Batch contiguous TTNN task completion on the shared mesh queue: enqueue a ready batch under the existing API-mutex ownership, perform exactly one `mesh_command_queue(0).finish()`, then release registry entries and complete every token in submission order from that batch without another finish.
- Preserve queue-destruction draining, per-token repeatable waits, and tensor lifetime protection, and handle finish failure for the whole batch.

## Implementation references

- **Modify:** `src/ttnn/device.cpp:489-530` — `TtnnQueue::complete_task`: collect a ready batch of contiguous outcomes, perform one `finish_native` for the batch, then complete each token in submission order.
- **Read:** `src/ttnn/device.cpp:438-450` — `TtnnQueue::execute` (batch enqueue under `api_mutex_`), `:336-345` — `finish_native`, and `TtnnQueue::fence_through_sequence` — the existing sequencing/lifetime machinery to drive batch cutover and failure handling.
- **Read:** `src/ttnn/device.cpp:459-482` — the failure/invalidation path established by ST-002 (03-ST-002-fence-ttnn-native-submission-failures); this task builds on its transaction-boundary and retention semantics, so it must land after that change.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:472-519` — TTNN async conformance anchors; `test/backend/test_backend_coexistence.cpp:337-453` — multi-queue coexistence.

## Requirements

- For N submitted non-no-op TTNN copies that form one ready batch, exactly one native queue finish occurs; all N tokens become waitable in submission order with repeatable results.
- On finish failure, invalidate or retain the failure for every affected token in the batch, and keep the ST-002 ownership guarantees (no in-flight native command without a completion owner).
- Preserve queue-destruction draining and tensor lifetime protection; do not remove the required finish behavior for host transfers or destruction paths.
- Multiple TTNN queues must be able to enqueue work without a finish-held mutex between every operation.

## Non-goals

- Changing TTNN native plane mapping, adding compute kernels, or altering per-operation native submission contents.
- Removing required finish behavior for host transfer/destruction.
- Changing CUDA/ROCm/SYCL scheduling.
- Redesigning `StagedWorker` publishing beyond what batch completion requires.

## Acceptance criteria

- [ ] A trace of N=1, 8, 32 non-no-op TTNN copies submitted without waits from one queue shows one native queue finish per ready batch rather than one per task, and all N tokens complete in submission order.
- [ ] Multiple queues submitting concurrently can enqueue without an API-mutex-held finish between every operation; registry/tensor lifetime checks still pass, failures remain repeatable per token, and queue destruction still drains.
- [ ] Existing TTNN conformance and backend coexistence suites pass; no-wait batch throughput and queue occupancy do not regress relative to the pre-change per-task finish (metrics recorded, see Verification).

## Verification

Actual validation: none (read-only review); proposed gates below.

Proposed gates (TTNN hardware on a remote host per remote-development; sync the workspace, then run via `remote-exec`):

- Add a temporary TTNN microbenchmark/trace around `TtnnQueue`: submit N=1, 8, 32 non-no-op copies without waits from one and from multiple queues, count `finish_native`/`mesh_command_queue(0).finish()` calls, and record queue occupancy and wall time; compare per-task vs batched completion.
- Run the existing TTNN conformance and coexistence suites for correctness: `ctest --test-dir <build> -R '^(iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests)$' --output-on-failure`.

- `ctest --test-dir <build> -R '^(iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests)$' --output-on-failure`
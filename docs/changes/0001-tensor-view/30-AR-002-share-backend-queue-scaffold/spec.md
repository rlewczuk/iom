# Share the four backend staged-worker queue scaffolds

**Order:** 30
**Priority:** P1 — medium, verified architecture defect. CPU, CUDA, ROCm, and TTNN duplicate the same queue staging, worker, completion, and destruction-drain state machine, with already-divergent publish and failure behavior.
**Blocked by:** `29-AR-001-share-cuda-rocm-copy-queue`, `28-ST-007-commit-sequence-before-queue-work`, `27-ST-006-bound-rocm-failure-wait`
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-002`
**Review severity:** medium
**Review verification:** verified, confidence 92

## Outcome

One backend-neutral `iom::detail::StagedWorker<Task>` class template in `include/iom/iom.hpp` owns the common staged-list, in-order worker, publish, completion, and shutdown-drain mechanics used by CPU, CUDA, ROCm, and TTNN. Each backend queue retains only its typed task, validation and no-op checks, native execute callback, native fence callbacks, and completion callback.

The helper has exactly four backend callbacks: `execute`, `fence_complete`, `fence_destroy`, and `complete`. It has no CUDA, HIP, TTNN, SYCL, allocator, `TensorView`, or `Device` knowledge. CUDA and ROCm's shared tiled-copy implementation comes from AR-001; AR-002 replaces AR-001's private `GpuQueue<Policy>` worker state machine with this helper instead of creating a second queue abstraction.

The helper preserves sequence-before-work, commit-before-complete, pre-link rollback, post-link retained-failure, repeated-wait, no-destruction-wait, and exactly-once completion contracts. ST-001, ST-002, ST-003, ST-006, and ST-007 consume the stable callback/task boundary established here.

## Current failure

The four queue classes duplicate the same state:

- `src/cpu/device.cpp` (`CpuQueue`)
- `src/cuda/copy.cu` (`CudaQueue`, after AR-001's shared copy migration)
- `src/rocm/copy.hip` (`RocmQueue`, after AR-001's shared copy migration)
- `src/ttnn/device.cpp` (`TtnnQueue`)

Each currently owns a mutex, completion condition, staged list, task list, worker thread, publish operation, worker loop, completion call, destructor drain, validation helper, identical-window check, and unsupported-operation helper. CPU/TTNN publish handling differs from CUDA/ROCm, and native post-link failure handling is not represented by one shared contract. A queue fix therefore requires four edits and can silently drift again.

The review's concrete invariant is that a task is staged before backend work, published only after its fence field is ready, completed once in queue order, and drained during destruction without a whole-stream wait. The remediation centralizes that invariant while leaving backend-native execution and failure classification in the composing queue.

## Scope

- **Create:** `iom::detail::StagedWorker<Task>` in `include/iom/iom.hpp` as a header-only class template. Add only the standard-library includes needed by its declarations and inline/template definitions.
- **Modify:** the four queue classes listed above to hold one non-copyable, non-movable `StagedWorker<Task>` value member and to construct it with the four callbacks.
- **Modify:** `DeviceOps` with protected backend-neutral `validate_copy`, `identical_window`, and `unsupported` helpers. Keep `DeviceOps::copy` and compute-operation methods pure virtual and keep the public API unchanged.
- **Migrate:** CUDA and ROCm queue execution from AR-001's `GpuQueue<Policy>` worker mechanics to `StagedWorker<BackendTask>`. Keep AR-001's shared tiled algorithms and typed runtime policies; remove the old shared queue state machine.
- Preserve each backend's native task payload and fence representation. Add only `void* fence = nullptr` as the common opaque fence slot; CUDA/ROCm store their event handle there after successful record, CPU/TTNN leave it null.
- Preserve backend-local `SubmissionFault` seams, native error categories/messages, and the ST-006 post-link retained-failure route through `DeviceOps::commit_failure`.

No public queue API, token encoding, tensor ownership, allocator, backend selection, or compute implementation changes are included.

## Implementation references

- **Modify:** `include/iom/iom.hpp` — add the helper and protected static `DeviceOps` helpers. The helper stores exactly eight data members: four callbacks, publish policy, mutex, condition variable, two `std::list<Task>` containers, shutdown flag, and worker thread. The public helper operations are `start()`, `submit_copy(Task)`, and `shutdown_and_drain()`; copy and move are deleted.
- **Modify:** `src/iom.cpp` only if the existing `DeviceOps` helper definitions cannot remain inline; `submit`, `complete`, `commit_failure`, token encoding, queue IDs, and sequence bit layout remain unchanged except for ST-007's already-specified contract.
- **Modify:** `src/cpu/device.cpp` — remove queue-local validation, identical-window, unsupported, publish, run, and drain scaffolding. Keep `copy_elements` as the execute callback. Use `PublishPolicy::CompleteOnThrow`; CPU's fence callbacks are null-safe no-ops.
- **Modify:** `src/cuda/copy.cu` — remove the queue worker body supplied by AR-001, retain CUDA policy and copy execution, and construct `StagedWorker<CudaTask>` with CUDA event synchronize/destroy callbacks and `PublishPolicy::Splice`.
- **Modify:** `src/rocm/copy.hip` — mirror CUDA with HIP event callbacks and `PublishPolicy::Splice`.
- **Modify:** `src/ttnn/device.cpp` — remove queue-local scaffolding, keep TTNN native copy execution and required API mutex, and construct `StagedWorker<TtnnTask>` with null fence callbacks and `PublishPolicy::CompleteOnThrow`.
- **Read:** `include/iom/iom.hpp` and `src/iom.cpp` — preserve `submit` reservation/rollback, `complete`, `commit_failure`, repeated wait, and token semantics from ST-006/ST-007.
- **Read:** `docs/changes/0001-tensor-view/29-AR-001-share-cuda-rocm-copy-queue/spec.md` — consume its shared tiled-copy body and remove only its `GpuQueue<Policy>` state machine.
- **Read:** `docs/changes/0001-tensor-view/27-ST-006-bound-rocm-failure-wait/spec.md` and `28-ST-007-commit-sequence-before-queue-work/spec.md` — preserve post-link null-fence failure routing and sequence rollback exactly.
- **Tests:** extend common CPU/backend tests with helper callback-order, pre-link cleanup, post-link retained-failure, null-fence, shutdown-drain, and no-destruction-wait cases. Do not remove existing backend conformance cases.

## Helper contract

```text
namespace iom::detail {

template <typename Task>
class StagedWorker {
public:
    using Execute = std::function<void(Task&)>;
    using FenceComplete = std::function<void(void*)>;
    using FenceDestroy = std::function<void(void*)>;
    using Complete = std::function<void(std::uint64_t, std::exception_ptr)>;

    enum class PublishPolicy { Splice, CompleteOnThrow };

    struct Callbacks {
        Execute execute;
        FenceComplete fence_complete;
        FenceDestroy fence_destroy;
        Complete complete;
    };

    explicit StagedWorker(Callbacks callbacks, PublishPolicy publish_policy);
    ~StagedWorker();
    StagedWorker(const StagedWorker&) = delete;
    StagedWorker& operator=(const StagedWorker&) = delete;
    StagedWorker(StagedWorker&&) = delete;
    StagedWorker& operator=(StagedWorker&&) = delete;

    void start();
    void submit_copy(Task task);
    void shutdown_and_drain();
};

}  // namespace iom::detail
```

`submit_copy` stages the task under the helper mutex before invoking `execute` outside the mutex. The composing queue has already reserved the sequence through `DeviceOps::submit`; the task's sequence is therefore committed before native work begins. On successful execute, the helper copies the task's opaque fence into the staged entry under the mutex and publishes it. On a pre-link throw, it erases the staged entry exactly once and rethrows the original exception so the outer `submit` rollback runs. On a post-link failure, the execute callback destroys any unrecorded native resource, clears the fence, calls the composing queue's protected `commit_failure(sequence, current_exception())`, and returns normally; the helper publishes the null-fence task and the later completion makes the retained failure observable through `wait`.

The worker pops tasks in submission order, unlocks the helper mutex, calls `fence_complete` only for a non-null fence, calls `fence_destroy` once, and calls `complete(sequence, failure)` once. A fence-completion exception becomes the completion failure; a retained post-link failure remains owned by `DeviceOps::commit_failure`. Completion callbacks run outside the helper mutex to avoid lock-order cycles.

`shutdown_and_drain` marks shutdown, notifies, joins the worker, then drains any staged entries. Both the worker's shutdown path and staged-list drain call `fence_destroy` for non-null fences and `complete(sequence, nullptr)` exactly once. They never call `fence_complete`, `cudaStreamSynchronize`, `hipStreamSynchronize`, TTNN mesh `finish`, or another whole-stream barrier. The composing queue destroys native streams and other queue resources only after the helper returns.

The helper must move-construct tasks from list nodes (`Task current(std::move(tasks_.front())); tasks_.pop_front();`), never default-construct and assign a task. This remains valid when ST-002 changes CPU/TTNN view members to non-assignable value members.

## Backend boundary

Each backend task preserves its existing payload and gains only the opaque fence member:

- CPU: sequence, source view pointer, destination view pointer, no-op flag, null fence.
- CUDA: sequence, source/destination view pointers, native event payload, no-op flag, opaque fence.
- ROCm: sequence, source/destination view pointers, native event payload, no-op flag, opaque fence.
- TTNN: sequence, source/destination view/native plane payloads, no-op flag, null fence; ST-002 later replaces caller-view pointers with value views.

The four callbacks are the only helper-to-backend interface. `execute` is a queue-owned closure that captures the composing queue's `this`; it performs native work and owns pre-link/post-link classification. `fence_complete` and `fence_destroy` are static or non-capturing typed adapters. `complete` forwards to the composing queue's existing `DeviceOps::complete`. No `commit_failure` callback is added.

`DeviceOps::validate_copy` checks both views belong to the queue's device and have identical `TensorSpec`; it throws the existing `std::invalid_argument` messages. `DeviceOps::identical_window` compares native handle, plane offset, and every plane stride. `DeviceOps::unsupported` constructs the existing backend-prefixed `std::runtime_error`. Every backend's six compute-operation rejectors call this helper.

A per-queue `submission_order_mutex_` serializes the entire public `copy` call through `DeviceOps::submit`, so concurrent callers cannot publish a higher reserved sequence before a lower one. The helper itself remains unaware of sequence allocation and device metadata.

## Requirements

1. `StagedWorker<Task>` is the only definition of staged-list publish, worker loop, completion, and shutdown-drain mechanics for CPU, CUDA, ROCm, and TTNN.
2. `Callbacks` has exactly four `std::function` fields named `execute`, `fence_complete`, `fence_destroy`, and `complete`; no fifth callback or backend-specific branch exists in the helper.
3. The helper contains no backend SDK include, native runtime type, `TensorView`, `Device`, allocator, backend switch, or active-backend registry.
4. Every task is staged before `execute`; no task is published until its fence field has been copied into the staged entry under the helper mutex.
5. Pre-link execute failures erase their staged entry exactly once, rethrow the original exception, and allow ST-007's reservation rollback. No completion is reported for an uncommitted sequence.
6. Post-link CUDA/ROCm failures retain the failure through the composing queue's `commit_failure`, publish a null-fence task, and return a valid token whose repeated waits rethrow the retained failure.
7. Worker success waits on a non-null native fence, destroys it exactly once, and reports completion exactly once. Null fences skip both native callbacks.
8. Shutdown drains every task and staged entry exactly once without waiting on the backend stream/device/mesh. Native resource destruction occurs before queue-owned stream/device cleanup.
9. Move construction, not move assignment, extracts tasks from both lists. The helper remains compatible with ST-002's deleted `TensorView` assignments.
10. Validation, identical-window no-op, unsupported-operation messages, sequence allocation, token encoding, and public `DeviceOps` signatures preserve existing behavior.
11. CUDA/ROCm consume AR-001's shared standard-tiled algorithms; no second `GpuQueue<Policy>` worker state machine remains after migration.
12. Existing CPU, CUDA, ROCm, and TTNN fault seams and conformance assertions remain present and observe the same exception categories/messages and token behavior.
13. Queue owners remain non-copyable/non-movable; the creating device and allocator lifetime contract is unchanged.

## Non-goals

- Changing `DeviceOps::submit`, `wait`, `complete`, `commit_failure`, token encoding, queue IDs, or sequence bit layout beyond the separate ST-006/ST-007 contracts.
- Changing tiled-copy algorithms, staging allocation, transfer streams, bit packing, or TTNN native storage; AR-001 and PF-002/PF-003/PF-006 own those changes.
- Changing CUDA/ROCm event fences to another synchronization primitive; ST-001 owns destruction-time event/stream fencing.
- Changing CPU/TTNN view ownership or tensor destruction; ST-002 and ST-003 own those hazards.
- Adding SYCL queue migration; the current finding covers CPU, CUDA, ROCm, and TTNN only.
- Adding public queue classes, virtual dispatch, shared ownership, cancellation, cross-queue dependencies, or new backend selection.
- Removing existing backend conformance cases or hiding unsupported operations behind no-op success.

## Acceptance criteria

- [ ] `include/iom/iom.hpp` contains one `StagedWorker<Task>` definition with exactly four callback fields and no backend token.
- [ ] CPU, CUDA, ROCm, and TTNN queue classes each contain one helper value member, construct it with the four callbacks, start it once, and call `shutdown_and_drain` before native queue-resource destruction.
- [ ] `src/cuda/copy.cu` and `src/rocm/copy.hip` no longer contain a second worker/publish/drain skeleton after consuming AR-001's shared copy body.
- [ ] Source audit finds one canonical `validate_copy`, one canonical `identical_window`, one canonical `unsupported`, and one staged-worker comment; backend sources contain no duplicate message literals or helper bodies.
- [ ] A helper-level test observes execute, fence-complete, fence-destroy, and complete in order on a successful fenced task.
- [ ] A pre-link execute-throw test observes staged-entry removal, no completion, original exception propagation, and sequence reuse on the next successful submission.
- [ ] CUDA/ROCm `event_create` failures preserve throw-before-token behavior; third-plane or equivalent post-link failures return a token and repeated waits rethrow the retained exception.
- [ ] A null-fence test verifies neither fence callback runs and completion still occurs once.
- [ ] Shutdown tests verify every task is completed once, every non-null fence is destroyed once, and no fence-complete or whole-stream synchronization is called during drain.
- [ ] CPU, CUDA, ROCm, and TTNN conformance suites pass without skipped cases; existing unsupported-operation, identical-window, ordering, repeated-wait, and lifetime assertions remain green.
- [ ] The helper compiles in the common target with accelerators disabled and in each enabled backend target.

## Verification

Run the local CPU/common build and tests with accelerator backends disabled. Add and run focused helper tests for callback order, pre-link cleanup, retained post-link failure, null fence, sequence rollback, and shutdown drain. Run CUDA and ROCm conformance and fault-injection tests on hardware using `.agents/skills/remote-development`, including event creation, third-plane launch, repeated wait, and queue destruction paths. Run TTNN conformance on the configured remote host, including native copy and no-destruction-wait cases. Use source audits to verify exactly one helper and no stale `GpuQueue<Policy>` worker body. Clean each remote mirror after verification.

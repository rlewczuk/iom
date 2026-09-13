# Park fixed-resource admission without recursive or blocking retry

**Order:** 02
**Priority:** P0 — accepted queue work can lose liveness
**Blocked by:** None
**Review source:** `cpp-inference-gpu-stability` — whole-codebase review, branch `main`, HEAD `cb88619865ff9bcda0a4e3bac4bae4bd4bf510a8` (`Move cpp-inference-review skill to .omp`), baseline `origin/main`; clean at scope capture; upstream `origin/main` ahead 4/behind 0
**Finding:** ST-001
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 95
**Review scope:** whole-codebase
**Backend scope:** common admission, CUDA, ROCm, SYCL
**Location:** `src/device_ops.cpp:125-186` — `DeviceOps::pump_admission`; `src/shared/gpu_queue_operations.inl:20-90` — fixed-resource dispatch; `src/sycl/queue.cpp:319-332` and `src/sycl/queue_binary.cpp:275-282` — blocking resource acquisition

## Outcome

Fixed-resource exhaustion remains a nonterminal parked FIFO state: accepted work never recurses or blocks the submitter/worker, proven release wakes one oldest node through a race-safe retry/recheck, and close deterministically terminalizes parked work without releasing or reusing quarantined resources.

## Current problem

This task has two explicit mechanisms under one progress invariant. In common admission, `DeviceOps::pump_admission` catches `detail::AdmissionResourceUnavailable`, restores the same FIFO node and provisional credit, clears `admission_pumping_`, and immediately calls `pump_admission` again (`src/device_ops.cpp:169-185`). CUDA/ROCm dispatch uses nonblocking `EventRing::try_acquire` and `MetadataSlotPool::try_acquire` (`src/shared/gpu_queue_operations.inl:20-27,75-86`). A quarantined event remains `in_use` until a successful covering drain (`src/shared/event_ring.hpp:249-289,303-318`), so recursive calls can make no state or resource progress and exhaust the stack in a `noexcept` path instead of returning an accepted token.

SYCL has a distinct backend mechanism: `SyclCompletionPool::acquire` and metadata acquisition wait on condition variables (`src/sycl/queue_support.hpp:41-58,85-106`; `src/sycl/queue.cpp:323-330`, `src/sycl/queue_binary.cpp:278-281`). `StagedWorker::execute` is synchronous in the callback (`include/iom/detail/staged_worker.hpp:59-75`), so a FIFO tail pumped from `complete()->pump_admission` can block the worker while all completion slots are protected. The worker cannot join during queue destruction, and protected slots are only released by covering proof. The source establishes these paths; stack exhaustion and the SYCL deadlock are strongly-supported consequences, not accelerator hardware reproductions. Existing fault/smoke tests destroy or drain affected queues and do not enqueue a resource-using tail on the live queue.

## Scope

- In common `DeviceOps` admission, restore an unavailable node as nonexecuting parked FIFO work, release the provisional credit, and return without recursion or busy-spin; retry only from an actual progress path.
- In SYCL, provide nonblocking completion and required metadata acquisition that reports the same nonterminal `AdmissionResourceUnavailable` signal, so `execute`/`execute_binary` cannot block the submitter or completion worker; preserve backend-specific pool ownership.
- Define race-safe release notification and recheck: a proven release must notify the admission progress path without holding pool locks, and the retry must recheck queue closing, FIFO head identity, resource availability, and node state under the owning synchronization so a release cannot be lost or wake a non-head node.
- Define the all-protected live-queue outcome: when every fixed completion/metadata resource is unavailable or quarantined, a newly accepted resource-using tail remains parked with its token/leases retained until covering proof makes a resource reusable, or until close terminalizes it; it is never recursively retried, synchronously waited, or silently dropped.

## Implementation references

- **Modify:** `src/device_ops.cpp` — `DeviceOps::pump_admission`, `complete`, and `close_and_drain`; own FIFO parking, credits, wake/recheck, close terminalization, and retained outcomes.
- **Modify:** `src/shared/gpu_queue_operations.inl` — `GpuQueue::copy_impl`/`binary_impl` fixed-resource try-acquire paths; keep CUDA/ROCm nonblocking dispatch and release ordering.
- **Read/modify as required:** `src/shared/event_ring.hpp` — `try_acquire`, quarantine, `on_worker_destroy`, `on_queue_drain`, and proven release; `src/shared/metadata_slot_pool.hpp` — release/protection; add at most one release-to-admission wake mechanism outside pool locks.
- **Modify:** `src/sycl/queue_support.hpp` — `SyclCompletionPool`/metadata pool try-acquire and protected release; `src/sycl/queue.cpp` and `src/sycl/queue_binary.cpp` — return the common nonterminal signal instead of blocking.
- **Read:** `include/iom/detail/staged_worker.hpp:59-105,127-148` — synchronous callback and completion ordering; do not make all `StagedWorker` users asynchronous.
- **Tests:** `test/cuda/test_cuda_smoke.cpp:742-815`, `test/rocm/test_rocm_smoke.cpp:362-438`, `test/sycl/test_sycl_smoke.cpp:812-875`, plus common admission/fault fixtures; add a focused repeated-unavailable admission case.

## Requirements

- On `AdmissionResourceUnavailable`, leave the FIFO node accepted, nonexecuting, and parked; restore only provisional admission state and return with bounded call depth. This signal must remain nonterminal and must not become an operation failure.
- Make normal proven completion release EventRing/metadata resources before `DeviceOps::complete` performs one admission pump attempt. Add no duplicate or pool-lock-held callback; if a failed dispatch can partially acquire resources, preserve/prove fixed-resource ordering or release and wake exactly once after the partial resource is safely returned.
- Implement SYCL nonblocking `try_acquire` for completion and required metadata resources and route exhaustion through `AdmissionResourceUnavailable`. Never wait on a condition variable from the submitting thread or the worker executing `complete()->pump_admission`.
- Make release notification/recheck race-safe: notification must be paired with a state transition that a retry observes while holding the correct admission/pool synchronization; every retry rechecks closing, node still parked/head, credits, and resource availability before dispatch.
- Preserve FIFO order, admission credits, OID/token and sequence encoding, owner/workspace lease rollback and retention, repeatable failures, native error categories, and EventRing/metadata quarantine. Unknown or unproved resources remain unavailable until covering proof.
- During close, stop acceptance and further pumping, then deterministically terminalize every parked node with the existing drained-before-completion/queue-closed error category or equivalent retained result. Close must complete even when all resources are protected, and must not rewrite prior token outcomes or weaken covering proof.

## Non-goals

- Do not remove quarantine, reuse unknown resources, weaken covering-drain proof, alter queue capacity/four-queue limits, change OID encoding/failure categories, add timeout-only recovery, or redesign backend resource ownership.
- Do not change operation arithmetic/capabilities, cross-queue ordering, SYCL host fallback, or unrelated admission/performance machinery.
- Do not split the common recursion and SYCL blocking acquisition into separate tasks; both mechanisms are required parts of this coordinated remediation.

## Acceptance criteria

- [ ] With a live CUDA or ROCm queue at `C=1`, inject the existing event-record and stream-drain failures, submit a valid non-no-op tail before completion, and observe prompt positive parked-token return with bounded pump depth, no tail native/resource effect before proof, repeatable first failure, FIFO dispatch after covering proof, and no process termination.
- [ ] With SYCL completion/metadata slots protected, a non-no-op copy or binary tail never blocks submitter or worker, remains parked through the same nonterminal signal, and queue destruction completes with deterministic retained outcomes; race-safe proven-release wake/recheck dispatches exactly the oldest eligible node while no protected resource is reused.

## Verification

- `ctest --test-dir <build> --output-on-failure -R '^(iom_cuda_conformance_tests|iom_rocm_conformance_tests|iom_sycl_conformance_tests|iom_backend_conformance_cpu_tests)$'` after adding the focused common admission case; accelerator runs must use remote-development hosts.
- Run the injected-fault C=1 live-queue scenarios under a watchdog, then covering proof/release and close/drain scenarios. The supplied root context passed local CPU targets `iom_tests`, `iom_scalar_add_tests`, and `iom_backend_conformance_cpu_tests` (3/3); CUDA/ROCm/SYCL sync attempts timed out before build, so recursive-crash/deadlock absence remains unverified here.

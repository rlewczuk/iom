# Adopt bounded admission on CUDA and ROCm

**Order:** 09
**Priority:** P1 — shared standard-GPU queues must use fixed resources without submission-time blocking or allocation.
**Blocked by:** `06-fixed-queue-resources`, `07-binary-workspace-cutover`, `08-common-fifo-admission`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

CUDA and ROCm adopt the common bounded FIFO admission contract through the policy-templated queue implementation. Once a device is set up, accepted work may park indefinitely in host memory, while native credits, completion events, metadata slots, upload mirrors, streams, workspaces, and tensor payload references are acquired, used, retired, and quarantined without synchronous submission-time waits or operation-time native allocation/free calls. CUDA and ROCm retain their existing kernel, copy, view, padding, alias, broadcast, numerical, and error behavior except for the specified admission and resource-lifetime rules.

## Scope

Implement the CUDA/ROCm adoption in the shared queue and event machinery, backend policies and copy translation units, device-owned queue state, and focused smoke/conformance coverage. The configured per-queue capacity is the immutable, nonzero `QueueConfig::max_in_flight_per_queue` value, denoted **C**. Every live queue owns one disjoint C-slot metadata partition and C precreated completion resources; a device permits at most four live queues. The queue state is the exact Device-owned state for that queue and must not borrow resources or ordering from another queue or Device.

Acceptance of a valid operation performs validation, an immutable request/view snapshot, owner/tensor/raw-workspace registration and exclusivity leasing, token/sequence reservation, and FIFO append transactionally. It may allocate host snapshot/node state, but it must not acquire native resources, synchronously wait for resources, allocate native memory, launch work, mutate an output, or otherwise create a native effect before returning the positive token. Invalid input and failed preacceptance host preparation return synchronously with no token or effects and roll back every registration, lease, and reservation.

At most C accepted requests per queue are dispatchable at once. At C saturation, at least C+2 additional valid requests must promptly return positive tokens and remain parked host-side. Parked nodes hold no native credit, completion event/resource, metadata slot, upload mirror, or native effect. They retain their immutable snapshot, token outcome state, owner registrations, and workspace lease. Only the oldest dispatchable FIFO head may acquire one native credit, one precreated completion event generation, and at most one metadata slot when its operation requires metadata. No later node can bypass that head.

Completion processing is autonomous and must not depend on a user calling `wait` or submitting another operation. After proving stream/event, descriptor, upload, workspace, tensor-payload, and host-mirror use is complete, it caches the token result independently of reusable native state, returns safe resources, and dispatches exactly the oldest parked head, one at a time. Inline, no-op, and no-metadata operations consume the same credit and FIFO position but never acquire dummy metadata. There is no cross-queue ordering or device-wide synchronization.

Failures discovered after acceptance remain terminal, repeatable token results. An unknown event or stream completion quarantines the entire unresolved lease: credit, event generation, metadata partition/slot, host mirror/upload backing, workspace, and tensor payload references. Quarantine reduces usable capacity and prevents partition or arena release until a covering queue/device proof. Queue destruction closes admission, finishes host preparation, drains every accepted request in FIFO to a terminal result, and transfers unresolved leases to Device-owned quarantine. Without covering proof it retains the queue partition, queue-count reservation, and referenced arena backing; it never silently cancels accepted work or reassigns unsafe slots.

## Implementation references

- `src/shared/gpu_queue.hpp`: planned bounded FIFO admission, parked-node state, dispatch/retirement, retained outcomes, queue close/drain, and Device quarantine integration in `detail::GpuQueue<Policy>`.
- `src/shared/event_ring.hpp`: planned fixed-capacity completion resources in `detail::EventRingState<Policy>`; eager creation of C events at queue setup, safe generation retirement, autonomous completion observation, and unknown-completion quarantine. Remove saturation waits, lazy creation, replacement, and resizing from the dispatch path.
- `src/shared/metadata_slot_pool.hpp`: consume the fixed device metadata arena and per-queue C-slot partition established by queue setup; no submission-time growth, replacement, or native allocation/free.
- `src/shared/standard_tiled_copy.inl` and `src/shared/standard_tiled_add.inl`: preserve existing CUDA/ROCm copy and binary kernels and translate their metadata/launch work to the borrowed admission lease without changing numerical or view semantics.
- `src/cuda/copy.hpp` and `src/cuda/copy.cu`: implement the CUDA `gpu_policy`, queue construction, copy-operation translation, completion/error hooks, and native instrumentation seams against the shared contract.
- `src/rocm/copy.hpp` and `src/rocm/copy.hip`: implement the corresponding ROCm `gpu_policy`, queue construction, copy-operation translation, completion/error hooks, and native instrumentation seams.
- `src/cuda/device.cpp` and `src/rocm/device.cpp`: retain queue/device ownership, four-queue reservation, partition lifetime, setup rollback, and Device quarantine through destruction.
- `test/cuda/test_cuda_smoke.cpp` and `test/rocm/test_rocm_smoke.cpp`: focused backend resource/admission, fault, construction, destruction, and post-setup native-call instrumentation.
- `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp`: shared observable FIFO, resource, isolation, snapshot, reuse, and failure-history conformance.
- `test/backend/test_backend_coexistence.cpp`: configured multi-backend and cross-device coexistence isolation only; do not make CUDA/ROCm admission depend on a global registry or mutex.

## Requirements

1. Use the fixed resources provisioned during queue/device setup. Queue construction must eagerly create all C completion resources, fixed host metadata mirrors, and queue-native bookkeeping, and must reserve exactly one disjoint C-slot metadata partition. Publish the queue only after all setup succeeds; rollback every reservation and native/host resource on construction failure.
2. Atomically admit only the FIFO head to native execution. A head acquires exactly one credit and precreated event generation, plus at most one metadata slot. A parked node performs no native call, resource acquisition, output mutation, upload, kernel launch, or stream/event operation. Submission must not synchronously wait for a credit, event, metadata slot, or staging resource.
3. Treat no-op, inline, and no-metadata operations as ordinary FIFO nodes: they consume one credit, cannot bypass a parked head, and never consume a dummy metadata slot. Metadata-bearing operations use no more than one fixed slot.
4. Make retirement autonomous and one-at-a-time FIFO. Proven completion returns only resources whose stream/event/descriptor/upload/workspace/tensor-payload use is complete, preserves the immutable token outcome independently of reused event/descriptor storage, and immediately attempts the oldest parked head without a user wait or new submission.
5. Preserve post-acceptance failures and quarantine unknown use. Repeated waits on an accepted failure must return the same terminal failure after native resource reuse. If event/stream completion is unknown, quarantine the whole lease, reduce the queue's usable capacity, retain the host mirror and payload references, and do not release its metadata partition, queue reservation, or arena range until a covering proof. A later proof may reclaim resources but must not erase token history.
6. Close and destroy queues safely. Stop new acceptance, drain all accepted and parked nodes FIFO to terminal outcomes, transfer unresolved leases to Device-owned quarantine before releasing queue-local state, and retain the queue partition/count and referenced arena backing when proof is absent. A different queue's proof must not reclaim this queue's resources.
7. Keep resource and identity isolation. Four queues on one Device have independent C-sized metadata partitions and completion resources; creating a fifth live queue throws `std::bad_alloc` before starting another stream/worker. Different Device objects, ordinals, and backend policies have independent quotas and quarantine. Do not introduce a device-wide synchronization primitive, cross-device mutex, cross-queue credit/partition borrowing, resource growth/replacement, or global registry.
8. Keep allocator behavior explicit. After setup, CUDA/ROCm submission, dispatch, retirement, waits, and view transforms make zero IOM native allocation/free calls and never resize resource arrays. Operation-time native allocation/free and missing-workspace allocation are forbidden; only caller-requested setup/data-arena suballocation remains outside this admission task. Host-transfer allocation removal is verified by task 11.
9. Preserve owner/view and payload lifetime contracts. Capture immutable metadata/view snapshots per accepted request, retain registered tensor and workspace payload references through completion or quarantine, and prevent later submissions from overwriting a live descriptor, upload mirror, view mapping, output, or padding.
10. Keep failure paths transactional and repeatable. Construction faults, event creation/record faults, metadata/resource setup faults, host snapshot/node allocation faults, dispatch/launch faults, and queue destruction faults must leave no leaked or duplicated lease, slot, credit, partition, count reservation, registration, or arena range. Faults after token acceptance remain observable as retained token failures rather than synchronous negative OIDs.
11. Add meaningful focused regression coverage for every behavior owned here. Instrument actual CUDA/ROCm IOM native allocation/free and EventRing/metadata acquisition boundaries, and assert parked work has no resource acquisition or native effect. Cover saturation with C+2 prompt tokens, autonomous single-step FIFO dispatch, inline/no-op/no-metadata ordering, four disjoint queues, cross-queue/device isolation, snapshot immutability, proof-based reuse, retained failure/history after reuse, construction/destruction fault paths, quarantine, and zero post-setup IOM native memory calls.

## Non-goals

- Do not change CUDA/ROCm kernel arithmetic, numerical results, copy layout, padding preservation, alias behavior, broadcasting, view addressing, or supported operation behavior.
- Do not add public operation or direction enums, duplicate query APIs, alternate admission modes, compatibility shims, or per-queue mutable configuration.
- Do not add device-wide synchronization, a cross-device mutex, a global registry, queue-resource borrowing, resource growth/replacement, lazy event/metadata allocation, or operation-time native allocation/free.
- Do not redesign the data/metadata arena, raw workspace API, factory signatures, or common FIFO contract owned by the blocked tasks; use those planned contracts at their existing boundaries.
- Do not implement SYCL, CPU, or TTNN admission in this task except where coexistence compilation observes their existing independent behavior.
- Do not claim that SDK-internal allocations are eliminated; the zero-call criterion applies to IOM native allocation/free boundaries after setup.
- Do not add benchmarks or performance claims, and do not make hardware-dependent multi-GPU evidence a universal CI requirement.

## Acceptance criteria

- With capacity C saturated, C+2 or more additional valid CUDA and ROCm operations promptly return positive tokens. Instrumentation proves parked nodes acquire no credit, event, metadata, upload, workspace, or tensor-payload resource and cause no stream call, launch, output mutation, or other native effect.
- Releasing one proven completion autonomously dispatches exactly the oldest parked node, and repeated one-credit releases observe strict FIFO, including no-op, inline, and no-metadata nodes. No explicit wait or new submission is needed to make progress.
- Four simultaneously live queues each use a disjoint C-slot metadata partition and independent completion resources; creating a fifth queue fails before native queue start. Queue construction rollback, safe drained partition reuse, and unknown-completion partition retention are all observable.
- Operations on separate queues and separate Device objects remain isolated in credits, partitions, streams, outcomes, workspace leases, quarantines, queue counts, and arena release. One queue/device cannot prove or reclaim another's unresolved lease.
- Different views of one owner and later submissions retain immutable metadata/view snapshots, descriptor bytes, output state, padding, workspace ranges, and tensor payload references through parking and completion. Reuse after a covering proof is safe and does not alter earlier token results.
- Injected post-acceptance launch/event/stream and completion-observation failures remain repeatable terminal failures; unknown completion quarantines the complete lease, reduces capacity, and blocks partition/arena release until proof. Queue destruction drains accepted work FIFO and preserves token history even when unresolved leases survive destruction.
- Construction and destruction faults leave transactional state and no unsafe release. After successful device/queue setup, instrumented CUDA/ROCm execution reports zero IOM native memory allocation/free calls across first and repeated submission, parking, dispatch, retirement, waits, and view transforms.
- Existing CUDA and ROCm copy/binary conformance remains numerically and behaviorally unchanged apart from the bounded admission/resource contract; host-transfer conformance remains owned by task 11.

## Verification

Proposed gates (not run by this task):

- Through the `remote-development` workflow, run `ctest --test-dir <cuda-build> --output-on-failure -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests)$'` on configured CUDA hardware. The smoke target must exercise native allocation/event/metadata instrumentation, setup and teardown faults, saturation, parking, autonomous FIFO dispatch, quarantine, and zero post-setup IOM native memory calls; the conformance target must exercise the shared observable contract.
- Through the `remote-development` workflow, run `ctest --test-dir <rocm-build> --output-on-failure -R '^(iom_rocm_smoke_tests|iom_rocm_conformance_tests)$'` on configured ROCm hardware with the same focused coverage and enabled-hardware failure behavior.
- When the coexistence configuration enables CUDA and/or ROCm (and the project creates the target), through the `remote-development` workflow run `ctest --test-dir <build> --output-on-failure -R '^iom_backend_coexistence_tests$'` to verify cross-backend/device isolation. Do not require this target when it is not configured.

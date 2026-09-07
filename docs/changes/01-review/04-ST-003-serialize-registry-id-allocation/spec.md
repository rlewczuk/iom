# Per-device registry queue/entry IDs race across concurrent queues

**Order:** 04
**Priority:** P0 — race/UB
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** ST-003
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 93
**Review scope:** whole-codebase
**Backend scope:** cuda, rocm, sycl, ttnn
**Location:** `include/iom/detail/outstanding_work_registry.hpp:579-649` (`RegistryState`, `allocate_queue_id`, `register_copy_entries`); queue constructors `src/cuda/copy.cu:108-117`, `src/rocm/copy.hip:107-116`, `src/sycl/copy.cpp:410-420`, `src/ttnn/device.cpp:392-400`

## Outcome

Mutable registry ID allocation is serialized per `RegistryState`. Every queue on one `Device` receives a unique registry queue ID, every source/destination entry pair receives a unique reserved ID pair, and destroying one queue invalidates only its own entries — under concurrent queue creation and submission from independent caller threads.

## Current problem

The four accelerator backends share a per-device registry whose ID counters are read/written without synchronization outside the registry map's internal mutex. Both `execute` paths read/write the shared plain `RegistryState::next_entry_id` without a lock (`register_copy_entries` → `register_registry_entries`, which reads, computes a pair, and mutates directly at `include/iom/detail/outstanding_work_registry.hpp:613-649`), producing duplicate or skipped source/destination IDs under concurrency; one registration can throw "already registered", or later cleanup can address the wrong entry. Concurrent `Device::create_ops()` calls similarly race `next_queue_id` (`allocate_queue_id` mutates `state.next_queue_id` directly at `:640-642`), producing duplicate `registry_queue_id_` values. If two queues receive the same registry queue ID, destroying one queue's `invalidate_entries_for_queue` invalidates the other queue's still-live entries, causing unrelated tensors to quarantine or later completion to lose ownership. `RegistryState` contains plain `EntryId next_entry_id` and `QueueId next_queue_id` (`:579-584`) and no counter mutex; the map mutex protects only map operations. All asynchronous queue constructors call `allocate_queue_id` and all queue execute paths call `register_copy_entries` (`src/cuda/copy.cu:108-117,274-278`; `src/rocm/copy.hip:107-116,273-277`; `src/sycl/copy.cpp:410-420,497-501`; `src/ttnn/device.cpp:392-400,459-464`). The public queue contract serializes calls only per individual queue ("Calls on one queue are serialized by the caller", `include/iom/iom.hpp:205-210`), leaving independent queues as the concurrency boundary; `DeviceOps::submit` explicitly documents concurrent submit reservations (`include/iom/iom.hpp:310-318`). Impact: undefined behavior/data races under concurrent queue creation/submission, intermittent synchronous copy failures, duplicate registry ownership, cross-queue invalidation and premature quarantine, or leaks/double cleanup.

## Scope

- Add one explicit per-`RegistryState` allocation lock (or move ID allocation into a registry-state owner that atomically reserves queue IDs and source/destination ID pairs) covering exactly the shared ID counters.
- Hold the lock across pair reservation — not merely individual increments — because a source/destination reservation must remain one reservation; do not make the two IDs independently atomic.
- Route all accelerator queue constructors and copy registrations through the same protected path.

## Implementation references

- **Modify:** `include/iom/detail/outstanding_work_registry.hpp:579-649` — `detail::RegistryState` gains the allocation lock; `allocate_registry_queue_id`/`allocate_queue_id` and `register_registry_entries`/`register_copy_entries` use it, keeping the existing registry map mutex for map operations.
- **Modify:** `src/cuda/copy.cu:108-117,274-278` (`CudaQueue`), `src/rocm/copy.hip:107-116,273-277` (`RocmQueue`), `src/sycl/copy.cpp:410-420,497-501` (`SyclQueue`), `src/ttnn/device.cpp:392-400,459-464` (`TtnnQueue`) — callers need no per-backend locking, only the shared protected allocation path.
- **Read:** `include/iom/detail/outstanding_work_registry.hpp` — `OutstandingWorkRegistry::{register_entry,invalidate_entries_for_queue,remove_entry_if_present}` map-mutex conventions to reuse without duplicating.
- **Tests:** `test/backend/test_backend_coexistence.cpp:337-453,591-633`, `test/cuda/test_cuda_smoke.cpp:446-506` — existing multi-queue/coexistence patterns; add new concurrent same-device queue tests.

## Requirements

- Queue IDs are unique per `Device` and source/destination ID pairs are unique per registration even when multiple queues submit concurrently from different threads.
- The reservation of the source/destination pair must be atomic as a pair — one lock covering the pair computation, not two independent increments.
- Keep the registry map mutex for map operations; do not widen it into a global lock over copy execution or native submission.
- Preserve separate `RegistryState` instances per `Device` and the process-global public token queue-ID pool; they represent different identities and must not be conflated.
- Destroying one queue invalidates only its own entries; existing sequential coexistence and queue-ID reuse behavior remain unchanged.

## Non-goals

- Changing process-global public `DeviceOps` token IDs, token generation/reuse policy, backend selection, runtime stream scheduling, or registry map indexing.
- Serializing unrelated backend execution or all native copies — only shared ID state needs protection.

## Acceptance criteria

- [ ] A barrier-based test concurrently creates multiple queues on one `Device` and submits multi-plane copies from each queue: every queue has a distinct registry identity, every operation obtains one unique source/destination pair, all operations complete, and destroying one queue invalidates only its own entries.
- [ ] No operation fails solely due to an ID allocation race; each originating queue can wait its own token, allocator/quarantine counts remain correct, and no duplicate/skipped entry ID is observed.
- [ ] Existing sequential coexistence and queue-ID reuse tests pass unchanged; under ThreadSanitizer (where the backend/toolchain supports it) the registry counter paths show no data race.

## Verification

Actual validation: none (read-only review); proposed gates below.

Proposed gates (accelerator hardware on remote hosts per remote-development, exercising CUDA, ROCm, SYCL, and TTNN separately when runtimes are available; sync the workspace, then run via `remote-exec`):

- Add and run the barrier-based concurrent queue-creation/submission test described under Acceptance criteria, with an internal testing accessor or observable allocator/quarantine assertions for registry queue-ID uniqueness.
- Run the scenario under ThreadSanitizer where the backend/toolchain supports it.
- Run the existing backend coexistence and CUDA smoke suites as regression coverage: `ctest --test-dir <build> -R '^(iom_backend_coexistence_tests|iom_cuda_smoke_tests)$' --output-on-failure`.

- `ctest --test-dir <build> -R '^(iom_backend_coexistence_tests|iom_cuda_smoke_tests)$' --output-on-failure`
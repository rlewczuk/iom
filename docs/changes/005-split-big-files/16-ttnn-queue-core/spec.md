# Split TTNN queue core

**Order:** 16
**Priority:** P1 — moves queue lifecycle/dispatch after private device and testing linkage exist
**Blocked by:** `07-ttnn-device-types`, `08-ttnn-testing`
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Extract the TTNN queue core from `src/ttnn/device.cpp` into one private declaration header and one queue implementation translation unit. The extraction must preserve the existing asynchronous submission, native completion, failure, registry, staging, and workspace-lease behavior exactly. `TtnnDevice::create_ops` moves with the queue. Copy execution remains temporarily in `src/ttnn/device.cpp` as the sole out-of-class `TtnnQueue::execute_copy(Task&)` definition until task 17 moves it to `src/ttnn/queue_copy.cpp`.

## Scope

Create only the following planned production files as part of this factoring task:

- `src/ttnn/queue_internal.hpp`: private declarations for `TtnnQueue`, its nested `Task` and `BinaryOutcome` state, and the TTNN fence helpers/declarations needed by the queue implementation. `Task` and `BinaryOutcome` must retain their `detail::WorkspaceLease` members.
- `src/ttnn/queue.cpp`: definitions for the queue lifecycle, submission, binary dispatch, completion, fence helpers, fence-through-sequence handling, and `TtnnDevice::create_ops`.

Update the TTNN library source list in `CMakeLists.txt` to include `src/ttnn/queue.cpp` and the private queue declaration header according to the existing TTNN target conventions. Retain all existing TTNN sources and headers; do not change test target source lists or public factory headers.

Move the queue-core responsibility represented by these current `src/ttnn/device.cpp` baseline spans:

- `src/ttnn/device.cpp:465-751`: TTNN fence capture/invocation/build and native finish helpers, `TtnnQueue` declaration and lifecycle/submission/binary dispatch portions through `publish_native_completion`.
- `src/ttnn/device.cpp:876-1046`: the remainder of queue completion, queue state, and `fence_through_sequence`.
- `src/ttnn/device.cpp:1130-1132`: the assigned queue/factory portion containing `TtnnDevice::create_ops` (symbol ownership controls if line numbers shift).

The exact symbol responsibilities, rather than stale physical line positions, control the cutover. The moved class declaration must be available through `queue_internal.hpp`, and `src/ttnn/device.cpp` must include that private header so its temporary `execute_copy` member definition remains well-formed.

Do not move `TtnnQueue::execute_copy(Task&)` in this task. Keep its current implementation as an out-of-class definition in `src/ttnn/device.cpp` until task 17, with its declaration in `queue_internal.hpp` and exactly one definition across the TTNN target. `queue.cpp` must not contain a duplicate or forwarding replacement.

Do not move `ttnn_detail::binary_planes`, its explicit operation instantiations, transfer/layout helpers, `region_from_host`, `region_to_host`, `region_from_host`/`region_to_host` support, or `copy_planes` from `src/ttnn/copy.cpp`. Those remain owned by the later TTNN binary/copy tasks. Do not move the TTNN testing API or fault-consumer definitions; use the linkage established by `08-ttnn-testing`.

## Implementation references

Use the current implementation in `src/ttnn/device.cpp` as the behavior reference:

- `TtnnFenceCapture`, its storage/alignment/triviality assertions, `ttnn_fence_invoke`, and `build_ttnn_fence` are the private fence-capture path.
- `finish_locked_mesh`, `finish_locked`, and `finish_native` preserve the API-mutex and native mesh-finish seams. Their existing distinction between copy-drain finish attempts and other fence/quarantine finish calls must remain intact.
- `TtnnQueue::Task` carries the sequence, borrowed source/destination views, no-op marker, binary operation/request snapshot, binary entry registration, workspace lease, and entry IDs.
- `TtnnQueue::BinaryOutcome` carries binary entry registration, workspace lease, native-submission/completion-proof flags, and retained failure.
- `TtnnQueue` lifecycle is its constructor's `detail::StagedWorker<Task>` callback wiring and `CompleteOnThrow` policy, plus the destructor's queue-entry invalidation and `shutdown_and_drain`.
- `TtnnQueue::copy_impl` preserves submission-order locking, identical-window detection, `submit`, and worker copy submission of a `Task` holding the existing borrowed view references. Its execution call remains the temporary `execute_copy` definition in `device.cpp`.
- `TtnnQueue::binary_impl` preserves submission-order locking, fence construction, `submit_binary`, the captured `BinaryRequest` snapshot, entry registration, workspace lease transfer, and worker submission.
- The binary branch of `TtnnQueue::execute` preserves outcome insertion before native work, API-mutex-guarded per-plane dispatch through `ttnn_detail::binary_planes`, operation selection, retained-failure handling, and execution-sequence publication.
- `publish_native_completion`, `complete_task`, and `fence_through_sequence` preserve contiguous completion bookkeeping, native finish retry, registry release/invalidation, workspace-lease completion, token completion, and post-completion failure retention.
- `TtnnDevice::create_ops` continues to return `std::make_unique<TtnnQueue>(*this)` and remains the only TTNN queue factory definition.

The private header must use the existing namespace, include, forward-declaration, and visibility idioms established by `src/ttnn/device_internal.hpp` and the TTNN registry/staging headers after blockers 07 and 08 land. It must not expose queue internals through `include/iom/ttnn/device.hpp` or any other public header.

## Requirements

1. Preserve the submit → `Task` → outcome ownership transfer. A submitted copy task keeps the existing borrowed `TensorView` references for exactly the same lifetime and ordering contract. A binary task keeps the captured request snapshot, `detail::BinaryEntryRegistration`, and `detail::WorkspaceLease` until completion proves native work finished or the existing quarantine/invalidation path takes over.
2. Preserve the existing `DeviceOps` override signatures and all private queue state: sequence/outcome maps, registry queue ID and state pointer, submission-order mutex, outcome mutex, worker, fence mutex, last-finished sequence, and executed sequence. Do not alter atomic memory-ordering or lock ordering.
3. Preserve queue lifecycle behavior. Construct and start `detail::StagedWorker<Task>` with the existing callbacks and `CompleteOnThrow` policy. Destruction must invalidate this queue's registry entries before `shutdown_and_drain`, drain every published task, retry retained native completion as before, and leave no queue-owned work or lease unaccounted for.
4. Preserve copy submission behavior without moving copy execution. `copy_impl` must retain the submission-order mutex, `identical_window` no-op detection, sequence allocation through `submit`, and worker submission. `execute(Task&)` must still route non-binary tasks to the one temporary out-of-class `execute_copy` definition in `src/ttnn/device.cpp`.
5. Preserve binary submission and snapshot behavior. `binary_impl` must build the queue fence, call the existing `submit_binary` transaction, copy the public request into the internal TTNN request representation, and transfer entries and `workspace_lease` into the worker `Task` and then `BinaryOutcome` without additional allocations or alternate ownership.
6. Preserve binary dispatch ordering and failure semantics. Insert the provisional binary outcome before native submission; hold `TtnnDevice::api_mutex()` while dispatching `ttnn_detail::binary_planes`; retain the existing operation switch for Add/Mul/Sub/Div; distinguish failure before any plane is submitted from failure after native work is accepted; publish execution/completion markers exactly as before; and make retained failures repeatable through every wait.
7. Preserve native finish batching. Fence helpers must retain their noexcept failure conversion, API mutex protection, and native queue selection. `complete_task` must perform one native finish for a contiguous batch containing native work, no finish for a no-op-only batch, and the existing finish retry for a submitted binary operation without proven completion. Do not introduce per-plane or per-task finish calls, new synchronization, or a reordered lock acquisition.
8. Preserve completion proof and release order. Registry entries may be released normally only after native completion is proven; otherwise retain/invalidate through the existing failure/quarantine path. Complete each workspace lease only with the existing completion-proof result, and deliver the same precedence among worker failure, retained submission failure, and native finish failure.
9. Preserve `fence_through_sequence` behavior: serialize through `fence_mutex_`, return for already-finished sequences, call the existing native finish path for an outstanding sequence, advance `last_finished_seq_` only on a successful finish, and record a post-completion failure through the existing mechanism otherwise. Waits and failures must remain repeatable.
10. Preserve private visibility and ODR. Every moved definition must occur exactly once; queue-only declarations stay private; no public header, signature, capability, error category/message contract, validation order, allocator seam, registry/quarantine semantic, or include-path contract may change.
11. Keep the extracted files responsibility-complete and within the line budgets: `src/ttnn/queue.cpp` at most 440 physical lines and `src/ttnn/queue_internal.hpp` at most 499 physical lines. With blockers 07 and 08 complete and all assigned queue-core definitions removed, every production file touched by this task, including the temporary `src/ttnn/device.cpp`, must be at most 499 physical lines after normal formatting. Task 17 must preserve that cap when it moves `execute_copy`.
12. Update only TTNN build integration needed for this extraction. The TTNN target must compile/link the new queue implementation, retain `device.cpp` for the temporary copy definition and device/tensor/workspace responsibilities, and retain `copy.cpp` for host transfer and copy/binary support. Do not add a queue-fence translation unit, factory-only translation unit, backend switch, global registry, compatibility alias, or shim.

## Non-goals

- Do not implement or alter production behavior beyond relocating definitions.
- Do not move `TtnnQueue::execute_copy` to `queue.cpp` or `queue_copy.cpp` yet; task 17 owns that cutover.
- Do not move `binary_planes`, host-transfer/layout helpers, `region_from_host`, `region_to_host`, or `copy_planes`; in particular, no `binary_planes` or host-transfer move is part of queue core.
- Do not move TTNN dtype/device-type code, testing seams/API definitions, `TtnnTensor`, `TtnnWorkspace`, device creation, or the public factory header; those belong to their specified tasks or remain in `device.cpp`.
- Do not change public headers, public signatures, test source lists, test expectations, error behavior, validation order, allocation behavior, queue synchronization design, native finish policy, or workspace/registry ownership contracts.
- Do not add tests, a permanent line-count test, compatibility aliases/shims, cross-backend abstractions, extra capabilities, or speculative helper layers.

## Acceptance criteria

- `src/ttnn/queue_internal.hpp` exists as a private queue linkage header and declares the complete queue state needed by out-of-class definitions, including `Task`/`BinaryOutcome` `WorkspaceLease` members and fence declarations, without changing public TTNN headers.
- `src/ttnn/queue.cpp` contains the queue-core and fence definitions listed in Scope/Implementation references, plus the sole moved `TtnnDevice::create_ops` definition; it does not define `TtnnQueue::execute_copy`.
- `src/ttnn/device.cpp` retains exactly one out-of-class `void TtnnQueue::execute_copy(Task&)` definition at its temporary location until task 17, and the TTNN target has no duplicate queue or factory definitions.
- TTNN copy/host-transfer ownership remains unchanged: `binary_planes` and its explicit instantiations remain in their designated copy/binary path, and no host-transfer helper is relocated by this task.
- The TTNN CMake target includes `src/ttnn/queue.cpp` and the private queue header while retaining required existing TTNN sources and headers; no test source list or public factory header changes.
- The queue and private header meet the 440/499 line budgets, and every production file touched or created by this task, including the temporary `src/ttnn/device.cpp`, is at most 499 physical lines after normal formatting.
- Existing TTNN behavior is unchanged: submission order, borrowed-view lifetime, binary snapshots and lease/entry ownership, API mutex use, one-finish contiguous batching, repeatable waits/failures, completion-proof release, destructor invalidation, and drain behavior all remain intact.

## Verification

The future implementer must run these checks after tasks 07 and 08 are integrated; they are verification instructions, not gates for this specification-writing task. Task 16 must compile and satisfy its line caps with the temporary `execute_copy` definition still in `device.cpp`; rerun the same gates after task 17 completes the copy cutover.

1. On the configured TTNN remote Linux host, configure or reuse the TTNN build with testing enabled and build the TTNN library and its smoke, conformance, and coexistence targets:

   ```sh
   cmake --build <ttnn-build> --target iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests
   ```

2. Run the TTNN tests without converting hardware failures into skips:

   ```sh
   ctest --test-dir <ttnn-build> --output-on-failure -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$'
   ```

3. Use a deterministic one-time scanner (not a committed test) over every production file touched or created by this task. Assert at most 499 physical lines immediately after task 16, specifically including the temporary `src/ttnn/device.cpp` and `src/ttnn/queue_internal.hpp`, and assert `src/ttnn/queue.cpp <=440`. Rerun the final tracked-production scan after task 17 moves `execute_copy`.

4. Confirm by compile/link and symbol inspection that each moved queue/fence/create-ops definition exists exactly once, `TtnnQueue::execute_copy(Task&)` exists exactly once in `src/ttnn/device.cpp` before task 17, private queue declarations remain private, and the TTNN public headers and factory interfaces are unchanged.

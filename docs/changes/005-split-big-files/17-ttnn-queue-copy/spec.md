# Extract TTNN queue copy execution

**Order:** 17
**Priority:** P1 — completes the TTNN queue responsibility split
**Blocked by:** `16-ttnn-queue-core`
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Extract only `TtnnQueue::execute_copy` from `src/ttnn/device.cpp` into the planned `src/ttnn/queue_copy.cpp`, and add that source to the `iom_ttnn` CMake target. The queue core left by Order 16 remains the owner of queue lifecycle, dispatch, completion, fence helpers, and `TtnnDevice::create_ops`; this task changes only the translation-unit placement of copy execution.

The resulting copy path must be behavior-identical. It must retain TTNN native per-plane submission, ownership protection, failure retention, and completion batching while leaving all public headers and signatures unchanged.

## Scope

Implementation may:

- create `src/ttnn/queue_copy.cpp` as the single new implementation file for `TtnnQueue::execute_copy`;
- remove the old `TtnnQueue::execute_copy` definition from `src/ttnn/device.cpp`, exactly once;
- consume the private `src/ttnn/queue_internal.hpp` declarations produced by Order 16 and the existing `src/ttnn/copy.hpp` interface;
- add `src/ttnn/queue_copy.cpp` to the `iom_ttnn` source list in `CMakeLists.txt` while retaining all existing TTNN sources and private support entries.

This task does not move any other `TtnnQueue` member. `src/ttnn/device.cpp` continues to contain the device, tensor, workspace, and factory responsibilities assigned by the parent specification. `src/ttnn/copy.cpp` continues to own transfer/layout helpers and `ttnn_detail::copy_planes`.

## Implementation references

- `TtnnQueue::execute_copy`: `src/ttnn/device.cpp:752-875` (the complete definition to relocate).
- `TtnnQueue::execute`: the queue-core dispatch that invokes `execute_copy` for non-binary tasks; preserve that call and its binary path unchanged.
- `TtnnQueue::Task`: copy task fields `sequence`, `source`, `destination`, `no_op`, and entry IDs must remain available through `queue_internal.hpp`.
- Queue state used by the moved definition: `state_`, `registry_queue_id_`, `outcomes_`, `outcome_mutex_`, `device_`, `executed_seq_`, and the queue/device API-lock and completion helpers declared by the private queue interface.
- `src/ttnn/copy.hpp:11-16`: `ttnn_detail::copy_planes`, whose `any_submitted` result distinguishes pre-native failure from partial submission.
- `src/ttnn/copy.cpp:404-418`: per-plane native copy loop and its testing fault seam; do not duplicate or relocate it.
- `include/iom/detail/outstanding_work_registry.hpp:715-717`: `detail::register_copy_entries`; source and destination registration must precede native work.
- `CMakeLists.txt:121-128`: `iom_ttnn` source list, which must gain `src/ttnn/queue_copy.cpp`.
- Existing conformance coverage: `test/ttnn/test_ttnn_conformance.cpp` cases `TTNN failed plane submissions drain before rethrow` (around line 1023), `TTNN registration and outcome insertion failures roll back ownership` (around line 1065), `TTNN identical-window copies complete without native finish` (around line 1308), `TTNN mixed no-op and native batches finish native work once` (around line 1323), and `TTNN no-wait bursts share one native finish per ready batch` (around line 1344). The shared identical-window copy matrix is in `test/backend/backend_conformance_copy_storage.hpp`.

## Requirements

1. Define `TtnnQueue::execute_copy` once, out of line in `src/ttnn/queue_copy.cpp`, with the same namespace, signature, access, includes, and private-symbol visibility as the original. Include `queue_internal.hpp` and reuse `copy.hpp`; do not introduce a public declaration or a second queue abstraction.
2. Build the TTNN fence and register source and destination entries before any native plane reaches the mesh. Insert the copy `SequenceOutcome` only after registration. If registration or outcome insertion fails before native submission, remove every entry that was created, discard the provisional outcome, and propagate the original failure so no work remains pending.
3. Preserve identical-window behavior: a `Task` marked `no_op` registers the normal ownership/completion record, submits no native plane and performs no device-wide finish, publishes its executed sequence, and lets normal completion release the entries. Waits remain valid and repeatable.
4. For a non-no-op task, hold the device API mutex while converting native handles and invoking `ttnn_detail::copy_planes`. Preserve the per-plane order and pass through its `any_submitted` result. Do not add allocation, staging, host transfer, validation, or synchronization behavior.
5. On a copy-plane failure before any plane is submitted, synchronously roll back source and destination registry entries and erase the copy outcome, then rethrow the submission failure. On a failure after one or more planes were submitted, mark native work as submitted and synchronously call the existing locked mesh-finish helper before ownership is removed.
6. If that partial-submission drain succeeds, remove the source/destination entries and outcome and rethrow the original plane-submission failure. If the drain itself fails, retain the registered entries and outcome, retain the native failure/token state, store the submission failure for the normal completion path, publish the executed sequence, and return without unsafe rollback. The retained token must continue to report the same failure on repeated waits, while completion retry/quarantine eventually proves cleanup safe.
7. On complete plane submission, mark native work submitted, publish `executed_seq_`, and return to the existing `complete_task` path. Do not call an extra finish: completion must preserve the existing one native finish per ready contiguous batch, submission order, registry release, and repeatable-wait behavior for mixed no-op/native and no-wait bursts.
8. Keep all existing `IOM_ENABLE_TESTING` seams observable through their current declarations and counters. Do not move or duplicate test-seam state, `finish_locked`, `finish_locked_mesh`, `complete_task`, or any fence/lifecycle implementation.
9. Add `src/ttnn/queue_copy.cpp` to `iom_ttnn` in `CMakeLists.txt`. Do not alter TTNN test source lists, public factory headers, backend switches, or unrelated target integration.
10. Keep `src/ttnn/queue_copy.cpp` approximately 180 physical lines and keep every touched or new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/` at most 499 physical lines after normal formatting. Use no `queue_fence.cpp` and no factory-only file.

## Non-goals

- No queue lifecycle, worker, dispatch, completion, fence, or `create_ops` redesign; those belong to Order 16 and remain in the queue core.
- No host-transfer or staging changes, including `region_from_host`, `region_to_host`, or retained staging ownership.
- No binary implementation or `binary_planes` changes; no changes to `src/ttnn/binary.cpp` scope.
- No changes to `src/ttnn/copy.cpp`, `src/ttnn/copy.hpp`, public TTNN headers, tensor/device creation, validation order, or capabilities.
- No new tests, test source-list changes, compatibility aliases/shims, backend switches, global registries, cross-backend abstractions, allocation changes, or synchronization redesign.
- No changes to registry/quarantine semantics, workspace/staging leases, native per-plane layout, public signatures, error categories, numerics, ownership/lifetimes, asynchronous in-order behavior, or ODR; this is source factoring only.

## Acceptance criteria

- `TtnnQueue::execute_copy` has exactly one definition, in `src/ttnn/queue_copy.cpp`; the old definition is absent from `src/ttnn/device.cpp` and no compatibility translation unit exists.
- The `iom_ttnn` target compiles and links with the new source while retaining `device.cpp`, `copy.cpp`, existing private headers, and public headers.
- Registration precedes every native plane submission; identical-window copies perform no native finish; per-plane copies preserve order; pre-native failures roll back; partial failures drain synchronously; drain failures retain entries/outcomes and repeatable token failure; successful drains roll back only after completion proof.
- Existing completion batching remains unchanged: no-op/native mixtures and no-wait bursts finish each ready native batch once and release ownership only through the established completion path.
- No queue lifecycle, host transfer, or binary implementation changes are present, and public APIs/capabilities, errors/validation order, ownership/lifetimes, numerics, and private visibility/ODR remain unchanged.
- `src/ttnn/queue_copy.cpp` is about 180 lines and every touched/new production source under `src/` or `include/` is at most 499 physical lines.

## Verification

The future implementer or integrator must run these checks; they are instructions, not gates for drafting this mini-spec:

1. On the configured TTNN remote-development host, configure or reuse the TTNN build and build the backend plus its smoke, conformance, and coexistence targets, for example:
   ```sh
   cmake --build <ttnn-build> --target iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests
   ```
2. On that same remote TTNN build, run:
   ```sh
   ctest --test-dir <ttnn-build> --output-on-failure \
     -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$'
   ```
   Hardware failures are failures; do not skip or convert them to passes. The TTNN conformance cases named above must cover partial-plane drain, rollback, identical-window no-op, retained failure, and completion batching.
3. Confirm the moved symbol occurs exactly once and the old translation unit no longer defines it:
   ```sh
   git grep -n 'TtnnQueue::execute_copy' -- src/ttnn
   ```
4. Run a deterministic one-time line scanner over `src/ttnn/device.cpp`, `src/ttnn/queue.cpp`, `src/ttnn/queue_internal.hpp`, `src/ttnn/queue_copy.cpp`, and any other production file touched by tasks 16–17. Assert each is at most 499 physical lines and report the approximate `queue_copy.cpp` count. Task 21 performs the final repository-wide scan; this scoped scanner is verification only and must not become a permanent test.

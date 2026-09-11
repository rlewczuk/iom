# Complete core DeviceOps cutover

**Order:** 05
**Priority:** P0 — removes the obsolete oversized translation unit and completes the core migration
**Blocked by:** `01-core-tensor-view`, `02-core-workspace`, `03-core-device-ops-binary`, `04-core-device-ops-copy-neural`
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

The backend-neutral DeviceOps implementation is completely cut over from `src/iom.cpp` to the planned `src/device_ops.cpp`. The new translation unit owns exactly the shared queue, validation, failure, token, wait, fence, completion, and sequence machinery; operation-family definitions remain in their responsibility-complete core shards. `src/iom.cpp` is removed rather than retained as a compatibility translation unit, and CMake links all seven planned core source files. Public headers, signatures, capabilities, error categories, validation order, ownership, lifetimes, asynchronous ordering, repeatable waits, workspace lease proof, and ODR behavior remain unchanged.

## Scope

- Extract the remaining shared DeviceOps definitions from `src/iom.cpp` into `src/device_ops.cpp`: the process-global 255-ID allocator state; queue-ID lease and release; both `DeviceOps` constructors and destructor; `queue_device()` and common `validate_views()` checks; the shared `invoke()`, `invoke_failure()`, and `map_failure()` path; `detail::WorkspaceValidation::validated()` and `address()`; token encoding; `wait()` and the default fence hook; completion and post-completion failure retention; and `complete()`, `commit_failure()`, and `seek_next_sequence()` bookkeeping.
- Make the one backend-neutral `UnsupportedOperation` type available through the planned private `src/iom_internal.hpp` linkage as needed by the core shards. `src/device_ops.cpp` supplies the single mapping of that type to `OidError::Unsupported`; no public declaration, duplicate local exception type, or second mapping is permitted. Keep `src/iom_internal.hpp` minimal and private: only the already-planned shared checked arithmetic, rank constants, and `UnsupportedOperation` linkage may live there.
- Keep operation-specific definitions in their planned destinations without omission: copy validation/identical-window logic and the `copy` facade in `src/device_ops_copy.cpp`; binary validation, snapshots, default binary hook, all four `add`/`mul`/`sub`/`div` facades, and their requirement queries in `src/device_ops_binary.cpp`; and default neural hooks plus `silu`/`linear`/`rmsnorm`/`sdpa` facades in `src/device_ops_neural.cpp`. This task coordinates the common linkage but must not duplicate those definitions in `src/device_ops.cpp`.
- Preserve the existing protected `DeviceOps` interface and the `submit`/`submit_binary` admission protocol in `include/iom/iom.hpp`. In particular, captured binary requests remain value snapshots; workspace acquisition precedes binary-owner registration and queue work; the existing catch path removes registrations and completes the workspace lease in its established order; and successful or failed asynchronous work releases a workspace lease only after a covering completion proof, otherwise retaining/quarantining it.
- Update `IOM_SOURCES` in `CMakeLists.txt` by replacing `src/iom.cpp` with exactly `src/tensor.cpp`, `src/tensor_view.cpp`, `src/workspace.cpp`, `src/device_ops.cpp`, `src/device_ops_copy.cpp`, `src/device_ops_binary.cpp`, and `src/device_ops_neural.cpp`. Do not add a compatibility TU or change test source lists/public factory headers.

## Implementation references

- `src/iom.cpp:649-656` — the global queue-ID mutex/bitset and `UnsupportedOperation` definition to relocate behind private core linkage.
- `src/iom.cpp:983-1012` — `DeviceOps::unsupported`, both constructors, `queue_device()`, and common view/device validation. Copy-specific validation at the neighboring source block belongs to `device_ops_copy.cpp`, not this shard.
- `src/iom.cpp:1038-1065` — `map_failure`, `invoke_failure`, and `invoke` and their exact OID error mapping.
- `src/iom.cpp:1212-1275` — `detail::WorkspaceValidation::validated`, including capacity, owner/device, alignment, checked range-end, overlap, and `address()` behavior.
- `src/iom.cpp:1328-1482` — destructor and queue-ID release, 55-bit token encoding, `wait`, fence-through-sequence hook, post-completion failure retention, completion, retained-failure commit, and skipped-sequence bookkeeping.
- `src/iom_internal.hpp` — planned private backend-neutral linkage; keep it narrow and do not expose it through installed headers.
- `include/iom/iom.hpp:244-552` — public `DeviceOps`, `detail::WorkspaceValidation`, protected snapshots/submission templates, completion hooks, and private state whose declarations and signatures must remain unchanged.
- `CMakeLists.txt:74-110` — `IOM_SOURCES` and `IOM_HEADERS`; only the seven planned core `.cpp` files replace `src/iom.cpp` in the source list, while private core headers need not be public headers.
- `test/test_iom.cpp:2005-2050,2416-2829,3567-3988` — public signature, OID mapping, queue-ID/token, submit rollback, wait/failure retention, workspace validation, and workspace-lease behavior exercised by the existing core tests.

## Requirements

1. **One definition per symbol.** Define every moved shared symbol exactly once in `src/device_ops.cpp`, and remove its old definition with `src/iom.cpp`. The resulting core files must link without missing or duplicate `DeviceOps`, workspace-validation, queue-ID, token, wait, completion, or failure symbols. Do not put operation-family definitions into the shared shard merely to make the old file disappear.

2. **Queue-ID ownership.** Preserve one process-global `std::mutex` and 255-bit live-ID state. `lease_queue_id()` allocates IDs 1 through 255 under that mutex and reports the established exhaustion error when all IDs are live. Both constructors lease exactly one ID, and the destructor calls the noexcept release path exactly once for its own ID. IDs do not select a backend, device, runtime context, or cross-backend registry; a released ID is reusable with the existing ordering.

3. **Common queue/device/view validation.** Preserve `queue_device()` throwing `std::logic_error` when a default-constructed queue has no device. Preserve `validate_views()` rejecting null views and views belonging to another exact `Device` with `std::invalid_argument`, before any sequence reservation, token, registry entry, metadata, lease, output mutation, or backend effect. Do not weaken or duplicate operation-specific validation owned by the copy, binary, and neural shards.

4. **Failure mapping and facade boundary.** Keep `invoke(result)` as the common no-throw result path and `invoke_failure(exception_ptr)` as the common no-throw exception path. `map_failure()` must preserve the established categories: `UnsupportedOperation` to `OidError::Unsupported`/OID -2, `std::bad_alloc` to `ResourceExhausted`/-4, `std::overflow_error` to `Overflow`/-3, `std::invalid_argument` to `InvalidArgument`/-1, other `std::exception` to `DeviceError`/-5, and unknown or empty failures to `InternalError`/-6. No synchronous exception may cross an OID-returning `noexcept` facade, and no operation shard may introduce a competing mapping.

5. **Workspace validation.** Keep `WorkspaceValidation::validated()` pure and allocation-free. A zero-byte requirement returns the unchanged view without imposing owner/device/range effects. A positive requirement must reject an empty view, foreign `Device`, dead/foreign owner, insufficient capacity, a base not aligned to 32 bytes, insufficient required alignment, checked range-end overflow, or overlap with any operand/output storage using the established exception categories and validation order. `address()` returns the checked `range_address()` and is noexcept. Validation must not register owners, acquire a lease, reserve a token, touch backend state, or mutate an operand/output.

6. **Capture, lease, rollback, and completion order.** Preserve the existing `submit` and `submit_binary` semantics declared in `include/iom/iom.hpp`: reserve the sequence before queue work; capture the validated binary request by value; acquire a positive workspace lease before registering binary owners; submit the captured request and registrations; on synchronous failure remove registrations before completing/releasing the workspace lease through the existing proof path; and let the outer sequence rollback restore the sequence only under its existing conditions. On accepted asynchronous success or failure, retain all owner and workspace protection through the completion proof. Unknown completion keeps the lease retained/quarantined, and a later covering proof may reclaim it without changing the token outcome. Do not reorder these operations or replace them with a new synchronization protocol.

7. **Token encoding and sequence limits.** Preserve the positive `oid` encoding `(queue_id << 55) | sequence`, the 55-bit sequence mask/maximum, and zero as invalid. `encode_token()` must not allocate or throw. `submit()` must reject sequence exhaustion before queue work and preserve the current transactional sequence rollback behavior for synchronous queue-work failures, including the inline-completion/post-completion-failure cases.

8. **Wait and fence semantics.** `wait()` must reject non-positive tokens, zero queue IDs/sequences, foreign queue IDs, future/unsubmitted sequences, and skipped/reserved-but-never-submitted sequences with `std::invalid_argument`. For an accepted token it waits on the completion condition until the sequence completes or has a retained failure, rethrows the retained failure on every wait, and otherwise invokes `fence_through_sequence(sequence)` before its final retained-failure check. Accepted successes remain repeat-waitable; accepted failures remain repeatably failing; a wait is observational and must not release workspace, owner, queue, or registry state.

9. **Completion and failure retention.** Preserve all mutex/condition-variable guards and monotonic bookkeeping for `completed_`, `next_sequence_`, `skipped_sequences_`, `failures_`, and `pending_failures_`. `complete()` rejects zero and never-submitted sequences, consumes any pending committed failure, advances completion monotonically, retains failures, and wakes waiters. `record_post_completion_failure()` must not overwrite an earlier failure. `commit_failure()` retains only a nonempty failure for a reserved, not-yet-completed, non-skipped sequence and preserves its established rejection messages/categories. `seek_next_sequence()` remains a forward-only test seam bounded by `kMaxSequence + 1` and records skipped ranges exactly as before.

10. **Operation-family cutover.** After the split, every public operation family remains available exactly once: `copy` in `device_ops_copy.cpp`; all four binary operations and all four binary requirement queries in `device_ops_binary.cpp`; and `silu`, `linear`, `rmsnorm`, and `sdpa` in `device_ops_neural.cpp`. Each facade still uses the shared queue/device/view validation and shared failure boundary where its current implementation does, while preserving its family-specific validation, unsupported behavior, workspace query purity, capture, and backend hook dispatch.

11. **Build and line cap.** Add all seven planned core sources to `IOM_SOURCES`, remove `src/iom.cpp` entirely, and leave public header availability/transitive includes unchanged. Every core production `.cpp` or private `.hpp` touched or created by tasks 01–05 must be at most 499 physical lines after normal formatting. Task 21 performs the final tracked-production scan after the independent backend/header/shared slices land. Use the fewest responsibility-complete files; do not add a numbered shard, compatibility TU, permanent line-count test, backend switch, global registry, cross-backend abstraction, or extra capability.

## Non-goals

- Do not change public headers, public signatures, `Device::create_ops()` shape, capabilities, supported operation sets, OID values, error categories, validation precedence, numerics, allocator behavior, queue ordering, context behavior, ownership, lifetimes, or private visibility/ODR semantics.
- Do not move binary validation/facades/queries, copy validation/facade, or neural hooks/facades into `src/device_ops.cpp`; consume the blocked shard contracts and keep one responsibility-complete definition of each operation family.
- Do not redesign `submit`, workspace lease acquisition, registry/quarantine behavior, fence synchronization, completion ordering, failure retention, sequence allocation, or queue-ID allocation. No implicit workspace allocation, alternate lease path, compatibility alias, or synchronization redesign is allowed.
- Do not alter `src/iom_internal.hpp` into a miscellaneous helper collection, expose it as an installed/public header, or create another backend-neutral abstraction or process-global registry.
- Do not modify tests or add new tests, change test target source lists, change public factory headers, or add a permanent line-count test. Existing `test/test_iom.cpp` is behavior proof for the unchanged contract.
- Do not leave `src/iom.cpp` as an empty, forwarding, include-only, or compatibility translation unit, and do not leave any obsolete duplicate definitions behind.

## Acceptance criteria

- `src/iom.cpp` is absent and no compatibility translation unit replaces it. `CMakeLists.txt:IOM_SOURCES` contains exactly the seven planned core files `tensor.cpp`, `tensor_view.cpp`, `workspace.cpp`, `device_ops.cpp`, `device_ops_copy.cpp`, `device_ops_binary.cpp`, and `device_ops_neural.cpp` in addition to the retained non-core sources.
- All shared DeviceOps infrastructure named in Scope is defined exactly once in `src/device_ops.cpp`; `src/iom_internal.hpp` contains only the planned minimal private linkage; and no moved definition is missing, duplicated, publicly exposed, or ODR-ambiguous.
- The queue-ID pool still has exactly 255 process-global IDs; constructors/destructor lease and release IDs transactionally; tokens preserve queue-ID plus 55-bit sequence encoding; sequence exhaustion, skipped ranges, rollback, foreign/future/zero tokens, repeatable waits, fence calls, completion, and retained failures behave exactly as before.
- `WorkspaceValidation` still performs the same pure zero/positive requirement checks, `address()` returns the same range address, and no validation path allocates, registers, leases, reserves a token, mutates data, or changes backend state. Accepted binary requests preserve capture, workspace acquisition, owner registration, rollback, completion-proof release, and quarantine/error order.
- All operation families remain linked exactly once: `copy`; `add`, `mul`, `sub`, and `div` plus their four requirement queries; and `silu`, `linear`, `rmsnorm`, and `sdpa`. No family silently loses its facade or default unsupported behavior.
- Every core production source/header touched or created by tasks 01–05 is at most 499 physical lines after normal formatting. Task 21 owns the final repository-wide scan. No test source list, public factory header, backend switch, global registry, compatibility alias, extra capability, or permanent scanner test is introduced.
- The exact CPU configure/build/test gates and the link and line-cap proofs below complete successfully after implementation; hardware tests, when applicable elsewhere, remain fail-not-skip and are not substituted by this core-only gate.

## Verification

Proposed gates for the future implementer; none are run while writing this specification:

1. From the project source root, configure exactly as required by the parent source specification:

   ```sh
   cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
   ```

2. Build exactly the core library and CPU targets:

   ```sh
   cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests
   ```

3. Run the exact CPU CTest expression:

   ```sh
   ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'
   ```

4. Run a one-time deterministic line scanner (verification only, never a permanent test) over the core production paths touched or created by tasks 01–05. Fail if any retained core `.cpp`/`.hpp` exceeds 499 physical lines, and separately assert `src/iom.cpp` does not exist. Do not include unrelated backend files whose independent factoring tasks have not landed; task 21 runs the final tracked `src`/`include` scan after all slices complete.

5. Provide compile/link proof for the preserved public umbrella headers and the extracted library using a temporary translation unit that includes both `iom/iom.hpp` and `iom/detail/outstanding_work_registry.hpp`, references a non-inline public symbol such as `iom::TensorShape`, links against `build/split-cpu/libiom.a` with the normal C++20 compiler/link flags and pthread support, runs the resulting executable, and removes all temporary files. The proof must show no missing or duplicate symbols and must not alter production or test sources. The normal `iom_tests` and CPU target links remain part of the build proof; this focused TU additionally proves umbrella include/transitive availability after deleting `src/iom.cpp`.

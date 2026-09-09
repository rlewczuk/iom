# Migrate SYCL copy to the OID boundary

**Order:** 05
**Priority:** P0 — required backend migration before ADD can begin
**Blocked by:** `02-add-oid-facades`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The SYCL copy queue is a clean consumer of task 02's common non-throwing OID facade and protected backend copy hook. Valid copies, including waitable no-ops, return positive encoded tokens; deterministic failures before acceptance return the required signed `OidError` result without a token or output effect; and accepted post-launch failures remain repeatably observable through `wait`.

## Scope

- Migrate `SyclQueue::copy` and its `Task` execution path from the old public virtual/throwing boundary to task 02's protected copy extension point and signed error-result boundary. The public facade remains responsible for common validation, sequence reservation, token encoding, exception mapping, and wait bookkeeping; do not reintroduce a public backend capability query or a public virtual OID operation.
- Preserve exact queue-device/context validation, in-order SYCL queue execution, submission ordering, positive token acceptance, and the existing no-op behavior: an identical source/destination window still consumes a sequence and returns a waitable token without a native movement.
- Preserve the SYCL metadata-slot pool, host/device staging-pool behavior, view metadata used by the tiled copy kernel, and all cleanup on both success and failure. Do not allocate, replace, relocate, or retarget caller-owned tensor storage or handles.
- Preserve owner registry tracking and quarantine/invalidation behavior for source and destination allocations. Entries remain protected until a completion proof or retained failure is resolved, and cleanup must remove all provisional registrations when acceptance fails.
- Preserve `SyclFenceState` event-fence caching and completion ordering. A completed fence must be reused for repeated waits rather than waited again; a retained post-launch failure must be returned by the valid token's wait every time with the existing runtime failure semantics.
- Update `src/sycl/copy.hpp` declarations and test seams as required by the new extension boundary, without exposing SYCL types through the common public API.

## Implementation references

- **Modify:** `src/sycl/copy.cpp` — `SyclQueue` and `Task` execution at lines 389–653; adapt the queue submission, fence, registry, metadata-pool, and cleanup paths to task 02's protected hook and signed OID result contract.
- **Modify:** `src/sycl/copy.hpp` — `SubmissionFault`, fault controls, transfer declarations, and `make_queue`; keep the SYCL-private seam usable without adding a public capability API.
- **Read:** `src/sycl/device.cpp` — `SyclDevice` and `SyclTensor` at lines 49–225; use the owned-context, allocator-backed storage, stable native-handle, transfer, and quarantine behavior as the device/storage analogue. This remains outside the copy redesign.
- **Tests:** `test/sycl/test_sycl_conformance.cpp` — SYCL-specific deterministic fault coverage at lines 422–487, including pre-enqueue failures and retained post-enqueue failures. Update only assertions needed for the signed OID boundary; broader shared/backend migration and coexistence coverage belongs to task 08.

## Requirements

- Use the protected copy hook supplied by `02-add-oid-facades`; the SYCL class must not bypass the common facade's validation, sequence/token encoding, owner registration, or synchronous error mapping.
- For deterministic failures before acceptance, return the exact signed category through the task 02 boundary and perform no SYCL enqueue, output write, token acceptance, or lasting owner-registry registration:
  - `SubmissionFault::state_allocation`, `fence_construction`, and `outcome_insertion` represent allocation/bounded-resource failures and map to `OidError::ResourceExhausted`.
  - `SubmissionFault::first_submit` represents a SYCL/runtime failure detected before any work is accepted and maps to `OidError::DeviceError`.
  - Any otherwise unclassified internal failure maps to `OidError::InternalError` through the common boundary; do not leak an exception from an OID-returning public call.
- Preserve common mappings for validation, unsupported work, checked arithmetic, and sequence exhaustion: `InvalidArgument`, `Unsupported`, and `Overflow` respectively. The SYCL backend must not duplicate or weaken these checks, and no negative result is waitable.
- Keep the current transactional distinction at submission: a failure before `submitted_any` must clean up the outcome, registry entries, fence state, and metadata ownership so the common sequence reservation can be reclaimed when permitted; a failure after the SYCL event/kernel has been submitted must not be converted into a negative result.
- For an accepted post-launch failure (`SubmissionFault::post_launch` or an equivalent SYCL runtime failure after event submission), return a positive token, retain the failure with the token's fence/completion state, and make repeated `wait(token)` calls rethrow it. Never retry an accepted native runtime failure.
- Keep the queue's `sycl::property::queue::in_order{}` execution, metadata transfer before the tiled kernel, event fence creation/caching, completion callback ordering, and destructor drain. Pending work must not permit premature allocator reuse; pool resources and event state must be released only after completion proof or safely quarantined.
- Preserve exact `Device` identity and context ownership: views from another `SyclDevice` (even the same ordinal) remain invalid before acceptance, and SYCL storage pointers remain associated with the owning context. Keep `SyclDevice`/`SyclTensor` lifetime and stable-handle behavior shown in `src/sycl/device.cpp:49-225`.
- Preserve the existing SYCL fault scenarios in `test/sycl/test_sycl_conformance.cpp:422-487`: each pre-enqueue fault yields the required negative result and leaves the next valid submission at the reusable first sequence when no later reservation exists; the post-enqueue fault yields sequence one as a positive token and its failure is repeatably observable.

## Non-goals

- Do not modify or redesign the common OID definitions, public facades, token allocator, wait implementation, or protected-hook design owned by `02-add-oid-facades`.
- Do not migrate shared conformance fakes/tests, cross-backend coexistence tests, skipped/foreign/future wait matrices, or other backend test suites owned by `08-migrate-backend-oid-tests`.
- Do not change SYCL tensor storage layout, host-transfer algorithms, metadata/staging-pool capacity policy, tiled copy semantics, allocator ownership, or device discovery beyond what is necessary to compile against the migrated boundary.
- Do not implement ADD or any other compute operation, add support/capability queries, introduce a generic fallback subsystem, or add caller-visible options or conversions.

## Acceptance criteria

- [ ] `SyclQueue` uses task 02's protected copy extension boundary; no SYCL public OID-returning override bypasses the common no-throw facade.
- [ ] A valid SYCL copy and an identical-window no-op return positive tokens, execute in queue order, and preserve metadata, event, owner-registry, staging, and cleanup behavior.
- [ ] State allocation, fence construction, and outcome insertion faults return `OidError::ResourceExhausted`; first-submit faults return `OidError::DeviceError`; all occur before any token/effect and leave the next permitted sequence behavior intact.
- [ ] A post-launch fault returns a positive token and `wait` rethrows the retained failure on repeated waits; source/destination storage is not reused until the fence/cleanup path proves it safe.
- [ ] SYCL device/context and stable storage-owner behavior remains consistent with `src/sycl/device.cpp:49-225`, and the SYCL-specific fault scenarios at `test/sycl/test_sycl_conformance.cpp:422-487` cover the migrated result boundary.
- [ ] Focused remote verification is runnable on configured host `sycl|bv2` after oneAPI initialization, using `sycl-ls` to confirm Level Zero GPU visibility, then the SYCL smoke target and the focused conformance fault cases; no coexistence or ADD checks are required here.

## Verification

- On the configured remote `sycl` host (`bv2`), sync the assigned workspace with the `remote-development` `remote-sync` procedure, then run commands through `remote-exec` with the setup inside the command: `set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls` (must enumerate the Level Zero GPU), followed by `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF`.
- Build the focused targets remotely: `cmake --build build/sycl --target iom_sycl_smoke_tests iom_sycl_conformance_tests`.
- Run `ctest --test-dir build/sycl --output-on-failure -R '^iom_sycl_smoke_tests$'`, then invoke the conformance executable with doctest filters for `SYCL pre-enqueue failures preserve submission sequences` and `SYCL submission remains transactional across post-enqueue failures` (the cases currently at `test/sycl/test_sycl_conformance.cpp:422-487`).

# Migrate TTNN copy to the protected OID boundary

**Order:** 06
**Priority:** P0 — required backend migration in the complete OID compatibility cutover before any ADD work.
**Blocked by:** `02-add-oid-facades`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

`TtnnQueue::copy` uses the protected backend copy hook and the common non-throwing OID facade established by the prerequisite migration, rather than overriding the public OID operation. TTNN copies retain their current nine-leaf storage capability and preserve serialized submission, per-plane native copies, ownership registration, completion batching, stable TTNN-owned planes/handles, and repeatable retained failures while returning the exact negative OID category for every synchronous failure before acceptance.

## Scope

- Migrate the TTNN queue copy entry point and its completion/registry path in `src/ttnn/device.cpp:429-790` from the current public `copy` override to the protected copy extension/error boundary supplied by `02-add-oid-facades`.
- Keep caller submission order serialized by `submission_order_mutex_`, and keep all TTNN runtime calls serialized by `TtnnDevice::api_mutex()`; preserve the worker's `CompleteOnThrow` behavior and the existing sequence/completion ordering.
- Preserve registration of source and destination owners before native submission, rollback of partial registrations when no work is accepted, and release/invalidation/quarantine cleanup after completion or an undrainable failure.
- Preserve per-plane `ttnn_detail::copy_planes` submission and its distinction between failure before the first plane and failure after one or more planes. A partially submitted operation must be synchronously mapped to an error only when its submitted work was successfully drained; if it cannot be drained, publish a positive token and retain the original failure for `wait`.
- Preserve one native mesh finish per ready contiguous batch containing native work, no finish for an identical-window no-op batch, and the existing submission-order failure precedence.
- Preserve stable TTNN-owned plane vectors and native handles. Do not replace, relocate, or free caller-owned tensor storage before the relevant native work is terminal.
- Preserve host-transfer staging lease safety and device teardown ordering while changing the copy boundary: leases must not be reused before a covering finish, and retired/poisoned staging must retain the existing discard or quarantine behavior.

## Implementation references

- **Modify:** `src/ttnn/device.cpp:429-790` — `finish_locked`, `finish_native`, `TtnnQueue`, `TtnnQueue::execute`, `TtnnQueue::complete_task`, and `TtnnQueue::fence_through_sequence`; move only the public-copy-specific behavior behind the protected hook and retain the completion/registry machinery.
- **Read:** `src/ttnn/copy.cpp:384-520` — `region_to_host` and `copy_planes`; preserve the per-plane mapping, submission-failure seam, and native queue-drain assumptions used by the queue.
- **Read:** `src/ttnn/staging.hpp:41-322` — `TtnnHostStaging`, `UploadLease`, and `DownloadLease`; preserve release, discard, retire, and reclaim ordering.
- **Read:** `src/ttnn/registry_state.hpp:66-108` — `TtnnNativeCleanupAction` and the TTNN registry test seams; preserve cleanup retry/quarantine behavior.
- **Read:** `test/ttnn/test_ttnn_conformance.cpp:654-1332` — current asynchronous copy, transfer, lifetime, registration rollback, partial-plane, retained-failure, native-finish, no-op, and batch-finish coverage to migrate to the new OID expectations without changing its TTNN storage matrix.
- **Use:** the protected copy hook and common facade contract produced by `02-add-oid-facades`; do not add a second public facade, capability query, or TTNN-specific public fallback API.

## Requirements

- The public `DeviceOps::copy` call must go through the common `noexcept` facade. `TtnnQueue` must no longer override the public operation or bypass common validation, token encoding, sequence reservation, or synchronous exception mapping.
- The TTNN protected hook must return accepted work only through the common submission path, which yields a positive OID. Zero and negative values must never be used as accepted tokens, and a synchronous failure must not mutate destination data or leave a submitted sequence behind.
- Map synchronous failures before OID acceptance exactly as follows: malformed/foreign/mismatched views, invalid metadata, and other caller validation failures to `OidError::InvalidArgument`; an unsupported/unavailable TTNN copy specification or operation to `OidError::Unsupported`; checked arithmetic or sequence exhaustion to `OidError::Overflow`; `std::bad_alloc` and bounded-resource failures to `OidError::ResourceExhausted`; TTNN/native runtime failures detected before acceptance to `OidError::DeviceError`; and otherwise unclassified failures to `OidError::InternalError`.
- A native runtime failure after a positive token has been accepted must not be converted into a synchronous negative OID or retried. Retain it in the existing completion record and make every `wait(token)` throw the same failure, including repeated waits.
- Keep validation and registration before the first native plane reaches the mesh. If registration or outcome insertion fails, remove every entry already registered, reclaim the reserved sequence according to the common submission contract, and return the mapped negative OID with no native effect.
- If a plane submission fails before any plane reaches the mesh, roll back the entries and return `DeviceError` (or the more specific mapped category) synchronously. If one or more planes reached the mesh, attempt the existing synchronous drain; map the failure synchronously only when that drain succeeds, otherwise publish a positive token with the submission failure retained.
- Keep `executed_seq_`, `outcomes_`, `last_finished_seq_`, and the contiguous-batch completion logic consistent with submission order. A finish failure for a ready native batch must be retained for every affected token in that batch, while later batches remain usable.
- Keep identical-window copies as accepted no-ops with no native mesh finish, and keep repeated waits successful for successful no-ops and repeatably failing for retained failures.
- Do not change `ttnn_supported_data_types()`, the current `kSupportedToNative` nine-leaf table, TTNN tensor creation/backing, native plane geometry, or host-transfer storage. Storage expansion is a separate task.
- Keep the existing stable native plane/handle lifetime guarantees through queue destruction, owner destruction, quarantine action failure, temporary derived views, and immediate owner reuse after a drained failure.

## Non-goals

- Expanding TTNN storage beyond the current `BOOL` plus eight numeric `NONE` leaves, adding emulation, or changing tensor creation and host-transfer support.
- Implementing ADD, changing any other operation, migrating common/backend tests outside the TTNN copy cases, or updating architecture/conformance documentation.
- Changing `copy_planes`, TTNN staging allocation policy, native tile geometry, registry data structures, queue token encoding, wait validation, or the common facade design except where required to consume the established protected hook/error boundary.
- Adding a public `add_support`/capability query, options object, public fallback knob, compatibility shim, or alternate TTNN copy API.

## Acceptance criteria

- [ ] `TtnnQueue` no longer exposes a public `copy` override; its backend implementation is reachable only through the protected common OID boundary, and no synchronous exception escapes the public copy call.
- [ ] Valid copies return positive OIDs and preserve logical contents, in-order visibility, temporary-view safety, stable owners/handles, and the existing nine-leaf TTNN copy capability.
- [ ] Validation, unsupported-spec, overflow, allocation, native-runtime, and unknown pre-acceptance failures produce exactly `InvalidArgument`, `Unsupported`, `Overflow`, `ResourceExhausted`, `DeviceError`, and `InternalError` respectively, with no destination or sequence side effects.
- [ ] Partial-plane failures drain before owner release when possible; an undrainable failure produces a positive token whose repeated waits report the retained failure, and later healthy copies remain usable.
- [ ] Registration/outcome rollback, quarantine cleanup, staging lease disposition, identical-window no-op completion, contiguous native-finish batching, one-failed-batch propagation, and repeated successful waits remain behaviorally unchanged.
- [ ] The focused TTNN conformance coverage remains green after its expected OID assertions are migrated: `TTNN conformance: asynchronous copies against the CPU reference`; `TTNN conformance: copy validation fails before writes and sequences`; `TTNN conformance: transfer failures keep metadata and ownership`; `TTNN host transfers reuse retained staging after warm-up`; `TTNN host-transfer failures discard poisoned staging`; `TTNN conformance: deferred queue lifetime and stability`; `TTNN copy survives derived-view temporaries`; `TTNN failed plane submissions drain before rethrow`; `TTNN registration and outcome insertion failures roll back ownership`; `TTNN un-drainable failed submission reports a repeatable failed token`; `TTNN native finish failure reports a repeatable failed token`; `TTNN identical-window copies complete without native finish`; `TTNN mixed no-op and native batches finish native work once`; `TTNN no-wait bursts share one native finish per ready batch`; and `TTNN one failed batch finish fails every token of that batch`.

## Verification

- `ctest --test-dir <build> -R '^iom_ttnn_conformance_tests$'` — focused TTNN conformance target; not run per assignment.
- Run the listed TTNN copy/lifetime/failure cases from `test/ttnn/test_ttnn_conformance.cpp:654-1332` with the TTNN hardware gate enabled, including repeated waits after retained failures and healthy-copy-after-failure checks; not run per assignment.
- No builds, tests, formatters, linters, or other gates are run for this specification task.

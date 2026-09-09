# Migrate core OID tests and shared fakes

**Order:** 07
**Priority:** P0 — the core OID tests and deterministic shared fixtures must prove the public compatibility cutover before any ADD work or backend-specific conformance migration.
**Blocked by:** `01-define-oid-contract`, `02-add-oid-facades`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The core `iom_tests` coverage and shared conformance fixtures exercise the migrated signed OID contract through the common non-throwing public facades. They prove token allocation, exact error values and mappings, sequence and wait behavior, protected backend hooks, and the retained negative behavior of every default compute hook without relying on the pre-migration throwing OID API.

## Scope

- Migrate the OID/signature/token tests in `test/test_iom.cpp:1695-2168` and the related fake queue definitions in `test/test_iom.cpp:767-958` to the contract established by tasks 01 and 02. Update only test seams or wiring needed to compile and observe that contract.
- Migrate shared fake queues in `test/backend/backend_conformance_other.hpp:42-190`: `DeferredCopyQueue`, `InstrumentedQueue`, and their deterministic completion/failure paths must use the protected backend extension hooks and signed OID results rather than overriding public OID facades.
- Migrate common token and signature helpers in `test/backend/backend_conformance_common.hpp:209-303`, including repeated asynchronous-failure checks, token decoding, and compile-time facade-signature assertions.
- Preserve the existing queue-lifetime, stable-owner/handle, in-order completion, repeat-wait, and retained-failure scenarios while changing synchronous failures to returned negative OIDs.
- Keep every default compute hook, including ADD, at negative `OidError::Unsupported` during this preliminary migration. Task 10 later introduces a positive fake ADD seam after all nine OID tasks complete.

## Implementation references

- **Modify:** `test/test_iom.cpp` — `FakeQueue`, `InlineQueue`, token helpers, and the `DeviceOps` signature, queue-id, sequence, submit, rollback, retained-failure, wait, and destruction test cases at lines 767-1003 and 1695-2168.
- **Modify:** `test/backend/backend_conformance_other.hpp` — `DeferredCopyQueue` and `InstrumentedQueue`; route copy/probe and deterministic failure behavior through the protected extension boundary while retaining observable records and destruction journaling.
- **Modify:** `test/backend/backend_conformance_common.hpp` — `expect_repeated_runtime_failure`, token decoding helpers, and common `DeviceOps` member-pointer/signature assertions.
- **Read:** `include/iom/oid.hpp` — planned public signed OID type, `OidError` values, and `to_oid`, `oid_is_error`, and `oid_is_token` helpers delivered by task 01.
- **Read:** `include/iom/iom.hpp` — `DeviceOps` public facade and protected extension-point declarations delivered by task 02; do not recreate or alter that API in this task.
- **Tests:** `test/CMakeLists.txt` — existing `iom_tests` and `iom_backend_conformance_cpu_tests` targets; change test wiring only if the migrated source requires it.

## Requirements

- Prove at compile time and through calls made via `DeviceOps` references that `iom::oid` is signed `std::int64_t`; every public OID-returning operation has its exact view-only signature, is `noexcept`, and is a common non-virtual facade backed by protected hooks. Retain the const/mutable view constraints and reject obsolete `Tensor` operand overloads at compile time.
- Prove the exact complete error enum and helper behavior: `OidError::InvalidArgument == -1`, `Unsupported == -2`, `Overflow == -3`, `ResourceExhausted == -4`, `DeviceError == -5`, and `InternalError == -6`; `to_oid` returns those values; every negative OID is an error; every positive OID is a token; and zero is neither.
- Replace the old 56-bit token assumptions with the exact encoding `((uint64_t{queue_id} << 55) | sequence)`. Exercise all 255 queue IDs `[1,255]`, preserve queue-ID reuse after destruction, verify queue ID bits 55-62 and the 55-bit sequence `[1, 2^55-1]`, and verify positive tokens never collide with the six negative error values.
- Exercise sequence exhaustion through the deterministic `seek_next_sequence` seam. A submission at sequence `2^55-1` is accepted and waitable; the next submission returns `to_oid(OidError::Overflow)` synchronously, invokes no backend hook, records no submission, has no output/storage effect, and does not accept a token. Invalid seek arguments and sequence zero remain rejected without creating a token.
- Assert that accepted fake operations return positive tokens and that synchronous validation or backend failures return the one mapped negative `OidError` before effects, submission, sequence consumption, or token acceptance. Cover deterministic seams for all six synchronous categories: invalid argument, unsupported, overflow, resource exhausted, device error, and internal error.
- Migrate `FakeQueue`, `InlineQueue`, `DeferredCopyQueue`, and `InstrumentedQueue` to protected backend hooks and negative result reporting. Public calls must exercise the common facade; fakes must not override or bypass that facade. Keep deterministic pre-acceptance and post-acceptance failure modes observable.
- Verify synchronous exception-to-error conversion and rollback semantics: no synchronous exception crosses an OID-returning facade, a failed pre-acceptance call has no submission/sequence effect, and an accepted inline operation remains waitable even when later work throws or records a failure according to the queue contract.
- Verify valid tokens support in-order completion and repeated successful waits. Verify a post-acceptance asynchronous failure is retained and rethrown identically on every repeated `wait`; preserve queue ordering and visibility for later completed tokens. Negative, zero, foreign, future, and otherwise unsubmitted values must be rejected by `wait` as invalid non-tokens where covered by core fixtures; do not add the explicit skipped-after-later-completion scenario reserved for task 08.
- Assert every default compute operation (`add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`) returns `to_oid(OidError::Unsupported)` through the facade, with no output mutation, submission, sequence consumption, or token acceptance. Remove old expectations that these unsupported calls throw `std::runtime_error`; retain `std::runtime_error` assertions only where they verify an intentionally retained asynchronous failure from `wait`.
- Keep tests consumer-observable: assert returned OIDs, effects, hook/record counters, wait behavior, and stable owners/handles, not implementation source text or private field layout. Do not add a public `add_support` query or any equivalent capability API.

## Non-goals

- Do not modify production OID declarations, facade implementation, token registry, queue implementation, or backend drivers; those belong to tasks 01-06 and are blockers rather than requirements to rediscover here.
- Do not migrate backend-specific conformance drivers, backend coexistence coverage, or accelerator/runtime tests; task 08 owns those changes.
- Do not add the skipped-token-after-a-later-completion test; the explicit `seek_next_sequence` skipped-value scenario is reserved for task 08.
- Do not make ADD positive in any fake or production backend, and do not implement ADD arithmetic, numeric codecs, validation, storage expansion, or any other ADD behavior. The ADD unsupported expectation remains until task 10, after the complete OID phase.
- Do not change non-OID `Device` queries, unrelated tests, documentation, CMake targets, or sibling task specifications.

## Acceptance criteria

- [ ] `test/test_iom.cpp` and both shared conformance headers compile against the migrated protected-hook API without public fake overrides or old 56-bit helpers.
- [ ] Core tests observe a signed `std::int64_t` OID, exact six negative values and helper predicates, exact `noexcept` view-only facade signatures, zero invalidity, accepted positive tokens, all 255 queue IDs, queue-ID reuse, the bits 55-62/55-bit sequence layout, and the maximum-sequence boundary.
- [ ] Sequence exhaustion returns `to_oid(OidError::Overflow)` before hook invocation or any observable effect; the maximum valid sequence remains waitable, and invalid seek/zero sequence inputs do not produce accepted tokens.
- [ ] Deterministic tests observe each of the six synchronous error mappings as a negative OID with no output/storage mutation, submission, sequence consumption, or token acceptance, and no synchronous exception crosses the public OID facade.
- [ ] Repeated successful waits, in-order completion, retained asynchronous failures, and repeated identical failure waits remain observable; invalid non-token waits are rejected, while the skipped-after-later-completion case remains solely in task 08.
- [ ] Every default compute hook, including ADD, returns `to_oid(OidError::Unsupported)` with no effects, and all obsolete runtime-error expectations for synchronous unsupported operations are removed. The focused shared CPU conformance path continues to use the migrated helper/fake behavior.

## Verification

- `cmake --build build --target iom_tests` — not run per assignment; focused core target.
- `cmake --build build --target iom_backend_conformance_cpu_tests` — not run per assignment; focused CPU shared-conformance target.
- `ctest --test-dir build --output-on-failure -R '^(iom_tests|iom_backend_conformance_cpu_tests)$'` — not run per assignment; focused runtime targets only.

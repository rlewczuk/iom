# Verify elementwise-add coexistence

**Order:** 17
**Priority:** P2 — required cross-backend integration proof after every backend ADD path is complete; this is not optional backend coverage.
**Blocked by:** `13-implement-cuda-rocm-add`, `14-implement-sycl-add`, `16-implement-ttnn-add`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

Extend the existing combined backend coexistence executable so one process exercises ADD concurrently and interleaved across every enabled CPU, CUDA, ROCm, SYCL, and TTNN participant. The test must distinguish backend or device mix-ups, process-global state leaks, registry and lifetime collisions, queue-ordering errors, and retained-failure poisoning while preserving the existing copy/coexistence proof. Register only the focused coexistence target changes needed to build and run this coverage.

## Scope

- Build the integration on the existing `BackendParticipant` data and factory helpers in `test/backend/test_backend_coexistence.cpp` (around lines 283–390), the interleaved participant/queue copy case (around lines 394–511), the independent-device identity case (around lines 513–594), and the queue identity, concurrency, and lifetime cases (around lines 645–880).
- Add cross-backend ADD submissions to the same process-wide participant arrangement. Under the corresponding `IOM_COEXIST_*` macros, include every enabled CPU, CUDA, ROCm, SYCL, and TTNN backend; do not turn a backend off or silently substitute another backend.
- Submit ADD operations round-robin across participants and across at least two queues per participant without waits between submissions. Assert every accepted result is a positive `oid`, and assert the process-wide queue/token identity remains unique and attributable to the submitting queue.
- Check representative integer and floating ADD results against the one independent scalar oracle owned by the ADD implementation tasks. The integration test may use a compact set of representative supported numeric leaves and shapes; it must not duplicate the 21-leaf or exhaustive encoding matrices owned by tasks 12–16.
- Exercise the existing asynchronous contract: wait from the originating queue, wait in an order different from submission order, repeat at least one successful wait, and verify results only after the relevant waits complete.
- Interleave copy and ADD on the same queue and on different queues where the existing participant setup permits it. Prove queue ordering with at least a copy-then-ADD dependency and an ADD-then-copy/readback dependency, without introducing host synchronization that bypasses the queue contract.
- Preserve and retain the existing copy interleaving, destination checks, queue-id uniqueness, concurrent registry, and lifetime coverage. New ADD assertions must not weaken or replace those checks.
- Exercise exact device identity validation with tensors from independent `Device` instances. A mixed-device ADD must return the appropriate negative `OidError` before effects or token acceptance, including same-backend independent instances where that configuration supports them and representative cross-backend mismatches. Confirm the rejected operation does not mutate output or poison a valid queue.
- Capture caller owner and handle identities before ADD, then verify they are unchanged after completion. Keep inputs, output, and caller-owned storage in place; no ADD integration test may accept replacement or relocation of a caller tensor or handle.
- Submit an ADD using derived/transformed view metadata held by short-lived local view objects, release those metadata wrappers before waiting, and verify completion uses the captured metadata and produces the oracle result. Do not require the backend to retain caller view objects.
- Where the shared integration fixtures make aliasing observable, cover exact permitted in-place aliases and cross-backend registry tracking of repeated aliases. Verify aliases are deduplicated for lifetime/cleanup purposes and are not double-released; leave backend-specific exhaustive alias matrices to their conformance tasks.
- Use the existing deterministic fault seams, or the smallest integration seam exposed by the prerequisite ADD tasks, to accept one operation and then produce a retained asynchronous failure on one backend queue. Repeated waits on that token must rethrow the retained failure, while independent queues and other backend participants continue to complete and validate successfully.
- Update `test/CMakeLists.txt` only as required to compile/link/register the existing `iom_backend_coexistence_tests` target with the new integration source or minimal test seam. Keep the existing enabled-backend macros, libraries, SYCL link handling, and single combined target model.

## Implementation references

- **Modify:** `test/backend/test_backend_coexistence.cpp` — `BackendParticipant`, `make_*_participant`, and coexistence test cases around lines 283–880; add the genuinely cross-backend ADD scenarios and assertions here while retaining the current copy, identity, queue, and lifetime cases.
- **Modify:** `test/CMakeLists.txt` — the `iom_backend_coexistence_tests` definition and `add_test` registration around lines 238–346; preserve conditional backend participation and add only required source/link/seam wiring.
- **Read:** `test/backend/test_backend_coexistence.cpp` — the existing interleaved-copy, foreign-device rejection, queue-id, concurrent registry, and destruction scenarios; reuse their participant/factory, allocator, token, and logical-byte checking conventions rather than creating a second coexistence harness.
- **Read:** `test/CMakeLists.txt` — `add_iom_backend_tests` and the combined target's conditional definitions/link options; follow the established CMake conventions for enabled backend libraries and SYCL.
- **Read:** `test/backend` ADD/conformance helpers introduced by tasks 10–16 — use the shared tensor/view construction, scalar oracle, deterministic failure, and logical-result checking seams instead of duplicating backend arithmetic or a second oracle. Any file not present until those tasks land is a planned reference.

## Requirements

- The focused integration must run as one process and must include every backend enabled by the build. In the all-backend configuration, CPU, CUDA, ROCm, SYCL, and TTNN must all participate in the same interleaved ADD scenario.
- ADD support is signaled only by `add(lhs, rhs, out)`. Do not add or call a public `add_support` query or another capability API. Accepted work returns a positive token; synchronous rejection returns the specified negative `OidError` and has no effects.
- Use the independent scalar oracle for expected numeric values. Assert representative exact integer results and the specified floating result class/envelope, including any selected special-value case, without making the coexistence test an exhaustive leaf or encoding test.
- Keep all three tensor views caller-owned and on the exact queue `Device` required by the operation. Validate that a foreign or otherwise independently owned view is rejected before submission and that a later valid submission still succeeds.
- Prove asynchronous ordering, repeat waits, and retained asynchronous failures. A failure retained by one accepted token must be isolated from other queues, devices, and backend participants.
- Prove positive-token uniqueness/attribution, stable device/backend identity, stable caller owners and handles, derived metadata lifetime, and alias-safe registry cleanup in the combined process. Do not rely on backend-kind globals or a test-only backend selector to make the assertions pass.
- Keep the existing copy/coexistence checks active and compatible with the ADD checks. The test must continue to detect queue-id collisions, cross-queue registry invalidation, and cleanup that releases storage more than once or not at all.
- Keep common test code backend-neutral: no vendor type, vendor-kind switch, generic CPU fallback, new public fallback API, or new production capability framework. Minimal integration-only seams are permitted when required to deterministically trigger the specified retained failure.
- Preserve the existing target's conditional compilation and linking for CUDA, ROCm, SYCL, and TTNN. Do not create a second executable for this integration.

## Non-goals

- Do not repeat each backend's 21-leaf ADD conformance, exhaustive F4/F6/FP8 encoding matrices, full broadcast matrix, or backend-specific alias matrix; tasks 12–16 own backend completeness.
- Do not implement ADD arithmetic, codecs, storage expansion, queue machinery, OID migration, or backend fallback/emulation in this task.
- Do not change public ADD semantics, add options, add capability queries, or introduce mixed-type promotion, allocation of caller operands/results, or public broadcast views.
- Do not replace or redesign the existing copy/coexistence tests, participant factories, registry model, or test target. Do not add benchmarks, broad documentation, unrelated production changes, or deferred backend fixes.
- Do not weaken a failed-backend assertion by skipping the backend, converting it to a copy-only check, or accepting a backend-specific result outside the frozen ADD contract.

## Acceptance criteria

- [ ] In the configured all-backend environment, `iom_backend_coexistence_tests` constructs CPU, CUDA, ROCm, SYCL, and TTNN participants in one process and interleaves ADD submissions across their queues; every accepted operation returns a positive, uniquely attributable token and all representative outputs match the independent oracle.
- [ ] The integration proves copy/ADD queue ordering, out-of-order waits, repeated waits, and visibility only after the relevant token completes, while the pre-existing interleaved-copy and queue-registry assertions remain active.
- [ ] Independent-device and backend/device mix-ups are rejected synchronously with the specified negative OID, with no output mutation or token side effect; subsequent valid work on the rejecting and unrelated queues succeeds.
- [ ] Caller owners and handles remain stable; short-lived derived metadata remains valid through completion; applicable exact aliases are deduplicated and cleaned up without double release or cross-queue invalidation.
- [ ] A deterministic accepted asynchronous failure is repeatably retained by its originating token and does not poison other backend queues or devices, which still complete and pass oracle checks.
- [ ] `test/CMakeLists.txt` builds and registers the existing `iom_backend_coexistence_tests` target for the enabled-backend combinations without adding a second target or changing unrelated test registration.

## Verification

- `ctest --test-dir build/all --output-on-failure -R '^iom_backend_coexistence_tests$'` — run in the configured all-backend environment; no gate is run while writing this mini-specification.

# Document the OID compatibility contract

**Order:** 09
**Priority:** P0 — completes the required public OID compatibility cutover before ADD work begins.
**Blocked by:** `08-migrate-backend-oid-tests`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

Update the public documentation and public-header comments to describe the migrated signed OID contract consistently with the declarations and migrated tests. The documentation must make token/error classification, encoding, synchronous error timing, wait behavior, backend extension boundaries, and caller-visible breaking changes unambiguous, while leaving later ADD semantics and TTNN capability expansion out of scope.

## Scope

- Document `oid` as signed `std::int64_t`, with all negative values reserved for errors, positive values reserved for accepted asynchronous tokens, and zero invalid (neither a token nor an error).
- Document exactly these stable error values: `OidError::InvalidArgument = -1`, `Unsupported = -2`, `Overflow = -3`, `ResourceExhausted = -4`, `DeviceError = -5`, and `InternalError = -6`.
- Document the public helpers `to_oid(OidError) noexcept`, `oid_is_error(oid) noexcept`, and `oid_is_token(oid) noexcept`, including that the predicates classify by sign and that `to_oid` returns the enum's negative underlying value.
- Document token encoding with every queue ID `q` in `[1, 255]` represented in bits 55 through 62 and a 55-bit sequence in `[1, 2^55 - 1]`: `static_cast<oid>((std::uint64_t{q} << 55) | sequence)`. Sequence zero is never submitted, all 255 queue IDs remain representable, and sequence exhaustion returns synchronous `Overflow` before effects or acceptance.
- Document that every public OID-returning operation (`copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`) is a common `noexcept` facade. Backend extension points are protected hooks behind that facade; they cannot bypass common validation, error mapping, token encoding, or lifetime registration. `wait` remains the operation that observes accepted asynchronous work and may throw.
- Document the exact synchronous mapping and timing: invalid caller input to `InvalidArgument`; unavailable operation or unsupported specification to `Unsupported`; checked arithmetic or sequence exhaustion to `Overflow`; allocation or bounded-resource failure to `ResourceExhausted`; backend/runtime failure detected before acceptance to `DeviceError`; and otherwise unclassified internal failure to `InternalError`. Each negative result is returned before effects or token acceptance, and no synchronous exception crosses an OID-returning public method.
- Document that accepted work always returns one positive token. Failures that occur after acceptance are retained and are repeatably rethrown by `wait(token)`, while successful completion and visibility remain repeatable and queue ordering is preserved.
- Document that `wait` immediately throws `std::invalid_argument` for a negative value, zero, a foreign queue token, a future value, a skipped/reserved-but-never-submitted value, or any otherwise unsubmitted value. A skipped value remains invalid even after a later submission completes. Preserve the existing in-order completion, repeated-wait, retained-failure, and caller serialization rules.
- Replace stale public wording in `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, `include/iom/oid.hpp`, `include/iom/iom.hpp`, and any other stale public API wording found in the OID contract surface. The resulting declarations and comments must agree with the migrated OID tests, including the all-255-ID and skipped-sequence behavior.
- State the breaking caller impact: callers must consume negative OID results instead of expecting synchronous exceptions from OID-returning operations, must only wait on accepted positive tokens, and must retain the existing `wait` exception handling for invalid tokens and retained asynchronous failures.
- Keep current compute hooks other than the later ADD implementation at `Unsupported`; documentation must not imply that `mul`, `silu`, `linear`, `rmsnorm`, or `sdpa` became implemented as part of this cutover.

## Implementation references

- **Modify:** `docs/ARCHITECTURE.md` — public API guide, current OID/token paragraph around lines 201–240; replace the old unsigned/56-bit/throwing description with the signed encoding, facade, error, wait, and compatibility contract.
- **Modify:** `docs/BACKEND_CONTRACT.md` — queue/token contract around lines 267–297 and compute capability contract around lines 323–338 (and related public contract wording around lines 241–338 and 347–392); make synchronous OID failures non-throwing while retaining throwing `wait` behavior and unsupported compute-hook requirements.
- **Modify:** `include/iom/oid.hpp` — planned public header; document the signed alias, exact `OidError` values, and `noexcept` helpers alongside their declarations.
- **Modify:** `include/iom/iom.hpp` — `DeviceOps` public comments and declarations; remove stale `uint64_t`, 56-bit, and throwing-compute prose and describe common non-throwing facades plus protected backend hooks. The OID alias must come from `oid.hpp`, not a duplicate typedef.
- **Read:** `include/iom/iom.hpp` — `DeviceOps::wait`, `submit`, `complete`, `commit_failure`, `seek_next_sequence`, and the compute declarations; use the migrated signatures and hook terminology established by tasks 01–08.
- **Read:** `docs/changes/002-eltwise-add/spec.md` — preliminary OID migration sections 41–43 and 80–102; these exact values, mappings, timing, and migration impact are normative for this documentation task.
- **Tests:** migrated core/backend OID suites from tasks 07–08 — use their asserted error values, all-255 queue-ID coverage, wait rejection cases, sequence-exhaustion timing, and retained asynchronous-failure behavior as the consistency baseline. Do not add or alter tests in this task.

## Requirements

- Use the exact spelling and values of `oid`, `OidError`, `to_oid`, `oid_is_error`, and `oid_is_token`; do not introduce an alternate error enum, capability query, `add_support` API, or compatibility alias.
- Explicitly distinguish synchronous negative results from accepted positive tokens and from `wait` exceptions. A negative result is terminal and is never waitable.
- Describe queue IDs as the complete range 1 through 255 and place them in bits 55..62; describe the sequence as exactly 55 bits. Do not retain wording that says `oid` is `uint64_t`, that the sequence is 56 bits, or that sequence exhaustion is synchronously thrown.
- State that validation, mapping, encoding, and lifetime registration belong to the common facade and occur before backend effects or token acceptance. Backend hooks are protected implementation extension points, not alternate public entry points.
- State the complete synchronous error mapping and the exact invalid-wait set: negative, zero, foreign, future, skipped, and unsubmitted values. Include the rule that a skipped value stays invalid after later completion.
- Preserve queue ordering, visibility, repeated waits, retained post-acceptance failures, and caller serialization in the documentation. Do not weaken the documented lifetime or asynchronous failure contract.
- Describe the clean breaking cutover for callers and keep non-OID `Device` queries outside this result contract.
- Keep the compute capability wording compatible with the migration: unimplemented compute hooks remain `Unsupported` before submission and output mutation, except that a later ADD task may replace ADD's unsupported expectation. Do not document ADD arithmetic, broadcasting, numeric support, or any TTNN storage expansion here.
- Make the documentation self-contained and aligned with the public declarations and migrated tests; do not refer readers to a parent or sibling task specification as a substitute for the contract.

## Non-goals

- Do not implement or change OID behavior, queue logic, error mapping, backend copy paths, or tests.
- Do not document or implement ADD arithmetic, broadcasting, numeric leaves, capability expansion, or TTNN storage/backing changes; those belong to later tasks.
- Do not add public capability queries, `add_support`, options objects, fallback APIs, or alternate token formats.
- Do not change non-OID exceptions used by tensor construction, synchronous host transfers, allocators, or unrelated APIs.
- Do not claim support for compute operations that remain unimplemented; later ADD work is the only planned exception to the current ADD `Unsupported` expectation.

## Acceptance criteria

- [ ] `docs/ARCHITECTURE.md` and `docs/BACKEND_CONTRACT.md` no longer describe OIDs as unsigned/`uint64_t`, use a 56-bit sequence, or report synchronous OID failures as thrown exceptions.
- [ ] Public comments/declarations in `include/iom/oid.hpp` (planned) and `include/iom/iom.hpp` state the signed type, exact six error values/helpers, 55-bit sequence, bits 55..62 queue-ID encoding, positive/negative/zero classification, common `noexcept` facades, protected hooks, and caller-visible cutover.
- [ ] The documented synchronous mapping and pre-effect timing exactly match the migrated declarations/tests, and negative results are explicitly non-waitable.
- [ ] The documented `wait` contract covers negative, zero, foreign, future, skipped, and unsubmitted values, including rejection of a skipped value after later submissions complete, while retaining repeat waits and retained asynchronous failures.
- [ ] The contract explicitly preserves all 255 queue IDs and states that current non-ADD compute hooks remain `Unsupported` before submission; no ADD behavior or TTNN capability expansion is introduced.
- [ ] Manual consistency verification finds no stale OID wording in the targeted public documentation/header surface and no contradiction with the migrated OID tests. No documentation-specific target exists, so this is a review-only check.

## Verification

- Manual consistency review only: compare the edited documentation and public-header comments against the exact OID declarations and migrated core/backend OID tests from tasks 01–08, including all-255 IDs, sequence exhaustion, error timing, and every invalid-wait case.
- No builds, tests, formatters, linters, or other gates are run for this documentation-only task.

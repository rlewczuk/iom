# Add common OID facades

**Order:** 02
**Priority:** P0 — establishes the common public compatibility boundary and token/lifetime safety that every backend and every later ADD task depends on.
**Blocked by:** `01-define-oid-contract`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

Replace the seven public OID-returning `DeviceOps` operations with one clean, backend-neutral cutover: `copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa` are common non-virtual `noexcept` facades, while backend work is reached only through protected hooks. Every facade owns validation, exact `Device` and tensor-owner identity checks, sequence reservation and token encoding, synchronous exception-to-`OidError` mapping, and outstanding-owner registration. Accepted work returns a positive token; every synchronous failure returns one negative error and escapes no exception. `wait` continues to throw for invalid tokens and repeatedly rethrows retained asynchronous failures.

## Scope

- Update `DeviceOps` in `include/iom/iom.hpp` and its implementation in `src/iom.cpp` so the public methods have their existing argument lists and return `oid`, but are non-virtual and `noexcept`:
  - `oid copy(const TensorView&, TensorView&) noexcept`;
  - `oid add(const TensorView&, const TensorView&, TensorView&) noexcept`;
  - `oid mul(const TensorView&, const TensorView&, TensorView&) noexcept`;
  - `oid silu(const TensorView&, TensorView&) noexcept`;
  - `oid linear(const TensorView&, const TensorView&, TensorView&) noexcept`;
  - `oid rmsnorm(const TensorView&, TensorView&, const TensorView&, float, size_t) noexcept`;
  - `oid sdpa(const TensorView&, const TensorView&, const TensorView&, size_t, size_t, size_t, TensorView&) noexcept`.
- Put backend extension points behind the facades as protected hooks. Hooks receive only the already-validated operation arguments and the common submission/lifetime context they need; they cannot reserve or encode a public token, bypass common validation, choose a different queue/device identity, or publish work without common registration. The exact protected hook names are an implementation detail, but no backend may override a public facade.
- Apply this synchronous result mapping uniformly to all seven operations:
  - invalid caller arguments, malformed or mismatched views/specs, wrong device identity, wrong owner identity, invalid alias/view relationships, and invalid operation parameters → `to_oid(OidError::InvalidArgument)`;
  - an unimplemented operation or unsupported recognized specification → `to_oid(OidError::Unsupported)`;
  - checked arithmetic failure or exhausted 55-bit submission sequence → `to_oid(OidError::Overflow)`;
  - allocation, registration, metadata, staging, or other bounded-resource failure before acceptance → `to_oid(OidError::ResourceExhausted)`;
  - a backend/runtime failure detected before work is accepted → `to_oid(OidError::DeviceError)`;
  - any otherwise unclassified synchronous failure → `to_oid(OidError::InternalError)`.
  All exceptions raised by common validation, hook dispatch, sequence management, or lifetime registration must be caught inside the facade and converted to one of these results. No synchronous exception may cross an OID-returning public method.
- Preserve the exact timing boundary: all validation, device/owner checks, checked arithmetic, resource/lifetime registration, backend path selection, and sequence-exhaustion checks happen before effects and before token acceptance. A negative result performs no operation effect and does not create a waitable token. Accepted work returns exactly one positive token, encoded with queue ID `q` in bits 55–62 and sequence in `[1, 2^55 - 1]`; sequence zero is never submitted.
- Make the common submission path retain asynchronous failures after a positive token is returned. `wait` must remain throwing and must immediately throw `std::invalid_argument` for negative, zero, foreign, future, skipped, or otherwise unsubmitted values. A valid token preserves in-order queue visibility, and a successful wait is repeatable; a retained asynchronous failure is rethrown on every repeated wait. Reserved/skipped values that were never submitted remain invalid even after later work completes.
- Add or adjust stable common plumbing for exact `Device` identity and tensor-owner identity. A queue created by a `Device` accepts only views whose `&view.device()` is that exact `Device` instance; backend kind or backend ordinal is not a substitute. Owner identities and view metadata must remain stable long enough for asynchronous work, with derived-view metadata snapshotted rather than retaining caller view objects. Registration must retain every owner/storage touched by accepted work, deduplicate exact aliases, and release or quarantine it only after completion/failure proof. This plumbing must support later multi-input ADD without exposing a backend registry, global active-backend switch, or vendor type in common headers.
- Adapt `include/iom/detail/outstanding_work_registry.hpp` only as needed to represent the common registration/lifetime contract for the operation argument set; preserve safe cleanup/quarantine behavior and make pre-acceptance registration failure map to a negative result rather than leaving accepted work untracked.
- Keep default compute hooks (`add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`) as synchronous `Unsupported` results through the common facade. Do not retain the current throwing `unsupported(...)` public behavior.

## Implementation references

- **Modify:** `include/iom/iom.hpp:200-387` — `oid`/`DeviceOps` declarations, public operation signatures, protected hook boundary, submission helpers, and token state. The dedicated `include/iom/oid.hpp` type and helpers are supplied by blocker task 01; consume them rather than redefining them.
- **Modify:** `src/iom.cpp:488-630` — `DeviceOps` construction/destruction, queue-ID/token encoding, `wait`, completion, retained-failure, and skipped-sequence bookkeeping. Move the sequence width and validation to the blocker’s 55-bit contract and make invalid waits immediate.
- **Modify:** `include/iom/device.hpp:22-42` — preserve non-copyable/non-movable stable `Device` identity and add only the common identity plumbing required by the facades; do not add a registry or backend selector.
- **Modify:** `include/iom/tensor.hpp:147-222` and `src/iom.cpp:250-280` — preserve `TensorView`’s creating-device and owner relationships and provide the common layer the stable owner identity/metadata needed for validation and lifetime registration.
- **Modify as needed:** `include/iom/detail/outstanding_work_registry.hpp:326-680` — common registration, alias deduplication, completion release, and quarantine for all owners captured by an accepted operation.
- **Read:** `docs/BACKEND_CONTRACT.md:37-40,81-88,147-175,182-212` — backend-neutral ownership, queue, error, and device identity invariants to preserve.
- **Read:** `test/test_iom.cpp:767-980,1695-1735` and `test/backend/backend_conformance_common.hpp:209-299` — deterministic fake queues, current signatures, and core/fake seams. These tests are migrated by task 07, not this task.
- **Read:** `test/backend/backend_conformance_other.hpp:42-350` — repeatable waits, retained failures, foreign/zero token rejection, and lifetime scenarios that the new facade must continue to support. Backend conformance migration belongs to task 08.

## Requirements

- The public API is a clean cutover: remove `virtual` from all seven public OID-returning methods, mark each `noexcept`, and remove public exception-based unsupported behavior. Do not add aliases, compatibility shims, `add_support`, an equivalent capability query, an options object, a signature knob, or a backend-specific public fallback API.
- Protected backend hooks are the only operation extension point. They must be unable to bypass the common validation boundary, exact queue `Device` identity, owner identity/lifetime registration, token encoding, sequence exhaustion handling, or error mapping. Common code must not include vendor headers, select a backend by switch, or keep global backend state.
- Map the six synchronous categories exactly as specified in Scope. In particular, unsupported compute hooks return `Unsupported`; sequence exhaustion is `Overflow`; a pre-acceptance temporary/metadata/registration allocation failure is `ResourceExhausted`; and a backend/runtime failure before acceptance is `DeviceError`.
- Perform rejection before any output/storage effect and before accepting a token. A negative OID is terminal and never waitable. A positive OID is the only indication that work was accepted.
- Preserve queue ordering, caller serialization, stable caller-created owners and native handles, derived-view metadata snapshots, repeat waits, and repeatable retained asynchronous failures. Never expose a backend-owned registry or replace/relocate caller tensor storage as part of this cutover.
- Validate `wait` inputs as tokens from this queue that were actually submitted. Negative, zero, foreign, future, skipped, and unsubmitted values must throw `std::invalid_argument` immediately; valid tokens retain completion visibility and failure behavior on every call.
- Keep `Device` non-copyable/non-movable and compare exact object identity, including when two devices share a backend kind or ordinal. Owner identity must likewise distinguish separate tensor owners even if native handles or storage addresses happen to match.
- Ensure lifetime registration covers all input/output owners an operation can touch and deduplicates exact aliases. Registration must be complete before acceptance; completion and failure paths must release safely or quarantine when completion cannot be proven.

## Non-goals

- Do not implement ADD validation, broadcasting, arithmetic, numeric codecs, supported leaves, backend emulation, or any other ADD behavior; those belong to tasks 10–18.
- Do not migrate CPU, CUDA/ROCm, SYCL, or TTNN `copy` implementations or backend queue fakes; tasks 03–06 own those changes.
- Do not migrate or add core/shared tests, backend conformance/coexistence tests, or their CMake wiring; tasks 07–08 own test changes. This task only preserves the seams those focused fake tests exercise.
- Do not update `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, or public API documentation; task 09 owns documentation.
- Do not add a generic capability framework, public support query, backend registry/switch, global runtime state, CPU fallback backend, or new public storage/ownership abstraction.

## Acceptance criteria

- [ ] Compile-time inspection sees all seven exact existing view-based signatures as non-virtual `noexcept` member functions, and backend subclasses can customize only protected hooks; no public OID operation can throw synchronously.
- [ ] A deterministic common/fake hook that returns each of the six synchronous categories produces the exact negative `OidError` value, with no output effect and no accepted/waitable token. The default compute hooks produce `Unsupported` rather than throwing.
- [ ] A valid accepted submission returns a positive token with the blocker’s exact queue-ID/55-bit-sequence encoding. Sequence exhaustion returns `Overflow` before hook effects; all negative results are non-waitable.
- [ ] Facade validation rejects wrong `Device` instances, mismatched owner identities, malformed operation arguments, and invalid view relationships before backend dispatch and before lifetime registration is published as accepted work.
- [ ] `wait` immediately rejects negative, zero, foreign, future, skipped, and unsubmitted values with `std::invalid_argument`; a submitted token can be waited repeatedly, and a retained asynchronous failure is rethrown on every wait.
- [ ] Accepted work retains every participating owner through completion, deduplicates exact aliases, and safely releases or quarantines registration on success and failure without a backend registry or global backend selection mechanism.
- [ ] Focused fake/core seams in `test/test_iom.cpp` and `test/backend/backend_conformance_common.hpp`/`backend_conformance_other.hpp` remain the intended proof points for signatures, token sequencing, error timing, waits, retained failures, and ownership; their migration is left to tasks 07–08.

## Verification

- No gates are run while generating this frozen mini-specification.
- The implementing work should be checked later with the focused core/fake DeviceOps tests and compile-time signature checks in `test/test_iom.cpp`, `test/backend/backend_conformance_common.hpp`, and `test/backend/backend_conformance_other.hpp`; backend copy and full coexistence gates are intentionally deferred to tasks 03–08.

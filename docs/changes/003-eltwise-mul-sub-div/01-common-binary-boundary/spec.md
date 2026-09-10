# Common binary operation boundary

**Order:** 01
**Priority:** P0 — establishes the public contract and shared safety/lifetime boundary for every backend.
**Blocked by:** None
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Replace the ADD-specific common request and submission machinery, and the raw-view MUL path, with one validated operation-neutral binary boundary used by `add`, `mul`, `sub`, and `div`. The four exact three-view `noexcept` facades must preserve existing ADD behavior while making valid supported MUL/SUB/DIV requests positive asynchronous OIDs and enforcing all common validation, alias, ownership, queue, and failure rules before backend dispatch.

## Scope

- In `DeviceOps`, expose exactly `oid add(const TensorView&, const TensorView&, TensorView&) noexcept`, and identical three-view `mul`, `sub`, and `div`; carry an internal operation identity through immutable snapshots and backend hooks.
- Validate, in order, recognized specs/rank and nonzero dimensions/device identity/owner and native handle/view/storage bounds/checked arithmetic; exact leaf and quantization matching; right-aligned broadcasting and exact output shape; mapping snapshot; exact unbroadcasted in-place aliasing; then operation-specific support.
- Support ADD/MUL/SUB on the 21 numeric `QuantizationFormat::NONE` leaves `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`. Support DIV only on the nine floating leaves `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.
- Preserve `[1,1]` scalar convention, right-aligned singleton broadcasting, transformed leading-view offsets/strides, arbitrary read/read overlap, and the rule that only an exact same-owner, identical unbroadcasted mapping may alias output.
- Accepted work snapshots metadata, reserves ordered positive OIDs, registers three distinct owners with exact-alias deduplication, rolls registration back on rejected submission, and retains immutable completion failures. Pre-acceptance rejection has no output effect, owner registration, or sequence consumption; positive-token waits are repeatable and rethrow the same retained failure.

## Implementation references

- **Modify:** `include/iom/iom.hpp:268-405` — `AddViewSnapshot`, `AddRequest`, protected hooks, and `submit_add`; make declarations and helper names/records operation-neutral.
- **Modify:** `src/iom.cpp:659-788,844-931` — `snapshot_add_view`, `validate_add`, default hooks, and `add`/`mul` facades; centralize validation and dispatch for all four operations.
- **Modify:** `include/iom/detail/outstanding_work_registry.hpp:518-747` — ADD-only registration records/helpers; preserve deduplication, cleanup, and quarantine semantics under neutral names.
- **Tests:** `test/test_iom.cpp:779-953,1818-2262` — extend fake queues and common validation/lifetime/failure tests rather than creating a second binary-operation fixture.
- **Read:** existing `add` completion and token helpers in `src/iom.cpp` and `DeviceOps` private state in `include/iom/iom.hpp:401-429`; reuse their sequence and retained-failure behavior.

## Requirements

- No synchronous exception may cross any facade. Map malformed specs/device/shape/view/alias errors to `InvalidArgument`, checked arithmetic to `Overflow`, bounded pre-acceptance allocation failures to `ResourceExhausted`, backend failures to `DeviceError`, and otherwise to `InternalError`.
- Matching BOOL, F8_E8M0, or non-NONE quantization returns `Unsupported` only after all earlier validation. DIV additionally rejects matching integer leaves only after those checks. A recognized dtype mismatch is `InvalidArgument`, even if the operation would not support one leaf.
- Snapshot each view's spec, device/owner/native identity, plane offset/strides, aligned logical plane strides, and broadcast flags; never retain caller `TensorView` objects or allocate/replace caller operands/output.
- Treat exact aliases among the three owner registrations as one registry entry, but retain all three logical owner references and roll back every entry if queue submission rejects. Preserve positive OID ordering, repeat waits, retained post-acceptance failures, and queue usability after failure.
- Remove superseded ADD-only and raw-view MUL request/registration paths; do not retain aliases, a compatibility hook, public options, support query, promotion, public broadcast view, or public zero stride.

## Non-goals

- Do not implement scalar arithmetic or any CPU, CUDA, ROCm, SYCL, or TTNN execution.
- Do not alter copy, silu, linear, rmsnorm, sdpa, tensor ownership/allocation, or unrelated ADD numerical semantics.
- Do not add a generic operation registry, fallback framework, backend-kind switch, runtime dependency, or global active-backend state.
- Do not change existing target names or add a separate binary-operation executable.

## Acceptance criteria

- [ ] Compile-time tests pin all four exact three-view `noexcept` signatures; valid fake-queue ADD behavior remains compatible and valid MUL/SUB/DIV requests reach one neutral hook with operation identity.
- [ ] Fake/common tests demonstrate the prescribed validation precedence, shape/mapping snapshot, exact alias/read-overlap policy, no-effect negative OIDs, sequence behavior, owner deduplication/rollback, repeat waits, and retained failures for every operation.
- [ ] Matching unsupported leaves and integer DIV reject only after earlier mismatch/device/shape/view/overflow checks, with no token, output mutation, registration, or sequence consumption.
- [ ] Source review shows one neutral validator, snapshot, submission/owner-registration, and completion path; no raw-view MUL or parallel ADD-only registration system remains.

## Verification

- `cmake --build build --target iom_tests` — not run per assignment.
- `ctest --test-dir build -R '^iom_tests$' --output-on-failure` — not run per assignment.
- Focused fake-queue scenario: submit interleaved ADD/MUL/SUB/DIV, inspect positive OIDs and operation-tagged completion, then repeat waits and injected failures; not run per assignment.

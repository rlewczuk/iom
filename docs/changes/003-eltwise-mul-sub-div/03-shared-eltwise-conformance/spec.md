# Shared elementwise conformance harness

**Order:** 03
**Priority:** P1 — supplies reusable behavioral coverage before real backend execution is migrated.
**Blocked by:** `01-common-binary-boundary`, `02-scalar-binary-arithmetic`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Turn the historical ADD-only backend conformance helpers into one backend-neutral harness for ADD, MUL, SUB, and DIV. It must invoke an independent oracle through real or fake queue adapters, cover the shared mapping/error/lifetime contract once, and leave backend drivers responsible only for runtime setup and backend-specific fault seams.

## Scope

- Generalize `test/backend/backend_conformance_common.hpp:278-557`, `test/backend/backend_conformance_other.hpp:535-624`, and `test/backend/backend_conformance_add_gpu.hpp:23-331` to operation-neutral compile-time signatures, fixtures, scalar packing, diagnostics, capability cases, and reusable four-operation value cases.
- Cover all required domains: ADD/MUL/SUB on the 21 NONE leaves `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; DIV on the nine floating leaves `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.
- Exercise equal shapes, opposite-direction singleton axes, rank promotion and `[1,1]`, ranks 2/3/6/8/9/16/17, final tiled row/column tails, nested transformed leading views, nonzero transformed output offset/stride, padding/other planes, read overlap, exact aliases, and invalid partial/shifted/broadcast same-owner windows.
- Add representative U8 MUL, U8 SUB, and F32 DIV mappings and reversed noncommutative operands; test mismatch/error precedence, integer DIV rejection, operation identity, independent-oracle invocation, and hooks for pre-acceptance and post-acceptance fault/lifetime behavior.
- Keep driver files out of this task: no backend execution implementation or runtime initialization belongs here.

## Implementation references

- **Modify:** `test/backend/backend_conformance_common.hpp:278-557` — shared fake/real adapter concepts, mapping fixtures, and operation cases.
- **Modify:** `test/backend/backend_conformance_other.hpp:535-624` — non-GPU backend capability/error and lifecycle helpers.
- **Modify:** `test/backend/backend_conformance_add_gpu.hpp:23-331` — historical GPU helper; generalize metadata/fault/value scenarios rather than copy it per operation.
- **Read:** `test/backend/backend_conformance_add.hpp:14-200` — independent scalar oracle contract from task 02; call it without embedding production arithmetic.
- **Read:** `test/AGENTS.md:1-7` — preserve shared backend-neutral coverage, independent encodings/sentinels, allocator traffic, exact mismatch, lifetime/error checks, and sentence-style doctest names.

## Requirements

- Compile-time checks must accept the exact three-view `noexcept` methods for all four operations and reject no valid operation solely because a backend lacks an SDK dtype.
- For every operation, compare every addressed logical element against the independent oracle; require exact integer MUL/SUB results and floating finite/special/zero behavior under the one-ULP contract.
- Verify validation precedence: malformed/mismatched/device/shape/view/checked-overflow errors precede operation support; matching BOOL/F8_E8M0/non-NONE quantization is Unsupported; integer DIV is Unsupported only after earlier checks.
- Verify snapshots and lifetime hooks: output is unchanged on negative OIDs, accepted requests preserve caller storage/owners/handles, three owners deduplicate exact aliases, repeated waits retain failures, and cleanup occurs exactly once.
- Keep backend-specific runtime setup, queue submission, and arithmetic outside this harness; expose explicit hooks so each driver can inject resource/runtime and fence/execution failures.

## Non-goals

- Do not implement scalar arithmetic, backend traversal, queue policy, or runtime setup.
- Do not duplicate the ADD test framework, dtype matrix, oracle, mapping, or fault machinery for each new operation.
- Do not alter public validation or create a new executable/target; backend drivers continue to invoke this shared harness.

## Acceptance criteria

- [ ] All five backend drivers can instantiate one neutral harness and select any of ADD/MUL/SUB/DIV without operation-specific fixture copies.
- [ ] Shared tests visibly cover ranks, broadcast directions, tails, transformed output mapping, padding, read overlap, exact and forbidden aliases, mismatch precedence, U8 MUL/SUB, F32 DIV, and integer DIV rejection.
- [ ] Independent oracle comparisons, owner/lifetime checks, operation identity, repeated waits, retained failures, and pre/post fault hooks are reusable by every enabled backend.
- [ ] Existing ADD shared coverage remains represented and no real backend execution code is introduced.

## Verification

- `cmake --build build --target iom_backend_conformance_cpu_tests` — not run per assignment.
- `ctest --test-dir build -R 'iom_(backend_conformance_.*)_tests' --output-on-failure` — not run per assignment.
- Focused scenario: instantiate the fake adapter for four operations with rank-17 transformed/tail views, then assert oracle values, precedence, alias rejection, and repeat-failure hooks; not run per assignment.

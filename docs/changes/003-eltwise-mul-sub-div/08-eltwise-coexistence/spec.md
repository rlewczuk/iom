# Elementwise backend coexistence

**Order:** 08
**Priority:** P2 — provides cross-backend integration proof after every real backend path is operation-neutral.
**Blocked by:** `04-cpu-eltwise-operations`, `05-cuda-rocm-eltwise-operations`, `06-sycl-eltwise-operations`, `07-ttnn-eltwise-operations`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Extend the existing combined coexistence process and target so every enabled backend interleaves ADD, MUL, SUB, DIV, and COPY across at least two queues per participant. The test must attribute every result independently, prove ordering/lifetime/failure isolation and no global dispatch state, and retain all existing ADD/COPY checks.

## Scope

- Modify only `test/backend/test_backend_coexistence.cpp:894-1073` and `test/CMakeLists.txt:263-363` when minimal source registration/wiring is required; keep the existing `iom_backend_coexistence_tests` executable and test name.
- Use the independent oracle and shared operation cases for ADD/MUL/SUB/DIV, with exact integer MUL/SUB and one-ULP/special/zero floating comparisons. Exercise representative noncommutative SUB and DIV operands, U8 MUL/SUB, and F32 DIV.
- Run across every enabled CPU, CUDA, ROCm, SYCL, and TTNN participant with at least two queues each, preserving runtime setup in existing driver/backend adapters and old COPY/ADD assertions.
- Cover unique attributable output tokens, in-order and out-of-order submissions, repeat waits, retained failure isolation, exact device rejection, stable caller owners/handles/derived-view lifetime, exact aliases, read overlap, and no global active-backend/dispatch state.
- Use all required operation domains: ADD/MUL/SUB on `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; DIV on `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.

## Implementation references

- **Modify:** `test/backend/test_backend_coexistence.cpp:894-1073` — existing ADD/COPY combined-process scenarios; extend interleaving and attribution in place.
- **Modify if needed:** `test/CMakeLists.txt:263-363` — preserve conditional enabled-backend wiring and target name; no second executable.
- **Read:** `test/backend/backend_conformance_common.hpp:278-557` and `backend_conformance_other.hpp:535-624` — reuse operation-neutral fixtures/oracle from task 03.
- **Read:** `test/AGENTS.md:1-7` — retain independent encodings, sentinels, allocator traffic, and lifecycle/error checks.

## Requirements

- Submit mixed ADD/MUL/SUB/DIV/COPY sequences from two queues per participant and verify each output against an independent token-associated expected result; do not assume commutativity for SUB/DIV.
- Prove positive OID uniqueness/attribution, ordering and out-of-order behavior, repeatable waits, and failure retention/isolation: one accepted failed operation must not corrupt another queue's result or later work.
- Verify exact device identity rejection and pre-acceptance no-effect behavior, stable caller storage/owners/native handles and derived views, exact lhs/out/rhs/out/all-three alias behavior, valid read/read overlap, and transformed mapping.
- Keep old COPY and ADD coexistence coverage green and ensure no global backend/operation dispatch state is needed to pass.

## Non-goals

- Do not duplicate per-backend dtype matrices, add backend implementation, modify scalar arithmetic, or replace conformance targets.
- Do not create another executable, generic scheduler, global queue, fallback API, public options, or support query.
- Do not remove or weaken existing COPY/ADD checks.

## Acceptance criteria

- [ ] The existing coexistence target runs one combined process covering every enabled backend and interleaves all four arithmetic operations with COPY on at least two queues per participant.
- [ ] Independent token attribution, noncommutative arithmetic, mapping/alias/lifetime checks, ordering/repeat waits, device rejection, and retained-failure isolation are observable.
- [ ] Existing ADD/COPY behavior remains covered and no global dispatch state or duplicate backend matrix is introduced.

## Verification

- `cmake --build build --target iom_backend_coexistence_tests` — not run per assignment.
- `ctest --test-dir build -R '^iom_backend_coexistence_tests$' --output-on-failure` — not run per assignment.
- Focused scenario with every enabled backend: interleave two-queue ADD/MUL/SUB/DIV/COPY, inject one retained failure, repeat its wait, and verify unrelated tokens, owners, and later work; not run per assignment.

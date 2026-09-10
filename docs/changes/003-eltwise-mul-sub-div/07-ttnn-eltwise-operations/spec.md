# TTNN elementwise operations

**Order:** 07
**Priority:** P1 — extends TTNN's native per-plane carrier and staging lifecycle to the four-operation contract.
**Blocked by:** `01-common-binary-boundary`, `02-scalar-binary-arithmetic`, `03-shared-eltwise-conformance`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Reuse TTNN's native per-plane carrier, API mutex, host staging/cache, mapping, queue, owner, and cleanup path for ADD, MUL, SUB, and floating DIV. Preserve TTNN's existing storage span (BOOL plus the 21 numeric leaves), caller handles/owners, transformed views, aliases, tails, and retained failures without introducing generic fallback storage.

## Scope

- Generalize `src/ttnn/device.cpp:513-648` task/carrier/mutex/queue/lifetime state and `src/ttnn/copy.cpp:421-635` ADD plane traversal, staging, mapping, and completion to carry operation identity once.
- Touch `src/ttnn/staging.hpp` or `src/ttnn/registry_state.hpp` only if the existing lifecycle seam must be generalized; preserve their current ownership and cleanup design.
- Execute ADD/MUL/SUB for `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; execute DIV for `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`. BOOL remains in TTNN's storage span but is Unsupported for arithmetic; F8_E8M0 and non-NONE quantization are Unsupported.
- Reuse native per-plane mapping and host staging for broadcast, nested transformed leading inputs, nonzero output offset/stride, row/column tails, padding/other planes, exact aliases, and read/read overlap.
- Update `test/ttnn/test_ttnn_conformance.cpp:1450-2062` for real TTNN runtime setup, all four operations, and operation-specific resource/API fault seams.

## Implementation references

- **Modify:** `src/ttnn/device.cpp:513-648` — native per-plane task/carrier, API mutex, queue, and lifecycle path.
- **Modify:** `src/ttnn/copy.cpp:421-635` — ADD-only operation traversal and completion.
- **Modify if needed:** `src/ttnn/staging.hpp`, `src/ttnn/registry_state.hpp` — only to carry neutral operation metadata through existing staging/registry seams.
- **Modify:** `test/ttnn/test_ttnn_conformance.cpp:1450-2062` — driver and conformance cases.
- **Read:** `src/shared/scalar_add.hpp:11-113` and shared harness from task 03 for codec/oracle and operation cases.

## Requirements

- Select arithmetic once per request/task and retain one per-plane traversal/staging/completion structure; do not duplicate ADD/MUL/SUB/DIV mapping or allocate caller operands/output.
- Preserve TTNN caller native handles and stable owners, API mutex ordering, host staging/cache bounds, queue visibility, exact alias load-before-store, three-owner deduplication, cleanup exactly once, positive OIDs, repeat waits, retained failures, and later queue usability.
- Perform all common validation before operation support: matching BOOL/F8_E8M0/non-NONE returns Unsupported; integer DIV returns Unsupported; mismatches, wrong devices, shapes, malformed views, aliases, and overflow retain earlier errors and no effects.
- Validate every addressed logical element against the independent oracle, with exact integer MUL/SUB and one-ULP floating behavior plus special-value and zero-sign rules. Cover ranks 2/3/6/8/9/16/17, tails, transformations, padding, aliases, and U8/F32 representative cases.
- Keep TTNN's existing storage/staging boundary and supported data-type span; no native SDK dtype rejection may narrow required numeric operations.

## Non-goals

- Do not redesign TTNN storage, add generic fallback or allocation APIs, add a device kernel, or change public signatures/capability queries.
- Do not touch CPU, CUDA, ROCm, or SYCL implementations.
- Do not alter unrelated TTNN operations, common validation, or scalar format policy.

## Acceptance criteria

- [ ] TTNN real-queue conformance executes all required ADD/MUL/SUB leaves and floating DIV leaves, while preserving BOOL storage behavior and rejecting unsupported arithmetic only under the common contract.
- [ ] Native per-plane mapping, staging/cache, API mutex, handles/owners, aliases, tails/transforms, repeat waits, retained failures, and post-failure queue use are covered.
- [ ] Source review finds one TTNN operation-neutral traversal/lifecycle path and no generic fallback or storage redesign.
- [ ] Existing TTNN ADD conformance remains compatible.

## Verification

- `cmake --build build --target iom_ttnn_conformance_tests` — not run per assignment.
- `ctest --test-dir build -R '^iom_ttnn_conformance_tests$' --output-on-failure` — not run per assignment.
- Focused TTNN scenario on configured hardware: interleave four operations with exact and transformed aliases, inject an API/cleanup failure, repeat waits, and verify stable handles/owners and later queue work; not run per assignment.

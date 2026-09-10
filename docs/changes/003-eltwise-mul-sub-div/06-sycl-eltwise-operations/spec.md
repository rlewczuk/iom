# SYCL elementwise operations

**Order:** 06
**Priority:** P1 — extends the existing host-staged in-order SYCL path without introducing a device kernel or storage redesign.
**Blocked by:** `01-common-binary-boundary`, `02-scalar-binary-arithmetic`, `03-shared-eltwise-conformance`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Generalize the existing SYCL validated request, bounded staging, host traversal, in-order queue, fence, and cleanup path for ADD, MUL, SUB, and floating DIV. Operation selection must happen outside logical loops while preserving queue visibility/order, caller storage boundaries, owner lifetime, retained failures, and the complete supported dtype contract.

## Scope

- Update `src/sycl/copy.cpp:161-234,399-402,481-760` to carry neutral operation metadata through staging/task/fence/cleanup and execute all four operations via the shared scalar path.
- Touch `src/sycl/add.cpp` only if the existing operation entry seam must be renamed or generalized; do not add a device kernel.
- Support ADD/MUL/SUB on exactly `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; support DIV on exactly `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.
- Preserve storage/staging boundaries, bounded pools, host encoding/decoding, transformed mappings, broadcast/tails/padding, exact alias and read-overlap rules, no caller storage replacement, ordered OIDs, repeat waits, and retained post-acceptance failures.
- Update `test/sycl/test_sycl_conformance.cpp:679-707` for runtime setup, four-operation shared cases, and operation-specific resource/fence fault seams.

## Implementation references

- **Modify:** `src/sycl/copy.cpp:161-234` — task/staging/fence state and ownership cleanup.
- **Modify:** `src/sycl/copy.cpp:399-402,481-760` — ADD-only dispatch/traversal and operation entry path.
- **Modify if needed:** `src/sycl/add.cpp` — existing SYCL ADD seam only where required for the neutral operation boundary.
- **Modify:** `test/sycl/test_sycl_conformance.cpp:679-707` — queue setup and conformance driver.
- **Read:** `src/shared/scalar_add.hpp:11-113` and shared harness from task 03; reuse codec/oracle contracts.

## Requirements

- Select the operation once per task/request, never switch per logical element and never duplicate staging, mapping, or cleanup structures.
- Ensure all required leaves execute through host staging even when no device-native dtype exists; matching BOOL/F8_E8M0/non-NONE and integer DIV remain common Unsupported after earlier validation.
- Prepare metadata and bounded staging resources before positive OID acceptance. Preserve no-effect negative errors, three-owner registration/deduplication, exact alias load-before-store, cleanup once, repeatable waits, and queue usability after fence/execution failure.
- Compare every addressed output element to the independent oracle: integer MUL/SUB exact modulo-$2^w$; floating DIV and other operations satisfy one-ULP and special-value/sign rules.
- Exercise ranks 2/3/6/8/9/16/17, both broadcast directions, `[1,1]`, transformed leading/output views, tails, padding, aliases, read overlap, and representative U8 MUL/SUB/F32 DIV through the real queue.

## Non-goals

- Do not add a new device kernel, generic fallback framework, storage redesign, public capability query, or mixed-type promotion.
- Do not touch CPU, CUDA, ROCm, or TTNN code.
- Do not alter public validation/order/error contracts or unrelated SYCL operations.

## Acceptance criteria

- [ ] SYCL real-queue conformance returns correct values and positive OIDs for all required ADD/MUL/SUB leaves and floating DIV leaves; valid MUL no longer reports Unsupported.
- [ ] Bounded staging, queue visibility/order, transformed/tail mapping, aliases, owner lifetime, pre/post fault retention, repeat waits, and no caller-storage replacement remain observable.
- [ ] Source review finds one host-staged traversal/task/fence/cleanup structure with operation selected outside loops and no new kernel.
- [ ] Existing SYCL ADD behavior remains compatible.

## Verification

- `cmake --build build --target iom_sycl_conformance_tests` — not run per assignment.
- `ctest --test-dir build -R '^iom_sycl_conformance_tests$' --output-on-failure` — not run per assignment.
- Focused SYCL scenario: submit interleaved operations on two in-order queues, inject a fence failure, repeat the failed wait, then submit later work and verify staging/owner cleanup; not run per assignment.

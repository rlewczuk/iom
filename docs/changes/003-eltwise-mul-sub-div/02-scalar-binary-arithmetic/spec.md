# Scalar binary arithmetic and oracle

**Order:** 02
**Priority:** P0 — defines the single production scalar codec and independent expected values required by all backends.
**Blocked by:** None
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Generalize the existing scalar ADD codec into one operation-specialized implementation that decodes and encodes each supported leaf once while providing ADD, MUL, SUB, and DIV arithmetic. Add an independent oracle and direct scalar tests that make integer wraparound, floating special values, operand order, and one-encoding-step behavior observable without duplicating codec or test frameworks.

## Scope

- Refactor `src/shared/scalar_add.hpp:11-113` (the historical filename may remain) so format definitions, decode, RNE encoding, special handling, saturation, signed zero, and gradual underflow are shared; select arithmetic through operation-specialized functors/templates rather than a per-element operation switch.
- Implement ADD/MUL/SUB for exactly the 21 NONE numeric leaves `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; implement DIV for exactly the nine floating leaves `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`.
- Integer MUL and SUB must return the low width bits modulo $2^w$ without signed-overflow UB or saturation. DIV has no integer implementation.
- Floating arithmetic decodes both raw operands in the destination format, computes in the extended/IEEE domain, then encodes once with round-to-nearest ties-to-even, gradual underflow, and the existing format policies. Preserve NaN/infinity/zero rules: MUL zero-times-infinity is NaN; SUB same-sign infinities are NaN; DIV zero/zero and infinity/infinity are NaN, signed-zero and infinity sign is XOR, and finite quotients use exact division before encoding.
- The oracle in `test/backend/backend_conformance_add.hpp:14-200` must be independent from production codec and kernel code while sharing only neutral packing/classification utilities as appropriate. Generalize `test/backend/test_scalar_add.cpp:7-27`; adjust `test/CMakeLists.txt:1-17` only to register the existing `iom_scalar_add_tests` source/target.

## Implementation references

- **Modify:** `src/shared/scalar_add.hpp:11-113` — `scalar_add_detail::Format`, `format`, `decode_small`, `encode_small`, and `scalar_add`; retain one format codec and expose operation-specialized entry points.
- **Modify:** `test/backend/backend_conformance_add.hpp:14-200` — existing independent expected-value/packing helpers; rename/generalize historical ADD-only interfaces without cloning them for three operations.
- **Modify:** `test/backend/test_scalar_add.cpp:7-27` — direct doctest cases; use sentence-style names required by `test/AGENTS.md`.
- **Modify if needed:** `test/CMakeLists.txt:1-17` — keep `iom_scalar_add_tests` and its `add_test` registration unless source registration must include a neutralized header.
- **Read:** `test/AGENTS.md:1-7`; preserve independent host encodings, sentinels, exact diagnostics, and focused test naming.

## Requirements

- Exhaustively test all ordered raw pairs for I2/U2/I4/U4 MUL and SUB, including unequal reversed pairs. Cover every applicable compact floating operation with ordered/reversed operands, and verify exact raw scalar results through the production codec boundary.
- Add wide integer vectors covering zero, signed/unsigned extrema, high bits, modulo products, and underflowing differences. Verify exact low-width results and no saturation.
- Add wide floating vectors for ordinary RNE boundaries, subnormals, overflow/underflow, both zero signs, infinities, NaNs, zero-times-infinity, same-sign infinity subtraction, signed-zero division, zero/zero, finite/infinity, and infinity/infinity.
- The production scalar boundary must match the independently encoded reference raw bits exactly after its single decode/operation/encode path; the oracle's one-adjacent-ULP comparison helper is for downstream backend conformance only. Require the correct NaN class, infinity sign, and reference zero sign.
- Keep ADD results and format-specific policies unchanged, including finite-only saturation for F4/F6/F8_E4M3FN and NaN/infinity handling for F8_E5M2/F16/BF16/F32/F64.
- Do not make tests depend on backend SDK dtypes, native arithmetic, or a copied production decode/encode implementation.

## Non-goals

- Do not change common validation, public APIs, queue submission, owner registration, or any backend traversal.
- Do not add integer DIV policy, mixed-type promotion, casts, options, fallback arithmetic, or a second scalar codec.
- Do not rename the target unless mechanically unavoidable; historical `add` filenames may remain when their contents are the sole neutral implementation.

## Acceptance criteria

- [ ] One shared decode/encode implementation serves all four operation-specialized arithmetic paths; source review finds no copied codec, format mapper, or per-element operation-selection loop.
- [ ] Direct tests pass for exhaustive compact integer pairs, reversed noncommutative operands, wide integer wraparound, and the specified floating boundary/special vectors against an independent oracle.
- [ ] ADD scalar outputs remain unchanged, while MUL/SUB exact integer outputs and DIV floating outputs obey the stated operation and encoding rules.
- [ ] Integer DIV is absent from scalar execution and cannot be mistaken for a supported integer result.

## Verification

- `cmake --build build --target iom_scalar_add_tests` — not run per assignment.
- `ctest --test-dir build -R '^iom_scalar_add_tests$' --output-on-failure` — not run per assignment.
- Focused scenario: compare every ordered compact raw pair and representative reversed SUB/DIV vectors with independently encoded expected bits; not run per assignment.

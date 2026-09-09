# Implement the independent scalar ADD reference boundary

**Order:** 11
**Priority:** P0 — this is the production scalar arithmetic and independent reference needed before any backend execution can claim ADD conformance.
**Blocked by:** `09-document-oid-contract`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

Add a backend-neutral internal scalar ADD boundary under `src/shared` and a deliberately independent test-only scalar oracle under `test/backend`. The production boundary must return the exact encoded result for one pair of same-type raw elements across all 21 required unquantized numeric leaves; the focused scalar proof must compare it against an independently implemented reference, without reusing production conversion or arithmetic code.

## Scope

- **Planned production boundary:** `src/shared/scalar_add.hpp`, with a concrete internal function that accepts a `DataType` and two raw element encodings and returns one raw encoding of the same type. It is internal shared machinery, not a public API; place its implementation in this header or the existing shared translation unit selected by the build.
- Implement exact integer addition for `I2/U2/I4/U4/I8/U8/I16/U16/I32/U32/I64/U64`: signed values use two's-complement interpretation, unsigned values use ordinary binary interpretation, and the result is the low `w` bits of the exact sum modulo `2^w`. Do not invoke signed-overflow behavior.
- Implement the scalar floating reference for `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. Decode the named format, compute the exact real sum, and encode exactly once with round-to-nearest, ties-to-even (RNE); do not perform an intermediate same-format rounding.
- Use the OCP Microscaling Formats v1.0 scalar-element tables for the F4/F6 formats, including subnormals, RNE, and saturation to signed maximum finite. `F8_E4M3FN` uses saturation and reference NaN `0x7f`; `F8_E5M2` uses OCP overflow/infinity behavior and reference NaN `0x7e`.
- Treat `F16`, `BF16`, `F32`, and `F64` as their named IEEE encodings: gradual underflow with no FTZ/DAZ; exact zero-sign rules (`-0 + -0` is `-0`, opposite zero signs and exact nonzero cancellation are `+0`, and a nonzero value rounded to zero keeps its mathematical sign); NaN or opposite infinities produce a NaN; otherwise an infinity operand determines the result. Use canonical reference NaNs F16 `0x7e00`, BF16 `0x7fc0`, F32 `0x7fc00000`, and F64 `0x7ff8000000000000`.
- **Planned independent oracle:** `test/backend/backend_conformance_add.hpp`. It must independently decode raw encodings, compute the exact reference, and encode expected raw values. It must not include, call, copy, or mirror production conversion/arithmetic helpers; an oracle-independence review must be possible from the test-only source.
- **Planned focused registration:** add `test/backend/test_scalar_add.cpp` to the dedicated scalar test target `iom_scalar_add_tests`. The target must link the normal test framework and library and be registered as a test; it must not require a backend device or queue.

## Implementation references

- **Modify (planned):** `src/shared/scalar_add.hpp` — internal scalar ADD boundary and format-dispatch declarations/definitions; keep it backend-neutral and usable by all five backends.
- **Add (planned):** `src/shared/scalar_add.cpp` or the existing shared implementation unit selected by the build — production raw-element arithmetic and encoding implementation if the project does not keep this boundary header-only.
- **Add (planned):** `test/backend/backend_conformance_add.hpp` — the sole independent scalar oracle and reusable raw-pair/reference helpers for this focused proof.
- **Add (planned):** `test/backend/test_scalar_add.cpp` — focused exhaustive and boundary tests for the production boundary against the independent oracle.
- **Modify (planned):** `test/CMakeLists.txt` — register `test/backend/test_scalar_add.cpp` in the dedicated `iom_scalar_add_tests` target and register that target as a test.
- **Read:** `src/shared/standard_tiled_copy.hpp` — `kStandardSupportedDataTypes`; reuse the existing `DataType` ordering and exact 21-leaf distinction without changing storage capability declarations.
- **Read:** `test/backend/backend_conformance_common.hpp` — existing backend-neutral raw bit-width and host-encoding conventions; the new oracle must remain independent rather than calling its production arithmetic.

## Requirements

- Cover exactly these 21 `QuantizationFormat::NONE` leaves and no other data type: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. `BOOL`, `F8_E8M0`, quantized formats, and promotion are outside this boundary.
- The production result must be the exact scalar reference encoding, not merely the later backend allowance of an adjacent finite encoding. The focused scalar tests therefore require exact raw equality to the independent oracle for every tested pair.
- Define floating ULP adjacency in the oracle for downstream use as one adjacency step among ordered finite representable values, with `-0` and `+0` equivalent. A reference NaN requires only a NaN result in backend tests; a reference infinity requires the same-sign infinity; a reference zero accepts either zero sign under the general envelope. This task's scalar boundary itself must still match the canonical reference encoding exactly.
- Exhaustively test every ordered raw pair for all encodings of `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, and `F8_E5M2`, plus every ordered raw pair for `I2`, `U2`, `I4`, and `U4`. Include representative boundary vectors for `I8/U8/I16/U16/I32/U32/I64/U64`, `F16`, `BF16`, `F32`, and `F64`, including zero signs, smallest subnormals, normal/subnormal transitions, ties, largest finite values, overflow thresholds, infinities, NaNs, and cancellation.
- The independent oracle must test both finite and special-value behavior: OCP F4/F6 saturation and subnormals; E4M3FN saturation and `0x7f` reference NaN; E5M2 infinity overflow and `0x7e` reference NaN; IEEE-like NaN/opposite-infinity behavior; the four canonical IEEE/BF16 NaNs; gradual underflow; and exact zero-sign rules.
- Keep the scalar boundary deterministic and portable in every host/device compilation path that consumes `src/shared`: use fixed-width representations and explicit bit manipulation, avoid architecture-endian assumptions, signed overflow, exceptions, allocation, global mutable state, host-only runtime facilities, dependence on the ambient floating-point rounding mode, and FTZ/DAZ or fast-math behavior. It must contain no vendor types, vendor runtime calls, backend-kind switch, queue access, or device-global state.
- Review the oracle for independence: production conversion, production arithmetic, production lookup tables, and production encoders must not be transitively included or reused by `backend_conformance_add.hpp`. The oracle may share only public type names and format specifications, not implementation machinery.
- Preserve the later ADD contract: this scalar work computes one element only, does not allocate or replace caller storage, and does not decide validation, broadcasting, aliasing, queue ordering, token handling, asynchronous failures, backend fallback, or device execution policy.

## Non-goals

- Do not add or change the public `DeviceOps::add` facade, OID behavior, capability query, `add_support`, or any public header/API.
- Do not implement CPU, CUDA, ROCm, SYCL, or TTNN kernels, queues, transfers, storage expansion, staging, fallback paths, or backend conformance drivers.
- Do not implement tensor shape validation, rank promotion, multidirectional broadcasting, transformed views, alias checks, owner/lifetime tracking, output allocation, or caller-storage writes.
- Do not add quantization codecs, mixed-type promotion, casts, options, or a generic numeric codec/fallback subsystem.
- Do not make the oracle a production dependency or use the production scalar boundary as the oracle's reference implementation.

## Acceptance criteria

- [ ] The planned shared scalar boundary returns exact reference raw encodings for all 21 required leaves, including integer modulo-low-`w` behavior and all specified OCP/IEEE/BF16 floating semantics.
- [ ] `iom_scalar_add_tests` is registered and runs without a backend device or queue; its exhaustive cases cover all ordered pairs for every F4/F6/FP8 leaf and I2/U2/I4/U4.
- [ ] The focused proof includes representative wider-leaf boundaries and visibly checks subnormals, RNE ties, signed zero, saturation/overflow, infinity, NaN classes, cancellation, and each listed canonical reference NaN.
- [ ] Production scalar output compares exactly with the independently encoded oracle output; the oracle source has no production arithmetic/conversion dependency, and the independence review is recorded in the implementation/test review.
- [ ] Shared code is portable across required host/device compilation paths and does not rely on vendor types, ambient rounding mode, FTZ/DAZ, fast-math, signed overflow, allocation, exceptions, or mutable global state.
- [ ] No public API, queue/broadcast/alias behavior, backend kernel, storage/transfer behavior, quantization codec, or promotion behavior is introduced by this task.

## Verification

- `ctest --test-dir <build-dir> -R '^iom_scalar_add_tests$'` (not run here; the project-wide owner runs gates after all frozen tasks land).
- Review `test/backend/backend_conformance_add.hpp` and its include graph to confirm the oracle does not reuse production arithmetic, conversion, tables, or encoders (review only; no gate run here).

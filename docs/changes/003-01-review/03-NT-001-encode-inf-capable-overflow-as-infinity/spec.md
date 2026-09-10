# Encode inf-capable finite overflow as infinity

**Order:** 03
**Priority:** P0 — corrects wrong inference values and independent oracle
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** NT-001
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** verified, confidence 98
**Review scope:** whole-codebase
**Backend scope:** CPU, CUDA, ROCm, SYCL, and TTNN through the shared host/device scalar codecs and independent test oracle
**Location:** `src/shared/scalar_add.hpp` — `scalar_add_detail::encode_small`; `src/shared/standard_tiled_add.inl` — `add_encode`/`add_value`; `test/backend/backend_conformance_add.hpp` — `add_oracle::round_encode`/`binary`; TTNN F64 comparison exception

## Outcome

Finite arithmetic results beyond the RNE overflow threshold encode as same-sign infinity for F8_E5M2, F16, BF16, F32, and F64 on every enabled backend. Finite-only F4/F6/F8_E4M3FN formats retain maximum-finite saturation, while NaN, signed zero, exact infinity, gradual underflow, and one-ULP finite behavior remain unchanged. The independent oracle enforces the same policy without reusing production code.

## Current problem

The numerical invariant in `docs/BACKEND_CONTRACT.md` and the 003 operation contract says the five infinity-capable formats encode finite overflow as infinity; only finite-only formats saturate. `encode_small` computes `finite_emax = emask - 1` for `f.infs` formats and returns `pack(finite_emax, fmask)` both when exponent range is exceeded and when RNE carries into the exponent. `add_encode` repeats this policy, and GPU `add_value` additionally clamps finite-operand F64 MUL/SUB/DIV device-double overflow to maximum finite. The independent `add_oracle::round_encode` mirrors saturation, while TTNN explicitly permits F64 ADD saturation. A focused private codec/public CPU probe reproduced max-finite*2 as saturation for F8_E5M2/F16/BF16/F32/F64; root CPU build/core tests and remote backend smoke/conformance/coexistence passed, but no candidate-specific accelerator overflow vectors ran.

## Scope

- Correct one format-aware production overflow policy for all four operations and all five infinity-capable formats, including ordinary exponent overflow and RNE carry overflow.
- Remove GPU F64 finite-overflow clamps so the device path reaches the corrected infinity encoding policy without destination-width intermediate rounding.
- Independently correct the test oracle and remove TTNN's F64 ADD saturation allowance; retain finite-only saturation as a negative boundary.

## Implementation references

- **Modify:** `src/shared/scalar_add.hpp` — `encode_small`, arithmetic/`scalar_binary`; own the host overflow decision and preserve long-double extended-domain computation.
- **Modify:** `src/shared/standard_tiled_add.inl` — `add_encode` and `add_value`; mirror the corrected policy in device-callable arithmetic and remove F64 overflow-to-max-finite branches.
- **Modify:** `test/backend/backend_conformance_add.hpp` — `add_oracle::round_encode`/`binary`; remain an independent implementation of the normative policy.
- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — F64 ADD comparison around the current saturation allowance; require infinity.
- **Read:** `test/backend/test_scalar_add.cpp` and backend conformance drivers — extend existing value conventions rather than creating another oracle.

## Requirements

- For `f.infs` formats, encode finite values outside the RNE representable range and upper-boundary RNE carry as signed infinity; for finite-only formats retain same-sign maximum finite.
- Exercise all applicable ADD/MUL/SUB/DIV operations with max-finite*2, max+max, max-(-max), max/0.5 where valid, sign-reversed variants, and threshold-adjacent RNE values. Preserve NaN, signed zero, exact infinity, gradual underflow, integer modulo, operation eligibility, and one final encoding step.
- Correct the independent oracle separately. It must not call production codec functions or mask a maximum-finite mutation.

## Non-goals

- Do not alter finite-only saturation, NaN payload canonicalization, integer arithmetic, operation domains, storage layouts, or ordinary finite tolerance policy.
- Do not introduce a second codec, backend-specific overflow exception, or bitwise-equality requirement for ordinary finite results.

## Acceptance criteria

- [ ] CPU scalar and real-queue outputs for F8_E5M2/F16/BF16/F32/F64 finite overflow are same-sign infinity, including RNE-carry boundaries; the old max-finite result fails.
- [ ] Enabled CUDA, ROCm, SYCL, and TTNN paths agree with the corrected independent oracle for all applicable overflow operations; F4/F6/F8_E4M3FN still saturate.
- [ ] NaN, signed zero, exact infinity, gradual underflow, and adjacent finite one-ULP cases retain their required classes and signs.
- [ ] A temporary production saturation mutation is rejected by the independent oracle/conformance checks.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_scalar_add_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^(iom_scalar_add_tests|iom_backend_conformance_cpu_tests)$'` — run with the focused overflow and RNE-boundary vectors.
- On configured CUDA, ROCm, SYCL, and TTNN hosts, run each enabled backend conformance target with real-queue max-finite overflow, sign-reversed, threshold-adjacent, finite-only, and special-value scenarios; apply a temporary saturation mutation and require failure.
- Actual evidence is the focused scalar/public CPU reproduction and supplied root CPU/remote smoke/conformance results. No accelerator overflow vector, candidate-specific mutation, sanitizer, or profiler has run; hardware vector execution remains a gap.

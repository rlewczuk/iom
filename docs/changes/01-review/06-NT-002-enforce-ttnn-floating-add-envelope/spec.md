# Enforce the TTNN floating ADD comparison envelope

**Order:** 06
**Priority:** P1 — vacuous floating conformance
**Blocked by:** None
**Review source:** `cpp-inference-numerical-testing` — `whole-codebase checked-out main/HEAD ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb ("Update remote hosts"), clean working tree at review start`
**Finding:** NT-002
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** verified, confidence 99
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `test/ttnn/test_ttnn_conformance.cpp:1427-1487` — `require_add_leaf_matches_oracle`

## Outcome

TTNN ADD conformance rejects every integer mismatch, wrong floating special-value class, and finite result more than one representable step from the independent oracle, while accepting an exact result or a finite result within one ULP. The test therefore reports the actual floating arithmetic contract instead of converting every floating mismatch into a passing assertion.

## Current problem

The conformance helper computes an independent `add_oracle::add` result, but when a floating result differs it emits `MESSAGE("floating leaf within envelope")` followed by `CHECK(true)` at `test/ttnn/test_ttnn_conformance.cpp:1482-1484`. That branch neither compares the observed and expected encodings nor distinguishes a one-ULP result from an arbitrary value, a wrong infinity or NaN class, or a signed-zero change. The integer branch is strict, so the same helper gives TTNN floating leaves a vacuous pass condition and can hide incorrect device arithmetic.

## Scope

- Replace the floating mismatch branch in `require_add_leaf_matches_oracle` with one type-aware comparison policy shared by all floating leaves exercised by the TTNN ADD conformance driver.
- Keep integer leaves exact and continue comparing every logical element against the independent oracle; do not move arithmetic responsibility into production TTNN code or use `detail::scalar_add` as the test oracle.
- Make the comparison policy cover F16, BF16, F32, and F64 raw encodings, including finite values, NaNs, infinities, and both signed zeros.

## Implementation references

- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — `require_add_leaf_matches_oracle` and its local comparison helpers; this is the sole TTNN ADD value assertion currently containing the vacuous floating branch.
- **Read:** `test/backend/backend_conformance_add.hpp` — `add_oracle::add` and existing independent encoding conventions; preserve the test-oracle boundary rather than deriving expected values from a production arithmetic helper.
- **Read:** `src/shared/scalar_add.hpp` — production scalar arithmetic only as a negative reference; it must not become the conformance oracle.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — `TTNN ADD exhaustively covers every compact leaf pair` and `TTNN ADD representative wide leaves match the oracle`, run on the configured TTNN device.

## Requirements

- Delete the `MESSAGE("floating leaf within envelope")` / `CHECK(true)` mismatch path. For every integer leaf, require exact raw equality with the oracle.
- For floating values, first classify the observed and expected raw encodings by the exact leaf type. NaN must match NaN, infinity must match infinity with the same sign, and zero must match zero with the same sign; any different special class fails.
- For finite nonzero values of the same type, map the type-width raw bits to a monotonic signed-aware integer order and require an absolute raw-order distance of at most one. The mapping must use the sign bit for the selected type width, must not compare F16/BF16/F32/F64 through a common widened host value, and must not make `-0` and `+0` equivalent.
- A one-ULP finite perturbation must pass, while a two-ULP perturbation, a finite-versus-infinity/NaN mismatch, an infinity-sign mismatch, and a signed-zero mismatch must fail. Diagnostics must identify the type, element, expected bits, observed bits, and comparison class.
- Keep expected values sourced from `iom_conformance::add_oracle::add(type, lhs, rhs)` and keep the oracle independent of the production TTNN path. Do not call `detail::scalar_add` or any backend implementation to produce expected values.
- Exercise the policy with deterministic positive and negative comparator cases or an equivalent test-only mutation seam so that the negative cases cannot be silently removed while the hardware output happens to be exact.

## Non-goals

- Do not alter TTNN ADD arithmetic, dtype conversion, storage, staging, queue behavior, or production error handling.
- Do not change the integer arithmetic contract, compact-pair generation, wide-value vectors, broadcast cases, or mapping tests except where they invoke the corrected comparison.
- Do not broaden the floating tolerance beyond one ULP or accept mismatched special-value classes for convenience.

## Acceptance criteria

- [ ] Every integer mismatch fails; exact integer output passes for every integer leaf covered by the TTNN ADD driver.
- [ ] Exact floating output and a finite one-ULP perturbation pass, while a finite two-ULP perturbation fails with a value-level diagnostic.
- [ ] NaN, infinity sign, and signed-zero classes are compared exactly; finite output is never accepted in place of a special value or vice versa.
- [ ] The TTNN hardware conformance cases compare every observed logical element against the independent oracle and no floating mismatch executes an unconditional passing assertion.
- [ ] No production scalar-add implementation is referenced as the expected-value oracle, and the corrected policy does not alter TTNN execution or caller-visible behavior.

## Verification

- `cmake --build <configured TTNN build> --target iom_ttnn_conformance_tests` — proposed focused build; not run for this specification-writing task.
- **Remote TTNN hardware:** use `.agents/skills/remote-development/scripts/remote-sync ttnn <unique-task-id>`, then `.agents/skills/remote-development/scripts/remote-exec ttnn <unique-task-id> 'cmake --build <configured-build> --target iom_ttnn_conformance_tests && ctest --test-dir <configured-build> --output-on-failure -R "^iom_ttnn_conformance_tests$"'`; expected observation is a passing TTNN conformance run with the compact and wide ADD cases performing value comparisons.
- Run the focused comparator positive/negative scenarios on the configured TTNN host; expected observations are pass for exact and one-ULP finite values and failure for two-ULP, wrong-special-class, infinity-sign, and signed-zero mutations. No local accelerator result substitutes for this remote check.

# Check CPU and SYCL ADD output values through the real queues

**Order:** 05
**Priority:** P1 — unprotected arithmetic contract
**Blocked by:** None
**Review source:** `cpp-inference-numerical-testing` — whole-codebase review of checked-out main HEAD `ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb` (`Update remote hosts`), clean tree at review start
**Finding:** NT-001
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** strongly-supported, confidence 94
**Review scope:** whole-codebase
**Backend scope:** multi-backend (`cpu`, `sycl`), shared conformance infrastructure
**Location:** `test/cpu/test_cpu.cpp:1361-1402`; `test/cpu/test_cpu_conformance.cpp:241-270`; `test/sycl/test_sycl_conformance.cpp:714-718`; `test/backend/backend_conformance_add.hpp:1-37` (`add_oracle::add`); `src/cpu/device.cpp:678-730`; `src/sycl/copy.cpp:537-607`

## Outcome

CPU and SYCL conformance submit value-bearing ADD requests through each backend's real queue, wait for completion, read back candidate logical bytes, and compare them with one independent ADD oracle. U8 uses exact modulo arithmetic; F32 checks canonical special classes and finite results within one ULP, so token/lifetime success cannot mask an arithmetic, packed-bit, broadcast, or coordinate regression.

## Current problem

The CPU test `CPU supports ADD and rejects other compute capabilities` submits `x+x`, asserts only that `add` returns a token, waits, and never reads `y` before proceeding. CPU conformance's `run_add_request_conformance` likewise tests request/lifetime protocol without numeric output. SYCL's ADD-specific test invokes that same common fake queue, which records requests and owner registrations but performs no arithmetic. Production `CpuQueue::add_elements` and `SyclQueue::add_elements` contain real packed load/store loops calling `detail::scalar_add`; the direct scalar test does not prove those queue paths supply correct operands, mapping, or output bits. A mutation that stores the left operand instead of the sum would still satisfy current token/wait assertions. Root-run CPU and remote SYCL conformance suites passed, but the numerical deliberate-output perturbation was not run; the root ledger records no such gate. Consequently wrong CPU/SYCL inference values can reach consumers while conformance reports success.

## Scope

- Add one shared value-level ADD conformance helper invoked by the CPU and SYCL conformance drivers.
- Have the helper create candidate tensors and queues through each backend factory, submit ADD, wait, read back logical bytes, and compare candidate output to independent `add_oracle::add` results.
- Cover U8 shape `[2,17,33]` with nonuniform inputs and exact modulo-256 comparison, and F32 shape `[1,33,17]` with special-value classes and finite one-ULP comparison across all 21 numeric `QuantizationFormat::NONE` leaves supported by ADD.
- Retain existing common request/lifetime assertions and exclude CUDA, ROCm, TTNN, and rank-boundary cases (those belong to other tasks).

## Implementation references

- **Modify:** `test/backend/backend_conformance_add.hpp` — own the backend-neutral value-level helper, independent logical-byte packing, candidate readback, and comparison policy.
- **Modify:** `test/cpu/test_cpu_conformance.cpp` and `test/sycl/test_sycl_conformance.cpp` — invoke the shared helper with each real candidate backend while retaining `run_add_request_conformance` for protocol checks.
- **Read:** `test/backend/backend_conformance_add.hpp` — `iom_conformance::add_oracle::add`, `set_add_logical_value`, and `get_add_logical_value`; use the oracle independently of production arithmetic.
- **Read:** `src/cpu/device.cpp` — `CpuQueue::add_elements/add_impl`; ensure the test exercises packed storage and queue submission rather than direct scalar calls.
- **Read:** `src/sycl/copy.cpp` — `SyclQueue::add_elements/add_impl`; ensure the test observes the SYCL staging/queue path and performs readback after wait.
- **Read:** `test/cpu/test_cpu.cpp:1361-1402` — existing token/lifetime test; preserve its unsupported-operation and sequence checks while adding value observation in the conformance helper.

## Requirements

- The shared helper must submit through the candidate backend queue, wait for the returned token, copy/read the candidate output, and compare every addressed logical element. It must not substitute a CPU reference execution for candidate output.
- For U8 `[2,17,33]`, use distinct, nonuniform packed logical inputs and require exact `(lhs + rhs) mod 256` bytes at all 1122 elements, including row/column tails and any broadcasted coordinates exercised by the helper.
- For F32 `[1,33,17]`, include `+0/-0`, subnormal, finite rounding-boundary, max finite, `+/-Inf`, and NaN values. Require canonical NaN, infinity, and signed-zero classes exactly and finite output within one ULP of `add_oracle::add`; do not use an unexplained broad epsilon.
- Exercise all 21 numeric `QuantizationFormat::NONE` leaves through the same helper, excluding `BOOL` and `F8_E8M0` according to ADD capability policy. Keep U8 and F32 cases mandatory.
- Preserve the existing request/lifetime test and ensure a deliberately mutated CPU or SYCL output (for example, storing lhs unchanged) fails the value-level test. Do not include CUDA/ROCm/TTNN or rank-8/9/16/17 boundary scenarios.

## Non-goals

- Do not replace common validation, owner, sequence, or lifetime assertions; this helper supplements them.
- Do not alter production `scalar_add`, CPU/SYCL arithmetic, dtype policy, backend selection, or add unsupported operators.
- Do not require bitwise identity for finite floating results where the contract permits one ULP, add a CPU fallback that hides SYCL execution, or duplicate the helper body in CPU and SYCL tests.
- Do not cover CUDA/ROCm/TTNN or rank-boundary cases; those are explicitly outside NT-001.

## Acceptance criteria

- [ ] CPU and SYCL real-queue ADD tests return positive tokens, wait successfully, read candidate output, and compare all mandatory U8/F32 values to the independent oracle.
- [ ] U8 `[2,17,33]` results are exact modulo 256 for nonuniform/tail elements; F32 `[1,33,17]` matches canonical special classes and finite values within one ULP.
- [ ] A test-only mutation that stores an input unchanged, uses wrong operands, corrupts packed bits, or maps a broadcast coordinate incorrectly fails the conformance test; the existing token/lifetime test remains present.
- [ ] Finite one-ULP deviations pass while two-ULP deviations and wrong NaN, infinity, or zero-sign classes fail.
- [ ] No CUDA, ROCm, TTNN, or rank-boundary cases are added to this shared CPU/SYCL task, and no production arithmetic accessor is used as the oracle.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_cpu_conformance_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^iom_cpu_conformance_tests$'` — candidate CPU queue readback matches U8/F32 oracle and existing protocol tests pass.
- `(remote-development: SYCL host)` sync the workspace and run the configured oneAPI/icpx build for `iom_sycl_conformance_tests`, followed by `ctest --test-dir build --output-on-failure -R '^iom_sycl_conformance_tests$'` — candidate SYCL queue readback matches the same oracle after wait.
- Run a focused throwaway differential mutation that stores lhs instead of `scalar_add(lhs,rhs)`; both CPU and SYCL value-level tests must fail. Run comparator boundary scenarios with exact classes, one finite ULP (pass), and two finite ULPs/wrong special class (fail).

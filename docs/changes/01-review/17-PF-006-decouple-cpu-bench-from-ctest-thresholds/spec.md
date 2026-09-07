# CPU performance CTest gate hard-codes one host's calibration

**Order:** 17
**Priority:** P2
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** PF-006
**Review area:** Performance
**Review severity:** low
**Review verification:** strongly-supported, confidence 94
**Review scope:** whole-codebase
**Backend scope:** cpu
**Location:** `test/cpu/test_cpu_bench.cpp:131-207 (TEST_CASE "CPU benchmark: blocked copy throughput vs memory floor"); test/CMakeLists.txt:36-50 (iom_cpu_bench target and unconditional add_test)`

## Outcome

Generic CTest runs of the CPU benchmark report measured samples plus workload/host/compiler identity without failing on the one-host absolute floors. Pass/fail calibration lives only in a deliberately controlled performance job whose runner identity and baseline are explicit; functional CPU correctness remains covered by the separate test suites. The cross-host false-failure mechanism was demonstrated by the root run: on remote host bv2 (AMD Ryzen 7 9700X — not the calibrated HX 370 host), `ctest -R iom_cpu_bench` FAILED with `I4 2048x2048 copy_from_host` at 0.136 GB/s below the 0.6 GB/s floor and an F32 16x16 queued submit+wait differential of ~0.3 us above the ~0.1 us allowance.

## Current problem

A benchmark gate must compare performance under a controlled, identified environment; a generic correctness CTest run must not fail solely because the host is slower, differently configured, or under ordinary load. Today the CPU benchmark violates that:

- `iom_cpu_bench` is registered as an unconditional CTest test (`add_test(NAME iom_cpu_bench COMMAND iom_cpu_bench)`) at `test/CMakeLists.txt:50` whenever `BUILD_TESTING` is enabled (target defined at `:36-49`).
- `TEST_CASE "CPU benchmark: blocked copy throughput vs memory floor"` (`test/cpu/test_cpu_bench.cpp:131-207`) asserts fixed absolute throughput floors — F32 copy_from_host &ge; 4.0 GB/s, copy_to_host &ge; 2.4 GB/s, queued copy &ge; 2.0 GB/s, I4 copy_from_host &ge; 0.6 GB/s — and a fixed queued-copy allowance (`kAllowanceSeconds = 7.0e-8` s) for the F32 16x16 submit+wait differential.
- The calibration comments state these values were derived on one AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release `-O2`, single-threaded, ordinary developer load (median-based, floors at half the measured median), and the calibration notes record that earlier constants were already re-derived when a prior task deleted the CPU `StagedWorker` path.
- The benchmark prints a single reference-host identity MESSAGE but does not validate or select that environment; no baseline input, runner guard, or variance policy accompanies the assertions. `median_seconds` uses only one warmup and five timed samples (`:31-47`).

The root run recorded the concrete false failure: on bv2 (AMD Ryzen 7 9700X), the generic gate failed on the I4 throughput floor (0.136 GB/s vs 0.6) and on the queued-latency differential (0.3 us vs the 0.1 us-class allowance), with no code change involved. Portable CI/developer runs can likewise fail on slower or differently loaded hosts while the output lacks baseline-relative variance.

## Scope

- Separate benchmark reporting from correctness CTest: make `iom_cpu_bench` (or a companion runner) report measured samples and host/compiler identity without failing on the one-host absolute floors, OR run the threshold assertions only in a deliberately controlled benchmark job whose runner identity and baseline are explicit.
- Keep the existing benchmark workload, warmup/sample methodology, and the MESSAGE reporting (including the reference-host identity line); remove unportable unconditional pass/fail calibration from generic CTest.
- Keep the functional CPU test suites (`iom_cpu_tests`, `iom_backend_conformance_cpu_tests`) as the correctness gates, unchanged and independent of benchmark registration.

## Implementation references

- **Modify:** `test/cpu/test_cpu_bench.cpp` — the `TEST_CASE` at `:131-207`: decouple the `CHECK_GE` throughput floors and the `kAllowanceSeconds` differential `CHECK_MESSAGE` from the generic CTest path while preserving the MESSAGE reporting and calibration-comment context; keep `median_seconds`/`measure` (`:31-47`, `:63-122`) so samples and variance remain reportable.
- **Modify:** `test/CMakeLists.txt:36-50` — the `iom_cpu_bench` target and `add_test` registration: either register the benchmark as reporting-only for generic CTest, or move threshold enforcement behind an explicit controlled-benchmark contract (environment/runner guard and recorded baseline).
- **Read:** `test/cpu/test_cpu_bench.cpp:31-47` — `median_seconds`/`measure` methodology (one warmup, five timed samples) to preserve when converting assertions into reporting plus variance output.
- **Tests (unchanged):** `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` — the functional CPU correctness gates; nothing about their registration or assertions changes.

## Requirements

- A generic `ctest -R iom_cpu_bench` run (with `BUILD_TESTING=ON`) MUST NOT fail solely on absolute throughput/latency thresholds on any supported host; the exact bv2 failure (I4 from_host 0.136 GB/s vs the 0.6 floor; queued differential 0.3 us vs the allowance) must no longer fail a generic run.
- Benchmark output MUST still record usable measurements: workload, compiler/OS, detected hardware, warmup/sample count, samples and variance, and — where thresholds are enforced — the explicit threshold/baseline decision.
- If a controlled performance job is retained, its runner identity and baseline MUST be explicit and gated (matching the named host/configuration), not implicit in an unguarded generic test.
- Functional CPU correctness tests remain separate and must pass unchanged.

## Non-goals

- Do not add accelerator benchmarks or a general profiling harness (out of scope for this task).
- Do not alter the CPU copy implementation; do not weaken a deliberately controlled performance job once its runner contract is explicit.
- Do not delete the benchmark workload or its MESSAGE reporting.

## Acceptance criteria

- [ ] On the reference host AND on a slower/different supported host (e.g., remote bv2, AMD Ryzen 7 9700X), a generic `ctest --test-dir <build> -R '^iom_cpu_bench$' --output-on-failure` run succeeds (no failure driven by the absolute floors) while still printing throughput/latency samples and workload/host/compiler identity.
- [ ] Threshold enforcement, if retained, is reachable only through an explicit controlled-run contract (identifiable job/flag/environment plus a recorded baseline); the 4.0/2.4/2.0/0.6 GB/s floors and the queued-latency allowance no longer gate the ordinary CTest path.
- [ ] `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` still pass unchanged; the benchmark case still executes the same workload (4096x4096 F32 and 2048x2048 I4 transfers, 16x16 latency pair) and reports the same metrics.

## Verification

- Actual root-run validation (recorded): on remote bv2 (AMD Ryzen 7 9700X, CPU-only — not the calibrated HX 370 host), configure succeeded and `ctest -R iom_cpu_bench` FAILED as a cross-host false-failure demonstration: `i4 from_host 0.136 GB/s < 0.6 GB/s floor` and `queued differential 0.3 us > 0.1 us allowance`. On the same host `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` passed. This task removes that host-only gate failure while keeping the measurements.
- Proposed implements gate: run `ctest --test-dir <build> -R '^iom_cpu_bench$' --output-on-failure` on the reference host and on bv2 (or another slower/different host or load state) and verify generic runs report samples without failing; re-run `ctest --test-dir <build> -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure` to confirm the functional gates are unchanged.

- `ctest --test-dir <build> -R '^iom_cpu_bench$' --output-on-failure`
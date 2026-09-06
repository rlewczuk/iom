# Replace the absolute queued-latency gate with a self-relative no-op baseline gate calibrated from post-51/post-62 ordinary-load distributions and re-baseline honest CPU throughput floors

**Order:** 67
**Priority:** P0 — the always-run ctest suite is deterministically red on the reference host, so the "ctest exits 0" acceptance of every applied sub-change spec is structurally unreachable and real regressions hide behind permanent red. Restoring a trustworthy exit-0 signal gates all remaining validation of this change.
**Blocked by:** `51-CC-001-enforce-cpu-tiled-row-walk` — the throughput floors must be calibrated against the corrected per-tile-column walk; calibrating before 51 lands would freeze numbers measured on the layout-invalid contiguous path that 51 deletes. Orders 61-AR-003-simplify-outstanding-work-registry and 62-AR-002-share-release-quarantine-protocol precede 67 in the change and shape the consumed production code path (the post-61/62 no-registry CPU copy state described throughout this spec), but they are not hard calibration blockers for this task.
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-001`
**Review verification:** verified, confidence 95

## Outcome

`iom_cpu_bench` stays in the always-run ctest suite but its 16×16 queued-latency gate becomes self-relative. The test measures, interleaved in the same run, the median submit+wait of a real 16×16 queued copy and of an identical-window queued no-op (`queue->copy(source->view(), source->view())` — the same staged worker hand-off, `tasks_` list splice, completion-CV notification, and wait path with only the data-movement inner kernel skipped — no registry registration/release on CPU after 61-AR-003) and gates `copy_median ≤ no_op_median + kAllowanceSeconds`, where `kAllowanceSeconds` is a single named constant in the test, derived once on the reference host by the deterministic calibration formula in Requirements 3–4 from the post-51/post-62 ordinary-load distribution and frozen into the source with documented measured inputs and headroom. The forbidden gate shape is an absolute total-latency ceiling (the deleted `CHECK_LE(..., 6.0e-6)`); the permitted gate shape is a measured/calibrated additive allowance over the self-relative no-op baseline. Absolute submit+wait values become `MESSAGE` reporting only. The four GB/s throughput floors are re-calibrated from post-CC-001 measurements on the documented reference host. The suite's exit-0 signal is trustworthy again: 20 consecutive runs pass under ordinary developer load, and a copy-path regression (e.g. a +100 µs injection into the data-movement path only) still trips the relative gate.

## Current failure

Invariant: a gate registered in the always-run ctest suite must pass on the reference machine under ordinary developer load, or the suite's exit-0 signal is meaningless.

`test/cpu/test_cpu_bench.cpp:115` asserts `CHECK_LE(f32_small.queued_small_seconds, 6.0e-6)` — an absolute wall-clock ceiling on the median submit+wait of one 16×16 F32 queued copy. The measured quantity is dominated by two OS thread transitions (caller → `StagedWorker` thread → waiter; `include/iom/iom.hpp:58-133` `submit_copy`/`run`, `src/cpu/device.cpp:762-788` `CpuQueue` ctor/destructor and worker wiring), not by code. On the reference development host the median is 7.1 µs against the 6 µs bound: the test fails deterministically (5/5 review runs, load ≈ 1.2). The 6 µs number originates from `docs/changes/0001-tensor-view/41-PF-001-block-cpu-tile-copy/spec.md:171` ("36.3 µs reduces by ≥ 6×" → 6 µs) — a host-specific ratio frozen into an absolute gate with zero jitter margin. This gate shape (a fixed total-latency constant) is the forbidden form; the new gate shape (a measured/calibrated additive allowance on the self-relative differential against a same-process no-op baseline) is the permitted form.

The bench target is registered unconditionally (`test/CMakeLists.txt:36-50`, `add_test(NAME iom_cpu_bench COMMAND iom_cpu_bench)`), so every full `ctest` run is red while `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` pass — the bench is the suite's only red. A permanently red always-run gate trains developers to ignore red and masks real regressions.

Additionally, the four throughput floors in the same case (`test/cpu/test_cpu_bench.cpp:111-114`: F32 `copy_from_host` ≥ 10 GB/s, F32 `copy_to_host` ≥ 10 GB/s, F32 queued ≥ 5 GB/s, I4 `copy_from_host` ≥ 1 GB/s) were calibrated on the layout-invalid contiguous fast path (24.3 / 85.6 GB/s F32, 19.5 GB/s I4) that task 51 deletes. They pass today for the wrong reason and must be honestly re-measured against the corrected per-tile-column walk once 51 lands.

## Scope

- Remove the absolute submit+wait ceiling at `test/cpu/test_cpu_bench.cpp:115`. No `CHECK` in the translation unit may compare any total wall-clock duration against an absolute constant. (The gate's mandatory shape is `copy_median ≤ no_op_median + kAllowanceSeconds` where `kAllowanceSeconds` is a measured/calibrated additive constant with documented inputs.)
- Add an identical-window queued no-op baseline measured in the same process and run as the gated copy: on the same 16×16 F32 tensors and queue used by the small-latency measurement, each sample submits `queue->copy(source->view(), source->view())` and waits. `CpuQueue::copy` classifies it `no_op` via `identical_window` (`src/cpu/device.cpp:795`) and `CpuQueue::execute` (`src/cpu/device.cpp:831-834`) skips only `copy_elements`. After 61-AR-003 the CPU copy path no longer registers registry entries (`src/cpu/device.cpp:831-863` already skips registration for no-op, and 61-AR-003 removes it from the real copy path too — see order 61's spec); the no-op baseline therefore pays exactly the staged worker hand-off, the `tasks_` list splice, the completion-CV notification, and the wait that the gated copy pays. The baseline is the same submit+wait path with no data movement and no outstanding-work bookkeeping.
- Interleave the two series so both see the same load: one warmup pass (one no-op sample, one copy sample), then 11 samples, each timing one no-op submit+wait followed by one real-copy submit+wait; median of each 11-sample series (sorted index 5) gives one paired `no_op_median` and `copy_median` per run.
- Compute the additive allowance once on the reference host by the deterministic, explicitly nonnegative calibration formula in Requirements 3–4 from 20 consecutive ordinary-load runs; freeze the resulting `kAllowanceSeconds` as a TU-local `constexpr double` in the test source with a comment block documenting the measured inputs (calibration dataset size, timer period `R`, computed `D`, computed `Q`, `diff_max`, computed `M`, unrounded `A_seconds`, rounding granularity `G`, and the raw `diff_median_1..20` values) that produced it. The constant is the only latency figure in pass/fail, and its derivation is recorded inline so the next reviewer can re-derive it without re-reading this spec.
- Keep absolute values outside pass/fail: the copy median, the no-op median, and a reference-host line become `MESSAGE` output alongside the existing throughput `MESSAGE`s. The historical 36.3 µs → ≥6× improvement comparison is reported evidence only.
- Re-calibrate the four throughput floors from post-CC-001 measurements using the protocol in Verification (median across a 20-run stability set, halved, on the documented reference host), and record the reference host, measured medians, and calibration date in a comment block adjacent to the floors.
- Affected surface: `test/cpu/test_cpu_bench.cpp` only, plus this spec. Zero production-code changes and zero CMake changes; `iom_cpu_bench` remains unconditionally registered and always-run.

## Implementation references

- **Modify:** `test/cpu/test_cpu_bench.cpp` — `BenchmarkResult` (`:46-51`) gains `queued_no_op_seconds`; `measure` (`:53-82`) measures the interleaved no-op/copy series inside the `measure_small_latency` branch (reusing the same `HostAllocator` (`:17-28`), device, 16×16 F32 tensors, and queue); the `TEST_CASE("CPU benchmark: blocked copy throughput vs memory floor")` (`:86-116`) replaces the `:115` ceiling with the self-relative gate. Keep the existing case name — it is referenced by the 41-PF-001 evidence conventions — and keep the median-of-5/one-warmup `median_seconds` helper (`:31-44`) for the four bandwidth medians, which are bandwidth-scale and not the failure mode. Add a single `constexpr double kAllowanceSeconds = …;` with a calibration provenance comment block (calibration date, N, computed `D`, computed `Q`, chosen multiplier).
- **Read:** `src/cpu/device.cpp:791-802` — `CpuQueue::copy` and the `identical_window` no-op selection that makes `queue->copy(v, v)` a true zero-data queued submission; `:831-863` — `execute`, which (after 61-AR-003) does not register registry entries for any CPU copy and skips `copy_elements` for the no-op branch.
- **Read:** `docs/changes/0001-tensor-view/61-AR-003-simplify-outstanding-work-registry/spec.md` — the registry simplification that lets the CPU path opt out of registration, so the no-op baseline is the staged worker / list / CV / wait round trip alone; the baseline must be re-derived under this state, not under the pre-61 four-multimap state.
- **Read:** `docs/changes/0001-tensor-view/62-AR-002-share-release-quarantine-protocol/spec.md` — the destructor/release/quarantine consolidation that may have touched `execute`'s bookkeeping shape; the calibration absorbs any incidental effect on the differential.
- **Read:** `include/iom/iom.hpp:58-133` (`StagedWorker::submit_copy`) and `:165-186` (`run`/`process`) — the staged publish, worker wake, fence no-ops, and completion path that constitute the measured two-transition round trip the no-op baseline isolates.
- **Read:** `docs/changes/0001-tensor-view/41-PF-001-block-cpu-tile-copy/spec.md` — the bench target's origin, the `:171` threshold provenance (now deleted), and the reference-host identification ("Ryzen AI 9 HX 370, single thread, g++ 15.2.0 `-O2`", `:20`) to reuse verbatim in the documentation comment.
- **Read:** `docs/changes/0001-tensor-view/51-CC-001-enforce-cpu-tiled-row-walk/spec.md` — the corrected walk the floors must be calibrated against, and its verification note excluding `iom_cpu_bench` pending this task.
- **Tests:** `test/CMakeLists.txt:36-50` — the unconditional `iom_cpu_bench` registration; unchanged by this task.

## Requirements

1. Delete the `CHECK_LE(f32_small.queued_small_seconds, 6.0e-6)` gate at `test/cpu/test_cpu_bench.cpp:115`. After the change, `grep -n "6.0e-6" test/cpu/test_cpu_bench.cpp` returns zero matches and no `CHECK` in the TU compares a total wall-clock duration against an absolute constant. The only latency `CHECK` permitted in the TU is `CHECK_LE(f32_small.queued_small_seconds, f32_small.queued_no_op_seconds + kAllowanceSeconds)` (Requirement 5).

2. Measure the no-op baseline on the same queue, tensors, and `measure_small_latency` flag path as the gated copy. Each no-op sample is `const iom::oid token = queue->copy(source->view(), source->view()); queue->wait(token);` — `Tensor::view()` returns `TensorView&` (`include/iom/tensor.hpp:206`), so the same view binds to both parameters and `identical_window` fires. Under the post-61 CPU code path (no registry registration), the baseline pays exactly the staged worker hand-off, the `tasks_` list splice, the completion-CV notification, and the wait — no outstanding-work bookkeeping, no data movement.

3. Interleave the series: one warmup pass (one no-op sample, then one copy sample), then 11 samples, each timing one no-op submit+wait immediately followed by one real-copy submit+wait (`queue->copy(source->view(), destination->view())` as today). Each series yields a median (sorted element 5). Store as `BenchmarkResult::queued_no_op_seconds` and the existing `queued_small_seconds`; shapes that do not measure latency keep `0.0` for both. Within one run this produces one paired `(no_op_median, copy_median)` observation; across 20 ordinary-load runs the calibration in Requirement 4 produces 20 such paired observations.

4. Compute the additive allowance once on the reference host under ordinary developer load by this deterministic, explicitly nonnegative formula with proven headroom above every calibration-run median differential. Executed after task 51 has landed and the self-relative gate code from Requirement 5 is in place; 61/62 precede 67 in the change and shape the consumed production code path but are not hard calibration blockers.

   - For each run `k ∈ [1, 20]`, extract the `MESSAGE`'d `no_op_median_k` and `copy_median_k` (both in seconds).
   - Per-run differential: `diff_median_k = copy_median_k − no_op_median_k` (seconds; 20 values).
   - Robust spread of the healthy differential distribution: `Q = median(|diff_median_k − D|)` for `k ∈ [1, 20]` where `D = median(diff_median_1, …, diff_median_20)`. `Q` is the median absolute deviation of the 20 per-run median differentials about the 20-run median.
   - Worst observed per-run median differential: `diff_max = max(diff_median_1, …, diff_median_20)`.
   - Documented timer granularity: `R = std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den` in seconds, recorded in the provenance comment. On Linux x86-64 `R = 1e-9`; the formula uses the implementation's actual period rather than a guessed latency.
   - Empirical minimum margin derived from measured dispersion and timer resolution (not from an invented latency):
     `M = max(2 · Q, 10 · R)`.
     This guarantees `M > 0` (since `R > 0` for `steady_clock` and `2·Q ≥ 0`) and ties the floor to actual measurement characteristics, not to a hard-coded microsecond guess.
   - Unrounded allowance: `A_seconds = max(0, diff_max) + M + 2 · Q`. The `max(0, diff_max)` term guarantees the formula is nonnegative even on a host where the differential is occasionally negative due to scheduler/timer noise (the no-op path occasionally pays more than the copy path); the `M + 2·Q` tail guarantees `A_seconds > diff_max` (strict headroom above every observed per-run median differential).
   - Rounding granularity (fixed, stated, applied upward): `G = 1.0e-8` seconds (10 ns — coarser than `steady_clock`'s 1 ns period, the next decade up so the constant is robust against sub-period jitter). The committed constant is `kAllowanceSeconds = std::ceil(A_seconds / G) * G` — round up to the nearest 10 ns, never down. This guarantees `kAllowanceSeconds ≥ A_seconds > diff_max` strictly, and that the constant cannot land below an observed calibration run by rounding error.
   - The resulting `kAllowanceSeconds` is the only number frozen into the test source. Its comment block records: calibration date, host string, `N = 20`, `R` (timer period used), computed `D`, `Q`, `diff_max`, computed `M`, unrounded `A_seconds`, rounding granularity `G`, and the raw `diff_median_1..20` values used.
   - Injector calibration (used by Requirements 8 and the Verification injection step): the injected delay is `max(100e-6, 10 * kAllowanceSeconds)` — at least 100 µs, or 10× the committed allowance if the committed allowance is smaller. Since `kAllowanceSeconds > 0`, `10 · kAllowanceSeconds > kAllowanceSeconds`, so the injection is strictly greater than the committed allowance and the gate trips by construction regardless of what `kAllowanceSeconds` resolves to.
   - Run `iom_cpu_bench` 20 times consecutively under ordinary load (no `taskset`, no pinning, no quiet machine); capture the `MESSAGE` lines from each invocation of `./build/test/iom_cpu_bench` directly to stdout (do not use `ctest --output-on-failure` for collection — it suppresses `MESSAGE` output on passing runs).

5. Gate with the named constant: `constexpr double kAllowanceSeconds = <value from Requirement 4>;` at TU scope (or as the first statement inside the `TEST_CASE`, immediately above the gate, so it is adjacent to the calibration provenance comment). The assertion is `CHECK_LE(f32_small.queued_small_seconds, f32_small.queued_no_op_seconds + kAllowanceSeconds)` wrapped in `CHECK_MESSAGE` whose message prints the copy median, the no-op median, the differential, and the allowance, all in µs with one decimal. The comment directly above the constant records the calibration date, host string, dataset size `N = 20`, timer period `R`, computed `D`, computed `Q`, `diff_max`, computed `M`, unrounded `A_seconds`, rounding granularity `G = 1.0e-8`, the injector value `max(100e-6, 10 · kAllowanceSeconds)`, and the rounded constant being committed.

6. Re-calibrate the four throughput floors after task 51 has landed, on the reference host, using the protocol in Verification step 6: per metric, take the median of the 20 reported values across the stability runs, halve it, round down to 0.1 GB/s, and write it into the corresponding `CHECK_GE` at `test/cpu/test_cpu_bench.cpp:111-114`. Do not carry over any pre-CC-001 number (24.3 / 85.6 GB/s F32, 19.5 GB/s I4 measured the deleted contiguous path). Record directly above those four `CHECK_GE` lines a comment block with: the reference host string ("AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release `-O2`, single-threaded, ordinary developer load"), the calibration date, and the measured post-51 medians the floors were derived from.

7. Keep `iom_cpu_bench` unconditionally registered in the always-run suite. `git diff test/CMakeLists.txt` must be empty — this task fixes the gate's robustness, not the target's registration, timeout, or skip state.

8. Zero production-code change: only `test/cpu/test_cpu_bench.cpp` (and this spec) differ at completion. `src/cpu/device.cpp` appears in this task solely as the transient injection site of the falsification experiment and must be reverted (`git diff src/` empty).

9. Preserve the existing measurement conventions: `median_seconds` median-of-5 with one warmup for the bandwidth metrics, `steady_clock` timing, `HostAllocator` fixture, `MESSAGE`-first reporting style of the TU.

10. Re-calibration trigger: `kAllowanceSeconds` is re-derived only when a change to the queued-copy submission path, the `StagedWorker` mechanics, the completion-CV discipline, or the wait path is expected to perturb the differential distribution. PF-002 (download pre-zero removal) does not touch the queued-copy submit+wait round trip and therefore does not trigger re-calibration. The throughput floors (not the latency allowance) are the surface PF-002 may move, and only via the PF-002 spec itself, not via this task.

## Non-goals

- Removing the download pre-zero pass (`std::fill` before the tile walk, `src/cpu/device.cpp:582`) or raising the `copy_to_host` floor for that gain: review PF-002 owns download-prezero and any subsequent throughput-floor change after this task's post-51 calibration.
- Any CMake change: no new target, no timeout/skip/wiliness attributes, no conditional registration of `iom_cpu_bench`, no separate perf target.
- Any production change to `CpuQueue`, `detail::StagedWorker`, `DeviceOps`, or the outstanding-work registry (orders 61/62 territory); the no-op baseline consumes the post-61/62 identical-window behavior as is, and re-calibration absorbs any incidental shape change rather than re-engineering the baseline.
- Absolute-total-latency pass/fail gates in any form: the forbidden shape is a `CHECK` that compares a total submit+wait duration against an absolute constant; the permitted shape is `copy_median ≤ no_op_median + kAllowanceSeconds` for a measured/calibrated `kAllowanceSeconds`. The historical 36.3 µs → ≥6× improvement figure is documented as reported evidence only.
- Hardcoding any specific additive allowance value without derivation in this task: the value of `kAllowanceSeconds` is determined by Requirement 4's calibration on the reference host, not pre-stated here.
- Benchmark-framework work: no harness integration, CSV/JSON output, trend dashboards, CPU pinning, `taskset` isolation, or cross-host normalization.
- Accelerator benchmarking: no CUDA/ROCm/SYCL/TTNN bench is added or modified (SYCL queued-copy observation is NT-002; lifetime regression coverage is NT-003).
- Revising the 41-PF-001 conformance `TEST_CASE`s or any file other than `test/cpu/test_cpu_bench.cpp`.

## Acceptance criteria

- [ ] `grep -n "6.0e-6" test/cpu/test_cpu_bench.cpp` returns zero matches; the only latency `CHECK` in the TU is the self-relative one (no absolute total-latency ceiling remains).
- [ ] The gate is `CHECK_LE(queued_small_seconds, queued_no_op_seconds + kAllowanceSeconds)` with `kAllowanceSeconds` a single `constexpr double` defined once in the TU with a calibration provenance comment (date, host, N=20, timer period `R`, computed `D`, computed `Q`, `diff_max`, computed `M`, unrounded `A_seconds`, rounding granularity `G = 1.0e-8`, injector value `max(100e-6, 10 * kAllowanceSeconds)`, and the rounded constant being committed); the `CHECK_MESSAGE` text prints copy median, no-op median, differential, and allowance in µs.
- [ ] `kAllowanceSeconds = std::ceil(A_seconds / G) * G` where `A_seconds = max(0, diff_max) + max(2·Q, 10·R) + 2·Q` with `G = 1.0e-8`, per Requirement 4. By construction `kAllowanceSeconds > diff_max ≥ max(diff_median_1..20)` (every healthy calibration per-run median differential lies strictly below the committed constant); the constant is nonnegative; the rounding step rounds upward at a stated 10 ns granularity. The provenance comment in the TU records `D`, `R`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, `G`, and the raw `diff_median_1..20` values used so the constant is re-derivable from the source. Step 4b's collection runs against the step-4a transient working tree (no reselection, no rerun, no exit-code filter) and step 4c is the sole commit of the derived `kAllowanceSeconds`.
- [ ] The no-op baseline is measured interleaved with the gated copy in the same run (1 warmup pass + 11 samples per series, median of each), via `queue->copy(source->view(), source->view())` exercising the `identical_window` no-op path (`src/cpu/device.cpp:795`, `:831-834`); under post-61 CPU code the baseline pays staged worker hand-off, `tasks_` list splice, completion-CV notification, and wait only — no registry registration/release and no data movement.
- [ ] Absolute submit+wait values (copy median, no-op median, reference host) appear only in `MESSAGE` output; `grep -n "CHECK" test/cpu/test_cpu_bench.cpp` shows no `CHECK` over any absolute latency constant.
- [ ] The four `CHECK_GE` floors equal half the post-51 calibration protocol's per-metric median (rounded down to 0.1 GB/s), pass on the reference host, and the adjacent comment block records the reference host string, calibration date, and measured medians.
- [ ] 20 consecutive `./build/test/iom_cpu_bench` invocations (post-commit, ordinary developer load, no isolation tooling) on the reference host all exit `0`. `ctest --output-on-failure` is reserved for the suite-level check in Verification step 6 — it suppresses `MESSAGE` output on passing runs and so cannot serve as the stability loop here.
- [ ] Injected-regression experiment passes: with a copy-only `std::this_thread::sleep_for(std::chrono::microseconds(injector_delay));` added inside the `if (!task.no_op)` branch of `CpuQueue::execute` (`src/cpu/device.cpp:832`, plus a temporary `#include <chrono>`), where `injector_delay = max(100, std::chrono::duration_cast<std::chrono::microseconds>(10 * std::chrono::duration<double>(kAllowanceSeconds)).count())` (i.e. `max(100 µs, 10·kAllowanceSeconds)` per Requirement 4's injector calibration), the bench fails at the self-relative `CHECK_MESSAGE` naming a differential ≈ `injector_delay` while the no-op median stays in its usual range; after reverting, the bench passes again and `git diff src/` is empty. By construction `injector_delay > kAllowanceSeconds`, so the gate trips regardless of the resolved constant value.
- [ ] `git diff test/CMakeLists.txt` is empty; `iom_cpu_bench` remains unconditionally registered (`test/CMakeLists.txt:36-50`).
- [ ] Full CPU-only `ctest --test-dir build --output-on-failure` exits 0 with `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, and `iom_cpu_bench` all passing — restoring the exit-0 acceptance every applied sub-change spec relies on.

## Verification

CPU-only evidence is complete for this task; no accelerator hardware is touched. All commands run at the repository root.


1. Configure and build the always-run CPU targets in Release:

   ```bash
   cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release \
     && cmake --build build -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench
   ```

2. Single binary run — exits 0 and prints the copy median, no-op median, differential, reference host, and the four GB/s values to stdout via `MESSAGE`. `ctest --output-on-failure` suppresses `MESSAGE` output on passing runs, so the always-run probes in this spec invoke the bench binary directly and capture stdout. `ctest` is used only for the suite-level exit-0 check in step 6.

   ```bash
   ./build/test/iom_cpu_bench
   ```

3. 20-pass ordinary-load stability set (the review's acceptance method): runs after step 4c has committed the calibrated `kAllowanceSeconds` — this is step 4d, not a separate pass. Steps 1–2 build and exercise the wiring; step 4 (a→d) calibrates and freezes the constant; step 4d is the post-commit 20-run stability check. `ctest` is reserved for step 6's suite-level exit-0 check; all 20-run loops invoke the bench binary directly per step 2's reason (ctest suppresses `MESSAGE` output on passing runs).

4. `kAllowanceSeconds` calibration protocol — executed in this exact order. `ctest --output-on-failure` suppresses `MESSAGE` output on passing runs, so calibration collection invokes the binary directly and captures stdout.

   **Step 4a — transient working-tree state (no commit):** in the working tree, edit `test/cpu/test_cpu_bench.cpp` to (i) delete the absolute gate at `:115`, (ii) add the `MESSAGE` lines for the copy median, the no-op median, and the reference host, (iii) declare `constexpr double kAllowanceSeconds = 0.0;` at TU scope adjacent to the calibration provenance comment (the provenance comment lists every field listed in Requirement 4's bullet for the constant except the rounded constant value itself). The `0.0` placeholder is the only value present at this stage; the working tree is *not* committed at the end of step 4a — there is no intermediate commit of a placeholder or any other value. Rebuild and proceed to 4b.

   **Step 4b — collect exactly 20 runs of calibration data, exactly once:** with the working tree in its step-4a transient state, run the bench binary directly 20 times consecutively under ordinary load (no `taskset`, no pinning, no quiet machine). The collection captures the `MESSAGE` lines for the copy median and no-op median from each run's stdout. The collection does **not** check the binary's exit code and does **not** rerun: each of the 20 runs is taken as-is, even if the `0.0` placeholder is so small that some runs would exit non-zero. (The pipeline appends `|| true` to the per-run command, or otherwise ignores that exit, so a non-zero gate from an undersized placeholder does not truncate the 20-run set; a later step sizes the constant from this dataset alone, with no reselection.) This is the only 20-run data the calibration ever sees; there is no reselection, no rerun, no "raise the placeholder and re-collect" loop, and no filtering by exit code:

   ```bash
   mkdir -p /tmp/nt001 && : > /tmp/nt001/calibration.log && \
   for i in $(seq 1 20); do \
     { ./build/test/iom_cpu_bench 2>&1 \
       | grep -E "queued submit\+wait \(copy\)|queued submit\+wait \(no-op\)|reference host" \
       | sed "s/^/run $i: /"; } \
     >> /tmp/nt001/calibration.log || true; \
   done
   ```

   **Step 4c — compute and commit the final constant (single commit):** from `/tmp/nt001/calibration.log` compute `D = median(diff_median_1..20)`, `Q = median(|diff_median_k − D|)`, `diff_max = max(diff_median_1..20)`, `R = std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den` (recorded in the provenance comment; `1e-9` on Linux x86-64), `M = max(2·Q, 10·R)`, `A_seconds = max(0, diff_max) + M + 2·Q`, and `kAllowanceSeconds = std::ceil(A_seconds / 1.0e-8) * 1.0e-8`. Replace the step-4a `0.0` placeholder in the working tree with the computed `constexpr double kAllowanceSeconds = <rounded>;`; fill in the calibration provenance comment with every measured input (`D`, `R`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, `G`, the rounded constant being committed, and the raw `diff_median_1..20` values used). This is the **only commit of `kAllowanceSeconds`** in this task — the step-4a state was never committed, the step-4b collection never triggered a commit, and the step-4c replacement is the sole event that lands the constant in the tree. Rebuild after the commit and proceed to 4d.

   **Step 4d — 20-pass post-commit stability set (the only pass/fail acceptance step):** run the bench binary directly 20 times consecutively and confirm every run exits `0`. This is the review's acceptance method executed against the committed `kAllowanceSeconds`:

   ```bash
   for i in $(seq 1 20); do \
     ./build/test/iom_cpu_bench > /dev/null \
     || { echo "stability run $i FAILED"; exit 1; }; \
   done && echo "iom_cpu_bench: 20/20 stable"
   ```

   `ctest` is used only for the suite-level exit-0 check in step 6; the 20-run protocol here uses the direct binary because every `MESSAGE` line must reach stdout (ctest suppresses `MESSAGE` on passing runs).

5. Injected queued-path regression trips the relative gate. The injection site is `src/cpu/device.cpp`, which is not the test TU and so cannot reference `kAllowanceSeconds` directly. Procedure:

   - Compute the numeric `injector_delay` from the already-committed `kAllowanceSeconds` value (read from the TU source after step 4c): `injector_delay = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double>(10 * kAllowanceSeconds)).count()` µs, with a floor of `max(100, injector_delay)` to honor Requirement 4's `max(100 µs, 10·kAllowanceSeconds)`. The implementer substitutes the resulting µs value as a literal at the injection site and records the literal in the procedure next to the edit (e.g. `// injector_delay = 100 µs (derived from kAllowanceSeconds = 3.4 µs at commit abc123)`). The `injector_delay` literal in the edit equals `max(100, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double>(10 * kAllowanceSeconds)).count())` and is documented inline so a reviewer can re-derive it without re-reading the calibration artifact.
   - Temporarily edit `src/cpu/device.cpp`: add `#include <chrono>` and insert `std::this_thread::sleep_for(std::chrono::microseconds(<literal>));` as the first statement of the `if (!task.no_op)` branch in `CpuQueue::execute`, where `<literal>` is the computed `injector_delay` µs integer.
   - Rebuild only the bench target and run it directly:

     ```bash
     cmake --build build -j --target iom_cpu_bench && ./build/test/iom_cpu_bench
     ```

   - Expected: the case fails at the self-relative `CHECK_MESSAGE`; the printed differential is ≈ `injector_delay` µs `> kAllowanceSeconds` (by Requirement 4's construction: `injector_delay = max(100 µs, 10·kAllowanceSeconds) > kAllowanceSeconds` strictly because `kAllowanceSeconds > 0`), and the printed no-op median stays within its usual (single-digit µs) range — proving the baseline is copy-path-specific and the gate is discriminating, not merely permissive. Revert the edit, rebuild, and confirm the binary exits `0` again with `git diff src/` empty.
6. Full always-run set exits 0 — the restored suite-wide acceptance:

   ```bash
   ctest --test-dir build --output-on-failure
   ```

   Expected: `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, `iom_cpu_bench` all pass.
7. Throughput-floor calibration protocol (run once, after task 51 has landed and this gate change is in place, on the reference host under ordinary load). The four GB/s values are `MESSAGE` lines, so collection invokes the bench binary directly:

   ```bash
   : > /tmp/nt001/throughput.log && \
   for i in $(seq 1 20); do \
     ./build/test/iom_cpu_bench 2>&1 \
       | grep -E "GB/s" \
       | sed "s/^/run $i: /" \
       >> /tmp/nt001/throughput.log; \
   done
   ```

   Per metric (F32 `copy_from_host`, F32 `copy_to_host`, F32 queued, I4 `copy_from_host`): take the median of the 20 reported GB/s values from `/tmp/nt001/throughput.log`, halve it, round down to 0.1 GB/s, and write that constant into the corresponding `CHECK_GE` with the host/date/medians comment block of Requirement 6. Expected observation: all four floors pass in step 2/4d runs with the measured medians comfortably (≥2×) above them, and no floor reuses a pre-CC-001 figure. If the medians move again later (e.g. after PF-002's download-prezero fix), that later recalibration belongs to the task that causes the movement, not this one.

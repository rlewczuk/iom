# Re-derive `kAllowanceSeconds` after `73-PF-003-reduce-staged-submission-overhead` deletes the CPU `StagedWorker` so the always-run ctest exit-0 signal is trustworthy against the post-73 inline-completion path

**Order:** 77
**Priority:** P0 — until `kAllowanceSeconds` is re-derived the always-run `iom_cpu_bench` gate is calibrated against a code path that no longer exists; the constant may pass vacuously (allowance far above the new differential, hiding real regressions) or fail spuriously (jitter above the new differential, re-training developers to ignore red — the exact defect class NT-001 was raised for). Restoring a trustworthy exit-0 signal gates every remaining validation of this change.
**Blocked by:** `73-PF-003-reduce-staged-submission-overhead` — order 73 deletes the CPU `StagedWorker` path that order 67's `kAllowanceSeconds` was calibrated against; the `copy_median − no_op_median` differential changes shape (both medians drop, by different amounts), and the frozen constant no longer corresponds to the produced distribution. The post-73 tree is the substrate this task calibrates against; before 73 lands the constant is not re-derivable on the correct code path. Order 67 (NT-001) precedes 73 in the change set and is not a blocker of 77 — 67's gate-form, no-op baseline structure, calibration formula, and provenance-comment fields are reused verbatim and do not move.
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-001` (Requirement 10 re-calibration trigger of `67-NT-001-stabilize-cpu-benchmark-gates`, fired by `73-PF-003-reduce-staged-submission-overhead`); no new review finding exists — `review.md` enumerates `CC-001..CC-004`, `ST-001..ST-005`, `AR-001..AR-007`, `NT-001..NT-004`, `PF-001..PF-006` and nothing else.
**Review severity:** high
**Review verification:** verified, confidence 95

## Outcome

`kAllowanceSeconds` in `test/cpu/test_cpu_bench.cpp` is re-derived on the post-73 tree against the inline-completion CPU copy path; the gate form (`CHECK_LE(queued_small_seconds, queued_no_op_seconds + kAllowanceSeconds)`), the measurement structure (1 warmup + 11 interleaved samples per series, median of each), the calibration formula and rounding granularity (`G = 1.0e-8`), the injector formula, the unbiased collection protocol, and the comment-block shape are reused verbatim from order 67's spec — none of those move. The re-derived constant's provenance comment records, in addition to every field order 67 records, an explicit note that this is the post-73 re-derivation superseding order 67's constant, so a future reader can see why two derivations exist. The four throughput `CHECK_GE` floors at `test/cpu/test_cpu_bench.cpp:111-114` are explicitly out of scope (see Scope and Non-goals — the throughput surface is 72-PF-002's territory, and order 72 explicitly defers any post-fix `copy_to_host` floor re-derivation to a follow-up task; this task is that follow-up only with respect to the latency allowance). The always-run `iom_cpu_bench` exit-0 signal is trustworthy again: `kAllowanceSeconds > diff_max ≥ every healthy calibration-run median differential` by construction, and an injected copy-path regression of `≈ injector_delay` µs trips the gate while a clean tree passes 20/20 consecutive direct-binary invocations under ordinary developer load.

## Current failure

`docs/changes/0001-tensor-view/review.md` NT-001 invariant: "A gate registered in the always-run ctest suite must pass on the reference machine under ordinary developer load, or the suite's exit-0 signal is meaningless."

Order 67's spec Requirement 10 states `kAllowanceSeconds` must be re-derived when "a change to the queued-copy submission path, the `StagedWorker` mechanics, the completion-CV discipline, or the wait path is expected to perturb the differential distribution." Order 73 deletes the CPU `StagedWorker` path entirely: `CpuQueue::copy` no longer hands a `Task` to `detail::StagedWorker::submit_copy`; instead it executes `copy_elements` inline on the submitting thread and calls `DeviceOps::complete(sequence, nullptr)` directly (order 73 Requirement 12, `CpuQueue::copy` post-73 shape). The previous round trip — caller → `StagedWorker` worker → completion-CV notify → waiter — collapses to caller-only; the condvar notification, the worker mutex acquisition, the worker-thread `process` invocation, and the `complete_task` callback lookup all disappear.

This perturbs the distribution order 67 calibrated against in two distinct ways:

1. **Both medians drop.** The copy median loses the worker-thread handoff (the `~3–6 µs` order 73 estimates on the pre-61/62 tree — an estimate, not a measurement); the no-op baseline median loses the same worker-thread handoff it was measuring as the "shared round-trip cost" the copy adds to. The relative magnitudes of those drops are not measured and not assumed by this task.

2. **The differential changes shape.** The differential order 67 measured was the worker's per-copy `execute`-vs-no-op work plus the worker-hand-off fixed cost shared by both paths (so a roughly constant per-call overhead added identically to both, leaving the differential to reflect `execute`'s work alone). The post-73 differential reflects only the work the inline copy does that the inline no-op doesn't — `copy_elements` itself, the `validate_copy`/`identical_window` checks, and the `DeviceOps::complete` record. The shape is the same quantity, but the value and its dispersion are different.

Order 67 froze `kAllowanceSeconds` against the pre-73 distribution. Until re-derived the always-run `iom_cpu_bench` gate is untrustworthy in two directions: it may pass vacuously (the committed allowance is far above the new, smaller differential, so a real regression in the copy path hides inside the constant's headroom), or it may fail spuriously (run-to-run jitter on the post-73 path occasionally exceeds the constant that was calibrated for a wider distribution, re-training developers to ignore red). The second is the exact defect class NT-001 was raised for: a permanently red always-run gate masks real regressions by making red ignorable. The first is the converse: a permanently green gate by a stale constant is equally corrupting — a real regression in the copy path passes the suite. Both directions are untrustworthy until the constant matches the distribution.

Order 73 Requirement 13 ("Bench re-calibration is NOT in this task") and the order 73 "Bench re-calibration" non-goal defer re-calibration to "a separate explicit follow-up task (a '67-b' or a later NT)". That follow-up is this task.

Order 72's "Re-baselining `test_cpu_bench.cpp:112`" non-goal (`docs/changes/0001-tensor-view/72-PF-002-avoid-full-download-prezero/spec.md`) defers any post-fix `copy_to_host` throughput-floor re-derivation to a follow-up change; order 67 Requirement 10 says PF-002 "does not touch the queued-copy submit+wait round trip and therefore does not trigger re-calibration" of the latency allowance. Order 72 is the natural owner of the four `CHECK_GE` floor re-derivation; this task owns the latency allowance only.

## Scope

Exactly one file modified: `test/cpu/test_cpu_bench.cpp`. Only the `kAllowanceSeconds` `constexpr double` value and its calibration provenance comment change. Specifically:

- The gate at `test/cpu/test_cpu_bench.cpp:115` keeps the order-67 form `CHECK_LE(queued_small_seconds, queued_no_op_seconds + kAllowanceSeconds)` wrapped in `CHECK_MESSAGE` (order 67 Requirement 5); only the constant value and the comment directly above it move. The gate is **not** redesigned — no absolute constant, no new multiplier, no invented latency floor, no absolute microsecond literal in pass/fail.
- The measurement structure (1 warmup pass + 11 interleaved samples per series, median of each, via `queue->copy(source->view(), source->view())` for the no-op series and `queue->copy(source->view(), destination->view())` for the copy series) is preserved verbatim from order 67 Requirement 3.
- The four `CHECK_GE` throughput floors at `test/cpu/test_cpu_bench.cpp:111-114` and their adjacent comment block are **out of scope**: order 72 owns any throughput-floor re-derivation after PF-002's pre-zero removal; order 67 Requirement 10 makes the latency allowance and the throughput floors independent re-calibration surfaces. The throughput floors stay at their post-51/order-67 values.
- `test/CMakeLists.txt:36-50` (the unconditional `iom_cpu_bench` registration) is unchanged.
- No CMake change: no new target, no timeout/skip attribute, no conditional registration.
- No production change: `src/cpu/device.cpp`, `include/iom/iom.hpp`, `include/iom/detail/outstanding_work_registry.hpp`, every accelerator backend TU, and `src/iom.cpp` are not edited by this task. The post-73 production tree is consumed as is.

The transient injection site for the falsification experiment is `src/cpu/device.cpp`'s `CpuQueue::copy` post-73 inline work lambda (the non-no-op path that calls `copy_elements`); the edit is reverted before completion so `git diff src/` is empty.

## Implementation references

- **Modify:** `test/cpu/test_cpu_bench.cpp` — the single `constexpr double kAllowanceSeconds = <rounded>;` declaration (where it currently sits at TU scope adjacent to the gate, per order 67 Requirement 5) is replaced with the re-derived value; the calibration provenance comment directly above the constant is replaced with the post-73 re-derivation block, retaining every field order 67 records and adding the explicit "supersedes order 67" note (Requirements 4 and 5 below). The gate line at `:115` keeps its order-67 shape; the `median_seconds` helper at `:31-44` is untouched; `BenchmarkResult` at `:46-51` is untouched (the `queued_no_op_seconds` field is the order-67 addition; this task does not add any field); `measure` at `:53-82` is untouched.

- **Read:** `docs/changes/0001-tensor-view/67-NT-001-stabilize-cpu-benchmark-gates/spec.md` — Requirements 3 (interleaved 1-warmup + 11-sample series), 4 (the calibration formula `D`, `Q`, `diff_max`, `R`, `M = max(2·Q, 10·R)`, `A_seconds = max(0, diff_max) + M + 2·Q`, `G = 1.0e-8`, `kAllowanceSeconds = ceil(A_seconds/G) * G`; injector formula `max(100, duration_cast<microseconds>(10·kAllowanceSeconds).count())`; protocol of 20 consecutive runs under ordinary load, exit codes ignored for collection, sole commit, post-commit 20-run stability set), 5 (the gate shape `CHECK_LE(queued_small_seconds, queued_no_op_seconds + kAllowanceSeconds)` wrapped in `CHECK_MESSAGE` printing copy median / no-op median / differential / allowance in µs with one decimal; the provenance-comment field list: calibration date, host string, `N = 20`, timer period `R`, computed `D`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, rounding granularity `G = 1.0e-8`, injector value, rounded constant, raw `diff_median_1..20`), 6 (throughput-floor re-calibration — NOT in this task), 7 (`iom_cpu_bench` unconditional registration), 8 (production-code isolation: `git diff src/` empty), 9 (measurement conventions), 10 (re-calibration triggers). Every clause of Requirements 3–5 and 10 is reused verbatim.

- **Read:** `docs/changes/0001-tensor-view/73-PF-003-reduce-staged-submission-overhead/spec.md` — Requirement 12 (post-73 `CpuQueue::copy` executes inline: calls `copy_elements` for non-no-op, then calls `complete(sequence, nullptr)`; `Task`, `execute`, `complete_task`, `worker_`, `shutdown_and_drain` are deleted), Requirement 13 ("Bench re-calibration is NOT in this task"), the order 73 "Bench re-calibration" non-goal ("PF-003 does **not** edit `test/cpu/test_cpu_bench.cpp`; PF-003 records the bench medians before and after in the change evidence ... re-calibration against the new path is the explicit follow-up task, not PF-003"), the order 73 Outcome paragraph naming this task's domain ("The CPU bench (post-inline-completion) is left to a separate explicit follow-up task that owns the bench re-calibration").

- **Read:** `docs/changes/0001-tensor-view/72-PF-002-avoid-full-download-prezero/spec.md` — the non-goal "Re-baselining `test_cpu_bench.cpp:112` (`copy_to_host` throughput floor) in this task" and its trailing sentence: "If the post-fix median moves the floor, the recalibration follows NT-001's documented protocol ... and is owned by a follow-up change that runs the recalibration procedure." Order 72 explicitly delegates throughput-floor re-derivation away from itself. Combined with order 67 Requirement 10 (PF-002 does not touch the queued-copy submit+wait round trip and therefore does not trigger re-calibration of the latency allowance), the boundary is: 77 owns the latency-allowance re-derivation only; the four throughput `CHECK_GE` floors at `test/cpu/test_cpu_bench.cpp:111-114` are not touched by this task and stay at their order-67 (post-51) values.

- **Read:** `include/iom/iom.hpp:28-196` — `iom::detail::StagedWorker` class body (the worker the post-73 `CpuQueue` no longer instantiates). Order 73 deletes `CpuQueue`'s `worker_` member, the `StagedWorker<Task>::Callbacks` construction, the `StagedWorker<Task>::PublishPolicy::CompleteOnThrow` argument, `worker_.start()`, and `worker_.shutdown_and_drain()` — this task assumes the post-73 tree state described by order 73 Requirement 12.

- **Read:** `include/iom/iom.hpp:212-368` — `DeviceOps` class body. The post-73 `CpuQueue::copy` lambda calls `DeviceOps::complete(sequence, nullptr)` directly. The submit / wait / complete contract documented here is unchanged by order 73 and by this task. The inline-completion seam at `include/iom/iom.hpp:295-329` (`DeviceOps::submit` documenting "An inline-completing backend may call `complete(sequence)` from `queue_work`") is the canonical reference for the post-73 CPU shape.

- **Read:** `src/cpu/device.cpp:734-962` — `CpuQueue` class body as it stands in the current tree (pre-73). The members order 73 deletes are `Task` (`:735-752`), `SequenceOutcome` (`:754-759` — the per-sequence outcome map 73 also drops), the `worker_` member (`:766-781` with its `Callbacks` and `PublishPolicy::CompleteOnThrow`), `~CpuQueue`'s `shutdown_and_drain` (`:784-788`), `execute` (`:831-863`), `complete_task` (`:865-890`); the call to `worker_.submit_copy(std::move(task))` at `:800`; the `submission_order_mutex_` (`:960`) and the `outcome_mutex_` (`:958`) and the `outcomes_` map (`:959`) are dropped by 73 as well. The post-73 `copy` body (73 Requirement 12) is:

  ```cpp
  return submit(
          [this, &source, &destination, no_op](
                  std::uint64_t sequence) {
              if (!no_op) {
                  copy_elements(source, destination);
              }
              complete(sequence, nullptr);
          });
  ```

  This is the substrate this task calibrates `kAllowanceSeconds` against; the transient injection site for the falsification experiment (Requirement 8 below) is the post-73 inline `if (!no_op) { copy_elements(source, destination); }` body in `CpuQueue::copy`'s lambda — **not** the order-67 site (`CpuQueue::execute`'s `if (!task.no_op)` branch at pre-73 `src/cpu/device.cpp:831-834`), which order 73 deletes entirely. The implementer must verify the post-73 shape against the merged tree at task-execution time before injecting, and locate the post-73 site by reading order 73 Requirement 12's snippet against the delivered `src/cpu/device.cpp`. The literal substituted at the injection site is the numeric `injector_delay` µs integer because `src/cpu/device.cpp` is production code and cannot reference the test TU's `kAllowanceSeconds` symbol.

- **Read:** `test/CMakeLists.txt:36-50` — `add_executable(iom_cpu_bench cpu/test_cpu_bench.cpp)`, the doctest link, and `add_test(NAME iom_cpu_bench COMMAND iom_cpu_bench)`. Unconditional registration, unchanged by this task.

- **Tests:** `test/cpu/test_cpu_bench.cpp` is the only test file in this task's scope; no conformance or regression tests are added (the gate's discriminating power is verified by the injector experiment in Requirement 8 and Verification step 5).

## Requirements

1. The gate at `test/cpu/test_cpu_bench.cpp:115` is **unchanged in form** from order 67 Requirement 5: `CHECK_LE(f32_small.queued_small_seconds, f32_small.queued_no_op_seconds + kAllowanceSeconds)` wrapped in `CHECK_MESSAGE` whose message prints copy median, no-op median, differential, and allowance in µs with one decimal. Only the value of `kAllowanceSeconds` and the calibration provenance comment directly above it move. `grep -n "6.0e-6" test/cpu/test_cpu_bench.cpp` returns zero matches; `grep -n "CHECK" test/cpu/test_cpu_bench.cpp` shows no `CHECK` over any absolute latency constant.

2. The no-op baseline is measured on the same queue, tensors, and `measure_small_latency` flag path as the gated copy, interleaved in the same run per order 67 Requirement 3: one warmup pass (one no-op sample, then one copy sample), then 11 samples, each timing one `queue->copy(source->view(), source->view()); queue->wait(token);` no-op submit+wait immediately followed by one `queue->copy(source->view(), destination->view()); queue->wait(token);` real-copy submit+wait. Median of each 11-sample series (sorted index 5) gives one paired `(no_op_median, copy_median)` observation per run; across 20 ordinary-load runs the calibration in Requirement 3 produces 20 such paired observations. The post-73 `CpuQueue::copy` path is the consumed path for both series.

3. The calibration formula is reused **exactly** from order 67 Requirement 4, with no new multiplier, no invented latency floor, and no absolute microsecond literal in pass/fail:

   - For each run `k ∈ [1, 20]`, extract the `MESSAGE`'d `no_op_median_k` and `copy_median_k` (both in seconds).
   - Per-run differential: `diff_median_k = copy_median_k − no_op_median_k` (seconds; 20 values).
   - Robust spread of the healthy differential distribution: `Q = median(|diff_median_k − D|)` for `k ∈ [1, 20]` where `D = median(diff_median_1, …, diff_median_20)`.
   - Worst observed per-run median differential: `diff_max = max(diff_median_1, …, diff_median_20)`.
   - Documented timer granularity: `R = std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den` in seconds, recorded in the provenance comment (`R = 1e-9` on Linux x86-64 `steady_clock`).
   - Empirical minimum margin: `M = max(2 · Q, 10 · R)`. `M > 0` since `R > 0` for `steady_clock` and `2·Q ≥ 0`.
   - Unrounded allowance: `A_seconds = max(0, diff_max) + M + 2 · Q`. The `max(0, diff_max)` term guarantees nonnegativity even when scheduler/timer noise occasionally makes the no-op path slower than the copy path; the `M + 2·Q` tail guarantees `A_seconds > diff_max` strictly.
   - Rounding granularity (fixed, applied upward): `G = 1.0e-8` seconds (10 ns, the next decade above `steady_clock`'s 1 ns period). The committed constant is `kAllowanceSeconds = std::ceil(A_seconds / G) * G` — round up to the nearest 10 ns, never down. By construction `kAllowanceSeconds ≥ A_seconds > diff_max` strictly.
   - The committed constant satisfies `kAllowanceSeconds > diff_max ≥ max(diff_median_1..20)`: every healthy calibration per-run median differential lies strictly below the constant. The constant is nonnegative, the rounding step rounds upward, and the formula is deterministic and reproducible from the recorded `D`, `R`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, and `G`.
   - Injector calibration: `injector_delay = max(100, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double>(10 * kAllowanceSeconds)).count())` µs. Since `kAllowanceSeconds > 0`, `10 · kAllowanceSeconds > kAllowanceSeconds`; the injection is strictly greater than the committed allowance and trips the gate by construction regardless of the value `kAllowanceSeconds` resolves to.

4. The calibration **collection protocol** is order 67's unbiased protocol, reused verbatim:

   - **Transient working-tree state is never committed.** The `kAllowanceSeconds` declaration in the working tree holds a placeholder value of `0.0` immediately after the gate form from order 67 has been put in place; the calibration provenance comment lists every field of Requirement 5 below except the rounded constant value. The placeholder's `0.0` is the only value present in the transient tree; no intermediate commit of a placeholder or any other value is made.
   - **20 consecutive runs are collected exactly once, with exit codes ignored for collection.** The bench binary is run directly 20 times in a row under ordinary developer load (no `taskset`, no CPU pinning, no quiet-machine protocol — matching order 67's ordinary-load convention). The collection captures the `MESSAGE` lines for the copy median and no-op median from each run's stdout; it does not check the binary's exit code and does not rerun. The pipeline appends `|| true` to the per-run command so a non-zero gate from an undersized placeholder does not truncate the 20-run set; a later step sizes the constant from this dataset. There is no reselection, no "raise the placeholder and re-collect" loop, and no exit-code filter.
   - **The bench binary is invoked directly, not via `ctest`.** `./build/test/iom_cpu_bench` is the collection command; `ctest --output-on-failure` suppresses `MESSAGE` output on passing runs, so `ctest` cannot serve as the collection vehicle. `ctest` is reserved for the suite-level exit-0 check in Verification step 6.
   - **The derived constant is the sole commit.** From the 20-run dataset compute `D`, `Q`, `diff_max`, `R`, `M`, `A_seconds`, and `kAllowanceSeconds = ceil(A_seconds / G) * G`. Replace the `0.0` placeholder in the working tree with the computed value; fill in the calibration provenance comment per Requirement 5. The single commit at the end of step 4c contains the rounded constant and the full provenance comment; no other commit precedes it.

5. The calibration provenance comment directly above `constexpr double kAllowanceSeconds = <rounded>;` records every field order 67 records, in the same shape: calibration date; reference host string ("AMD Ryzen AI 9 HX 370, Linux x86-64, g++ 15.2.0, Release `-O2`, single-threaded, ordinary developer load" — verbatim from order 67 Requirement 6's reference-host line, since the reference host has not changed); dataset size `N = 20`; timer period `R` in seconds; computed `D`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, rounding granularity `G = 1.0e-8`, the rounded constant, the injector value `max(100, duration_cast<microseconds>(10*kAllowanceSeconds).count())` µs, and the raw `diff_median_1..20` values. In addition, the comment block records an explicit note: **this is the post-73 re-derivation superseding order 67's constant**, naming order 73's deletion of the CPU `StagedWorker` path (`src/cpu/device.cpp` `CpuQueue::Task`, `execute`, `complete_task`, `worker_`, `shutdown_and_drain`) as the reason a second derivation exists, so a future reader can distinguish the two derivations without re-reading this spec or order 67's spec.

6. Ordinary developer load, no `taskset`, no pinning, no quiet machine — matching order 67's convention. Calibration collection, the stability set, and the suite-level check all run under the same convention.

7. The four throughput `CHECK_GE` floors at `test/cpu/test_cpu_bench.cpp:111-114` are explicitly **out of scope** for this task; their comment block above `:111-114` is unchanged. Order 72 owns the `copy_to_host` floor re-derivation (order 72's "Re-baselining `test_cpu_bench.cpp:112`" non-goal defers it to a follow-up change), and order 67 Requirement 10 says PF-002 does not touch the queued-copy submit+wait round trip and therefore does not trigger latency-allowance re-calibration. The throughput floors and the latency allowance are independent re-calibration surfaces, and this task touches only the latency allowance.

8. The injector falsification experiment uses the **post-73 inline injection site**, not the order-67 site. Order 67's site (`src/cpu/device.cpp:832`, the `if (!task.no_op) { copy_elements(...) }` branch of `CpuQueue::execute`) does not exist after order 73 lands — order 73 deletes `CpuQueue::execute` entirely (`src/cpu/device.cpp:831-863` in the current tree; deleted by order 73 Requirement 12). The correct post-73 site is inside the `CpuQueue::copy` lambda's `if (!no_op)` branch (order 73 Requirement 12's snippet above), where `copy_elements(source, destination)` runs synchronously on the submitting thread. The implementer verifies the post-73 shape at task-execution time by reading order 73 Requirement 12 against the merged `src/cpu/device.cpp` and locating the inline copy call. The injector substitution is a numeric literal at the injection site: `injector_delay = max(100, duration_cast<microseconds>(10·kAllowanceSeconds).count())` µs (the `kAllowanceSeconds` already committed at step 4c), substituted as `std::this_thread::sleep_for(std::chrono::microseconds(<literal>));` as the first statement of the inline `if (!no_op) { ... }` body, with a temporary `#include <chrono>` added at the top of the TU if not already present (most likely already present via `src/cpu/device.cpp:4` in the current tree). The literal value is recorded next to the edit (e.g. `// injector_delay = <n> us (derived from kAllowanceSeconds = <v> us at commit <hash>)`). After the experiment, the edit is reverted; `git diff src/` is empty. The expected observation: the bench fails at the self-relative `CHECK_MESSAGE` with a differential ≈ `injector_delay` µs > `kAllowanceSeconds` (by Requirement 3's construction), and the no-op median stays in its usual (sub-µs to low-µs) range — proving the gate is copy-path-specific and discriminating, not merely permissive. After reverting, the bench passes again.

9. Zero production-code change at completion. `git diff src/cpu/device.cpp include/iom/iom.hpp include/iom/detail/outstanding_work_registry.hpp src/iom.cpp src/cuda src/rocm src/sycl src/ttnn` is empty. The only file modified is `test/cpu/test_cpu_bench.cpp` (and this spec). `git diff test/CMakeLists.txt CMakeLists.txt` is empty. `git diff include/` is empty. `iom_cpu_bench` remains unconditionally registered (`test/CMakeLists.txt:36-50`).

10. The always-run `ctest --test-dir build --output-on-failure` suite exits 0 with `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, and `iom_cpu_bench` all passing — restoring the exit-0 acceptance every applied sub-change spec relies on.

## Non-goals

- Re-deriving the four throughput `CHECK_GE` floors at `test/cpu/test_cpu_bench.cpp:111-114`. The throughput floors are a separate surface owned by order 72 ("Re-baselining `test_cpu_bench.cpp:112`" non-goal) and the post-72 follow-up change it names. The throughput floors and the latency allowance are independent re-calibration triggers (order 67 Requirement 10). This task touches only the latency allowance.
- Changing the gate form. The self-relative gate `CHECK_LE(queued_small_seconds, queued_no_op_seconds + kAllowanceSeconds)` is the order-67 form and stays; only the constant value moves.
- Changing the measurement structure (1 warmup + 11 samples per series, median of each, via the interleaved no-op/copy series in `measure`). Order 67's structure is preserved verbatim.
- Changing the calibration formula. Order 67's `D`, `Q`, `diff_max`, `R`, `M`, `A_seconds`, `G = 1.0e-8`, `kAllowanceSeconds = ceil(A_seconds/G) * G`, and the injector formula are reused verbatim — no new multiplier, no invented latency floor, no absolute microsecond literal in pass/fail.
- Reselection, "raise the placeholder and re-collect" loops, exit-code filters, or any other deviation from order 67's unbiased 20-run collection protocol.
- Any CMake change: no new target, no timeout/skip/wiliness attribute, no conditional registration of `iom_cpu_bench`, no separate perf target, no target rename.
- Any production change to `CpuQueue`, `DeviceOps`, `detail::StagedWorker`, the outstanding-work registry, or any other source file (`src/cpu/device.cpp`, `include/iom/iom.hpp`, `include/iom/detail/outstanding_work_registry.hpp`, `src/iom.cpp`, every accelerator backend TU). Order 73 owns the production changes; this task only consumes the post-73 tree.
- Any cross-backend or cross-host normalization. CPU-only, single-host calibration, single-threaded ordinary developer load.
- Any benchmark framework work: no harness integration, no CSV/JSON output, no trend dashboards, no `taskset` / CPU pinning, no quiet-machine protocol.
- Any accelerator benchmarking. CUDA/ROCm/SYCL/TTNN bench is not added or modified (this task is CPU-only).
- Revising order 67's spec or order 73's spec. The two prior specs are the upstream contracts this task consumes; only this spec (and the test TU's constant + comment) is authored.
- Re-litigating order 73's design. The post-73 inline-completion path is the consumed substrate; this task does not propose alternatives.
- Inventing a new review finding. This task is the implementation of order 67 Requirement 10's re-calibration trigger, fired by order 73; it is not a new finding and adds no entry to `docs/changes/0001-tensor-view/review.md` (which contains `NT-001..NT-004` and nothing more).

## Acceptance criteria

- [ ] `grep -n "6.0e-6" test/cpu/test_cpu_bench.cpp` returns zero matches; the only latency `CHECK` in the TU is the self-relative one (no absolute total-latency ceiling remains).
- [ ] The gate at `test/cpu/test_cpu_bench.cpp:115` is `CHECK_LE(f32_small.queued_small_seconds, f32_small.queued_no_op_seconds + kAllowanceSeconds)` wrapped in `CHECK_MESSAGE` whose message prints copy median, no-op median, differential, and allowance in µs with one decimal; `kAllowanceSeconds` is the single `constexpr double` declared at TU scope adjacent to the calibration provenance comment.
- [ ] `kAllowanceSeconds = std::ceil(A_seconds / G) * G` where `A_seconds = max(0, diff_max) + max(2·Q, 10·R) + 2·Q` with `G = 1.0e-8`, per Requirement 3. By construction `kAllowanceSeconds > diff_max ≥ max(diff_median_1..20)` (every healthy calibration per-run median differential lies strictly below the committed constant); the constant is nonnegative; the rounding step rounds upward at a stated 10 ns granularity. The provenance comment in the TU records `D`, `R`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, `G`, the rounded constant, the injector value, the raw `diff_median_1..20` values, the calibration date, and the reference host string, plus an explicit note that this is the post-73 re-derivation superseding order 67's constant. Step 4b's collection runs against the step-4a transient working tree (no reselection, no rerun, no exit-code filter) and step 4c is the sole commit of the constant.
- [ ] The no-op baseline is measured interleaved with the gated copy in the same run (1 warmup pass + 11 samples per series, median of each), via `queue->copy(source->view(), source->view())` exercising the post-73 inline no-op classification (`DeviceOps::identical_window`, the post-73 inline `CpuQueue::copy` shape from order 73 Requirement 12). The post-73 baseline pays `validate_copy`, `identical_window`, the `DeviceOps::submit` sequence reservation, the no-op branch's skip of `copy_elements`, and `DeviceOps::complete(sequence, nullptr)` — no `StagedWorker` worker thread, no completion-CV round trip, no registry handoff.
- [ ] Absolute submit+wait values (copy median, no-op median, reference host) appear only in `MESSAGE` output; `grep -n "CHECK" test/cpu/test_cpu_bench.cpp` shows no `CHECK` over any absolute latency constant.
- [ ] The four throughput `CHECK_GE` floors at `test/cpu/test_cpu_bench.cpp:111-114` and their adjacent comment block are unchanged. `git diff` of the throughput-floor region shows zero hunks. Order 72 owns any throughput-floor re-derivation; this task touches only the latency allowance.
- [ ] 20 consecutive `./build/test/iom_cpu_bench` invocations (post-commit, ordinary developer load, no isolation tooling) on the reference host all exit `0`. `ctest --output-on-failure` is reserved for the suite-level check in Verification step 6 — it suppresses `MESSAGE` output on passing runs and so cannot serve as the stability loop.
- [ ] Injected-regression experiment passes: with a copy-only `std::this_thread::sleep_for(std::chrono::microseconds(<literal>));` added as the first statement of the post-73 inline `if (!no_op) { copy_elements(source, destination); }` body in `CpuQueue::copy`'s lambda (post-73 `src/cpu/device.cpp`, verified at task-execution time against order 73 Requirement 12's snippet — the order-67 site at pre-73 `src/cpu/device.cpp:831-834` no longer exists), where `<literal> = max(100, duration_cast<microseconds>(10·kAllowanceSeconds).count())` µs (Requirement 3's injector formula applied to the committed `kAllowanceSeconds`), the bench fails at the self-relative `CHECK_MESSAGE` naming a differential ≈ `<literal>` µs > `kAllowanceSeconds` (by construction), and the no-op median stays in its usual range. After reverting, the bench passes again and `git diff src/` is empty.
- [ ] `git diff test/CMakeLists.txt` is empty; `iom_cpu_bench` remains unconditionally registered (`test/CMakeLists.txt:36-50`).
- [ ] `git diff src/cpu/device.cpp include/iom/iom.hpp include/iom/detail/outstanding_work_registry.hpp src/iom.cpp src/cuda src/rocm src/sycl src/ttnn` is empty at completion.
- [ ] Full CPU-only `ctest --test-dir build --output-on-failure` exits 0 with `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, and `iom_cpu_bench` all passing — restoring the suite-wide exit-0 acceptance every applied sub-change spec relies on.

## Verification

CPU-only evidence is complete for this task; no accelerator hardware is touched. All commands run at the repository root. The build directory is `build/` (one directory for the full task; the bench binary path is `./build/test/iom_cpu_bench`).

1. Configure and build the always-run CPU targets in Release:

   ```bash
   cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release \
     && cmake --build build -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench
   ```

2. Single binary run — exits 0 and prints the copy median, no-op median, differential, reference host, and the four GB/s values to stdout via `MESSAGE`. `ctest --output-on-failure` suppresses `MESSAGE` output on passing runs, so the always-run probes in this spec invoke the bench binary directly and capture stdout. `ctest` is used only for the suite-level exit-0 check in step 6.

   ```bash
   ./build/test/iom_cpu_bench
   ```

3. Pre-conditions check (confirms the post-73 tree substrate this task calibrates against):

   ```bash
   grep -n "StagedWorker\|submit_copy\|shutdown_and_drain" src/cpu/device.cpp
   grep -n "struct Task\|void execute(Task\|void complete_task(" src/cpu/device.cpp
   grep -n "copy_elements" src/cpu/device.cpp
   ```

   Expected on the post-73 tree: the first grep finds zero matches (`detail::StagedWorker<Task>`, `worker_.submit_copy`, and `shutdown_and_drain` are deleted by order 73 Requirement 12); the second grep finds zero matches (`struct Task`, `void execute(Task&)`, and `void complete_task(` are the in-class spellings the current tree uses at `src/cpu/device.cpp:735`, `:831`, `:865` — never qualified as `CpuQueue::execute`, so grep the in-class tokens, not the qualified names); the third grep still finds `copy_elements` at its post-73 location inside the inline `CpuQueue::copy` lambda (`copy_elements` itself survives — only its `execute`/`complete_task` callers are deleted). Note `class CpuQueue` (`:734`) is NOT expected to disappear: order 73 keeps the queue class and only removes its worker machinery, so do not grep for its absence. If any first- or second-grep token is still present, the post-73 tree has not landed and this task cannot proceed.

4. `kAllowanceSeconds` calibration protocol — executed in this exact order on the post-73 tree. `ctest --output-on-failure` suppresses `MESSAGE` output on passing runs, so calibration collection invokes the bench binary directly and captures stdout.

   **Step 4a — transient working-tree state (no commit):** in the working tree, edit `test/cpu/test_cpu_bench.cpp` so the gate at `:115` is `CHECK_LE(f32_small.queued_small_seconds, f32_small.queued_no_op_seconds + kAllowanceSeconds)` wrapped in `CHECK_MESSAGE` (the order-67 form, unchanged), the `MESSAGE` lines for copy median / no-op median / reference host are present, and `constexpr double kAllowanceSeconds = 0.0;` is declared at TU scope adjacent to a calibration provenance comment listing every field of Requirement 5 except the rounded constant value itself. The `0.0` placeholder is the only value present at this stage; the working tree is **not** committed at the end of step 4a — there is no intermediate commit of a placeholder or any other value. Rebuild and proceed to 4b.

   **Step 4b — collect exactly 20 runs of calibration data, exactly once:** with the working tree in its step-4a transient state, run the bench binary directly 20 times consecutively under ordinary load (no `taskset`, no pinning, no quiet machine). The collection captures the `MESSAGE` lines for the copy median and no-op median from each run's stdout. The collection does **not** check the binary's exit code and does **not** rerun: each of the 20 runs is taken as-is, even if the `0.0` placeholder is so small that some runs would exit non-zero. The pipeline appends `|| true` to the per-run command so a non-zero gate from an undersized placeholder does not truncate the 20-run set; a later step sizes the constant from this dataset.

   ```bash
   mkdir -p /tmp/nt001 && : > /tmp/nt001/calibration.log && \
   for i in $(seq 1 20); do \
     { ./build/test/iom_cpu_bench 2>&1 \
       | grep -E "queued submit\+wait \(copy\)|queued submit\+wait \(no-op\)|reference host" \
       | sed "s/^/run $i: /"; } \
     >> /tmp/nt001/calibration.log || true; \
   done
   ```

   **Step 4c — compute and commit the final constant (single commit):** from `/tmp/nt001/calibration.log` compute `D = median(diff_median_1..20)`, `Q = median(|diff_median_k − D|)`, `diff_max = max(diff_median_1..20)`, `R = std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den` (recorded in the provenance comment; `1e-9` on Linux x86-64), `M = max(2·Q, 10·R)`, `A_seconds = max(0, diff_max) + M + 2·Q`, and `kAllowanceSeconds = std::ceil(A_seconds / 1.0e-8) * 1.0e-8`. Replace the step-4a `0.0` placeholder in the working tree with the computed `constexpr double kAllowanceSeconds = <rounded>;`; fill in the calibration provenance comment with every field of Requirement 5: calibration date, reference host string, `N = 20`, `R`, computed `D`, `Q`, `diff_max`, `M`, unrounded `A_seconds`, `G = 1.0e-8`, the rounded constant, the injector value `max(100, duration_cast<microseconds>(10*kAllowanceSeconds).count())` µs, the raw `diff_median_1..20` values, and the explicit note that this is the post-73 re-derivation superseding order 67's constant. The single commit at the end of step 4c contains the constant and the full provenance comment; no other commit precedes it. Rebuild and proceed to 4d.

   **Step 4d — 20-pass post-commit stability set (the only pass/fail acceptance step):** run the bench binary directly 20 times consecutively and confirm every run exits `0`. This is the review's acceptance method executed against the committed `kAllowanceSeconds`:

   ```bash
   for i in $(seq 1 20); do \
     ./build/test/iom_cpu_bench > /dev/null \
     || { echo "stability run $i FAILED"; exit 1; }; \
   done && echo "iom_cpu_bench: 20/20 stable"
   ```

   `ctest` is used only for the suite-level exit-0 check in step 6; the 20-run protocol here uses the direct binary because every `MESSAGE` line must reach stdout (ctest suppresses `MESSAGE` on passing runs).

5. Injected queued-path regression trips the relative gate. The injection site is **post-73** `src/cpu/device.cpp`'s `CpuQueue::copy` inline lambda, the non-no-op path that calls `copy_elements(source, destination)` synchronously on the submitting thread (order 73 Requirement 12's snippet). The order-67 site (`CpuQueue::execute`'s `if (!task.no_op)` branch at pre-73 `src/cpu/device.cpp:831-834`) does not exist after order 73 lands and must not be used. The implementer verifies the post-73 shape at task-execution time by reading order 73 Requirement 12 against the merged `src/cpu/device.cpp` and locating the inline copy call. The injection site is production code and so cannot reference `kAllowanceSeconds` directly; the numeric literal is substituted instead. Procedure:

   - Compute the numeric `injector_delay` from the already-committed `kAllowanceSeconds` value (read from the TU source after step 4c): `injector_delay = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double>(10 * kAllowanceSeconds)).count()` µs, with a floor of `max(100, injector_delay)` to honor Requirement 3's `max(100, 10·kAllowanceSeconds)` µs. Record the literal next to the edit (e.g. `// injector_delay = <n> us (derived from kAllowanceSeconds = <v> us at commit <hash>)`).
   - Temporarily edit `src/cpu/device.cpp`: add `#include <chrono>` to the include block (mirroring order 67's verification step; `<chrono>` is not currently included in the current tree's `src/cpu/device.cpp` — `src/cpu/device.cpp:3-15` lists `<array>`, `<condition_variable>`, `<cstdint>`, `<deque>`, `<exception>`, `<mutex>`, `<cstring>`, `<new>`, `<stdexcept>`, `<string>`, `<thread>`, `<utility>`, `<vector>` but not `<chrono>`) and insert `std::this_thread::sleep_for(std::chrono::microseconds(<literal>));` as the first statement of the post-73 inline `if (!no_op) { copy_elements(source, destination); }` body in `CpuQueue::copy`'s lambda.
   - Rebuild only the bench target and run it directly:

     ```bash
     cmake --build build -j --target iom_cpu_bench && ./build/test/iom_cpu_bench
     ```

   - Expected: the case fails at the self-relative `CHECK_MESSAGE`; the printed differential is ≈ `injector_delay` µs > `kAllowanceSeconds` (by Requirement 3's construction), and the printed no-op median stays within its usual (sub-µs to low-µs) range — proving the baseline is copy-path-specific and the gate is discriminating, not merely permissive. Revert the edit, rebuild, and confirm the binary exits `0` again with `git diff src/cpu/device.cpp src/` empty.

6. Full always-run set exits 0 — the restored suite-wide acceptance:

   ```bash
   ctest --test-dir build --output-on-failure
   ```

   Expected: `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, `iom_cpu_bench` all pass.

7. Final scope audit:

   ```bash
   git diff --stat include/ test/CMakeLists.txt CMakeLists.txt src/cpu src/cuda src/rocm src/sycl src/ttnn include/iom/iom.hpp include/iom/detail src/iom.cpp
   ```

   Expected: only `test/cpu/test_cpu_bench.cpp` shows non-zero changed lines (the `kAllowanceSeconds` value and its calibration provenance comment); every other path above shows zero changed lines. `test/CMakeLists.txt` shows zero changes — `iom_cpu_bench` remains unconditionally registered.
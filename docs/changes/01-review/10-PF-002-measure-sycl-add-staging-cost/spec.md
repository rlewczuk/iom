# Measure SYCL ADD staging cost

**Order:** 10
**Priority:** P2 — required independent measurement work
**Blocked by:** None
**Review source:** `cpp-inference-performance` — `whole-codebase checked-out main/HEAD ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb ("Update remote hosts"), clean working tree at review start`
**Finding:** PF-002
**Review area:** Performance
**Review severity:** medium
**Review verification:** hypothesis, confidence 78
**Review scope:** whole-codebase
**Backend scope:** SYCL
**Location:** `src/sycl/copy.cpp:622-763` — `SyclQueue::execute` staged ADD path

## Outcome

A controlled measurement produces a decisive conclusion about whether the SYCL ADD staging sequence is material on the configured Level Zero device. It reports warmed ADD latency/throughput and the absolute USM allocation/free, kernel-launch, memcpy, wait, host-loop, and overlap costs for representative and tail workloads, then applies a predeclared triage gate. If a gate is exceeded, the result opens a separate optimization task; if no production/representative ADD caller exists or all gates remain below threshold, the staging-cost hypothesis is falsified without prescribing an implementation change here.

## Current problem

`SyclQueue::execute` allocates three host-USM and three device-USM buffers at `src/sycl/copy.cpp:645-680`, launches byte-copy kernels at `:687-713`, enqueues three memcpys at `:714-716`, waits at `:717`, runs the scalar host loop `add_elements` at `:718-722`, copies the result back at `:723-734`, and waits again at `:738`. Cleanup frees all six allocations at `:746-751`. This creates an obvious full staging and synchronization mechanism for every accepted ADD request, but the review had no controlled Level Zero measurements of its wall-time share, allocation cost, launch count, transfer bytes, wait cost, host-loop time, or overlap loss. The mechanism is therefore a performance hypothesis, not a measured regression.

## Scope

- Measure the existing SYCL ADD path only; do not prescribe or implement a production kernel, staging reuse, allocator, synchronization, scheduling, or API change in this task.
- Use warmed ADD microbenchmarks on the configured Level Zero device for `{8,1024,1024}`, `{1,1024,1024}`, and tail `{2,33,65}`. Exercise BF16 and F32 for the large shapes and I64 for the tail/carrier-sensitive case, recording dtype and shape for every sample.
- Run at least 100 timed iterations per workload after a documented warm-up. Count and time USM alloc/free operations, kernel launches, memcpy bytes, `wait_and_throw` calls/duration, host-loop time, and overlap with adjacent independent work.
- Apply one predeclared triage gate: open a follow-up optimization task only when at least one measured phase consumes >=10% of ADD phase wall time, overlap falls by >=10 percentage points versus the independent-work baseline, or measured staging memory exceeds 5% of the declared benchmark memory budget. Below every threshold, or absent a representative production caller, falsifies material overhead for the measured path.

## Implementation references

- **Read:** `src/sycl/copy.cpp` — `SyclQueue::execute`, `add_view_staging_bytes`, `add_elements`, staging allocation/free, copy kernels, and both `wait_and_throw` regions; these are the measured phases.
- **Read:** `src/sycl/copy.cpp:293-329` — `launch_scatter_words` and `launch_gather_words`; distinguish ADD staging-copy launches from logical view-transfer helpers in the timeline.
- **Read:** `src/sycl/copy.cpp:455-501` — `SyclQueue`, `add_impl`, and queue submission/fence ownership; preserve asynchronous token semantics while measuring.
- **Read:** `README.md` and `docs/changes/002-eltwise-add/05-migrate-sycl-copy/spec.md` — configured SYCL/Level Zero setup and remote-development command conventions.
- **Tests:** `test/sycl/test_sycl_conformance.cpp` — existing SYCL ADD/conformance driver and configured device construction; use its real queue/device setup for the measurement scenario without weakening correctness cases.

## Requirements

- Keep this task hypothesis-only. Do not modify `src/sycl/copy.cpp`, public interfaces, production queue scheduling, or staging ownership to improve the measured result. A disposable out-of-tree measurement harness or profiler configuration may observe the path and add temporary measurement probes, but it must not become a committed optimization.
- Warm each workload before timing, then execute at least 100 ADD iterations with stable allocations and inputs. Separate warm-up/setup from measured samples and report median, p95, minimum, maximum, total elapsed time, and effective logical throughput.
- Measure BF16 and F32 at `{8,1024,1024}` and `{1,1024,1024}`, plus I64 at tail `{2,33,65}`. Record logical bytes, staging bytes, iteration count, device identity, Level Zero runtime, and compiler/runtime setup for each result.
- Report absolute counts and durations for all six USM allocations and corresponding frees per request, every staging-copy and output-copy kernel launch, every memcpy byte total, every `wait_and_throw` call and duration, and the host `add_elements` loop. Do not report percentages without the underlying count, byte, or time values.
- Capture an overlap timeline with an adjacent independent SYCL operation and a no-ADD baseline of that operation. Report overlap duration and percentage-point loss using the same warmed sample window; if the device cannot run the independent operation, record the limitation and do not claim overlap evidence.
- Declare the benchmark memory budget before measuring, retain it in the report, and apply the 5% staging-memory threshold exactly. Do not tune the budget or thresholds after observing results.
- Check for a real production/representative ADD caller and state its path. If none exists, mark the hypothesis falsified for lack of a material consumer regardless of synthetic timings.
- Conclude with exactly one of: gate exceeded (name the phase and open a separate optimization task), all gates below threshold (hypothesis falsified for the measured representative workloads), or no representative production caller (hypothesis falsified). Do not silently convert the suspected cost into an implementation requirement.

## Non-goals

- Do not implement a SYCL ADD kernel, staging reuse/pool, allocation reduction, wait removal, queue overlap change, host-loop replacement, or public fallback/path-selection API.
- Do not claim a performance regression from source inspection alone, extrapolate from CPU/local hardware, or treat a non-Level-Zero device as the configured accelerator result.
- Do not replace SYCL numerical, lifetime, error, or ownership conformance with a benchmark, and do not make benchmark thresholds fail generic correctness CTest runs.
- Do not open the optimization task automatically when a gate is exceeded; preserve the measurements and exact follow-up trigger for a separate decision.

## Acceptance criteria

- [ ] The report contains warmed BF16/F32 `{8,1024,1024}` and `{1,1024,1024}` results plus the I64 tail `{2,33,65}` result, each with at least 100 timed iterations and complete workload/device metadata.
- [ ] USM alloc/free counts, kernel-launch counts, memcpy bytes, `wait_and_throw` count/duration, host-loop time, latency/throughput, and overlap timeline are all reported as absolute measurements before derived percentages.
- [ ] The report names the production/representative ADD caller or explicitly records that none exists, and states that execution occurred on the configured Level Zero device.
- [ ] The predeclared gate is applied without threshold or budget changes after measurement: >=10% phase wall-time share, >=10 percentage-point overlap loss, or >5% of the declared staging-memory budget triggers a follow-up optimization task; below all gates is a falsifier.
- [ ] No production implementation change is required or smuggled into this hypothesis task, and the conclusion is explicitly marked measured, falsified, or limited by missing representative caller/evidence.

## Verification

- `(remote-development: SYCL host)` `.agents/skills/remote-development/scripts/remote-sync sycl <unique-task-id> && .agents/skills/remote-development/scripts/remote-exec sycl <unique-task-id> 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/sycl --target iom_sycl_conformance_tests && ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_conformance_tests$"'` — expected observation is a passing baseline conformance run and Level Zero GPU enumeration. This accelerator gate is remote and is not run for this specification-writing task.
- **Remote measurement scenario:** on the same synchronized remote workspace, run the disposable profiler/harness against BF16/F32 `{8,1024,1024}` and `{1,1024,1024}`, and I64 `{2,33,65}`, after warm-up for >=100 iterations each, with an adjacent independent operation and no-ADD baseline. Expected observation is a complete absolute-counter report and one unambiguous triage conclusion; local or non-Level-Zero execution is not evidence for this task.

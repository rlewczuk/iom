# Measure TTNN ADD host-emulation cost

**Order:** 09
**Priority:** P2 — required independent measurement work
**Blocked by:** None
**Review source:** `cpp-inference-performance` — `whole-codebase checked-out main/HEAD ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb ("Update remote hosts"), clean working tree at review start`
**Finding:** PF-001
**Review area:** Performance
**Review severity:** medium
**Review verification:** hypothesis, confidence 75
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `src/ttnn/copy.cpp:427-585` — TTNN ADD host-emulation loop in `ttnn_detail::add_planes` (called by the TTNN queue ADD path)

## Outcome

A controlled measurement produces a decisive conclusion about whether TTNN ADD host emulation is material on configured TTNN hardware: it reports warmed representative latency/throughput and the absolute host/device transfer, queue-finish, allocation, mutex, overlap, and scratch-space costs, then applies a predeclared triage gate. If any gate is exceeded, the report is the input for a separately opened optimization task; if no representative production ADD caller exists or every gate remains below threshold, the host-emulation hypothesis is falsified without prescribing an implementation change here.

## Current problem

The TTNN ADD path performs device-to-host reads through `ttnn::copy_to_host(..., blocking=true)` in `src/ttnn/copy.cpp:438-450`, decodes and adds every logical element with `detail::scalar_add` at `:522-525`, uploads output planes, and calls `mesh_command_queue(0).finish()` at `:580-582` before releasing staging leases. The owning `TtnnDevice` serializes these operations under its API mutex (`src/ttnn/device.cpp:277-299` and queue callers), while the path also uses per-plane host vectors/caches at `:435-437`. The mechanism is mechanically clear, but the review collected no warmed workload measurements, so its latency, bandwidth, synchronization, allocation, mutex, overlap, and scratch-space materiality are unverified. Treating the suspected cost as fact would conflate a code-path hypothesis with a measured performance conclusion.

## Scope

- Measure the existing TTNN ADD path only; do not prescribe or implement a production optimization, native kernel, fallback, batching change, or API change as part of this task.
- Use warmed representative ADD workloads with BF16 and F32 at `{8,1024,1024}` and `{1,1024,1024}`, plus one explicit carrier-backed integer case using I64 at `{1,1024,1024}`. Run at least 100 timed iterations per workload after a documented warm-up and report the exact iteration count.
- Measure and report latency and throughput, H2D and D2H bytes, `queue.finish` count and cumulative time, host allocation count, API-mutex hold time, overlap with adjacent independent work, and peak host scratch bytes. Report current-path counters as absolute counts, bytes, seconds, and bytes-at-peak before deriving percentages.
- Apply one predeclared triage gate: open a follow-up optimization task only when at least one measured phase consumes >=10% of ADD phase wall time, overlap falls by >=10 percentage points versus the independent-work baseline, or peak host scratch exceeds 5% of the declared benchmark host-memory budget. A result below every threshold is a falsification of material overhead for the measured representative path.

## Implementation references

- **Read:** `src/ttnn/copy.cpp` — `add_planes` host loop, `load`, `read`, `plane_at`, output upload, and queue-finish/release sequence; these are the measured critical-path phases.
- **Read:** `src/ttnn/device.cpp` — `TtnnDevice::api_mutex`, `TtnnQueue` submission/finish path, and TTNN queue ownership; attribute mutex hold and queue synchronization to the correct owner.
- **Read:** `src/ttnn/staging.hpp` — `TtnnHostStaging`, upload leases, retained allocations, and `g_host_staging_allocations`; use existing counters where available and report them absolutely rather than inferring allocation behavior from elapsed time.
- **Read:** `docs/BACKEND_CONTRACT.md` and `README.md:65-107` — TTNN ADD/storage behavior and the configured build/test commands; preserve the real-device requirement.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — `TTNN ADD broadcast, tail, transformed views, and aliases` and the representative wide ADD setup; use the same device construction and logical shapes for the measurement harness, without weakening conformance assertions.

## Requirements

- Keep this task hypothesis-only. Do not modify `src/ttnn/copy.cpp`, `src/ttnn/device.cpp`, public interfaces, or production scheduling to improve the measured result. A disposable out-of-tree measurement harness or profiler configuration may observe the path and may add temporary counters, but it must not become a production optimization or committed behavior change.
- Warm each workload before timing; then execute at least 100 ADD iterations with stable tensor allocation and input data. Separate setup/warm-up from measured samples and report median, p95, min, max, and total elapsed time, plus effective logical throughput.
- Measure BF16 and F32 for both `{8,1024,1024}` and `{1,1024,1024}`, and I64 for `{1,1024,1024}` as the carrier-backed integer case. Record dtype, shape, logical bytes, native plane count, and iteration count for every row.
- Count and time every H2D/D2H transfer attributable to the ADD request, every `mesh_command_queue(0).finish()`, host allocation event and byte total, API-mutex hold interval, and host scratch allocation/peak. Do not report only percentages: include absolute values and the method used to attribute each event.
- Capture an overlap timeline with an adjacent independent TTNN operation and a no-ADD baseline of the same independent operation. Report overlap duration and percentage-point loss using the same warmed sample window; if the backend cannot create the independent operation, record that limitation and do not claim overlap evidence.
- Declare the host-memory budget before the measurement run, retain it in the report, and apply the 5% scratch threshold exactly. Do not tune the budget after observing results.
- Check the configured codebase for a real production/representative ADD caller and state the caller path. If no such caller exists, mark the hypothesis falsified for lack of a material consumer regardless of synthetic timing.
- Conclude with exactly one of: gate exceeded (name the phase and open a separate optimization task), all gates below threshold (hypothesis falsified for the measured representative workloads), or no representative production caller (hypothesis falsified). Do not silently convert a hypothesis into an implementation requirement.

## Non-goals

- Do not implement a TTNN native ADD kernel, alter host emulation, remove synchronization, change staging ownership, change API-mutex scope, or add a fallback/path-selection API.
- Do not claim a performance regression from source inspection alone, extrapolate from CPU/local hardware, or use unconfigured TTNN hardware as evidence.
- Do not replace numerical conformance, lifetime/error tests, or the independent oracle with a benchmark, and do not make benchmark thresholds fail generic correctness CTest runs.
- Do not open the optimization task automatically when a gate is exceeded; record the measured conclusion and the exact follow-up trigger only.

## Acceptance criteria

- [ ] The report contains warmed BF16/F32 `{8,1024,1024}` and `{1,1024,1024}` results plus the I64 carrier-backed `{1,1024,1024}` result, each with at least 100 timed iterations and reproducible workload metadata.
- [ ] Latency/throughput, H2D/D2H bytes, finish count/time, host allocation count/bytes, API-mutex hold time, overlap timeline, and peak host scratch are all reported as absolute measurements before percentages or derived conclusions.
- [ ] The report names the production/representative ADD caller or explicitly records that none exists, and states the configured TTNN device and runtime.
- [ ] The predeclared gate is applied without threshold or budget changes after measurement: >=10% phase wall-time share, >=10 percentage-point overlap loss, or >5% of the declared host-memory budget is sufficient to trigger a follow-up optimization task; below all gates is a falsifier.
- [ ] No production implementation change is required or smuggled into this hypothesis task, and the conclusion is explicitly marked measured, falsified, or limited by missing representative caller/evidence.

## Verification

- `(remote-development: TTNN host)` `.agents/skills/remote-development/scripts/remote-sync ttnn <unique-task-id> && .agents/skills/remote-development/scripts/remote-exec ttnn <unique-task-id> 'cmake -S . -B build/ttnn -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF && cmake --build build/ttnn --target iom_ttnn_conformance_tests && ctest --test-dir build/ttnn --output-on-failure -R "^iom_ttnn_conformance_tests$"'` — expected observation is a passing baseline conformance run on the configured TTNN device. This accelerator gate is remote and is not run for this specification-writing task.
- **Remote measurement scenario:** on that same synchronized remote workspace, run the disposable profiler/harness against BF16/F32 `{8,1024,1024}`, BF16/F32 `{1,1024,1024}`, and I64 `{1,1024,1024}` after warm-up, for >=100 iterations each, with an adjacent independent operation and no-ADD baseline. Expected observation is a complete absolute-counter report and one unambiguous triage conclusion; source inspection or a local run is not an accelerator measurement.

# Measure whether per-plane host-transfer launches are materially costly

**Order:** 06
**Priority:** P2 — establish materiality before considering a production batching change
**Blocked by:** None
**Review source:** `cpp-inference-performance` — `whole-codebase reviewed state: branch main, clean HEAD 86533aef347935405cb86d465cc5489f5c3d530a (Remove review files.)`
**Finding:** `PF-001`
**Review area:** Performance
**Review severity:** medium
**Review verification:** hypothesis, confidence 75
**Review scope:** whole-codebase
**Backend scope:** CUDA + ROCm/HIP + SYCL standard tiled host-transfer paths
**Location:** `src/shared/standard_tiled_copy.inl` — `launch_view_transfer`/`synchronous_transfer_impl`; `src/sycl/copy.cpp` — `launch_view_transfer`

## Outcome

Measure and falsify the suspected per-plane host-transfer launch overhead before making any production batching change. Using fixed release/power/device conditions and F32 shapes `{P,1,32,64}` for `P = 1, 8, 32, 64, 128`, determine separate upload/download medians, exact bytes/padding, and launch counts on CUDA, ROCm, and SYCL. Materiality is established only when launch/driver cost is at least 10% of representative high-P end-to-end time and an ephemeral one-launch prototype improves high-P median end-to-end latency by at least 15% including metadata preparation, with no extra waits/allocations and no P=1 regression; otherwise finalize with no implementation.

## Current problem

The standard CUDA/ROCm path loops over every logical plane in `launch_view_transfer` and submits one scatter or gather kernel per plane, then synchronizes once. SYCL independently submits one `parallel_for` per plane and waits once. The queued device-to-device counterpart already uses one metadata-driven grid-stride dispatch over all words, making P-dependent host-transfer launch fragmentation a concrete mechanism. However, no accelerator timing/profile or P sweep has established that launch/driver work is material relative to transfer time, and metadata preparation/ownership may affect a one-launch design. Existing smoke/conformance results verify behavior only, not this hypothesis.

## Scope

- Run a throwaway measurement/falsification experiment on suitable remote CUDA, ROCm/HIP, and SYCL hosts; do not commit a production batching change or permanent benchmark solely from this task.
- Warm up and separately time uploads and downloads for F32 `{P,1,32,64}` at P values 1, 8, 32, 64, and 128 with repeated samples/medians under fixed release, power, and device conditions. Record logical bytes, padded bytes, end-to-end latency, and launch counts.
- Build an ephemeral one-launch metadata-driven scatter/gather prototype only for measurement, including metadata preparation in timing. Accept any later implementation follow-on only for backends that meet all thresholds; reject/finalize without implementation when they do not.

## Implementation references

- **Modify:** None in the repository; use a throwaway focused driver/prototype outside the checked-in source, and remove it after measurement. This accepted task is measurement/falsification only.
- **Read:** `src/shared/standard_tiled_copy.inl` — `launch_view_transfer`, `synchronous_transfer_impl`, and queued `launch_grid_stride_copy`; establish current P launches, one final synchronization, staging ownership, and the existing metadata representation.
- **Read:** `src/sycl/copy.cpp` — SYCL `launch_view_transfer`, `region_from_host`, `region_to_host`, and `launch_calls.kernel_launched`; count SYCL host-transfer submissions and waits through the existing seam.
- **Read:** `src/cuda/copy.cu` and `src/rocm/copy.hip` — shared standard-path instantiation and backend policies; preserve backend-specific behavior in any ephemeral prototype.
- **Tests:** `test/sycl/test_sycl_smoke.cpp` — existing launch-count seam for queued copies; backend conformance copy cases — exact logical bytes, padding, errors, and transfer behavior.

## Requirements

- Measure upload and download separately, after warm-up, over exactly F32 `{P,1,32,64}` with P=1/8/32/64/128, repeated samples and medians. Hold release build, power mode, device selection, and relevant runtime settings fixed; report logical and padded byte counts for every case.
- Current launch counts MUST be observed as P scatter/gather kernel launches per CUDA/ROCm host transfer and P SYCL `parallel_for` submissions through `iom::sycl_detail::launch_calls.kernel_launched`. Use `nsys profile --trace=cuda,nvtx,osrt --sample=none --stats=true` for CUDA and `rocprofv3 --hip-trace --kernel-trace --output-file <trace> -- <driver>` for ROCm; use the existing SYCL seam plus timing for SYCL.
- The ephemeral one-launch prototype MUST include metadata preparation in end-to-end timing and compare same-byte uploads/downloads against the current implementation. It MUST preserve exact logical output, padding, and error behavior in the measured cases and add no extra waits or allocations.
- Materialize no production batching implementation from a sub-threshold result. If only some backends pass, any later task MUST be backend-scoped; this task itself ends with measurement/falsification and no permanent implementation.

## Non-goals

- Do not modify `src/shared/standard_tiled_copy.inl`, `src/sycl/copy.cpp`, CUDA/ROCm/SYCL production code, CPU loops, TTNN native per-plane representation, staging-pool ownership, queued device-to-device semantics, or public synchronous transfer behavior.
- Do not add a scheduler, cache, generic runtime layer, permanent benchmark, extra synchronization, or allocation to chase an unproven gain.
- Do not infer materiality from launch count alone or from the absence of existing accelerator profiles; vendor/runtime constraints that require per-plane dispatch are valid falsification evidence.

## Acceptance criteria

- [ ] The experiment reports separate upload/download repeated medians, logical/padded bytes, and launch counts for all five P values on each available CUDA, ROCm, and SYCL backend under fixed release/power/device conditions; P=1 is measured as a no-regression baseline.
- [ ] A representative high-P case is material only if profiler-attributed launch/driver cost is at least 10% of end-to-end transfer time and the one-launch prototype improves high-P median end-to-end latency by at least 15% after metadata preparation, with exact logical/padding/error parity and no extra waits/allocations.
- [ ] If either threshold fails, the task is finalized as rejected/no implementation. If thresholds pass only for a subset, the result explicitly scopes any separate follow-on to those backends and does not change production code here.

## Verification

- `nsys profile --trace=cuda,nvtx,osrt --sample=none --stats=true -- <fixed-release-measurement-driver>`
- `rocprofv3 --hip-trace --kernel-trace --output-file <trace> -- <fixed-release-measurement-driver>`
- Run the same fixed-release measurement driver on SYCL while resetting and reading `iom::sycl_detail::launch_calls.kernel_launched`; compare current and ephemeral one-launch paths for all P values and both directions. The supplied ledger reports no profiler/P sweep; no TTNN padding, SafeTensors cap, sanitizer, static-analysis, or other unrelated gate substitutes for this experiment, and no performance measurement has been run.
- Baseline validation ledger only (not PF-001 coverage): local GNU 15.2 CPU build/tests/bench/conformance passed 4/4; remote CUDA 13.2.78 smoke+conformance, ROCm HIP Clang 23 smoke+conformance, SYCL IntelLLVM 2026.1 smoke+conformance on two enumerated Arc Pro B60 Level Zero GPUs, and TTNN smoke+conformance each passed 2/2. No profiler/P sweep or ephemeral prototype measurement has been run.

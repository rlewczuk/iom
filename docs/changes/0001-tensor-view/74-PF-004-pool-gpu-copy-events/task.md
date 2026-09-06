# Task: Pool CUDA/ROCm queued-copy events

Status: done

Implemented the PF-004 leaf task on branch `run-task/0001-tensor-view--74-PF-004-pool-gpu-copy-events`.

## Delivered

- Added shared `EventRingState<Policy>` with lazy 16-event creation, blocking reuse, cached worker results, metadata-slot attachment/release, and destructor-only native event destruction.
- Replaced CUDA/ROCm per-submission fence resources and intrusive refcounts with the 32-byte `EventLeaseWithFailure` shared-state capture.
- Moved `event_create` fault consumption to ring acquisition and preserved retained-failure, event-record, staged-worker, and queue teardown semantics.
- Added the 240-byte rank-10 inline metadata kernel argument and retained pooled metadata/H2D fallback for rank 11 and above.
- Added CUDA/ROCm ring smoke coverage and rank-10/rank-11 full-storage conformance coverage.

## Verification

- CUDA remote host `bv1`: implementation configured and built with CUDA 13.2; CUDA smoke and conformance tests passed; common `iom_tests` passed; full CTest excluding the unrelated CPU benchmark passed (6/6); Compute Sanitizer memcheck for queue-destruction fencing passed with `ERROR SUMMARY: 0 errors`.
- ROCm remote host `bv2`: implementation configured and built with ROCm HIP; ROCm smoke and conformance tests passed; common `iom_tests` passed; full CTest excluding the unrelated CPU benchmark passed (6/6).
- Baseline remote runs passed before the change. Timing helper observations, median of five runs:
  - CUDA: sequential 16x16 submit+wait 14.0811 us -> 13.871 us; 64-submit burst 717.781 us -> 455.11 us; main-thread allocation counter 7 -> 6.
  - ROCm: sequential 62.0429 us -> 53.8954 us; 64-submit burst 1350.65 us -> 1555.61 us; main-thread allocation counter 15 -> 11.
  - The allocation counter includes registry/worker bookkeeping and is observational, not a latency gate.
- CUDA Nsight profiling of the rank-boundary case observed one metadata H2D copy for the pooled rank-11 submission and no metadata H2D copy for the inline rank-10 submission; event creation occurred once per queue, with destruction deferred to queue-state teardown. ROCm profiler tooling was unavailable (`rocprof`/`rocprofv2` absent; `rocgdb` only).
- Source audits passed: forbidden per-task resource/refcount symbols absent; shared ring contains no CUDA/HIP symbols; native event create/destroy calls remain policy-owned; patch whitespace and file boundary checks passed.

Note: unfiltered remote CTest also runs `iom_cpu_bench`; it failed on both hosts' transient throughput floors, while all six non-benchmark tests passed on each backend configuration. Remote mirrors were cleaned after verification.

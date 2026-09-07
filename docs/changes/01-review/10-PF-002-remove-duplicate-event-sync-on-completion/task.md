**Status:** done

## Summary

Duplicate CUDA/ROCm event synchronization on normal completion is removed: the per-submission record gains a synchronized_on_complete_ marker set only when on_worker_complete event synchronization succeeds; on_worker_destroy skips its extra activate+wait on marked records while retaining the cleanup wait for queue drain, registration-failure, and failed/partially-observed completions. Reset-on-acquire is implicit via fresh record construction per pooled-event reuse.

## Verification

- ctest -R "iom_cuda_(smoke|conformance)_tests" on bv1 — 2/2 passed (conformance 35.42s)
- ctest -R "iom_rocm_(smoke|conformance)_tests" on bv2 — 2/2 passed (conformance 39.40s)
- combined wave-2 train: CUDA 3/3, ROCm 3/3

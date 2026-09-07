**Status:** done

## Summary

CUDA and ROCm queue protocols are consolidated into one policy-templated detail::GpuQueue<Policy> with unchanged behavior, diagnostics, sync/event counts, per-submission fence semantics (02-ST-001) and one-wait sync-skip (10-PF-002); factory signatures and backend labels unchanged.

## Verification

- ctest -R "iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests" on bv1 — 3/3 passed (36.37s)
- ctest -R "iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests" on bv2 — 3/3 passed (40.52s)
- combined wave-3 train: CUDA/ROCm 3/3 each

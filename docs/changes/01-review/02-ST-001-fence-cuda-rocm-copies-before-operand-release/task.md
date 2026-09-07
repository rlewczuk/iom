**Status:** done

## Summary

CUDA/ROCm copies are fenced before operand release: each submission owns a shared completion record whose pooled event slot is retained until the last fence reference drops (registry entries and destructor snapshots); fences read their own per-submission result so slot reuse never alters an earlier fence result, and failed submissions report retained failure through their fence. Deterministic conformance covers pre-wait destruction (interleaving-independent end-state contracts) and queue-destroy invalidation quarantine with exact once-only storage accounting.

## Verification

- ctest -R "iom_cuda_(smoke|conformance)_tests" on bv1 (RTX 5090) — 2/2 passed, conformance 38.64s
- ctest -R "iom_rocm_(smoke|conformance)_tests" on bv2 — 2/2 passed, conformance 44.61s
- combined train: cuda+coexistence 3/3 on bv1, rocm+coexistence 3/3 on bv2

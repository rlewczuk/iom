**Status:** done

## Summary

TTNN queue completion is batched: complete_task collects the ready contiguous batch via an executed_seq_ marker (proof the native work reached the mesh), performs one finish_locked mesh finish per batch, then releases/invalidates entries and completes every token in submission order with per-member failure precedence. Two conformance cases pin one-finish-per-ready-batch bounds and the failed-batch-finish prefix semantics; harness finish-count seam added under IOM_ENABLE_TESTING.

## Verification

- ctest -R "^(iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests)$" on bv1 — 3/3 passed, conformance 18.23s (incl. ST-002 failure-injection cases re-verified)
- combined wave-2 train on bv1 — TTNN 3/3 passed (coexistence 1.80s)

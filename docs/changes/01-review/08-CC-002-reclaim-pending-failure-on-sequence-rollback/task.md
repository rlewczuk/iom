**Status:** done

## Summary

A pending failure committed before a synchronous submit throw is reclaimed from pending_failures_ on sequence rollback, so a reissued sequence cannot be taken over by a stale complete() and rethrown. Regression test asserts the reclaimed sequence reissue completes successfully with repeated wait() (a surviving pending failure would be rethrown); existing submit/rollback and concurrent-gap cases unchanged.

## Verification

- ctest -R "^iom_tests$" and "^iom_cpu_tests$" local CPU — 2/2 passed
- combined wave-2 train: iom_tests green on CPU and all accelerator builds; CUDA/ROCm/SYCL/TTNN conformance+coexistence 3/3 each

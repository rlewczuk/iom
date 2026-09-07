**Status:** done

## Summary

Transfer-stream synchronization is no longer duplicated: Scope records explicit successful synchronization and skips the destructor stream wait, keeping exactly one sync per successful CUDA/ROCm host transfer; poisoned and errored paths still synchronize or drop the stream before reuse or destruction. Pool idle accounting and byte/error behavior unchanged.

## Verification

- ctest -R "iom_cuda_(smoke|conformance)_tests" on bv1 — 2/2 passed (conformance 35.16s)
- ctest -R "iom_rocm_(smoke|conformance)_tests" on bv2 — 2/2 passed (conformance 39.54s)
- combined wave-2 train: CUDA 3/3, ROCm 3/3, SYCL 3/3

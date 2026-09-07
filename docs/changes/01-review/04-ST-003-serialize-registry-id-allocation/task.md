**Status:** done

## Summary

Registry queue-id and source/destination entry-pair allocation is serialized behind an explicit RegistryState allocation mutex shared by all accelerator backends; concurrent queue creation and copy submission reserve unique ids atomically as pairs. Barrier-based concurrent coexistence scenario runs under every enabled backend with counting-allocator free/quarantine assertions, plus an always-running concurrent RegistryState unit test asserting unique non-skipped queue ids and contiguous non-overlapping entry-id pairs.

## Verification

- ctest -R "^iom_tests$" local CPU — passed (includes concurrent RegistryState unit test)
- iom_backend_coexistence_tests on combined train: CUDA 0.18s, ROCm 0.16s, SYCL 0.78s, TTNN 1.79s — all passed 3/3 per backend
- accelerator conformance suites on combined train (CUDA/ROCm/SYCL/TTNN) — all passed

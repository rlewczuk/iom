**Status:** done

## Summary

Test-only backend-local native-allocation instrumentation for CUDA (cuMemAlloc/cuMemFree), ROCm (hipMalloc/hipFree), and SYCL (sycl::malloc_device/free): attempt records completed with success/failure, byte count, operation class, lifecycle phase, allocation/free kind; data/metadata backing, operation-metadata, staging, and setup-resource classification at the IOM call sites; stable allocate/free pairing for transient-churn visibility; restorable hooks, no production global state, no public API change

## Verification

- ctest --test-dir <build> --output-on-failure -R "^iom_cuda_smoke_tests$" (remote CUDA host bv1, RTX 5090) — Passed 0.57s
- ctest --test-dir <build> --output-on-failure -R "^iom_rocm_smoke_tests$" (remote ROCm host bv2, Radeon AI PRO R9700) — Passed 0.90s
- ctest --test-dir <build> --output-on-failure -R "^iom_sycl_smoke_tests$" (remote SYCL host bv2, Level-Zero Intel Arc Pro B60) — Passed 0.89s

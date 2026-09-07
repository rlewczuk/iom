**Status:** done

## Summary

Physical-storage oracle and CPU StorageModel use an independent canonical tile-slot encoder instead of delegating to the production mapper; {2,3,17,33} added to conformance shape coverage; perturbed-oracle negative fixture still fails as designed.

## Verification

- ctest -R "^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$" local CPU — 2/2 passed (conformance 28.69s)
- accelerator conformance suites on combined train (CUDA/ROCm/SYCL/TTNN) exercising the independent oracle — all passed

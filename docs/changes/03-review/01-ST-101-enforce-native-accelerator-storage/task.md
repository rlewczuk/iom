**Status:** done

## Summary

Enforced CUDA and ROCm unmanaged native storage provenance at tensor construction with owner checks, synchronous rejection, and exact-once allocator rollback; documented factory preconditions.

## Verification

- remote CUDA verification: cmake --build build-cuda --target iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-cuda -R "iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests" --output-on-failure — 3/3 tests passed on CUDA 13.2.78.
- remote ROCm verification: cmake --build build-rocm --target iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-rocm -R "iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests" --output-on-failure — 3/3 tests passed on ROCm HIP Clang 23.
- final-train combined backend verification repeated CUDA and ROCm smoke/conformance plus coexistence — 3/3 tests passed on each backend.

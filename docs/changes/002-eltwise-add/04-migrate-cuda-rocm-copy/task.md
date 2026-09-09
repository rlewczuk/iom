**Status:** done

## Summary

Preserved the shared policy-templated CUDA/ROCm copy queue through the protected OID hook and initialized its registry-state lifetime ownership.

## Verification

- remote-exec cuda 002-eltwise-add-04-cuda: cmake configure/build iom_cuda_smoke_tests — CUDA library and smoke executable built successfully
- remote-exec cuda 002-eltwise-add-04-cuda: ctest --test-dir build/cuda --output-on-failure -R ^iom_cuda_smoke_tests$ — CUDA smoke test passed
- remote-exec rocm 002-eltwise-add-04-rocm: cmake configure/build iom_rocm_smoke_tests — ROCm library and smoke executable built successfully
- remote-exec rocm 002-eltwise-add-04-rocm: ctest --test-dir build/rocm --output-on-failure -R ^iom_rocm_smoke_tests$ — ROCm smoke test passed

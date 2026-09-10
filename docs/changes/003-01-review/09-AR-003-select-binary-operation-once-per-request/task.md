**Status:** done

## Summary

Selected operation-specialized CUDA/ROCm kernels and TTNN binary traversal once per request, removed GPU metadata and TTNN internal operation dispatch, and added explicit TTNN template instantiations for the out-of-line binary traversal.

## Verification

- remote-sync cuda ar003-cuda-final && remote-exec cuda ar003-cuda-final cmake --build build --target iom_cuda_conformance_tests && ctest --test-dir build --output-on-failure -R ^iom_cuda_conformance_tests$ — 22 test cases and 131,641 assertions passed
- remote-sync rocm ar003-rocm && remote-exec rocm ar003-rocm cmake --build build --target iom_rocm_conformance_tests && ctest --test-dir build --output-on-failure -R ^iom_rocm_conformance_tests$ — 22 test cases and ROCm conformance passed
- remote-sync ttnn ar003-ttnn && remote-exec ttnn ar003-ttnn cmake --build build --target iom_ttnn_conformance_tests && ctest --test-dir build --output-on-failure -R ^iom_ttnn_conformance_tests$ — TTNN conformance passed after explicit add/mul/sub/div binary_planes instantiations repaired the initial link failure
- Temporary CUDA Add-to-Mul dispatch mutation failed iom_cuda_conformance_tests: 5 test cases failed with 36,258 failed assertions; restored Add selection and reran clean CUDA conformance successfully
- Source inspection found no BinaryMetadata operation field, no metadata.operation device switch, no TTNN request.operation per-element switch, and only execute-level operation selection before templated traversal

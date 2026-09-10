**Status:** done

## Summary

Unified host and CUDA/ROCm compile-time binary scalar operations behind one shared format, decode, arithmetic, and encode codec; retained established NaN, infinity, rounding, and storage behavior.

## Verification

- cmake --build build/review-cpu --target iom_scalar_add_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R "^(iom_scalar_add_tests|iom_backend_conformance_cpu_tests)$" — both CPU targets passed
- remote-sync cuda 003-01-review-07-ar002 && remote-exec cuda ... iom_cuda_conformance_tests --test-case="CUDA binary conformance: ADD MUL SUB DIV real queue" — 46,190 assertions passed
- remote-sync rocm 003-01-review-07-ar002 && remote-exec rocm ... iom_rocm_conformance_tests --test-case="ROCm binary conformance: ADD MUL SUB DIV real queue" — 45,958 assertions passed
- Temporary F32 codec mutation (23 fraction bits to 22) made iom_scalar_add_tests fail 22 assertions, CUDA fail 3,095 assertions, and ROCm fail 3,095 assertions; restored exact format

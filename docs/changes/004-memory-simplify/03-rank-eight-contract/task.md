**Status:** done

## Summary

Full-shape rank interval 2-8 enforced at TensorShape construction, TensorSpec::validate, Tensor::Tensor, with_leading_dimensions, TensorView slice/select/permute/reshape_leading, and binary result validation — rank nine rejected with std::invalid_argument / OidError::InvalidArgument before allocation, registration, sequence/token acceptance, metadata upload, or native dispatch; validation precedence preserved; leading-dimension spans are not full shapes; tests migrated to rank-eight success and rank-nine rejection in test_iom.cpp, shared run_binary_rank_boundary_conformance, and CUDA/ROCM fixtures

## Verification

- cmake --build build --target iom_tests iom_backend_conformance_cpu_tests — clean
- ctest --test-dir build --output-on-failure -R "^(iom_tests|iom_backend_conformance_cpu_tests)$" — 2/2 passed (0.07s, 30.46s)
- ctest --test-dir build --output-on-failure -R "^iom_cuda_conformance_tests$" (remote bv1, RTX 5090) — Passed 34.57s
- ctest --test-dir build --output-on-failure -R "^iom_rocm_conformance_tests$" (remote bv2, Radeon AI PRO R9700) — Passed 40.99s
- ctest --test-dir build --output-on-failure -R "^iom_sycl_conformance_tests$" (remote bv2, Level-Zero Intel Arc Pro B60) — Passed 7.07s
- TTNN conformance not run: tt-nn CMake SDK (tt-nnConfig.cmake) is absent on the configured host bv1, so TTNN is not an enabled backend in this environment; per project rule optional backends fail configuration when SDK unavailable

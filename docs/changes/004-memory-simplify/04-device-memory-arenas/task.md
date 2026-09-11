**Status:** done

## Summary

Standard-GPU devices (CUDA/ROCm/SYCL) reserve exactly two fixed 32-aligned disjoint native backings at factory setup (tensor data ListAllocator + device-wide FixedSizeAllocator of exactly 4*C 512-byte slots) with checked 4*C*512 sizing, transactional RAII rollback, and bad_alloc/overflow_error/invalid_argument mapping; tensor create/destroy suballocate/free data-arena ranges under device-boundary bookkeeping locks with interior-pointer identity validation; QueueConfig + DeviceMemoryConfig cut over to all five factories (CPU keeps borrowed allocator, TTNN keeps native per-plane storage) with no legacy GPU overload; every caller, test, example, and tools/sycl_add_staging_measure.cpp migrated; task-01 seam classifies the two setup backings as data/metadata and focuses the no-churn/rollback/fragmentation/concurrency instrumentation tests

## Verification

- ctest --test-dir build --output-on-failure -R "^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$" (local) — 3/3 passed (0.07s, 0.15s, 43.65s)
- ctest --test-dir build --output-on-failure -R "^iom_cuda_smoke_tests$" (remote bv1, RTX 5090) — Passed 0.31s
- ctest --test-dir build --output-on-failure -R "^iom_cuda_conformance_tests$" (remote bv1) — Passed 34.69s
- ctest --test-dir build --output-on-failure -R "^iom_backend_coexistence_tests$" (remote bv1) — Passed 0.21s
- ctest --test-dir build --output-on-failure -R "^iom_rocm_smoke_tests$" (remote bv2, Radeon AI PRO R9700) — Passed 0.93s
- ctest --test-dir build --output-on-failure -R "^iom_rocm_conformance_tests$" (remote bv2) — Passed 40.22s
- ctest --test-dir build --output-on-failure -R "^iom_backend_coexistence_tests$" (remote bv2) — Passed 0.19s
- ctest --test-dir build --output-on-failure -R "^iom_sycl_smoke_tests$" (remote bv2, Level-Zero Intel Arc Pro B60) — Passed 0.24s
- ctest --test-dir build --output-on-failure -R "^iom_sycl_conformance_tests$" (remote bv2) — Passed 6.52s
- ctest --test-dir build --output-on-failure -R "^iom_backend_coexistence_tests$" (remote bv2) — Passed 0.78s
- focused instrumented arena tests (in the three smoke binaries): exactly two setup backing calls (data+metadata), zero post-setup tensor create/destroy native calls, disjoint 32-aligned domains, rollback at each injected setup failure, fragmentation/reuse/coalescing, concurrent bookkeeping race-freedom — all passed
- TTNN gates not run: tt-nn CMake SDK (tt-nnConfig.cmake) absent on the configured host bv1; TTNN is not an enabled backend in this environment

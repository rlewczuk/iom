**Status:** done

## Summary

Backend-neutral raw workspace contract: RawWorkspace (non-copyable/non-movable, exact-Device ownership) + RawWorkspaceView (empty default, copy-constructible, non-retargetable, checked owner-absolute 32-aligned subranges) + Device::create_workspace; standard-GPU positive creation suballocates the existing data arena with no new native backing and std::bad_alloc on exhaustion, CPU/TTNN accept empty only; pure DeviceOps binary and TensorView host-transfer workspace-requirement queries with exact {0,1} / SYCL checked staging-sum / {compute_staging_size,32} results and zero side effects; shared validation plus exclusive completion-controlled workspace-range lease primitives on the outstanding-work registry with rollback, overlap ResourceExhausted, and unknown-completion quarantine

## Verification

- ctest --test-dir build --output-on-failure -R "^(iom_tests|iom_backend_conformance_cpu_tests)$" (leaf worktree, 1089a52b) — 2/2 passed (0.07s, 30.33s)
- ctest -R "^iom_cuda_smoke_tests$" / "^iom_cuda_conformance_tests$" / "^iom_backend_coexistence_tests$" (remote bv1) — Passed 0.57s / 35.96s / 0.19s
- ctest -R "^iom_rocm_smoke_tests$" / "^iom_rocm_conformance_tests$" / "^iom_backend_coexistence_tests$" (remote bv2) — Passed 0.92s / 40.14s / 0.18s
- ctest -R "^iom_sycl_smoke_tests$" / "^iom_sycl_conformance_tests$" / "^iom_backend_coexistence_tests$" (remote bv2) — Passed 0.24s / 6.52s / 0.78s
- combined train gates: local iom_tests+iom_cpu_tests+iom_backend_conformance_cpu_tests 3/3; CUDA smoke 0.32s + conformance 35.93s; ROCm smoke 0.95s + conformance 39.07s; SYCL smoke 0.18s + conformance 6.66s; coexistence Passed on CUDA/ROCM/SYCL after the four-live-queue migration
- TTNN gates not run: tt-nn CMake SDK absent on configured host bv1; TTNN is not an enabled backend in this environment

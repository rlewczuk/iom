**Status:** done

## Summary

Factored shared GpuQueue lifecycle and operation definitions into private included fragments while preserving the single declaration, template visibility, and backend source registrations.

## Verification

- remote-sync cuda 005-split-big-files-20-cuda and remote-sync rocm 005-split-big-files-20-rocm — synchronized exact worktree to both configured accelerator hosts.
- CUDA remote configure/build targets libiom iom_cuda iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests — all targets built successfully; exact ctest regex — 3/3 passed.
- ROCm remote configure/build targets libiom iom_rocm iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests — all targets built successfully; exact ctest regex — 3/3 passed.
- wc -l src/shared/gpu_queue.hpp src/shared/gpu_queue_lifecycle.inl src/shared/gpu_queue_operations.inl — 269, 177, and 284 lines, all within the 499-line cap.

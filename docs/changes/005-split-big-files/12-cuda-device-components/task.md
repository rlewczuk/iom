**Status:** done

## Summary

Split CUDA device core from native tensor/workspace ownership into private, budgeted translation units without changing factory, context, arena, registry, or queue behavior.

## Verification

- remote-sync cuda 005-split-big-files-12 — exact CUDA worktree synchronized to bv1:agent-work/iom/005-split-big-files-12.
- remote-exec cuda 005-split-big-files-12 cmake configure and cmake --build build/split-cuda --target libiom iom_cuda iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests — configure and all targets built successfully with CUDA 13.2.78.
- remote-exec cuda 005-split-big-files-12 ctest --test-dir build/split-cuda --output-on-failure -R ^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$ — 3/3 tests passed.
- wc -l src/cuda/device.cpp src/cuda/device_tensor.cpp src/cuda/device_internal.hpp — counts 426, 179, and 106, within 480/220/220 budgets.

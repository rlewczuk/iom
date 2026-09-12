**Status:** done

## Summary

Split ROCm device core from native tensor/workspace ownership, reused the authoritative HIP error helpers, and preserved runtime, arena, registry, quarantine, and factory boundaries.

## Verification

- remote-sync rocm 005-split-big-files-13 — exact ROCm worktree synchronized to bv2:agent-work/iom/005-split-big-files-13.
- remote-exec rocm 005-split-big-files-13 cmake configure and cmake --build build/split-rocm --target libiom iom_rocm iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests — configure and all targets built successfully with HIP/ROCm.
- remote-exec rocm 005-split-big-files-13 ctest --test-dir build/split-rocm --output-on-failure -R ^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$ — 3/3 tests passed.
- wc -l src/rocm/device.cpp src/rocm/device_tensor.cpp src/rocm/device_internal.hpp — counts 404, 193, and 113, within 480/220/220 budgets.
- moved-definition search in src/rocm/device.cpp — no moved RocmTensor/RocmWorkspace/create/validator definitions or anonymous HIP helper definitions remain; retained calls use rocm_detail::check_hip.

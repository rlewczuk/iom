**Status:** done

## Summary

Common allocator construction now checks address-range, alignment, and payload arithmetic before derived state, preserving typed errors and valid allocation behavior.

## Verification

- local CPU cmake/ctest gate on allocator leaf worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed, including allocator cases
- remote-sync/remote-exec cuda 07-CC-001-allocator-checked-constructor-arithmetic — CUDA build and smoke, conformance, and coexistence tests passed
- remote-sync/remote-exec rocm 07-CC-001-allocator-checked-constructor-arithmetic — ROCm build and smoke, conformance, and coexistence tests passed
- remote-sync/remote-exec sycl 07-CC-001-allocator-checked-constructor-arithmetic with setvars override and sycl-ls — Level Zero devices enumerated; SYCL build and smoke, conformance, and coexistence tests passed
- combined final-train local CPU and TTNN/CUDA/ROCm/SYCL remote gates — every configured build succeeded and all selected smoke, conformance, and coexistence tests passed

**Status:** done

## Summary

CUDA, ROCm, and SYCL metadata-slot growth now relies on slot-local completion and avoids redundant whole-queue synchronization while preserving pooled-copy behavior.

## Verification

- local CPU cmake/ctest gate on metadata leaf worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- remote-sync/remote-exec cuda 08-PF-002-metadata-pool-growth-without-queue-sync — CUDA build and smoke, conformance, and coexistence tests passed
- remote-sync/remote-exec rocm 08-PF-002-metadata-pool-growth-without-queue-sync — ROCm build and smoke, conformance, and coexistence tests passed
- remote-sync/remote-exec sycl 08-PF-002-metadata-pool-growth-without-queue-sync with setvars override and sycl-ls — Level Zero devices enumerated; SYCL build and smoke, conformance, and coexistence tests passed
- combined final-train local CPU and TTNN/CUDA/ROCm/SYCL gates — every configured build succeeded and all selected smoke, conformance, coexistence, and rank-boundary coverage passed

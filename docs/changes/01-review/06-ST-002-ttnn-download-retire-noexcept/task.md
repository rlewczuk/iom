**Status:** done

## Summary

TTNN download retirement is an in-place non-allocating state transition, rejects reuse until covering completion, and preserves failed-transfer ownership.

## Verification

- local CPU cmake/ctest gate on download leaf worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- remote-sync/remote-exec ttnn 06-ST-002-ttnn-download-retire-noexcept — build and TTNN smoke, conformance, and coexistence tests passed
- combined final-train local CPU and TTNN/CUDA/ROCm/SYCL remote gates — every configured build succeeded and all selected smoke, conformance, and coexistence tests passed

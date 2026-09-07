**Status:** done

## Summary

Identical-window TTNN copies complete without mesh finish while mixed batches retain one native finish and ordered registry release.

## Verification

- local CPU cmake/ctest gate on no-op leaf worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- remote-sync/remote-exec ttnn 05-PF-001-ttnn-noop-completion-without-mesh-finish — build and TTNN smoke, conformance, and coexistence tests passed
- combined final-train local CPU and TTNN/CUDA/ROCm/SYCL remote backend gates — every configured build succeeded and all selected smoke, conformance, and coexistence tests passed

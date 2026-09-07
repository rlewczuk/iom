**Status:** done

## Summary

TTNN native planes remain retained through failed finishes, quarantine drains once per action without allocation, and terminal teardown avoids unsafe destruction.

## Verification

- local CPU cmake/ctest gate on quarantine worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- remote-sync/remote-exec ttnn 02-ST-003-ttnn-quarantine-native-plane-retention with TTNN smoke, conformance, and coexistence targets — build succeeded and all 3 tests passed
- combined final-train TTNN/CUDA/ROCm/SYCL remote backend gates — each configured backend built and its smoke, conformance, and coexistence tests passed

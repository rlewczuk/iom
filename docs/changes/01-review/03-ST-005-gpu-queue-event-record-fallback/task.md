**Status:** done

## Summary

CUDA and ROCm event completion now requires a recorded event or successful stream drain, retaining work when neither proves completion.

## Verification

- local CPU cmake/ctest gate on GPU leaf worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- remote-sync/remote-exec cuda 03-ST-005-gpu-queue-event-record-fallback — build and CUDA smoke, conformance, and coexistence tests passed after event-record test repair
- remote-sync/remote-exec rocm 03-ST-005-gpu-queue-event-record-fallback — build and ROCm smoke, conformance, and coexistence tests passed
- combined final-train CUDA/ROCm remote backend gates — both configured backends built and all smoke, conformance, and coexistence tests passed

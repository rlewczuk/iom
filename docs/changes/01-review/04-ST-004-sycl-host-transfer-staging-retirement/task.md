**Status:** done

## Summary

SYCL transfer staging is released only after transfer-queue terminality, preserves original enqueue failures, and waits before teardown destruction.

## Verification

- local CPU cmake/ctest gate on SYCL leaf worktree — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- remote-sync/remote-exec sycl 04-ST-004-sycl-host-transfer-staging-retirement with setvars override and sycl-ls — Level Zero devices enumerated; build and SYCL smoke, conformance, and coexistence tests passed
- combined final-train SYCL remote backend gate — Level Zero devices enumerated; build and all SYCL smoke, conformance, and coexistence tests passed

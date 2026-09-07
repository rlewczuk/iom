**Status:** done

## Summary

CPU no longer uses the outstanding-work registry: registry wiring, quarantine/entry bookkeeping, and dead CpuDevice/CpuTensor members are deleted from the CPU backend; immediate-free/address-recycling and queue-destruction-with-unwaited-token semantics preserved; shared accelerator registry/fence/quarantine code untouched.

## Verification

- ctest -R "^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$" local CPU — 3/3 passed (34.62s)
- combined wave-3 train: CPU triptych green; CUDA/ROCm/SYCL/TTNN conformance+coexistence 3/3 each (registry paths untouched)

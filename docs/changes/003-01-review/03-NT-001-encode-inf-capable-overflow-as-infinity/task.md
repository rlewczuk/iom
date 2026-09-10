**Status:** done

## Summary

Inf-capable formats encode finite overflow as same-sign infinity across host/device scalar paths and the independent oracle, while finite-only formats retain saturation.

## Verification

- Local CPU review-cpu build and CTest for iom_scalar_add_tests and iom_backend_conformance_cpu_tests — both targets passed.
- Combined final train CPU, CUDA, ROCm, and TTNN backend conformance/coexistence gates — all executed targets passed.

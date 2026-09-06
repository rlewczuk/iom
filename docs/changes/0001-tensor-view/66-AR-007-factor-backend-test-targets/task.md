**Status:** done

## Summary

Extracted the ROCm, CUDA, SYCL, and TTNN smoke/conformance CMake wiring into one parameterized `add_iom_backend_tests()` function while preserving backend gates, source properties, target names, test registrations, and link ordering. Documented the SYCL remote setup workaround in the top-level `AGENTS.md`.

## Verification

- CPU-only configure/build succeeded after rebasing; CTest order and set diffs remained empty, CPU `link.txt` diffs remained empty, and `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` passed 3/3.
- ROCm remote profile — configure/build succeeded, `ctest -N` enumerated seven tests, and `iom_rocm_smoke_tests` passed.
- CUDA remote profile — configure/build succeeded, `ctest -N` enumerated seven tests, and `iom_cuda_smoke_tests` passed.
- TTNN remote profile — configure/build succeeded, `ctest -N` enumerated seven tests, and `iom_ttnn_smoke_tests` passed.
- SYCL remote profile — `set +u; source /opt/intel/oneapi/setvars.sh; set -u` initialized the oneAPI environment, `sycl-ls` enumerated two Level Zero GPU devices, configure/build succeeded, `ctest -N` enumerated seven tests, and `iom_sycl_smoke_tests` passed.

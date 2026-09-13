**Status:** done

## Summary

Fixed-resource exhaustion now parks accepted FIFO work without recursive or blocking retry; SYCL acquisition is nonblocking with deterministic close handling and preserved resource ownership.

## Verification

- cmake --build /tmp/iom-01-review-02-cpu -j2 --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir /tmp/iom-01-review-02-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$' — all 3 local CPU tests passed
- remote CUDA and ROCm conformance/smoke gates after rebase — each built its conformance, smoke, and CPU control targets; iom_cuda_smoke_tests, iom_cuda_conformance_tests, iom_rocm_smoke_tests, iom_rocm_conformance_tests, and both CPU controls passed
- remote SYCL gate after setvars initialization and sycl-ls device enumeration — built SYCL conformance, smoke, and CPU control targets; all 3 tests passed

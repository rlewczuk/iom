**Status:** done

## Summary

Standard GPU binary operations now handle exact operand aliases safely, with corrected F64 and F6 exact-raw alias witnesses.

## Verification

- Local CPU build and iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests passed.
- Remote CUDA build passed iom_cuda_smoke_tests, iom_cuda_conformance_tests, and iom_backend_conformance_cpu_tests.
- Remote ROCm build passed iom_rocm_smoke_tests, iom_rocm_conformance_tests, and iom_backend_conformance_cpu_tests.
- Remote SYCL build passed iom_sycl_smoke_tests, iom_sycl_conformance_tests, and iom_backend_conformance_cpu_tests; sycl-ls enumerated two Level Zero GPUs.
- Remote TTNN build passed iom_ttnn_smoke_tests, iom_ttnn_conformance_tests, and iom_backend_conformance_cpu_tests.

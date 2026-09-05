**Status:** done

## Summary

Implemented device-owned outstanding-work registries and quarantines for CPU, CUDA, ROCm, and TTNN tensor destruction. Queued copy tasks register exact source/destination entry IDs before publication; tensor destruction fences live work, preserves failed or invalidated storage in device quarantine, and device teardown drains quarantine after queue and runtime cleanup. Queue teardown invalidates retained entries. Added deterministic registry/quarantine coverage and CUDA/ROCm allocator-recycling regressions.

## Verification

- CPU configure/build and CTest: `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` passed 2/2.
- CPU AddressSanitizer configure/build and CTest: both CPU suites passed 2/2 with no ASan diagnostics.
- CUDA remote configure/build and CTest: `iom_cuda_smoke_tests` and `iom_cuda_conformance_tests` passed 2/2.
- CUDA Compute Sanitizer memcheck: all 12 test cases and 39,883 assertions passed; error summary 0.
- CUDA Compute Sanitizer racecheck: completed successfully on `iom_cuda_conformance_tests`.
- ROCm remote configure/build and CTest: `iom_rocm_smoke_tests` and `iom_rocm_conformance_tests` passed 2/2.
- ROCm `rocprofv3 --runtime-trace --stats`: 13/13 test cases and 37,982 assertions passed; profiler completed successfully. The profiler reported non-fatal timestamp corrections from the host runtime.
- TTNN remote configure/build and CTest: `iom_ttnn_smoke_tests` and `iom_ttnn_conformance_tests` passed 2/2.
- Source audits found no backend SDK terms in the common registry header, retained address indexes through invalidation, and exact entry-ID registration/release/invalidation across backends.
- Remote accelerator mirrors were cleaned after verification.

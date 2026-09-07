**Status:** done

## Summary

TTNN upload staging remains retained or retired until completion is proven, preserves original failures, and skips unsafe reuse.

## Verification

- remote-sync/remote-exec ttnn 01-ST-001-ttnn-upload-staging-retirement with TTNN smoke, conformance, and coexistence targets — build succeeded and all 3 tests passed
- combined final-train local CPU cmake/ctest gate — iom_tests, iom_cpu_tests, and iom_backend_conformance_cpu_tests all passed
- combined final-train TTNN/CUDA/ROCm/SYCL remote backend gates — each configured backend built and its smoke, conformance, and coexistence tests passed

**Status:** done

## Summary

Consolidated CUDA and ROCm copy metadata, grid-stride launch, staging-slot, transfer-stream, and metadata-slot infrastructure under `src/shared/` policy templates. Promoted backend policies into `copy.hpp`, retained backend-local queue and fence code, removed the six duplicated pool files, and updated target source lists and smoke-test includes.

## Verification

- `cmake -S . -B build-ar004-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-ar004-cpu -j2 --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests` — all three CPU targets built successfully.
- `ctest --test-dir build-ar004-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — 3/3 tests passed.
- CUDA remote workflow on `bv1` using task directory `ar004-20260906` — configured with `CUDA_ENABLED=ON`, built `iom_cuda_smoke_tests` and `iom_cuda_conformance_tests`, and passed both suites 2/2.
- CUDA focused runs for staging accounting, poisoned/error stream drops, and queue-destruction fencing — 4/4 selected tests passed with 116/116 assertions.
- ROCm remote workflow on `bv2` using task directory `ar004-20260906-rocm` — configured with `ROCM_ENABLED=ON`, built `iom_rocm_smoke_tests` and `iom_rocm_conformance_tests`, and passed both suites 2/2.
- ROCm focused runs for staging accounting, odd-tail transfer reuse, and queue-destruction fencing — 3/3 selected tests passed with 109/109 assertions.
- Source audits — legacy CUDA/ROCm metadata names and shared-runtime names returned no matches; each shared definition and each backend `.inl` include occurred once; deleted pool files were absent; copy translation units shrank from 938/937 to 431/420 lines. Both remote mirrors were cleaned.

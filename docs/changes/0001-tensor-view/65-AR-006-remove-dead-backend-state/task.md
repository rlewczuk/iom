**Status:** done

## Summary

Removed the unused shared staging-pool identity and shared outcome success field, default-constructed CUDA/ROCm pools, nulled CUDA/ROCm task view pointers after binding synchronous locals, and passed backend-derived fence success directly to the shared resolver. Updated CUDA/ROCm smoke construction sites while preserving the SYCL and TTNN completion protocols.

## Verification

- `cmake -S . -B build-ar006-cpu -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build-ar006-cpu -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests` — CPU targets built successfully.
- `ctest --test-dir build-ar006-cpu --output-on-failure -R 'iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests'` — 3/3 CPU/common suites passed.
- CUDA remote workflow `ar006-20260906` on `bv1` — configured and built `iom_cuda_smoke_tests`, `iom_cuda_conformance_tests`, and `iom_tests`; backend 2/2 and common 1/1 passed.
- ROCm remote workflow `ar006-20260906-rocm` on `bv2` — configured and built `iom_rocm_smoke_tests`, `iom_rocm_conformance_tests`, and `iom_tests`; backend 2/2 and common 1/1 passed.
- SYCL remote workflow `ar006-20260906-sycl` on `bv2` with the required oneAPI environment override — built `iom_sycl_smoke_tests`, `iom_sycl_conformance_tests`, and `iom_tests`; backend 2/2 and common 1/1 passed.
- TTNN remote workflow `ar006-20260906-ttnn` on `bv1` — built `iom_ttnn_smoke_tests`, `iom_ttnn_conformance_tests`, and `iom_tests`; backend 2/2 and common 1/1 passed.
- Final source audits and `git diff --check` — no deleted `SequenceOutcome::fence_succeeded` member access, no staging-pool identity fields, exactly four CUDA/ROCm task-pointer entry references, and one shared resolver definition with four backend callsites.

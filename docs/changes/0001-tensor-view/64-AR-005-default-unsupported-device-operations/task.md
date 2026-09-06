**Status:** done

## Summary

`DeviceOps` now supplies default unsupported-operation bodies for `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`, using one protected `backend_label()` hook per queue. CPU, CUDA, ROCm, SYCL, and TTNN queues no longer carry duplicate rejection stubs; test fakes inherit the defaults while `FakeQueue` retains its recording overrides. Capability conformance supports exact backend-message checks for direct backend drivers while preserving the suite-level call shape.

## Verification

- `cmake -S . -B build -DCPU_ENABLED=ON && cmake --build build -j && ctest --test-dir build --output-on-failure -R "iom_tests|iom_cpu_tests"` — build succeeded and both tests passed.
- `ctest --test-dir build --output-on-failure -R iom_backend_conformance_cpu_tests` — passed.
- CUDA remote build of `iom_cuda_smoke_tests` and `iom_cuda_conformance_tests`, followed by their individual CTest runs under `/tmp/iom-bv1.lock` — both targets built and passed.
- ROCm remote build of `iom_rocm_smoke_tests` and `iom_rocm_conformance_tests`, followed by their individual CTest runs under `/tmp/iom-bv2.lock` — both targets built and passed.
- SYCL remote build of `iom_sycl_smoke_tests` and `iom_sycl_conformance_tests`, followed by their individual CTest runs with the required oneAPI/UMF runtime library paths under `/tmp/iom-bv2.lock` — both targets built and passed.
- TTNN remote build of `iom_ttnn_smoke_tests` and `iom_ttnn_conformance_tests`, followed by their individual CTest runs under `/tmp/iom-bv1.lock` — both targets built and passed.
- Static searches for `throw unsupported` under `src/` and `test/`, and for compute-operation declarations under `src/` — no matches.
- `git diff --check` — passed.

**Status:** done

## Summary

Restored no-throw CUDA and HIP event synchronization in `cuda_fence_destroy` and `hip_fence_destroy` before their unchanged resource-destruction paths. Strengthened both queue-destruction regressions to submit 64 successful copies around the retained-failure submission, ensuring published fenced work remains for shutdown draining.

## Verification

- `cmake -S . -B build -DCUDA_ENABLED=ON && cmake --build build -j` via CUDA `remote-exec` — configured and built all CUDA, CPU, and test targets successfully.
- `ctest --test-dir build -R iom_cuda_conformance_tests --output-on-failure`, focused `CUDA queue destruction fences pending copies`, and `build/test/iom_cuda_smoke_tests` via CUDA `remote-exec` — conformance 1/1 passed; focused case 1/1 with 78/78 assertions passed; smoke suite 12/12 with 112/112 assertions passed.
- `compute-sanitizer --tool memcheck build/test/iom_cuda_conformance_tests --test-case="CUDA queue destruction fences pending copies"` via CUDA `remote-exec` — focused case passed and `ERROR SUMMARY: 0 errors`.
- `cmake -S . -B build -DROCM_ENABLED=ON && cmake --build build -j` via ROCm `remote-exec` — configured and built all ROCm, CPU, and test targets successfully.
- `ctest --test-dir build -R iom_rocm_conformance_tests --output-on-failure`, focused `ROCm queue destruction fences pending copies`, and `build/test/iom_rocm_smoke_tests` via ROCm `remote-exec` — conformance 1/1 passed; focused case 1/1 with 75/75 assertions passed; smoke suite 5/5 with 62/62 assertions passed.
- `rocprofv3 --runtime-trace --stats -- build/test/iom_rocm_conformance_tests --test-case="ROCm queue destruction fences pending copies"` via ROCm `remote-exec` — focused case passed with 75/75 assertions and profiler exit 0; no busy-event or invalid-handle diagnostics appeared.
- `cmake -S . -B build-st002-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-st002-cpu -j && ctest --test-dir build-st002-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — 3/3 CPU/common tests passed.
- Source audit and `git diff --check` — only `src/cuda/copy.cu`, `src/rocm/copy.hip`, `test/cuda/test_cuda_conformance.cpp`, and `test/rocm/test_rocm_conformance.cpp` changed before this annotation; each backend has one new no-throw synchronize helper call before its unchanged resource cleanup.
- CUDA and ROCm remote mirrors were removed with `remote-clean` after verification.

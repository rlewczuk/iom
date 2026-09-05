**Status:** done

## Summary

Added CUDA and ROCm no-throw event fence-and-destroy callbacks and stream synchronize-and-destroy finalizers after `StagedWorker` shutdown. Added multi-entry queue-destruction regressions with a retained null-fence failure between successful copies.

## Verification

- `cmake -S . -B build-st001-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-st001-cpu -j && ctest --test-dir build-st001-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — 3/3 CPU/common tests passed.
- CUDA remote build and selected CTest suite — 6/6 tests passed.
- ROCm remote build and selected CTest suite — 6/6 tests passed.
- CUDA focused queue-destruction regression — 22 assertions passed; Compute Sanitizer memcheck reported 0 errors and racecheck reported 0 hazards.
- ROCm focused queue-destruction regression — 19 assertions passed; `rocprofv3 --runtime-trace --stats` completed successfully and generated trace output.
- ROCm AddressSanitizer — intentionally skipped per user instruction because the RDNA4 target does not provide working ASan support.
- Source audit — each backend has one `fence_and_destroy` and one `synchronize_and_destroy_stream`; queue destructors drain the worker before guarded stream finalization.

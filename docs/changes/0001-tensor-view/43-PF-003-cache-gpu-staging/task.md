**Status:** done

## Summary

Added device-owned CUDA and ROCm transfer-stream and staging-slot pools. Synchronous host transfers now reuse bounded, geometrically growing staging allocations and backend transfer streams, poison failed resources, zero only the rounded tail word, and preserve logical host-transfer behavior. Added private accounting/failure seams, pool lifecycle tests, concurrent-transfer coverage, and backend build wiring.

## Verification

- `cmake -S . -B build-cpu -DBUILD_TESTING=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-cpu -j` — core production configuration built successfully.
- `ctest --test-dir build-cpu -E iom_cpu_bench --output-on-failure` — 3/3 CPU/core tests passed.
- Remote CUDA `cmake -S . -B build-cuda -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-cuda -j && flock /tmp/iom-cuda0.lock ctest --test-dir build-cuda -E iom_cpu_bench --output-on-failure` — build succeeded; 6/6 tests passed, including CUDA smoke and conformance.
- Remote ROCm `cmake -S . -B build-rocm -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-rocm -j && flock /tmp/iom-rocm0.lock ctest --test-dir build-rocm -E iom_cpu_bench --output-on-failure` — build succeeded; 6/6 tests passed, including ROCm smoke and conformance.
- Source audit — one `compute_staging_size` call in the shared synchronous-transfer path and zero backend allocation/stream lifecycle calls in that shared transfer body.

**Status:** done

## Summary

Implemented SYCL standard-layout USM tensor storage, context-compatible allocator validation, synchronous logical host transfers, in-order asynchronous view copies, and unsupported-operation capability failures. Added SYCL storage/copy conformance coverage, updated smoke expectations, and wired the implementation and tests into the optional SYCL CMake targets.

## Verification

- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DSYCL_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/bin/icpx -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF` on `bv2` — configuration succeeded.
- `cmake --build build/sycl --target iom_sycl_conformance_tests iom_sycl_smoke_tests -j` on `bv2` — both SYCL targets built successfully.
- `LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/umf/latest/lib:/opt/intel/oneapi/tbb/latest/lib ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build/sycl --output-on-failure -R ^iom_sycl_conformance_tests$` on `bv2` — 1/1 test passed on Intel Arc Pro B60 accelerator hardware.
- `LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/umf/latest/lib:/opt/intel/oneapi/tbb/latest/lib ONEAPI_DEVICE_SELECTOR=level_zero:gpu ctest --test-dir build/sycl --output-on-failure -R ^iom_sycl_smoke_tests$` on `bv2` — 1/1 test passed on Intel Arc Pro B60 accelerator hardware.
- `cmake -S . -B build/sycl-disabled -DBUILD_TESTING=ON -DSYCL_ENABLED=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/sycl-disabled --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j && ctest --test-dir build/sycl-disabled --output-on-failure -R '^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$'` — all 3 core/CPU tests passed with SYCL disabled.

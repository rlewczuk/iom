**Status:** done

## Summary

Completed the core DeviceOps cutover into seven responsibility-complete sources, removed src/iom.cpp, and kept shared queue/validation/failure/completion machinery in device_ops.cpp.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests — all targets built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$ — 4/4 tests passed.
- wc -l src/tensor.cpp src/tensor_view.cpp src/workspace.cpp src/device_ops.cpp src/device_ops_copy.cpp src/device_ops_binary.cpp src/device_ops_neural.cpp src/iom_internal.hpp — 197, 337, 111, 488, 54, 463, 81, and 39 lines; all <=499; src/iom.cpp absent.
- temporary umbrella consumer including iom/iom.hpp and iom/detail/outstanding_work_registry.hpp compiled, linked against build/split-cpu/libiom.a, and ran successfully.

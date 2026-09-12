**Status:** done

## Summary

Split CPU device, tensor, transfer-helper, and queue responsibilities into private components while preserving the public factory, allocation, tiled transfers, queue ordering, and numerics.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF — configure completed successfully.
- cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests — all CPU targets built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$ — 4/4 tests passed.
- wc -l src/cpu/device.cpp src/cpu/tensor.cpp src/cpu/queue.cpp src/cpu/device_internal.hpp src/cpu/transfer_helpers.hpp — counts 57, 176, 320, 32, and 319; all assignment and universal caps passed.

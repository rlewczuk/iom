**Status:** done

## Summary

Extracted core binary validation, snapshot, facade, and workspace-requirement definitions into src/device_ops_binary.cpp with one compiled owner while retaining iom.cpp for remaining responsibilities.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF — configure completed successfully.
- cmake --build build/split-cpu --target libiom iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests — all requested targets built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$ — 3/3 tests passed.
- wc -l src/device_ops_binary.cpp src/iom.cpp src/tensor.cpp src/tensor_view.cpp src/workspace.cpp src/iom_internal.hpp — device_ops_binary.cpp is 463 lines; iom.cpp remains an intermediate 773-line carrier for later cutover; other extracted files are 197, 337, 111, and 39 lines.

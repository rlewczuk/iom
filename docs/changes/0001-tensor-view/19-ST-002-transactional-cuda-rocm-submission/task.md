**Status:** done

## Summary

CUDA and ROCm copy submission now links an owned event task before any kernel launch, retains post-enqueue failures until the worker fence completes, and publishes staged tasks with non-allocating list splices. Added the common deferred retained-failure scenario, backend fault seams, hardware tests for launch/event/pre-enqueue failures, repeated waits, sequence preservation, and allocator-slot canaries.

## Verification

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cpu --target iom_tests iom_backend_conformance_cpu_tests -j` — CPU contract targets built successfully.
- `cmake --build build/cpu --target iom_cpu_tests -j && ctest --test-dir build/cpu --output-on-failure -R '^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$|^iom_tests$'` — all 3 CPU tests passed.
- `./build/cpu/test/iom_tests --test-case='DeviceOps queue*,DeviceOps view signatures*'` — 10 test cases and 356 assertions passed.
- CUDA remote `cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build -j --target iom_cuda_conformance_tests` followed by `flock /tmp/agent-gpu0.lock ./build/test/iom_cuda_conformance_tests -tc="*remains transactional across post-enqueue failures*"` — 1 test case and 46 assertions passed on CUDA hardware.
- CUDA remote CTest runs for `iom_cuda_conformance_tests`, `iom_backend_conformance_cpu_tests`, and `iom_tests` — all passed.
- ROCm remote `cmake -S . -B build -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm && cmake --build build -j --target iom_rocm_conformance_tests` followed by `flock /tmp/agent-gpu0.lock ./build/test/iom_rocm_conformance_tests -tc="*remains transactional across post-enqueue failures*"` — 1 test case and 42 assertions passed on ROCm hardware.
- ROCm remote CTest runs for `iom_rocm_conformance_tests`, `iom_backend_conformance_cpu_tests`, and `iom_tests` — all passed.

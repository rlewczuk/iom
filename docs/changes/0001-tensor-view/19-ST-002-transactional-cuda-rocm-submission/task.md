**Status:** done

## Summary

CUDA and ROCm copy submission now links an owned event task before any kernel launch, retains post-enqueue failures until the worker fence completes, and publishes staged tasks with non-allocating list splices. Added the common deferred retained-failure scenario, backend fault seams, hardware tests for launch/event/pre-enqueue failures, repeated waits, sequence preservation, and allocator-slot canaries.

## Verification

- `cmake --build build/cpu --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j && ctest --test-dir build/cpu --output-on-failure -R '^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$|^iom_tests$'` — post-merge build succeeded and all 3 CPU tests passed.
- `./build/cpu/test/iom_tests --test-case='DeviceOps queue*,DeviceOps view signatures*'` — 10 test cases and 356 assertions passed.
- `.agents/skills/remote-development/scripts/remote-exec cuda st002-cuda 'cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build -j --target iom_cuda_conformance_tests iom_backend_conformance_cpu_tests iom_tests && flock /tmp/agent-gpu0.lock ./build/test/iom_cuda_conformance_tests -tc="*remains transactional across post-enqueue failures*" && ctest --test-dir build --output-on-failure -R iom_cuda_conformance_tests && ctest --test-dir build --output-on-failure -R iom_backend_conformance_cpu_tests && ctest --test-dir build --output-on-failure -R iom_tests'` — post-merge CUDA build passed; focused test passed with 46 assertions; all 3 CTest runs passed on CUDA hardware.
- `.agents/skills/remote-development/scripts/remote-exec rocm st002-rocm 'cmake -S . -B build -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm && cmake --build build -j --target iom_rocm_conformance_tests iom_backend_conformance_cpu_tests iom_tests && flock /tmp/agent-gpu0.lock ./build/test/iom_rocm_conformance_tests -tc="*remains transactional across post-enqueue failures*" && ctest --test-dir build --output-on-failure -R iom_rocm_conformance_tests && ctest --test-dir build --output-on-failure -R iom_backend_conformance_cpu_tests && ctest --test-dir build --output-on-failure -R iom_tests'` — post-merge ROCm build passed; focused test passed with 42 assertions; all 3 CTest runs passed on ROCm hardware.

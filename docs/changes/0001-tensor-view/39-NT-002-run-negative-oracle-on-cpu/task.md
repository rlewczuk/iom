**Status:** done

## Summary

Added the CPU conformance negative storage-oracle fixture. The test derives a one-leaf span from `Device::supported_data_types()`, verifies the perturbed transfer map is rejected, and verifies an identity map passes.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build --target iom_backend_conformance_cpu_tests && ctest --test-dir build -R iom_backend_conformance_cpu_tests --output-on-failure` — configured and built successfully; 1/1 test passed.
- `./build/test/iom_backend_conformance_cpu_tests '-tc=CPU conformance: storage oracle identifies perturbed transfer map'` — 1 test passed with 341 assertions.

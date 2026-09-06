**Status:** done

## Summary

Implemented SYCL queue lifetime hardening across `src/sycl/registry_state.hpp`, `src/sycl/device.cpp`, `src/sycl/copy.hpp`, `src/sycl/copy.cpp`, `CMakeLists.txt`, and `test/sycl/test_sycl_conformance.cpp`. The backend now uses device-owned registry/quarantine state, fences tensor destruction, drains `StagedWorker` tasks during queue teardown, and rolls back pre-enqueue failures while retaining post-enqueue failures.

## Verification

- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF` via remote-development `sycl` profile — configured successfully with IntelLLVM 2026.1.
- `cmake --build build/sycl --target iom_sycl_conformance_tests iom_sycl_smoke_tests iom_tests -j` via remote-development — all targets built successfully.
- Dedicated pre-enqueue fault run under exclusive device access — 1 test passed, 26 assertions passed.
- Valgrind Memcheck runs for destruction-before-wait, post-enqueue transactional failure, queue teardown, and `SyclFenceState::result()` — each test passed and reported `ERROR SUMMARY: 0 errors from 0 contexts`.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"` via remote-development — 2/2 tests passed.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"` via remote-development — 1/1 test passed.
- SYCL source audit and `git diff --check` — passed.
- Remote SYCL mirror cleaned after verification.

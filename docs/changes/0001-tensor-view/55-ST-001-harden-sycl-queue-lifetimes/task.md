**Status:** blocked

## Summary

Implemented SYCL queue lifetime hardening across `src/sycl/registry_state.hpp`, `src/sycl/device.cpp`, `src/sycl/copy.hpp`, `src/sycl/copy.cpp`, `CMakeLists.txt`, and `test/sycl/test_sycl_conformance.cpp`. The backend now uses device-owned registry/quarantine state, fences tensor destruction, drains `StagedWorker` tasks during queue teardown, and rolls back pre-enqueue failures while retaining post-enqueue failures.

## Verification

- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF` via remote-development `sycl` profile — configured successfully with IntelLLVM 2026.1.
- `cmake --build build/sycl --target iom_sycl_conformance_tests iom_sycl_smoke_tests -j` via remote-development — both targets built successfully.
- Focused SYCL destruction, pre-enqueue, post-enqueue, queue-teardown, no-op, and `SyclFenceState::result()` runs under exclusive device access — all passed.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"` via remote-development — 2/2 tests passed.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"` via remote-development — 1/1 test passed.
- SYCL source audit and `git diff --check` — passed.

## Errors

- Integration checkout is currently dirty (`test/CMakeLists.txt`), so `spec-run-task` cannot merge the latest integration branch or rerun the required diagnostics yet; the implementation remains committed on the retained feature branch.

**Status:** done

## Summary

Bound ROCm post-enqueue failure completion without recording or synchronizing a fence on the errored stream. ROCm marks the staged event nullable, the shared GPU worker and shutdown drains skip null-event destruction/synchronization while completing the retained failure, and the ROCm conformance test re-execs itself under a five-second `posix_spawn` watchdog for both injected post-enqueue faults.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DROCM_PATH=/opt/rocm` via remote-development on `bv2` — configured successfully.
- `cmake --build build -j --target iom_rocm_conformance_tests` via remote-development on `bv2` — built successfully.
- `flock /tmp/agent-gpu0.lock ./build/test/iom_rocm_conformance_tests -tc="*remains transactional across post-enqueue failures"` via remote-development — 1 test passed, 77 assertions passed.
- `ctest --test-dir build --output-on-failure -R iom_rocm_conformance_tests` via remote-development — 1/1 tests passed.
- `cmake --build build -j --target iom_tests` and `ctest --test-dir build --output-on-failure -R iom_tests` via remote-development — 1/1 tests passed.
- ROCm source audit — `src/rocm/copy.hip` contains one `hipEventRecord` and one `hipEventSynchronize` call site; the failure path uses the nullable staged event branch.

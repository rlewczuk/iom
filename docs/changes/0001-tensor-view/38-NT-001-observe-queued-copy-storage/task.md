**Status:** done

## Summary

Extended shared asynchronous copy conformance with an optional `AcceleratorStorageOracle` path. Each accelerator copy case now seeds and verifies independent full source and destination owner allocations, waits its single queued candidate copy, preserves the existing CPU reference and logical assertions, and compares complete candidate source and destination storage against the modeled expectations. CUDA, ROCm, and TTNN direct and full-suite registrations pass their native storage oracles through the shared harness.

## Verification

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON && cmake --build build/cpu --target iom_backend_conformance_cpu_tests -j2` — completed successfully.
- `ctest --test-dir build/cpu -R iom_backend_conformance_cpu_tests --output-on-failure` — 1/1 test passed.
- CUDA remote sync/build/test using `nt-001-queued-copy-oracle` — target built successfully and `iom_cuda_conformance_tests` passed 1/1.
- ROCm remote sync/build/test using `nt-001-queued-copy-oracle` — target built successfully and `iom_rocm_conformance_tests` passed 1/1.
- TTNN remote sync/build/test using `nt-001-queued-copy-oracle` — target built successfully and `iom_ttnn_conformance_tests` passed 1/1.
- Temporary widened CUDA and ROCm kernel discriminators wrote a destination padding byte without changing the logical window; the corresponding conformance targets failed at byte 1297 with `candidate oracle destination full to full: storage diverges`, then the temporary source change was removed.
- `remote-clean cuda nt-001-queued-copy-oracle`, `remote-clean rocm nt-001-queued-copy-oracle`, and `remote-clean ttnn nt-001-queued-copy-oracle` — all mirrors removed.

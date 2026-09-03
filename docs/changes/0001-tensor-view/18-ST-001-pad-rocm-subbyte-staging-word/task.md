**Status:** blocked

## Summary

Implemented checked 32-bit-word round-up for ROCm transfer staging, zeroed the complete rounded staging allocation for device-to-host gathers, and added the focused `{1,17}` I2/F6 host-read regression in `src/rocm/copy.hip` and `test/rocm/test_rocm_conformance.cpp`.

## Verification

- `cmake -S . -B build/rocm -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm && cmake --build build/rocm --target iom_rocm_conformance_tests -j` through `remote-exec rocm 18-st001-subbyte-staging` — build succeeded on the configured ROCm host.
- `flock /tmp/agent-gpu0.lock ./build/rocm/test/iom_rocm_conformance_tests -tc="*sub-byte odd-length host reads*"` through `remote-exec rocm 18-st001-subbyte-staging` — 1 test case and 22 assertions passed without sanitizer instrumentation.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/rocm --output-on-failure -R "^iom_rocm_conformance_tests$"` through `remote-exec rocm 18-st001-subbyte-staging` — 1/1 ROCm conformance test passed without sanitizer instrumentation.
- `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cpu --target iom_backend_conformance_cpu_tests -j && ctest --test-dir build/cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — CPU backend conformance passed locally.

## Errors

- `remote-exec rocm 18-st001-subbyte-staging-unfixed` running the uncorrected focused binary at `./build/rocm/test/iom_rocm_conformance_tests` — exited 1 with `Your application is linked against incompatible ASan runtimes`; `ldd` showed system `libasan.so.8` and uninstrumented `/usr/lib/x86_64-linux-gnu/libamdhip64.so.7`, while `rocminfo` reported `XNACK enabled: NO`. The configured ROCm host lacks the instrumented ROCm runtime/XNACK prerequisite required to produce the mandated device-ASan reproduction and fixed ASan proof. The task worktree and feature branch are retained and unmerged.

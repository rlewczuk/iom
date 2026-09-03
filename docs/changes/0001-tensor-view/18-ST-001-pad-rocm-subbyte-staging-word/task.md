**Status:** blocked

## Summary

Implemented checked 32-bit-word round-up for ROCm transfer staging, zeroed the complete rounded staging allocation for device-to-host gathers, and added the focused `{1,17}` I2/F6 host-read regression in `src/rocm/copy.hip` and `test/rocm/test_rocm_conformance.cpp`.

## Verification

- `cmake -S . -B build/rocm -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" && cmake --build build/rocm --target iom_rocm_conformance_tests -j` through `remote-exec rocm 18-st001-subbyte-staging` on `bh2` — build succeeded.
- `ASAN_OPTIONS=detect_leaks=0 flock /tmp/agent-gpu0.lock ./build/rocm/test/iom_rocm_conformance_tests -tc="*sub-byte odd-length host reads*"` through `remote-exec rocm 18-st001-subbyte-staging` — 1 test case and 22 assertions passed; this build did not instrument the HIP translation unit.
- `flock /tmp/agent-gpu0.lock ./build/rocm-normal/test/iom_rocm_conformance_tests -tc="*sub-byte odd-length host reads*"` through `remote-exec rocm 18-st001-subbyte-staging` — 1 test case and 22 assertions passed on `bh2`.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/rocm-normal --output-on-failure -R "^iom_rocm_conformance_tests$"` through `remote-exec rocm 18-st001-subbyte-staging` — 1/1 ROCm conformance test passed on `bh2`.
- `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cpu --target iom_backend_conformance_cpu_tests -j && ctest --test-dir build/cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — CPU backend conformance passed locally.

## Errors

- `remote-exec rocm 18-st001-subbyte-staging-unfixed` running the uncorrected ASan-configured binary at `./build/rocm/test/iom_rocm_conformance_tests` — exited 1 with `Your application is linked against incompatible ASan runtimes`, not the required device-ASan out-of-bounds report.
- The remote `compile_commands.json` showed `-fsanitize=address` on host C++ files but not `src/rocm/copy.hip`; an attempted HIP device-ASan configure with `-DCMAKE_HIP_ARCHITECTURES=gfx1201:xnack+` failed with `invalid target ID`. `rocminfo` reported `XNACK enabled: NO`, and `/opt/rocm/lib/asan` was absent. The required device-ASan reproduction and fixed device-ASan proof remain unavailable; the task worktree and feature branch are retained and unmerged.

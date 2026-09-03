**Status:** done

## Summary

Implemented checked 32-bit-word round-up for ROCm transfer staging, zeroed the complete rounded staging allocation for device-to-host gathers, and added the focused `{1,17}` I2/F6 host-read regression in `src/rocm/copy.hip` and `test/rocm/test_rocm_conformance.cpp`.

## Verification

- `cmake -S . -B build/rocm -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm -DCMAKE_CXX_FLAGS="" -DCMAKE_C_FLAGS="" -DCMAKE_EXE_LINKER_FLAGS="" && cmake --build build/rocm --target iom_rocm_conformance_tests -j` through `remote-exec rocm 18-st001-subbyte-staging` on `bh2` — build succeeded.
- `flock /tmp/agent-gpu0.lock ./build/rocm/test/iom_rocm_conformance_tests -tc="*sub-byte odd-length host reads*"` through `remote-exec rocm 18-st001-subbyte-staging` — 1 test case and 22 assertions passed on ROCm hardware.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/rocm --output-on-failure -R "^iom_rocm_conformance_tests$"` through `remote-exec rocm 18-st001-subbyte-staging` — 1/1 ROCm conformance test passed on ROCm hardware.
- `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cpu --target iom_backend_conformance_cpu_tests -j && ctest --test-dir build/cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — CPU backend conformance passed locally.
- ROCm device-ASan verification — intentionally skipped per the explicit user instruction after the installed gfx1201 compiler reported that `-fsanitize=address` is unsupported for that offload target.

**Status:** done

## Summary

Removed the full destination pre-zero from CPU and TTNN host downloads. CPU now zeroes only the trailing byte when a sub-byte logical element count leaves unused tail bits; byte-aligned CPU downloads and all supported TTNN downloads rely on their complete tile or plane writes.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench` — configured and built all requested CPU targets.
- `ctest --test-dir build --output-on-failure -E iom_cpu_bench` — passed 3/3 tests: `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests`.
- `ctest --test-dir build --output-on-failure -R '^iom_cpu_bench$'` — passed 1/1 benchmark gate.
- CPU and TTNN source audits — exactly one `std::fill_n` remains in `src/cpu/device.cpp`, with no `std::fill(` matches in either rewritten download path; protected paths remained unchanged.
- `.agents/skills/remote-development/scripts/remote-exec ttnn pf002-ttnn-download-prezero 'cmake -S . -B build/ttnn-pf002 -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/ttnn-pf002 --target iom_ttnn_conformance_tests iom_ttnn_smoke_tests'` — remote TTNN build passed.
- `.agents/skills/remote-development/scripts/remote-exec ttnn pf002-ttnn-download-prezero 'ctest --test-dir build/ttnn-pf002 --output-on-failure -R "^iom_ttnn_(conformance|smoke)_tests$"'` — passed 2/2 remote TTNN smoke and conformance tests.
- Remote `grep -nF "std::fill(" src/ttnn/copy.cpp` — returned no matches; remote task mirror was cleaned afterward.

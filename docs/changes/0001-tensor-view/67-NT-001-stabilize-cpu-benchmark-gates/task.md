**Status:** done

## Summary

Replaced the absolute 16x16 queued-copy latency ceiling with an interleaved identical-window no-op baseline and a calibrated self-relative allowance in `test/cpu/test_cpu_bench.cpp`. Recalibrated the four post-CC-001 throughput floors from a 20-run stability dataset; no production or CMake files changed.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench` — configured and built all required CPU targets.
- Transient calibration collection — 20 direct `iom_cpu_bench` runs recorded in `/tmp/nt001/calibration.log`; derived `kAllowanceSeconds = 1.46e-6` from `D = -1.245e-7`, `Q = 2.655e-7`, `diff_max = 3.91e-7`, `R = 1e-9`, `M = 5.31e-7`, and `A_seconds = 1.453e-6`.
- Post-commit stability loop — `iom_cpu_bench: 20/20 stable`.
- Injected copy-only 100 us delay — benchmark failed only at the relative latency gate with copy 161.4 us, no-op 9.0 us, differential 152.4 us, and allowance 1.5 us; after reverting the injection, the benchmark passed and `git diff -- src/` was empty.
- `ctest --test-dir build --output-on-failure` — 4/4 tests passed: `iom_tests`, `iom_cpu_tests`, `iom_cpu_bench`, and `iom_backend_conformance_cpu_tests`.
- Static/scope audits — no `6.0e-6` match; `git diff -- test/CMakeLists.txt src/ include/ CMakeLists.txt` was empty.

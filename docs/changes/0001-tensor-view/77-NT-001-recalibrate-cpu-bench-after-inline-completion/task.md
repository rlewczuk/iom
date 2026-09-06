**Status:** done

## Summary

Re-derived `kAllowanceSeconds` for the post-73 inline CPU queued-copy path from one 20-run ordinary-load dataset, preserving the self-relative gate and interleaved no-op baseline. Updated only `test/cpu/test_cpu_bench.cpp`; no production or CMake changes remain.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench` — configured and built all required CPU targets.
- Twenty direct placeholder calibration runs with exit codes ignored — captured 20 copy/no-op median pairs; computed `D = 6.0e-8`, `Q = 0.0`, `diff_max = 6.0e-8`, `M = 1.0e-8`, `A_seconds = 7.0e-8`, and `kAllowanceSeconds = 7.0e-8`.
- Twenty direct post-commit stability runs — `iom_cpu_bench: 20/20 stable`.
- Post-73 injection of a 100 us copy-only delay — bench exited 1 at the relative gate with a 151.424 us differential and 0.1 us no-op median; after reverting, the bench exited 0 and `git diff --quiet -- src/` passed.
- `ctest --test-dir build --output-on-failure` — 100% of 4 tests passed: `iom_tests`, `iom_cpu_tests`, `iom_cpu_bench`, and `iom_backend_conformance_cpu_tests`.

**Status:** done

## Summary

Removed the CPU wide-row transfer shortcuts so host transfers and queued copies always walk one standard 16x16 tile column at a time. Added production assertions to both row-copy helpers, corrected callback accounting, and added direct-storage regressions for tile-aligned F32, I4, and U8 shapes plus queued F32 copy storage equality.

## Verification

- `cmake -S . -B build -DCPU_ENABLED=ON && cmake --build build -j && ctest --test-dir build --output-on-failure -E iom_cpu_bench` — configured, built all CPU targets, and passed `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` (3/3 tests passed).
- `ctest --test-dir build -R '^iom_cpu_tests$' --output-on-failure` — passed (1/1 test).
- Source audit with no matches for the removed wide-row and wide-element predicates, exactly two `assert(elements <= TensorSpec::TILE)` occurrences, and unchanged `src/iom.cpp` — passed.

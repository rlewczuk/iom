**Status:** done

## Summary

Removed the duplicated `QuantizationFormat` recognizer from `src/iom.cpp`. `TensorSpec::validate()` now checks the contiguous declared range using the authoritative `NONE` through `TT_BFP8A` boundary, then rejects every declared non-`NONE` format as unsupported while preserving invalid-enum and data-type validation behavior.

## Verification

- `cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug` — configured successfully.
- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests` — both targets built successfully.
- `ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests)$'` — 2/2 tests passed.
- `cmake --build cmake-build-debug --target iom_backend_conformance_cpu_tests` — target built successfully.
- `ctest --test-dir cmake-build-debug --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — 1/1 test passed.

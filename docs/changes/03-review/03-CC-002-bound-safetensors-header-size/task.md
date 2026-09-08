**Status:** done

## Summary

Bound untrusted SafeTensors headers at the inclusive 100,000,000-byte format cap before size conversion, file-range arithmetic, or JSON parsing while retaining existing guards and runtime errors.

## Verification

- focused cap fixture: build/test/iom_tests --test-case="SafeTensorsFile rejects a header beyond the format cap before parsing" — 1 test case and 15 assertions passed.
- focused ordinary-header regression: build/test/iom_tests --test-case="SafeTensorsFile still parses an ordinary valid header" — 1 test case and 6 assertions passed.
- SafeTensors suite: cmake --build build --target iom_tests && ctest --test-dir build -R ^iom_tests --output-on-failure — 1/1 test passed.
- final-train CPU combined verification: cmake --build build/cpu --target iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/cpu -R "^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$" --output-on-failure — 2/2 tests passed.

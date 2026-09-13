**Status:** done

## Summary

Host-transfer workspace policy now comes from one const Tensor owner hook receiving checked logical bytes; common TensorView delegates both pure queries and concrete owners preserve backend-specific values.

## Verification

- cmake -S . -B /tmp/iom-01-review-04-cpu -DCMAKE_BUILD_TYPE=Debug && cmake --build /tmp/iom-01-review-04-cpu -j2 --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir /tmp/iom-01-review-04-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$' — configured, built all three targets, and all 3 tests passed
- common production source check over src/*.cpp and src/shared/* — no BackendKind policy switch or BackendKind:: selector found

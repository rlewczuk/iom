**Status:** done

## Summary

Common submit_binary_operation facade owns binary validation, workspace validation, request capture, dispatch, and failure mapping while add/mul/sub/div remain enum-forwarding public wrappers.

## Verification

- cmake -S . -B /tmp/iom-01-review-05-cpu -DCMAKE_BUILD_TYPE=Debug && cmake --build /tmp/iom-01-review-05-cpu -j2 --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir /tmp/iom-01-review-05-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$' — configured, built all three targets, and all 3 tests passed
- source structure check in src/device_ops_binary.cpp — one submit_binary_operation definition and four public enum-forwarding wrappers observed

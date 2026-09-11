**Status:** done

## Summary

Host transfers use caller-provided workspace staging and fixed transfer resources, with migrated APIs, callers, and obsolete pools removed.

## Verification

- cmake --build /tmp/iom-build-11 -j2 && ctest --test-dir /tmp/iom-build-11 --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests|iom_scalar_add_tests)$' — build succeeded and 4/4 tests passed

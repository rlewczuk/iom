**Status:** done

## Summary

CPU and TTNN use common bounded FIFO admission with transactional snapshots, parking, autonomous retirement, repeatable outcomes, queue-local limits, and safe teardown rollback.

## Verification

- cmake --build /tmp/iom-build-08 -j2 && ctest --test-dir /tmp/iom-build-08 --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests|iom_scalar_add_tests)$' — build succeeded and 4/4 tests passed

**Status:** done

## Summary

Binary operations use explicit borrowed workspace views with matching queries, transactional validation and leases, backend propagation, and exact SYCL workspace slices.

## Verification

- cmake --build /tmp/iom-build-07 -j2 && ctest --test-dir /tmp/iom-build-07 --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests|iom_scalar_add_tests)$' — build succeeded and 4/4 tests passed

**Status:** done

## Summary

Migrated core and shared backend OID assertions to signed non-throwing results, protected queue hooks, and device-bound queue construction.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-07-build --target iom_tests && ctest --test-dir /tmp/iom-002-eltwise-add-07-build --output-on-failure -R ^iom_tests$ — focused core suite passed 112/112
- ctest --test-dir /tmp/iom-002-eltwise-add-wave-build --output-on-failure -R ^(iom_tests|iom_cpu_tests)$ — finalized local wave core and CPU suites passed

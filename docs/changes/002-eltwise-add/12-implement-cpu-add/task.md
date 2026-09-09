**Status:** done

## Summary

Implemented CPU ADD for all required numeric NONE leaves through AddRequest snapshots, including packed and wide scalar arithmetic, right-aligned broadcast, transformed leading views, exact aliases, and three-owner inline completion; updated CPU capability expectations.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-12-build --target iom_cpu_tests iom_backend_conformance_cpu_tests iom_tests; ctest --test-dir /tmp/iom-002-eltwise-add-12-build --output-on-failure -R "^(iom_cpu_tests|iom_backend_conformance_cpu_tests|iom_tests)$" — all passed; throwaway packed I2, [1,1] broadcast/tail, and I16 in-place alias smoke passed.

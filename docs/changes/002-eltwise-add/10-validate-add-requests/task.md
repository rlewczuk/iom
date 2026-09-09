**Status:** done

## Summary

Implemented common ADD validation, immutable request snapshots, submit_add lifetime registration seam, deduplicated three-owner registry support, and deterministic fake/shared conformance coverage.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-10-build --target iom_tests iom_backend_conformance_cpu_tests; ctest --test-dir /tmp/iom-002-eltwise-add-10-build --output-on-failure -R "^(iom_tests|iom_backend_conformance_cpu_tests)$" — both passed; throwaway snapshot-aware ADD facade smoke passed.

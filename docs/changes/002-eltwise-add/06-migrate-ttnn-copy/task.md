**Status:** done

## Summary

Migrated TTNN copy conformance failures to signed OID results while retaining native storage and unsupported-operation contracts.

## Verification

- remote-exec ttnn 002-eltwise-add-06-ttnn: cmake configure/build iom_ttnn_smoke_tests — TTNN library and smoke executable built successfully
- remote-exec ttnn 002-eltwise-add-06-ttnn: ctest --test-dir build/ttnn --output-on-failure -R ^iom_ttnn_smoke_tests$ — TTNN smoke test passed

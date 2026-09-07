**Status:** done

## Summary

TTNN full-storage oracle observes padded native TILE cells and every owner plane via raw physical readback with sentinel-seeded standard-slot observation, plus a decisive padding-mutation probe test proving the observer exposes native padding writes while logical projection stays byte-identical.

## Verification

- ctest -R "iom_ttnn_(smoke|conformance)_tests" on combined train bv1 — 2/2 passed, conformance 17.20s
- iom_backend_coexistence_tests on combined train bv1 — passed 1.79s

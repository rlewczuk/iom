**Status:** done

## Summary

Delivered the operation-neutral shared conformance harness, independent-oracle cases, mapping coverage, backend fault/lifecycle hooks, and repaired missing namespace closures in both common and other harness headers.

## Verification

- Focused CPU build diagnosis: cmake --build build --target iom_backend_conformance_cpu_tests reached downstream CPU migration first, then exposed missing shared-header namespace closures; both closures are now repaired, with full CPU conformance rerun pending downstream leaf replay.

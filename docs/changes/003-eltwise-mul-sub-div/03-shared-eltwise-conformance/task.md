**Status:** done

## Summary

Delivered operation-neutral shared four-operation conformance fixtures, independent-oracle cases, mapping coverage, and backend fault/lifecycle hooks.

## Verification

- Focused CPU conformance gate: cmake -S . -B build && cmake --build build --target iom_backend_conformance_cpu_tests — blocked before harness compilation by intentional downstream CPU migration of removed ADD-specific hooks; shared harness source validation remains required after task04.

**Status:** done

## Summary

TTNN host transfers reuse retained staging: per-dtype per-plane upload slots and one download byte buffer owned by the device under the api_mutex, padding-only zeroing, failure-poisoning (discard/retire) with one-finish-per-region preserved; allocation-counter and failure-injection test seams plus two conformance cases pin zero fresh allocations across repeated same/smaller transfers and clean recovery after injected upload/download/staging faults.

## Verification

- ctest -R "^(iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests)$" on bv1 — 3/3 passed (conformance 18.23s)
- combined wave-2 train on bv1 — TTNN 3/3 passed (coexistence 1.80s)

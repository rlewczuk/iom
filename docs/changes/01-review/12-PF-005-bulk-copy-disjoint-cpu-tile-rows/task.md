**Status:** done

## Summary

CPU tile-row copies bulk-copy disjoint byte-aligned rows with a single std::memcpy where the standard encoding guarantees contiguity, with a retained element-wise fallback for non-contiguous layouts; behavior byte-identical to the prior per-element path per conformance oracles.

## Verification

- ctest -R "^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$" local CPU — 2/2 passed (34.50s)
- combined wave-3 train: CPU iom_tests+iom_cpu_tests+iom_backend_conformance_cpu_tests green

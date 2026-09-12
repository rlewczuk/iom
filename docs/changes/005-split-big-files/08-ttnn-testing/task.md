**Status:** done

## Summary

Centralized TTNN device/copy testing seams in testing.cpp/testing_internal.hpp and repaired batched completion so each successful admission completes exactly once through its queued callback.

## Verification

- After rebase onto integration head 590929a5e2b2711eb76882f9f331e2db31277541, remote-sync ttnn 005-split-big-files-08 and TTNN configure/build targets libiom iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests completed successfully.
- After the same rebase, exact TTNN ctest regex ^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$ passed 3/3; repaired coexistence path remained passing.
- wc -l src/ttnn/testing.cpp src/ttnn/testing_internal.hpp — 227 and 17 lines, both <=499.

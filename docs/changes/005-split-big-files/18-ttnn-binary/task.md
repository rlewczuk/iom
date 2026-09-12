**Status:** done

## Summary

Extracted TTNN binary plane execution and explicit scalar operation instantiations into binary.cpp while retaining host transfer/layout/copy paths in copy.cpp.

## Verification

- remote-sync ttnn 005-split-big-files-18 and remote TTNN configure/build targets libiom iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests — all targets built successfully after rebase onto TTNN testing integration.
- remote TTNN exact ctest regex ^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$ — 3/3 tests passed.
- wc -l src/ttnn/copy.cpp src/ttnn/binary.cpp src/ttnn/copy.hpp — 395, 209, and 45 lines; copy <=430, binary <=225, all touched files <=499.

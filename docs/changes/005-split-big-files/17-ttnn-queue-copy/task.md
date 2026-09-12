**Status:** done

## Summary

Moved the sole TTNN TtnnQueue::execute_copy definition into queue_copy.cpp and retained queue-core, transfer, binary, testing, and device ownership boundaries.

## Verification

- remote-sync ttnn 005-split-big-files-17 and remote TTNN configure/build targets libiom iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests — all targets built successfully.
- remote TTNN exact ctest regex ^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$ — 3/3 tests passed.
- wc -l src/ttnn/device.cpp src/ttnn/queue.cpp src/ttnn/queue_internal.hpp src/ttnn/queue_copy.cpp — 282, 415, 120, and 136 lines, all <=499.
- Source inspection found exactly one TtnnQueue::execute_copy definition in queue_copy.cpp and none in device.cpp; queue_copy.cpp is the only new copy-execution translation unit.

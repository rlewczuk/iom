**Status:** done

## Summary

Extracted TTNN queue lifecycle, submission, binary dispatch, completion, fence, and factory definitions into private queue.cpp/queue_internal.hpp while retaining the sole temporary execute_copy definition in device.cpp.

## Verification

- remote-sync ttnn 005-split-big-files-16 and remote TTNN configure/build targets libiom iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests — all targets built successfully after rebase onto TTNN binary integration.
- remote TTNN exact ctest regex ^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$ — 3/3 tests passed.
- wc -l src/ttnn/device.cpp src/ttnn/queue.cpp src/ttnn/queue_internal.hpp — 404, 415, and 120 lines; queue.cpp <=440 and all <=499.
- Source ownership inspection found one TtnnQueue::execute_copy definition in device.cpp, one TtnnDevice::create_ops definition in queue.cpp, one private TtnnQueue declaration in queue_internal.hpp, and binary_planes calls retained in queue.cpp.

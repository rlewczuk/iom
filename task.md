**Status:** done

## Summary

Implemented the backend-neutral `iom::detail::StagedWorker<Task>` and migrated CPU, CUDA, ROCm, and TTNN queue scaffolds to it. Centralized copy validation, identical-window detection, and unsupported-operation errors in `DeviceOps`; retained native task payloads, event fences, failure seams, and public APIs. Removed the shared CUDA/ROCm `GpuQueue<Policy>` state machine while preserving shared tiled-copy algorithms. Added helper callback-order, pre-link rollback, null-fence, and shutdown-drain coverage.

## Verification

- Local CPU/common configure, build, focused `StagedWorker*` tests, and full CTest passed: 3/3 CTest tests; focused helper tests 3/3 with 17 assertions.
- CUDA remote configure/build passed; CUDA smoke and conformance passed: 2/2 tests.
- ROCm remote configure/build passed; ROCm smoke and conformance passed: 2/2 tests.
- TTNN remote configure/build passed; TTNN smoke and conformance passed: 2/2 tests.
- Source audit found one `StagedWorker` definition, no stale `GpuQueue` state machine, and canonical `DeviceOps` helper definitions.
- Remote mirrors were cleaned after verification.

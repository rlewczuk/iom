**Status:** done

## Summary

Implemented the backend-neutral `iom::detail::StagedWorker<Task>` and migrated CPU, CUDA, ROCm, and TTNN queue scaffolds to it. Centralized copy validation, identical-window detection, and unsupported-operation errors in `DeviceOps`; retained native task payloads, event fences, failure seams, and public APIs. Removed the shared CUDA/ROCm `GpuQueue<Policy>` state machine while preserving shared tiled-copy algorithms. Added helper callback-order, pre-link rollback, null-fence, and shutdown-drain coverage.

## Verification

- `cmake -S . -B build-ar002-cpu -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-ar002-cpu -j && ./build-ar002-cpu/test/iom_tests -tc='StagedWorker*' && ctest --test-dir build-ar002-cpu --output-on-failure` — focused helper tests passed 3/3 with 17 assertions; full CTest passed 3/3.
- CUDA remote configure/build — CUDA smoke and conformance passed 2/2.
- ROCm remote configure/build — ROCm smoke and conformance passed 2/2.
- TTNN remote configure/build — TTNN smoke and conformance passed 2/2.
- Source audit — one `StagedWorker` definition, no stale `GpuQueue` state machine, and canonical `DeviceOps` helper definitions.
- Remote mirrors were cleaned after verification.

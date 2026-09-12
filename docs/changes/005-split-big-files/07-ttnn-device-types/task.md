**Status:** done

## Summary

Split TTNN device type tables and private TtnnDevice linkage into device_types.cpp and device_internal.hpp while preserving native per-plane ownership, context, registry, and factory behavior.

## Verification

- remote-sync ttnn 005-split-big-files-07 — exact TTNN worktree synchronized to bv1:agent-work/iom/005-split-big-files-07.
- remote-exec ttnn 005-split-big-files-07 cmake configure and cmake --build build/split-ttnn --target libiom iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests — all targets built successfully.
- remote-exec ttnn 005-split-big-files-07 ctest --test-dir build/split-ttnn --output-on-failure -R ^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$ — 3/3 tests passed.
- wc -l src/ttnn/device.cpp src/ttnn/device_types.cpp src/ttnn/device_internal.hpp — new files are 149 and 71 lines; device.cpp remains a 1026-line intermediate carrier for later queue extraction.

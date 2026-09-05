**Status:** blocked

## Summary

Added CUDA and ROCm no-throw event fence-and-destroy callbacks and stream synchronize-and-destroy finalizers after `StagedWorker` shutdown. Added multi-entry queue-destruction regressions with a retained null-fence failure between successful copies.

## Errors

- ROCm device AddressSanitizer verification — the documented host-instrumented build linked incompatible ASan runtimes and exited before the test; a HIP-only sanitizer build completed but ROCm clang reported AddressSanitizer unsupported for the target `gfx1201` offload image. The required ROCm memory-diagnostic acceptance evidence is therefore unavailable on the configured host. The implementation and retained worktree remain available for rerun on a host with supported ROCm device diagnostics.

## Verification

- Local CPU-only CMake/CTest — 3/3 tests passed.
- CUDA and ROCm queue-destruction focused regressions — passed, 22 and 19 assertions respectively.
- CUDA Compute Sanitizer memcheck and racecheck focused regression — zero errors and zero hazards.
- CUDA and ROCm complete CTest suites — 6/6 tests passed on each configured host.
- ROCm `rocprofv3 --runtime-trace --stats` focused regression — test passed and trace output was generated.
- Source audit — each backend has one `fence_and_destroy` and one `synchronize_and_destroy_stream`; queue destructors drain the worker before guarded stream finalization.

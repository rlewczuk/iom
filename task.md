**Status:** done

## Summary

Completed the CUDA driver-call injection seam and centralized the driver error helpers. `DriverCalls` now routes all six claimed lifecycle, factory, and context calls; `driver.cpp` owns the single table definition; `cuda_error` and `check_cuda` are inline in the private driver header; and CUDA smoke coverage verifies every seam entry and context-release lifecycle.

## Verification

- `remote-sync cuda ar005-cuda` from the assigned worktree — synchronized to `bv1:agent-work/iom/ar005-cuda`.
- Remote CMake configure/build with CUDA enabled for `iom_cuda_smoke_tests` and `iom_cuda_conformance_tests` — both targets built successfully.
- `iom_cuda_smoke_tests -tc="*driver-call seam intercepts every claimed driver call*"` — 1 test case and 13 assertions passed.
- The activation-failure, allocation-failure, and primary-context-success CUDA smoke cases — each ran individually; 1 test case passed in each run with 8, 7, and 8 assertions respectively.
- `ctest --test-dir build --output-on-failure -R "^iom_cuda_smoke_tests$"` — 1/1 test passed.
- `ctest --test-dir build --output-on-failure -R "^iom_cuda_conformance_tests$"` — 1/1 test passed.
- Remote source/default/helper audits — no direct claimed driver calls, exactly six driver default initializers, and one inline helper definition site in `src/cuda/driver.hpp`.
- `remote-clean cuda ar005-cuda` — verified remote mirror removed.

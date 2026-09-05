**Status:** blocked

## Summary

AR-006 was not implemented because both required predecessor tasks remain unfinished on the integration branch: AR-003 has not added `Device::supported_data_types()`, and AR-005 has not completed the CUDA driver-call seam. The exact deterministic worktree is retained for a later run.

## Errors

- `31-AR-003-expose-device-capabilities` — `docs/changes/0001-tensor-view/31-AR-003-expose-device-capabilities/task.md` is absent and `include/iom/device.hpp` still has no `supported_data_types()` method.
- `33-AR-005-unify-cuda-driver-seam` — `docs/changes/0001-tensor-view/33-AR-005-unify-cuda-driver-seam/task.md` is absent; `src/cuda/driver.hpp` still exposes only the partial three-entry `DriverCalls` table and `src/cuda/device.cpp` still defines `driver_calls`.

**Status:** done

## Summary

Added an injectable private CUDA primary-context lifecycle seam and an RAII guard in `make_cuda_device`. Retained contexts are released exactly once on activation or `CudaDevice` allocation failure, while successful construction defers the sole release to `CudaDevice` destruction. Added three hardware-backed CUDA smoke regressions and wired the private header into the CUDA targets.

## Verification

- `.agents/skills/remote-development/scripts/remote-exec cuda st004-ctx-guard 'cmake -S . -B build/cuda-st004 -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build/cuda-st004 --target iom_cuda_smoke_tests iom_cuda_conformance_tests'` — both targets built successfully on the CUDA host.
- `.agents/skills/remote-development/scripts/remote-exec cuda st004-ctx-guard 'ctest --test-dir build/cuda-st004 --output-on-failure -R "^iom_cuda_(smoke|conformance)_tests$"'` — 2/2 tests passed, including all three injected lifecycle cases.

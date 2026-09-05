**Status:** blocked

## Summary

Implemented a device-owned CUDA `TransferStreamPool` with exclusive non-blocking streams, poisoned-scope cleanup, and device-lifetime draining. Routed CUDA synchronous host transfers through the pool with asynchronous host-to-device copies and asynchronous staging memset, while preserving the existing public API. Added CUDA smoke coverage for concurrent transfers, multi-queue lifetime, failed-transfer stream dropping, and poisoned-scope cleanup.

## Verification

- `cmake -S . -B build/cuda-st005 -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build/cuda-st005 --target iom_cuda_smoke_tests iom_cuda_conformance_tests` — succeeded on CUDA host `bv1`.
- `ctest --test-dir build/cuda-st005 --output-on-failure -R "^iom_cuda_(smoke|conformance)_tests$"` — 2/2 tests passed on CUDA host `bv1`.
- `iom_cuda_smoke_tests --test-case="CUDA errored host transfer drops its stream from the pool"` — passed on CUDA host `bv1`.

## Errors

- `nsys profile --stats=true ...` overlap, multi-queue, and errored-transfer commands; `nsys profile --cuda-um-cpu-page-faults=true ...` poisoned-scope command — CUDA host `bv1` has no `nsys` executable (`command -v nsys` returned no path and `/usr/local/cuda/bin/nsys` is absent). The required Nsight overlap and invalid-handle observations therefore remain unverified; the feature branch is retained and not integrated.

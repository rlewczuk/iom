**Status:** done

## Summary

Implemented a device-owned CUDA `TransferStreamPool` with exclusive non-blocking streams, poisoned-scope cleanup, and device-lifetime draining. Routed CUDA synchronous host transfers through the pool with asynchronous host-to-device copies and asynchronous staging memset, while preserving the existing public API. Added CUDA smoke coverage for concurrent transfers, multi-queue lifetime, failed-transfer stream dropping, and poisoned-scope cleanup.

## Verification

- `cmake -S . -B build/cuda-st005 -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build/cuda-st005 --target iom_cuda_smoke_tests iom_cuda_conformance_tests` — passed on CUDA host `bv1` after merging current `main`.
- `ctest --test-dir build/cuda-st005 --output-on-failure -R "^iom_cuda_(smoke|conformance)_tests$"` — 2/2 tests passed.
- `nsys profile --stats=true --output=st005-overlap ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA host transfers on independent threads do not share a stream"` — test passed; exported synchronization data showed distinct streams 13 and 14 with overlapping intervals, including stream 14 `[361921224,361953094]` and stream 13 `[361932475,361970187]`.
- `nsys profile --stats=true --output=st005-multi-queue ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA host transfers from multiple queues on one device share the transfer-stream pool"` — test passed; no nonzero CUDA runtime return values in the exported report.
- `nsys profile --stats=true --output=st005-errored ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA errored host transfer drops its stream from the pool"` — test passed; no nonzero CUDA runtime return values in the exported report.
- `nsys profile --cuda-um-cpu-page-faults=true --stats=true --output=st005-poison ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA poisoned Scope drops its stream from the pool"` — test passed; no nonzero CUDA runtime return values or unified-memory page faults.
- `nsys profile --cuda-um-cpu-page-faults=true --stats=true --output=st005-error-quiescence ./build/cuda-st005/test/iom_cuda_conformance_tests --test-case="CUDA conformance: transfer failures keep metadata and ownership"` — 702/702 assertions passed; no nonzero CUDA runtime return values or unified-memory page faults.
- Transfer-path search for null stream arguments in `src/cuda/copy.cu` — no matches.

**Status:** done

## Summary

Implemented a device-owned CUDA `TransferStreamPool` with exclusive non-blocking streams, poisoned-scope cleanup, and device-lifetime draining. Routed CUDA synchronous host transfers through the pool with asynchronous host-to-device copies and asynchronous staging memset, while preserving the existing public API. Added CUDA smoke coverage for concurrent transfers, multi-queue lifetime, failed-transfer stream dropping, and poisoned-scope cleanup.

## Verification

- `cmake -S . -B build/cuda-st005 -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build/cuda-st005 --target iom_cuda_smoke_tests iom_cuda_conformance_tests` — passed on CUDA host `bv1` after merging the current integration branch.
- `ctest --test-dir build/cuda-st005 --output-on-failure -R "^iom_cuda_(smoke|conformance)_tests$"` — 2/2 tests passed.
- The same CUDA build and CTest command was rerun after the latest integration merge — passed; both CUDA targets built and both tests passed.
- `nsys profile --stats=true --output=st005-overlap-final ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA host transfers on independent threads do not share a stream"` — passed; exported synchronization data showed distinct streams 13 and 14 with overlapping intervals, including stream 14 `[359920897,359954811]` and stream 13 `[359945243,359971644]`.
- `nsys profile --stats=true --output=st005-multi-queue-final ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA host transfers from multiple queues on one device share the transfer-stream pool"` — passed.
- `nsys profile --stats=true --output=st005-errored-final ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA errored host transfer drops its stream from the pool"` — passed.
- `nsys profile --cuda-um-cpu-page-faults=true --stats=true --output=st005-poison-final ./build/cuda-st005/test/iom_cuda_smoke_tests --test-case="CUDA poisoned Scope drops its stream from the pool"` — passed with no unified-memory page faults.
- `nsys profile --cuda-um-cpu-page-faults=true --stats=true --output=st005-error-quiescence-final ./build/cuda-st005/test/iom_cuda_conformance_tests --test-case="CUDA conformance: transfer failures keep metadata and ownership"` — 702/702 assertions passed with no unified-memory page faults.
- Final Nsight SQLite reports for overlap, multi-queue, errored-transfer, poisoned-scope, and error-quiescence runs each reported zero nonzero CUDA runtime return values.
- Transfer-path search for null stream arguments in `src/cuda/copy.cu` — no matches.

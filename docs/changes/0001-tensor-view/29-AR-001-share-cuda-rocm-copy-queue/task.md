**Status:** blocked

## Summary

Extracted the standard-tiled CUDA/ROCm kernels, view mapping, host-transfer dispatcher, and pre-AR-002 policy queue into `src/shared/standard_tiled_copy.inl`. Added the checked backend-neutral staging-size helper in `include/iom/gpu_algorithm.hpp`, typed CUDA and ROCm policy front ends, and CMake source-list entries.

## Verification

- `cmake -S . -B build -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON && cmake --build build -j2` — passed.
- `ctest --test-dir build --output-on-failure -R 'iom_(tests|cpu_tests|backend_conformance_cpu_tests)'` — 3/3 tests passed.
- Remote ROCm configure/build — passed on `bv2`.

## Errors

- Remote ROCm `ctest --test-dir build --output-on-failure -R "iom_(rocm_conformance_tests|backend_coexistence_tests)"` — both runs failed at the first HIP scatter/gather launch with `hipErrorIllegalState`; the same conformance case also failed from the unmodified integration checkout, so the configured accelerator host cannot provide passing runtime verification.
- Remote CUDA synchronization via `remote-sync cuda ar001` — `Host key verification failed`; CUDA configure/build and conformance verification remain unavailable.

**Status:** blocked

## Summary

Extracted the standard-tiled CUDA/ROCm kernels, view mapping, host-transfer
dispatcher, and pre-AR-002 policy queue into
`src/shared/standard_tiled_copy.inl`. Added the checked backend-neutral
staging-size helper in `include/iom/gpu_algorithm.hpp`, typed CUDA and ROCm
policy front ends, the existing CUDA driver-call seam, and CMake source-list
entries. The task branch retains the implementation for a later verification
retry.

## Verification

- `cmake -S . -B build-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-cpu -j && ctest --test-dir build-cpu --output-on-failure` — 3/3 tests passed.
- C++20 staging-helper smoke program — `compute_staging_size(0) == 0`,
  `(1) == 4`, `(4) == 4`, and `size_t` maximum throws
  `std::overflow_error` with the required message.
- ROCm target configured and built on `bv2` with
  `cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=ON -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DROCM_PATH=/opt/rocm && cmake --build build -j --target iom_rocm_conformance_tests` — passed.
- `ctest --test-dir build --output-on-failure -R "^iom_rocm_conformance_tests$"`
  on `bv2` — failed at the first HIP scatter/gather launch with
  `hipErrorIllegalState`; the same conformance failure reproduced from a clean
  unmodified integration checkout.
- Source audit — no vendor tokens in shared files, exactly one shared inclusion
  per CUDA/ROCm frontend, no backend duplicate tiled-copy or queue definitions,
  one staging-size helper, and the exact five-field shared `Task`.

## Errors

- CUDA remote synchronization via `remote-sync cuda ar001` — `Host key
  verification failed`; CUDA configure, build, and conformance verification
  remain unavailable. Do not accept the unknown host key automatically.

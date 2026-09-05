**Status:** blocked

## Summary

Extracted the standard-tiled CUDA/ROCm kernels, view mapping, host-transfer
dispatcher, and pre-AR-002 policy queue into
`src/shared/standard_tiled_copy.inl`. Added the checked backend-neutral
staging-size helper in `include/iom/gpu_algorithm.hpp`, typed CUDA and ROCm
policy front ends, the existing CUDA driver-call seam, and CMake source-list
entries. Fixed ROCm staging zeroing to use the operation stream, avoiding a
default-stream race with the nonblocking gather stream. Unoptimized ROCm
builds now use the minimum `-O1` device optimization required to avoid
unsupported hostcall code objects on the target APU.

## Verification

- `cmake -S . -B build-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-cpu -j && ctest --test-dir build-cpu --output-on-failure` — 3/3 tests passed.
- C++20 staging-helper smoke program — `compute_staging_size(0) == 0`,
  `(1) == 4`, `(4) == 4`, and `size_t` maximum throws
  `std::overflow_error` with the required message.
- Clean pre-change `main`, configured as ROCm Release on `bv2`, reproduced the
  original ROCm logical transfer mismatch for `DataType::BOOL` at byte 1.
- Fixed branch, configured on `bv2` both without `CMAKE_BUILD_TYPE` and as
  Release, built all targets and passed `ctest --test-dir build
  --output-on-failure`: 6/6 tests in each configuration, including ROCm
  conformance and backend coexistence.
- Source audit — no vendor tokens in shared files, exactly one shared inclusion
  per CUDA/ROCm frontend, no backend duplicate tiled-copy or queue definitions,
  one staging-size helper, and the exact five-field shared `Task`.

## Errors

- CUDA remote synchronization via `remote-sync cuda ar001` — `Host key
  verification failed`; CUDA configure, build, and conformance verification
  remain unavailable. Do not accept the unknown host key automatically.

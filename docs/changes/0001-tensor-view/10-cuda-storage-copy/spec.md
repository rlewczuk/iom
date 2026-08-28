# Implement CUDA storage, transfers, and asynchronous copy

**Order:** 10
**Priority:** P1 — this completes CUDA's required materialization and copy contract.
**Blocked by:** `09-cuda-buildable-scaffold`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

The CUDA device materializes every unquantized leaf type through its injected device allocator, performs exact synchronous logical host transfers, executes in-order asynchronous same-device view copies, and passes the shared CPU-reference conformance suite on hardware.

## Scope

- Replace the CUDA scaffold's tensor/queue capability failures with complete standard-layout storage, transfer, and copy behavior.
- Instantiate the existing shared conformance suite unchanged.

## Implementation references

- **Modify:** `src/cuda/device.cpp` — implement `create_tensor`, `create_ops`, allocation ownership, CUDA stream/queue construction, and error propagation.
- **Create (planned):** `src/cuda/copy.cu` — implement logical standard-layout host transfer/copy work honoring plane offsets, strides, padding, and every leaf width.
- **Create (planned):** `test/cuda/test_cuda_conformance.cpp` — provide a recording CUDA-device allocator, factory, and complete supported-type table to `test/backend/backend_conformance.hpp`.
- **Modify:** `CMakeLists.txt` — compile CUDA sources and runtime links only into `iom_cuda`.
- **Modify:** `test/CMakeLists.txt` — add `iom_cuda_conformance_tests` under `CUDA_ENABLED`.
- **Read:** `src/cpu/device.cpp` — match validation and logical behavior without importing CPU storage assumptions.
- **Read:** `test/backend/backend_conformance.hpp` — invoke the shared cases without CUDA-specific forks.

## Requirements

- `create_tensor` validates before allocation and supports every declared `DataType` with `QuantizationFormat::NONE`; grouped formats and invalid enums fail before allocator use.
- The caller allocator returns CUDA device addresses valid for the factory-owned context. Allocate exactly `tiled_storage_nbytes()` once, enforce 32-byte base alignment, free a misaligned or partially constructed allocation once, and free successful storage once at tensor destruction.
- Storage is the exact standard 16x16 tile format, including leaf widths, little-endian multi-byte values, and least-significant-bit-first sub-byte/six-bit slots.
- Host transfers are synchronous when the public call returns, require exact logical byte counts, honor all safe view offsets/strides, hide padding, perform no conversion, ignore input tail bits, zero output tail bits, and reject invalid `BOOL` before any device write.
- Each `create_ops()` owns a backend-default in-order CUDA stream/queue and uses the common queue ID and submission sequence contract. Multiple queues from one device are independent.
- Asynchronous `copy` accepts only exact-device views with identical logical metadata, moves logical values in view-coordinate order without padding, conversion, or host staging, and supports every safe transformed view.
- Identical windows consume a sequence and return a waitable no-op. Non-identical overlap has unspecified values. Validation and synchronous launch errors occur before sequence consumption or destination writes.
- CUDA execution errors are stored and rethrown on every wait for the failed sequence. View, owner, native allocation, and variable metadata addresses remain caller-stable until completion.
- Unsupported compute methods throw `std::runtime_error` during capability validation without submission or output changes.
- Storage allocator calls occur only at tensor construction/destruction, never during transfers, view transforms, or copies.
- The conformance executable covers every leaf type and fails rather than skips if an enabled CUDA runtime/device is unavailable.

## Non-goals

- Numerical compute kernels, implicit host staging, or a CUDA-specific public layout.
- Borrowed contexts, stream tuning options, or multi-backend runtime dispatch.
- Beginning SYCL work before this target passes on CUDA hardware.

## Acceptance criteria

- [ ] Every leaf type materializes with one aligned exact-size CUDA allocation and deterministic cleanup for all failure paths.
- [ ] Full and transformed host transfers match independent CPU bytes for padding, high rank, nested transforms, sub-byte packing, endianness, and `BOOL`.
- [ ] Full/view asynchronous copies match CPU and obey same-device, metadata, validation, no-op, error, and wait rules.
- [ ] Unsupported compute operations fail before submission, and transfer/copy performs no tensor-storage allocation or implicit staging.
- [ ] CUDA-enabled conformance runs without skips; CUDA-disabled core and already-complete ROCm builds remain unaffected.

## Verification

- `cmake -S . -B build/cuda -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda`
- `cmake --build build/cuda --target iom_cuda_conformance_tests`
- `ctest --test-dir build/cuda --output-on-failure -R '^iom_cuda_conformance_tests$'`

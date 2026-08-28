# Implement SYCL storage, transfers, and asynchronous copy

**Order:** 12
**Priority:** P1 — this completes SYCL's required materialization and copy contract.
**Blocked by:** `11-sycl-buildable-scaffold`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

The SYCL device materializes every unquantized leaf type through a caller allocator compatible with its owned context, performs exact synchronous logical host transfers, executes in-order asynchronous same-device view copies, and passes shared CPU-reference conformance on hardware.

## Scope

- Replace the SYCL scaffold's capability failures with complete standard-layout tensor, transfer, and copy implementations.
- Prove allocator/context compatibility and instantiate the shared conformance harness unchanged.

## Implementation references

- **Modify:** `src/sycl/device.cpp` — implement `create_tensor`, `create_ops`, allocation ownership, in-order queue construction, and SYCL exception propagation.
- **Create (planned):** `src/sycl/copy.cpp` — implement SYCL logical transfer/copy work for standard tiled storage and arbitrary safe plane descriptors.
- **Create (planned):** `test/sycl/test_sycl_conformance.cpp` — provide a recording device allocator compatible with the selected ordinal/context, factory, and complete supported-type table to `test/backend/backend_conformance.hpp`.
- **Modify:** `CMakeLists.txt` — compile and link SYCL implementation only into `iom_sycl`.
- **Modify:** `test/CMakeLists.txt` — add `iom_sycl_conformance_tests` when SYCL is enabled.
- **Read:** `src/cpu/device.cpp` — match logical validation and transfer semantics.
- **Read:** `test/backend/backend_conformance.hpp` — reuse all cases without backend-specific exceptions.

## Requirements

- `create_tensor` validates before allocation and supports every declared `DataType` with `QuantizationFormat::NONE`; grouped formats and invalid enums fail before allocator use.
- The injected allocator's returned pointer must be valid for copy and kernel operations in the factory-owned SYCL context. Detect and reject an incompatible pointer before submitting transfer/copy work; do not expose the native context or change the `Allocator` or factory signatures.
- Allocate exactly `tiled_storage_nbytes()` once, enforce 32-byte base alignment, free rejected/partially constructed storage once, and free successful storage once at tensor destruction.
- Use the exact engine-standard 16x16 layout, including every leaf width, little-endian multi-byte encoding, and least-significant-bit-first sub-byte/six-bit slots.
- Host transfers are synchronous at return, exact-size, view-aware, padding-hidden, conversion-free, tail-bit canonical, and atomic with respect to argument/`BOOL` validation before device writes.
- Every `create_ops()` owns a backend-default `sycl::queue` with in-order semantics in the device's owned context. Queues are independent and use common queue IDs/sequences.
- Asynchronous `copy` requires exact-device views and identical logical metadata, honors all safe offsets/strides, and copies logical values only, with no numeric conversion, padding copy, or implicit host staging.
- Identical windows return a waitable no-op. Validation or synchronous submit failure consumes no sequence and writes nothing; non-identical overlap remains unspecified.
- Asynchronous SYCL errors are retained per failed sequence and rethrown on repeated waits. Views, owners, allocations, and metadata remain stable through completion.
- Unsupported compute methods reject capability before submission or output changes.
- Tensor-storage allocator calls are limited to tensor construction/destruction.
- Enabled conformance covers every leaf type on real eligible SYCL hardware and never silently skips or falls back to a CPU device.

## Non-goals

- Numerical compute kernels, public SYCL context exposure, borrowed contexts, or allocator API changes.
- Host fallback, implicit staging, a SYCL-specific public layout, or queue tuning options.
- Starting TTNN before this hardware conformance target passes.

## Acceptance criteria

- [ ] Every leaf type materializes with one exact-size aligned allocation accepted by the owned context; incompatible allocator pointers fail deterministically with correct cleanup.
- [ ] Full and transformed transfers match independent CPU bytes for padding, high ranks, nested transforms, sub-byte packing, endianness, and `BOOL`.
- [ ] Asynchronous copies match CPU and obey device/metadata/no-op/error/wait rules without host staging or allocator calls.
- [ ] Unsupported compute operations fail before submission.
- [ ] Enabled SYCL conformance uses accelerator hardware without skips; disabled builds retain no SYCL dependency and other backends remain independently selectable.

## Verification

- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF`
- `cmake --build build/sycl --target iom_sycl_conformance_tests`
- `ctest --test-dir build/sycl --output-on-failure -R '^iom_sycl_conformance_tests$'`

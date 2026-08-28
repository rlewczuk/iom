# Implement ROCm storage, transfers, and asynchronous copy

**Order:** 08
**Priority:** P1 — this completes the first accelerator's required materialization and copy behavior.
**Blocked by:** `07-rocm-buildable-scaffold`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

The ROCm device materializes every unquantized leaf type through its injected device allocator, performs exact logical host transfers, executes in-order asynchronous same-device view copies, and passes the shared CPU-reference conformance suite on hardware.

## Scope

- Replace the scaffold's tensor/queue capability failures with complete standard-layout ROCm tensor, transfer, and copy implementations.
- Instantiate the existing shared conformance harness without forking its observable cases.

## Implementation references

- **Modify:** `src/rocm/device.cpp` — implement `create_tensor`, `create_ops`, allocator ownership, queue construction, and runtime error propagation.
- **Create (planned):** `src/rocm/copy.hip` — implement logical standard-layout host transfer/copy kernels or runtime operations that honor plane offsets and strides for every leaf width.
- **Create (planned):** `test/rocm/test_rocm_conformance.cpp` — provide a recording HIP-device allocator, ROCm device factory, and the complete supported-type table to `test/backend/backend_conformance.hpp`.
- **Modify:** `CMakeLists.txt` — compile HIP sources only into `iom_rocm` with target-local settings.
- **Modify:** `test/CMakeLists.txt` — add hardware-backed `iom_rocm_conformance_tests` when ROCm is enabled.
- **Read:** `src/cpu/device.cpp` — reuse validation order and logical behavior, not CPU storage assumptions.
- **Read:** `test/backend/backend_conformance.hpp` — invoke the shared suite unchanged.

## Requirements

- `create_tensor` validates before allocation and supports every declared `DataType` with `QuantizationFormat::NONE`; every grouped format and invalid enum is rejected before allocator use.
- The injected allocator returns addresses in ROCm device address space accepted by the owned context's copy and kernel APIs. Each tensor performs one exact-size allocation, enforces at least 32-byte alignment, frees misaligned or partially constructed storage once, and frees successful storage once at destruction.
- ROCm storage uses the exact standard 16x16 tiled layout, leaf widths, little-endian scalar representation, and sub-byte least-significant-bit packing used by CPU.
- Host transfers are synchronous at the public return boundary, require exact logical byte counts, never expose padding, perform no conversion, honor every safe view offset/stride, zero output tail bits, and reject invalid `BOOL` before any device write.
- `create_ops()` creates a backend-default in-order ROCm queue bound to the device context. Separate queues are independent and use the common process-wide queue-ID/token implementation.
- Asynchronous `copy` accepts only views from the exact same `Device`, requires identical logical metadata, copies logical values without padding/conversion/host staging, and honors all safe leading-view transforms.
- Identical windows submit a waitable no-op. Non-identical overlap has unspecified destination values. Validation and synchronous launch failure consume no sequence and do not modify output.
- Runtime execution failures surface from repeated `wait(oid)` calls as the same stored failure. Every participating view, owner, allocation, and variable-length metadata remains caller-owned and stable through completion.
- `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa` reject unsupported capability before submission or output changes.
- Transfer, transform, and copy paths never call the tensor-storage allocator after construction and before destruction.
- The conformance executable covers every declared leaf type and must fail, not skip, when `ROCM_ENABLED=ON` but required hardware/runtime is unavailable.

## Non-goals

- Numerical compute kernels or implicit CPU staging.
- A ROCm-specific layout, public native descriptor, borrowed context, or queue tuning options.
- Work on another accelerator before this hardware-backed conformance target passes.

## Acceptance criteria

- [ ] Every leaf type materializes in one aligned exact-size ROCm allocation with deterministic cleanup on all failures.
- [ ] Directly seeded full and transformed views transfer bit-for-bit against CPU for padding, ranks above four, nested transforms, sub-byte values, endianness, and `BOOL`.
- [ ] Asynchronous full/view copies match CPU, preserve token/wait semantics, and reject metadata/foreign-device failures before writes.
- [ ] No required case skips on an enabled backend, and every unsupported compute method fails before submission.
- [ ] Core-plus-CPU still builds with ROCm disabled; ROCm-specific sources and dependencies disappear from that build.

## Verification

- `cmake -S . -B build/rocm -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm`
- `cmake --build build/rocm --target iom_rocm_conformance_tests`
- `ctest --test-dir build/rocm --output-on-failure -R '^iom_rocm_conformance_tests$'`

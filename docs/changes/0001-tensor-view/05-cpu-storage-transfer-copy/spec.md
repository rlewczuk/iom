# Implement CPU materialization, transfers, and asynchronous copy

**Order:** 05
**Priority:** P1 — CPU is the executable reference path required before any accelerator backend.
**Blocked by:** `04-deviceops-queue-contract`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

The always-built CPU backend materializes every unquantized leaf type in the standard 16x16 tiled layout, owns storage through the injected allocator, performs exact synchronous logical host transfers, and executes in-order asynchronous view copies.

## Scope

- Add the CPU factory, concrete device/tensor/queue, build target integration, and focused CPU tests.
- Deliver the complete CPU storage/transfer/copy path in this task; mathematical compute operations remain explicit unsupported capabilities.

## Implementation references

- **Create (planned):** `include/iom/cpu/device.hpp` — declare `make_cpu_device(Allocator&)` and no backend implementation types.
- **Create (planned):** `src/cpu/device.cpp` — implement the CPU device, standard-layout tensor ownership, logical host transfers, in-order queue, and asynchronous copy.
- **Create (planned):** `test/cpu/test_cpu.cpp` — recording allocator, direct-layout inspection, transfer, device, ownership, and copy tests.
- **Modify:** `CMakeLists.txt` — compile the CPU backend into the always-built `libiom`; do not create an option that disables CPU.
- **Modify:** `test/CMakeLists.txt` — add a focused `iom_cpu_tests` executable and CTest entry.
- **Read:** `include/iom/alloc.hpp` — reuse `Allocator::alloc/free/reset` unchanged and its constructor-selected alignment model.
- **Read:** `src/iom.cpp` — reuse the checked standard-layout address calculation and common queue-token machinery rather than duplicating them.

## Requirements

- `make_cpu_device(Allocator&)` returns a distinct `Device` with `BackendKind::CPU` and backend ordinal zero. The allocator and device must outlive their tensors and queues.
- `create_tensor` validates before allocation, supports every declared `DataType` only with `QuantizationFormat::NONE`, and calls `allocator.alloc(spec.tiled_storage_nbytes())` exactly once.
- A null allocation throws `std::bad_alloc` without `free`. A non-null address not aligned to at least 32 bytes is freed exactly once and rejected with `std::runtime_error`. Any later construction failure frees owned storage before propagation.
- Successful tensor destruction calls `allocator.free(address)` exactly once. Do not catch or normalize allocator exceptions; `free` throwing remains a caller contract violation.
- CPU `native_handle()` is the allocation base address. Views combine that handle with their logical plane offset and strides; view transforms never alter the owner allocation.
- Standard storage is contiguous 16x16 tiles ordered by leading plane, tile row, tile column, row, and column. Padding is stored but never exposed to host buffers and never affects logical copy results.
- `copy_from_host` and `copy_to_host` are synchronous, require exactly `view.spec().logical_nbytes()`, traverse the view in logical row-major coordinate order, and honor arbitrary safe plane offsets and strides.
- Host encoding performs no numeric conversion: multi-byte values are little-endian; sub-byte and six-bit values begin at `i * bits_per_element`, least-significant bit first; input tail bits are ignored and output tail bits are zero. `BOOL` bytes are exactly zero or one.
- Host writes validate byte count and every `BOOL` byte before modifying any destination value. Wrong-size host reads/writes also fail before destination writes.
- `create_ops()` returns a new in-order asynchronous CPU queue on every call. Multiple queues from one device may address that device's tensors and have independent queue IDs.
- `copy` requires both views to belong to the exact creating `Device` instance and have identical shape, leaf type, and quantization. Cross-device and cross-backend copies fail before writes and never stage through a host buffer.
- Copy traverses source and destination in logical view-coordinate order, honors every safe dense or strided view, copies no padding, and performs no conversion. Identical windows still submit a waitable no-op. Overlapping non-identical windows have unspecified destination values.
- A window is identical only when owner, logical shape, plane offset, and plane strides all match; equal native handles alone are insufficient.
- Validation completes before submission. Successful work is reported through the common `oid`/`wait` contract, executes in call order, and retains asynchronous failures for repeated waits.
- `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa` reject capability with `std::runtime_error` before submission, sequence consumption, or output changes.
- Host transfer, view transform, queue submission, and copy call neither `alloc` nor `free` on tensor storage.
- Queue destruction does not implicitly wait or cancel; tests and callers wait for every submitted operation first.

## Non-goals

- Numerical compute kernels or compute conformance.
- Implicit host/device synchronization, transfer conversion, or allocator replacement.
- Tensor move support, storage reset, a default tensor, or a CPU global singleton.

## Acceptance criteria

- [ ] Every unquantized leaf type materializes with one exact-size, 32-byte-aligned allocation and one matching free.
- [ ] Null, misaligned, unsupported, throwing-construction, and destruction paths exhibit the specified allocation counts and exception classes.
- [ ] Independently encoded host patterns for every leaf type land in exact tiled slots; direct allocation inspection covers endianness, sub-byte positions, both padded dimensions, multiple leading planes, and rank above four.
- [ ] Inverse reads hide padding, zero unused tail bits, and do not rely only on round-trip cancellation.
- [ ] Full, interior, stepped, selected, permuted, reshaped, and nested views transfer and copy the exact logical planes without allocator calls.
- [ ] Identical-window no-op, metadata mismatch, foreign device, wrong span, invalid `BOOL`, and injected asynchronous failures obey validation/write/sequence rules.
- [ ] Two CPU queues are independent and same-queue submissions execute in order.
- [ ] Every unimplemented compute method fails capability validation without submitting or modifying output.

## Verification

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON`
- `cmake --build build/cpu --target iom_cpu_tests`
- `ctest --test-dir build/cpu --output-on-failure -R '^iom_cpu_tests$'`

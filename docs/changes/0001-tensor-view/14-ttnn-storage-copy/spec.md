# Implement TTNN native storage, transfers, and asynchronous copy

**Order:** 14
**Priority:** P1 — this completes TTNN's required native materialization and copy contract.
**Blocked by:** `13-ttnn-buildable-scaffold`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

The TTNN device materializes an explicit nonempty leaf-type set including `BF16` in native tiled storage, maps common logical-plane views onto that storage, performs exact host transfers and in-order asynchronous copies, and passes shared CPU-reference conformance for every supported type.

## Scope

- Replace the TTNN scaffold's capability failures with native tensor ownership, logical host transfers, queues, and same-device asynchronous copy.
- Define one explicit supported-type table and use it for creation validation and conformance parameterization.

## Implementation references

- **Modify:** `src/ttnn/device.cpp` — implement native tensor creation/destruction, logical-plane mapping, `create_ops`, queue ownership, and runtime failures.
- **Create (planned):** `src/ttnn/copy.cpp` — implement native logical region transfers and copies for arbitrary safe plane offsets/strides.
- **Create (planned):** `test/ttnn/test_ttnn_conformance.cpp` — declare the nonempty supported-type table including `BF16`, instantiate `test/backend/backend_conformance.hpp`, and verify rejection of every other leaf type.
- **Modify:** `CMakeLists.txt` — compile TTNN implementation only into `iom_ttnn`.
- **Modify:** `test/CMakeLists.txt` — add hardware-backed `iom_ttnn_conformance_tests` under `TTNN_ENABLED`.
- **Read:** `src/cpu/device.cpp` — match logical behavior while retaining TTNN-native physical padding/storage.
- **Read:** `test/backend/backend_conformance.hpp` — run the same logical transfer/view/copy cases for every supported type.

## Requirements

- Define one explicit, nonempty TTNN supported-`DataType` table containing at least `BF16`. `create_tensor` accepts exactly that table with `QuantizationFormat::NONE` and rejects every other leaf type and every grouped format before native allocation.
- TTNN allocates and owns native materialized tensors through its runtime, never through `iom::Allocator`. Native storage byte count and padding may differ from `TensorSpec::tiled_storage_nbytes()`.
- `TensorView::spec()` retains the requested logical shape/type. Plane offsets and strides count logical leading planes and must map every safe slice/select/permutation/contiguous reshape onto the corresponding TTNN-native region.
- `native_handle()` is a pointer to the backend-owned native tensor object, not an exposed TTNN value type or standard-layout byte address. Common headers reveal no TTNN internals.
- Host transfers are synchronous at return, require exact logical byte size, preserve the specified host leaf encoding without conversion, honor arbitrary safe views, expose no native padding, canonicalize output tail bits, and validate `BOOL` before writes when `BOOL` is supported.
- `create_ops()` produces an in-order backend-default TTNN queue using the common process-wide token contract; multiple queues from one device are independent.
- Asynchronous `copy` accepts only views from the exact same TTNN `Device` with identical logical metadata, copies values in logical view-coordinate order, and performs no implicit host staging or format conversion.
- Identical windows submit a waitable no-op. Non-identical overlap has unspecified values. Validation and synchronous submission failures occur before sequence consumption or destination writes.
- Runtime failures are retained and rethrown by repeated waits. Participating views, owner tensors, native objects, and metadata stay stable through completion.
- Unsupported compute methods reject capability before submission or output changes.
- Enabled conformance runs every shared case for every supported table entry, verifies rejection of every non-entry, and fails rather than skips when TTNN hardware/runtime is unavailable.

## Non-goals

- Requiring TTNN to use the engine-standard physical layout or support every leaf type.
- Grouped TT block-float/quantized formats, numerical compute kernels, or native-storage exposure.
- Host staging, borrowed contexts, queue tuning, or allocator integration.

## Acceptance criteria

- [ ] The supported table is explicit, nonempty, includes `BF16`, and creation rejects every type outside it before native allocation.
- [ ] Native tensor/context resources are created and destroyed exactly once with no `iom::Allocator` calls.
- [ ] Full and nested transformed views transfer and copy bit-for-bit against CPU for all supported types, padding, and high leading ranks.
- [ ] Copy obeys exact-device, metadata, no-op, validation, asynchronous error, and repeated-wait rules without host staging.
- [ ] Unsupported compute and unsupported leaf types fail before submission/output changes; enabled hardware tests never skip.

## Verification

- `cmake -S . -B build/ttnn -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF`
- `cmake --build build/ttnn --target iom_ttnn_conformance_tests`
- `ctest --test-dir build/ttnn --output-on-failure -R '^iom_ttnn_conformance_tests$'`

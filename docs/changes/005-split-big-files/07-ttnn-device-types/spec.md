# Split TTNN device types

**Order:** 07
**Priority:** P0 — exposes private device linkage and type helpers required by queue extraction.
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

TTNN device linkage and device-type arithmetic are split into responsibility-complete private source files without changing the public API or runtime behavior. `src/ttnn/device_internal.hpp` provides the private `TtnnDevice` declaration and the private declarations needed by other TTNN translation units. `src/ttnn/device_types.cpp` owns the canonical supported-type table, native-type mapping, carrier and checked shape helpers, and `ttnn_supported_data_types()`. `src/ttnn/device.cpp` continues to own the concrete tensor/workspace owners, device create methods, and factory; `TtnnTensor` remains local to that translation unit.

## Scope

Create only these production files and modify the TTNN target source list:

- `src/ttnn/device_internal.hpp` — private declaration/linkage boundary for `TtnnDevice` and the extracted type/check helpers.
- `src/ttnn/device_types.cpp` — extracted supported/native type data and pure type/shape checks.
- `CMakeLists.txt:114-142` — add the new TTNN private header and `device_types.cpp` to `iom_ttnn` while retaining every existing TTNN source, private header, and public header.

Extract the current type/check block from `src/ttnn/device.cpp:122-249` and `ttnn_supported_data_types()` from `src/ttnn/device.cpp:1110-1112`. Include the private header from the TTNN implementation files that need `TtnnDevice` or the extracted helpers. Keep the device class's fields, accessors, destructor/quarantine drain, and private visibility unchanged. Keep `TtnnTensor`, `TtnnWorkspace`, tensor/workspace creation, device creation, and the factory in `src/ttnn/device.cpp`; this task does not split queue, testing, binary, or copy responsibilities.

## Implementation references

- Existing TTNN type table, key array, uniqueness assertion, dtype lookup, carrier factor, ordinal error helper, checked narrowing, and leading-plane count: `src/ttnn/device.cpp:117-249` (the type table begins at line 122).
- Existing `TtnnDevice` declaration and exact field/destruction/accessor behavior: `src/ttnn/device.cpp:251-313`.
- Existing `TtnnTensor` construction, native plane layout, ownership, registry registration, release/quarantine, and host-transfer forwarding: `src/ttnn/device.cpp:315-453`.
- Existing zero-byte `TtnnWorkspace`: `src/ttnn/device.cpp:455-464`.
- Existing tensor/workspace/device factory definitions: `src/ttnn/device.cpp:1114-1155`.
- Private transfer declarations and native cleanup/quarantine owner: `src/ttnn/copy.hpp` and `src/ttnn/registry_state.hpp`.
- Retained staging lifetime and device ownership: `src/ttnn/staging.hpp`.
- Public TTNN capability/factory declarations, which must remain unchanged: `include/iom/ttnn/device.hpp:12-31`.
- Canonical table order, unsupported leaf/quantization rejection, boundary overflow behavior, tiled storage, and post-rejection health: `test/ttnn/test_ttnn_conformance.cpp:448-604`.
- Native 32x32 padded storage, logical transfer behavior, ownership/lifetime, and device recreation smoke coverage: `test/ttnn/test_ttnn_conformance.cpp:606-704, 886-889` and `test/ttnn/test_ttnn_smoke.cpp:15-69`.

## Requirements

1. **Private linkage only.** Declare `TtnnDevice` in `src/ttnn/device_internal.hpp` with its existing final `Device` inheritance, constructor, deleted copy operations, destructor, `backend_kind()`, `backend_device()`, `supported_data_types()`, `create_tensor()`, `create_workspace()`, `create_ops()`, `mesh()`, `api_mutex()`, `host_staging()`, and `registry_state()` interfaces. Preserve the exact member declaration order and types: ordinal, `TtnnHostStaging`, native mesh `shared_ptr`, API mutex, and `detail::RegistryState`. Keep this header private; do not add it to any public include directory or public backend header. The header may forward-declare native/private types as needed, but must not introduce a new abstraction, registry, alias, or shim.

2. **Extract all type/check definitions exactly once.** `src/ttnn/device_types.cpp` must own the canonical table/mapping and private helper definitions. Make helper linkage explicit across TTNN translation units through the private header, with no duplicate anonymous copies in `device.cpp` or elsewhere. Keep `ttnn_supported_data_types()` as the sole public definition and return a stable span over the canonical key array.

3. **Preserve the exact 22-entry table and order.** The table is, in order:
   `BOOL -> UINT8`; `I2 -> UINT32`; `U2 -> UINT32`; `I4 -> UINT32`; `U4 -> UINT32`; `I8 -> UINT8`; `U8 -> UINT8`; `I16 -> UINT16`; `U16 -> UINT16`; `I32 -> UINT32`; `U32 -> UINT32`; `I64 -> UINT32`; `U64 -> UINT32`; `F4_E2M1 -> UINT32`; `F6_E2M3 -> UINT32`; `F6_E3M2 -> UINT32`; `F8_E4M3FN -> UINT32`; `F8_E5M2 -> UINT32`; `F16 -> UINT32`; `BF16 -> BFLOAT16`; `F32 -> FLOAT32`; `F64 -> UINT32`.
   Build the public key span from this same table, retain the size assertion and compile-time pairwise key-uniqueness assertion, and do not sort, deduplicate, append, or condition the entries. `F8_E8M0` remains unsupported. `DataType` values not in the table remain rejected by `TtnnDevice::create_tensor()` with `std::runtime_error` after `TensorSpec::validate()`, and non-`NONE` quantization remains rejected by the existing validation path.

4. **Preserve helper semantics and error categories.** `is_supported(DataType)` performs the existing key-span membership test. `carrier_factor(DataType)` returns `2` exactly when `detail::leaf_bits(type) > 32`, otherwise `1`. `native_dtype(DataType)` performs the table lookup and throws `std::invalid_argument` with the existing no-native-dtype meaning when no mapping exists. Preserve the `invalid_ordinal(std::uint32_t, std::size_t)` message/category and use it from the unchanged factory path.

5. **Check before any native object or allocation.** `checked_to_uint32(std::size_t dimension, std::size_t index)` must reject values greater than `std::numeric_limits<std::uint32_t>::max()` with `std::overflow_error`, including the dimension index/value and native limit in the existing diagnostic, and otherwise return the exact narrowed value. `checked_plane_count(const TensorSpec&)` must first check both final matrix dimensions with `checked_to_uint32`, then multiply only leading dimensions in order with checked overflow detection, throwing `std::overflow_error` at the first overflowing dimension. Rank-derived access is valid only after the existing rank-2-through-rank-8 `TensorSpec::validate()` contract; never underflow `rank - 2`, silently wrap a plane count, or narrow before validation. The carrier-column multiplication in `TtnnTensor` must retain its checked overflow and checked uint32 sequence before `ttnn::create_device_tensor()`.

6. **Preserve native storage and ownership.** `TtnnTensor` remains one TTNN-owned native tensor per leading-plane combination, with a native shape `{1, rows, carrier_columns}`, TILE layout, and 32x32 native tile padding. Non-native logical leaves continue using their prescribed UINT32 carrier; widths above 32 use two carrier columns. No `iom::Allocator` storage or dummy raw workspace is introduced. Keep the existing plane vector, registry registration, API mutex locking, staging lifetime, release-versus-quarantine decision, covering finish action, and device-destructor quarantine drain unchanged. `TtnnWorkspace` remains the zero-byte owner; positive raw workspace still throws the established `std::invalid_argument` without touching TTNN allocation.

7. **Preserve public and build contracts.** Do not modify `include/iom/ttnn/device.hpp`, any public signature, capability, error category, validation order, or factory signature. Update only the TTNN `iom_ttnn` source list at `CMakeLists.txt:121-128` to include `src/ttnn/device_types.cpp` and `src/ttnn/device_internal.hpp`; retain `device.cpp`, `copy.cpp`, `copy.hpp`, `registry_state.hpp`, `staging.hpp`, and the public device header. Do not change test source lists. Every extracted definition must have one ODR-valid definition and all current TTNN consumers must compile against the private declaration.

8. **Line budget.** `src/ttnn/device_internal.hpp` and `src/ttnn/device_types.cpp` must each be at most 499 physical lines immediately after this factoring. The final combined change, after the related later queue extraction reduces `src/ttnn/device.cpp`, must leave `src/ttnn/device.cpp` and every touched/new production `.cpp`/`.hpp` under `src/` or `include/` at most 499 physical lines after normal formatting. Use the fewest responsibility-complete files; do not add a permanent line-count test.

## Non-goals

- Do not change the public TTNN header, public signatures, supported capabilities, quantization policy, validation order, error categories, or diagnostics semantics.
- Do not move or redesign `TtnnTensor`, `TtnnWorkspace`, tensor/workspace creation, the device factory, transfer/layout code, native storage, registry/quarantine, or staging ownership.
- Do not split queue/fence code, copy code, binary code, testing seams/API definitions, or conformance/smoke tests in this task. The private device declaration is only a linkage boundary for the separately specified later work.
- Do not add backend switches, global registries, cross-backend abstractions, synchronization changes, allocations, compatibility aliases/shims, extra TTNN capabilities, new tests, or a permanent line-count test.
- Do not replace TTNN native per-plane storage with host storage, an `iom::Allocator`, one flattened tensor, or a different tile/packing scheme.

## Acceptance criteria

- `src/ttnn/device_internal.hpp` contains the exact private `TtnnDevice` declaration and helper declarations needed by TTNN translation units, with no public-header exposure and unchanged fields, ownership, destructor, and accessors.
- `src/ttnn/device_types.cpp` contains exactly one canonical 22-entry table in the specified order, a compile-time key uniqueness check, the prescribed native mappings and `carrier_factor`, the checked uint32/plane-count/rank arithmetic, the ordinal check helper, and the sole `ttnn_supported_data_types()` definition.
- `src/ttnn/device.cpp` retains local `TtnnTensor` and `TtnnWorkspace`, their create methods, the factory, native 32x32 per-plane allocation, one plane per leading combination, API locking, and release/quarantine behavior. No moved definition remains duplicated there, and no queue/testing/binary/copy split is introduced by this task.
- `F8_E8M0`, unsupported data types, and non-`NONE` quantization remain rejected exactly as before; all 22 supported keys retain the exact public span order and `Device::supported_data_types()` points at that canonical span. Overflow and narrowing failures occur before any TTNN native object or allocation, and a later valid tensor can still be created after a rejected request.
- `CMakeLists.txt:114-142` builds and links the new implementation while retaining all existing TTNN sources and headers, with no public factory/header or test-list changes.
- The new files are each at most 499 physical lines now, and the final post-queue-extraction TTNN production files satisfy the same cap. Existing smoke and conformance behavior, ownership/quarantine, context recreation, and coexistence isolation remain unchanged.

## Verification

Proposed gates for the future implementation (not run while writing this mini-spec):

- Through the `remote-development` workflow on configured TTNN hardware, configure the TTNN-enabled build, then build `libiom`/`iom_ttnn` plus `iom_ttnn_smoke_tests`, `iom_ttnn_conformance_tests`, and `iom_backend_coexistence_tests`. The build must compile/link both new files and private-header consumers.
- Through the same remote TTNN environment, run:
  `ctest --test-dir <ttnn-build> --output-on-failure -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$'`.
  Hardware absence or native failure is a test failure, never a skip. Inspect the supported-table, rejection/overflow, tiled-storage, lifetime/quarantine, context-recreation, and coexistence cases named in the implementation references.
- Run a one-time scoped physical-line scanner (not a permanent test) over `src/ttnn/device.cpp`, `src/ttnn/device_internal.hpp`, `src/ttnn/device_types.cpp`, and every other touched/new production suffix under `src/` or `include/`; assert `<=499` after normal formatting, including the final post-queue-extraction state.
- Confirm by source/build inspection that each extracted symbol has one definition, `include/iom/ttnn/device.hpp` is unchanged, and the TTNN target retains its prior sources while adding only the planned device-types implementation/private declaration for this task.

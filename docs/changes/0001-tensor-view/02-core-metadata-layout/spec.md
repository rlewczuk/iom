# Implement tensor metadata and standard tiled layout

**Order:** 02
**Priority:** P0 — all tensor construction, views, transfers, and backends depend on this public contract.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

`TensorShape` and `TensorSpec` provide validated variable-rank host metadata, exact logical and padded byte sizes for every leaf `DataType`, and checked addressing for the common 16x16 tiled layout.

## Scope

- Replace the placeholder shape and empty tile declarations with `BackendKind`, `TensorShape`, and `TensorSpec`.
- Implement checked shape, padding, element-count, byte-count, and standard-layout slot calculations.
- Add deterministic metadata and pure-layout tests; no device allocation is introduced in this task.

## Implementation references

- **Modify:** `include/iom/tensor.hpp` — retain the existing `DataType` and `QuantizationFormat` enumerators, remove `TileSize`, and declare `BackendKind`, `TensorShape`, and `TensorSpec`.
- **Modify:** `src/iom.cpp` — implement metadata validation, checked arithmetic, leaf bit widths, standard padding, and a checked pure standard-layout slot helper used by tests and later backends.
- **Modify:** `test/test_iom.cpp` — replace the dummy test with focused metadata and layout cases.
- **Read:** `include/iom/alloc.hpp` — confirm allocation alignment remains an allocator-construction property; do not change this interface.

## Requirements

- `TensorShape` owns `std::vector<std::size_t>`, requires rank at least two and every dimension nonzero, exposes `rank()`, checked `dimension()`, `dimensions()`, `element_count()`, and value equality.
- `dimension(rank())` throws `std::out_of_range`. Rank, element-count, stride, round-up, and multiplication overflow must throw `std::overflow_error` rather than wrap.
- `TensorSpec` contains shape, leaf `DataType`, and `QuantizationFormat` defaulting to `NONE`; `TILE` is exactly 16.
- `validate()` accepts every declared `DataType` with `QuantizationFormat::NONE`, throws `std::runtime_error` for every other recognized quantization format, and throws `std::invalid_argument` for values outside either enum.
- `standard_padded_shape()` rounds only the final two dimensions to multiples of 16. Leading dimensions remain unchanged.
- `logical_nbytes()` rounds the logical bit count up to bytes. `tiled_storage_nbytes()` uses every padded slot in every leading plane. Both validate before calculating.
- Leaf widths are exactly: 8 bits for `BOOL`; 2 for `I2/U2`; 4 for `I4/U4/F4_E2M1`; 6 for `F6_E2M3/F6_E3M2`; 8 for `I8/U8/F8_E4M3FN/F8_E5M2/F8_E8M0`; 16 for `I16/U16/F16/BF16`; 32 for `I32/U32/F32`; and 64 for `I64/U64/F64`.
- The standard layout orders leading planes row-major, then tile row, tile column, row in tile, and column in tile. The pure helper must validate coordinates and all intermediate arithmetic.
- Rank-two tensors have one plane. For a 32x32 matrix, slots `[0,0]`, `[0,16]`, `[16,0]`, and `[16,16]` are 0, 256, 512, and 768.
- Every 16x16 tile payload for every leaf type is a multiple of 32 bytes.

## Non-goals

- Tensor ownership, device factories, host transfers, or asynchronous operations.
- Grouped or quantized storage for any `QuantizationFormat` other than `NONE`.
- A second public tensor descriptor or a configurable tile size.

## Acceptance criteria

- [ ] Valid shapes at ranks 2, 3, 4, and above 4 round-trip their dimensions; ranks 0 and 1 and a zero in every dimension position are rejected.
- [ ] `[1,17]` pads to `[16,32]` and `[2,8,31,33]` pads to `[2,8,32,48]`.
- [ ] Every leaf type validates, reports the specified width, and produces checked logical and tiled byte counts.
- [ ] Every recognized grouped format and every invalid enum value fails with the specified exception class.
- [ ] Tests cover element-count, padding, stride, logical-byte, and storage-byte overflow.
- [ ] Pure layout tests cover multiple leading planes, rank above four, non-square matrices, padding in both tiled dimensions, and the four fixed 32x32 slot examples.

## Verification

- `c++ -std=c++20 -Iinclude -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN src/iom.cpp test/test_iom.cpp -o /tmp/iom-core-metadata-tests && /tmp/iom-core-metadata-tests --test-case="TensorShape*,TensorSpec*,Standard tiled layout*"`

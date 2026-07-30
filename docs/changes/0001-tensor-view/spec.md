# Minimal materialized tiled-tensor API

## 1. Purpose

Define the tensor metadata, storage layout, ownership, and data-access contract used by IOM kernels and the existing `iom::DeviceOps` interface.

This change replaces the empty `iom::Tensor` definition in `include/iom/iom.hpp`. It reuses `iom::DataType` from that header and `iom::Allocator` from `include/iom/alloc.hpp`.

The design has four goals:

1. every engine tensor is a materialized tiled tensor;
2. tensor metadata is small, fixed-size, and host-resident;
3. CPU, CUDA, and ROCm share one persistent layout that CUDA and ROCm kernels can consume with 16x16 matrix instructions;
4. operations continue to use caller-created input, output, and scratch tensors.

## 2. Scope

This specification defines:

- `iom::TileSize`, `iom::BackendKind`, and `iom::TensorSpec`;
- the common `iom::Tensor` interface used by callers and kernels;
- the standard CPU, CUDA, and ROCm tiled layout;
- allocator ownership for manual-storage tensors;
- synchronous host access and whole-tensor copies;
- lifetime rules required by asynchronous `iom::DeviceOps` calls;
- observable validation and test requirements.

This specification does not define:

- `Backend`, execution-context, status, result, or workspace classes;
- new compute operations or changes to the signatures of existing `DeviceOps` compute methods;
- kernel-specific scratch-memory management; scratch storage is an ordinary, explicitly created `Tensor` when an operation needs it;
- tensor views, slices, strides, reshapes, or non-materialized tensors;
- graphs, autograd, implicit transfers, distributed tensors, or memory planning;
- quantization scales, zero points, or other format-specific metadata;
- the concrete factory that creates a backend-specific `Tensor`;
- TTNN native-storage internals.

## 3. Verified current constraints

The repository currently provides:

- `iom::DataType` in `include/iom/iom.hpp`;
- an empty polymorphic `iom::Tensor` base class;
- asynchronous `iom::DeviceOps` methods that return an `iom::oid` and whose completion can be observed through `DeviceOps::wait(oid)`;
- in-place `DeviceOps` call sites such as `silu(g, g)` and `rmsnorm(t, t, ...)`;
- `iom::Allocator::alloc(size_t)`, `free(void*)`, and `reset()` in `include/iom/alloc.hpp`;
- allocator implementations whose alignment is selected when the allocator is constructed.

Consequences:

- the tensor API uses exceptions for immediate failures, matching the existing allocator API; it does not introduce `Status` or `Result`;
- this specification imposes no blanket input/output aliasing rule on `DeviceOps`;
- allocation alignment is a property of the injected allocator, not an argument to each tensor allocation;
- no separate public and kernel tensor descriptors are required.

## 4. Tensor metadata

All tensor-related definitions are in namespace `iom`.

```cpp
namespace iom {

enum class TileSize : std::uint8_t {
    TILE_16 = 16,
    TILE_32 = 32,
};

enum class BackendKind : std::uint8_t {
    CPU,
    CUDA,
    ROCM,
    TTNN,
};

struct TensorSpec {
    static constexpr std::size_t MAX_RANK = 4;

    std::uint8_t rank = 0;
    std::array<std::size_t, MAX_RANK> dimensions{};
    DataType dtype = DataType::F32;
    TileSize tile_size = TileSize::TILE_16;

    [[nodiscard]] std::size_t dimension(std::size_t index) const;
    [[nodiscard]] std::size_t element_count() const;
    [[nodiscard]] std::size_t logical_nbytes() const;
    [[nodiscard]] std::array<std::size_t, MAX_RANK>
    padded_dimensions() const;
    [[nodiscard]] std::size_t tiled_storage_nbytes() const;

    void validate() const;

    friend bool operator==(const TensorSpec&, const TensorSpec&) = default;
};

}  // namespace iom
```

There is no separate `Shape`, `TileShape`, or `PaddedShape` type.

`backend_kind()` and `backend_device()` identify the tensor's execution target. CPU tensors use device `0`; CUDA, ROCm, 
and TTNN tensors use the nonnegative runtime device ordinal supplied during construction. Both values remain fixed for 
the Tensor lifetime.

### 4.1 Shape rules

- `rank` is 2, 3, or 4.
- `dimensions[0]` through `dimensions[rank - 1]` are nonzero logical dimensions.
- Unused entries from `dimensions[rank]` through `dimensions[3]` are zero. This gives one canonical representation for equality and tests.
- Leading dimensions are row-major tensor planes.
- Only the final two dimensions are tiled.
- A logical vector of length `N` is represented as shape `[1, N]`; rank-one tensors are invalid.
- `dimension(index)` throws `std::out_of_range` when `index >= rank`.
- Metadata calculations detect multiplication, addition, and round-up overflow and throw `std::overflow_error` rather than wrapping.

`padded_dimensions()` returns the same fixed-size array as `dimensions`, with only the final two logical dimensions rounded up to the selected tile size. It does not allocate.

Examples:

```text
rank=2, dimensions=[1, 17, 0, 0], TILE_16
    padded_dimensions=[16, 32, 0, 0]

rank=4, dimensions=[2, 8, 31, 33], TILE_32
    padded_dimensions=[2, 8, 32, 64]
```

### 4.2 Data widths

`TensorSpec` reuses every existing `iom::DataType` enumerator. Metadata byte calculations use these storage widths:

| `DataType` | bits per element |
|---|---:|
| `BOOL`, `U8`, `I8`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0` | 8 |
| `U16`, `I16`, `F16`, `BF16` | 16 |
| `U32`, `I32`, `F32` | 32 |
| `U64`, `I64`, `F64` | 64 |
| `F6_E2M3`, `F6_E3M2` | 6 |
| `F4` | 4 |

`logical_nbytes()` is the byte count of the unpadded logical elements, rounded up to a whole byte. `tiled_storage_nbytes()` 
is the byte count of all padded element slots in the standard tiled layout. Because one 16x16 microtile contains 256 elements, 
every currently defined element width produces a whole number of bytes per microtile.

The numerical scalar encoding represented by each `DataType` is a `DataType` contract. This layout additionally defines 
packed ordering: for `F4` or either `F6` type, element `i` begins at bit offset `i * bits_per_element`; bit offset zero 
is the least-significant bit of byte zero, and each element is stored least-significant bit first into increasing bit offsets. 
Unused tail bits in a logical host buffer are ignored on input and written as zero on output.

## 5. Standard tiled layout

CPU, CUDA, and ROCm tensors use the same engine-defined layout. TTNN may retain its native materialized layout but must 
present the same logical shape and copy behavior.

The physical unit is a 16x16 microtile:

- microtiles are contiguous;
- elements within a microtile are row-major;
- element slots for sub-byte types are consecutive bits;
- padding occupies ordinary element slots but is not part of the logical tensor.

A `TILE_16` logical tile contains one microtile. A `TILE_32` logical tile contains four 16x16 microtiles in this order:

```text
0: rows  0..15, columns  0..15
1: rows  0..15, columns 16..31
2: rows 16..31, columns  0..15
3: rows 16..31, columns 16..31
```

Complete tensor storage order is:

1. flattened leading-dimension plane, in row-major order;
2. logical tile row;
3. logical tile column;
4. 16x16 microtile row within the logical tile;
5. 16x16 microtile column within the logical tile;
6. row within the microtile;
7. column within the microtile.

For rank 2, `plane` is zero. For rank 3, it is the index in dimension 0. For rank 4 coordinates `[d0, d1, row, column]`, it is `d0 * dimensions[1] + d1`.

For logical coordinates `[..., row, column]`, let:

```text
T = 16 or 32
S = T / 16

tile_row = row / T
tile_column = column / T
microtile_row = (row % T) / 16
microtile_column = (column % T) / 16
in_microtile_row = row % 16
in_microtile_column = column % 16
```

The element-slot index is:

```text
plane_tile_count = ceil(rows / T) * ceil(columns / T)
logical_tile_index = plane * plane_tile_count
                   + tile_row * ceil(columns / T)
                   + tile_column
microtile_index = logical_tile_index * (S * S)
                + microtile_row * S
                + microtile_column
element_slot = microtile_index * 256
             + in_microtile_row * 16
             + in_microtile_column
```

A checked pure helper used by `TensorSpec` and tests may implement this calculation. It is not an additional public tensor 
representation.

### 5.1 Matrix-instruction compatibility

Manual-storage tensor base addresses must be at least 32-byte aligned. The existing `Allocator` interface receives no 
alignment argument, so a backend must inject an allocator configured to satisfy this requirement.

Every 16x16 microtile payload starts at a 32-byte boundary for every current `DataType`: 256 elements multiplied by 
4, 6, 8, 16, 32, or 64 bits is a multiple of 32 bytes. A 16x16 half-precision microtile therefore has a 32-byte-aligned 
base and a row stride of 16 elements. A `TILE_32` kernel consumes four such microtiles rather than first converting 
a row-major 32x32 block.

This directly satisfies the pointer-alignment and leading-dimension constraints of 16x16 CUDA WMMA loads and matches the 
16x16 wave-matrix shape exposed for modern ROCm RDNA devices. It does not claim native matrix-unit support for every 
`DataType`; unsupported types require an operation-specific kernel or are rejected by that operation.

Padding values are unspecified. Kernels must not allow padding to affect logical outputs, and host reads never expose padding.

## 6. Tensor interface

`Tensor` is the single common surface for metadata, backend-native kernel access, host transfer, and tensor transfer. 
No other class, or separate kernel descriptor is introduced.

```cpp
namespace iom {

class Tensor {
public:
    virtual ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    [[nodiscard]] const TensorSpec& spec() const noexcept;
    [[nodiscard]] BackendKind backend_kind() const noexcept;
    [[nodiscard]] std::uint32_t backend_device() const noexcept;

    // Backend-native kernel handle. See section 6.1.
    [[nodiscard]] virtual void* native_handle() noexcept = 0;
    [[nodiscard]] virtual const void* native_handle() const noexcept = 0;

    // Synchronous, complete-tensor transfers. See section 6.2.
    virtual void copy_from_host(std::span<const std::byte> source) = 0;
    virtual void copy_to_host(std::span<std::byte> destination) const = 0;
    virtual void copy_from(const Tensor& source) = 0;

protected:
    Tensor(TensorSpec spec,
           BackendKind backend_kind,
           std::uint32_t backend_device);

private:
    TensorSpec spec_;
    BackendKind backend_kind_;
    std::uint32_t backend_device_;
};

}  // namespace iom
```

The protected constructor validates and stores the metadata. Concrete CPU, CUDA, ROCm, and TTNN tensor implementations 
provide storage and transfer behavior. Object creation remains backend-specific and is outside this change; 
the public API does not add a `Backend` or `ExecutionContext` abstraction.

A tensor is always materialized after successful construction. There is no default constructor, empty state, 
`valid()`, `has_storage()`, `reset()`, or replaceable public view.

### 6.1 Native handle

`native_handle()` exposes the storage already owned by the concrete tensor:

- CPU: the allocation address;
- CUDA: the device allocation address returned by the CUDA allocation API;
- ROCm: the device allocation address;
- TTNN: a pointer to the backend-owned native `ttnn::Tensor` object.

Only code selected for `backend_kind()` may interpret this type-erased handle. It must not be retained beyond the tensor lifetime. 
The handle is stable for the tensor lifetime on CPU, CUDA, and ROCm. TTNN may update the native value behind its stable handle, 
but the public `Tensor` object, `TensorSpec`, backend kind, and device remain unchanged.

This intentionally uses the same `Tensor` object at the `DeviceOps` and kernel boundaries. It does not duplicate metadata 
in a persistent device-side descriptor.

### 6.2 Host and tensor copies

All three copy methods are synchronous: when a method returns, the destination contains the copied logical values and is 
ready for the caller to use.

`copy_from_host` and `copy_to_host` use a contiguous logical row-major host buffer:

- the host encoding is exactly the tensor's `DataType`;
- the span size must equal `spec().logical_nbytes()`;
- the methods translate between row-major logical order and the tensor's native tiled order;
- padding is not read from or written to the host buffer;
- no dtype conversion occurs.

`destination.copy_from(source)`:

- copies complete logical tensors only;
- requires equal logical ranks, dimensions, and dtypes;
- permits different tile sizes and retiles as part of the copy;
- permits different backend kinds or devices only when the destination implementation has a direct transfer path;
- never silently stages through a host buffer;
- performs no numeric conversion;
- treats `destination.copy_from(destination)` as a no-op.

An unsupported direct path throws `std::runtime_error`. Region copies and asynchronous copies are outside this change.

Before calling a synchronous host or tensor copy, the caller must wait for every outstanding `DeviceOps` operation that 
writes a participating tensor. Before submitting a `DeviceOps` operation that consumes a copy destination, the synchronous copy must have returned.

## 7. Storage ownership and allocation

CPU, CUDA, and ROCm concrete tensors receive an existing `iom::Allocator&` during backend-specific construction.

- The allocator must outlive every tensor using it.
- Construction calls `allocator.alloc(spec.tiled_storage_nbytes())` exactly once.
- A successful construction owns the returned non-null allocation.
- Destruction calls `allocator.free(address)` exactly once.
- If construction fails after allocation, it frees the allocation before propagating the exception.
- Manual-storage tensor transfers do not call `alloc` or `free` on the injected `iom::Allocator`.
- The allocator is assumed to return a pointer aligned to at least 32 bytes.

TTNN owns native materialized storage through its runtime and does not force that storage through `iom::Allocator`.

`Allocator::reset()` invalidates allocations owned by the allocator. Calling it while any corresponding tensor is alive 
is a caller error. Tensor destruction after such a reset is also invalid unless the concrete allocator explicitly documents 
otherwise; the normal lifetime order is tensors first, allocator second.

## 8. Asynchronous `DeviceOps` integration

Existing compute methods on `iom::DeviceOps` remain asynchronous and continue to return `oid`. No status return, 
validation wrapper, backend dispatch interface, or execution context is added.

The existing `DeviceOps::copy(const Tensor&, Tensor&)` method is replaced by `Tensor::copy_from`, because data movement 
belongs to the tensor surface. Existing call sites change from:

```cpp
dev.copy(source, destination);
```

to:

```cpp
destination.copy_from(source);
```

All other `DeviceOps` method signatures remain unchanged.

For every submitted compute operation:

- every referenced input, output, and scratch `Tensor` object must remain alive and at the same address until `DeviceOps::wait(returned_oid)` completes;
- their underlying native storage must also remain alive and unchanged during that interval;
- callers must wait before reading an output through `copy_to_host`, overwriting it through a copy, or destroying it;
- whether a particular operation supports input/output aliasing remains part of that existing operation's contract. This tensor specification does not reject the in-place calls already present in the repository.

Deleting Tensor moves makes the object-address rule explicit and prevents accidental relocation while an asynchronous operation may hold references.

## 9. Error behavior

The API uses exceptions, consistently with `iom::Allocator`:

- `TensorSpec::validate()` and the Tensor constructor throw `std::invalid_argument` for invalid rank, zero logical dimensions, nonzero unused dimensions, invalid enum values, or unsupported metadata combinations;
- checked size and padding calculations throw `std::overflow_error`;
- `dimension()` throws `std::out_of_range` for an invalid logical index;
- copy methods throw `std::invalid_argument` for a wrong host byte count or incompatible tensor metadata;
- copy methods throw `std::runtime_error` for an unsupported transfer path or backend failure;
- allocator exceptions propagate after ownership cleanup.

Copy methods validate host sizes, tensor metadata, and direct-path support before writing. Those validation and capability failures leave the destination unchanged. A backend failure after a transfer has begun may leave destination values unspecified; it must not change the Tensor metadata or storage ownership.

## 10. Implementation touchpoints

The implementation of this specification is limited to these current surfaces:

- `include/iom/iom.hpp`: add `TileSize`, `BackendKind`, `TensorSpec`, and the `Tensor` contract; remove `DeviceOps::copy` while leaving other `DeviceOps` methods unchanged;
- `src/iom.cpp`: implement checked metadata helpers and common Tensor accessors;
- backend-specific concrete Tensor implementations: own native storage and implement the native-handle and copy methods;
- `src/llama.cpp`: replace the current `dev.copy(x, r)` call with `r.copy_from(x)` and use the final host-copy method for tensor initialization;
- `test/test_iom.cpp`: replace the dummy test with the behavior tests below.

`include/iom/alloc.hpp` is reused without interface changes.

## 11. Automated acceptance criteria

### 11.1 Metadata and validation tests

Unit tests must verify:

- ranks 2, 3, and 4 with nonzero dimensions validate;
- rank 0, rank 1, rank greater than 4, zero logical dimensions, and nonzero unused dimensions are rejected;
- invalid `TileSize`, `DataType`, and `BackendKind` values are rejected at their owning API boundary;
- `[1, N]` is the valid vector representation;
- `dimension(rank)` is rejected;
- element-count, logical-byte-count, padding, and storage-byte overflow are detected;
- every existing `DataType` maps to the width in section 4.2.

### 11.2 Padding and layout tests

Pure deterministic tests must cover both tile sizes, all supported ranks, multiple leading planes, non-square shapes, and padding on each final dimension.

At minimum, tests verify:

```text
[1, 17] with TILE_16 pads to [16, 32]
[2, 8, 31, 33] with TILE_32 pads to [2, 8, 32, 64]
```

For one 32x32 tile, slot offsets must demonstrate the four-microtile order:

```text
[0, 0]   ->   0
[0, 16]  -> 256
[16, 0]  -> 512
[16, 16] -> 768
```

For every current `DataType`, tests verify that the byte size of a 16x16 microtile is a multiple of 32. An allocator-backed integration test verifies the actual native base address is 32-byte aligned.

### 11.3 Ownership tests

A recording `iom::Allocator` must verify:

- successful manual tensor construction performs one allocation of exactly `tiled_storage_nbytes()`;
- destruction frees the same pointer exactly once;
- a construction failure after allocation frees the pointer;
- host and manual-storage tensor copies do not invoke `alloc` or `free`;
- copy and move construction and assignment are disabled at compile time.

### 11.4 Copy tests

A CPU test tensor provides deterministic end-to-end tests:

1. copy row-major host bytes into a padded tensor;
2. copy them back and compare the complete logical host buffer;
3. run the round trip for `TILE_16` and `TILE_32` and for ranks 2 through 4;
4. verify padding never appears in the host buffer;
5. copy between equal specs with different tile sizes and compare logical values;
6. verify self-copy is a no-op;
7. verify mismatched rank, dimensions, dtype, and host byte count fail without changing the destination;
8. use a fake unsupported backend pair to verify that capability checking occurs before any destination write and that no hidden host staging occurs.

At least one packed type and one 16-bit floating type must be included so tests distinguish element-slot ordering from byte-only assumptions.

### 11.5 Asynchronous lifetime test

A fake `DeviceOps` implementation must deliberately defer a write until `wait(oid)`. The integration test verifies that:

- submission returns a token without performing the deferred write;
- `wait(token)` completes the write;
- the same stable Tensor objects are seen at submission and completion.

This test exercises the Tensor lifetime contract without requiring CUDA, ROCm, or TTNN hardware.

## 12. Completion criteria

The change is complete when:

- the public types and behavior above compile in namespace `iom`;
- no `Shape`, `TileShape`, `PaddedShape`, tensor-view, allocator replacement, `Backend` class, execution-context, workspace, status, result, or operation-dispatch abstraction is added;
- the existing `DeviceOps` compute interface remains asynchronous and otherwise unchanged;
- CPU layout and copy tests prove logical row-major round trips across both tile sizes;
- allocator tests prove fixed storage ownership and 32-byte alignment;
- the test suite deterministically covers every validation and failure guarantee listed above.

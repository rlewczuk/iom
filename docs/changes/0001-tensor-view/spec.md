# Materialized tiled tensors and leading-dimension views

## 1. Purpose and scope

Define tensor metadata, tiled storage, ownership, safe leading-dimension views, a minimal device construction boundary, and the `iom::DeviceOps` operand contract for CPU, CUDA, ROCm, SYCL, and TTNN.

This change replaces the placeholder declarations in `include/iom/tensor.hpp`, updates `include/iom/iom.hpp`, and adds independently buildable backend implementations. It reuses `iom::Allocator` from `include/iom/alloc.hpp`.

Goals:

1. every engine tensor is materialized and created explicitly by the caller through a `Device`;
2. tensor metadata is host-resident, with a variable-length dimensions array;
3. CPU, CUDA, ROCm, and SYCL use one persistent 16x16 tiled layout; TTNN retains its native tiled storage;
4. operations use caller-created input, output, and scratch tensors and never allocate operands;
5. the last two dimensions are tiled while leading dimensions support safe slicing, selection, permutation, and reshape views;
6. every tensor-operand method on `DeviceOps`, including `copy`, consumes `TensorView`;
7. CPU is the reference implementation, and every enabled accelerator backend is conformance-tested against it;
8. any combination of enabled backends can coexist in one build and process.

Non-goals:

- status, result, workspace, graph, autograd, implicit-transfer, distributed-tensor, memory-planning, or operation-dispatch abstractions;
- new compute operations or changes to their mathematical contracts beyond accepting and honoring views;
- transforms of the last two dimensions, raw `as_strided`, negative or broadcast strides, or non-materialized tensors;
- implicit synchronization or host staging for device-to-device copies;
- support for grouped or quantized storage (`QuantizationFormat` values other than `NONE`) until each format has an exact scale, block, auxiliary-metadata, packing, and host-encoding contract;
- exposing TTNN native-storage internals.

This is a source-breaking clean cutover in a pre-release API. Every in-repository caller is migrated in the same change; no compatibility aliases or deprecated tensor overloads remain. Tensor contents are not persisted by this API, so no data migration or rollback format is required.

## 2. Verified current constraints

The repository currently provides:

- `iom::DataType` and `iom::QuantizationFormat` in `include/iom/tensor.hpp`; `DataPrecision` and `DataFormat` do not exist;
- leaf `DataType` enumerators for ordinary integers and floats, including `F4_E2M1` and `F8_E4M3FN`, and a taxonomy of grouped `QuantizationFormat` values;
- placeholder `iom::TensorShape` and `iom::Tensor` definitions and an empty `TileSize` enum in `include/iom/tensor.hpp`;
- asynchronous `iom::DeviceOps` methods in `include/iom/iom.hpp` that return an `iom::oid` (currently a `std::uint64_t`) and whose completion can be observed through `DeviceOps::wait(oid)`;
- `DeviceOps::copy(const Tensor&, Tensor&)`; its only call site is `dev.copy(x, r)` in `src/llama.cpp`;
- `src/llama.cpp` calls `Tensor::set_float`, which is not declared by the current `Tensor`;
- in-place `DeviceOps` call sites such as `silu(g, g)` and `rmsnorm(t, t, ...)`;
- `iom::Allocator::alloc(size_t)`, `free(void*)`, and `reset()` in `include/iom/alloc.hpp`, with alignment selected when the allocator is constructed (`LinearAllocator` defaults to 32);
- no concrete tensor backend, common device factory, SYCL build option, or TTNN build option;
- mutually exclusive CUDA and ROCm CMake options, which do not satisfy the required multi-backend process model;
- safetensors already exposes `DataType`, but `src/safetensors.cpp` still refers to the nonexistent `DataType::F4` and `DataType::F8_E4M3` enumerators.

The current tree does not build: `src/llama.cpp` fails on `set_float`, while `src/safetensors.cpp` fails on the two stale enumerator names above.

Consequences:

- the tensor API uses exceptions for immediate failures, matching the allocator API; it does not introduce `Status` or `Result`;
- `TensorSpec` uses the existing `DataType` and `QuantizationFormat` split rather than recreating a precision/encoding pair;
- this change supports only `QuantizationFormat::NONE`; retaining the other enum values does not define or partially implement their storage contracts;
- this specification imposes no blanket input/output aliasing rule on `DeviceOps`;
- allocation alignment is a property of the injected allocator, not an argument to each tensor allocation;
- no separate public and kernel tensor descriptors are required.

## 3. Tensor metadata

All definitions below are in namespace `iom`. The existing `DataType` and `QuantizationFormat` enumerator names remain authoritative:

```cpp
enum class BackendKind : std::uint8_t {
    CPU,
    CUDA,
    ROCM,
    SYCL,
    TTNN,
};

enum class DataType {
    BOOL,
    I2, U2,
    I4, U4,
    I8, U8,
    I16, U16,
    I32, U32,
    I64, U64,
    F4_E2M1,
    F6_E2M3,
    F6_E3M2,
    F8_E4M3FN,
    F8_E5M2,
    F8_E8M0,
    F16,
    BF16,
    F32,
    F64,
};

enum class QuantizationFormat {
    NONE,
    INT8_SYMMETRIC,
    INT8_ASYMMETRIC,
    INT4_SYMMETRIC,
    INT4_ASYMMETRIC,
    OCP_MXFP4,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    NVIDIA_NVFP4,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q8_0,
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    TT_BFP2,
    TT_BFP2A,
    TT_BFP4,
    TT_BFP4A,
    TT_BFP8,
    TT_BFP8A,
};

class TensorShape {
public:
    explicit TensorShape(std::vector<std::size_t> dimensions);

    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::size_t dimension(std::size_t index) const;
    [[nodiscard]] std::span<const std::size_t> dimensions() const noexcept;
    [[nodiscard]] std::size_t element_count() const;

    friend bool operator==(const TensorShape&, const TensorShape&) = default;

private:
    std::vector<std::size_t> dimensions_;
};

struct TensorSpec {
    static constexpr std::size_t TILE = 16;

    TensorShape shape;
    DataType data_type;
    QuantizationFormat quantization = QuantizationFormat::NONE;

    [[nodiscard]] TensorShape standard_padded_shape() const;
    [[nodiscard]] std::size_t logical_nbytes() const;
    [[nodiscard]] std::size_t tiled_storage_nbytes() const;
    void validate() const;

    friend bool operator==(const TensorSpec&, const TensorSpec&) = default;
};
```

`DataType` identifies the exact leaf scalar encoding. `QuantizationFormat` identifies grouping, scale, auxiliary-metadata, and packing rules layered over leaf values. The two concepts are not interchangeable: `F16` and `BF16` are distinct leaf types, while `OCP_MXFP4` and `GGML_Q4_0` are grouped storage formats.

Only `QuantizationFormat::NONE` is supported by this change. `TensorSpec::validate()` accepts every declared `DataType` with `NONE`. It throws `std::runtime_error` for every other recognized `QuantizationFormat` before allocation because this change does not define those formats. Invalid values cast into either enum throw `std::invalid_argument`. `logical_nbytes()` and `tiled_storage_nbytes()` validate the specification before calculating a size.

Shape rules:

- rank is `dimensions().size()` and must be at least two; there is no maximum rank or separate rank field;
- every dimension is nonzero;
- leading dimensions index physical planes in row-major order, while the final two dimensions form the tiled matrix;
- only the final two dimensions are tiled and padded; a logical vector of length `N` is represented as `[1, N]`;
- `dimension(index)` throws `std::out_of_range` when `index >= rank()`;
- metadata calculations detect multiplication and round-up overflow and throw `std::overflow_error` rather than wrapping.

`TensorSpec::standard_padded_shape()` returns a new shape with only the final two dimensions rounded up to multiples of `TensorSpec::TILE`. The name distinguishes this engine layout calculation from TTNN's native padding:

```text
[1, 17]          -> [16, 32]
[2, 8, 31, 33]  -> [2, 8, 32, 48]
```

Leaf storage widths are:

| `DataType` | bits per logical element |
|---|---:|
| `BOOL` | 8 |
| `I2`, `U2` | 2 |
| `I4`, `U4`, `F4_E2M1` | 4 |
| `F6_E2M3`, `F6_E3M2` | 6 |
| `I8`, `U8`, `F8_E4M3FN`, `F8_E5M2`, `F8_E8M0` | 8 |
| `I16`, `U16`, `F16`, `BF16` | 16 |
| `I32`, `U32`, `F32` | 32 |
| `I64`, `U64`, `F64` | 64 |

`logical_nbytes()` is the unpadded logical bit count rounded up to whole bytes. `tiled_storage_nbytes()` uses all padded element slots in the standard layout of section 4. TTNN native storage may have a different byte count.

`BOOL` uses one byte containing zero or one. Signed integers use two's-complement and unsigned integers use ordinary unsigned binary. `F16`, `F32`, and `F64` use IEEE binary formats; `BF16` uses bfloat16; the remaining floating names identify their exact exponent/mantissa encodings, including the finite-only `F8_E4M3FN` variant. Multi-byte scalar fields are little-endian. For widths below eight bits and both six-bit floating types, logical element `i` begins at bit offset `i * bits_per_element`; bit offset zero is the least-significant bit of byte zero, and each element is written least-significant bits first into increasing bit offsets. Unused tail bits in a logical host buffer are ignored on input and written as zero on output.

`SafeTensorView` remains an unquantized leaf-data view and continues to expose one `DataType`. The accepted safetensors dtype mapping is:

| safetensors dtype | `DataType` |
|---|---|
| `BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64` | same-named enumerator |
| `F16`, `BF16`, `F32`, `F64` | same-named enumerator |
| `F8_E5M2`, `F8_E8M0`, `F6_E2M3`, `F6_E3M2` | same-named enumerator |
| `F8_E4M3` | `F8_E4M3FN` |
| `F4` | `F4_E2M1` |

Every other dtype string is rejected. In particular, the presence of `I2`, `U2`, `I4`, and `U4` in `DataType` does not add nonstandard safetensors dtype strings.

## 4. Standard tiled layout

CPU, CUDA, ROCm, and SYCL tensors with `QuantizationFormat::NONE` use one engine-defined layout. TTNN tensors retain their native materialized layout but expose the same logical shape and leading-plane addressing contract.

The physical unit in the standard layout is one 16x16 tile (`TensorSpec::TILE`):

- tiles are contiguous;
- elements within a tile are row-major;
- sub-byte element slots use the bit order from section 3;
- padding occupies ordinary element slots but is not part of the logical tensor; padding values are unspecified, kernels must not let padding affect logical outputs, and host reads never expose padding.

Storage order is:

1. leading-dimension plane, row-major;
2. tile row, then tile column within the plane;
3. row, then column within the tile.

For leading coordinates `d[0] ... d[n-1]`, where `n = rank - 2`, the dense owner plane index is the ordinary row-major index:

```text
plane = d[0]
for i in 1 .. n-1:
    plane = plane * dimensions[i] + d[i]
```

Rank-two tensors have one plane. For logical coordinates `[..., row, column]`:

```text
tile_row    = row / 16
tile_column = column / 16

tile_index   = plane * ceil(rows / 16) * ceil(columns / 16)
             + tile_row * ceil(columns / 16)
             + tile_column
element_slot = tile_index * 256 + (row % 16) * 16 + (column % 16)
```

A checked pure helper may implement this calculation for the standard layout and tests; it is not an additional public tensor representation.

In one 32x32 logical matrix:

```text
[0, 0]   -> slot 0
[0, 16]  -> slot 256
[16, 0]  -> slot 512
[16, 16] -> slot 768
```

### 4.1 Alignment

- Standard-layout tensor base addresses are at least 32-byte aligned. The existing `Allocator` interface receives no alignment argument, so a backend injects an allocator configured to satisfy this requirement.
- Every 16x16 tile payload is a multiple of 32 bytes for every `DataType` in section 3.
- Views address whole planes, so view offsets preserve tile alignment.
- This layout does not imply that every backend matrix instruction supports every `DataType`. An operation rejects a type it cannot compute before submitting work.

## 5. TensorView and Tensor interface

`Tensor` is the materialized owner. `TensorView` is the non-owning operand and transfer surface. A tensor exposes one stable full-storage view through `Tensor::view()`; `Tensor` does not inherit from `TensorView`.

Composition avoids corrupting an owner's full-view metadata through assignment to a public base-class reference. It also keeps storage ownership out of copied views.

### 5.1 TensorView

```cpp
namespace iom {

class Device;
class Tensor;

class TensorView {
public:
    TensorView(const TensorView&) = default;
    TensorView(TensorView&&) = default;
    TensorView& operator=(const TensorView&) = delete;
    TensorView& operator=(TensorView&&) = delete;

    [[nodiscard]] const TensorSpec& spec() const noexcept;
    [[nodiscard]] const Device& device() const noexcept;
    [[nodiscard]] BackendKind backend_kind() const noexcept;
    [[nodiscard]] std::uint32_t backend_device() const noexcept;

    [[nodiscard]] void* native_handle() noexcept;
    [[nodiscard]] const void* native_handle() const noexcept;

    // Offset and strides count complete logical planes, not bytes or elements.
    [[nodiscard]] std::size_t plane_offset() const noexcept;
    [[nodiscard]] std::span<const std::size_t> plane_strides() const noexcept;

    [[nodiscard]] TensorView slice(std::size_t dim, std::size_t first,
                                   std::size_t count, std::size_t step = 1) const;
    [[nodiscard]] TensorView select(std::size_t dim, std::size_t index) const;
    [[nodiscard]] TensorView permute(
        std::span<const std::size_t> leading_order) const;
    [[nodiscard]] TensorView reshape_leading(
        std::span<const std::size_t> leading_dimensions) const;

    // Synchronous host transfers. See section 5.4.
    void copy_from_host(std::span<const std::byte> source);
    void copy_to_host(std::span<std::byte> destination) const;

private:
    friend class Tensor;
    TensorView(Tensor& owner, TensorSpec spec, std::size_t plane_offset,
               std::vector<std::size_t> plane_strides);

    Tensor* owner_;
    TensorSpec spec_;
    std::size_t plane_offset_ = 0;
    std::vector<std::size_t> plane_strides_;
};

}  // namespace iom
```

The view map for leading logical coordinates is:

```text
owner_plane = plane_offset
for i in 0 .. leading_rank-1:
    owner_plane += coordinate[i] * plane_strides[i]
```

The full tensor view has offset zero and dense row-major plane strides. Rank-two views have no leading strides.

Safe transforms are:

- `slice(dim, first, count, step)` keeps the rank, sets dimension `dim` to `count`, advances the offset by `first * stride[dim]`, and multiplies that stride by `step`. `step` and `count` must be nonzero, and `first + (count - 1) * step` must be in bounds.
- `select(dim, index)` fixes one leading coordinate, advances the offset by `index * stride[dim]`, and erases that dimension and its matching stride. It does not recompute other strides.
- `permute(leading_order)` requires an exact permutation of all leading-dimension indices and reorders the leading dimensions and their matching strides. For a rank-two view, the required empty permutation is a no-op. The final two dimensions remain in place.
- `reshape_leading(leading_dimensions)` may split, merge, insert, or remove size-one leading dimensions. The product of the new leading dimensions must equal the current leading-plane count, with an empty span having product one. In reverse leading-dimension order, each dimension larger than one must have the next dense stride, beginning at one; size-one dimensions do not constrain their stored stride. Otherwise the method rejects the reshape. The final two dimensions remain unchanged.

For every transform:

- `dim` indexes only a leading dimension; the final two tiled dimensions cannot be selected, sliced, permuted, or reshaped;
- offset, stride, product, and last-index calculations are checked for overflow;
- the result remains in bounds and non-overlapping;
- no raw `as_strided`, negative stride, broadcasting stride, or last-two-dimension transform is exposed;
- only metadata vectors may allocate; tensor storage is neither allocated nor moved;
- transforms work for every backend, including TTNN, because their public offset and strides count logical planes rather than backend bytes.

`TensorView` is non-owning. The owner `Tensor` and its native storage must outlive every derived view.

### 5.2 Tensor

```cpp
namespace iom {

class Tensor {
public:
    virtual ~Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    [[nodiscard]] TensorView& view() noexcept;
    [[nodiscard]] const TensorView& view() const noexcept;

protected:
    Tensor(TensorSpec spec, Device& device);

    [[nodiscard]] virtual void* storage_handle() noexcept = 0;
    virtual void region_from_host(const TensorView& destination,
                                  std::span<const std::byte> source) = 0;
    virtual void region_to_host(const TensorView& source,
                                std::span<std::byte> destination) const = 0;

private:
    friend class TensorView;
    Device* device_;
    TensorView full_view_;
};

}  // namespace iom
```

The protected constructor validates the metadata and creates `full_view_` with zero offset and dense row-major plane strides. Concrete backends provide storage and host-region transfer hooks; their `Device` creates the concrete tensor.

A tensor is materialized after successful construction. It has no default constructor, empty state, `valid()`, `has_storage()`, or `reset()`. Deleting copy and move keeps ownership unambiguous and the full-view and owner addresses stable for asynchronous operations. The creating `Device` must outlive the tensor.

### 5.3 Native handle

`native_handle()` always exposes the owner's storage handle:

- CPU: the allocation address;
- CUDA, ROCm, and SYCL: the device allocation address;
- TTNN: a pointer to the backend-owned native tensor object.

Only code selected for the owner's `backend_kind()` may interpret the type-erased handle. Standard-layout kernels combine it with the view's `spec()`, `plane_offset()`, and `plane_strides()`. TTNN maps the same logical-plane descriptor onto its native layout. The handle must not be retained beyond the owner lifetime.

### 5.4 Host transfers

`copy_from_host` and `copy_to_host` are synchronous. They move the view's logical region through a contiguous row-major host buffer:

- the host encoding is exactly the view's `DataType` with `QuantizationFormat::NONE`, using the sub-byte packing in section 3;
- the span size equals `spec().logical_nbytes()`;
- the methods translate between logical row-major order and backend storage, honoring the view's offset and strides;
- padding is neither read from nor written to the host buffer;
- no numeric or encoding conversion occurs;
- `copy_from_host` rejects a `BOOL` source containing a byte other than zero or one before writing any destination value.

Before a host read, the caller waits for outstanding `DeviceOps` writes to participating storage. Before a host write, the caller waits for every outstanding operation that reads or writes participating storage. A host write returns before a later `DeviceOps` submission may consume its destination.

## 6. Device construction and layering

`Device` owns one backend runtime context and creates tensors and operation queues for that context:

```cpp
namespace iom {

class Device {
public:
    virtual ~Device() = default;

    [[nodiscard]] virtual BackendKind backend_kind() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t backend_device() const noexcept = 0;

    [[nodiscard]] virtual std::unique_ptr<Tensor> create_tensor(
        const TensorSpec& spec) = 0;
    [[nodiscard]] virtual std::unique_ptr<DeviceOps> create_ops() = 0;
};

}  // namespace iom
```

`create_tensor` is an explicit caller action; compute and copy methods never allocate operands. It validates the specification and rejects unsupported types before allocating. `create_ops` returns one in-order asynchronous queue bound to the device. A caller may create multiple queues from one device and use tensors from that device on any of them. The Device must outlive every tensor and queue it created, and every caller-supplied allocator must outlive its Device.

Calls on one `DeviceOps` queue are serialized by the caller; concurrent submission or waiting on the same queue is outside this contract. Different queues may be driven concurrently. A cross-queue dependency is established by waiting for the producer's `oid` on its originating queue before submitting the consumer.

Layering and customization requirements:

- the common `Device` interface and tensor/view types contain no CUDA, HIP, SYCL, or TTNN header types;
- each backend is a separate library with a separate factory returning `std::unique_ptr<Device>`: `make_cpu_device(Allocator&)`, `make_cuda_device(std::uint32_t device_ordinal, Allocator&)`, `make_rocm_device(std::uint32_t device_ordinal, Allocator&)`, `make_sycl_device(std::uint32_t device_ordinal, Allocator&)`, and `make_ttnn_device(std::uint32_t device_ordinal)`;
- `backend_device()` returns the configured backend-local ordinal; the CPU device returns zero;
- an accelerator factory creates and owns the runtime context for its ordinal; borrowed native contexts and queue-tuning option abstractions are outside this change;
- every `create_ops()` call creates an in-order queue with backend defaults;
- a standard-layout backend always uses the caller-supplied allocator; TTNN always uses its native storage allocator;
- factories are independent functions, not a global registry, singleton, or `BackendKind` switch, and `Device` remains subclassable by user code;
- factories reject an unavailable or invalid ordinal before creating tensor or queue resources;
- CPU, CUDA, ROCm, and SYCL tensor creation supports every declared `DataType` with `QuantizationFormat::NONE`; TTNN supports at least `BF16` and may explicitly support additional leaf types in its backend test table;
- any number and combination of backend devices may coexist in one process.

## 7. Storage ownership and allocation

The CPU, CUDA, ROCm, and SYCL device factories receive an `iom::Allocator` whose returned pointers belong to that backend's address space and are accepted by its runtime copy and kernel APIs. For every standard-layout tensor:

- `Device::create_tensor` calls `allocator.alloc(spec.tiled_storage_nbytes())` exactly once;
- a null return throws `std::bad_alloc` without calling `free`;
- a non-null address that is not 32-byte aligned is freed once and rejected with `std::runtime_error`;
- successful construction owns the returned allocation;
- destruction calls `allocator.free(address)` exactly once: at tensor destruction, or deferred to device destruction for storage quarantined because a queued operation referencing it failed or was invalidated;
- failure after allocation frees the allocation before propagating the exception;
- the injected allocator's `free` must not throw; `Allocator` lacks a `noexcept` declaration, but throwing from tensor destruction or construction cleanup is a caller contract violation;
- host transfers, device copies, compute submission, and view transforms never call `alloc` or `free` on the configured tensor-storage allocator.

Quarantined storage is released by the device destructor's quarantine drain, after all tensors and queues of that device are destroyed; this deferral is why the allocator must outlive the device.

TTNN owns native materialized storage through its runtime and does not route that storage through `iom::Allocator`.

`Allocator::reset()` invalidates its allocations. Calling it while a corresponding tensor is alive is a caller error. Tensor destruction after such a reset is invalid unless that allocator explicitly documents otherwise.

## 8. Asynchronous `DeviceOps` integration

Every tensor-operand method takes views. Existing compute methods remain asynchronous and retain their mathematical contracts:

```cpp
virtual oid copy(const TensorView& source, TensorView& destination) = 0;
virtual oid add(const TensorView& a, const TensorView& b, TensorView& c) = 0;
virtual oid mul(const TensorView& a, const TensorView& b, TensorView& c) = 0;
virtual oid silu(const TensorView& x, TensorView& y) = 0;
virtual oid linear(const TensorView& x, const TensorView& w, TensorView& y) = 0;
virtual oid rmsnorm(const TensorView& x, TensorView& y, const TensorView& w,
                    float eps, std::size_t dim) = 0;
virtual oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
                 std::size_t n_heads, std::size_t n_kv_heads,
                 std::size_t head_dim, TensorView& attn_out) = 0;
```

`oid` remains `std::uint64_t`. Its most-significant eight bits are a queue ID and its least-significant 56 bits are that queue's submission sequence:

```text
oid = (queue_id << 56) | sequence
```

Queue ID zero and sequence zero are invalid. Every live `DeviceOps` owns one process-unique queue ID in `[1, 255]`; creation fails with `std::runtime_error` when all 255 IDs are in use. Queue IDs may be reused after destruction, and an `oid` ceases to be valid when its originating queue is destroyed. A queue's sequence begins at one and increases for every successful submission, including identical-window no-ops. Submission after sequence `2^56 - 1` throws `std::overflow_error` before queuing work.

`wait(oid)` is idempotent while the originating queue remains alive. After successful completion it returns immediately on every later call; after asynchronous failure it rethrows the stored failure on every later call. Because the queue is in order, waiting for sequence `N` also completes every earlier sequence, and later waits for those earlier IDs observe their individual stored result.

The common `DeviceOps` base implementation, not each backend, leases and releases queue IDs from one process-wide pool. This queue-ID pool is the only shared process state introduced by the queue contract; it does not select or activate a backend. Per-queue submission sequences require no global synchronization.

Idempotent successful waits require only a completed-sequence watermark. Implementations retain an exception for each failed sequence until queue destruction so repeated waits rethrow the same failure.

`DeviceOps::copy` is part of this change. It:

- requires source and destination to belong to the same `Device` as the queue;
- requires identical logical shape, data type, and quantization format;
- copies logical values in view coordinate order, honoring arbitrary offsets and strides produced by section 5;
- copies no padding and performs no numeric or format conversion;
- returns an `oid` scoped to the submitting `DeviceOps`; completion is observed through that queue's `wait(oid)`;
- treats identical source and destination windows as a no-op submission with a waitable `oid`;
- gives unspecified destination values for overlapping but non-identical source and destination windows.

Two views are an identical window only when they have the same owner, logical shape, plane offset, and plane strides. No overlap guarantee is inferred merely from equal native handles.

Cross-device and cross-backend copies are rejected. Callers may perform explicit host staging with section 5.4; `DeviceOps::copy` never stages implicitly.

All current and future compute implementations must honor every safe view produced by section 5. An operation may reject unsupported shapes, data types, quantization formats, or aliasing, but it must not reject an otherwise valid operand merely because its leading dimensions are sliced, stepped, selected, or permuted. Shared backend indexing maps a leading coordinate through `plane_offset()` and `plane_strides()` before entering the operation's per-plane kernel.

Submission rules:

- validation completes before work is queued;
- a submission that fails validation or fails synchronously before work is queued does not consume a sequence number;
- every participating `TensorView`, its owner tensor, and the native allocation remain alive at stable addresses until `wait(returned_oid)` completes; the queue is not required to copy variable-length view metadata;
- callers wait before host access, transfer overwrite, or owner destruction;
- submissions to one `DeviceOps` queue execute in call order, permitting a later operation to consume an earlier output without an intervening host wait;
- the caller observes completion of every successful submission before destroying its `DeviceOps`; destruction does not implicitly wait or cancel work;
- queues created from the same `Device` are independent; the caller waits on the producer queue before submitting a cross-queue consumer;
- an `oid` is valid only while its originating queue is alive; `wait` compares the high queue-ID bits before the sequence and throws `std::invalid_argument` for zero, unknown, or live foreign-queue tokens.

This change migrates compute signatures and the common view-indexing contract but does not implement or conformance-test the mathematical compute kernels. CPU compute behavior and accelerator compute conformance are separate operation changes. The backend queues delivered here implement asynchronous `copy`; until a separate operation change adds a compute method, that method throws `std::runtime_error` during capability validation without submitting work.

## 9. Error behavior

The API uses exceptions for immediate failures:

- `TensorShape` construction throws `std::invalid_argument` for rank below two or a zero dimension;
- `TensorSpec::validate()` throws `std::invalid_argument` for invalid enum values and `std::runtime_error` for a recognized but unsupported quantization format; tensor creation throws `std::runtime_error` when that backend cannot store a validated `DataType`;
- `dimension()` throws `std::out_of_range` for an invalid index;
- checked shape, padding, byte-size, offset, stride, and last-index calculations throw `std::overflow_error`;
- `slice` and `select` throw `std::out_of_range` for a non-leading dimension or an out-of-range coordinate and `std::invalid_argument` for a zero count or step;
- `permute` throws `std::invalid_argument` unless its indices are an exact permutation of the leading dimensions;
- `reshape_leading` throws `std::invalid_argument` for a zero new dimension, plane-count mismatch, or non-contiguous source view;
- host transfers throw `std::invalid_argument` for the wrong byte count or a non-canonical `BOOL` input byte;
- `DeviceOps::copy` throws `std::invalid_argument` for incompatible metadata or a view from another `Device`;
- waiting on a zero, unknown, or live foreign-queue `oid` throws `std::invalid_argument`;
- unsupported tensor specifications, operations, and backend capabilities throw `std::runtime_error`;
- exceptions from `alloc` and backend construction propagate after owned resources are cleaned up; the injected allocator's `free` must not throw;
- exhaustion of the 56-bit per-queue submission sequence throws `std::overflow_error`, while exhaustion of the live eight-bit queue-ID space throws `std::runtime_error`.

Argument and capability failures occur before submission or destination writes. A synchronous host-transfer backend failure may leave destination values unspecified but does not alter metadata or ownership. An asynchronous backend failure is reported by `wait(oid)` and follows the same metadata and ownership guarantee.

## 10. Implementation touchpoints and staged build structure

Current files:

- `include/iom/tensor.hpp`: retain the existing `DataType` and `QuantizationFormat` enumerators, remove the empty `TileSize`, and implement `TensorShape`, `TensorSpec`, `TensorView`, `BackendKind`, and `Tensor`;
- `include/iom/iom.hpp`: keep `oid` as `std::uint64_t`, centralize process-wide queue-ID leasing in the common `DeviceOps` base, change every tensor operand to `TensorView`, retain `DeviceOps::copy`, and make `silu` input const;
- `src/safetensors.cpp` and `test/test_safetensors.cpp`: keep the `DataType` API and repair stale dtype mappings, specifically safetensors `F4` to `DataType::F4_E2M1` and `F8_E4M3` to `DataType::F8_E4M3FN`;
- `src/iom.cpp`: implement checked metadata, standard-layout, view-transform, and common accessor behavior;
- `src/llama.cpp`: pass `tensor.view()` to every `DeviceOps` call and replace nonexistent `set_float` calls with typed host transfers;
- `test/test_iom.cpp`: replace the dummy test with common metadata, layout, view, lifetime, and CPU reference tests;
- `CMakeLists.txt` and `test/CMakeLists.txt`: split optional backend targets and their conformance tests.

New layered surfaces:

- `include/iom/device.hpp`: common `Device` contract;
- backend-specific public headers under `include/iom/{cpu|cuda|rocm|sycl|ttnn}/`;
- backend-specific source targets under `src/{cpu|cuda|rocm|sycl|ttnn}/`;
- backend-specific tests under `test/{cpu|cuda|rocm|sycl|ttnn}/`;
- a shared storage, transfer, view, and copy conformance harness under `test/backend/`.

The always-built `libiom` target contains core code and the CPU reference backend. CUDA, ROCm, SYCL, and TTNN each build as an independent optional library with its own runtime dependencies and compiler settings. Existing `CUDA_ENABLED` and `ROCM_ENABLED` become independent; `SYCL_ENABLED` and `TTNN_ENABLED` are added. No option disables another backend. Linking several backend libraries into one executable must not create duplicate symbols, a global active-backend state, or runtime selection side effects.

`include/iom/alloc.hpp` is reused without interface changes.

Implementation is delivered in dependency order:

1. implement and test core metadata, layout, tensor/view contracts, CPU storage, host transfers, and CPU asynchronous copy;
2. select exactly one accelerator backend;
3. add that backend's dependency discovery, public factory, source target, and construction smoke test;
4. implement its storage, host transfers, asynchronous copy, and shared conformance cases;
5. configure and build it independently and pass its hardware-backed conformance target;
6. only then begin the next accelerator backend;
7. after all individual backends pass, add the combined build/link and coexistence checks.

The accelerator order is chosen according to dependency availability; the constraint is one active backend scaffold at a time, not a fixed vendor order. A scaffold is an intermediate implementation milestone, not completion: no backend is considered delivered while its factory or required storage/copy path is a stub.

## 11. Automated acceptance criteria

### 11.1 Metadata and encoding

Deterministic unit tests verify:

- valid nonzero shapes at ranks 2, 3, 4, and greater than 4; ranks 0 and 1 and every zero-dimension position are rejected;
- `[1, N]` is the vector representation and `dimension(rank())` is rejected;
- every declared `DataType` validates with `QuantizationFormat::NONE`;
- CPU, CUDA, ROCm, and SYCL materialize every validated `DataType`, while TTNN materializes at least `BF16`;
- every other declared `QuantizationFormat` is rejected as unsupported before allocation, and values outside both enums are rejected as invalid;
- element-count, logical-byte-count, padding, stride, and storage-byte overflows are detected;
- every `DataType` has the bit width in section 3 and every 16x16 tile payload is a multiple of 32 bytes;
- every safetensors dtype in section 3 maps to the stated `DataType`, with all other strings rejected.

Safetensors mapping tests use generated minimal headers or a direct parser seam. They do not depend on an external model path and cannot skip when model files are absent.

### 11.2 Standard layout and host transfers

Pure layout tests cover multiple leading planes, ranks above four, non-square matrices, and padding on each final dimension. They include:

```text
[1, 17]          -> [16, 32]
[2, 8, 31, 33]  -> [2, 8, 32, 48]

[0, 0]   ->   0
[0, 16]  -> 256
[16, 0]  -> 512
[16, 16] -> 768
```

CPU tests write independently chosen encoded byte patterns into padded tensors and inspect the allocation directly, then perform the inverse host read. This prevents a matching pair of incorrect pack/unpack functions from passing through round-trip cancellation. The cases cover every `DataType`, multi-byte endianness, sub-byte bit positions, zero output tail bits, hidden padding, exact 32-byte base alignment, wrong-span rejection before writes, and no numeric conversion. Invalid `BOOL` host bytes are rejected before writes, while valid zero and one bytes round-trip unchanged.

### 11.3 Safe view transforms

CPU tests verify:

- `Tensor::view()` returns the same stable full-view object with zero offset and dense row-major plane strides; `TensorView` is copy/move constructible but not assignable;
- stepped and unstepped `slice`, `select`, and every leading-axis permutation produce the expected shape, offset, and strides;
- all four metadata transforms work on const views;
- `reshape_leading` accepts contiguous split/merge and size-one edits, preserves coordinate order, and rejects non-contiguous or product-changing inputs;
- rank-two empty permutations and rank-two/size-one leading reshapes cover the empty-product boundary;
- nested transforms map every logical leading coordinate to the same owner plane as an independently calculated reference map;
- interior, stepped, and permuted views round-trip through host transfers, and writing one view changes exactly its addressed owner planes;
- invalid leading dimensions, zero count/step, out-of-range last elements, duplicate/missing permutation indices, and arithmetic overflow are rejected;
- transforms never call the tensor-storage allocator or move tensor data.

### 11.4 Device and ownership

A recording allocator and CPU device verify one exact-size allocation, rejection of null and misaligned returns, one matching free, cleanup after construction failure, rejection of an unsupported specification before allocation, no tensor-storage allocator calls from host transfers/view transforms/device copies, deleted tensor copy/move operations, and the documented device/allocator lifetime order.

Device tests also verify:

- a compile fixture implements a custom `Device`;
- each factory and every view reports the expected `BackendKind` and backend-local device ordinal;
- each accelerator factory rejects an unavailable ordinal before tensor or queue resource creation;
- contexts created by accelerator factories are destroyed exactly once after all tensors and queues;
- two queues created from one device can address tensors from that device;
- devices from every simultaneously enabled backend can be constructed and used in one process without shared active-backend state;
- queue IDs remain unique when queues are constructed concurrently, and releasing a queue makes its ID available for reuse;
- a queue rejects views created by another `Device`, including another instance for the same backend ordinal.

### 11.5 Asynchronous copy and lifetime

The CPU reference queue and a deterministic deferred fake verify:

- submission returns an `oid` with the queue ID in its high eight bits and a nonzero monotonic sequence in its low 56 bits;
- `wait(oid)` observes completion, is idempotent after both success and failure, and same-queue submissions execute in order;
- zero, unknown-sequence, and live foreign-queue tokens are rejected deterministically;
- queue-ID exhaustion, ID reuse after queue destruction, and sequence exhaustion follow the bounds in section 8;
- validation and synchronous submission failures do not consume sequence numbers;
- every safe dense or strided view shape copies in logical coordinate order;
- identical-window copy is a no-op, metadata mismatch and foreign-device views fail before writes, and no implicit host staging occurs;
- the deferred fake observes the same derived-view and owner addresses at submission and completion;
- a deterministic fake can initialize its next sequence near `2^56 - 1`, so sequence exhaustion is tested without issuing impractically many operations;
- all submitted work is waited before queue destruction, and queue destruction performs no implicit synchronization;
- injected synchronous-host and asynchronous-copy failures leave metadata, ownership, and native handles unchanged, with the asynchronous error reported by `wait`;
- views, owners, and storage remain stable through completion.

### 11.6 Accelerator conformance

For each enabled CUDA, ROCm, SYCL, and TTNN backend, the same parameterized suite:

1. creates matching CPU-reference and accelerator tensors;
2. seeds identical logical host buffers;
3. performs full-tensor and transformed-view host transfers and same-device asynchronous copies;
4. reads logical results back;
5. compares them bit-for-bit with CPU.

The cases cover every `DataType` for CUDA, ROCm, and SYCL. The TTNN suite contains an explicit, nonempty supported-type table that includes `BF16`, covers every type in that table, and verifies rejection of every other type. All suites cover only `QuantizationFormat::NONE`, plus padding, multiple leading ranks including rank greater than four, stepped slices, selects, permutations, contiguous reshapes, nested transforms, and validation failures. Enabling a backend's conformance target requires its hardware/runtime; the test must not silently skip an enabled backend.

Compute methods are compile-checked for the `TensorView` signatures and share the tested plane-indexing helper. Each not-yet-implemented compute method is also verified to fail capability validation without submitting work or changing an output. Numerical compute tests are intentionally absent here; the conformance fixture and CPU-reference device remain reusable by later operation specifications.

### 11.7 Build matrix

Automated builds proceed in the staged order from section 10. Core-plus-CPU and each optional backend independently must configure, compile, link, and run their applicable tests before work starts on another backend. After the individual milestones pass, a combined job enables and links all four optional backend libraries into one executable. Backend-specific public headers are not included by the core target, and disabling one backend removes only its library, factory, and tests.

## 12. Completion criteria

The change is complete when:

- the normalized metadata, materialized tensors, safe views, `Device`, and asynchronous view copy behave as specified;
- `TensorSpec` uses `DataType` plus `QuantizationFormat`; only `QuantizationFormat::NONE` is accepted, all leaf types are materialized by the standard-layout backends, and TTNN materializes at least `BF16`;
- CPU is the executable reference and all four accelerator storage/copy implementations pass the same conformance suite on their supported runtime;
- every `DeviceOps` tensor operand is a `TensorView`, all affected callers are migrated, and `DeviceOps::copy` remains asynchronous;
- the cutover leaves no deprecated tensor overloads or compatibility aliases;
- safetensors builds against the current `DataType` enumerators and its accepted dtype strings have deterministic mapping tests;
- each backend is scaffolded, implemented, built, and tested as a separate milestone before the next backend begins;
- any enabled backend combination builds and coexists without a global active backend;
- no grouped storage contract, raw-stride API, last-two-dimension view, allocator replacement, borrowed-context mode, status/result, graph, workspace, or operation-dispatch abstraction is added;
- every validation, ownership, layout, transfer, lifetime, and build guarantee above has deterministic automated coverage.

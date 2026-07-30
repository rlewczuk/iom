
# Minimal tiled-tensor API for an LLM inference engine

## 1. Design rules

The API is based on the following invariants:

1. **Every engine tensor is tiled.**

    * A vector is represented as a two-dimensional tensor such as `[1, N]`.
    * The last two dimensions are tiled.
    * Leading dimensions are ordinary batch dimensions.

2. **Operations never allocate output tensors.**

    * Public operations always receive output tensors by mutable reference.
    * Example: `linear(ctx, x, weight, bias, y)`.
    * No operation returns a newly allocated engine tensor.

3. **Tensor transfers are always explicit.**

    * No operation automatically moves tensors between host and device.
    * No operation automatically moves tensors between devices.
    * No operation automatically changes backend.

4. **CPU, CUDA, and ROCm tensors use fixed storage allocated during initialization.**

    * Tensor metadata remains on the host.
    * An injected allocator supplies buffer regions.
    * Operations perform no allocator calls.

5. **TTNN tensors use native `ttnn::Tensor` storage.**

    * The public tensor object remains stable.
    * The TTNN backend may replace the native `ttnn::Tensor` held by an output tensor.
    * Replacing the native value releases the previous value through normal RAII.

6. **The engine is imperative.**

    * There is no graph construction, graph compilation, autograd, or backward pass.

7. **Higher-level systems manage streaming and parallelism.**

    * Expert streaming uses explicit `copy` calls.
    * Expert parallelism is implemented by maintaining tensors on individual devices and invoking communication explicitly.
    * Paged KV cache logic owns page tables and invokes explicit cache operations.

# 2. Tiled tensor model

## 2.1 Logical shape

The public tensor shape always has at least two dimensions.

Examples:

```text
Single-token hidden state: [1, hidden_size]
Token batch:               [batch, hidden_size]
Attention tensor:          [batch, heads, sequence, head_dim]
Expert weights:            [experts, input_dim, output_dim]
```

Rank-one device tensors are not supported. A logical vector of length `N` is represented as `[1, N]`.

```c++
namespace llm {

constexpr std::size_t kMaxRank = 6;

struct Shape {
    std::uint8_t rank = 0;
    std::array<std::uint32_t, kMaxRank> dimensions{};

    [[nodiscard]] std::uint32_t operator[](std::size_t index) const noexcept {
        return dimensions[index];
    }
};

struct TileShape {
    std::uint16_t rows = 0;
    std::uint16_t columns = 0;

    friend bool operator==(const TileShape&, const TileShape&) = default;
};

enum class DType : std::uint16_t {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int8,
    UInt8,

    // Add stable packed or quantized formats as required.
    QInt8,
    QInt4
};

struct TensorSpec {
    Shape shape;
    DType dtype;
    TileShape tile;

    friend bool operator==(const TensorSpec&, const TensorSpec&) = default;
};

} // namespace llm
```

## 2.2 Standard physical layout

CPU, CUDA, and ROCm use one engine-defined physical representation:

* leading dimensions are stored in row-major order;
* the final two dimensions are divided into tiles;
* tiles are stored in row-major tile order;
* elements inside each tile are stored in row-major order;
* final dimensions are padded to complete tiles.

For a tensor:

```text
[D0, D1, ..., M, N]
```

with tile shape:

```text
[TR, TC]
```

the logical element `[..., m, n]` is located using:

```text
tile_row = m / TR
tile_col = n / TC
in_tile_row = m % TR
in_tile_col = n % TC
```

Conceptually, the physical dimensions are:

```text
[D0, D1, ..., ceil(M / TR), ceil(N / TC), TR, TC]
```

Padding is not part of the logical tensor. Its values are unspecified and must not be observable through the public API.

The Tenstorrent backend may use TTNN’s native tiled representation rather than this byte-level format. 
It must nevertheless expose equivalent logical semantics.

## 2.3 Shape utilities

Layout calculations should be pure functions so they can be unit-tested independently.

```c++
struct PaddedShape {
    Shape shape;
};

[[nodiscard]] Result<PaddedShape>
compute_padded_shape(const TensorSpec& spec);

[[nodiscard]] Result<std::size_t>
logical_element_count(const TensorSpec& spec);

[[nodiscard]] Result<std::size_t>
standard_tiled_storage_bytes(const TensorSpec& spec);
```

The backend remains responsible for the final storage requirement because quantized formats or native alignment requirements may alter the required size.

# 3. Tensor object

A tensor is a stable logical object consisting of:

* immutable logical metadata;
* its owning backend;
* backend-specific storage.

The tensor should be move-only. Accidental copying would otherwise make ownership and output replacement ambiguous.

```c++
namespace llm {

class Backend;

namespace detail {
class TensorStorage;
class TensorAccess;
}

class Tensor {
public:
    Tensor() noexcept = default;

    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    ~Tensor();

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool has_storage() const noexcept;

    [[nodiscard]] const TensorSpec& spec() const noexcept;
    [[nodiscard]] Backend& backend() const noexcept;

    // Releases the underlying manual allocation or native tensor.
    void reset() noexcept;

private:
    friend class detail::TensorAccess;

    Backend* backend_ = nullptr;
    TensorSpec spec_{};
    std::unique_ptr<detail::TensorStorage> storage_;
};

} // namespace llm
```

The `TensorSpec` does not change after construction.

For CPU, CUDA, and ROCm, the storage address also remains fixed until `reset()`.

For TTNN, only the backend-private storage object may be replaced. The public `Tensor` object and its `TensorSpec` remain unchanged.

## 3.1 Lifetime rule

A backend must outlive:

* all tensors created by that backend;
* all execution contexts created by that backend.

An allocator must outlive the backend and all tensors using that allocator.

This avoids shared ownership between every tensor, backend, and allocator.

# 4. Allocator contract

Only CPU, CUDA, and ROCm use this allocator interface.

The allocator is injected when the backend is created. It is only used during tensor and execution-context initialization.

```c++
namespace llm {

struct Allocation {
    void* address = nullptr;
    std::size_t bytes = 0;
    std::size_t alignment = 0;

    // Returned unchanged to deallocate(). It may identify a pool block,
    // offset, CUDA allocation, HIP allocation, or test allocation.
    std::uintptr_t token = 0;
};

class Allocator {
public:
    virtual ~Allocator() = default;

    [[nodiscard]] virtual Result<Allocation>
    allocate(std::size_t bytes, std::size_t alignment) = 0;

    virtual void deallocate(Allocation allocation) noexcept = 0;
};

struct StorageRequirements {
    std::size_t bytes = 0;
    std::size_t alignment = 0;
};

} // namespace llm
```

An allocator instance is associated with one memory domain. For example:

* CPU allocator: host memory;
* CUDA allocator: memory on one CUDA device;
* ROCm allocator: memory on one ROCm device.

The backend therefore does not pass a device argument to every allocator call.

## 4.1 Fixed-pool allocator

A normal CPU, CUDA, or ROCm configuration would use an allocator backed by one externally created pool:

```c++
class FixedPoolAllocator final : public Allocator {
public:
    FixedPoolAllocator(void* base, std::size_t bytes);

    Result<Allocation>
    allocate(std::size_t bytes, std::size_t alignment) override;

    void deallocate(Allocation allocation) noexcept override;

    void reset();
};
```

Two implementations are reasonable:

* a monotonic allocator whose `deallocate` is a no-op and whose entire pool is reset together;
* a free-list allocator that can reclaim tensor ranges individually.

The allocator policy is not visible to tensors or operations.

## 4.2 Storage requirement queries

Manual-memory backends expose exact requirements before creating a tensor:

```c++
class Backend {
public:
    virtual Result<StorageRequirements>
    storage_requirements(const TensorSpec& spec) const = 0;
};
```

For TTNN, this function may return `StatusCode::Unsupported`, because the native runtime owns allocation and placement.

This query makes static planning straightforward:

```cpp
auto requirements = cuda_backend.storage_requirements(spec);
auto tensor = cuda_backend.create_tensor(spec);
```

The same function can be used by tests to verify padding, tile sizes, alignment, and quantized storage calculations.

# 5. Backend and execution context

A backend instance represents one execution target.

Examples:

* one CPU backend;
* CUDA device 0;
* CUDA device 1;
* ROCm device 0;
* one TTNN device or mesh configuration.

```c++
namespace llm {

enum class BackendKind {
    Cpu,
    Cuda,
    Rocm,
    Ttnn
};

struct ContextOptions {
    // Preallocated scratch memory for CPU/CUDA/ROCm kernels.
    // Zero is valid for backends or kernels that do not require it.
    std::size_t workspace_bytes = 0;
};

class ExecutionContext {
public:
    ExecutionContext(ExecutionContext&&) noexcept;
    ExecutionContext& operator=(ExecutionContext&&) noexcept;

    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;

    ~ExecutionContext();

    [[nodiscard]] Backend& backend() const noexcept;

    // Waits for all previously submitted operations on this context.
    Status synchronize();

private:
    friend class Backend;

    Backend* backend_ = nullptr;
    std::unique_ptr<detail::BackendExecutionContext> implementation_;
};

class Backend {
public:
    virtual ~Backend() = default;

    [[nodiscard]] virtual BackendKind kind() const noexcept = 0;

    [[nodiscard]] virtual Status
    validate_tensor_spec(const TensorSpec& spec) const = 0;

    [[nodiscard]] virtual Result<StorageRequirements>
    storage_requirements(const TensorSpec& spec) const = 0;

    [[nodiscard]] virtual Result<Tensor>
    create_tensor(const TensorSpec& spec) = 0;

    [[nodiscard]] virtual Result<ExecutionContext>
    create_context(const ContextOptions& options) = 0;
};

} // namespace llm
```

The execution context owns:

* a CPU execution queue, CUDA stream, HIP stream, or TTNN command-queue state;
* one fixed workspace allocation where applicable;
* backend-specific launch state.

Operations are submitted in order to an `ExecutionContext`.

No operation creates temporary allocations through the general allocator. A kernel that needs scratch memory uses the context workspace.

If the workspace is insufficient, the operation returns an error before launch.

# 6. Row-major host access

Raw tiled device data should not be exposed through the public API.

Instead, the public API provides explicit copies between engine tensors and contiguous logical row-major host views.

```c++
namespace llm {

struct ConstHostTensorView {
    const void* data = nullptr;
    std::size_t bytes = 0;
    Shape shape;
    DType dtype;
};

struct HostTensorView {
    void* data = nullptr;
    std::size_t bytes = 0;
    Shape shape;
    DType dtype;
};

template<typename T>
ConstHostTensorView make_host_view(
    std::span<const T> values,
    const Shape& shape,
    DType dtype);

template<typename T>
HostTensorView make_host_view(
    std::span<T> values,
    const Shape& shape,
    DType dtype);

Status copy_from_host(
    ExecutionContext& context,
    const ConstHostTensorView& source,
    Tensor& destination);

Status copy_to_host(
    ExecutionContext& context,
    const Tensor& source,
    const HostTensorView& destination);

} // namespace llm
```

These operations:

* require matching logical shapes;
* require matching dtypes;
* convert between logical row-major ordering and backend tiled ordering;
* ignore physical padding;
* are submitted to the context;
* do not implicitly synchronize.

Example:

```c++
std::vector<float> input_values(hidden_size);

Shape input_shape{
    .rank = 2,
    .dimensions = {1, hidden_size}
};

auto host_input = make_host_view(
    std::span<const float>(input_values),
    input_shape,
    DType::Float32);

LLM_RETURN_IF_ERROR(
    copy_from_host(cuda_context, host_input, input));

LLM_RETURN_IF_ERROR(cuda_context.synchronize());
```

For reading:

```c++
std::vector<float> output_values(vocabulary_size);

auto host_output = make_host_view(
    std::span<float>(output_values),
    output.spec().shape,
    DType::Float32);

LLM_RETURN_IF_ERROR(
    copy_to_host(cuda_context, output, host_output));

LLM_RETURN_IF_ERROR(cuda_context.synchronize());
```

This API is also the primary mechanism for backend-independent numerical unit tests.

# 7. Tensor-to-tensor copies

Tensor transfers are represented by one explicit operation:

```c++
Status copy(
    ExecutionContext& context,
    const Tensor& source,
    Tensor& destination);
```

The operation copies logical tensor values, not raw storage bytes.

Requirements:

* source and destination logical shapes must match;
* source and destination dtypes must match;
* tile shapes may differ;
* the responsible backend may retile during the copy;
* no numeric dtype conversion is performed;
* no hidden host staging is permitted.

The execution context identifies the backend responsible for the transfer.

Examples:

```c++
copy(cuda0_context, cpu_expert, cuda0_expert);
copy(cuda1_context, cuda0_tensor, cuda1_tensor);
copy(ttnn_context, host_tensor, ttnn_tensor);
```

A backend may reject a transfer path.

For example, a CUDA-to-TTNN copy may not have a direct implementation. The caller must then stage it explicitly:

```cpp
copy_to_host(cuda_context, cuda_tensor, host_view);
cuda_context.synchronize();

copy_from_host(ttnn_context, host_view, ttnn_tensor);
ttnn_context.synchronize();
```

There is no implicit fallback through host memory.

Initially, `copy` should operate on complete tensors only. Expert streaming can model each streamable expert as a separate tensor. Tile-aligned region copies can be added later if concrete use cases require them.

# 8. Public operation API

Operations are free functions in an `ops` namespace.

Every operation:

* receives an execution context;
* receives input tensors by `const Tensor&`;
* receives outputs by `Tensor&`;
* returns `Status`;
* performs validation before backend dispatch;
* never creates an engine tensor;
* never performs an implicit transfer.

```cpp
namespace llm::ops {

struct LinearOptions {
    bool transpose_weight = false;
};

Status linear(
    ExecutionContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    const LinearOptions& options = {});

Status matmul(
    ExecutionContext& context,
    const Tensor& left,
    const Tensor& right,
    Tensor& output);

Status add(
    ExecutionContext& context,
    const Tensor& left,
    const Tensor& right,
    Tensor& output);

Status multiply(
    ExecutionContext& context,
    const Tensor& left,
    const Tensor& right,
    Tensor& output);

struct RmsNormOptions {
    float epsilon;
};

Status rms_norm(
    ExecutionContext& context,
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    const RmsNormOptions& options);

struct SoftmaxOptions {
    std::int32_t axis = -1;
};

Status softmax(
    ExecutionContext& context,
    const Tensor& input,
    Tensor& output,
    const SoftmaxOptions& options = {});

} // namespace llm::ops
```

A decoding step looks like:

```cpp
LLM_RETURN_IF_ERROR(
    ops::rms_norm(ctx, hidden, norm_weight, normalized, {.epsilon = 1e-5f}));

LLM_RETURN_IF_ERROR(
    ops::linear(ctx, normalized, qkv_weight, qkv_bias, qkv));

LLM_RETURN_IF_ERROR(
    ops::linear(ctx, attention_output, output_weight, nullptr, projected));

LLM_RETURN_IF_ERROR(
    ops::add(ctx, hidden, projected, next_hidden));
```

No output allocation occurs during this sequence.

## 8.1 Validation rules

The public operation layer should perform inexpensive common validation:

* context is valid;
* all non-copy tensors belong to `context.backend()`;
* logical ranks and dimensions are valid;
* output shape and dtype match the operation contract;
* output has storage;
* tile shapes are supported by the backend;
* inputs and outputs do not alias unless the operation explicitly permits it.

The backend performs any additional kernel-specific validation.

## 8.2 Aliasing

Aliasing should be disallowed by default.

Instead of permitting ambiguous calls such as:

```cpp
add(ctx, x, y, x);
```

provide a specifically documented in-place operation where required:

```cpp
Status add_in_place(
    ExecutionContext& context,
    Tensor& destination,
    const Tensor& source);
```

This keeps kernel requirements and tests unambiguous.

# 9. Backend dispatch API

The public operation functions dispatch to backend methods.

The backend-facing interface can remain direct rather than introducing an operation graph or generalized command object.

```cpp
namespace llm {

class Backend {
public:
    // Resource methods omitted here.

    virtual Status enqueue_copy_from_host(
        ExecutionContext& context,
        const ConstHostTensorView& source,
        Tensor& destination) = 0;

    virtual Status enqueue_copy_to_host(
        ExecutionContext& context,
        const Tensor& source,
        const HostTensorView& destination) = 0;

    virtual Status enqueue_copy(
        ExecutionContext& context,
        const Tensor& source,
        Tensor& destination) = 0;

    virtual Status enqueue_linear(
        ExecutionContext& context,
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        Tensor& output,
        const ops::LinearOptions& options) = 0;

    virtual Status enqueue_matmul(
        ExecutionContext& context,
        const Tensor& left,
        const Tensor& right,
        Tensor& output) = 0;

    virtual Status enqueue_add(
        ExecutionContext& context,
        const Tensor& left,
        const Tensor& right,
        Tensor& output) = 0;

    virtual Status enqueue_rms_norm(
        ExecutionContext& context,
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        const ops::RmsNormOptions& options) = 0;
};

} // namespace llm
```

This creates one virtual method per public operation. For an inference engine with a controlled operation set, that is simpler than:

* a general command union;
* runtime operation registration;
* type-erased attribute dictionaries;
* graph nodes;
* generic variadic tensor arrays.

The public wrapper remains responsible for portable validation:

```c++
Status ops::linear(
    ExecutionContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    const LinearOptions& options) {

    LLM_RETURN_IF_ERROR(
        validate_linear(context, input, weight, bias, output, options));

    return context.backend().enqueue_linear(
        context, input, weight, bias, output, options);
}
```

# 10. Kernel API for CPU, CUDA, and ROCm

CPU, CUDA, and ROCm share a common host-side tiled descriptor.

```c++
namespace llm::kernel {

struct TiledTensorDesc {
    DType dtype;

    std::uint8_t rank;
    std::array<std::uint32_t, kMaxRank> logical_dimensions;
    std::array<std::uint32_t, kMaxRank> padded_dimensions;

    std::uint16_t tile_rows;
    std::uint16_t tile_columns;

    std::size_t storage_bytes;
};

template<typename Address>
struct ConstTensorArg {
    Address address{};
    TiledTensorDesc desc{};
};

template<typename Address>
struct TensorArg {
    Address address{};
    TiledTensorDesc desc{};
};

} // namespace llm::kernel
```

The backend converts the public `Tensor` into this form through a private accessor.

No persistent tensor metadata is stored on the device.

For CUDA and ROCm:

* metadata is maintained in the host-side `Tensor`;
* the launch wrapper extracts the required fields;
* small kernel parameters are passed by value during launch;
* a device-side metadata allocation is not created.

## 10.1 CPU kernel interface

```cpp
namespace llm::cpu::kernel {

using ConstTensorArg =
    llm::kernel::ConstTensorArg<const std::byte*>;

using TensorArg =
    llm::kernel::TensorArg<std::byte*>;

struct Context {
    std::byte* workspace = nullptr;
    std::size_t workspace_bytes = 0;
    ThreadPool* thread_pool = nullptr;
};

Status linear(
    Context& context,
    const ConstTensorArg& input,
    const ConstTensorArg& weight,
    const ConstTensorArg* bias,
    TensorArg& output,
    const ops::LinearOptions& options);

} // namespace llm::cpu::kernel
```

## 10.2 CUDA kernel interface

```cpp
namespace llm::cuda::kernel {

using ConstTensorArg =
    llm::kernel::ConstTensorArg<CUdeviceptr>;

using TensorArg =
    llm::kernel::TensorArg<CUdeviceptr>;

struct Context {
    CUstream stream = nullptr;
    CUdeviceptr workspace = 0;
    std::size_t workspace_bytes = 0;
};

Status linear(
    Context& context,
    const ConstTensorArg& input,
    const ConstTensorArg& weight,
    const ConstTensorArg* bias,
    TensorArg& output,
    const ops::LinearOptions& options);

} // namespace llm::cuda::kernel
```

## 10.3 ROCm kernel interface

```cpp
namespace llm::rocm::kernel {

using ConstTensorArg =
    llm::kernel::ConstTensorArg<const void*>;

using TensorArg =
    llm::kernel::TensorArg<void*>;

struct Context {
    hipStream_t stream = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
};

Status linear(
    Context& context,
    const ConstTensorArg& input,
    const ConstTensorArg& weight,
    const ConstTensorArg* bias,
    TensorArg& output,
    const ops::LinearOptions& options);

} // namespace llm::rocm::kernel
```

The kernel launch interfaces do not need to be binary-compatible with one another. They only need equivalent behavior behind the common public operation contract.

## 10.4 Operation-specific launch parameters

Individual kernels should not be required to consume the entire generic tensor descriptor.

The backend launch wrapper may lower it into a smaller operation-specific structure:

```cpp
struct LinearLaunchParams {
    std::uint32_t batch;
    std::uint32_t input_features;
    std::uint32_t output_features;

    std::uint16_t input_tile_rows;
    std::uint16_t input_tile_columns;

    std::uint32_t input_tile_columns_count;
    std::uint32_t output_tile_columns_count;
};
```

This keeps device kernel interfaces small while preserving one common host-side tensor representation.

# 11. Backend-private tensor access

Raw storage access is not public.

Backends use one internal helper:

```cpp
namespace llm::detail {

struct ManualStorageView {
    void* address = nullptr;
    std::size_t bytes = 0;
};

class TensorAccess {
public:
    static ManualStorageView manual_storage(Tensor& tensor);
    static ManualStorageView manual_storage(const Tensor& tensor);

    static TensorStorage& storage(Tensor& tensor);
    static const TensorStorage& storage(const Tensor& tensor);

    static void replace_storage(
        Tensor& tensor,
        std::unique_ptr<TensorStorage> replacement);

    static void clear_storage(Tensor& tensor) noexcept;
};

} // namespace llm::detail
```

`replace_storage` performs common checks:

* the replacement belongs to the same backend;
* its logical shape matches `Tensor::spec()`;
* its dtype matches `Tensor::spec()`;
* its tile shape is compatible;
* the old storage is destroyed only after validation succeeds.

CPU, CUDA, and ROCm normally call only `manual_storage()`.

The TTNN adapter uses `replace_storage()` and `clear_storage()`.

# 12. TTNN backend adapter

The TTNN backend uses a backend-private storage implementation:

```cpp
namespace llm::ttnn_backend {

class TensorStorage final : public detail::TensorStorage {
public:
    explicit TensorStorage(ttnn::Tensor tensor);

    ttnn::Tensor& native() noexcept;
    const ttnn::Tensor& native() const noexcept;

private:
    ttnn::Tensor tensor_;
};

} // namespace llm::ttnn_backend
```

The adapter exposes internal helpers:

```cpp
class TtnnBackend final : public Backend {
private:
    ttnn::Tensor& native(Tensor& tensor);
    const ttnn::Tensor& native(const Tensor& tensor) const;

    Status replace_native(
        Tensor& destination,
        ttnn::Tensor replacement);

    void clear_native(Tensor& tensor) noexcept;
};
```

A TTNN linear operation may then use a return-value-based native API while preserving the engine’s predeclared output API:

```cpp
Status TtnnBackend::enqueue_linear(
    ExecutionContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& output,
    const ops::LinearOptions& options) {

    const auto& native_input = native(input);
    const auto& native_weight = native(weight);

    ttnn::Tensor native_output;

    if (bias != nullptr) {
        native_output = ttnn::linear(
            native_input,
            native_weight,
            native(*bias));
    } else {
        native_output = ttnn::linear(
            native_input,
            native_weight);
    }

    return replace_native(output, std::move(native_output));
}
```

The public call remains:

```cpp
ops::linear(ctx, input, weight, bias, output);
```

The old native output tensor is released when its storage wrapper is replaced.

## 12.1 TTNN output rules

For TTNN, an output tensor is a stable logical slot rather than necessarily a stable allocation.

The following remain fixed:

* public `Tensor` identity;
* logical shape;
* dtype;
* tile dimensions;
* owning backend.

The following may change after an operation:

* native `ttnn::Tensor`;
* TTNN buffer handle;
* TTNN memory placement;
* backend-private native metadata.

If a TTNN operation returns a tensor that is incompatible with the declared output `TensorSpec`, the adapter returns an error and leaves the existing output unchanged.

# 13. Tensor creation examples

## 13.1 CUDA with a static pool

```cpp
void* cuda_pool_address = allocate_cuda_pool(pool_bytes);

FixedPoolAllocator allocator(cuda_pool_address, pool_bytes);
CudaBackend cuda_backend(/*device=*/0, allocator);

TensorSpec hidden_spec{
    .shape = Shape{
        .rank = 2,
        .dimensions = {1, hidden_size}
    },
    .dtype = DType::Float16,
    .tile = TileShape{16, 16}
};

auto hidden = LLM_TRY(cuda_backend.create_tensor(hidden_spec));
auto normalized = LLM_TRY(cuda_backend.create_tensor(hidden_spec));
auto output = LLM_TRY(cuda_backend.create_tensor(hidden_spec));

auto context = LLM_TRY(cuda_backend.create_context({
    .workspace_bytes = 16 * 1024 * 1024
}));
```

All tensor and workspace allocations occur before inference.

## 13.2 TTNN

```cpp
TtnnBackend ttnn_backend(mesh_device, default_memory_config);

TensorSpec hidden_spec{
    .shape = Shape{
        .rank = 2,
        .dimensions = {1, hidden_size}
    },
    .dtype = DType::BFloat16,
    .tile = TileShape{32, 32}
};

auto hidden = LLM_TRY(ttnn_backend.create_tensor(hidden_spec));
auto normalized = LLM_TRY(ttnn_backend.create_tensor(hidden_spec));
auto output = LLM_TRY(ttnn_backend.create_tensor(hidden_spec));

auto context = LLM_TRY(ttnn_backend.create_context({}));
```

`create_tensor` may initially create native TTNN storage. Subsequent output-producing operations may replace that storage.

# 14. Error handling

Use explicit `Status` and `Result<T>` types rather than exceptions in the core API.

```cpp
enum class StatusCode {
    Ok,
    InvalidArgument,
    InvalidShape,
    InvalidDType,
    InvalidTileShape,
    BackendMismatch,
    Unsupported,
    InsufficientStorage,
    InsufficientWorkspace,
    TransferNotSupported,
    BackendError
};

class Status {
public:
    [[nodiscard]] bool ok() const noexcept;
    [[nodiscard]] StatusCode code() const noexcept;
    [[nodiscard]] std::string_view message() const noexcept;
};

template<typename T>
class Result {
public:
    [[nodiscard]] bool ok() const noexcept;
    [[nodiscard]] const Status& status() const noexcept;

    T& value() &;
    T&& value() &&;
};
```

Backend-native errors should be translated into stable engine error codes while retaining a diagnostic message.

# 15. Unit-test design

## 15.1 Pure layout tests

Test without any backend:

* logical-to-tiled offset calculation;
* tile padding;
* storage size calculation;
* rank validation;
* `[1, N]` vector layout;
* multiple leading dimensions;
* non-square tiles.

Example:

```cpp
TEST(TiledLayout, VectorUsesOneByNTiling);
TEST(TiledLayout, PadsFinalTwoDimensions);
TEST(TiledLayout, LeadingDimensionsAreIndependentTilePlanes);
```

## 15.2 Allocator tests

Use a fake allocator that records calls:

```cpp
class RecordingAllocator final : public Allocator {
public:
    std::vector<AllocationRequest> allocations;
    std::vector<Allocation> deallocations;
};
```

Verify:

* tensor creation requests the expected size and alignment;
* tensor destruction calls `deallocate`;
* no allocation occurs during `linear`, `add`, or `copy`;
* execution-context workspace is allocated only during context creation.

## 15.3 Mock backend tests

A mock backend can record public operation dispatch:

```cpp
struct RecordedLinearCall {
    const Tensor* input;
    const Tensor* weight;
    const Tensor* bias;
    Tensor* output;
};

class MockBackend final : public Backend {
public:
    std::vector<RecordedLinearCall> linear_calls;
};
```

Verify:

* `ops::linear` validates before dispatch;
* backend mismatch is rejected;
* incorrect output shape is rejected;
* output aliasing is rejected;
* valid calls reach the backend exactly once.

## 15.4 Host-copy tests

For every backend:

1. write row-major host values to a tensor;
2. read them back;
3. compare only the logical region;
4. test shapes that require padding;
5. test `[1, N]` decode tensors.

This verifies the backend’s tile conversion independently of compute kernels.

## 15.5 Manual-storage stability tests

For CPU, CUDA, and ROCm:

* record output storage address;
* execute an operation;
* verify that the address did not change;
* verify that no allocator call occurred.

## 15.6 TTNN replacement tests

Use a fake native tensor wrapper when unit-testing without hardware.

Verify:

* a returned native tensor replaces the previous output storage;
* the previous native object is destroyed exactly once;
* incompatible replacement metadata is rejected;
* a failed operation leaves the old output unchanged;
* `Tensor::reset()` frees the current native object.

## 15.7 Copy-path tests

Verify that:

* supported direct copies succeed;
* unsupported backend pairs return `TransferNotSupported`;
* no hidden host staging occurs;
* different tile shapes are correctly retiled;
* dtype conversion is rejected.

# 16. Intentionally omitted abstractions

The initial API should not include:

* compute graphs;
* automatic memory planning during execution;
* automatic transfers;
* automatic layout conversion inside compute operations;
* a distributed tensor abstraction;
* implicit tensor parallelism;
* implicit expert streaming;
* implicit KV-page movement;
* arbitrary strided tensor views;
* general operation registration;
* dynamic attribute maps;
* runtime tensor allocation by operations;
* a common binary kernel ABI across CPU, CUDA, ROCm, and TTNN;
* persistent device-side tensor metadata.

Higher-level code can represent multiple devices with ordinary containers:

```cpp
std::array<Tensor, 4> weight_shards;
std::array<ExecutionContext, 4> device_contexts;
```

Tensor parallelism can then explicitly invoke:

```cpp
for (std::size_t rank = 0; rank < 4; ++rank) {
    ops::linear(
        device_contexts[rank],
        input_shards[rank],
        weight_shards[rank],
        nullptr,
        output_shards[rank]);
}

collective_all_reduce(device_contexts, output_shards);
```

Expert streaming can similarly issue explicit copies:

```cpp
copy(
    gpu_context,
    host_experts[selected_expert],
    gpu_expert_slot);

ops::linear(
    gpu_context,
    routed_tokens,
    gpu_expert_slot,
    nullptr,
    expert_output);
```

The tensor layer therefore supplies the required mechanisms without embedding policy.

# 17. Suggested source layout

```text
include/llm/
    status.hpp
    shape.hpp
    tensor_spec.hpp
    tensor.hpp
    allocator.hpp
    backend.hpp
    execution_context.hpp
    copy.hpp
    ops/
        linear.hpp
        elementwise.hpp
        normalization.hpp
        attention.hpp

src/core/
    tensor.cpp
    validation.cpp
    tiled_layout.cpp
    copy.cpp

src/backends/cpu/
    cpu_backend.hpp
    cpu_backend.cpp
    cpu_kernel_api.hpp

src/backends/cuda/
    cuda_backend.hpp
    cuda_backend.cpp
    cuda_kernel_api.hpp

src/backends/rocm/
    rocm_backend.hpp
    rocm_backend.cpp
    rocm_kernel_api.hpp

src/backends/ttnn/
    ttnn_backend.hpp
    ttnn_backend.cpp
    ttnn_tensor_storage.hpp
```

The core library does not include CUDA, HIP, or TTNN headers. Those remain isolated in their respective backend modules.

# 18. Final API summary

The minimum public surface consists of:

```cpp
// Tensor lifecycle.
Backend::create_tensor(spec);
Backend::create_context(options);
Tensor::reset();

// Explicit data movement.
copy_from_host(context, host_view, tensor);
copy_to_host(context, tensor, host_view);
copy(context, source_tensor, destination_tensor);

// Imperative preallocated operations.
ops::linear(context, x, weight, bias, y);
ops::matmul(context, a, b, output);
ops::add(context, a, b, output);
ops::rms_norm(context, input, weight, output, options);

// Completion.
context.synchronize();
```

The minimum backend surface consists of:

```cpp
validate_tensor_spec();
storage_requirements();
create_tensor();
create_context();

enqueue_copy_from_host();
enqueue_copy_to_host();
enqueue_copy();

enqueue_linear();
enqueue_matmul();
enqueue_add();
enqueue_rms_norm();
```

CPU, CUDA, and ROCm lower tensors into:

```cpp
address + host-side tiled descriptor
```

TTNN lowers tensors into:

```cpp
ttnn::Tensor
```

and may replace the native value held by an output tensor.

This keeps the core API uniform without forcing fundamentally different storage systems into one artificial low-level representation.

The design can next be narrowed into concrete compilable headers and a mock backend test scaffold without changing its core contracts.


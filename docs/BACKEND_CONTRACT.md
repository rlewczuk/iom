# Backend contract

This document is the normative contract for an IOM backend. It complements
[ARCHITECTURE.md](ARCHITECTURE.md): the architecture document explains how the
existing implementations are arranged; this document defines what a new
implementation must preserve. “MUST”, “MUST NOT”, and “SHOULD” are normative.

The contract is intentionally expressed in terms of observable `Device`,
`Tensor`, `TensorView`, and `DeviceOps` behavior. A backend may use a vendor
runtime, native storage layout, streams, events, staging buffers, or a
synchronous CPU implementation internally, provided that it meets this
contract. It MUST NOT add a global active-backend selector, make common code
include a vendor runtime, or silently change the device/context that owns an
object.

## Part I — Implementing a backend

### What a backend contributes

A complete backend is a separate optional library and a public factory header.
It contributes all of the following:

1. a public `make_<backend>_device(...)` factory returning
   `std::unique_ptr<iom::Device>`;
2. a concrete, non-copyable/non-movable `Device` owning one backend runtime
   context and implementing capability reporting, tensor creation, and queue
   creation;
3. a concrete `Tensor` storage owner with synchronous logical host transfers;
4. one or more `DeviceOps` implementations that provide the required
   asynchronous `copy` operation and either implement or explicitly reject
   every compute capability;
5. storage/queue lifetime management that prevents storage reuse or release
   while the backend can still access it;
6. a smoke test, a conformance driver, CMake registration, and coexistence
   coverage when enabled with other backends.

The backend owns runtime state and storage mechanics. The common layer owns
logical tensor metadata, view transformations, queue-token encoding, and
wait/failure bookkeeping. Do not duplicate common validation or create a
backend-specific version of `TensorView`.

### Choose the storage model first

There are two valid implementation paths.

| Path | Use when | Required result |
| --- | --- | --- |
| **Standard tiled** | The backend can store the common 16x16 encoding directly. | Allocate `TensorSpec::tiled_storage_nbytes()`, report the immutable 23-type standard capability table in its exact order, and preserve every standard physical tile and padding byte. CPU, CUDA, ROCm, and SYCL follow this path. |
| **Native storage** | The runtime requires a different physical tile, plane, or allocation model. | Keep the public logical shape, view, transfer, copy, ownership, and queue contract unchanged; advertise only supported types; translate host logical bytes and view plane mappings correctly; supply a native test oracle that observes the canonical standard padded allocation. TTNN follows this path. |

A native path is not permission to change public tensor semantics. In
particular, final tensor dimensions are still the public 16x16-tiled matrix
contract even when a vendor resource uses a larger physical tile. Internal
native-only padding MUST be initialized/treated so it cannot contaminate
logical results or a later valid transfer.

### Recommended implementation order

1. **Add build and public surface.** Add the new `BackendKind` enumerator and
   `include/iom/<backend>/device.hpp` with only the factory/capability
   declarations required by users. Add an optional static library in the root
   `CMakeLists.txt`, link it privately to `libiom` and its SDK, and fail
   configuration if the enabled SDK is unavailable. Do not expose vendor types
   through `include/iom/device.hpp` or add a backend registry.
2. **Create and own one runtime context.** Resolve the backend-local ordinal,
   reject invalid ordinals before creating a usable device, and make that
   context/device identity stable for the `Device` lifetime. Borrowed
   allocator and `Device` lifetimes must outlast all tensors and queues they
   create. TTNN-style backends that own native storage may omit the allocator
   only when their factory documents that ownership model.
3. **Implement capabilities and tensor creation.** Return an immutable,
   nonempty span from `supported_data_types()`. Validate `TensorSpec` and
   reject any unadvertised type before native allocation. Allocate one stable
   storage owner, construct the common `Tensor` base with the creating device,
   and ensure the allocation respects the 32-byte base-alignment contract for
   allocator-backed storage.
4. **Implement synchronous host transfers.** Implement `region_from_host` and
   `region_to_host` for arbitrary valid transformed `TensorView`s. Convert
   between contiguous logical host bytes and physical storage without numeric
   conversion. Preserve every untouched owner plane and required padding.
5. **Implement `DeviceOps::copy`.** Validate operands before reserving work,
   submit a sequence through the common `DeviceOps` helpers, retain or fence
   backend work, and complete it in order. Start with `copy`; leave compute
   methods inherited until their real semantics, validation, and tests exist.
6. **Make shutdown and failure paths real.** Account for work after native
   submission has begun. Release normal resources only after a completion
   proof; quarantine/retire resources if failure prevents proving completion.
   A synchronous pre-submit error must leave no work and no token behind.
7. **Add the backend conformance driver.** Create a CPU reference device, the
   candidate device, and an independent foreign device where the runtime
   permits it. Supply allocator/context setup and, for an accelerator or
   native layout, an independent `AcceleratorStorageOracle`.
8. **Register smoke, conformance, and coexistence tests.** Use
   `add_iom_backend_tests` in `test/CMakeLists.txt`; create separate smoke and
   conformance executables. Extend `test/backend/test_backend_coexistence.cpp`
   and its CMake wiring so all enabled factory headers and libraries coexist in
   one process. Enabled hardware tests MUST fail when hardware is unavailable;
   they do not skip.
9. **Run the complete backend gate.** Run the backend smoke and conformance
   tests plus coexistence with the intended option combinations. Accelerator
   build/test execution follows the repository `csw-remote` procedure.

### Minimal conformance-driver shape

The shared harness is backend-neutral. A backend driver SHOULD mirror the CPU,
CUDA, ROCm, SYCL, or TTNN driver structure:

```cpp
struct BackendDevices {
    // Own allocators before the devices that borrow them.
    std::unique_ptr<iom::Device> reference; // CPU
    std::unique_ptr<iom::Device> candidate; // backend under test
    std::unique_ptr<iom::Device> foreign;   // independently created device

    iom_conformance::ConformanceDevices conformance() const {
        return {*reference, *candidate, *foreign};
    }
};
```

The driver calls the shared scenarios with **every** datatype returned by the
candidate capability span. A `TrafficGate`-aware allocator is strongly
recommended: it makes an allocation or free during a transfer, view
transformation, or operation a test failure. The driver must not test an
internal copy helper against itself. Instead, an accelerator/native oracle
seeds and observes storage through an independent vendor path, using the
canonical standard tiled byte representation defined by
`backend_conformance_oracle.hpp`.

### Exit criteria

Before considering a backend complete, confirm all of the following:

- its factory/header/library are individually usable and coexist with every
  enabled backend without global dispatch;
- supported types, creation, transfers, transformed views, copies, errors,
  tokens, lifetime, and unsupported capabilities pass the shared suite;
- an independent oracle catches a deliberately perturbed physical mapping;
- no operation/transfer/view allocates or releases operand/output tensor
  storage after setup;
- every asynchronous path has a completion proof or a retained failure and
  cannot permit unsafe resource reuse;
- the backend does not skip enabled-hardware tests.

## Part II — Normative behavioral specification

### 1. Device identity, capabilities, and factories

1. A `Device` instance owns exactly one backend runtime context and creates
   tensors and queues for that context. It is non-copyable and non-movable.
   The `Device` owner address is therefore stable for all objects it creates.
2. `backend_kind()` MUST return the backend's fixed `BackendKind` value and
   `backend_device()` MUST return the resolved backend-local ordinal.
3. There is no active-backend global, backend switch, or global runtime
   selection service. Backends expose their own factory and objects retain the
   identity of their creating `Device`.
4. `supported_data_types()` MUST return an immutable, nonempty storage span.
   Standard backends retain the shared 23-entry sequence. ADD, MUL, and SUB
   accept with `NONE` exactly the 21 numeric leaves `I2`, `U2`, `I4`, `U4`,
   `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`,
   `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, `F64`.
   DIV accepts only the nine floating leaves
   `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
   `F32`, and `F64`. Matching `BOOL`, `F8_E8M0`, non-`NONE` quantization, and
   integer DIV are `Unsupported` after earlier validation. Required numeric
   leaves MUST use internal staging or emulation rather than SDK narrowing.
5. A capability table describes storage, not an operation query. It MUST reject
   unadvertised storage types before native allocation; operation support is
   reported only by the operation facade.
6. `create_tensor` MUST call/observe `TensorSpec::validate()`. Today that
   means rank at least two, nonzero dimensions, checked size arithmetic, a
   declared leaf type, and `QuantizationFormat::NONE`; every other declared
   quantization format is invalid.
7. `create_ops` MUST return an independent queue associated with that exact
   device. A queue created by one device MUST reject foreign views.
8. A backend factory MUST leave no partially usable device/context behind when
   initialization fails. Enabled hardware is never an implicit CPU fallback.

### 2. Tensor owner and allocation contract

1. A backend `Tensor` is materialized only through its creating `Device`. It
   MUST initialize the common `Tensor` base with the requested spec and that
   device, expose one stable full view through `view()`, and remain
   non-copyable/non-movable.
2. The tensor's opaque `native_handle()` MUST remain stable for the full owner
   lifetime. A backend MUST NOT relocate storage because a view is made,
   transformed, transferred, or submitted to a queue.
3. The creating `Device` and its borrowed allocator MUST outlive every tensor
   and queue they create. A caller must retain tensor owners until all work
   accessing their storage is complete. A backend MUST snapshot every view
   metadata field it needs before `copy` returns: a derived `TensorView`
   temporary may be destroyed before the caller waits. A backend MUST make its
   own destruction safe when it owns outstanding runtime state; it MUST NOT
   free or reuse a resource the runtime can still access.
4. Allocator-backed standard storage MUST allocate exactly the checked
   `spec.tiled_storage_nbytes()` amount and require a 32-byte-aligned base
   address. A null allocation is `std::bad_alloc`; a returned misaligned
   address MUST be released and reported as an error.
5. Accelerator allocators may have stricter validity rules. CUDA allocations
   must be unmanaged device allocations from the exact owning context/ordinal;
   ROCm allocations must be unmanaged device allocations from the exact
   ordinal; SYCL allocation pointers must be known to the owned context.
   Validate these rules before accepting storage.
6. Native storage may be runtime-owned and have a different physical byte
   count, but it MUST remain associated with the same public spec/device/view
   contract. It MUST validate native extent conversion and leading-plane count
   overflow before creating native resources.
7. Operations, view transformations, and host transfers MUST NOT allocate or
   free operand or output tensor storage. Resource pools may allocate internal
   staging/metadata capacity, but they must not replace, retarget, or move a
   user tensor.

### 3. Logical representation and physical layout

1. The public shape has a minimum rank of two. All leading dimensions are
   row-major logical planes. The last two dimensions are a matrix stored in
   fixed 16x16 tiles for the standard representation.
2. Standard physical slot order is: row-major leading plane, tile row, tile
   column, row within tile, column within tile. The final axes are padded up to
   multiples of 16; all leading axes retain their logical extents. Standard
   storage size includes that padding. A newly created standard allocation, and
   every region not targeted by a later view write, MUST remain zero-filled;
   writing a view MUST leave padding and all other owner planes unchanged.
3. The logical host representation is contiguous row-major elements packed
   least-significant-bit first at each `DataType` width. Multi-byte fields are
   little-endian. Transfers MUST move bit patterns, not numerically convert
   values. Boolean host input bytes MUST be canonical `0` or `1`.
4. Upload scatters logical bytes into the backend's physical layout; download
   gathers them back to exactly the logical byte stream. Padding is not part of
   the host payload. A standard-layout backend MUST preserve zero padding; a
   native backend MUST ensure native-only padding cannot leak into logical or
   canonical observed storage.
5. The shared physical oracle re-derives the canonical slot mapping without
   calling production layout code. An accelerator/native conformance driver
   MUST implement `AcceleratorStorageOracle::seed` and `observe` with an
   independent native access path. `observe` returns the complete canonical
   padded owner allocation, not only logical view bytes, so it detects wrong
   tile maps, nonzero padding, and writes to untouched owner planes.

### 4. View and transfer contract

1. `TensorView` is a non-owning, copyable but non-assignable window into one
   tensor owner. Its plane offset and plane strides count whole logical planes,
   never bytes or elements. `native_handle()` remains the owner handle and
   does not retain ownership.
2. Only leading dimensions may be transformed. `slice`, `select`, `permute`,
   and `reshape_leading` are common metadata operations that preserve the
   owner. Their output must be honored by every backend transfer and copy.
   The final two matrix dimensions MUST NOT be transformed.
3. A transformed view maps its leading logical indices through its plane offset
   and strides. A backend MUST support full, offset, stepped, selected,
   permuted, nested, and contiguous-reshaped views; it MUST leave all planes
   outside the destination view untouched.
4. `copy_from_host` and `copy_to_host` are synchronous and take exactly
   `view.spec().logical_nbytes()`. A short or long source/destination span
   MUST throw `std::invalid_argument` before altering host read buffers. A
   noncanonical BOOL upload MUST throw `std::invalid_argument`.
5. A failed host transfer MUST leave the view's spec, plane offset, plane
   strides, device identity, and native handle unchanged. Destination value
   bytes after a failed transfer are otherwise unspecified.
6. Host transfers do not implicitly wait for queues. The caller must wait for
   outstanding writes before a host read, and for every outstanding read/write
   before a host write or owner destruction. A backend MUST document/implement
   no hidden relaxation of this ordering rule.

### 5. Queue and token contract

1. `DeviceOps` is non-copyable/non-movable and represents one in-order queue.
   The caller serializes calls on a queue. The backend may synchronize or use
   asynchronous runtime work internally, but observable completion and
   visibility order is submission order.
2. Every successful submission uses the common facade and returns a positive
   `oid = static_cast<oid>((std::uint64_t{q} << 55) | sequence)`. Queue ID `q`
   is process-unique in the complete range `[1,255]`, represented in bits
   `55..62`; `sequence` is exactly 55 bits, begins at one, and is in
   `[1,2^55-1]`. Sequence zero is never submitted.
3. OID facades are common `noexcept` entry points. They validate, map errors,
   encode tokens, and register lifetimes before backend effects or acceptance;
   protected backend hooks cannot bypass those obligations. Synchronous
   failures return exactly `InvalidArgument=-1`, `Unsupported=-2`,
   `Overflow=-3`, `ResourceExhausted=-4`, `DeviceError=-5`, or
   `InternalError=-6`. Invalid input, unavailable operation/specification,
   checked arithmetic or sequence exhaustion, bounded-resource failure,
   pre-acceptance runtime failure, and otherwise unclassified failure map to
   those values respectively. No synchronous exception crosses an OID facade.
4. Sequence exhaustion returns `Overflow` before backend work, effects, or
   token acceptance. Every accepted submission returns one positive token.
   Failures after acceptance remain attached to that token and `wait` rethrows
   them repeatably; successful completion and visibility are repeatable.
5. `wait(token)` blocks for accepted work and immediately throws
   `std::invalid_argument` for a negative value, zero, a foreign queue token,
   a future value, a skipped/reserved-but-never-submitted value, or any other
   unsubmitted value. A skipped value remains invalid after later completion.
   Completing sequence *n* makes earlier submitted work observable as complete.
6. The common base releases a queue ID at destruction; it does not define an
   implicit wait or cancellation of outstanding work. A concrete backend MUST
   nevertheless perform runtime cleanup preventing use-after-free of streams,
   events, storage, and lifetime registrations. Callers must not use a token
   or queue after its owner is destroyed.
7. Backend queues MUST retain enough state to complete and fail submitted work
   correctly even if their worker observes a runtime error after launch. A
   completion proof can be a runtime event, successful stream drain, or
   equivalent native fence. If no proof is available, affected storage and
   metadata MUST be quarantined/retired rather than reused.

### 6. Memory, admission, and workspace contract

For CUDA, ROCm, and SYCL, setup MUST reserve exactly two IOM-native backing
allocations: the caller-selected tensor-data arena and a separate metadata
arena with checked capacity `4 * C * 512` bytes. `C` is the immutable,
nonzero `QueueConfig::max_in_flight_per_queue`; the metadata arena is
device-wide and one `FixedSizeAllocator` spans exactly `4 * C` 512-byte,
32-byte-aligned slots. The data arena uses one device-owned coalescing
allocator. Tensor and `RawWorkspace` addresses MUST stay in the data domain;
descriptor and metadata-slot addresses MUST stay in the metadata domain.

An arena suballocation changes allocator bookkeeping only; it is not a native
allocation. A metadata-slot lease, completion resource, host allocation,
caller workspace range, and accepted token are separate ownership records.
After setup, operations, transfers, views, waits, retirement, queue
recreation, parking, and failure recovery MUST make no additional IOM-native
device allocation/free calls and MUST NOT grow or resize fixed resource
arrays. Vendor/SDK-internal allocations are outside that IOM boundary and
MUST be reported separately rather than treated as IOM evidence.

Every accepted request retains its immutable host snapshot, token outcome,
owner registrations, and workspace lease. At most `C` requests per queue hold
native credits; later requests are host parked with no native effect and are
dispatched strictly FIFO when a completion proof returns a credit. No-op,
inline, and no-metadata requests use the same admission order and MUST NOT
bypass a parked head. Completion resources, metadata leases, and workspace
ranges are reusable only after proven completion. Unknown native use MUST
quarantine the complete unresolved lease, including the queue partition and
queue-count reservation, until that lease's own covering proof.

Positive workspace requirements MUST be queried through the pure operation or
transfer query before submission. A missing, undersized, misaligned, stale,
foreign, overlapping, or already-leased range is invalid or resource
exhausted as specified by the public facade; the caller retains ownership
through proven completion. CPU retains borrowed host/reference storage
without a device metadata arena or fabricated native slots. TTNN retains
native per-plane tensor ownership and host-only scratch; it MUST NOT claim the
standard-GPU two-backing guarantee, and vendor-internal runtime allocation is
unproven where it cannot be observed.

The four-live-queue cap and quota are local to the exact `Device`; there is no
global ordinal cap, active-backend registry, or queue selector. Two-/eight-GPU
isolation and matched-baseline first-use/warmed performance are
environment-dependent evidence only: record them when matching hardware and
baseline exist, otherwise report the unavailable result as a non-universal
risk. Report serialized host-transfer throughput separately from compute
throughput. This repository has no `examples/` directory, so caller audits
must record that fact rather than adding an example.

### 7. Copy contract

1. `copy(source, destination)` is required on every backend. It MUST call the
   equivalent of `DeviceOps::validate_copy` before reserving a sequence or
   causing writes: both views must belong to the queue's exact `Device`, and
   their specs must compare equal (shape, leaf type, quantization).
2. Shape/type/quantization mismatch and reference/foreign-device views MUST
   return the negative `InvalidArgument` OID result. A rejected copy MUST not
   write the destination and MUST not consume a token; the next valid copy
   receives the next unused sequence.
3. A valid copy maps every logical element from the source view to the matching
   logical element of the destination view. It MUST honor both independent
   plane offsets and strides; source and destination may be full, offset,
   stepped, selected, permuted, or nested views.
4. Same-queue copies execute in call order without an intervening host wait.
### 8. Binary elementwise contract

1. `add`, `mul`, `sub`, and `div` each have exactly the common non-virtual
   signature `oid op(const TensorView&, const TensorView&, TensorView&) noexcept`.
   Positive OIDs accept work; synchronous rejection returns one of the six
   negative `OidError` values. There are no options, promotion, public query,
   fallback selector, or signature knobs.
2. Validate in this order before effects, owner registration, sequence
   consumption, or token acceptance: recognized specs/rank/dim/device/owner/
   handle/view/storage and checked arithmetic; matching leaf/quantization;
   right-aligned broadcasting/output shape; mapping snapshot; exact alias rule;
   operation support. Mismatched recognized leaves or quantization are
   `InvalidArgument`; malformed view/device/shape/alias errors are
   `InvalidArgument`; checked arithmetic is `Overflow`; bounded resources,
   pre-acceptance runtime, and other failures map to `ResourceExhausted`,
   `DeviceError`, and `InternalError`.
3. Ranks below two are invalid. `[1,1]` broadcasts over all output axes (two
   scalars yield `[1,1]`); otherwise ranks right-align with conceptual leading
   ones, each axis equal or one, and output exactly the maximum shape.
   Singleton coordinates, including tiled tails, map to zero before tile-slot
   mapping; padding is never read. Transformed leading views preserve offsets
   and strides; broadcasting is internal, not a public zero-stride view.
4. ADD, MUL, and SUB support the 21 NONE numeric leaves listed in section 1;
   DIV supports the nine NONE floating leaves listed there. Matching BOOL,
   F8_E8M0, non-NONE quantization, and integer DIV return `Unsupported` only
   after earlier checks. Operand order is observable: SUB is lhs-rhs and DIV
   is lhs/rhs.
5. Integer MUL and SUB return low `w` bits modulo `2^w` with two's-complement
   signed or ordinary unsigned interpretation, without signed-overflow UB.
   Floating operands decode in their named format, compute in the extended
   mathematical/IEEE domain, and encode once with RNE. Gradual underflow and
   no FTZ/DAZ are required. MUL zero×infinity and NaN are NaN; SUB same-sign
   infinities and NaN are NaN; DIV NaN, 0/0, and infinity/infinity are NaN.
   Format-specific saturation and infinity/NaN classes follow the scalar
   reference; finite outputs are within one ULP.
6. Read/read overlap is valid. Same-owner input/output is allowed only for an
   exact unbroadcasted alias with identical spec, offset, strides, and mapping;
   all other relationships are rejected. Capture both inputs before each store,
   snapshot metadata rather than views, and register all three distinct owners
   with exact aliases deduplicated.
7. Binary work is in-order asynchronous (CPU may complete inline), repeat
   waits are valid, and accepted failures are retained and rethrown on every
   wait. Invalid waits throw `std::invalid_argument`. Pre-submit errors do not
   mutate output, consume a sequence, or register an owner. No caller operand/
   output storage is allocated, replaced, or relocated; staging, conversion,
   workspace, and emulation are internal. Accepted failures are not retried.

### 9. Other compute capabilities

`silu` and `sdpa` remain unsupported and return negative `Unsupported` before
submission, mutation, or token acceptance. The `linear` hooks are owned by
[Linear projections](#linear-projections): CPU, CUDA, ROCm, and SYCL implement
all twenty-one applicable leaves — the twenty non-BF16 leaves on the shared
scalar projection path plus `BF16` on the scalar recurrence on CPU and on a
separate native specialization on CUDA, ROCm, and SYCL, whose availability is a
runtime device and loaded-image fact — while TTNN implements the mandatory
`BF16` leaf alone on its direct Metalium route and explicitly rejects the other
twenty applicable leaves after structural validation. A backend without its own
linear port keeps reporting `Unsupported`. CUDA and ROCm
provide source-inspected RMSNorm launch wrappers over the shared core, and
TTNN now provides its preallocated BF16/F32 queue path; only backends without
one of those ports keep reporting RMSNorm `Unsupported`.

#### TinyLlama forward layout — Embedding and projection boundaries

This subsection freezes the backend-neutral boundaries and records the
implemented embedding ports. The embedding methods below are declared and
admitted by the current neural facade: common structural, device, view, shape,
alias, checked-arithmetic validation, and the pure requirement query are
implemented, while CPU, CUDA, and ROCm provide their operation hooks. SYCL and
TTNN remain explicitly capability-gated according to their own port state.
The normative `linear` surface is exactly the frozen `LinearOutputLayout` form
below, owned by [Linear projections](#linear-projections): there is no second
`linear` declaration, and the earlier three-view `linear(x, w, y)` facade is
not retained as an overload, alias, shim, or re-export. Until a backend port
lands, a well-formed request for that frozen ABI returns `Unsupported` as
specified above.
The operation-owned [Embedding lookup](#embedding-lookup) and
[Linear projections](#linear-projections) sections and their backend gates
publish the exact operation ABI and remain the normative source for per-backend
capability, workspace, status, and failure policy. The embedding declarations
and the frozen `LinearOutputLayout` and `linear` methods are exactly:

```cpp
enum class LinearOutputLayout { ordinary, head_planar };

oid embedding(
        const TensorView& table, const TensorView& indices, TensorView& out,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements embedding_workspace_requirements(
        const TensorView& table, const TensorView& indices,
        const TensorView& out);

oid linear(
        const TensorView& x, const TensorView& w, TensorView& out,
        size_t s, size_t R, LinearOutputLayout layout, size_t H, size_t D,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements linear_workspace_requirements(
        const TensorView& x, const TensorView& w, const TensorView& out,
        size_t s, size_t R, LinearOutputLayout layout, size_t H, size_t D);
```

Both requirement queries receive the same semantic arguments as submission,
including a const output view, but no workspace. They are pure: they allocate
nothing and perform no registration, submission, queue/OID consumption, or
state change. They validate operands, shapes, scalar parameters, aliases,
capability, and checked arithmetic, then return `WorkspaceRequirements` or
throw the established validation exception. Exact-device, size, alignment,
overlap, staleness, and lease checks on a supplied workspace occur only during
submission. The `noexcept` submission facades instead map synchronous failures
to OID errors and otherwise return a positive accepted token.
Synchronous rejection has no output effect, registration, sequence
consumption, or accepted token. A failure after acceptance remains attached to
its positive token and is rethrown on every repeat wait.

For embedding, the table is rank-two `[V,F]` and is shared unchanged by every
independent leading plane. `indices[...,1,R]` and `out[...,R,F]` have the same
leading tuple and each has rank 2 through 8. There is no implicit
index-to-output plane broadcast and no singleton output-rank inflation. With
`b` denoting the complete leading tuple, the mapping is

```text
out[b,r,f] = table[indices[b,0,r],f].
```

Runtime shapes, never checkpoint constants, determine `R`, `V`, and `F`.
Indices use the caller-selected integral tensor index type. A signed negative
index or an index at least `V` is invalid. A value discoverable only in device
storage may instead fail after acceptance as a device-data error; an
implementation MUST NOT introduce a host round trip or hidden host-side index
scan. The selected table payload bits are copied unchanged, with no numeric
conversion or rounding. Index representation, applicable leaves, fixtures,
tolerances, workspace and status protocol, ownership and failure rules, and
backend support are owned by [Embedding lookup](#embedding-lookup).

For linear projection, `x` is `[...,T,I]` and rank-two `w[O,I]` is shared
unchanged across all independent leading planes. This is the Hugging Face
`[out,in]` weight orientation: output coordinate `o` selects weight row
`w[o,*]`; the boundary never transposes a checkpoint weight. Ordinary layout
produces `out[...,R,O]` and requires `H=1,D=O`. Head-planar layout produces
`out[...,H,R,D]` and requires the checked equality `O=H*D`. Input and output
leading tuples must match after excluding the inserted head axis; the
rank-two weight's empty leading tuple is not matched and does not imply state
broadcast. The final output rank cannot exceed 8. Thus an ordinary rank-8
output is structurally valid, while adding a head axis that would make a
rank-9 head-planar output is `InvalidArgument`.

The ordinary and head-planar equations are, respectively,

```text
out[b,r,o]   = sum_i x[b,s+r,i] * w[o,i]
out[b,h,r,d] = sum_i x[b,s+r,i] * w[h*D+d,i].
```

Here `T` is the source row extent, `s` is its starting row, and `R` is the
independently requested row count; `R` is neither inferred from `s` nor
confused with `T` or `O`. Likewise `O` is the weight's output-row extent and
equals `H*D` only after checked validation in head-planar mode.

The normative non-square case is
`I=3,O=10,H=2,D=5,T=19,s=2,R=17`. It selects exactly `x[2]` through
`x[18]`. Ordinary output columns `o=0..9` consume `w[0]` through `w[9]`.
Head-planar `(h=0,d=0..4)` maps to `o=0..4`, and
`(h=1,d=0..4)` maps to `o=5..9`. With all other values unchanged, `R=1`
selects `x[2]`, `R=15` selects `x[2..16]`, and `R=16` selects
`x[2..17]`. These cases make a checkpoint transpose, an inferred or
off-by-one row window, and an `h`/`d` shuffle observably wrong.

The final untied LM head uses ordinary mode with `s=T-1`, `R=1`, `H=1`, and
`D=V`. For one sequence it writes `[1,V]` logits directly into the supplied
output; it requires no final-axis `TensorView` slice. Projection rearranges
only newly computed selected rows. It MUST NOT materialize repeated KV heads,
make a persistent host copy or transpose of checkpoint weights, or expose a
public transpose, head-pack, or row-extraction operation.

All ranks remain in 2 through 8 and all dimensions are nonzero. Embedding and
linear validate, in the common precedence and before effects: well-formed
views, ranks, shapes, leading mappings, exact device ownership, leaf type and
quantization; scalar and range rules; aliasing and capability; then checked
element, stride, byte-range, output-size, and other shape products. Binary and
embedding share one bounded, allocation-free checked-view admission path for
the common structural facts: recognized encodings, rank and nonzero extents,
exact live owner and stable native handle, leading-only view metadata,
selected-plane bounds, and checked plane, tile, element, bit, and byte
arithmetic. It allocates, snapshots, registers, and leases nothing, and
shaping, capability, alias, and workspace policy stay with each operation.
Linear additionally validates a recognized `layout`, `s <= T`, `R > 0`,
`R <= T-s`, `H > 0`, and `D > 0`; head-planar mode checks `H*D` for overflow
before comparing it with `O`, while ordinary mode requires `H=1,D=O`.
Logical mapping excludes all 16x16 tile padding and honors every transformed
leading offset and stride independently.

The rejection results are normative:

| Condition | Result |
| --- | --- |
| malformed view, rank outside 2..8, zero dimension, mismatched leading plane, invalid final shape, or invalid row window | `InvalidArgument` |
| signed negative index or index `>= V` | `InvalidArgument`, or an accepted device-data failure when the value is discoverable only on device |
| unsupported recognized dtype, device, or quantization | `Unsupported` after earlier validation |
| invalid `LinearOutputLayout` value | `InvalidArgument` |
| output/input or workspace overlap, including forbidden transformed-owner overlap | `InvalidArgument` |
| rank growth beyond 8 | `InvalidArgument` |
| checked element, stride, byte-range, output-size, or `H*D` overflow | `Overflow` |
| bounded-resource, runtime/device, or internal submission failure | `ResourceExhausted`, `DeviceError`, or `InternalError` |

Inputs may overlap other inputs: valid read/read overlap and exact aliases are
not rejected merely because both accesses are reads. New output and workspace
storage must each be disjoint from every input and from each other; these
gather/projection operations reject every output/input overlap. Workspaces are
caller-owned reusable scratch subject to the shared exact-device, nonoverlap,
size, alignment, and lifetime rules. The operations allocate, relocate, or
silently convert no operand or output. Submission snapshots metadata,
registers distinct owners through the existing queue protocol, and retains
the workspace lease through proven completion; it never retains borrowed
`TensorView` objects beyond submission.

Every result is stored in the caller-provided output leaf. Embedding preserves
payload bits; each distinct projection result is rounded and stored at the
operation boundary according to [Linear projections](#linear-projections).
BF16 is mandatory for CPU, CUDA, ROCm, SYCL, and TTNN when those ports are
implemented. The operation-owned sections classify datatype leaves, special
values, numerical references, tolerances, fixture provenance, and backend
feasibility. Delivery order is contract/reference/all-five feasibility, then
CPU, CUDA, ROCm, SYCL, and TTNN, closing the five-backend gate before the next
operation. This planned boundary makes no kernel, profiler, or native-backend
conformance claim.

Conformance records must distinguish shape failures, plane/leading mismatch,
invalid row windows, dtype/device/quantization support, output/workspace
overlap, rank growth, and checked arithmetic overflow. They cover independent
leading planes, transformed mappings, logical padding exclusion, and the
`1/15/16/17` row cases. Unsupported probes migrate only when their operation
is implemented; an unsupported port never counts as numerical conformance.
#### TinyLlama forward layout — Dtype and numerical policy

This subsection defines semantic applicability and shared numerical boundaries
for embedding lookup, linear, RMSNorm, RoPE, cache-row append, SiLU, and SDPA.
It does not advertise a current implementation: the neural hooks remain
`Unsupported` as stated above until an actual backend port changes that
capability. The binary rules in section 8 remain unchanged and are not
restated here.

##### Semantic applicability

In this table, “payload” means embedding-table and embedding-output data, and
cache means cache-row copy data. “ID” means an embedding index. “Linear” is a
general signed numeric matrix product. “Real arithmetic” comprises RMSNorm,
RoPE, SiLU, and SDPA. `applicable` is a semantic classification, not a claim
that every backend currently stores or computes that leaf.

| Leaf | Embedding payload / cache append | Embedding ID | Linear | RMSNorm / RoPE / SiLU / SDPA | Semantic reason |
| --- | --- | --- | --- | --- | --- |
| `BOOL` | applicable | inapplicable | inapplicable | inapplicable | Payload bits can be copied; boolean is neither an integer token ID contract nor a real-valued arithmetic result. |
| `I2` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic; real-valued normalization, trigonometry, activation, and attention are not defined. |
| `U2` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic; there is no implicit conversion to real arithmetic. |
| `I4` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U4` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I8` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U8` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I16` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U16` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I32` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U32` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `I64` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `U64` | applicable | applicable | applicable | inapplicable | Integer payload/ID and linear arithmetic only. |
| `F4_E2M1` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed low-precision real value for numeric operations, never an ID. |
| `F6_E2M3` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed low-precision real value, never an ID. |
| `F6_E3M2` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed low-precision real value, never an ID. |
| `F8_E4M3FN` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed finite real format, never an ID. |
| `F8_E5M2` | applicable | inapplicable | applicable | applicable | Bit-preserving payload; ordinary signed real format, never an ID. |
| `F8_E8M0` | applicable | inapplicable | inapplicable | inapplicable | Payload bits can be preserved, but an unsigned exponent-only encoding is not a general signed numeric result. |
| `F16` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and ordinary signed real arithmetic. |
| `BF16` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and the mandatory TinyLlama real-arithmetic/storage path. |
| `F32` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and ordinary signed real arithmetic. |
| `F64` | applicable | inapplicable | applicable | applicable | Bit-preserving payload and ordinary signed real arithmetic; it is not silently narrowed to FP32. |

Embedding payload/cache copy therefore has all 23 applicable leaves.
Embedding IDs have exactly the 12 integer leaves (`I2`, `U2`, `I4`, `U4`,
`I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and `U64`): negative signed
IDs and every ID greater than or equal to vocabulary size `V` are invalid,
while `BOOL` and every floating leaf are inapplicable. Linear has those 12
integer leaves plus exactly the nine ordinary signed floating leaves
(`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
`F32`, and `F64`). RMSNorm, RoPE, SiLU, and SDPA have exactly those nine
floating leaves. `F8_E8M0` MUST NOT be promoted to a general signed numeric
result.

Semantic applicability is independent of native storage or arithmetic
capability. The 23 stored leaves on the standard CPU, CUDA, ROCm, and SYCL
paths, and TTNN's 22 stored leaves excluding `F8_E8M0` plus its encoded-carrier
limits, are capability evidence rather than permission to narrow this table.
Each operation contract MUST enumerate a backend implementation or a justified
limitation for every applicable dtype. BF16 weights, activations, and caches
are mandatory for TinyLlama on all five backends: CPU, CUDA, ROCm, SYCL, and
TTNN. Integer linear accumulation, overflow behavior, and any conversion
before a kernel belong to the [Linear projections](#linear-projections)
contract; this table does not infer integer normalization or silently mean
“all floats.” Only `QuantizationFormat::NONE` is applicable; every other
quantization format is rejected.

##### Storage, rounding, masking, and nonfinite values

BF16 storage boundaries MUST use round-to-nearest, ties-to-even (RNE) at each
operation boundary. Embedding and append are bit-preserving copies and MUST
NOT re-encode their payloads. For BF16 operands:

- linear and normalization reductions MUST use FP32 or a demonstrably
  sufficiently wide equivalent;
- RMSNorm MUST multiply by its scale in the wide domain before rounding the
  stored result once;
- SiLU MUST evaluate stably in the wide domain and store BF16 once;
- RoPE MUST evaluate its angle, trigonometric values, and rotation
  intermediates in the wide domain and store BF16 once; and
- SDPA MUST compute QK/scale and stable masked softmax in FP32, round
  probability `P` to BF16 with RNE before a native BF16 `P V`, accumulate PV
  in FP32, and round its output to BF16.

The SDPA rule is not a guarantee that attention uses exact-FP32 `P`, and it
MUST NOT impose a BF16 probability-storage boundary on other
operation-supported dtypes. Precision rules for those dtypes belong to their
operation contracts; in particular, `F64` MUST NOT be narrowed to FP32.

The following cases are normative checks on the operation boundaries:

1. Around one, adjacent BF16 values are `1.0` and `1.0 + 2^-7`. Their midpoint
   is `1.0 + 2^-8`; RNE chooses `1.0` because the lower significand is even.
2. The midpoint between `1.0 + 2^-7` and `1.0 + 2*2^-7` is
   `1.0 + 3*2^-8`; RNE chooses the upper value because its significand is
   even.
3. The softmax probability `0.501953125` (`0.5 + 2^-9`) is exactly halfway
   between BF16 `0.5` and `0.50390625`; RNE stores `0.5`.
4. The probability `0.505859375` (`0.50390625 + 2^-9`) is exactly halfway
   between BF16 `0.50390625` and `0.5078125`; RNE stores `0.5078125` because
   the upper significand is even.

A candidate that first rounds through a BF16 intermediate where this policy
requires FP32 is non-conforming. In SDPA, omitting the explicit rounded-BF16
`P` boundary before a BF16-native PV is also non-conforming. Conversely, the
BF16 destination does not permit rounding QK, the softmax sum, or every FP32
accumulation.

Ordinary floating values follow their mathematically defined IEEE classes.
Masking MUST exclude inaccessible values entirely; admission MUST NOT scan
generically for nonfinite values or force a host transfer. Before kernels are
implemented, the owning operation contracts MUST fix nonfinite tensor
behavior, exact SiLU infinity and signed-zero rules, finite-format
saturation/NaN handling, the supported RoPE position/trigonometric domain, and
per-dtype tolerances. TinyLlama model parameters require positive finite
epsilon and finite positive theta. RMSNorm with `eps=0` may yield NaN for a
zero row, and the final selector rejects every otherwise-valid nonfinite logit.
No NaN payload equality is promised.

Format-specific finite overflow, saturation, signed zero, and representable
special-value classes remain owned by each operation's numerical contract.
Implementations MUST NOT invent NaN or infinity encodings for finite-only
formats and MUST reuse the existing named-format encoding rules. `F64` and
other dtypes MUST NOT be forced through FP32.

##### Reference ownership and coverage

Every future detailed numerical reference, fixture provenance, tolerance, and
snapshot/hook rule has exactly one owner:

- 03 embedding owns bit-preservation and index cases;
- 04 linear owns numeric, integer/orientation, and native-GEMM cases;
- 05 RMSNorm owns zero, scaled, reduction, and epsilon cases;
- 06 RoPE owns split-half positions and trigonometric cases;
- 07 append owns preservation and bounds;
- 08 SDPA owns GQA, causal masking, and probability rounding;
- 09 SiLU owns extremes; and
- 14 integration-reference-validation owns independent model and
  intermediate-logit fixtures.

Each owner MUST select fixed, justified tolerances and pin the reference
software or artifact identity before measuring a candidate. A production
implementation MUST NOT serve as its own oracle. This subsection adds no
duplicate fixture or kernel.

Future shared coverage MUST exercise logical `R=1,15,16,17`, non-tile feature
sizes, rank and planes, independent logical/tiled encodings, padding and tail
perturbations, causal future-token exclusion, cached/full equivalence, exact
capacity, and accepted/rejected failures. Exact token goldens are appropriate
only when the independent reference states its margin. All-five gates and
existing supported-operation coverage remain required. Unsupported neural
probes migrate only when a backend is actually ported and do not constitute
numerical conformance.
#### TinyLlama forward layout — Normalization and MLP boundaries

The following interfaces are the planned ABI of this subsection. The RMS
normalization declarations and their admission contract are frozen and
declared by [RMS normalization](#rms-normalization); the SiLU declarations are
still planned and undeclared:

```cpp
oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rmsnorm_workspace_requirements(
        const TensorView& x, const TensorView& scale,
        const TensorView& out, float eps);

oid silu(const TensorView& x, TensorView& out,
         RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements silu_workspace_requirements(
        const TensorView& x, const TensorView& out);
```

The RMS normalization cutover removed the redundant `dim` argument from
declarations, definitions, callers, tests, and documentation; no overload,
alias, compatibility shim, or re-export survives. The last logical axis is
the feature width. For `x` and `out` shaped `[...,R,F]`, `scale` MUST have
exactly `[1,F]`; it is shared explicitly across every row and independent
leading plane, without introducing a general hidden-state or leading-plane
broadcast. The loader alone adapts a checkpoint normalization vector `[H]` to
the public rank-two `[1,H]` scale. RMSNorm MUST NOT accept a rank-one scale.

For every logical leading-plane index `b`, row `r`, and feature `f`, RMSNorm is
row-local:

```text
out[b,r,f] = round(
    x[b,r,f]
    * rsqrt(sum(i=0..F-1, x[b,r,i] * x[b,r,i]) / F + eps)
    * scale[0,f])
```

The reduction includes exactly the `F` logical features: tiled padding, other
rows, and other planes MUST NOT contribute. `eps` MUST be finite and
nonnegative; the model configuration supplies a finite positive value. With
`eps == 0`, an all-zero row produces a quiet NaN, with no NaN-payload promise.
For the model's BF16 boundary, squares, reduction, reciprocal square root, and
scale multiplication use FP32 or a demonstrated sufficiently wide equivalent,
then the result is rounded once to the output leaf.

SiLU requires rank 2 through 8, identical input/output shapes, and independent
leading planes. It computes stable `x * sigmoid(x)` as one wide computation and
rounds once to the output leaf. The MLP MUST compose that stored result with
the existing binary operation as
`mul(SiLU(gate), up, product)`—in that operand order—and MUST NOT add a fused
SiLU-times-multiply operation.

Each requirement query returns `WorkspaceRequirements`, is deterministic and
pure, and has the same semantic arguments as its operation except for a const
output and no workspace argument. It performs operand, shape, scalar, alias,
capability, and checked-arithmetic validation only. It MUST NOT allocate,
register owners, acquire a lease, reserve or consume a sequence, submit work,
read queue occupancy or arena availability, mutate output, or otherwise change
state. Query validation errors are standard exceptions. The eventual
`noexcept` operation facades instead map pre-acceptance errors to the common
negative OID categories.

Admission MUST finish before work, owner registration, sequence consumption,
or output mutation. It validates ranks 2 through 8, nonzero dimensions,
leading tuples, exact device identity, dtype and quantization, aliases,
workspace device/size/alignment/nonoverlap, scalar finiteness, and checked
shape/range/element/byte/address arithmetic. Neural outputs are disjoint from
their inputs and scratch. Submission snapshots metadata rather than retaining
borrowed views, registers all required owners and workspace through completion,
and preserves repeated-wait observation of an accepted device-data failure.

Within each decoder layer, normalization and MLP boundaries are:

1. store RMSNorm of residual `X [R,F]` in distinct `N [R,F]`;
2. after attention output projection stores `B [R,F]`, use existing
   `add(X,B,X2)` to store the first residual `X2 [R,F]`;
3. store RMSNorm of `X2` in distinct `N2 [R,F]`;
4. independently store gate and up linear outputs `Gate [R,M]` and
   `Up [R,M]`;
5. store `SiLU(Gate)` in `ActivatedGate [R,M]`;
6. use existing `mul(ActivatedGate,Up,Product)` to store `Product [R,M]`;
7. store the down projection in `Down [R,F]`; and
8. use existing `add(X2,Down,NextX)` to store the second residual and next
   layer input `NextX [R,F]`.

Every numbered boundary has its own BF16 result store and operation-boundary
rounding; fusion MUST NOT erase either residual, either normalization boundary,
or the SiLU/multiply boundary. Existing `add` and `mul` broadcasting, alias,
workspace, queue, and numerical rules remain unchanged. Names denote logical
values, not a requirement to retain one persistent activation bank per layer:
storage MAY be reused only after all readers and accepted work have completed.
Each layer's K and V caches remain distinct persistent owners.

For the worked non-tile case `F=8`, `M=12`, `Hq=4`, `Hkv=2`, and `D=2`,
`F=Hq*D`. Each of `R=1,15,16,17` therefore has `X/N/N2/B/Down/NextX [R,8]`,
Q `[4,R,2]`, K/V `[2,R,2]`, merged attention `[R,8]`, and
`Gate/Up/ActivatedGate/Product [R,12]`. Consumers match those logical shapes
for all four runs; padding is not a logical row or feature. A conformance
example with leading extent `P` applies the same mapping independently:
`[P,R,8]` produces Q `[P,4,R,2]`, K/V `[P,2,R,2]`, and MLP intermediates
`[P,R,12]`. It does not broadcast state between planes or transform either
final tensor axis.

Setup MUST use checked arithmetic and a deterministic sizing order covering
embedding output, per-layer normalization and Q/K/V projections, RoPE,
separate bounded K/V caches, SDPA and maximum operation workspace, attention
projection, both residual stores, all MLP intermediates, final normalization,
and one-row logits. It keeps cache capacity and maximum workspace explicit,
does not duplicate persistent checkpoint weights, and never allocates or
relocates an operand or result inside an operation. Actual allocation,
initialized cache length, reset/growth/failure policy, and selector invocation
belong to the session component; operation-local minimum scratch belongs to
the corresponding operation contract.

The detailed datatype applicability, special-value behavior, references,
tolerances, fixture provenance, snapshots/hooks, kernels, and evidenced
backend limitations remain solely owned by the
[RMS normalization](#rms-normalization) and planned **SiLU** operation
sections. Until each operation's backend ports land, the `rmsnorm` and `silu`
calls continue to return negative `Unsupported` before submission, mutation,
or token acceptance, and this plan subsection declares no ABI of its own.

Delivery is operation-first, not mathematical-forward order: embedding,
linear, RMSNorm, RoPE, cache append, SiLU, and finally SDPA. For each operation,
finish its contract, independent reference, and all-five feasibility record,
then close CPU, CUDA, ROCm, SYCL, and TTNN in that order before starting the
next operation. The CPU baseline is scalar/wide arithmetic over existing tiled
storage, not native-accelerator evidence. CUDA/ROCm/SYCL/TTNN native BF16
matrix feasibility for linear and SDPA prefill and logical `R=1` MUST be
evidenced before those implementations; host computation, round trips,
elementwise substitutes, and padded extra logical tokens do not count.
Unavailable ports remain reported as unsupported and do not count as
numerical conformance. Future all-five conformance covers success, runs
`1/15/16/17`, non-tile features, independent planes, padding isolation,
rejections, and repeatable accepted failures. Only after SDPA closes may the
session assemble and verify the complete layer from these boundaries.
#### TinyLlama forward layout — Positions and cache boundaries

The common layer now declares the RoPE facade and freezes its
backend-neutral admission contract. It does not add a kernel or claim positive
support: until a backend leaf replaces the protected hooks, a valid request
returns `Unsupported` and the pure query throws the same category.

```cpp
oid rope(
        const TensorView& x, TensorView& out, std::size_t a, double theta,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rope_workspace_requirements(
        const TensorView& x, const TensorView& out, std::size_t a,
        double theta);
```

The two facades have exactly the same semantic arguments in the stated order.
`a` is an explicit absolute position and `theta` is an explicit runtime
scalar; neither is inferred from a session, cursor, cache, or model state.
`x` and `out` MUST have identical logical shape `[...,H,R,D]`, rank three
through eight, identical nonzero leading dimensions, `H > 0`, `R > 0`, and
positive even `D`. Every logical output element is written from its
corresponding input under the two views' independent valid offset and
leading-stride mappings. Tile padding and uninitialized physical slots are not
logical values. Q and K are separate calls, so unequal Q/K head counts are
valid and there is no cross-request Q/K alias category.

For every leading coordinate tuple `b`, head `h`, run index `r`, and
`0 <= j < D/2`, the frozen split-half equation is:

```text
angle = (a + r) * theta^(-2*j/D)
y[b,h,r,j]       = x[b,h,r,j]       * cos(angle)
                   - x[b,h,r,j+D/2] * sin(angle)
y[b,h,r,j+D/2]   = x[b,h,r,j+D/2] * cos(angle)
                   + x[b,h,r,j]       * sin(angle)
```

The implementation MUST pair the first and second halves, never adjacent
elements, and MUST use the explicit `a+r`. It adds no scaling, position
broadcast, reset, cursor, cache update, transpose, head packing, or alternate
angle convention. Admission accepts only finite `theta` in
`[1, std::numeric_limits<float>::max()]`. Checked arithmetic MUST compute
`a + R - 1` and require the result to be no greater than `2^24 - 1` before
any queue, owner, output, or backend effect. `a` is `std::size_t`, so a
position-range overflow is `std::overflow_error`; a representable position
above the bound is `std::invalid_argument`.

At position zero, the encoding is an exact bitwise identity for every
admitted leaf, including signed-zero representations and all nonfinite
payloads. At nonzero positions, the eight non-F64 leaves use binary32
exponent, frequency, angle, sine, cosine, and pair intermediates. Each pair
uses separate noncontracted multiply operations followed by the add/subtract;
accidental FMA/contraction and fast-math are forbidden. Exactly one existing
named-format round-to-nearest-even/saturation encode stores each destination
value. `F64` uses binary64 exponent, frequency, angle, trigonometry, and
intermediates throughout and is never narrowed.

The nine semantically applicable ordinary signed floating leaves are exactly
`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
`F32`, and `F64`. The fourteen recognized but semantically inapplicable leaves
are exactly `BOOL`, `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`,
`U32`, `I64`, `U64`, and `F8_E8M0`. RoPE has no boolean/integer trigonometric
result, and exponent-only `F8_E8M0` cannot represent a general signed
rotation. Unknown enum values are `InvalidArgument`; only
`QuantizationFormat::NONE` is admitted. Recognized non-`NONE` formats and the
fourteen inapplicable leaves return `Unsupported` only after every earlier
shape, device, alias, parameter, range, overflow, and metadata check.

Admission MUST NOT scan tensor values. IEEE nonfinite classes follow the
written expression, including infinity-times-zero becoming NaN; nonzero
position NaN payload equality is not promised, and NaN comparisons are by
class. Finite acceptance uses an independent high-precision reference and
fixed destination tolerances: each of the eight non-F64 leaves permits at
most one destination ULP plus
`2^-9 * (abs(x_first) + abs(x_second))`; `F64` permits at most one
destination ULP plus `2^-44 * pair_norm`, where `pair_norm` is the
corresponding pair magnitude. Analytic and golden coefficients are pinned to
`mpmath 1.3.0` at 100 decimal digits in generated/static fixtures only;
production code has no runtime Python dependency.

Both facades perform the same host-side checks before capability dispatch,
snapshot allocation with externally visible effects, sequence reservation,
owner registration, workspace lease, queue submission, native metadata
transfer, or data access, in this order:

1. Validate both full specs, rank three through eight, all nonzero extents,
   positive even `D`, and identical `[...,H,R,D]` shape.
2. Validate exact queue-device identity, non-null/live owner registrations and
   native handles, value-consistent owner metadata, and checked plane,
   leading-stride, and view bounds.
3. Validate identical leading tuple/final axes and every checked shape, tile,
   element, byte, stride, address, and `a + R - 1` arithmetic, including
   the position conversion bound.
4. Reject any input/output owner identity overlap and any conservatively
   overlapping complete owner storage or transformed view range. Input and
   output owners MUST be distinct; in-place encoding is not promised.
5. Validate finite `theta`, its `[1,float_max]` bound, and all checked
   exponent/frequency/angle and narrowing intermediates.
6. Validate matching dtype and quantization, then classify the recognized
   dtype and quantization capability.

`rope_workspace_requirements` is pure and deterministic. After the same
admission validation it returns exactly `{0,1}` for an admitted supported
capability and performs no allocation, retained request construction, owner
registration, lease acquisition, token/sequence reservation, queue
submission, native metadata upload, data read/write, or queue/session
mutation. Its result is independent of queue occupancy, allocator state,
registration state, and prior submissions. Malformed and overflow input
propagates as `std::invalid_argument` or `std::overflow_error`; unsupported
capability propagates as `UnsupportedOperation`, never as an OID.

Submission repeats admission, obtains fixed-capacity value-copied
`RopeViewSnapshot`s and an immutable `RopeRequest`, and accepts only an empty
`RawWorkspaceView`. A supplied nonempty workspace is `InvalidArgument` and is
never silently ignored on an admitted supported path. Because the common
requirement is zero, no positive workspace, angle cache, caller scratch,
hidden tensor allocation, host round trip, or cursor exists. The request
retains exact owner/native identities and queue metadata through the existing
prepared registration and rollback mechanics until proven in-order
completion. Any setup failure rolls back every earlier registration and lease.
Admission failures map through `invoke_failure` to established negative OIDs,
consume no sequence/token, submit no work, and leave output bytes, owner
registry, workspace state, and queue/session state unchanged.

A default `UnsupportedOperation` hook is a pre-acceptance failure after common
admission: `rope` returns negative `OidError::Unsupported` with no accepted
token, registration, output mutation, or retained request. A positive OID is
never replaced by a later negative result. An accepted runtime/device failure
may leave output indeterminate or partial, is retained and rethrown
identically on every repeat wait, and prohibits output/session reuse until the
failure is drained under existing queue rules; no rollback of already-written
device data is promised.

The eventual backend matrix is explicit but is not a support claim in this
common leaf:

| Backend | Applicable leaves | Explicitly rejected leaves |
| --- | --- | --- |
| CPU | all nine ordinary signed floating leaves | the fourteen inapplicable leaves |
| CUDA | all nine ordinary signed floating leaves | the fourteen inapplicable leaves |
| ROCm | all nine ordinary signed floating leaves | the fourteen inapplicable leaves |
| SYCL | the eight non-F64 applicable leaves | `F64` and the fourteen inapplicable leaves |
| TTNN | `BF16` and `F32` | the other seven applicable leaves and the fourteen inapplicable leaves |

Backend branches consume only their own frozen producer outputs and add no
common backend-kind switch or capability registry. Every later port preserves
this ABI, validation order, zero-workspace result, aliases, absolute
positions, arithmetic, nonfinite behavior, and OID/lifetime contract.
Calls with `R=1` and `a=1,15,16,17`, and multi-row calls spanning those
positions, use consecutive absolute positions with no tile-boundary reset,
skipped position, or reused cursor. The common leaf owns no cache/session
initialized length.

For cache append, `source` MUST have shape `[...,H,R,D]` and `destination`
shape `[...,H,C,D]`, each of rank three through eight and with every extent
nonzero. Their leading tuples, `H`, `D`, dtype, quantization, and exact device
MUST match; `R` is the independent source length and `C` is cache capacity.
Admission MUST check `a <= C`, then `R <= C-a`, along with all byte,
stride, address, and range arithmetic before effects. For every logical
coordinate it writes

```text
destination[b,h,a+r,d] = source[b,h,r,d]
```

for `0 <= r < R` and MUST leave every other logical destination row unchanged.
The copy is bit-preserving: it performs no numerical conversion or rounding
and is semantically valid for every payload leaf that the exact device supports
for storage. Unused tile padding MUST remain unchanged and, like the
uninitialized cache tail, is not logical data and MUST NOT be read as an
initialized value. Transformed leading offsets and strides of source and
destination MUST be honored independently, including partial final tiles.

The boundary cases are normative. With exact-end capacity `C=a+R`, the cases
`(a,R,C)=(1,1,2)`, `(15,15,30)`, `(16,16,32)`, and `(17,17,34)` write exactly
rows `[a,C)` and preserve every row `[0,a)`; there is no special behavior at
`R=1`, `R=15`, `R=16`, or `R=17`, nor at positions 15, 16, and 17. With
capacity larger than `a+R`, rows `[0,a)` and `[a+R,C)` likewise remain
untouched. `a>C`, `R>C-a`, or overflow while evaluating a position or storage
range MUST be rejected before mutation. Exact-end append is valid, while
padding after a partial tile and every uninitialized logical row remain
excluded from initialized length.

K and V MUST use separate `cache_append` submissions with distinct cache
owners; there is no atomic two-cache operation. Neither RoPE nor cache append
owns initialized length `L`, reset, clearing, growth, or session failure
policy. The session MAY publish `L=a+R` only after both K and V append OIDs
have been successfully waited. Positive admission alone is insufficient, and
the session MUST NOT submit a dependent consumer of a failed producer.
Attention receives the initialized length explicitly and MUST NOT infer it
from capacity or padding.

New output and workspace storage MUST be disjoint from every input and from
each other. Cache source and destination overlap MUST be rejected. Read/read
overlap is harmless where all shape rules hold. No operand, output, or
temporary tensor may be allocated, relocated, or silently converted, and
neither operation may introduce a host round trip. Workspace is caller-owned
through proven completion; a supplied range is checked only at submission for
the exact device, the query's exact required size and alignment, overlap,
freshness, and lease availability.

Both facades MUST validate malformed views, rank and dimensions, leading
tuples, dtype, device, quantization, aliases, scalar finiteness, ranges,
workspace, and checked arithmetic before effects. Host-checkable invalid input
maps to `InvalidArgument`, arithmetic overflow to `Overflow`, and a recognized
but unsupported leaf to `Unsupported` only after earlier validation.
In particular, an odd `D` for RoPE is `InvalidArgument`.
Pre-acceptance resource or runtime failures use the established OID error
categories. A pre-submit failure mutates no output, registers no owner, and
consumes no sequence. Each operation snapshots required metadata without
retaining borrowed views. Once accepted, a positive OID is never replaced by a
negative result: completion failure is retained and rethrown on every wait.

Future delivery order is contract, independent reference, and all-five
feasibility, followed by CPU, CUDA, ROCm, SYCL, and TTNN closure for RoPE; only
then does cache append follow the same sequence. Each operation's five-backend
gate MUST close before the next operation begins. CPU feasibility is scalar or
wide arithmetic over existing tiled storage; this planned contract makes no
native accelerator or profiler claim. The operation-specific RoPE and Cache
row append siblings own detailed support, numerical, snapshot/hook, and
conformance rules. Their shared regression coverage MUST extend the existing
copy/storage suite and independent physical oracle for applicable payload
leaves, transformed leading mappings, partial tiles and padding, aliases,
overflow, accepted failures and repeat waits, and the `1/15/16/17` cases.
#### TinyLlama forward layout — Causal grouped-query attention

This subsection freezes the planned TinyLlama caller boundary for the
operation-owned [Scaled dot-product attention](#scaled-dot-product-attention)
section. It does not add declarations or implementation. The current `sdpa`
facade remains `Unsupported` before submission, output mutation, or token
acceptance until that operation is implemented.

The clean-cutover target has exactly the following submission and pure
workspace-query signatures:

```cpp
oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
         TensorView& out, size_t a, size_t L,
         RawWorkspaceView workspace={}) noexcept;
WorkspaceRequirements sdpa_workspace_requirements(
        const TensorView& q, const TensorView& k, const TensorView& v,
        const TensorView& out, size_t a, size_t L);
```

There are no head-count or head-dimension scalar arguments. The query has the
same semantic arguments as submission, makes `out` const, accepts no
workspace, returns only deterministic capacity/alignment requirements, and
throws the standard validation exceptions. It is pure: it allocates nothing,
registers no owner or workspace, consumes no queue sequence or native credit,
submits no work, and does not depend on queue occupancy or free capacity.

For one common, possibly empty, leading tuple `B`, shapes are
`q[B...,Hq,R,D]`, `k[B...,Hkv,C,D]`, `v[B...,Hkv,C,D]`, and
`out[B...,R,Hq*D]`. Derive `Hq`, `R`, and `D` from `q`; `Hkv` and `C` from
the matching `k`/`v` axes. The Q/K/V ranks are 3 through 8 and the merged
output rank is 2 through 7. Every dimension is nonzero; in particular
`Hq>0`, `Hkv>0`, `D>0`, and `C>0`. The K and V shapes must match, every
leading tuple must be identical, `Hq % Hkv == 0`, `Hq*D` must be checked,
and the output shape must match exactly. There is no cache, state, or
leading-plane broadcast. For each leading coordinate `b`, the head merge is
exactly

```text
out[b,r,h*D+d]
```

for `0<=r<R`, `0<=h<Hq`, and `0<=d<D`. Logical elements use the shared
16x16 tiled mapping and independent leading-plane mappings; a backend-native
physical representation, including TTNN native32 preservation, must preserve
these logical coordinates and tail rules.

The scalar range contract is `0<L<=C`, `a<C`, and `R<=C-a`; implementations
must perform the last check by subtraction rather than by first forming
`a+R`. For query row `r`, only the nonempty initialized causal set

```text
T(r) = { t : 0 <= t < L and t <= a+r }
```

is logical input. Generic SDPA deliberately permits `L<a+R`: rows whose
absolute positions reach or pass `L` attend to all and only the shorter
initialized prefix. The session specialization calls SDPA only after both
successful cache appends and sets `L=a+R`. Thus a cached/chunked row `r`
uses absolute position `a+r` and the same keys `0..a+r` as that row in a
full-sequence causal evaluation; full prefill is the `a=0`, `L=R` case.

Let `G=Hq/Hkv` and map query head `h` to KV head
`g(h)=floor(h/G)`. Independently for every leading coordinate, compute the
following target equations:

```text
S[h,r,t] = (sum(d=0..D-1, q[h,r,d] * k[g(h),t,d])) / sqrt(D)
m[h,r]   = max(t in T(r), S[h,r,t])
Pfp[h,r,t] =
    exp(S[h,r,t] - m[h,r])
    / sum(u in T(r), exp(S[h,r,u] - m[h,r]))
Pfp[h,r,t] = 0 exactly when t is masked
Pbf[h,r,t] = BF16_RNE(Pfp[h,r,t])
out[r,h*D+d] =
    BF16_RNE(sum(t in T(r), Pbf[h,r,t] * v[g(h),t,d]))
```

Q and K operands are BF16, their products accumulate in FP32, and the
`1/sqrt(D)` scale and stable max-subtracted softmax are FP32. Probabilities
are rounded RNE to BF16 before the BF16-probability/BF16-V PV product and
FP32 accumulation. That probability store and the final RNE BF16 merged
store are observable numerical boundaries; fusion may not erase either one.
The Scaled dot-product attention operation owns detailed special-value policy,
datatype applicability, tolerances, reference implementation, fixture
provenance, snapshots/hooks, backend support, and kernels, using these same
equations and rounding boundaries.

Excluded logical K or V values must not be read or participate in output,
reductions, or nonfinite checks. In particular, an implementation must not
turn an excluded V row into `0 * NaN`. Physical tensor padding is masked
separately from the causal and initialized-prefix masks. A native kernel may
read explicitly initialized neutral padding in caller-owned packed scratch,
but that padding is not a logical token and cannot change a logical result.

For the independent worked case
`Hq=4,Hkv=2,D=3,a=2,R=2,L=4,C=7`, `G=2`: query heads 0 and 1 map to KV
head 0, while heads 2 and 3 map to KV head 1. The exact permitted sets are
`T(0)={0,1,2}` and `T(1)={0,1,2,3}`. Independently perturbing K row 3 or V
row 3 cannot change output row 0, because that initialized future row is
causally masked there. Independently perturbing any K or V row in the
uninitialized capacity tail `4..6` cannot change either output row. The same
invariances apply per leading plane without cross-plane broadcast.

Output and workspace storage must each be disjoint from Q, K, and V and from
one another. Exact read-only aliases, and harmless read/read overlap, among
Q/K/V are allowed when every aliased view independently satisfies its shape
contract; validation must not ban them. The supplied workspace must have a
live owner on the exact device, meet the query's byte capacity and alignment,
not overlap operand or output storage, and not conflict with an outstanding
lease. Output and scratch remain caller-owned. Device-local packing and head
merge may use only that scratch; repeated KV-head materialization is not
required. The operation allocates or relocates no operand, output, or
temporary tensor and performs no hidden host round trip. It introduces no
public softmax, transpose, packing, cache, or session-mutation helper.

Both query and submission validate operand/view metadata, nonzero dimensions,
ranks, exact shapes and leading tuples, exact device identity, dtype and
quantization, read/write alias rules, mode-independent `a` and `L`, workspace
device/capacity/alignment/overlap, and checked `Hq*D`, range, element, byte,
stride, and address arithmetic before owner registration, sequence
consumption, output mutation, or submission. Malformed shape, device, alias,
or range is `InvalidArgument`; checked arithmetic is `Overflow`; a recognized
but unsupported dtype or capability is `Unsupported` only after those earlier
checks. Bounded-resource, runtime/device, and otherwise unclassified failures
map to `ResourceExhausted`, `DeviceError`, and `InternalError`. The `noexcept`
facade returns negative OIDs for synchronous failures. Accepted failures stay
observable on every repeated wait, while a pre-submit failure changes no
output and consumes no token. Submission snapshots metadata and never retains
a borrowed `TensorView` beyond the call.

Delivery order is contract and independent reference, all-five native
feasibility, then CPU, CUDA, ROCm, SYCL, and TTNN closure; SDPA is final among
the seven missing TinyLlama operations. The shared CPU scalar/wide baseline
over existing tiled storage is correctness coverage, not an accelerator
claim. Native evidence must cover both QK and PV for prefill and logical
`run=1`, not merely one GEMM, and excludes host computation or round trips,
elementwise substitutes, and padded extra logical tokens. Unsupported
hardware must be reported, and this design subsection makes no native kernel
or profiler-conformance claim.

Future shared conformance covers runs `1/15/16/17`, non-tile dimensions and
features, independent leading planes, padding/tail isolation, causal prefill,
cached decode, generic shorter initialized prefixes, every applicable
dtype/backend combination, wrong rank/empty-key/shape/GQA ratio, alias,
device, workspace, range and overflow rejection, and accepted-failure
repeated waits. An unsupported port is not numerical conformance. Only after
the all-five SDPA gate closes may the session component claim complete
decoder-layer assembly and verification.
#### TinyLlama forward layout — Workspace and execution

This subsection records the shared facade boundary. The common layer now
declares embedding, linear, RMSNorm, and RoPE; cache append, SiLU, and SDPA
remain planned until their own leaves land. The target `DeviceOps` facades
return `oid`, are `noexcept`, and have exactly these signatures:

```cpp
oid embedding(const TensorView& table, const TensorView& indices,
              TensorView& out, RawWorkspaceView workspace = {}) noexcept;
oid linear(const TensorView& x, const TensorView& w, TensorView& out,
           std::size_t s, std::size_t R, LinearOutputLayout layout,
           std::size_t H, std::size_t D,
           RawWorkspaceView workspace = {}) noexcept;
oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace = {}) noexcept;
oid rope(const TensorView& x, TensorView& out, std::size_t a, double theta,
         RawWorkspaceView workspace = {}) noexcept;
oid cache_append(const TensorView& source, TensorView& destination,
                 std::size_t a,
                 RawWorkspaceView workspace = {}) noexcept;
oid silu(const TensorView& x, TensorView& out,
         RawWorkspaceView workspace = {}) noexcept;
oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
         TensorView& out, std::size_t a, std::size_t L,
         RawWorkspaceView workspace = {}) noexcept;
```

Every dimension, row, head, position, and extent is `std::size_t`. Tensor
inputs are const, the submitted output (or cache destination) is mutable, and
the optional caller workspace is last. `enum class LinearOutputLayout {
ordinary, head_planar }` is the fixed output-mode type; output rank does not
select the mode. `ordinary` uses `H=1,D=O`. `head_planar` requires checked
`H*D=O` and inserts a head axis without exceeding rank eight. Embedding derives
the vocabulary extent from its rank-two table. RMSNorm derives feature width
from its last axis and has no `dim` argument. SDPA derives head counts and head
width from validated tensor shapes and receives only the explicit positions
`a,L`.

Each facade has one pure workspace query. The query has the same semantic
arguments in the same order, changes only its output/destination to const,
omits only the workspace argument, returns `WorkspaceRequirements`, and is
allowed to throw the established validation exceptions:

```cpp
WorkspaceRequirements embedding_workspace_requirements(
        const TensorView& table, const TensorView& indices,
        const TensorView& out);
WorkspaceRequirements linear_workspace_requirements(
        const TensorView& x, const TensorView& w, const TensorView& out,
        std::size_t s, std::size_t R, LinearOutputLayout layout,
        std::size_t H, std::size_t D);
WorkspaceRequirements rmsnorm_workspace_requirements(
        const TensorView& x, const TensorView& scale,
        const TensorView& out, float eps);
WorkspaceRequirements rope_workspace_requirements(
        const TensorView& x, const TensorView& out, std::size_t a,
        double theta);
WorkspaceRequirements cache_append_workspace_requirements(
        const TensorView& source, const TensorView& destination,
        std::size_t a);
WorkspaceRequirements silu_workspace_requirements(
        const TensorView& x, const TensorView& out);
WorkspaceRequirements sdpa_workspace_requirements(
        const TensorView& q, const TensorView& k, const TensorView& v,
        const TensorView& out, std::size_t a, std::size_t L);
```

A query MUST be deterministic for the supplied values and current backend
capability. It MUST validate operation support; every operand and output spec;
rank `2..8`; nonzero dimensions; exact queue-device identity; live owners;
leading-view bounds and strides; scalar, range, and mode values; output shape;
aliases; checked element, byte, address, plane, tile, and stride arithmetic;
and backend capability. It MUST NOT allocate, register an owner, acquire a
lease, reserve a queue credit or token, submit backend work, inspect queue
occupancy or free-arena capacity, or depend on prior completion. A successful
query reserves nothing. The actual supplied workspace is deliberately not a
query operand and is validated only by facade admission after the requirement
is known.

Before owner registration, sequence consumption, token acceptance, metadata
effects, or backend work, a facade MUST validate, in order:

1. host-known rank, nonzero shape, mode, position, and range constraints;
2. exact device identity, live owner identity, native handles, and leading
   view bounds and strides;
3. all checked multiplication, byte, address, plane, tile, and stride
   arithmetic;
4. operand/output shape and alias rules, followed by supported
   `QuantizationFormat::NONE`, dtype, and backend capability; and
5. the caller's actual workspace against the query's `{bytes, alignment}`,
   including liveness, exact-device identity, size, alignment, overlap, and
   lease availability.

A malformed request returns `InvalidArgument=-1`; a well-formed but unsupported
operation or matching input returns `Unsupported=-2`; checked arithmetic or
sequence exhaustion returns `Overflow=-3`; bounded resource or lease failure
returns `ResourceExhausted=-4`; a pre-acceptance runtime failure returns
`DeviceError=-5`; and any other unclassified failure returns
`InternalError=-6`. An admission failure accepts no OID and causes no backend
effect. Zero is never accepted. Every positive result is an accepted submitted
token; it proves admission only, not successful initialization of output or
cache data. Device-data out-of-vocabulary or nonfinite-policy failures MAY be
accepted without a hidden host round trip. Their output is unusable, their
session is poisoned, and every wait for that OID MUST retain and rethrow the
same failure.

The caller owns every output and all scratch. Nonzero scratch MUST be a live
`RawWorkspace` owner created by the queue's exact `Device`, with sufficient
bytes, the queried alignment, and no overlap with any operand or output.
Workspace subrange offsets are checked and 32-byte aligned, and owner
subranges are at least 32-byte aligned. An empty workspace is valid exactly
when the query reports zero bytes. CPU still rejects creation of a positive
`RawWorkspace`; TTNN creation of a positive `RawWorkspace` now owns one
replicated DRAM native page whose page size is the checked request rounded up
to 32 bytes while `byte_size()` stays exactly the caller-requested logical
bytes, and its native allocation is released only after proven completion.
The first operation that actually requires positive scratch on a backend owns
the minimal factory support and conformance tests. A capability-blocked
backend MUST name the missing evidence instead of inventing a byte
requirement.

The reusable session scratch capacity is the maximum of the actual
prefill/decode operation and host-transfer requirements, not the sum of
mutually exclusive live ranges. Independent submissions MAY use disjoint
aligned subranges of one owner. Overlapping use requires the existing lease
serialization. Outputs, operands, and scratch MUST remain alive until every
accepted reader and writer completes, and a workspace range is reusable only
after its covering completion is proved. When terminality cannot be proved,
the entire affected range and storage remain retained or quarantined; they
MUST NOT be released or reused by an unrelated queue or device destruction.

At each facade call, common code MUST snapshot every required tensor spec,
exact device and owner identity, native handle, leading offset and strides,
scalar, mode, and workspace range. It MUST NOT retain a borrowed `TensorView`
or other borrowed view reference beyond the call. Submission follows the
existing snapshot, owner-registration, and exact-alias owner-deduplication
pattern. This contract adds no global active-backend registry, duplicate
validation framework, or no-allocation guarantee beyond the existing queue
machinery.

A queue is in order, but it does not propagate predecessor errors: current
queues can execute a successor after a predecessor fails, and `wait(last)`
does not report an earlier failed OID. A future TinyLlama session therefore
MUST successfully wait for every direct producer before submitting a consumer
of that producer. It MUST NOT submit that consumer after a producer wait
fails. Independent branches MAY be submitted together and then waited
individually; Q, K, and V projections are independent after their shared input
succeeds, as are gate and up projections. These are explicit correctness
boundaries, not hidden operator round trips or waits introduced only for
disabled tracing.

For one token row or prefill block, the required producer-success schedule is:

1. submit embedding and wait successfully before submitting its RMSNorm
   consumer; submit that attention RMSNorm and wait successfully;
2. submit Q, K, and V linear projections together, then wait successfully for
   all three OIDs individually;
3. submit the Q and K RoPE branches together and wait successfully for each;
4. submit separate K and V `cache_append` operations and wait successfully for
   both append OIDs; only then publish checked `initializedL = a + R` and
   submit SDPA with that initialized prefix;
5. wait successfully for SDPA before its output projection, wait for that
   projection before the residual add, and wait for the residual before the
   MLP RMSNorm;
6. after the MLP RMSNorm succeeds, submit gate and up linear projections
   together and wait successfully for both; wait for SiLU of gate before the
   elementwise gate/up product, then wait in turn for the down projection and
   final residual add; and
7. commit the logical token/request only after the final producer succeeds.

For a concrete failure boundary, suppose accepted `token1` completes,
accepted `token2` fails, and independently accepted `token3` completes.
Successful `wait(token3)` MUST NOT be interpreted as success of `token2` or of
the whole chain. The session waits all three separately, submits no consumer
of `token2`, marks itself failed, and drains every already accepted OID. The
same rule applies to cache append: if accepted K append `tokenK` fails while
accepted V append `tokenV` succeeds, the session still waits/drains both,
publishes no new `initializedL`, and submits no SDPA. Physical cache writes
that did complete remain distinct from the uncommitted logical token; a later
failure may therefore leave updated cache storage while poisoning the session.

For an accepted OID, `wait(oid)` blocks through completion and rethrows its
retained failure. Every repeated wait for that OID MUST rethrow the same
failure; successful waits and completion visibility are likewise repeatable.
`wait` immediately throws `std::invalid_argument` for negative, zero, foreign,
future, skipped, reserved-but-never-submitted, or otherwise unsubmitted
values. There is no cumulative wait, fail-fast cancellation, KV rollback,
retry, or reuse of poisoned state.

After any admission failure or accepted completion failure, the session MUST
stop new dependent submissions, become poisoned, and observe or drain every
accepted OID. Each wait is attempted independently, and draining MUST continue
after an individual wait throws. Storage can be released or reused only where
terminality is proved; otherwise it remains retained or quarantined. Token
commit is separate from physical cache initialization, and no failed cache
prefix may be published.

The planned neural signatures and queries above are not current API
declarations.
The existing neural hooks remain `Unsupported` until their real operation
ports land; this subsection changes no facade, kernel, queue, session, or
selector implementation.

#### TinyLlama forward layout — Final-logits selection and ownership

This subsection freezes and publishes the backend-neutral synchronous selection
boundary for the final untied LM-head result described by
[Embedding and projection boundaries](#tinyllama-forward-layout--embedding-and-projection-boundaries).
The public seam is:

```cpp
struct TokenSelectorScratch {
    std::span<std::byte> host;
    RawWorkspaceView device;
};

struct TokenSelectorScratchRequirements {
    std::size_t host_bytes;
    WorkspaceRequirements device;
};

class TokenSelector {
public:
    virtual ~TokenSelector() = default;
    virtual TokenSelectorScratchRequirements scratch_requirements(
            const TensorView&, std::size_t) const = 0;
    virtual std::size_t select(
            DeviceOps&, const TensorView&, std::size_t, oid,
            std::span<const std::size_t>, TokenSelectorScratch) = 0;
};
```

`scratch_requirements(logits, valid_vocabulary)` is a pure query over the
supplied view and extent. It MUST perform no allocation, transfer, queue
reservation, wait, submission, owner registration, or data mutation. A
concrete implementation defines its own host byte count, device
`WorkspaceRequirements`, alignment, placement, and admission details; this
seam mandates no host transfer, staging direction, vendor type, scratch
capacity, or backend capability.

`TokenSelectorScratch` is a per-call descriptor. Its `host` span and `device`
`RawWorkspaceView` refer only to caller-provisioned reusable storage. They
MUST remain live through the synchronous `select` return and MAY be reused
only after that call completes. The selector MAY write its scratch during the
call but MUST NOT retain either span, the `RawWorkspaceView`, or any hidden
scratch allocation after return. No persistent selector ownership is implied.

`select` is synchronous and deliberately not `noexcept`. Its arguments are,
in order, the queue, logits, valid vocabulary, producing OID, complete
history span, and caller-owned scratch descriptor. Invalid input, readiness,
runtime, and device-data failures are delivered with standard exceptions. The
method returns one token ID only after the producing work and all selector
work have completed successfully; it exposes no asynchronous selection result
or selector OID.

`logits` MUST be a borrowed const BF16 view with exact logical rank-two shape
`[1,V]`, where both dimensions are nonzero. The view and its live storage
owner MUST belong to the exact device served by the queue. `valid_vocabulary`
MUST be nonzero and exactly equal the logical `V`; the only selectable IDs are
`[0,V)`. Physical 16x16 tile padding, and any physical row or feature outside
that logical extent, MUST NOT be read as a candidate or included in
validation of logit values.

The producing `oid` MUST be positive and MUST identify work actually
submitted by this same live `DeviceOps` queue. An OID from a different queue
is invalid even when both queues serve the same backend device; zero,
negative, foreign, future, skipped, reserved-but-never-submitted, and
otherwise unsubmitted values remain invalid under the existing queue
contract. The session MUST successfully wait for every direct prerequisite
before submitting a dependent final projection. Selection itself MUST then
successfully observe `queue.wait(producer)` before using the logits. Positive
admission alone is not readiness. An already successful wait does not remove
this requirement because successful waits are repeatable. If the producer has
a retained completion failure, this call rethrows it, and every later wait
for that OID rethrows the same failure.

The queue, logits view, logits storage owner, and the storage owner's exact
device identity MUST remain live and unchanged throughout the call.
`history` is likewise borrowed only for the call, and the caller-provisioned
scratch storage remains live until return. The selector MUST NOT retain the
queue, either borrowed view, either span, the scratch descriptor, or any
referenced storage after return. Selection MUST NOT mutate logits, history,
KV storage, or any owner identity. The selector MAY mutate only the supplied
scratch storage. Scratch placement, capacity, alignment, and lifetime are
caller/implementation boundaries described by the requirements query, not
hidden ownership in this abstract seam.

No full-vocabulary host transfer is required by this seam. Backend-neutral
`DeviceOps` and `TensorView` access is sufficient, and the concrete selector
chooses a permitted device-local or host-transfer strategy. This boundary
mandates neither a hidden transfer nor a vendor API or vendor type.

`history` MUST contain the entire prompt/input sequence, including every
policy-added special token, followed in order by every successfully committed
generated ID. It therefore includes the latest committed token even when that
token has not yet been appended to KV for a later decode step. The selector
neither appends nor removes history and owns no session state.

After a successful selector return, the session MUST independently validate
the returned ID against `[0,V)` before committing it. Only then does the
session append it to history, increment the committed generated-token count,
apply its EOS/new-token/context precedence, and decide whether another model
step and KV append are needed. EOS inclusion, simultaneous-stop precedence,
capacity, absolute positions, initialized cache length, cache mutation,
failure/poisoning, draining, and reset remain exclusively session-owned. The
selector MUST NOT interpret an ID as EOS or mutate those policies.

The following prompt and decode transitions are normative:

1. After successful prefill of prompt IDs `[p0,p1,p2]`, the session has
   `history=[p0,p1,p2]`, committed generated-token count zero, and
   `initializedL=3`. The first selector call borrows that complete history and
   the final `[1,V]` logits plus their producing queue/OID.
2. If it returns valid `t0`, the session commits
   `history=[p0,p1,p2,t0]` and generated-token count one. If EOS or an
   applicable new-token/context limit ends the request, `t0` remains included
   in committed output while `initializedL` remains three: no subsequent KV
   append is needed merely to record an already committed terminal token.
3. If generation continues instead, the session processes committed `t0` at
   absolute position three. Only after both K and V append OIDs succeed may it
   publish `initializedL=4`. The following selector call then receives
   `history=[p0,p1,p2,t0]`; history is not reconstructed from cache length.

Committed generated-token count and initialized cache length are therefore
distinct state. Selection cannot publish a cache prefix, and committing a
terminal token cannot imply a physical KV write.

The production selector sibling owns exactly greedy behavior. It MUST examine
every logical logit in `[0,V)` and establish that each is finite, including
values that cannot win. It MUST throw on any NaN, positive infinity, or
negative infinity; a nonwinning NaN is still a data failure. For an all-finite
input it returns the ID of the maximum value and resolves an exact tie by the
lowest ID. There is no stochastic path, RNG, temperature, top-k, top-p, beam
search, or hidden fallback.

Failure ownership and no-mutation behavior are normative:

- An invalid logits shape, extent, owner/device relationship, or producer OID
  is rejected by the selector boundary with a standard exception and no
  returned ID.
- A producer readiness/completion failure, selector runtime failure, or
  nonfinite logical logit is a selector-call failure. It returns no ID; the
  session commits no token and does not change history or initialized cache
  state. Supplied scratch may contain partial private work after a failure,
  but the selector retains no scratch ownership. Session poisoning and
  draining of already accepted OIDs remain the session's responsibility.
- A deterministic injected selector MAY replace greedy behavior only in tests.
  If it returns an ID outside `[0,V)`, the session rejects that result before
  history, generated count, or cache mutation. Injection cannot bypass
  producing-OID readiness, ID-range validation, history ownership, or cache
  rules.

Thus a failed or out-of-range selection consumes no invalid token. Selector
scratch may contain partial private work after a failure, but logits, history,
KV contents, owner identities, committed-token count, and initialized cache
length remain unmodified by selection.

This public seam adds no concrete greedy selector, selector source, permanent
test target, session implementation, transfer strategy, allocation, facade,
kernel, or current-support claim. Existing neural hooks remain `Unsupported`
until their real operation ports land; the concrete selector sibling owns the
deterministic implementation, oracle, and any exact reusable scratch
requirements.
#### TinyLlama forward layout — Session sizing and lifetime

This subsection is the normative bounded-storage plan for the planned
single-sequence session. It does not add a model, session, loader, selector,
facade, kernel, or current support claim. The existing neural hooks remain
`Unsupported`; the operation-owning sections and all-five gates above remain
authoritative.

##### Parameters and two-stage setup

`Nlayers,F,M,Hq,Hkv,D,V,C,R` are runtime values. Each is nonzero when used,
and the session validates checked `F=Hq*D`. `C` is the context and per-cache
logical row capacity. The separately configured nonzero `Qcap` is queue
admission capacity only: it bounds accepted in-flight work but neither changes
`C` nor multiplies weights, caches, banks, or scratch. All tensor ranks remain
2 through 8; an inserted head axis or additional leading axes must still fit
that interval. Plane counts and transformed leading offsets remain independent
and are never state broadcast.

Setup occurs in two stages:

1. **Session setup, before requests.** Validate configuration and checkpoint
   shapes; establish mapped source -> `SafeTensors` -> caller-created device
   ownership; create persistent BF16 weights; allocate each layer's bounded K
   and V cache owners; and allocate immutable-final-axis logical-`R1=1`
   decode banks. These resources are not recreated for a different prompt
   length.
2. **Request setup, after tokenization.** Validate exact `R>0` and `R<=C`
   before submission, reject overflow or excessive prompts without
   truncation, allocate exact-`R` prefill banks, create every operand and
   output needed to query actual operation and transfer requirements, and
   provision reusable prefill/decode workspace plus separate synchronous
   selector scratch. Replacing a request's banks or workspace requires a
   complete drain of the prior request first.

There is no one-bank-per-possible-`R` family, capacity-row logical-padding
trick, final-axis slicing, retargeted owner or view, per-token allocation, or
exposure of physical capacity as initialized logical length. An owner has one
shape and stable identity for its entire lifetime. Exact-R prefill and
fixed-R1 decode banks are reused only after all readers and workspace leases
that reference them complete.

The loader validates model configuration, tensor names, exact logical shapes,
and checked storage before copying mapped bytes into caller-created device
tensors. It alone adapts rank-one normalization scales to `[1,F]`. Persistent
BF16 weight shapes are:

| Scope | Weight shape |
| --- | --- |
| model | embedding `[V,F]`, final norm `[1,F]`, untied LM head `[V,F]` |
| each layer | attention norm `[1,F]`, post-attention norm `[1,F]` |
| each layer | Q `[F,F]`, K `[Hkv*D,F]`, V `[Hkv*D,F]`, O `[F,F]` |
| each layer | gate `[M,F]`, up `[M,F]`, down `[F,M]` |

The mapped source and SafeTensor views remain source ownership, not a
persistent second checkpoint bank. Device/arena/queue metadata overhead and
host transfer staging are accounted separately from tensor bytes. This BF16
session plan adds no additional F64 implementation or precision obligation.

##### Checked cache and activation storage

For one sequence, each layer owns distinct K and V BF16 tensors with exact
logical shape `[Hkv,C,D]`; no implicit leading-plane or session-state
broadcast is allowed. All products are evaluated one factor at a time with
checked arithmetic:

```text
cache_logical_bytes =
    2 (K and V) * Nlayers * Hkv * C * D * 2 (BF16 bytes)

per_cache_standard_bytes =
    Hkv * ceil(C / 16) * ceil(D / 16) * 256 * 2
all_standard_cache_bytes =
    2 * Nlayers * per_cache_standard_bytes.
```

The two final axes are padded independently; `256` is the number of BF16
elements in a 16x16 tile. A checked ceiling division must not first perform an
unchecked `extent+tile-1`. Standard storage is the result of the existing
`TensorSpec::tiled_storage_nbytes()` contract, not its logical byte count.

TTNN does not use the standard formula. Its assessment requires one native
32x32 allocation per leading plane, checked native extents and `uint32_t`
conversions, and an evidenced carrier/runtime allocation size:

```text
per_cache_native32_bytes =
    Hkv * ceil(C / 32) * ceil(D / 32) * 1024 * carrier_bytes.
```

That expression is an assessment form, not a fabricated TTNN allocation
promise. If the backend reports another carrier representation or owns an
opaque allocation size, the reported value is used. Unknown exact native
allocation remains a capability-evidence gap.

For the independent checked case
`Nlayers=2,F=8,M=12,Hq=4,Hkv=2,D=2,V=19,C=17`, `F=Hq*D`.
Logical K+V storage is `2*2*2*17*2*2 = 544` bytes. A standard cache is
`2*ceil(17/16)*ceil(2/16)*256*2 = 2048` bytes, so both caches in both
layers require `2*2*2048 = 8192` bytes. A native32 assessment is
`2*ceil(17/32)*ceil(2/32)*1024*carrier_bytes`; with a confirmed two-byte
BF16 carrier this illustration is `4096` bytes per cache and `16384` bytes
total. Another measured carrier or runtime-owned size replaces that
illustration; native extents, plane counts, byte products, and conversions
remain checked.

`F=8` and `M=12` deliberately exercise non-tile final axes. Exact BF16
activation accounting is:

| Exact logical `R` | `[R,8]` logical bytes | `[R,8]` standard bytes | `[R,12]` standard bytes |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 512 | 512 |
| 15 | 240 | 512 | 512 |
| 16 | 256 | 512 | 512 |
| 17 | 272 | 1024 | 1024 |

The logical expression is `R*8*2`. Standard storage is
`ceil(R/16)*ceil(width/16)*256*2`; both widths occupy one feature tile in this
case. A decode bank is the fixed `R1=1` row. `R=17` is exact capacity and is
accepted without truncation; `R=18` is rejected before allocation. The
physical padded rows in the table are neither initialized tokens nor
permission to allocate a logical `[C,width]` prefill bank.

Every rank, element, plane, tile, byte, alignment, subrange-end, native
conversion, and address-offset calculation is checked stepwise before
allocation or submission. This includes `Hkv*D`, `Hq*D`, `a+(R-1)`,
`C-a`, weight products, all leading-plane products, tile ceiling/round-up,
and conversion of bits to bytes. Overflow is an error; it never wraps into a
smaller allocation.

##### Forward stores and live ranges

The session embeds the input once. Each configured decoder layer performs, in
order, RMSNorm; Q/K/V projections; separate Q and K RoPE; separate K and V
append; causal GQA SDPA; output projection; residual; RMSNorm; independent
gate/up projections; `SiLU(gate)`; the existing `mul(SiLU(gate),up)`; down
projection; and residual. After all layers, final RMSNorm and the untied
ordinary LM head's final-row window produce `[1,V]` logits.

Ordinary and head-planar projections retain the equations fixed above:

```text
Y[b,r,o]   = sum_i X[b,s+r,i] * W[o,i]
Y[b,h,r,d] = sum_i X[b,s+r,i] * W[h*D+d,i].
```

Q is `[Hq,R,D]`; K and V are `[Hkv,R,D]`. RoPE uses the split-half pair at
absolute `a+r`, separate K/V append writes
`cache[b,h,a+r,d]=new[b,h,r,d]`, and GQA selects
`g(h)=floor(h/(Hq/Hkv))`. SDPA reads only initialized `0<=t<L` that also
satisfies `t<=a+r`, applies its stable masked softmax and explicit BF16
probability boundary, and stores merged `[b,R,Hq*D]`. The final LM head uses
ordinary mode `s=run-1,R=1` and writes `[1,V]`; no extraction operation or
final-axis view transform is involved.

All operands and outputs below are BF16 except integer token indices and wide
caller scratch. Outputs are disjoint from inputs/readers. Read/read weight
reuse is allowed, but no owner moves and no physical bank or scratch range is
reused before every direct reader and lease completes.

| Value/resource | Shape/storage | Lifetime and reuse rule |
| --- | --- | --- |
| `X`, `X2`, next residual | exact prefill `[R,F]`, decode `[1,F]` | Each bank remains live through all direct readers; reuse follows their terminal OIDs. |
| Norm outputs | `[R,F]` or `[1,F]` | Separate output and read owners; these paths do not use an in-place alias. |
| Q/K/V projections | `[Hq,R,D]` / `[Hkv,R,D]` | Q/K remain through RoPE; K/V remain through their respective appends. |
| Rotated Q/K | matching head-planar shapes | Rotated Q remains through SDPA and rotated K through append; neither view is retargeted. |
| K/V caches | per layer `[Hkv,C,D]` | Persistent through the request; append alone initializes rows, only the published prefix is readable, and reset/destruction follows a safe drain. |
| Attention merged/output projection | `[R,Hq*D]`, then `[R,F]` | Output/read storage is disjoint and scratch reuse waits for all readers. |
| Gate/up, SiLU, product, down | `[R,M]`, `[R,M]`, `[R,M]`, `[R,F]` | Gate and up may enqueue independently; subsequent consumers wait for every direct producer. |
| Final norm/logits | `[R,F]`, then `[1,V]` | Logits are borrowed by the selector only for its synchronous call. |
| Token-index input | integer `[1,R]` or explicit independent planes | Caller-owned, validated before embedding, retained through embedding completion. |
| Host transfer/staging | caller-owned wide scratch and backend staging | Sized from actual transfer requirements; never hidden persistent checkpoint duplication. |
| Selector resources | caller-owned synchronous scratch | Provisioned separately from its actual requirement; no per-selection allocation or async task. |

The table defines logical values, not one simultaneously resident owner per
row of the table or per layer. Storage sharing is allowed only when these
actual nonoverlapping lifetimes prove it; no memory saving may be claimed from
an assumed final OID, queue order alone, or an aliased output.

##### Actual requirement maximum and backend staging

Create all tensor operands and outputs first, then call the pure workspace
queries and host-transfer requirement queries with the actual exact-R
prefill and fixed-R1 decode views. The reusable operation/transfer owner is
sized as:

```text
scratch_bytes =
    max(each actual prefill operation requirement,
        each actual decode operation requirement,
        each actual host-transfer requirement)
scratch_alignment =
    max(32, each actual required power-of-two alignment).
```

Both calculations use checked alignment and address arithmetic. The selector's
synchronous scratch is separately provisioned from its own requirement.
Mutually exclusive live ranges are not added; simultaneous independent
submissions use checked disjoint subranges or the existing lease
serialization. A changed request cannot replace this owner until every
accepted use is terminal.

The four feasibility records below constrain, but do not replace, those actual
queries:

| Backend | Matrix staging that request setup must include when selected |
| --- | --- |
| CUDA | Direct linear may query zero global scratch because its assessed route uses fixed kernel-local tiles. SDPA's assessed caller range contains checked, 32-byte-aligned FP32 score and BF16 probability segments. |
| ROCm | The conservative assessed linear range contains checked aligned `x_pack` and `y_pack`. Its SDPA range contains `q_pack`, sequentially reused per-`Hkv` K/V pack, FP32 scores, BF16 probabilities, BF16 PV, and merged staging. |
| SYCL | The assessed linear range contains checked FP32 product staging. Its conservative SDPA range contains FP32 scores, BF16 probabilities, FP32 PV, and BF16 head staging, with reuse only after the producing stage completes. |
| TTNN | Native32 score and probability storage is a minimum SDPA need; optional packs/PV storage depend on the selected Metalium path. The assessed buffer route may require 4096-byte alignment. The TTNN factory/range path now exists (one owning replicated DRAM native page per positive request); the exact positive SDPA requirement itself remains unassessed, and neither hidden TTNN tensors nor guessed bytes are allowed. |

Standard CUDA/ROCm/SYCL tensor data and raw workspace subranges retain the
existing 32-byte arena guarantees. TTNN uses its assessed native32 allocation
and checked `uint32` limits, never the standard tensor byte formula. CPU
continues to reject positive `create_workspace` requests; TTNN now owns real
positive scratch, so an operation that needs it names its own queried
requirement instead of inheriting a missing backend-private factory gap.
Unknown capability-dependent requirements are recorded as missing evidence,
not filled with a speculative constant.

##### Producer schedule, cache publication, and abort

A positive OID proves admission only. Consumers are submitted only after all
their direct producer waits succeed:

1. wait for embedding before attention RMSNorm, and wait for that norm;
2. enqueue Q, K, and V projections as independent branches, then attempt all
   three waits and require all three successes;
3. enqueue Q and K RoPE independently, then attempt and require both waits;
4. enqueue separate K and V appends, attempt both waits, and only after both
   succeed publish checked `initializedL=a+R` and submit SDPA;
5. wait successively for SDPA, output projection, first residual, and MLP
   RMSNorm;
6. enqueue gate and up together, attempt and require both waits, then wait in
   turn for SiLU, product, down projection, and the second residual; and
7. after the final layer, wait for final norm and the one-row LM head before
   synchronous selection and logical token commit.

The two append waits are independent correctness barriers. If K append fails
and V append succeeds, the V write may be physical but no new initialized
prefix is published and SDPA is not submitted. Every cache row is initialized
only by its append. No SDPA call may read a row outside the last successfully
published prefix.

Queues are in order but do not propagate predecessor errors. A later
successful OID, including a final OID, says nothing about an earlier failed
OID. On admission or completion failure, the session stops new dependent work,
becomes poisoned, and attempts to wait/drain every accepted OID; one thrown
wait does not stop later drain attempts. Token commit is separate from
physical cache initialization, and there is no rollback, retry, or reuse of a
possibly updated failed-session cache.

Storage whose terminality is unknown remains retained or quarantined. A
request reset may publish valid length zero, reset an allocator, destroy
owners, or replace exact-R banks/workspace only after a safe drain and after
all live owners and leases permit it. A replacement request never reads the
prior request's cache and never installs new storage before that boundary.

##### Attribution ownership seam

Attribution is correlation metadata owned by the caller, not a tracing or
timing implementation. For every phase—`load`, `tokenization`, `prefill`,
`decode`, and `selection`—the caller may associate:

- phase and operation name;
- decoder layer where applicable;
- absolute row/token position;
- the operation's positive OID where one exists; and
- explicit host-enqueue and completion-observation boundaries.

Host enqueue elapsed time, completion-observed elapsed time, and genuine
device timestamps are different measurements and must not be conflated.
Selection is synchronous and has no selector OID. This seam allocates no
telemetry event, adds no timing API, retains no view, creates no async selector
task, and does not force a per-operation wait merely because tracing is
disabled. The correctness waits above remain mandatory with or without
attribution.

Complete mathematical layer assembly remains gated until SDPA closes its
all-five backend gate. Incremental model/session/selector integration is owned
by its later siblings, not by this documentation plan.

#### TinyLlama forward layout — TTNN matrix feasibility

This is a bounded feasibility record for the planned BF16 TinyLlama
operations, not a declaration of current support. TTNN still returns
`Unsupported` for the neural facades before accepting an OID, and no result
below is production conformance or profiler evidence. In particular, the
host-computed implementation in `src/ttnn/binary.cpp` and the existing smoke,
native-storage, copy, and unsupported-capability tests are not matrix
evidence.

##### Evidence and installed target

The assessment used the configured `ttnn` host and the isolated remote
workspace `forward-layout-ttnn-feasibility`. Installed facts, source facts,
documented capability, and runtime evidence are deliberately separated:

- **Installed package and source.** `TT_METAL_HOME` was
  `/home/rlew/tt/src/tt-metal`. A clean `main` checkout reported commit
  `06994d4afdaa61e89753d73a59d7fd241187f37b`, description
  `v0.76.0-dev20260801-268-g06994d4afda`, and the generated TT-NN and Metalium
  CMake package-version files both report `0.76.0`. The source's own configured
  version is `0.76.0-dev20260801+268.06994d4afd`. IOM already requires
  `find_package(tt-nn CONFIG REQUIRED)` and privately links
  `TT::Metalium` and `TTNN::TTNN`; it has no TTNN version bound.
- **Installed device and storage.** A standalone inventory program, compiled
  against those two imported targets and then removed, observed one initialized
  Blackhole device: device 0, one hardware command queue, eight DRAM channels,
  1,572,864 bytes L1 per core, 4,278,190,080 bytes per DRAM channel, a
  1,350 MHz clock, physical grid `17x12`, logical grid `12x10`,
  compute/storage grid `11x10`, and DRAM grid `8x1`. The runtime logged firmware
  bundle `19.13.1`, KMD `2.8.0`, and IOMMU disabled. This proves installed
  runtime availability and device identity, not a matrix result.
- **IOM source-declared storage.** `src/ttnn/device.cpp` creates one TILE-layout
  `ttnn::Tensor` per logical leading plane. BF16 is native
  `DataType::BFLOAT16`; rows and columns are physically rounded to 32 and each
  plane is capped at 1 GiB. `src/ttnn/copy.cpp` identifies the physical order
  as row-major 32x32 tiles, each tile containing four row-major 16x16 faces.
  `src/ttnn/device_types.cpp` checks each native extent against `uint32_t`.
  These facts establish native32 storage only.
- **Installed high-level API.** At the installed commit,
  [`matmul.hpp`](https://github.com/tenstorrent/tt-metal/blob/06994d4afdaa61e89753d73a59d7fd241187f37b/ttnn/cpp/ttnn/operations/matmul/matmul.hpp)
  declares BF16-capable `ttnn::matmul`/`linear`, transpose flags, compute
  configuration, and an `optional_output_tensor`.
  [`matmul_device_operation.cpp`](https://github.com/tenstorrent/tt-metal/blob/06994d4afdaa61e89753d73a59d7fd241187f37b/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp)
  requires allocated, same-device, tiled floating inputs; a 32-wide inner tile;
  matching logical and padded K; and positive M/K/N. Its optional-output path
  requires exact computed logical shape, selected dtype, and memory
  configuration, and reuses that tensor; the absent-output path calls
  `create_device_tensor`.
- **High-level rejection and allocation cases.**
  [`matmul.cpp`](https://github.com/tenstorrent/tt-metal/blob/06994d4afdaa61e89753d73a59d7fd241187f37b/ttnn/cpp/ttnn/operations/matmul/matmul.cpp)
  rejects optional-output volume mismatch. `matmul_batched_weights` rejects
  both transpose flags, activation, output-tile selection, and every optional
  output, so it cannot satisfy caller-owned projection or QK/PV output.
  Depending on its chosen program, ordinary `matmul` may call allocating
  transpose wrappers; its bias and unary post-processing can also become
  separate operations. TinyLlama uses no linear bias or fused activation, but
  the HF `[out,in]` weight still requires either a program that consumes
  transposed B without an allocating wrapper or a checked caller-owned pack.
  Therefore an optional output is only a candidate, never proof of the IOM
  ownership contract.
- **Documented lower-level capability.** The
  [Metalium single-core matmul](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html)
  uses explicit source and destination DRAM buffers, a selected mesh command
  queue, 32x32 BF16 tiles, reader/compute/writer kernels,
  `matmul_init`/`matmul_tiles`, and `pack_tile`. This demonstrates a route
  whose writer can target a pre-existing buffer. The
  [Tensix compute/dataflow documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html)
  documents the FPU/SFPU, circular-buffer synchronization, and the distinction
  between storage format and compute registers. It also explicitly warns that
  `fp32_dest_acc_en=true` only makes each destination element 32 bits and does
  not prove FP32 computation; the matrix engine's stated maximum is TF32.
- **Runtime matrix evidence.** Parent verification configured and built the
  existing IOM TTNN smoke target, then observed its single device test pass in
  1.50 seconds on the installed Blackhole. That establishes availability only:
  no project neural test, logical `R=1/15/16/17` matmul sample, QK/PV sample,
  production conformance run, or profiler run was executed for this record. An
  attempted build of the installed upstream single-core example produced no
  binary because that SDK build tree was configured with programming examples
  disabled and its regeneration stopped at a missing `AMDDeviceLibs` package.

The repeatable inventory commands are
`git -C "$TT_METAL_HOME" rev-parse HEAD`,
`git -C "$TT_METAL_HOME" describe --tags --always --dirty`, and inspection of
`build_RelWithDebInfo/lib/cmake/{tt-nn,tt-metalium}/*-config-version.cmake`.
Every TTNN execution uses a 300-second timeout. Parent verification synchronized
this exact worktree and ran:

```text
cmake -S . -B build/ttnn -DBUILD_TESTING=ON -DTTNN_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF
cmake --build build/ttnn --target iom_ttnn_smoke_tests -j2
ctest --test-dir build/ttnn --output-on-failure -R "^iom_ttnn_smoke_tests$" --timeout 300
```

Smoke can establish device availability only. It cannot change a matrix cell
below from blocked to supported.

##### Contract fit and matrix decision

Let `B` be the checked product of all independent leading dimensions and let
`P32(n)=checked_mul(checked_ceil_div(n,32),32)`. Every public dimension is
nonzero and remains `size_t` through common validation. Before constructing a
TTNN shape or a `uint32_t` runtime argument, the backend checks the conversion
against `UINT32_MAX`; all plane, tile, element, byte, stride, and address
products and additions are checked first. Ranks remain 2 through 8,
`QuantizationFormat::NONE` is required, transformed leading offsets and
strides select independent native planes, and neither 16x16 logical nor 32x32
physical padding creates a logical row, feature, head, key, or token.

The preferred route is one bounded backend-private Metalium program family,
not a public matrix layer. Its dataflow readers consume existing IOM native32
owners, mask the logical tails, select linear row window `s..s+R`, read HF
weights `[O,I]` in transposed orientation, and map GQA head `h` directly to
`g(h)=floor(h/(Hq/Hkv))`. Its writer targets the supplied IOM output or a
caller-workspace subrange. Any padding placed in workspace is explicitly
initialized to neutral values. It does not materialize repeated KV heads,
transpose or copy persistent weights, read K/V capacity tail `L..C`, expose a
TTNN type publicly, or use a host/staging/elementwise substitute.

For the table, **blocked/native route present** means that installed TTNN and
Metalium expose the necessary matrix mechanism on the observed Blackhole, but
the current IOM port correctly remains `Unsupported` until the operation
implements that route, proves its precision, ownership, queue, and lifetime
behavior, and passes its native gates. The two linear rows are **implemented**
at this revision: the mandatory `BF16` leaf runs on the direct Metalium route
recorded in [Linear projections](#linear-projections) and the twenty other
applicable leaves are explicit capability rejections, so those rows no longer
carry that blocker; the QK and PV rows still do.

| Product | Logical `R=1` | `R=15` | `R=16` | `R=17` | Required native mapping and present blocker |
| --- | --- | --- | --- | --- | --- |
| ordinary linear | implemented (`BF16`) | implemented (`BF16`) | implemented (`BF16`) | implemented (`BF16`) | Per `b`, multiply selected `x[b,s:s+R,I]` by transposed shared `w[O,I]` and write exactly `out[b,R,O]`. Non-tile I/O and every leading plane are valid with the tail-masked readers of the landed route; no high-level transpose and no hidden temporary. |
| head-planar linear | implemented (`BF16`) | implemented (`BF16`) | implemented (`BF16`) | implemented (`BF16`) | The same product writes `out[b,h,R,d]` from weight row `h*D+d`, after checked `O=H*D`; the landed writer scatters logical head columns without padded-column shuffling. |
| SDPA QK | blocked/native route present | blocked/native route present | blocked/native route present | blocked/native route present | For each `b,h`, BF16 `q[b,h,R,D]` multiplies the logical transpose of `k[b,g(h),0:L,D]` into FP32 scores `[R,L]`, followed by FP32 scale and causal/prefix mask. Non-tile D/L and GQA require custom address generation; TTNN's high-level output and FP32-accumulation contract are unproved. |
| SDPA PV | blocked/native route present | blocked/native route present | blocked/native route present | blocked/native route present | Explicit RNE-rounded BF16 `P[b,h,R,L]` multiplies `v[b,g(h),0:L,D]` with FP32 accumulation and writes or merges exactly `out[b,R,h*D+d]`. It may not fuse away the probability store, read masked V, or create repeated KV heads; the current positive-workspace and numerical paths are absent. |

These four row cases are shape-generic tile counts, not special kernels:
`P32(R)` is 32 for `R=1/15/16/17`, while logical readers and writers admit
exactly 1, 15, 16, or 17 rows. Physical padding is masked independently of the
SDPA condition `0<=t<L && t<=a+r`. The SDPA route also requires
`Hq%Hkv=0`, `0<L<=C`, `a<C`, and `R<=C-a`; K/V public owners remain
`[B...,Hkv,C,D]`, even when internal readers address only prefix `L`.

For BF16 linear, QK, and PV, configure a candidate kernel with
`math_approx_mode=false`, high fidelity, and 32-bit destination accumulation,
but do not infer conformance from those knobs. BF16 products must be accumulated
with the contract's FP32 behavior and stored with RNE. QK scale, maximum,
subtraction, exponentiation, and denominator remain FP32; masked probability is
exactly zero; the probability is then explicitly RNE-rounded to BF16 before PV;
PV accumulates in FP32 and the merged result is RNE-rounded to BF16. Installed
headers and the inventory run do not prove those mathematical properties.
Numerical comparison against the independent references, including cancellation
and long-K cases that distinguish partial BF16/TF32 accumulation, is therefore a
hard support gate. This BF16 assessment makes no additional F64 claim.

##### Checked dataflow, output, and scratch

The following table freezes the admissible data flow. `logical bytes` account
for caller data; `native capacity` accounts for 32x32 storage. Each
multiplication and the final sum of aligned subranges is checked. BF16 tiles
are 2,048 bytes and FP32 tiles are 4,096 bytes, so a Metalium buffer-backed
workspace route uses 4,096-byte alignment (which also satisfies the common
32-byte minimum). The operation query is pure: it computes these bounds from
validated values without allocating, registering, submitting, consulting
queue occupancy or allocator capacity, or depending on a native handle state.

| Purpose | Existing owner and logical traffic | Caller-owned temporary, if required | Native invocation and destination | Lifetime and checked capacity |
| --- | --- | --- | --- | --- |
| linear A/window | `x[B...,T,I]`; `checked(B*R*I*2)` bytes read | none for an offset/tail-aware reader; otherwise BF16 packed A of `checked(B*P32(R)*P32(I)*2)` | reader supplies `[R,I]` tiles to Metalium matmul | x and any packed A remain registered through native completion |
| linear B/orientation | shared persistent `w[O,I]`; logical weight bytes are not scratch | none for a transpose-aware reader; otherwise one BF16 packed `[I,O]` region of `checked(P32(I)*P32(O)*2)`, not one per plane | `matmul_init`/`matmul_tiles`; never an allocating `ttnn::transpose` | w and any pack remain live through every submitted plane; no duplicate persistent checkpoint copy |
| ordinary/head-planar result | ordinary output traffic `checked(B*R*O*2)`; head-planar is the equal checked value `B*H*R*D*2` | none when the writer maps result columns directly; otherwise BF16 product region `checked(B*P32(R)*P32(O)*2)` | writer stores into caller output `[B...,R,O]` or scatters `o=h*D+d` to `[B...,H,R,D]` | output and any product region remain leased until completion; result padding is not logical |
| QK operands | q traffic `checked(B*Hq*R*D*2)`; K initialized-prefix traffic `checked(B*Hkv*L*D*2)` from capacity owners `[B...,Hkv,C,D]` | no repeated K heads; optional tail-neutral BF16 pack is at most `checked(B*(Hq*P32(R)*P32(D)+Hkv*P32(L)*P32(D))*2)` | per-head native QK, with K read in transposed orientation and `g(h)` address mapping | Q/K and any pack live through QK; bytes for `L..C` are neither scratch nor logical input |
| QK scale/mask/scores | logical FP32 score requirement `checked(B*Hq*R*L*4)` | FP32 native32 scores `checked(B*Hq*P32(R)*P32(L)*4)` | matrix writer targets the score subrange; device-local scale, causal/prefix mask, max and sum operate there | score range remains exclusive through probability production |
| probability preparation | logical BF16 probability requirement `checked(B*Hq*R*L*2)` | distinct BF16 native32 P region `checked(B*Hq*P32(R)*P32(L)*2)` so the RNE boundary is observable | device-local FP32 softmax writes RNE BF16 P; masked cells and physical padding are initialized zero | scores and P may not overlap while score values are live; P remains through PV |
| PV and merge | V prefix traffic `checked(B*Hkv*L*D*2)`; PV logical result `checked(B*Hq*R*D*2)`; merged caller output `checked(B*R*Hq*D*2)` | no PV temporary when the writer maps `(h,d)` directly; otherwise BF16 native32 PV region `checked(B*Hq*P32(R)*P32(D)*2)` | native BF16 P/V matmul writes caller output at `h*D+d`, or writes the checked PV region followed by device-local merge | V, P, output, and any PV region remain live until the final native completion |

The query reports zero linear workspace only when direct readers and writers
eliminate every DRAM pack/product temporary. Otherwise it returns the aligned
sum of only the live packed regions for the chosen path. SDPA requires at least
nonoverlapping native32 score and P regions, plus a PV region only if direct
merge is unavailable; optional Q/K packs are added only when direct native32
readers cannot express the installed layout. Scratch never includes persistent
weights, Q/K/V owners, output, or duplicate caches. The scheduler may reuse
nonoverlapping-lifetime subranges only after their native completion is proven.

TTNN `create_workspace(bytes)` now owns real positive scratch, so the score/P
route is blocked by its own kernels and alignment needs rather than by the
absence of a native factory or by permission for a hidden tensor allocation.
The embedding-lookup leaf under
`.cswd/tasks/006-tinyllama/03-embedding-lookup` owns that minimal TTNN-private
`RawWorkspace` factory: one owning replicated DRAM `MeshBuffer` on the existing
mesh, exposed as exactly the requested logical bytes over one contiguous native
page whose page size is the checked request rounded up to 32 bytes, with
checked owner-absolute range access and release only after proven completion.
Creation, rebind, subrange addressability, exact-device checks, and
destruction/reset follow those semantics. Workspace and output are disjoint
from all operands and each other; new output/read and scratch/operand overlap
are rejected, while valid read/read overlap, including exact Q/K/V aliases, is
accepted.

One facade submission must enqueue the bounded program on the exact
`TtnnDevice` mesh command queue used by that `DeviceOps`, in call order and
without an internal host wait. It snapshots specs, native handles, plane
mappings, scalar values, and workspace ranges; registers each distinct owner;
and retains the workspace lease until the existing native fence proves
completion. A pre-admission failure consumes no sequence and mutates nothing.
After a positive OID, device failure is retained and rethrown by every repeated
wait. Workspace reset/destruction must wait for proven use or quarantine the
range; device destruction must drain or retain unresolved work. The high-level
matmul signature does not select an IOM queue or establish these lifetime
facts, which is another reason its optional output is not yet adopted.

##### Fit of all seven operations

| Planned facade | Installed facility and contract disposition |
| --- | --- |
| embedding | `ttnn::embedding` declares an optional output, but installed evidence does not verify every integral index carrier, bit-preserving BF16 payload, independent IOM planes, transformed mappings, no hidden temporary, or IOM queue/lifetime behavior. A bounded device-local gather can use the same owner/writer rules; current support is blocked, not replaced by a host index scan. |
| linear | Implemented for the mandatory `BF16` leaf by the direct per-plane Metalium route: tail-masked readers, the native matrix facility with FP32 accumulation, and one RNE BF16 writer per output tile row, with checked `O=H*D` head-planar scatter, no high-level transpose, no hidden temporary, and the pure `{0, 1}` requirement query. The twenty other applicable leaves are explicit capability rejections; the executed evidence is recorded in [Linear projections](#linear-projections). |
| rmsnorm | Installed `sum`/`mean`/`max` reductions return tensors and do not expose caller output, while unary `rsqrt` has an optional output. Composing them would allocate intermediates and does not prove logical-tail exclusion or wide reduction. A backend-private fused row reduction is required; current support is blocked. |
| rope | Installed unary `sin` and `cos` accept optional outputs, but composing transpose/arithmetic wrappers can allocate and does not prove FP32 angle/trig behavior or split-half pairing. A device-local kernel must write caller output, mask tails, and use `a+r`; current support is blocked. |
| cache append | Installed slice/data-movement APIs expose optional outputs, and Metalium readers/writers can address tiles, but partial native32 updates must preserve every cell outside `[a,a+R)`, including other logical rows and physical padding, without host staging. Exact destination-window, queue, and alias behavior is unproved; current support is blocked. |
| silu | Installed `ttnn::silu` declares an optional output and is a plausible direct route only after exact shape/dtype/memory configuration, no-hidden-temporary, stable wide evaluation, RNE, queue, and lifetime behavior are verified. Until then a backend-private unary kernel is required and support remains blocked. |
| sdpa | Installed `scaled_dot_product_attention` returns a new tensor and has no optional output; its public mask/layout and fused numerical boundaries do not establish IOM's GQA, capacity-owner, explicit BF16-P, caller-scratch, or merged-output contract. Use the bounded QK/softmax/PV flow above; current support is blocked. |

Allocating wrappers are never repaired by copying their result into the caller
output: that still violates the no-hidden-allocation rule. Likewise, current
host transfer/staging and elementwise binary paths prove none of gather,
reduction, trig, partial-tile mutation, unary numerics, linear, QK, or PV.

##### Dependency decision and remaining gates

**Decision: add no matrix library.** The already-required
`TTNN::TTNN`/`TT::Metalium` pair provides high-level matmul and the lower-level
Blackhole matrix/dataflow primitives. A BLAS package or umbrella dispatcher
would add discovery, link, redistribution, handle, and workspace lifecycle
without solving native32 addressing, BF16/FP32 fidelity, IOM queue retention,
or caller-owned output/scratch. Prefer the single bounded TTNN-private
Metalium program family above when the high-level optional-output path cannot
be proven exact. Do not add a plugin layer, a second handle system, or separate
matrix facilities for linear, QK, and PV. Absence of BLAS linkage is not a
capability gap, and unavailable hardware remains `Unsupported`.

The eventual TTNN linear and SDPA leaves must first recheck the exact installed
commit/API and fail TTNN-enabled configuration explicitly if their required
TTNN/Metalium features are absent; TTNN-disabled and CPU-only builds acquire no
new dependency. If later evidence shows even lower-level Metalium cannot meet
the contract, that operation leaf must document the precise failure before
proposing a minimal alternative with an exact package, minimum version,
imported target, private linkage, redistribution terms, runtime footprint,
context/handle initialization, caller-workspace binding, queue integration,
and reset/destruction synchronization. No such need is evidenced here.

Parent verification observed the configured TTNN smoke gate pass as recorded
above. Production ports later owe
`cmake --build build --target iom_ttnn_conformance_tests` and
`ctest --test-dir build --output-on-failure -R '^iom_ttnn_conformance_tests$' --timeout 300`,
plus native runtime/profiler evidence for ordinary and head-planar linear and
both QK and PV at logical `R=1/15/16/17`, non-tile I/O/D/L, independent
leading planes, GQA, tails, aliases, checked failures, in-order OIDs, and
repeated waits. None of those production gates or profiler runs was run by
this assessment.
#### TinyLlama forward layout — CUDA matrix feasibility

This is a bounded capability record for the planned interfaces above, not a
CUDA neural implementation or a support claim; each interface-owning port
carries its own evidence. On the inventoried device the native CUDA Toolkit
route is **supported for implementation feasibility** for ordinary and
head-planar linear, QK, and PV at every required row count. Production status
is **closed for linear** — the operation-owning port supplies the runtime
numerical, conformance, and execution-connected evidence recorded in
[Linear native evidence](#linear-native-evidence) — and remains **blocked for
QK, PV, RoPE, SiLU, and SDPA** until their own ports supply the same evidence.
A device below compute capability 8.0 is **unsupported** for this BF16 WMMA
route; it does not earn a fallback pass.

##### Linear native evidence

Recorded on 2026-09-18 through the configured `cuda` `csw-remote` profile from
the exact `run-task/006-tinyllama--04-linear-projections--06-cuda-native-bf16`
worktree, mirror `lp-cuda-bf16-a1`, host `bv1`. This record covers the native
BF16 linear specialization (`src/cuda/linear.cu`) only; it makes no QK, PV,
RoPE, SiLU, or SDPA claim.

- **Backend, device, and toolchain.** CUDA; NVIDIA GeForce RTX 5090, compute
  capability 12.0, driver 595.71.05; `nvcc` release 13.2 (`V13.2.78`); runtime
  and driver API 13020 as reported by `cudaRuntimeGetVersion` and
  `cudaDriverGetVersion` inside the conformance binary. The CUDA target pins
  `CUDA_ARCHITECTURES=75;80;90;120`, and the loaded image of the specialization
  reports its own architecture at runtime through the arch-dependent device
  constant of `src/cuda/linear.cu` (`linear_bf16_wmma_image_arch() = 1200`), so
  the executed image is the sm_120 one and not a pre-Ampere JIT image.
- **Kernel and exercised submissions.** Symbol
  `standard_tiled_linear_bf16_kernel` (CUDA-local `__global__`, eight warps of
  32 lanes, one warp per `(plane, head, row tile, column tile)` unit, launched
  by `launch_linear_bf16` on the queue's own in-order stream). The
  conformance binary emitted one record per required row run from the real
  queue/OID path (`cuda-linear-bf16-record ...` lines of
  `test/cuda/test_cuda_conformance.cpp`), each with `P=2`, `T=19`, `I=3`,
  `O=10`, `s=2`, padded `Rp/Op/Ip = 16/16/16` for `R=1,15,16` and `32/16/16`
  for `R=17`, accepted OID `36028797018963969` at sequence 1 of its fresh
  queue, and launch geometry `blocks=2/2/2/4` (ordinary, `H=1,D=10`) and
  `blocks=4/4/4/8` (head-planar, `H=2,D=5`) for `R=1,15,16,17`; every run was
  compared element-by-element against the shared independent reference and
  passed. A device below compute capability 8.0, or a build whose loaded image
  carries no BF16 WMMA statements, reports the leaf `Unsupported` instead of
  producing a record, and the difference is verified through the injected
  capability fact in
  `test/cuda/test_cuda_conformance.cpp` (`CUDA BF16 linear is Unsupported on a
  device without the WMMA facility`).
- **Observed native facility.** The executed `wmma` BF16/FP32 operation behaves
  as the tensor-core datapath and not as any per-step FP32 accumulation: for
  the discriminative product set `{+4.014e38, +4.014e38, -7.603e38}` (every
  product beyond `FLT_MAX`, exact sum `4.256e37`) the device returns
  `0x1p+125 = 4.2535e37`, while every left-to-right or tree FP32 accumulation
  order of the same products overflows to `±Inf`; BF16-subnormal products are
  preserved (`2^-133` exactly, no flush-to-zero), `NaN` multiplicands admit
  `NaN`, and `(+Inf, -Inf)` products give `NaN` while `+Inf` plus a finite
  product stays `+Inf`. The compiled image of that kernel contains
  `HMMA.16816.F32.BF16` (`cuobjdump -sass build/libiom_cuda.a`, six
  occurrences across the three architectures at or above the floor and none in
  the sm_75 image). The disassembly is supplementary here: capability is
  decided at runtime from the device attribute and the loaded image, never from
  compilation or disassembly presence.
- **Profiler.** `ncu` 2026.1.1.0 is installed, but counter collection is denied
  to this account (`ERR_NVGPUCTRPERM`; no passwordless root), so no
  counter-based instruction-level observation was obtained. `nsys` 2026.4.1
  does run: `nsys profile --force-overwrite=true -o /tmp/lp-nsys` around the
  native-evidence test case, read back with
  `nsys stats --report cuda_gpu_kern_sum`, reports the executed symbol
  `iom::cuda_detail::<unnamed>::standard_tiled_linear_bf16_kernel(iom::detail::LinearMetadata)`
  with eight instances — one per recorded submission, in the same order as the
  eight emitted records — beside the shared copy and gather kernels. The record
  therefore reports the executed kernel, its loaded image architecture, and the
  observed facility behaviour above, and claims no counter-based instruction
  observation; the missing counters are a host-permission limitation, not a
  device or implementation result.
- **Fixture class agreement.** The complete BF16 case matrix (33 cases, 45023
  reference elements) is scanned for the invariant that the mandated FP32
  fused-multiply-add recurrence and the FP64 equation rounded once to BF16
  agree in class and satisfy the frozen threshold:
  `class_disagreements=0`, `threshold_failures=0`. The scan is what motivated
  the fixture scale constraint recorded on `linear_fixture_special_code` in
  `test/backend/backend_conformance_linear.hpp`, which scales the BF16
  saturation-magnitude ladder position to `2^48`; before that constraint
  exactly one element (the LM-head case, element 11) disagreed, `+inf` against
  `-inf`, because no reassociating native reduction reproduces per-step FP32
  overflow.
- **Conclusion.** Supported on this device and this loaded image for `R=1`,
  `15`, `16`, and `17` in ordinary and head-planar mode, each established by an
  executed submission whose output was compared against the independent
  reference; a device without the facility reports `Unsupported`.

##### Evidence boundary and installed capability

The following evidence was collected on 2026-09-14 through the configured
`cuda` `csw-remote` profile in the unique
`forward-layout-cuda-feasibility` workspace:

- `nvcc --version` reported CUDA compilation tools 13.2,
  `V13.2.78`; `/usr/local/cuda` resolved to `/usr/local/cuda-13.2`.
  Installed package inventory reported `cuda-compiler-13-2` 13.2.1-1,
  `cuda-nvcc-13-2` 13.2.78-1, and `cuda-cudart-dev-13-2` 13.2.75-1.
  `dpkg-query -S` assigned both `mma.h` and `cuda_bf16.h` under the
  13.2 target include directory to `cuda-cudart-dev-13-2`.
- `nvidia-smi` reported driver 595.71.05, driver-supported CUDA 13.2,
  and one NVIDIA GeForce RTX 5090. A bounded Runtime API inventory executable
  reported header `CUDA_VERSION=13020`, `CUDART_VERSION=13020`, runtime
  13020, driver API 13020, compute capability 12.0, warp size 32, and
  33,670,758,400 bytes of global memory.
- A temporary installed-header translation unit instantiated
  `<mma.h>` BF16/FP32 fragments for `16x16x16` with all four
  row-major/column-major A/B combinations, and `32x8x16` and `8x32x16`
  with row-major A and column-major B. It used `__nv_bfloat16`
  multiplicands, `float` accumulators, `load_matrix_sync`, `mma_sync`, and
  row-major `store_matrix_sync`. It compiled successfully with
  `nvcc -std=c++20 -arch=sm_120`; its executable performed only the device
  inventory above. The translation unit was removed after the bounded sample.
  This is installed SDK/header and target-code-generation evidence, **not**
  a matrix launch, numerical result, throughput result, or profiler result.

The [CUDA 12.8 Warp Matrix Functions documentation][cuda-wmma] defines WMMA
as warp-cooperative `D=A*B+C`, permits `row_major` and `col_major`
multiplicands, and permits a row- or column-major accumulator store. Its
[alternate-floating-point section][cuda-wmma-alt] requires compute capability
8.0 or newer for BF16 Tensor Core use and requires `__nv_bfloat16` fragments
to use `float` accumulators. Its
[element-type and matrix-size table][cuda-wmma-sizes] lists BF16/FP32
`16x16x16`, `32x8x16`, and `8x32x16`. The canonical route below uses only
`16x16x16`; the other shapes are optional tuning choices, not semantic
requirements.

`load_matrix_sync` and `store_matrix_sync` require a 256-bit (32-byte)
aligned base. `ldm` is in elements and describes the distance between
successive rows or columns. The guide requires a 16-byte stride for the
listed 16-bit and FP32 cases, and states that BF16 has the same shapes and
operations as FP16. This route therefore uses BF16 leading dimensions
divisible by 8 and FP32 leading dimensions divisible by 4; padding every
native dimension to 16 satisfies both. Each warp sees identical fragment
parameters and participates unconditionally. Fragments never cross a
translation-unit or external ABI boundary; the guide explicitly calls their
register layout architecture-specific.

These installed and documented facts establish that the primitives needed by
a direct port exist on the sampled host. They do not establish current IOM
support. The repository currently links the CUDA target only to
`CUDA::cuda_driver` and `CUDA::cudart`, allocates exact-context standard-tiled
storage, uses one in-order CUDA stream per queue, and leaves the neural
capability probes unsupported. No CUDA linear, QK, PV, end-to-end
conformance, sanitizer, or profiler command was run for this assessment.
The repository facts above are bounded by `CMakeLists.txt`'s CUDA target,
`src/cuda/device_tensor.cpp`'s exact-context standard-tiled allocation,
`src/cuda/copy.cu`'s shared queue factory, device enumeration in
`test/cuda/test_cuda_smoke.cpp`, and the still-unsupported neural probes in
`test/cuda/test_cuda_conformance.cpp`.

##### Fixed ABI and native lowering

This route changes none of the preceding ABI. Calls return `oid`, are
`noexcept`, take const input views, a mutable caller output, `std::size_t`
positions and extents, and an optional final `RawWorkspaceView`. Their
workspace queries have the same semantic arguments, a const output, and no
workspace. A query validates shapes, aliases, capability, and every checked
size but never allocates, registers, submits, consumes an OID, inspects a
handle, or depends on arena capacity. Submission additionally validates the
workspace's exact device, capacity, 32-byte alignment, nonoverlap, and lease.
Negative OIDs report pre-acceptance failure; an accepted device failure
remains attached to its positive OID on every repeated wait.
Only `QuantizationFormat::NONE` is admitted. Valid read/read aliases remain
explicitly supported; output/read and workspace/operand/output overlap remain
invalid.

Let `P` be the checked product of the complete leading tuple (one when it is
empty), and define, with checked addition and multiplication,

```text
pad16(x) = checked_add(x, 15) / 16 * 16
A32(n)   = checked_align_up(n, 32)
Rp=pad16(R), Ip=pad16(I), Op=pad16(O),
Dp=pad16(D), Lp=pad16(L).
```

All runtime ranks remain 2 through 8 and dimensions are nonzero. A CUDA kernel
maps each transformed leading coordinate through its snapshotted owner offset
and strides; it never collapses or broadcasts independent planes. Standard
IOM storage is tile-major, with row-major values inside each 16x16 tile.
That is not one globally contiguous WMMA matrix, and an arbitrary `s` or
transformed plane is not a legal aligned WMMA base. Each warp therefore
copies only its current logical tile into an `alignas(32)` kernel-local shared
tile, writes zero to out-of-range physical cells, synchronizes, and loads the
WMMA fragment with leading dimension 16. This bounded on-chip tile is not an
allocation, persistent weight copy, or caller workspace. It also prevents
perturbed owner padding, uninitialized cache capacity, and excluded keys from
being read as multiplicands.

For linear, A is selected `x[b,s+r,i]` in row-major order. A rank-two
HF `w[O,I]` tile is staged without transposing checkpoint storage and is
interpreted as column-major B, so `B[i,o]=w[o,i]`. The result fragment is
stored to an aligned FP32 shared tile, converted once with RNE, and scattered
directly to ordinary `out[b,r,o]` or head-planar
`out[b,h,r,d]`, `o=h*D+d`. There is no bias, hidden row extraction, complete
weight pack, duplicate checkpoint allocation, or host work.

For QK, A is row-major Q `[R,D]`; the selected
`K[g(h),0:L,D]` tile is column-major B representing `K` transposed, where
`g(h)=floor(h/(Hq/Hkv))`. Only `t<L` and `t<=a+r` survives the separate
FP32 scale/mask step. For PV, rounded BF16 P is row-major A and the selected
V `[L,D]` tile is row-major B. K and V are addressed at their one owning
`Hkv` head for every query head; no buffer materializes `Hq/Hkv` copies.
QK and PV accumulate FP32. Stable max-subtracted softmax, exact zero for
masked P, RNE FP32-to-BF16 probability preparation, final RNE conversion, and
head merge all remain CUDA kernels on the same queue stream.

The logical/physical row mapping and status are:

| Logical `R` | Physical WMMA M | Ordinary linear | Head-planar linear | QK | PV |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |
| 15 | 16 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |
| 16 | 16 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |
| 17 | 32 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |

The same 16-padding applies independently to non-tile `I`, `O`, `D`, and
`L`. In linear the native problem is `(Rp,Op,Ip)`; in QK it is
`(Rp,Lp,Dp)`; in PV it is `(Rp,Dp,Lp)`. For `R=1/15`, physical rows through
15 are neutral staging only; for `R=17`, rows 17 through 31 are neutral.
They are never logical token rows, never enter a softmax reduction, and never
reach output. The analogous column and reduction tails are zero
multiplicands or discarded FP32 results. Public K/V owners remain
`[P,Hkv,C,D]`; packing never changes them to public `[P,Hkv,L,D]` views.
All four modes still require `Hq%Hkv=0`, `0<L<=C`, `a<C`, and
`R<=C-a`, and only `0<=t<L,t<=a+r` contributes.

##### Checked data flow and scratch

Every expression in this table is evaluated with checked products before
bytes are formed. `logical bytes` accounts for the required payload even when
the data stays in an existing owner. `segment` names caller workspace only;
kernel-local 16x16 shared tiles have a fixed launch-time size and lifetime.

| Product | Existing owner and logical bytes | Caller-owned temporary / checked segment | Native invocation and caller-owned result |
| --- | --- | --- | --- |
| Ordinary linear | selected BF16 X is `P*R*I*2`; persistent BF16 W is `O*I*2` logical bytes and is never scratch | none; tile-local staging handles `s`, strides, and neutral tails; query may return zero bytes | BF16/FP32 WMMA `(Rp,Op,Ip)`; RNE directly to standard-tiled `out[P,R,O]`, `P*R*O*2` logical bytes |
| Head-planar linear | the same X and persistent W; `O=H*D` is checked | none; no complete weight or output pack | the same WMMA product; RNE/scatter to `out[P,H,R,D]`, checked `P*H*R*D*2 = P*R*O*2` |
| QK | BF16 Q logical bytes `P*Hq*R*D*2`; K owner remains `P*Hkv*C*D*2`, while only its initialized causal prefix is staged tile-by-tile | `scores`: `A32(P*Hq*Rp*Lp*4)` bytes, 32-byte aligned; FP32 QK/scale/mask, alive from QK through probability preparation | BF16/FP32 WMMA `(Rp,Lp,Dp)` per query head and `g(h)`; logical FP32 scores account for `P*Hq*R*L*4`; excluded/padded cells are masked and are not tokens |
| Softmax / P preparation | logical FP32 scores are `P*Hq*R*L*4` | `probability`: `A32(P*Hq*Rp*Lp*2)` bytes after `scores`; stable FP32 row reduction writes exact-zero masked, RNE BF16 P; alive until PV completion | no matrix substitute: a device reduction/conversion kernel produces the observable BF16 boundary, whose logical payload is `P*Hq*R*L*2` |
| PV and merge | probability segment plus BF16 V owner `P*Hkv*C*D*2`; only mapped initialized V tiles are read | no KV-head expansion and no full PV buffer; an FP32 accumulator tile is kernel-local | BF16/FP32 WMMA `(Rp,Dp,Lp)`; the logical planar PV payload is `P*Hq*R*D*2` after RNE and is scattered directly, without duplicating it, to merged `out[P,R,Hq*D]`, also `P*R*Hq*D*2` |

The deterministic SDPA query therefore returns

```text
alignment = 32
score_bytes = A32(P*Hq*Rp*Lp*4)
probability_bytes = A32(P*Hq*Rp*Lp*2)
bytes = checked_add(score_bytes, probability_bytes)
```

with `probability` beginning at checked offset `score_bytes`. Both segments
belong to one live `RawWorkspace` on the queue's exact CUDA device and are
disjoint from Q, K, V, and output. Valid Q/K/V read/read overlap, including
exact aliases that independently satisfy their shape contracts, remains
allowed; output/read, scratch/read, scratch/output, and conflicting scratch
lease overlap is rejected. Linear needs no global temporary in this route,
so an empty workspace is valid exactly when its query reports zero.

Scores and P are per query head because those values differ under GQA; K and V
are not copied per query head. The score and probability segments coexist
while P is prepared. Once QK scores are no longer needed the score segment
may be reused only after the queued phase proves that lifetime transition;
the probability segment and all owners remain live through PV and its
completion event. Across submissions, neither segment is reusable before the
accepted OID completes successfully or fails terminally. Unknown completion
retains or quarantines the lease. Queue submission uses the existing stream,
snapshots metadata instead of retaining borrowed views, records completion
after all packing/matrix/reduction/merge kernels, preserves in-order OIDs,
and does not hide a stream synchronization. Sessions must still wait every
producer individually before submitting its consumer and drain every accepted
OID after a failure.

##### Seven-operation fit and incompatibilities

| Interface operation | CUDA assessment |
| --- | --- |
| gather | Contract-compatible device kernel for integral index payloads and bit-preserving BF16 table values; a device-discovered bad index must become an accepted retained failure. WMMA is irrelevant. A host index scan or round trip is forbidden. Current implementation remains blocked. |
| matmul | Supported feasibility on the sampled CC 12.0 device by direct BF16/FP32 WMMA with tile-local standard-layout staging. Global owner layout, arbitrary `s`, transformed leading strides, and physical tails are incompatible with direct unguarded `load_matrix_sync`; the bounded staging above is required. |
| reduction | Contract-compatible FP32 CUDA block/warp reduction for linear accumulation helpers, RMSNorm, and stable softmax. Reductions must exclude masked/padded cells and retain the specified wide intermediates; WMMA does not replace max, sum, or normalization. The CUDA RMSNorm wrapper is source-inspected, the linear accumulation helpers are implemented through the landed port (see [Linear projections](#linear-projections)), and the softmax kernel remains blocked pending its own SDPA port. |
| trig | Contract-compatible CUDA device FP32/wide-domain sine and cosine with finite positive `theta`; it neither uses nor is evidenced by WMMA. A host math substitute is forbidden. Current RoPE kernel remains blocked. |
| partial-tile copy | Existing CUDA standard-tiled copy machinery establishes device-local tile addressing, but not neural matrix support. A native port must use guarded logical loads/stores and neutral shared cells so owner padding, cache capacity tail, and rows outside `R` are unobservable. |
| unary | Contract-compatible CUDA device FP32/wide-domain SiLU and BF16 RNE output. It is elementwise by design and is not an invalid matrix substitute. Current kernel remains blocked. |
| attention | Supported matrix feasibility for both QK and PV, including GQA and every required `R`, only as the complete device-local flow above. FP32 masking/softmax and explicit P-to-BF16 preparation are mandatory. Elementwise QK/PV, repeated KV heads, host work, hidden allocation, or treating physical rows as tokens is incompatible. Current SDPA port and evidence remain blocked. |

This matrix result says nothing about the other applicable dtype leaves; their
kernel selection, numerics, and tolerances remain with the operation-owned
specifications. In particular this record neither adds nor narrows any F64
path.

##### Dependency decision and remaining gates

**Decision: no added matrix library.** The installed `<mma.h>`,
`cuda_bf16.h`, compiler, driver, and runtime provide the required direct
primitive, orientation combinations, target code generation, and queue-local
kernel launch route. The project already discovers the CUDA Toolkit and links
the CUDA backend privately to the driver and runtime. The only identified
layout mismatch is solved by bounded device shared-memory tiles and checked
caller scratch; it is not a reason to add a BLAS dependency. Absence of
cuBLAS linkage is therefore not a gap, cuBLAS is not a default fallback, and
the first CUDA linear/SDPA ports must reuse this backend-private direct route.

This decision also avoids a concrete lifecycle conflict. The
[cuBLAS `cublasCreate`/`cublasDestroy` documentation][cublas-create] says
creation allocates host and device resources and destruction implicitly calls
`cudaDeviceSynchronize`. [`cublasSetStream`][cublas-stream] unconditionally
resets a user workspace to the default pool.
[`cublasSetWorkspace`][cublas-workspace] binds user-owned device memory only
for the current stream, requires 256-byte alignment, and warns that calls must
be serialized while kernels retain it. Those handle allocations, default-pool
fallback, stream rebinding, and destruction synchronization are not
allocation-neutral conveniences and cannot override caller-owned scratch,
exact-stream lifetime, or asynchronous teardown rules. If a future consuming
port finds a direct-WMMA blocker, it must first document that concrete blocker
and reject any library route that cannot write caller output, bind
deterministic caller scratch, initialize per-context bookkeeping before
submission, rebind after every stream change, and retain resources through
completion. No such blocker is evidenced here.

The following are exact reproducibility and future production commands, run
from the assigned local worktree through the same unique remote workspace:

```sh
.omp/csw/bin/csw-remote-sync cuda forward-layout-cuda-feasibility
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility 'nvcc --version && nvidia-smi'
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader'
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility "dpkg-query -W 'cuda-*'"
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility 'cmake --build build --target iom_cuda_conformance_tests'
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility "ctest --test-dir build --output-on-failure -R '^iom_cuda_conformance_tests$'"
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility "ncu --set full --target-processes all --kernel-name regex:'.*(linear|qk|pv).*' ./build/test/iom_cuda_conformance_tests --test-case='CUDA TinyLlama BF16 matrix paths cover R=1,15,16,17'"
```

The first four inventory/capability steps were run for this assessment; the
last three are deliberately unrun production-port gates. The future
conformance case must cover ordinary and head-planar linear plus both QK and
PV at logical `R=1,15,16,17`, non-tile `I/O/D/L`, independent transformed
planes, GQA, perturbed physical padding and excluded K/V rows, exact
BF16/FP32/RNE boundaries, every scratch rejection, accepted failure with
repeat waits, and no host traffic or hidden allocation. The profiler must
show the direct matrix kernels for both decode and prefill; inventory,
successful compilation, or one GEMM cannot substitute for that evidence.

[cuda-wmma]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-matrix-functions
[cuda-wmma-alt]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#alternate-floating-point
[cuda-wmma-sizes]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes
[cublas-create]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
[cublas-stream]: https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream
[cublas-workspace]: https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace
#### TinyLlama forward layout — ROCm matrix feasibility

This is a bounded feasibility record for the planned ABI above, not a ROCm
implementation or a support advertisement. The existing ROCm neural probes
continue to return negative `Unsupported`; the observations below do not
replace operation conformance, a production kernel run, or profiler evidence.
They establish that compiler-native BF16 matrix instructions are a viable
implementation route on a checked supporting target without adding a matrix
library.

##### Evidence boundary and installed target

The evidence was collected on the configured `rocm` host from the exact
task worktree, synchronized to the unique remote workspace
`forward-layout-rocm-feasibility`. The following evidence classes are kept
separate:

| Evidence class | Observation | What it does not prove |
| --- | --- | --- |
| Project source declaration | `CMakeLists.txt` requires HIP at least 7.2 and privately links `hip::host`; ROCm storage uses the standard tiled allocation and the existing in-order GPU queue. | A version constraint, storage support, or link target does not prove a native matrix kernel. |
| Installed compiler/SDK | `hipcc --version` reported HIP `7.15.26333-0000000`, AMD Clang `23.0.0git`, commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`; `hipconfig --full` reported `HIP_PATH=ROCM_PATH=/opt/rocm/core-10.0`; `.info/version` reported ROCm `10.0.0`; `hip_version.h` reports `7.15.26333`; Clang's resource directory is `/opt/rocm/core-10.0/lib/llvm/lib/clang/23`. The installation is not represented by the queried Debian `hip-*`/`rocm-*` package names, so those names are not used as version evidence. | Upstream-main or ROCm-7.2.4 source is not assumed to equal this installed compiler. |
| Installed target definitions | Installed generated `clang/Basic/BuiltinsAMDGPU.inc` exposes the BF16-to-FP32 GFX11 WMMA `_w32`/`_w64`, GFX12 `_w32_gfx12`/`_w64_gfx12`, and BF16 MFMA names. Its guards are respectively `wmma-256b-insts` plus the selected wave size, `wmma-128b-insts` plus the selected wave size, and `mai-insts`. | A name in a compiler table is not proof that it is invocable on every target. |
| Installed runtime/device | `rocminfo` reported HSA runtime `1.21`, runtime extension `1.30`, a Radeon AI PRO R9700 `gfx1201` agent with wavefront size 32, and a separate `gfx1036` agent. `hipGetDeviceProperties` enumerated ordinal 0 as the R9700 (`gfx1201`, `warpSize=32`, 32 reported multiprocessors, 1024 maximum threads per block, 34,208,743,424 bytes global memory) and ordinal 1 as `gfx1036`; `hipRuntimeGetVersion` and `hipDriverGetVersion` both returned `70152801`. AMD's [GPU specification table](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html) independently identifies the R9700 as RDNA4/gfx1201 with wave32 or wave64. | The inventory does not make the older `gfx1036` matrix-capable and does not benchmark the R9700. |
| Bounded capability samples | `hipcc --offload-arch=gfx1201 -O2` compiled a 32-lane kernel calling `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`; execution on ordinal 0 completed and preserved an FP32 accumulator for a zero BF16 product (`gfx12_w32_bf16_wmma_zero_product=pass`). The same installed compiler compiled, but did not run, a `--offload-arch=gfx1100` kernel calling the GFX11 `_w32` builtin with its replicated input vector shape. Both temporary `_local` sources and the remote workspace were removed. | The zero-product sample proves target compilation, dispatch, all-wave participation, and FP32 result transport only. It is not a layout, tail, numerical, throughput, or production conformance test; the GFX11 result is compile-only. |
| Deliberately unrun production gates | No `iom_rocm_conformance_tests` build or CTest run, production native linear/QK/PV kernel, numerical matrix, profiler, tuning run, or all-five gate was executed. | No current ROCm neural support or performance conclusion follows. |

The exact inventory command was:

```text
.omp/csw/bin/csw-remote-exec rocm \
  forward-layout-rocm-feasibility 'hipcc --version && rocminfo'
```

The bounded runtime probe was compiled with
`hipcc --offload-arch=gfx1201 -O2` and used
`hipRuntimeGetVersion`, `hipDriverGetVersion`,
`hipGetDeviceProperties`, a 32-thread launch, and
`hipDeviceSynchronize`. A separate compile-only probe used
`hipcc --offload-arch=gfx1100 -O2 -c`. These observations are reproducible
installed facts, while the eventual production commands remain:

```text
cmake --build build --target iom_rocm_conformance_tests
ctest --test-dir build --output-on-failure \
  -R '^iom_rocm_conformance_tests$'
```

Those commands must run remotely after a port exists and must then cover
native linear, QK, and PV at logical prefill and `R=1`; they were not run for
this assessment.

##### Native instruction route

The [ROCm 7.2.4 release-pinned builtin definitions](https://raw.githubusercontent.com/ROCm/llvm-project/rocm-7.2.4/clang/include/clang/Basic/BuiltinsAMDGPU.def)
define BF16-to-FP32 WMMA `_w32` and `_w64`, BF16 MFMA including
`__builtin_amdgcn_mfma_f32_16x16x8bf16`, and the GFX12-suffixed WMMA
variants. The current [upstream definitions](https://raw.githubusercontent.com/llvm/llvm-project/main/clang/include/clang/Basic/BuiltinsAMDGPU.td)
and [builtin documentation](https://raw.githubusercontent.com/llvm/llvm-project/main/clang/include/clang/Basic/BuiltinsAMDGPUDocs.td)
give the target-feature and operand details. AMD's
[RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/) independently
documents the cooperative `D=A*B+C` 16x16x16 operation and GFX11
replication. The general
[HIP C++ language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)
describe kernels, waves, and stream launches but expose no CUDA-style
`nvcuda::wmma` facade; that omission is not evidence that AMDGPU compiler
builtins are absent.

For the installed, selected `gfx1201` wave32 target the route is exactly
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`. Each lane supplies
two `_ExtVector<8, short>` BF16-bit fragments and one
`_ExtVector<8, float>` accumulator and receives an
`_ExtVector<8, float>` result. The operation is a 16x16x16 `A*B+C`;
all 32 lanes must participate with a converged execution mask. The `short`
elements carry the 16-bit BF16 representation, not integer numerical input,
and accumulation/output registers are FP32. Installed HIP declares raw BF16
as a two-byte-aligned `unsigned short` payload and provides `__bf16`/
`__hip_bfloat16` conversion support; operation stores must still demonstrate
the contract's explicit RNE boundaries.

GFX11 wave32 instead uses
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` with
`_ExtVector<16, short>` A and B fragments and `_ExtVector<8, float>` C/D.
The full 256-bit A/B operands encode the required two-copy replication
between half-waves. Its wave64 variant has four-copy GFX11 replication and
four FP32 C/D elements per lane. GFX12 removes explicit A/B replication:
the `_w32_gfx12` operands are 128 bits and `_w64_gfx12` operands are 64 bits.
The suffixes and vector widths are ABI-significant and may not be selected by
a runtime branch inside one incorrectly compiled device image.

BF16 MFMA remains a distinct CDNA/`mai-insts` route. For example,
`__builtin_amdgcn_mfma_f32_16x16x8bf16` consumes two two-short vectors plus
a four-float accumulator and control immediates and returns four floats per
lane. It is suitable only in an image compiled for a matching MFMA target and
wave64 execution. The installed `gfx1201` target does not satisfy `mai-insts`;
its checked route is GFX12 WMMA, not MFMA. No installed CDNA device was
available, so MFMA runtime feasibility is **blocked on a matching device
sample**, rather than inferred from compiler-table presence. Conversely,
`gfx1036` has neither the checked GFX11/GFX12 WMMA route nor checked BF16 MFMA
and remains unsupported for these matrix products.

Direct builtins take registers, not pointers, strides, alignments, or layout
tags. The backend-private loader therefore owns every memory-layout decision.
It loads logical A row-major and emits logical C/D row-major. For linear and
QK it presents B as column-major without changing the public owner:
`B[k,o]=w[o,k]` preserves HF `[out,in]`, and
`B[d,t]=k[g(h),t,d]` is K-transpose at fragment load. PV presents
`A[r,t]=P[r,t]` and row-major `B[t,d]=V[g(h),t,d]`. Standard tiled owner
offsets are resolved element-wise; transformed leading offsets and strides are
honored per plane. A scalar BF16 loader needs only the owner's guaranteed
two-byte element alignment. Packed scratch rows begin at 32-byte-aligned
subranges with multiples-of-16 leading dimensions. Any vectorized load must
prove its stronger alignment or fall back to scalar/coalesced loads; the
builtin itself justifies no pointer-alignment assumption. The
[rocWMMA API reference](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html)
confirms that row/column combinations are representable, but rocWMMA is not
required for this direct route.

##### Logical products and tail mapping

Let `P` be the checked product of all independent leading/batch extents and
let `pad16(n)` be checked round-up to a multiple of 16. Define
`Mp=pad16(R)`, `Ip=pad16(I)`, `Op=pad16(O)`, `Lp=pad16(L)`, and
`Dp=pad16(D)`. Every physical tile is 16x16; K loops advance by 16. The
following matrix is the feasibility result on the installed `gfx1201`.
“Feasible” means the compiler/device route and a caller-scratch mapping exist,
not that the operation is currently implemented.

| Product/mode | Logical `R` | Physical mapping | Result |
| --- | ---: | --- | --- |
| ordinary linear | 1 | `M=16`, `K=Ip`, `N=Op`; store only row 0 and `o<O` | Feasible on checked gfx1201 WMMA and implemented by the ROCm native `BF16` specialization on that device (see [Linear projections](#linear-projections)); the twenty scalar leaves use the shared raw-word kernel. |
| ordinary linear | 15 | `M=16`; row 15 is neutral padding and is never logical | Feasible, including non-tile `I/O` and independent planes. |
| ordinary linear | 16 | one logical M tile | Feasible, including non-tile `I/O`. |
| ordinary linear | 17 | `M=32`; rows 17..31 are neutral padding | Feasible; padded rows are discarded, not extra tokens. |
| head-planar linear | 1 | same product, then `o=h*D+d` into `[H,1,D]` | Feasible with checked `O=H*D`; no head broadcast. |
| head-planar linear | 15 | `M=16`, `N=Op`; logical head scatter only | Feasible for non-tile `D/O` and independent planes. |
| head-planar linear | 16 | one logical M tile; logical `h,d` mapping retained | Feasible. |
| head-planar linear | 17 | two M tiles; padded rows discarded before head scatter | Feasible. |
| QK | 1 | `M=16`, `K=Dp`, `N=Lp`; cached `L` remains independent of `R` | Feasible with GQA head map and separate initialized/causal masks. |
| QK | 15 | `M=16`; full prefill has `Lp=16`, generic cached use has `pad16(L)` | Feasible; physical K/L padding is masked separately. |
| QK | 16 | one M tile; full prefill has one N tile | Feasible. |
| QK | 17 | `M=32`; full prefill has `Lp=32` | Feasible; padded rows/columns never enter softmax. |
| PV | 1 | `M=16`, `K=Lp`, `N=Dp`; masked P entries are exact BF16 zero | Feasible with native BF16 P/V and FP32 accumulation. |
| PV | 15 | `M=16`; non-tile `L/D` use neutral K/N tails | Feasible. |
| PV | 16 | one logical M tile | Feasible. |
| PV | 17 | `M=32`; padded output rows and `d>=D` are discarded | Feasible; output then merges to `[R,Hq*D]`. |

For every row of this table, linear computes the selected source window
`x[b,s+r,*]` and never infers `R` from `s`. Q owns `[P,Hq,R,D]`; K and V
owners remain `[P,Hkv,C,D]` and only their initialized prefix `0..L-1` is
packed. `G=Hq/Hkv` and `g(h)=floor(h/G)` select a KV head at load time; no
full KV-head replication is stored. QK accumulates BF16 products in FP32,
applies `1/sqrt(D)` and both initialized-prefix and `t<=a+r` masks in FP32,
then uses stable FP32 softmax. Masked probabilities are exactly zero. Every
logical probability is RNE-rounded to BF16 before the BF16 P/V WMMA loop,
whose FP32 result is RNE-rounded to BF16 before the device-local head merge.
Excluded K/V rows are not read; initialized neutral packed padding is not a
logical token.

##### Caller-owned flow and checked scratch

Every segment below starts at `align_up(previous_end,32)` in one exact-device
`RawWorkspaceView`; every multiplication, `pad16`, byte conversion, alignment
round-up, and segment addition is checked. `A32(x)` below means checked
`align_up(x,32)`. The query reports the aligned sum and alignment 32 without
allocating, inspecting allocator capacity, creating a handle, or submitting.

| Segment | Checked payload bytes before `A32` | Purpose and lifetime |
| --- | --- | --- |
| linear `x_pack` | `P*Mp*Ip*2` | Gather exactly `x[b,s+r,i]`, zero physical M/K tails, and retain through the last linear WMMA read. This includes at least `P*R*I*2` logical bytes. |
| linear `y_pack` | `P*Mp*Op*2` | RNE BF16 matrix result before ordinary copy or `o=h*D+d` head-planar scatter to caller output; retain through that copy. This includes at least `P*R*O*2` logical bytes. |
| SDPA `q_pack` | `P*Hq*Mp*Dp*2` | Head-planar Q with neutral M/K tails; retain through QK. |
| SDPA `kv_pack` | `P*Hkv*Lp*Dp*2` | Pack only K rows `t<L` for QK, then reuse the same non-overlapping-in-time segment for only V rows `t<L` before PV. No `Hq` KV replication and no capacity tail. |
| SDPA `scores` | `P*Hq*Mp*Lp*4` | FP32 QK output, scale, mask, max, and sum storage; includes at least `P*Hq*R*L*4` logical score bytes and retains through probability production. |
| SDPA `p_bf16` | `P*Hq*Mp*Lp*2` | Explicit RNE BF16 probability boundary; masked and physical-tail entries are zero; retain through PV. Includes at least `P*Hq*R*L*2` logical bytes. |
| SDPA `pv_bf16` | `P*Hq*Mp*Dp*2` | Per-head RNE BF16 PV result; retain through merge. Includes at least `P*Hq*R*D*2` logical bytes. |
| SDPA `merged` | `P*R*(Hq*D)*2` | Checked `Hq*D` and device-local `[P,R,Hq*D]` merge before copy to the caller-owned standard-tiled output. |

The conservative linear requirement is
`A32(P*Mp*Ip*2)+A32(P*Mp*Op*2)`. The conservative SDPA requirement is the
aligned sum of the six SDPA rows, with `kv_pack` reused sequentially for K and
V. An implementation may prove a smaller direct-load/store requirement, but
it may not omit a logically required value, use hidden allocation, or make
the pure query depend on queue or allocator state. Persistent checkpoint
weights are read in place and are never scratch, transposed persistently, or
duplicated. Cache capacity `C` is not charged as prefix scratch, and neither
K nor V is materialized per query head.

The product flow is consequently:

| Product | Existing owner to temporary | Native invocation | Caller-owned result |
| --- | --- | --- | --- |
| ordinary linear | standard-tiled `x[...,T,I]` selected rows to `x_pack`; standard-tiled HF `w[O,I]` loaded in place as conceptual column-major B | one full wave per 16x16 C tile, K-loop over `Ip`, GFX12 BF16 WMMA to FP32 registers | RNE to `y_pack`, then logical `[P,R,O]` copy |
| head-planar linear | same input/weight path; no persistent weight packing | same product | RNE to `y_pack`, then `o=h*D+d` scatter to `[P,H,R,D]` |
| QK | standard-tiled Q to `q_pack`; only K prefix `L` to per-`Hkv` `kv_pack` | per `P,h` QK with `g(h)`, FP32 WMMA accumulation | FP32 `scores`, device-local scale/mask/stable softmax, RNE to `p_bf16` |
| PV | reuse `kv_pack` for only the V prefix; read `p_bf16` | per `P,h` BF16 P/V WMMA with FP32 accumulation | RNE to `pv_bf16`, device-local head merge to `merged`, logical copy to caller output |

Submission validates ranks 2..8 as applicable, nonzero runtime
`B,R,I,O,Hq,Hkv,D,L,C`, mode/window and GQA rules, transformed leading
mappings, `QuantizationFormat::NONE`, aliases, exact device, checked
arithmetic, and the actual workspace before registration or OID consumption.
Output/read, scratch/read, and scratch/output overlap are rejected. Valid
read/read overlap, including exact Q/K/V aliases that independently satisfy
their shapes, remains allowed. The HIP stream is the existing queue's stream;
there is no default-stream detour. The queue retains every operand/output owner
and the scratch lease through terminal completion. A full wave stays active
at every builtin, while tail predicates apply only to loads/stores and
device-local mask stages.

Every accepted call returns its positive OID before asynchronous completion.
Negative admission errors, producer-success waits, repeated observation of an
accepted failure, and exact-device destruction/quarantine follow the existing
queue contract unchanged. No consumer is submitted after a failed producer,
and K/V initialized length is published only after both append OIDs have
completed successfully.

##### Seven-operation fit and dependency decision

| Planned operation | ROCm feasibility and concrete incompatibility |
| --- | --- |
| embedding/gather | A device-local index-and-copy kernel can preserve BF16 payload bits and independent planes under the same queue/alias rules. WMMA is irrelevant. Device-resident invalid indices may fail after acceptance; a host scan or round trip is forbidden. |
| linear | The checked GFX12 WMMA route covers ordinary and head-planar BF16 products above. GFX11 may use its replicated-input variant; matching CDNA may use MFMA after device proof. `gfx1036` is unsupported. No host or elementwise matrix substitute is acceptable. |
| RMSNorm/reduction | Device-local FP32 accumulation/reduction, reciprocal square root, scaling, and one BF16 RNE store fit the queue. WMMA does not perform the reduction; tiled padding and other planes must be excluded. Detailed kernel and tolerances remain operation-owned. |
| RoPE/trig | Device-local FP32 angle/sine/cosine and split-half rotation fit; the `double theta` ABI does not require FP64 device trig. WMMA is irrelevant. Host trig tables/round trips, adjacent-pair rotation, and padded features are incompatible. |
| cache append/partial-tile copy | Existing device-local tiled copy machinery is a suitable basis for a bit-preserving logical row append, but it must preserve all other logical rows and physical padding. WMMA is irrelevant; K and V remain separate owners/submissions. |
| SiLU/unary | Device-local stable FP32 unary evaluation followed by one BF16 RNE store fits. It remains a distinct operation before existing `mul`; WMMA does not justify fusion or hidden scratch. |
| SDPA/attention | WMMA covers QK and PV only. Separate device-local FP32 scale, causal/prefix mask, max/sum/exp softmax, explicit BF16 probability preparation, and head merge are required. Missing either native product, reading excluded K/V, full KV-head replication, exact-FP32 P in PV, host work, or hidden allocation is incompatible. |

**Decision: no added matrix library.** The installed compiler, HIP runtime,
and checked `gfx1201` device provide the direct GFX12 wave32 BF16-to-FP32 WMMA
route. The release-pinned and installed builtin tables also define the GFX11
and MFMA alternatives. Backend-private fragment loaders are enough to preserve
the fixed layouts and caller workspace; a missing CUDA-style HIP WMMA facade
is not a reason to add a dependency.

If direct builtins later prove insufficient on a required supporting target,
the first facility to assess is rocWMMA, not rocBLAS. The
[rocWMMA programming guide](https://rocwmma.readthedocs.io/en/latest/conceptual/programmers-guide.html)
documents a header-only C++17 wrapper around AMDGCN intrinsics, all-wave
participation, and no external kernel invocation. No rocWMMA package, minimum
version, CMake target, redistribution footprint, or fallback is prescribed
here because the checked direct route needs none. rocBLAS is not selected:
its [programming guide](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/Programmers_Guide.html)
requires a device/stream-bound handle, documents synchronizing stream changes,
and documents default temporary allocation as a synchronizing event. Those
handle and allocation semantics would require separate proof of pure
requirements, deterministic caller scratch, stream rebinding, reset/destruction
synchronization, and retained lifetime, with no demonstrated benefit for this
fixed route.

The first production consumer is the ROCm leaf under
`.cswd/tasks/006-tinyllama/04-linear-projections`; its private loader/kernel is
then reused by the ROCm SDPA leaf under
`.cswd/tasks/006-tinyllama/08-causal-grouped-attention`. Enabled ROCm builds
need no new discovery or private link. At runtime, a compiler image/device
without a matching checked WMMA/MFMA feature fails capability explicitly as
`Unsupported`; it never turns an unsupported device into a dependency pass.
Any future need for a helper must be established by those consuming leaves
with exact package/header/imported-target, minimum-version, license,
compile/link/runtime, handle, stream, workspace, and lifetime evidence before
changing enabled-ROCm-only CMake. CPU-only and ROCm-disabled builds acquire no
such dependency.
#### TinyLlama forward layout — SYCL matrix feasibility

This is the bounded feasibility record for the seven-operation ABI. Embedding
lookup has its device-native raw-word `parallel_for` port, and linear
projections are implemented as the twenty non-BF16 leaves on the
operation-local in-order path plus the native BF16 `joint_matrix` route with
explicit RNE packing, recorded in [Linear projections](#linear-projections).
RMSNorm, RoPE, cache append, SiLU, and SDPA remain unimplemented. A future port
MUST preserve the signatures, pure requirement queries, validation precedence,
owner rules, and producer-wait schedule above; SYCL types remain
backend-private.

The evidence layers are deliberately separate:

| Evidence layer | Observation on 2026-09-14 | What it establishes |
| --- | --- | --- |
| Repository declaration | `CMakeLists.txt` selects `icpx`/`dpcpp`, `sycl/sycl.hpp`, and `libsycl` with `-fsycl`, but has no compiler-version or matrix-extension check. Standard BF16 storage uses 16x16 row-major physical tiles in device USM from the exact context; the data arena and every suballocation are 32-byte aligned. | Storage and build wiring only; neither is native matrix evidence. |
| Installed compiler and headers | `icpx --version` reported Intel oneAPI DPC++/C++ Compiler `2026.1.0 (2026.1.0.20260617)`, package `intel-oneapi-compiler-dpcpp-cpp-2026.1 2026.1.0-235`, and `SYCL_EXT_ONEAPI_MATRIX=1`. The 2026.1 installation contains `sycl/ext/oneapi/matrix/{query-types.hpp,matrix.hpp,matrix-unified.hpp,matrix-intel.hpp}` dated 2026-06-17; the API remains in `sycl::ext::oneapi::experimental`. | The experimental API and runtime query compile in the installed SDK. It is not a stability promise. |
| Installed runtime and device | Required `sycl-ls` enumeration reported two Level Zero V2 devices, each `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, driver `1.15.38646+7`, PCI device `8086:e211`, architecture `intel_gpu_bmg_g21`, and subgroup sizes `16,32`. PCI inventory reported the in-kernel `xe` driver on Linux `7.0.0-31-generic`. The selected device reports `aspect::ext_intel_matrix=true` and maximum work-group size 1024. | An eligible Level Zero XMX device and subgroup 16 are installed. The eventual queue MUST bind one exact enumerated device, not an OpenCL or other-device fallback. |
| Documented native capability | The installed `matrix_combinations` query returned 53 combinations per B60. For `A=BF16,B=BF16,C=FP32,D=FP32`, it returned continuous `M<=8,N=16,K=16`, exact `16x16x16`, and exact `1x64x16`, `32x64x16`, `1x64x32`, and `32x64x32`. It also returned BF16-output variants, but this contract does not rely on their conversion rounding. | `M=1`, `M=16`, and a `16+1` decomposition are legal with BF16 inputs and FP32 accumulation; N/K tails require physical padding or tile decomposition. |
| Bounded sample | A removed standalone sample allocated device USM, required subgroup 16, invoked `joint_matrix_load`, `joint_matrix_mad`, and `joint_matrix_store`, and checked all-one BF16 products. Separate Level Zero executions printed `BF16xBF16->FP32 1x16x16 PASS` and `16x16x16 PASS`. | Representative native XMX execution exists for the row shapes used below. It does not implement IOM linear, QK, or PV and is not conformance or profiler evidence. |
| Production evidence (2026-09-14 snapshot) | Not run at that date: IOM linear, QK, PV, masking, softmax, RNE packing, head merge, all shape cases, conformance, tuning, and profiling. | At that date the SYCL port was blocked/unimplemented, and inventory or the sample cannot close a native-operation gate. The linear portion is since superseded by the implemented port and its executed record in [Linear projections](#linear-projections); QK, PV, masking, softmax, RNE packing, head merge, and the remaining operations are still unrun. |

These conclusions use the
[experimental matrix extension at revision `cf12c378`](https://github.com/intel/llvm/blob/cf12c3783cc6a7adaf76e54c6a4f11f81ec8599b/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc),
the [Intel compiler 2026.1 release record](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html),
and the
[oneMath usage model at revision `0feb864`](https://github.com/uxlfoundation/oneMath/blob/0feb864ddaf49d12aa35e5492f6899fa823d9d9b/README.md).
The extension identifies itself as experimental, supports subgroup scope only,
requires A/B to be row- or column-major and accumulators to have dynamic
layout, and says an unsupported type/shape submission fails synchronously with
`kernel_not_supported`. For `intel_gpu_bmg_g21`, its XMX table contains the
queried BF16/FP32 combinations. Its installed-device restrictions require
`stride*sizeof(element)` to be a multiple of 8 and at most `2^24`, and a
4-byte-aligned load/store base. IOM's 32-byte arena, 16-element tile strides,
and 512-byte BF16 or 1024-byte FP32 tile steps satisfy the alignment minima,
but every offset, stride product, and byte address still requires checked
validation. A future kernel MUST require subgroup 16 explicitly; subgroup 32
was enumerated but was not sampled as a matrix execution shape.

**Product/row decision.** Let `p16(x)` mean checked round-up to 16. Every
entry below is *native-shape feasible on the enumerated B60*: the device query
supports the decomposition and the two primitive M shapes executed. The two
linear rows are now implemented by the landed SYCL linear port and evidenced
in [Linear projections](#linear-projections); the QK and PV rows remain
unlanded production ports, and each such product must keep returning
`Unsupported` until its real SYCL implementation and native tests land.

| Product | logical `R=1` | logical `R=15` | logical `R=16` | logical `R=17` |
| --- | --- | --- | --- | --- |
| Ordinary linear, `X[R,I] * W[O,I]^T` | One queried/sampled `1x16x16` M/N/K tile family. | One queried `16x16x16` family with physical row 15 zero and never stored as a logical row. | One queried/sampled `16x16x16` family. | One `16x16x16` family plus one queried/sampled `1x16x16` M tail. |
| Head-planar linear, same product followed by `o=h*D+d` | Same M decomposition; a device-local RNE/scatter writes only `[h,0,d]`. | Same 16-row physical tile; the neutral row is discarded before head-planar output. | Same exact 16-row tile. | Same 16+1 decomposition; scatter honors each head's independent output plane. |
| QK, `Q[h,R,D] * K[g(h),L,D]^T` | Same M=1 family; N walks 16-key tiles and K walks 16-wide D tiles. | Same padded M=16 family; causal masking excludes the neutral row and key tails. | Same exact M=16 family. | Same 16+1 family; each query head directly selects `g(h)` without expanding a KV head. |
| PV, `P[h,R,L] * V[g(h),L,D]` | Same M=1 family after FP32 softmax and explicit RNE BF16 P preparation. | Same padded M=16 family; padded probabilities and V/D tails are neutral. | Same exact M=16 family. | Same 16+1 family; FP32 results are explicitly RNE-packed before merge. |

The standard physical tile order makes full 16x16 X/Q/P and W/K/V tiles
directly loadable. X, Q, P, and V are row-major A/B operands. HF weights remain
the sole persistent `[out,in]` owner: a row-major W tile is interpreted as the
column-major B tile of `W^T`, so there is no persistent transpose. Likewise a
row-major K cache tile is the column-major B tile of QK. No checkpoint copy and
no repeated `Hq/Hkv` KV materialization are permitted. If an installed
compiler/device cannot accept those column-major loads, the capability query
returns `Unsupported`; a host transpose is not a fallback.

For non-tile `I`, `O`, `D`, or `L`, matrix hardware sees neutral physical K/N
tails only. `R=15` uses a physical M=16 tile, but row 15 is not a token;
`R=17` uses logical rows 0..15 and a distinct M=1 tail. Output conversion,
masking, and merge predicates exclude all padded coordinates. Leading-only
views select checked whole physical planes; every rank-2..8 leading tuple is
walked independently with its transformed offset and strides and is never
broadcast. Ordinary output writes `Y[b,r,o]`; head-planar scatter writes
`Y[b,h,r,d]` from global `o=h*D+d`. Q owns `[b,Hq,R,D]`, K/V owners remain
`[b,Hkv,C,D]`, and the kernel reads only `t<L` from `g(h)` while also enforcing
`t<=a+r`. The public cache shape is never narrowed to L.

**Checked flow and scratch.** `P` below is the checked product of actual
linear leading-plane extents and `B` is the checked attention batch/leading
product. Every displayed product, `p16`, addition, alignment, and conversion to
bytes is checked in `size_t` before admission. “Logical” is payload accounting;
“physical” is the conservative deterministic caller-workspace requirement.
All scratch subranges start at 32-byte boundaries in a live `RawWorkspace`
from the queue's exact device. They are disjoint from operands/output and from
simultaneously live scratch, and the lease and all operand/output registrations
remain retained until the accepted OID is terminal. Reuse is allowed only
after the in-order predecessor that last touches a subrange completes.

| Segment | Checked elements/bytes | Device-local flow and purpose |
| --- | --- | --- |
| Linear selected X | logical read `P*R*I*2`; existing owner physical tiles | Existing BF16 tiled owner -> row-major A loads. Window starts at `s`; no relocation or scratch copy. |
| Persistent W | logical read `O*I*2`; existing shared owner | Existing BF16 `[O,I]` tiles -> column-major B loads. It is never scratch, transposed checkpoint state, or duplicated persistent storage. |
| Linear FP32 products | physical scratch `P*p16(R)*p16(O)*4` | `joint_matrix` FP32 C/D tiles -> caller scratch -> explicit device-local RNE pack. Ordinary pack writes caller `[P,R,O]`; head-planar pack/scatter writes caller `[P,H,R,D]`. Logical input/output accounting is respectively `P*R*I*2` and `P*R*O*2`. |
| Q and K | logical reads `B*Hq*R*D*2` and `B*Hkv*L*D*2`; existing owners retain physical R/D and C/D tiles | Q A tiles plus direct K B tiles selected by `g(h)` -> QK MAD. Only the initialized prefix is addressed; K remains capacity C. |
| QK scores | logical `B*Hq*R*L*4`; physical scratch `B*Hq*p16(R)*p16(L)*4` | FP32 QK accumulator/store -> tiled caller scratch -> device-local scale, causal mask, max/sum reductions, and stable FP32 softmax. Padded positions are never reduction members. |
| BF16 probabilities | logical `B*Hq*R*L*2`; physical scratch `B*Hq*p16(R)*p16(L)*2` | FP32 probabilities -> explicit device-local RNE BF16 preparation -> PV row-major A tiles. This buffer remains live through PV; elementwise Q*K is not an alternative to MAD. |
| V and FP32 PV | V logical read `B*Hkv*L*D*2`; FP32 physical scratch `B*Hq*p16(R)*p16(D)*4` | Existing V row-major B tiles selected by `g(h)` plus BF16 P -> PV MAD -> FP32 caller scratch. V is not expanded per query head. |
| BF16 PV/head staging | logical `B*Hq*R*D*2`; physical scratch `B*Hq*p16(R)*p16(D)*2` | FP32 PV -> explicit RNE BF16 head tiles. A device-local merge then writes the caller-owned `[B,R,Hq*D]` output, whose logical payload is `B*R*(Hq*D)*2`; the checked `Hq*D` precedes byte sizing. |

An implementation may prove a smaller tile-streamed requirement or reuse
non-overlapping lifetimes (for example QK score storage after probability
preparation), but its pure query and submission MUST use the same documented
schedule. It may not replace deterministic caller scratch with USM allocation,
a library handle's hidden workspace, host staging, or allocator-capacity
inspection. The conservative linear requirement is the aligned FP32-product
segment. The conservative SDPA requirement is the checked, aligned sum of the
score, probability, FP32-PV, and BF16-head segments above; the reusable session
capacity remains the maximum of mutually exclusive operation/transfer
requirements, not their sum.

**Seven-operation fit.**

| Planned operation | SYCL feasibility and concrete present blocker |
| --- | --- |
| Embedding gather | Implemented by `src/sycl/embedding.cpp` and `src/sycl/queue_embedding.cpp`: one native `parallel_for` work item owns each destination word, with queued status reset, gather, and four-byte host-USM status copy. Runtime conformance and Level Zero scheduling evidence remain execution obligations; no host scan, host task, or operand staging is permitted. |
| Linear | Implemented by `src/sycl/queue_linear.cpp` and `src/sycl/queue_internal.hpp`: the twenty non-BF16 leaves queue an operation-local in-order `parallel_for` with the pure `{0, 1}` requirement query and an `aspect::fp64` device gate on `F64`, and `BF16` uses the backend-private subgroup-16 `joint_matrix` BF16/BF16/FP32 route with one explicit RNE BF16 pack/scatter kernel per output tile row over the caller-owned alignment-32 `A32(P*pad16(R)*pad16(O)*4)` product scratch, covering ordinary and head-planar output over every transformed leading plane and both `M=16` and single-row tails. It returns `Unsupported` before submission when `ext_intel_matrix`, subgroup 16, or the BF16/FP32 matrix combination is absent. Executed device facts and per-`R` conclusions are recorded in [Linear projections](#linear-projections). |
| RMSNorm reduction | BF16 loads, FP32 squares/reduction, normalization, and explicit BF16 result packing fit subgroup/work-group kernels and caller scratch. It is nonmatrix work; a matrix MAD substitute is unnecessary, and no native SYCL reduction port exists. |
| RoPE trig | Device-local FP32 range reduction/trig and BF16 output packing fit an ordinary kernel over the tiled owner. It must preserve position `a+r`, pair boundaries, tails, aliases, and the queue error model. No native SYCL RoPE port exists. |
| Cache partial-tile copy | A device-local predicated tile copy can append only `[a,a+R)` while preserving untouched/padded cache bytes. Source/destination overlap, capacity C, initialized-prefix publication, and separate K/V OIDs remain as specified. Existing generic copy is not this cache operation. |
| SiLU unary | Device-local FP32 evaluation followed by BF16 packing fits an elementwise kernel with no matrix claim. The current SYCL queue has no SiLU kernel or neural facade implementation. |
| Causal grouped-query attention | The two MAD products are feasible as tabulated; masking, stable softmax, RNE P preparation, GQA head selection, caller scratch, and merged output remain device-local nonmatrix kernels. Missing either native QK or PV is `Unsupported`; host attention, host staging, and elementwise multiplication do not qualify. |

Every requirement query remains pure and has the same semantic arguments and
const output as its facade, with no workspace argument. It validates runtime
`B,R,I,O,Hq,Hkv,D,L,C`, nonzero extents, rank 2..8, modes/windows, shapes,
leading offsets/strides, `QuantizationFormat::NONE`, BF16 capability, aliases,
matrix combinations, subgroup 16, stride/alignment, and all checked arithmetic.
It allocates, registers, submits, or consumes no OID and does not inspect queue
or arena capacity. Submission additionally validates the actual workspace's
exact device, liveness, capacity, 32-byte alignment, lease, and nonoverlap.
Valid read/read overlap, including exact SDPA input aliases, remains allowed;
new output/read and scratch/operand/output overlap is rejected.

The backend-private kernels submit to the existing in-order queue and retain
the exact context/device, tensor owners, workspace lease, and native events
through terminal completion. They introduce no host wait inside an operation.
The caller still waits every direct producer before submitting a consumer;
negative OIDs report `noexcept` admission failure, and accepted asynchronous
failures stay visible on every repeated wait. Workspace reset/destruction must
therefore synchronize through the existing lease/quarantine path rather than
freeing or rebinding live USM.

**Dependency decision: no added matrix library.** The installed compiler,
Level Zero runtime, extension headers, positive device query, and native M=1
and M=16 samples establish that small direct kernels can cover linear, QK, and
PV. Direct kernels also expose tile padding, column-major W/K interpretation,
explicit RNE boundaries, exact queue, and caller workspace without a handle or
hidden allocator. By contrast, oneMath is a selector plus backend-wrapper and
third-party-library stack; its host BLAS usage models add discovery/link/runtime
surfaces and do not establish the IOM-specific no-hidden-allocation,
head-planar, GQA, or logical-row contracts. Convenience or broad GEMM tuning
does not justify that dependency, and a library cannot create matrix hardware
when the required combination is absent.

The landed SYCL linear-projection leaf is the first consumer of this facility
and shares the same minimal backend-private matrix route with the later SYCL
SDPA leaf. That consumer makes enabled-SYCL configuration fail clearly if the
pinned compiler/header contract is absent, and its runtime queries return
`Unsupported` before submission when the feature macro is not 1,
`ext_intel_matrix` or subgroup 16 is absent, no BF16/BF16/FP32 combination
covers the chosen tile, or stride/alignment cannot be satisfied. Disabled
SYCL/CPU-only builds gain no dependency. If future measured evidence proves
direct kernels cannot meet this contract, that consuming leaf—not this
assessment—must justify a minimal replacement with exact package, minimum
version/features, CMake target, private link and runtime footprint,
redistribution terms, deterministic caller-workspace/handle lifecycle, and
configuration/runtime failure behavior before adoption. Unsupported hardware
never becomes a dependency pass.

The installed inventory is reproducible remotely after the required oneAPI
initialization with:

```text
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u;
icpx --version && sycl-ls
sycl-ls --verbose
```

The future production port gates, intentionally not run by this assessment,
are the remote initialized-environment commands
`cmake --build build --target iom_sycl_conformance_tests` and
`ctest --test-dir build --output-on-failure -R
'^iom_sycl_conformance_tests$'`, plus native profiler evidence for logical
`R=1` and prefill linear, QK, and PV. They must cover all `1/15/16/17`,
non-tile dimensions, planes, GQA, padding exclusion, numerics, aliases,
workspace failures, device rejection, accepted failures, and repeated waits.

#### RMS normalization

This is the operation-owned contract for `DeviceOps::rmsnorm`. The common
facade, its admission rules, and its pure requirement query are declared and
frozen here. The CPU port has landed and queues all nine applicable floating
leaves at the exact `{0, 1}` zero-workspace requirement; CUDA and ROCm provide
their own source-inspected launch wrappers over the shared core, and TTNN
provides a preallocated BF16/F32 queue path. The remaining backends report
the operation `Unsupported` exactly as section 9 states above, and an
unsupported port never counts as numerical conformance. The exact ABI is:

```cpp
oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rmsnorm_workspace_requirements(
        const TensorView& x, const TensorView& scale,
        const TensorView& out, float eps);
```

The redundant `dim` argument is removed from declarations, definitions,
callers, tests, and documentation in one clean cutover; no overload, alias,
compatibility shim, or re-export remains. The feature extent is the
operation's `F`, and no operation-specific host span, implicit conversion,
integer norm, quantization, or storage-format staging is introduced.

1. **Layout.** `x` and `out` have identical logical shape `[...,R,F]` with
   rank two through eight and nonzero extents. `scale` is exactly rank-two
   `[1,F]`, shared explicitly across every independent leading plane and row;
   RMSNorm MUST NOT accept a rank-one scale or introduce a general
   hidden-state or leading-plane broadcast. Every plane and row is
   independent: the reduction covers exactly the `F` logical features of its
   own row, and tiled padding, other rows, other planes, and other requests
   MUST NOT contribute. Selected plane offsets and strides are honored, and
   final-axis transforms or shape inflation are rejected. All three views use
   the same applicable leaf type and `QuantizationFormat::NONE`, output
   storage is disjoint from both inputs, and read/read overlap between `x` and
   `scale` remains valid.
2. **Arithmetic.** For every leading-plane index `b`, row `r`, and feature
   `f`, RMSNorm is row-local and evaluates
   `out[b,r,f] = x[b,r,f] * rsqrt(sum(i=0..F-1, x[b,r,i] * x[b,r,i]) / F + eps)
   * scale[0,f]` in the direct accumulator domain of its leaf, never against
   an unbounded-real oracle. `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`,
   `F8_E5M2`, `F16`, `BF16`, and `F32` decode to FP32, and every square,
   product, and reduction step is FP32, with FP32 overflow/underflow and
   permitted reduction reassociation within the comparison policy below.
   `F64` stays FP64 throughout. The mean, epsilon addition, reciprocal square
   root, normalization, and scale multiplication use that accumulator, and
   the result is encoded exactly once to the output leaf using
   round-to-nearest, ties-to-even together with the existing named-format
   special-value and saturation rules.
3. **Special values and epsilon.** `eps` MUST be finite and nonnegative.
   `eps == 0` on an all-zero row produces quiet NaNs with no NaN payload
   promise. NaN in `x` makes the shared row reduction NaN and poisons that
   row. Infinite `x` makes the reduction infinite: finite features normalize
   to signed zero, and infinite features become quiet NaN before the scale
   multiply. A nonfinite `scale` affects only its own feature after the shared
   norm. Admission MUST NOT scan for nonfinite data, and no RMS-specific
   queued data failure is added.
4. **Capability matrix.** Applicability is exactly the nine ordinary signed
   floating leaves; the remaining fourteen leaves have no integer, boolean,
   or exponent-only normalization contract. The matrix below is the complete
   leaf-by-backend capability record.

   | Leaf | Contract | CPU | CUDA | ROCm | SYCL | TTNN |
   | --- | --- | --- | --- | --- | --- | --- |
   | `BOOL` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I2`, `U2` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I4`, `U4` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I8`, `U8` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I16`, `U16` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I32`, `U32` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `I64`, `U64` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `F4_E2M1` | applicable | supported | supported | supported | supported | `Unsupported` |
   | `F6_E2M3` | applicable | supported | supported | supported | supported | `Unsupported` |
   | `F6_E3M2` | applicable | supported | supported | supported | supported | `Unsupported` |
   | `F8_E4M3FN` | applicable | supported | supported | supported | supported | `Unsupported` |
   | `F8_E5M2` | applicable | supported | supported | supported | supported | `Unsupported` |
   | `F8_E8M0` | unsupported | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
   | `F16` | applicable | supported | supported | supported | supported | `Unsupported` |
   | `BF16` | applicable | supported | supported | supported | supported | supported |
   | `F32` | applicable | supported | supported | supported | supported | supported |
   | `F64` | applicable | supported | supported | supported | `aspect::fp64` only | `Unsupported` |

   CPU, CUDA, and ROCm support all nine applicable floating leaves. SYCL
   supports the eight non-`F64` leaves and supports `F64` only when the device
   reports `aspect::fp64`; otherwise `F64` is `Unsupported`. TTNN supports only
   `BF16` and `F32`: its seven encoded-carrier floats (`F4_E2M1`, `F6_E2M3`,
   `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, and `F64`) are `Unsupported`
   because native TILE compute cannot consume those carrier layouts without
   the forbidden host staging. On every backend, an unknown dtype
   enumeration value is `InvalidArgument`, while a recognized inapplicable or
   unsupported leaf and a recognized non-`NONE` quantization format are
   `Unsupported`.
5. **Admission order.** Before capability dispatch or any queue effect, a
   submission validates in exactly this order: (1) rank, nonzero extents,
   identical `[...,R,F]` shape, and `scale == [1,F]`; (2) exact queue `Device`
   identity for every view, stable live owner registration and native handle,
   selected plane and leading bounds, and stride and view metadata; (3)
   checked element, byte, address, stride, plane, feature, and tile
   arithmetic, rejecting overflow before narrowing or pointer calculation;
   (4) exact same leaf type and `QuantizationFormat::NONE`, output
   disjointness, and conservative output/input alias rejection; (5) finite
   nonnegative epsilon; (6) immutable backend capability; and (7) supplied
   workspace validation and lease. This is the generic facade order of
   *TinyLlama forward layout — Workspace and execution* above, split so that
   leaf applicability and conservative aliasing are host-checkable admission,
   epsilon precedes immutable capability, and supplied-workspace admission
   follows capability and therefore never masks an unported backend. Read/read
   `x`/`scale` aliases are allowed, but any output alias is rejected even when
   transformed windows appear disjoint, and no data is inspected to admit a
   request.
6. **Requirement query and workspace.** The throwing query is pure and
   deterministic: it allocates no host or native metadata, constructs no
   request or vector snapshot, registers or leases no owner, mutates no
   queue, token, or state, reads no data, and submits nothing, so it stays
   pure while the queue is occupied. Its result depends only on the validated
   views, `eps`, and immutable capability, and every supported implementation
   returns exactly `{0, 1}` and consumes no `RawWorkspace`. Because the
   requirement is `{0, 1}`, only the empty `RawWorkspaceView{}` is admissible;
   any supplied workspace with an owner is `InvalidArgument` before dispatch,
   with no hidden allocation, relocation, host arithmetic, or host roundtrip.
   Unsupported capability and malformed query inputs surface as the
   established throwing exceptions instead of OID mapping.
7. **Request snapshot and ownership.** Successful admission constructs one
   immutable `RmsnormRequest` that owns value-copied metadata for `x`,
   `scale`, and `out` (shape, leaf type, quantization, plane offset, plane
   strides, and validated bounds), their exact live owner and native-handle
   identities, the validated epsilon, and the admitted `{0, 1}` requirement.
   No borrowed `TensorView` or caller-owned metadata is retained, and
   snapshot values do not change when caller views or their backing metadata
   are mutated or destroyed. Read/read owner identities are deduplicated;
   output storage stays disjoint. Every required owner is registered and
   retained through proven in-order completion by the existing
   prepare/register/dispatch/rollback, OID sequencing, completion-release,
   and quarantine machinery — this contract adds no second registry or
   workspace framework. Temporary caller views may die immediately after the
   call returns.
8. **Errors and default hooks.** Host-checkable shape, metadata, device,
   alias, dtype, epsilon, workspace, and overflow failures are admission
   failures: the `noexcept` facade maps them to the established negative OIDs
   without consuming a token, registering an owner, submitting work, or
   mutating queue or output state, and successful accepted work returns a
   positive token. A runtime failure after acceptance leaves output unusable,
   and every wait for that token repeats the same failure. The default common
   RMSNorm hooks keep a well-formed request `Unsupported`, so a valid-shape
   request on an unported backend reaches explicit capability rejection
   before workspace inspection or execution and is never counted as
   successful conformance. A backend port replaces only its own capability
   predicate and implementation.
9. **Comparison policy.** The conformance policy is fixed before measurement:
   special-value class and signed-zero checks are exact while NaN payloads
   are ignored; finite `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`,
   `F8_E5M2`, `F16`, and `BF16` outputs must lie within two adjacent
   destination encodings of the reference; `F32` requires
   `abs_err <= 1e-6 + 2e-5 * abs(reference)`; and `F64` requires
   `abs_err <= 1e-15 + 1e-12 * abs(reference)`. The independent reference
   rounds every FP32 intermediate and uses `double` for `F64`; production
   code MUST NOT serve as its own oracle.

**Shared CUDA/ROCm queue and tiled-kernel core.** The two accelerator
backends share one queue branch and one tiled device operation.
`src/shared/gpu_queue.hpp` carries the immutable RMSNorm task and completion
record beside the copy and binary variants, and
`src/shared/gpu_queue_operations.inl` submits, dispatches, and completes it on
the same fixed partition as every accelerator copy: the exactly `C` eagerly
created completion resources, the fixed 512-byte metadata slots, the
preallocated worker, the common owner registration output, and the `{0, 1}`
zero-workspace requirement are reused with no queue growth, per-call
allocation, second stream, or submission-side wait, and all terminal paths
go through the existing completion release and quarantine rules. Immutable
request scalars, view snapshots, owner registrations, the fixed metadata
lease, and the event-ring submission are captured by value before the worker
runs; no borrowed view, workspace lease, or metadata slot outlives proven
completion. `src/shared/standard_tiled_rmsnorm.inl` is the complete
backend-parameterized CUDA/HIP device kernel and device codec: independent
leading planes and rows, a logical-`F`-only traversal that never reads or
writes tile padding, the frozen FP32 accumulator (FP64 for `F64`) with the
special-value and signed-zero rules above, and exactly one
round-to-nearest-even output encode with no host decode, hidden transfer, or
hidden allocation. Each backend contributes only its own `gpu_policy` launcher,
which must use the queue's already-created nonblocking stream. The CUDA
launcher in `src/cuda/copy.cu` calls the shared
`launch_standard_tiled_rmsnorm` kernel directly on that stream, while
`src/cuda/copy.hpp` advertises all nine applicable signed floating leaves.
This source-inspected capability evidence does not claim hardware execution;
the future CUDA conformance target owns that gate. ROCm supplies the
corresponding wrapper in its backend leaf, while any backend without its own
wrapper remains Unsupported.

**ROCm capability and implementation evidence.** `src/rocm/copy.hpp` exposes
the ROCm policy's immutable RMSNorm capability for the nine applicable
floating leaves. `src/rocm/copy.hip` invokes the shared
`launch_standard_tiled_rmsnorm` HIP kernel on the queue's existing
nonblocking stream, preserving the device-local logical-`F` reduction,
FP32/FP64 accumulator domains, one destination encode, and zero-workspace
queue protocol.
**TTNN capability and implementation evidence.** `src/ttnn/device_types.cpp`
classifies the complete RMSNorm matrix and `TtnnQueue` advertises only BF16
and F32. `src/ttnn/queue.cpp` snapshots and registers all three native
per-plane owners, submits one preallocated adapter launch for every mapped
leading plane under the TTNN API mutex and mesh queue, and retains or
quarantines those owners until native completion is proven. The adapter binds
the caller's output plane directly, so no output relocation, host roundtrip,
positive workspace, or hidden host arithmetic participates. Logical feature
width alone reaches the preallocated primitive; physical tile padding and
untouched planes remain outside the reduction and write mapping. The TTNN
conformance driver owns the BF16/F32 capability probe and future bounded
remote runtime gate.


The common owner is `src/device_ops_rmsnorm.cpp` behind
`include/iom/iom.hpp`. `test/test_iom.cpp` owns signature/cutover, query
purity, validation precedence, rejection-effect, snapshot-lifetime,
registration-lifetime, workspace, and repeat-wait coverage;
`test/backend/backend_conformance_other.hpp` and `test/cpu/test_cpu.cpp` own
the valid-shape `Unsupported` probes. Backend kernels and their
backend-specific launch code remain owned by their own ports.

### 10. Model loading and weight layout

Model ingestion begins with an explicit model directory and stays
backend-neutral. Reading a configuration never searches tokenizer, generation,
distribution, or fallback files, never creates a mapping, device, tensor, or
workspace, and never copies checkpoint payload bytes. Configuration validation
precedes every later loading stage, and the mapped-file -> `SafeTensors` ->
caller-created tensor flow with the host-transfer rules of
[section 4](#4-view-and-transfer-contract) is unchanged. Later loading stages
extend this section instead of defining competing contracts.

#### Supported TinyLlama configuration

`iom::load_tinyllama_config(const std::filesystem::path& model_directory)` in
`include/iom/model.hpp` reads exactly `<model_directory>/config.json` and
returns a fully populated `iom::TinyLlamaConfig`. The public header declares no
JSON type, there is no partial or empty success value, and only runtime values
are stored:

```cpp
struct TinyLlamaConfig {
    std::size_t num_hidden_layers;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t num_attention_heads;
    std::size_t num_key_value_heads;
    std::size_t vocab_size;
    std::size_t max_position_embeddings;
    std::size_t head_dim;
    std::size_t bos_token_id;
    std::size_t eos_token_id;
    float rms_norm_eps;
    double rope_theta;
};
```

The supported fields, their defaults, and their required values are normative:

| JSON field | Rule | Result |
| --- | --- | --- |
| `architectures` | Required array, exactly `["LlamaForCausalLM"]` | no stored field |
| `model_type` | Required string, exactly `"llama"` | no stored field |
| `num_hidden_layers` | Required positive integer | `N` |
| `hidden_size` | Required positive integer | `H` |
| `intermediate_size` | Required positive integer | `I` |
| `num_attention_heads` | Required positive integer | `Hq` |
| `num_key_value_heads` | Required positive integer | `Hkv` |
| `vocab_size` | Required positive integer | `V` |
| `max_position_embeddings` | Required positive integer | `C` |
| `hidden_act` | Required string, exactly `"silu"` | no stored field |
| `rms_norm_eps` | Required finite positive number that stays finite and positive as `float` | `float` epsilon |
| `rope_theta` | Required finite positive number representable as `double` | `double` theta |
| `attention_bias` | Optional; absent means `false`, and a present value must be the boolean `false` | no stored field |
| `mlp_bias` | Optional with the `attention_bias` rule | no stored field |
| `tie_word_embeddings` | Optional with the `attention_bias` rule | no stored field |
| `rope_scaling` | Optional; absent means no scaling, and a present value must be `null` | no stored field |
| `pretraining_tp` | Optional; absent means the integer `1`, and a present value must be the integer `1` | no stored field |
| `torch_dtype` | Required string, exactly `"bfloat16"` | selected source dtype |
| `bos_token_id` | Required integer, exactly `1`, and below `V` | `1` |
| `eos_token_id` | Required integer, exactly `2`, and below `V` | `2` |
| `initializer_range`, `transformers_version`, `use_cache`, `pad_token_id` | Optional metadata: no validation and no merged default, and no effect on any stored value | not stored |
| any other key, including a head-dimension override | Rejected as unknown | none |

Every integer field accepts JSON integers only. Floating-point values including
`1.0`, booleans, strings, arrays, and `null` are rejected, and each accepted
value must be positive and representable as `std::size_t`; token ids are never
list-valued. The configuration declares no independent head width, and the head
plan is derived rather than read:

- `H % Hq == 0`, and `Hq % Hkv == 0`;
- `head_dim` is `D = H / Hq`, which must be positive and even;
- `Hkv * D` is the grouped K/V width and is computed with checked arithmetic.

`N`, `H`, `I`, `Hq`, `Hkv`, `D`, `V`, and `C` are the runtime parameters of
[TinyLlama forward layout — Session sizing and lifetime](#tinyllama-forward-layout--session-sizing-and-lifetime),
which owns the checked `F = Hq * D` equality and every storage consequence of
those values. This section adds no checkpoint-specific dimension limit: a
one-layer checkpoint and the two-layer `N=2, H=8, I=12, Hq=4, Hkv=2, V=19,
C=17` boundary fixture are both supported.

#### Configuration failures

| Condition | Exception |
| --- | --- |
| Missing `config.json`, an unreadable file, or a failed read | `std::runtime_error` carrying the path and the system reason |
| Malformed, empty, or non-object JSON | `std::invalid_argument` carrying the path |
| Missing field, wrong type or integer kind, unsupported value, unknown key, nonrepresentable narrowing, invalid head ratio, odd `head_dim`, or invalid token policy | `std::invalid_argument` carrying the path, the field, the actual value or `<missing>`, and the violated constraint |
| Overflow of a checked derived width | `std::overflow_error` |
| Allocation failure while reading or decoding | `std::bad_alloc` |

A missing `config.json` is an I/O failure, never a missing-field error, and
decode exceptions from the JSON library are translated at this boundary only,
so the standalone SafeTensors parser keeps its own behavior. Configuration
reading allocates configuration and schema metadata alone: it creates no device
tensor, transfer workspace, queue, or checkpoint payload copy. The mapped
source, destination preflight, and synchronous realization stages that follow
extend this section with their own normative subsections.

#### Mapped weight source and logical inventory

`iom::load_tinyllama_safetensors(const std::filesystem::path& model_directory)`
in `include/iom/model.hpp` reuses the configuration validation above and then
validates the complete required SafeTensors inventory of that same directory.
It returns an owning `iom::ModelSource` only after every required role passed,
and it still creates no device, tensor, or workspace:

```cpp
enum class ModelWeightRole { token_embedding, final_norm, lm_head,
    input_norm, post_attention_norm, query, key, value, attention_output,
    mlp_gate, mlp_up, mlp_down };
struct ModelWeightId { ModelWeightRole role; std::optional<std::size_t> layer; };
struct ModelWeightInfo { ModelWeightId id; TensorShape logical_shape; };
class ModelSource final {
public:
    ~ModelSource();
    ModelSource(const ModelSource&) = delete;
    ModelSource& operator=(const ModelSource&) = delete;
    ModelSource(ModelSource&&) = delete;
    ModelSource& operator=(ModelSource&&) = delete;
    const TinyLlamaConfig& config() const noexcept;
    std::span<const ModelWeightInfo> weights() const noexcept;
    const TensorSpec& tensor_spec(std::size_t index) const;
    WorkspaceRequirements upload_workspace_requirements(
        const Device& device, std::span<Tensor* const> destinations) const;
    void upload_weights(Device& device, std::span<Tensor* const> destinations,
        RawWorkspaceView workspace = {}) const;
private:
    struct Impl;
};
std::unique_ptr<ModelSource> load_tinyllama_safetensors(
    const std::filesystem::path& model_directory);
```

This ABI is fixed. `ModelSource` is non-copyable and non-movable, its private
constructor is defined out of line, and its `Impl` privately retains exactly one
owning `SafeTensorsDir` for the whole source lifetime, so the owner is
constructed only after the complete inventory passed. That `Impl` additionally
retains one borrowed mapped payload span per published entry, pointing into the
store it owns, so the synchronous realization below reads the retained mapping
instead of a retained host copy. `weights()` borrows the
published inventory for that lifetime; `tensor_spec(index)` returns the selected
`TensorSpec` of one entry and throws the ordinary `std::out_of_range` outside
`[0, weights().size())`.

**Inventory order.** `ModelWeightId.layer` is absent for the three globals and
is in `[0, N)` for the other nine roles. The published order is exactly the
globals `token_embedding`, `final_norm`, `lm_head`, followed by increasing
layer and, within each layer, `input_norm`, `post_attention_norm`, `query`,
`key`, `value`, `attention_output`, `mlp_gate`, `mlp_up`, `mlp_down`.

**Required checkpoint schema.** `N`, `H`, `I`, `Hq`, `Hkv`, `D`, `V`, and `C`
are the validated runtime parameters of
[TinyLlama forward layout — Session sizing and lifetime](#tinyllama-forward-layout--session-sizing-and-lifetime),
whose per-layer weight inventory these selected sizes reproduce. Every required
name below is mandatory, uniquely named, BF16, and of exact source rank/shape:

| SafeTensors name | Source shape | Role | Logical shape |
| --- | --- | --- | --- |
| `model.embed_tokens.weight` | `[V, H]` | `token_embedding` | source shape |
| `model.norm.weight` | `[H]` | `final_norm` | `[1, H]` |
| `lm_head.weight` | `[V, H]` | `lm_head` | source shape |
| `model.layers.{l}.input_layernorm.weight` | `[H]` | `input_norm` | `[1, H]` |
| `model.layers.{l}.post_attention_layernorm.weight` | `[H]` | `post_attention_norm` | `[1, H]` |
| `model.layers.{l}.self_attn.q_proj.weight` | `[H, H]` | `query` | source shape |
| `model.layers.{l}.self_attn.k_proj.weight` | `[Hkv*D, H]` | `key` | source shape |
| `model.layers.{l}.self_attn.v_proj.weight` | `[Hkv*D, H]` | `value` | source shape |
| `model.layers.{l}.self_attn.o_proj.weight` | `[H, H]` | `attention_output` | source shape |
| `model.layers.{l}.mlp.gate_proj.weight` | `[I, H]` | `mlp_gate` | source shape |
| `model.layers.{l}.mlp.up_proj.weight` | `[I, H]` | `mlp_up` | source shape |
| `model.layers.{l}.mlp.down_proj.weight` | `[H, I]` | `mlp_down` | source shape |

Every required role carries exactly the checked logical byte length
`2 * product(source shape)`. HF `[out, in]` orientation and row-major element
order are preserved unchanged: no transpose, host repack, or pre-tiled form is
introduced. A rank-one normalization vector becomes logical metadata `[1, H]`
only, its mapped span stays exactly `2*H` bytes, no rank-one `TensorShape` is
exposed, and no final-axis view or reshape is used. `ModelWeightInfo.logical_shape`
stays independent of the selected encoding, the physical row-major mapped
region, and the retaining mapping owner, and `tensor_spec(index)` exposes
selected logical metadata only — never host bytes, container types, or a native
allocation size.

**Checked sizing.** The validated configuration yields the checked required
count `3 + 9*N` before any name is generated. Counts, element products, logical
byte counts, standard 16x16 tiled byte counts, and the aggregate weight totals
are all computed with checked arithmetic; `logical_nbytes()` and
`tiled_storage_nbytes()` of each selected `TensorSpec` are the numbers used. The
repeated-layer aggregate is derived from the nine representative role sizes and
`N`, so an impossible configuration fails without allocating a huge payload or
generating a huge name list. Standard tiled totals are never presented as
TTNN-native sizes; native allocation, stride, and address checks remain
`Device`-factory responsibilities.

**Ownership and validation order.** The order is fixed: validate the
configuration, compute the checked required count `3 + 9*N`, construct exactly
one private `SafeTensorsDir` so exact-extension discovery, every tensor header,
byte, and range check, and duplicate-shard rejection happen before required-name
filtering, reject an empty store, check the element products, logical byte
counts, standard tiled byte counts, and repeated-layer aggregate from the
validated dimensions, reject a store whose tensor count is below `3 + 9*N`
before enumerating any layer, then validate each required role and publish.
Extras are ignored only after complete parser validation, and a valid
extra may use another dtype. No index JSON and no duplicate-JSON-key parsing is
added. `ModelSource` privately owns the mapping, no payload byte is traversed or
copied beyond container metadata validation, and no persistent second
checkpoint image or host copy exists. Every failure is cleaned up by RAII and
leaves the source artifacts unchanged.

**Source failures.** These categories are normative:

| Condition | Exception |
| --- | --- |
| Missing required weight, or a wrong source rank, shape, dtype, or byte length | `std::invalid_argument` naming the directory, the logical checkpoint key, the required rank/shape/dtype, and the actual rank/shape/dtype or `<missing>` |
| Empty store, or fewer than `3 + 9*N` stored tensors | `std::invalid_argument` naming the directory and the required and actual tensor counts, before any required weight is named |
| Impossible required count or checked byte total | `std::overflow_error` |
| Malformed container header JSON | `std::runtime_error` naming the directory; only this boundary translates that container leak, and the standalone SafeTensors API keeps its own behavior |
| Malformed extra tensor, invalid container fields, or a duplicate name across shards | the established container categories of the SafeTensors parser, unchanged |
| Allocation failure | `std::bad_alloc` |

A missing required name is therefore a model-schema rejection, not the store's
missing-name `std::out_of_range`, while `tensor_spec(index)` keeps ordinary
bounds behavior. The mapped-file -> SafeTensors -> caller-created tensor flow,
the host-transfer rules of [section 4](#4-view-and-transfer-contract), and the
realization ownership rules remain authoritative; upload, preflight, and
device realization are separate later stages that consume this published
source.

**Extension boundary.** This adapter is the TinyLlama SafeTensors/BF16 source
only. It must not expose container types, make Hugging Face file keys a
universal storage identity, or define a logical weight as permanently having
one BF16 host representation, so a later GGUF adapter or a container holding
several differently quantized or differently stored representations of one
logical role stays expressible. Such a re-selection must never retarget an
existing immutable owner or view or invalidate an in-flight user. Today's
read-only mmap and row-major bytes establish no direct-DMA, unified-memory,
or zero-copy capability, so this stage adds no DMA handle, registration API,
zero-copy guarantee, variant list, codec, selector, eviction policy, or format
registry, and a future aliased destination must independently retain its
backing. Existing copy/staging remains the current path.

#### Destination preflight and reusable transfer workspace

Before it provisions scratch or transfers a weight, a caller preflights the
complete destination binding of a published source:

```cpp
WorkspaceRequirements ModelSource::upload_workspace_requirements(
    const Device& device, std::span<Tensor* const> destinations) const;
```

The caller first creates exactly one independent `Tensor` owner per `weights()`
entry from that entry's `tensor_spec(index)`, then supplies those owners as
pointers in exactly the published inventory order. The binding is one-to-one
and ordered: the list holds exactly `weights().size()` entries, and one owner
may not serve two roles, not even two roles whose selected logical metadata is
equal, such as the `[1, H]` normalization scales.

**Fixed ordering.** Create destination tensors on the chosen device, then query
this requirement, then provision the reusable scratch it reports, and only then
transfer. The query provisions, leases, submits, waits, allocates, and destroys
nothing, so a caller neither guesses a workspace size nor allocates device
scratch before the binding is known to be complete. The query deliberately takes
no workspace: positive workspace liveness, ownership, alignment, and overlap
rules belong to the synchronous upload, which enforces them with the shared
`detail::WorkspaceValidation::validated` rules of
[section 4](#4-view-and-transfer-contract).

**Complete validation before the first owner query.** Every destination is
validated against the published inventory before any per-owner requirement
query runs, so a rejected binding reports no partial requirement and performs
no query, allocation, or transfer. Validation covers, per destination in order:

- a non-null `Tensor*`;
- the owner's own full view, taken from `Tensor::view()`: the ABI binds owners
  rather than views, and a view whose owner identity is another object is
  rejected, so no transformed view, plane subrange, or retargeted owner can
  enter the binding;
- a `Tensor` created by the exact `device` argument instance; equal backend
  kind and ordinal on another instance are still a foreign binding;
- the destination device's own BF16 storage capability from
  `supported_data_types()`, checked once for the binding before any destination
  metadata is compared;
- the supplied destination's checked logical and standard 16x16 tiled sizing,
  and the checked aggregate logical and tiled byte totals of the whole binding;
- a full `TensorSpec` match with `tensor_spec(index)`: rank, dimensions, BF16
  leaf type, and `QuantizationFormat::NONE`.

The supplied destination is sized before its schema is compared, so an
unsizeable declared shape reports the checked arithmetic failure rather than
the schema mismatch. Live `Device`, allocator, and `Tensor` lifetimes and
distinct owners' nonoverlapping valid storage remain caller preconditions: this
query claims no dangling-reference or invalid-allocation detection beyond exact
owner and device identity.

**Reported requirement.** The result is the maximum `bytes` and the maximum
`alignment` over the binding's per-owner pure
`copy_from_host_workspace_requirements()` results, so one serial scratch range
satisfies every weight. It is never a sum of mutually exclusive scratch
requirements, a guessed native storage size, or an aggregate logical byte
count. A binding whose requirements are all zero returns exactly `{0, 1}`, so
the CPU and TTNN zero-workspace policy never requests a positive allocation.
The per-owner requirement hook is pure and deterministic, so the same complete
destination list reports the same maxima regardless of arena capacity, queue
occupancy, leases, or completion state.

**Destination failures.** These categories are normative:

| Condition | Exception |
| --- | --- |
| Missing or excess destinations, a null `Tensor*`, or one owner bound twice | `std::invalid_argument` naming the position, and both positions for a repeat |
| A destination of another `Device` instance, or not the owner's own full view | `std::invalid_argument` naming the position |
| A device whose `supported_data_types()` omits BF16 | `std::invalid_argument` |
| A wrong rank, dimensions, leaf type, or quantization for the published index | `std::invalid_argument` naming the position, the required rank/shape/dtype/quantization, and the actual one |
| Impossible checked destination or aggregate sizing | `std::overflow_error` |
| Backend requirement-query failure | that backend's established category |

`ModelSource` keeps exactly one private complete-binding validator, and the
preflight query and the synchronous upload share it: there is no optional
bypass, duplicate public binding container, or generic planning API. The query
stays backend-neutral — it contains no backend-kind switch and no native
runtime type — and it never claims readiness: successful validation of a
binding is not a completed upload and publishes no usable device weights. Only a
normal return of the synchronous realization below is a publication permission.

#### Synchronous realization and publication

A caller realizes the same complete ordered binding it preflighted by calling

```cpp
void ModelSource::upload_weights(
    Device& device, std::span<Tensor* const> destinations,
    RawWorkspaceView workspace = {}) const;
```

The normal `void` return is the only successful completion and the only
publication permission: there is no ready wrapper, returned readiness object,
per-weight status, or partial success, so a caller exposes its usable model only
after this call returned. The call runs the shared complete-binding validator of
the previous subsection first, so the count, order, owner identity, exact
`Device` instance, BF16 capability, full selected specification, and checked
sizing of the whole binding are settled before the first copy, and it then copies
each published entry's exact private mapped row-major BF16 span directly into
that entry's destination full owner view, in exactly the published inventory
order. HF `[out, in]` orientation and row-major element order are preserved
unchanged, a rank-one normalization source stays a `2*H` byte payload realized
into its `[1, H]` logical destination, and no transpose, intermediate CPU copy,
per-weight checkpoint buffer, `DeviceOps::copy`, queued copy, OID, or
asynchronous work exists on this path.

**Workspace.** The binding's aggregate requirement is the maximum serial
per-owner `copy_from_host` requirement that the same validator reports. When
that requirement is positive, the supplied `workspace` is validated against
every destination's full owner view with the shared
`detail::WorkspaceValidation::validated` rules of
[section 4](#4-view-and-transfer-contract), even when the caller supplied the
empty view, so a missing, insufficient, misaligned, foreign, dead, or
destination-overlapping scratch range fails before the first copied role. A
zero-byte requirement of `{0, 1}` consumes no workspace at all, which preserves
the CPU and TTNN unused-workspace behavior: those devices realize their weights
with the default empty view.

**Ownership.** The caller creates and owns every destination and any scratch it
provisions from the preflight result. This call allocates no tensor, no
workspace, and no persistent host payload, retains no second checkpoint image,
and neither provisions nor destroys caller resources. Destinations and scratch
must be quiescent and alive for the whole call, and the source must outlive it:
every supplied source mapping stays alive through all synchronous copies. After
a successful call the source may be destroyed once no borrowed source metadata
is used, and the copied device tensors remain independently valid. Any future
aliased or direct-DMA realization must retain its backing independently through
use; it is not implemented here, so today's path adds no DMA handle,
registration, or zero-copy claim.

**Realization failures.** These categories are normative:

| Condition | Exception |
| --- | --- |
| Any complete-binding violation listed by the previous subsection | `std::invalid_argument` or `std::overflow_error` as listed there, before the first copied role |
| Missing, insufficient, misaligned, foreign, dead, or destination-overlapping positive workspace | `std::invalid_argument`, before the first copied role |
| A failed synchronous weight upload | that backend's established category, unchanged |

A validation failure uploads nothing. A later synchronous upload failure
propagates its original category and stops immediately: earlier destinations may
already hold copied bytes, later destinations stay untouched, and every
destination, workspace, and view keeps its caller ownership and its identity.
The call promises neither rollback nor retry and destroys or retargets no caller
resource; only setup-owned RAII resources follow the existing backend safe
release/quarantine rules. No exception category is wrapped at this boundary:
the fixed model-boundary container-JSON translation of the mapped-source
subsection remains the only translation, and `std::bad_alloc`,
`std::overflow_error`, and backend failures pass through unchanged.

#### CPU loading conformance and observable inventory

`test/backend/backend_conformance_model_loading.hpp` owns the one shared,
backend-neutral loading scenario,
`iom_conformance::run_model_loading_conformance(const ConformanceDevices&)`. It
is the only model-loading conformance surface: it contains no backend-kind
switch, no accelerator header, no failure-injection seam, and no second
inventory generator, so every backend driver calls it unchanged with its own
devices, in the fixed CPU, CUDA, ROCm, SYCL, then TTNN integration order.
`test/cpu/test_cpu_conformance.cpp` registers it as
`iom_backend_conformance_cpu_tests` case `CPU model loading*`, which is the CPU
invocation:

```text
cmake --build build --target iom_backend_conformance_cpu_tests
./build/test/iom_backend_conformance_cpu_tests --test-case="CPU model loading*"
```

The CPU case loads four independent synthetic checkpoints through
`load_tinyllama_config` and `load_tinyllama_safetensors` only: the two-layer
`N=2, H=8, I=12, Hq=4, Hkv=2, D=2, V=19, C=17` boundary, its one-layer form,
and the H16 `N=1, H=16, I=20, Hq=4, Hkv=2, D=4` and H18
`N=1, H=18, I=22, Hq=3, Hkv=1, D=6` normalization/tile-boundary forms. The
observable inventory is exactly `3 + 9*N` selected roles — 12 for each one-layer
checkpoint and 21 for the two-layer one — with the three globals
`token_embedding` `[V, H]`, `final_norm` `[1, H]`, `lm_head` `[V, H]` and the
nine per-layer roles `input_norm` and `post_attention_norm` `[1, H]`, `query`
`[H, H]`, `key` and `value` `[Hkv*D, H]`, `attention_output` `[H, H]`,
`mlp_gate` and `mlp_up` `[I, H]`, and `mlp_down` `[H, I]`, each selected
`TensorSpec` being BF16/NONE with `2 * product(logical shape)` logical bytes.

For every checkpoint the case creates one destination per published index from
`tensor_spec(index)`, queries `upload_workspace_requirements`, provisions
reusable scratch only when the reported maximum is positive, uploads through
`upload_weights`, and reads every destination back with real `TensorView`
transfers whose separately queried download requirement is never assumed from
the upload one. The CPU binding reports `{0, 1}` and therefore realizes with the
default empty view, and no positive raw-workspace factory call is made on a
device that rejects one. Exact BF16 bits of every role are compared with the
independent fixture bytes; there is no numeric tolerance, no CPU-computation
oracle, and no persistent second whole-checkpoint host bank. The same source is
realized on the driver's CPU reference device and on the selected device, and
the realized destinations stay readable after the fixture artifacts are deleted
and the source is released.

The case also observes the failure policy of this section: valid extra tensors,
a second shard, and unrelated documents change no selected result; an absent
final required role, a wrong final role schema, a transposed final destination,
a foreign-`Device` destination, a null final destination, a count one short or
one long, and one owner bound to two equal-metadata roles all preserve the
seeded destination sentinels with no published source or upload.

#### CUDA loading conformance and observable inventory

The CUDA driver adds no loader port and no second scenario:
`test/cuda/test_cuda_conformance.cpp` registers the `CUDA model loading*` case
of the existing `iom_cuda_conformance_tests` target, whose conditional
registration is unchanged, and calls
`iom_conformance::run_model_loading_conformance(devices.conformance())`
unchanged with its own devices. The checkpoints, selected inventory, ordered
destination binding, workspace ordering, byte comparison, and failure policy are
exactly the CPU contract above; this subsection records only what is
CUDA-specific.

**CUDA setup.** The case opens with `REQUIRE(cuInit(0) == CUDA_SUCCESS)` and
constructs the driver's `CudaDevices`: a CPU device owning the driver's checking
host allocator as the comparison reference, the selected CUDA device on ordinal
0 as the candidate, and a distinct second CUDA `Device` instance as the foreign
destination owner, each with the driver's synthetic `DeviceMemoryConfig` arena.
Both model-loading devices are real CUDA runtime, storage, and allocator
machinery, and the same published source realizes on the CPU reference and on
the CUDA candidate bit-for-bit. BF16 storage capability is required rather than
a skip reason: the shared scenario creates every destination from the published
BF16 specifications and its binding preflight rejects a device whose
`supported_data_types()` omits BF16, so a CUDA configuration without BF16 fails
the case.

**CUDA workspace.** A CUDA destination reports the positive staging requirement
of its own `host_transfer_workspace_requirements`, so the complete binding
reports the maximum serial per-owner requirement, the caller provisions real
`Device::create_workspace` scratch from that reported maximum, and the empty
default view is refused before the first copied role with every destination byte
left intact. The `{0, 1}` zero-workspace policy stays the behavior of branches
whose own requirement is zero; CUDA neither subdivides nor extends it.

**CUDA limitations.** No per-backend loader, CUDA-specific expectation,
backend-kind switch, or CUDA runtime header enters the common scenario, and the
expected bytes stay the fixture's own independent role bytes.
`CudaStorageOracle` remains a native-layout diagnostic of the existing storage
and transfer cases and is never a model-loading expected-byte generator. No
host-transfer fault injection is introduced: the existing CUDA
queued-operation injection seams prove queue submission behavior, not a
synchronous loader failure, so this case asserts no loader runtime failure
category of its own and the established CUDA and container categories of this
section stay unchanged.

```text
cmake --build build --target iom_cuda_conformance_tests
./build/test/iom_cuda_conformance_tests --test-case="CUDA model loading*"
```

#### ROCm loading conformance and observable inventory

ROCm adds no loader and no second scenario. The driver registers the shared
CPU-first case of
[CPU loading conformance and observable inventory](#cpu-loading-conformance-and-observable-inventory)
unchanged on this backend's own devices, so every common rule — the published
inventory, destination creation from `tensor_spec`, the complete-binding
preflight, the upload and download workspace and ownership rules, the rejection
policy, and source lifetime — is the shared one and is not restated here.
`test/rocm/test_rocm_conformance.cpp` registers it as the
`iom_rocm_conformance_tests` case `ROCm model loading*`, and the driver's custom
doctest `main` is unchanged. The focused selections are:

```text
cmake --build build --target iom_rocm_conformance_tests
./build/test/iom_rocm_conformance_tests --test-case="ROCm model loading*"
ctest --test-dir build --output-on-failure -R '^iom_rocm_conformance_tests$'
```

Setup follows this driver's existing conformance convention. The CPU reference
runs over a caller `LinearAllocator`; the candidate and the foreign device are
two independent `make_rocm_device(0, …)` instances whose `DeviceMemoryConfig`
reserve is the caller-selected tensor-data arena. The foreign role is therefore
a distinct `Device` instance of the same ordinal, so exact `Device` identity
rejects a foreign destination — never an equal backend and ordinal.

ROCm host transfers need real positive scratch, so this instantiation exercises
the positive half of the workspace policy rather than the CPU `{0, 1}` half:
each destination view reports `{compute_staging_size(logical_nbytes), 32}`, the
complete binding reports that same maximum from
`upload_workspace_requirements`, the case provisions exactly one caller-owned
`create_workspace` range from that result, and that range is suballocated from
the reserved data arena with no new native backing. The empty default scratch is
refused as `std::invalid_argument` before the first copied role with every
seeded destination sentinel intact, and the readback requirement is queried
independently of the upload one.

Limitations. BF16 is a mandatory capability of the real device and never a skip
condition. This case uses no diagnostic seam: `HipStorageOracle` is diagnostic
only, and queued `inject_submission_fault_for_testing` faults cannot prove a
synchronous loader failure, so neither belongs to loading coverage. Real
accelerator build and execution are remote-only, and closure requires the
nonempty focused selection above on the actual enabled ROCm device together with
the full `iom_rocm_conformance_tests` suite.

#### SYCL loading conformance and observable inventory

`test/sycl/test_sycl_conformance.cpp` registers the same shared scenario as
`iom_sycl_conformance_tests` case `SYCL model loading*`, which is the fourth
invocation of the CPU-first case above, after the ROCm case and before TTNN:

```text
cmake --build build --target iom_sycl_conformance_tests
./build/test/iom_sycl_conformance_tests --test-case="SYCL model loading*"
```

The case is the existing `SyclDevices` fixture, unchanged and not duplicated:
one independent CPU reference device, the selected eligible SYCL device as
candidate, and a second distinct SYCL device instance at the same ordinal as
the foreign device, whose owned context differs from the candidate's, so the
foreign-destination rejection is judged against the exact candidate instance
rather than the ordinal. Every destination is created on the candidate from
`ModelSource::tensor_spec(index)`, the binding preflight is compared with the
real maximum serial per-owner `copy_from_host` requirement reported by those
destination views, positive scratch is provisioned from that exact candidate
device through `create_workspace` and used as the upload workspace, and the
CPU reference realizes the same binding through its zero-byte `{0, 1}` path.
Both workspace policies are therefore observed by actual queries rather than a
backend-kind switch, and readback is a real `TensorView` transfer with a
separately queried download requirement.

SYCL setup and limitations:

- SYCL execution is remote-only. Configure with the actual `SYCL_ENABLED` flag
  and the existing SDK arguments under a profile override whose `REMOTE_SETUP`
  is empty; before every build and test run execute
  `set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u`,
  preserve that environment, and run `sycl-ls` to confirm a Level Zero GPU.
- BF16 storage is mandatory and the fixture requires an eligible device: an
  unavailable Level Zero GPU, or a device that cannot hold the required BF16
  roles, is a failure, never a skip.
- The loader stays backend-neutral: SYCL types remain backend-private, no
  accelerator header enters common code, `model.cpp` keeps its internal
  staging, and `src/sycl/device_tensor.cpp` and `src/sycl/copy.cpp` change only
  if a loading failure demonstrates a transfer defect.

#### TTNN loading conformance and observable inventory

The shared contract of this
[section 10](#10-model-loading-and-weight-layout) is unchanged; this subsection
records only TTNN setup, native-storage consequences, and the exact invocation.
`test/ttnn/test_ttnn_conformance.cpp` registers the same backend-neutral
scenario as `iom_ttnn_conformance_tests` cases `TTNN model loading*`, after the
SYCL predecessor, through the existing `TtnnDevices` fixture and the target's
custom runtime registration. The two registered cases are `TTNN model loading:
complete checkpoint inventories realize exact BF16 weights` and `TTNN model
loading: failed weight uploads publish nothing and retain staging`, so the
wildcard selection covers both.

**Native per-plane storage.** TTNN destinations are created from
`tensor_spec(index)` on the selected TTNN device exactly as on every other
backend, but realization lands in TTNN-native per-plane storage: one TTNN-owned
plane per leading logical index, each padded to 32x32 tiles with its own native
padding. The logical readback is compared bit-for-bit with the fixture's
independent BF16 bytes, so native padding or a wrong slot map can never
compensate a wrong result, and no numeric tolerance, matrix-performance claim,
or fixed native byte formula is used. Native storage stays TTNN-owned and is
not an `iom::Allocator` arena; unsupported quantization and unsupported leaf
types are still rejected before native allocation by the existing
supported-type table.

**Unused workspace.** Every TTNN per-owner host-transfer requirement is
`{0, 1}`, so a complete binding of these checkpoints reports exactly `{0, 1}`
and realizes with the default empty `RawWorkspaceView`: the scenario never
provisions the real owning TTNN-native positive scratch an operation
requirement can create on this device, and a standard 16x16 tiled byte count is
never mistaken for a native32 requirement.

**One live context.** `TtnnDevices` owns exactly one TTNN context
(`make_ttnn_device(0)`); a second TTNN device of the same ordinal cannot coexist
in one process, so its `reference` and `foreign` slots are independently created
CPU devices with their own allocator. The foreign-destination rejection
therefore proves exact `Device` identity across two backend kinds rather than an
equal backend kind and ordinal, and the CPU reference device remains the
independent comparison device for every role.

**Host-transfer failures and retained staging.** Realization is synchronous, so
there is no queued token to wait for: every role's `copy_from_host` runs through
the device's retained host-transfer staging (one byte slot per native upload
dtype per plane plus one download buffer). The second case exercises the
existing staging-allocation and single-plane submission seams on a weight
upload. A failed retained allocation raises `std::bad_alloc` before any
destination byte is written and retains nothing. A submission fault armed for a
later upload throws that backend's `std::runtime_error` before the failed role's
native submission, so the complete ordered binding stops immediately: every
destination still holds the sentinel bytes of the earlier successful upload of
that binding, no later role of the aborted call is uploaded, and the aborted
call publishes nothing. The bounded recovery drain then completes the work the
failed transfer had already submitted, after which the acquired staging returns
to the facility; only the next normal `void` return is the publication
permission, and it reuses the same retained slot rather than the failed attempt
having freed or replaced storage that could still be read. Retired staging is
freed only with the device. This adds no loader hook, no backend-kind switch, no
second loader or harness, and no whole-checkpoint host copy or native format
rewrite; the shared scenario keeps containing no failure-injection seam.

**Invocation.** The TTNN gate is remote-only through `skill://csw-remote`
(profile alias `ttnn`), which syncs this workspace to a unique mirror and
preserves the existing `TTNN_ENABLED` configuration and SDK arguments. TTNN
device execution can hang, so every invocation is bounded, and
`/home/rlew/bin/ttnn_reset` resets the host device:

```text
cmake --build build --target iom_ttnn_conformance_tests
timeout 300s ./build/test/iom_ttnn_conformance_tests --test-case="TTNN model loading*"
ctest --test-dir build --output-on-failure -R '^iom_ttnn_conformance_tests$' --timeout 300
```

The first two commands are the focused model-loading gate and must report a
nonempty selection; the third is the complete TTNN conformance regression that
must keep passing. Session, tokenizer, and neural-operation coverage is not
owned here.

#### Opt-in real-checkpoint loading verification

The synthetic five-driver integration above stays required and unchanged. In
addition, one explicit, default-off check loads a caller-supplied official
checkpoint on the selected device of each existing backend conformance
executable. `test/CMakeLists.txt` owns the option and defines
`IOM_TEST_REAL_MODEL_LOADING=1` for those five executables only:

```cmake
option(IOM_TEST_REAL_MODEL_LOADING
    "Compile the opt-in real-checkpoint loading cases into the existing backend conformance tests"
    OFF
)
```

With the default `OFF`, an ordinary configuration compiles and runs the whole
suite with no model artifact, no hardcoded path, and no download: the entire
real-loading section of `test/backend/backend_conformance_model_loading.hpp` is
compiled out, so no environment read, checkpoint path, or real-loading code
exists in that build. With the option `ON`, each existing executable compiles
exactly one additional case, and no new executable, harness, or test target is
created:

| Backend | Existing executable | Added case |
| --- | --- | --- |
| CPU | `iom_backend_conformance_cpu_tests` | `CPU real model loading` |
| CUDA | `iom_cuda_conformance_tests` | `CUDA real model loading` |
| ROCm | `iom_rocm_conformance_tests` | `ROCm real model loading` |
| SYCL | `iom_sycl_conformance_tests` | `SYCL real model loading` |
| TTNN | `iom_ttnn_conformance_tests` | `TTNN real model loading` |

**Shared case.** The one case is
`iom_conformance::run_real_model_loading(iom::Device&)` and the standard-GPU
environments are read by `iom_conformance::real_model_memory_config()`, which
returns the exact `iom::DeviceMemoryConfig` the driver passes to its factory,
because the arena is a device-construction value. The production model API of
the previous subsections is unchanged, and this check adds no loader, selector,
digest, downloader, or device-selection API.

**Environment intake.** Nothing is defaulted, and no directory is searched:

| Variable | Required | Rule |
| --- | --- | --- |
| `IOM_TEST_MODEL_DIR` | every backend | Nonempty path of the caller-supplied official checkpoint directory; the case requires an existing directory and reads exactly that directory. |
| `IOM_TEST_MODEL_ID` | every backend | Nonempty pinned identity of that artifact (a Hugging Face revision or a digest-manifest identity); recorded verbatim in the evidence block and never interpreted by the loader. |
| `IOM_TEST_MODEL_ARENA_BYTES` | CUDA, ROCm, SYCL | Positive decimal byte count divisible by 32, passed through `DeviceMemoryConfig.tensor_arena_bytes`; it must cover the checked aggregate standard weight total plus the queried scratch maximum, and the synthetic conformance arena is deliberately not reused. |
| `IOM_TEST_MODEL_SHA256` | optional | The externally collected `sha256sum` manifest of that directory, recorded verbatim; when it is absent the evidence block records the exact collection command instead. |

**Fixed order.** The case reads the environment, loads and validates the
checkpoint, checks the pinned identity and complete inventory, computes the
checked aggregate standard tiled weight total, rejects an arena below that
total before any data tensor exists, creates one destination per published role
from `tensor_spec(index)` on the selected device, queries the real maximum
serial workspace requirement of that binding, rejects an arena below the weight
total plus that queried maximum, provisions caller scratch only when the
requirement is positive, and then synchronously uploads every role. Only the
normal `void` return of that upload is success: there is no readback,
computation, logits, generation, token, or matrix-performance claim, and no
partial model is loaded to fit memory.

**Pinned reference.** The case validates `N22, H2048, I5632, Hq32, Hkv4, D64,
V32000, C2048` and exactly `3 + 9*N = 201` required BF16 roles against an
expectation encoded independently of the loader: the globals `token_embedding`
`[V, H]`, `final_norm` `[1, H]` from its rank-one `[H]` source, and untied
`lm_head` `[V, H]`, plus the nine per-layer roles `input_norm` and
`post_attention_norm` `[1, H]`, `query` `[H, H]`, `key` and `value`
`[Hkv*D, H]`, `attention_output` `[H, H]`, `mlp_gate` and `mlp_up` `[I, H]`,
and `mlp_down` `[H, I]`, each BF16/NONE with `2 * product(logical shape)`
logical bytes. No subset is accepted, and the identity is settled before the
first destination exists.

**Failure policy.** There is no skip, fallback, default, download, empty
success, or synthetic substitute on this path. A missing or non-directory
`IOM_TEST_MODEL_DIR` fails the case before loading; an empty `IOM_TEST_MODEL_ID`
fails it before loading; a malformed or wrong checkpoint keeps the established
schema, container, overflow, and allocation categories of this section; a
missing, nonnumeric, zero, misaligned, or too-small `IOM_TEST_MODEL_ARENA_BYTES`
fails before the first data tensor exists, and any later native exhaustion
(`std::bad_alloc` from destination creation or scratch provisioning) propagates
unchanged; an unavailable enabled device fails its factory; and a synchronous
upload failure keeps that backend's established category.

**Ownership and storage.** Only the driver's selected device is used: CPU
creates its destinations through the driver's own allocator, TTNN through its
native per-plane storage with the `{0, 1}` requirement that consumes no
workspace, and the standard GPUs through the caller-selected arena. No
reference or foreign device receives a second copy of the full inventory, no
equally large reference payload is built, and the source, the destinations, and
any scratch stay alive through every call.

**Evidence record.** Each case prints one evidence block to the run's standard
output: the exact invocation (the process argv), the backend kind, the backend
ordinal, the `Device` instance identity, the model directory, the pinned
identity, the artifact manifest or its exact collection command, the arena bytes
or `native storage`, the required role count and dtype, the checked aggregate
standard weight total, the queried scratch requirement, and the `PASS`/`FAIL`
result, which reports failure unless the synchronous upload returned. The
manifest is collected outside the test and retained with that block:

```text
sha256sum "$IOM_TEST_MODEL_DIR/config.json" "$IOM_TEST_MODEL_DIR"/*.safetensors
```

**Invocation.** Configure with the existing SDK arguments and the option, then
build and run that host's existing conformance target; the focused filter must
report a nonempty selection. Reconfigure with `-DIOM_TEST_REAL_MODEL_LOADING=OFF`
for the ordinary suite, which needs none of these variables.

| Backend | Build target | Focused enabled runtime command |
| --- | --- | --- |
| CPU | `iom_backend_conformance_cpu_tests` | `./build/test/iom_backend_conformance_cpu_tests --test-case="*real model loading*"` |
| CUDA | `iom_cuda_conformance_tests` | `./build/test/iom_cuda_conformance_tests --test-case="*real model loading*"` |
| ROCm | `iom_rocm_conformance_tests` | `./build/test/iom_rocm_conformance_tests --test-case="*real model loading*"` |
| SYCL | `iom_sycl_conformance_tests` | `./build/test/iom_sycl_conformance_tests --test-case="*real model loading*"` |
| TTNN | `iom_ttnn_conformance_tests` | `timeout 300s ./build/test/iom_ttnn_conformance_tests --test-case="*real model loading*"` |

All accelerator commands execute only in their synchronized remote mirrors
through `skill://csw-remote`, with the SYCL toolchain initialization and TTNN
timeout rules of the sections above unchanged. Missing artifacts, missing
hardware, or an unrun required gate prevents closure; none of them is a skip
reason.

### 11. Backend integration and conformance obligations

The following source map is executable contract coverage. Shared scalar,
common validation, rank, queue, workspace, and memory-boundary scenarios live
in `test/backend/backend_conformance_common.hpp`,
`test/backend/backend_conformance_memory.hpp`, and
`test/backend/backend_conformance_add.hpp`; storage, transforms, tails, padding,
and copies live in `test/backend/backend_conformance_copy_storage.hpp` and the
independent `AcceleratorStorageOracle`; model loading and weight realization
live in `test/backend/backend_conformance_model_loading.hpp` with its
`test/model_loading_fixture.hpp` checkpoint fixture, registered for CPU as the
`iom_backend_conformance_cpu_tests` case `CPU model loading*`. Backend-local
targets are
`iom_cpu_conformance_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, `iom_sycl_conformance_tests`, and
`iom_ttnn_conformance_tests`, registered by `add_iom_backend_tests`.
`test/backend/test_backend_coexistence.cpp` and target
`iom_backend_coexistence_tests` provide the combined coexistence gate.

Each backend driver exercises ADD, MUL, SUB, and DIV through its real queue for
every required leaf, as well as unsupported domains, validation precedence,
broadcasting, transformed mappings, exact aliases, owner deduplication, repeat
waits, and retained failures. CPU may complete inline; accelerator queues and
TTNN staging/emulation preserve the same contract without SDK dtype narrowing.

CMake registration uses `add_iom_backend_tests` to create smoke and
conformance targets. Drivers provide allocator/context setup, CPU reference,
foreign-device identity checks, hardware gating, and native storage oracles;
enabled hardware runs and never skips.

### 12. Contract source map

Use these sources when changing or extending the contract:

- public device/tensor/queue API: `include/iom/device.hpp`,
  `include/iom/tensor.hpp`, `include/iom/iom.hpp`;
- standard capability and transfer interface:
  `src/shared/standard_tiled_copy.hpp`;
- common binary validation, operation dispatch, and independent scalar oracle:
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_add.hpp`;
- common RMS normalization layout, admission, capability, request snapshot,
  and pure requirement query: `src/device_ops_rmsnorm.cpp`, with its API,
  purity, precedence, rejection-effect, snapshot, and ownership tests in
  `test/test_iom.cpp` and its valid-shape `Unsupported` probes in
  `test/backend/backend_conformance_other.hpp` and `test/cpu/test_cpu.cpp`;
- storage, transfer, copy, and physical oracle:
  `test/backend/backend_conformance_copy_storage.hpp`,
  `test/backend/backend_conformance_oracle.hpp`;
- native setup/allocation seams: `test/cuda`, `test/rocm`, and `test/sycl`
  smoke/conformance drivers;
- model configuration intake and its isolated fixture: `include/iom/model.hpp`,
  `src/model.cpp`, `test/test_model_loading.cpp`, and
  `test/model_loading_fixture.hpp`;
- shared model loading and realization scenario:
  `test/backend/backend_conformance_model_loading.hpp`;
- backend-local full suites: `test/cpu/test_cpu_conformance.cpp`,
  `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`,
  `test/sycl/test_sycl_conformance.cpp`, and
  `test/ttnn/test_ttnn_conformance.cpp`;
- coexistence integration: `test/backend/test_backend_coexistence.cpp` and
  `test/CMakeLists.txt`.

If implementation and this document differ, update implementation, conformance,
`ARCHITECTURE.md`, and this contract together. Do not weaken shared tests.

## Part III — Operation contracts

Each compute operation owns exactly one normative section in this part. An
operation section fixes that operation's public ABI, logical layout, tensor and
index classification, backend capability matrix, workspace and control-status
protocol, admission, alias, and error rules, queued-data and deferred-failure
behavior, implementation recipe, and independent-reference obligations. It does
not restate Part II: where a shared contract already exists (device identity,
tensor ownership, views, queue and tokens, memory and workspace, error
categories, and conformance obligations), the operation section states only
what an implementer of that operation must additionally observe.

Operation sections are the single normative source for their operation. An
implementation, its declarations, its shared conformance cases, and the section
MUST stay synchronized; a defect found later is corrected in the section, the
affected implementation, and a behavioral regression together, followed by
revalidation of every completed backend port.

### Embedding lookup

Embedding lookup is a bit-preserving row gather. It selects whole table rows by
integral IDs and copies their raw payload bits into caller-owned output. It
performs no arithmetic, numeric conversion, re-encoding, accumulation,
rounding, or tolerance comparison, and it never reads or writes a logical
element outside the equations below.

#### Public ABI, layout, and view semantics

The complete public surface is exactly `DeviceOps::embedding` and
`DeviceOps::embedding_workspace_requirements`:

```cpp
oid embedding(const TensorView& table, const TensorView& indices,
              TensorView& out, RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements embedding_workspace_requirements(
        const TensorView& table, const TensorView& indices,
        const TensorView& out);
```

There is no host-index overload, public status handle, capability registry,
generic gather/pack/transpose API, or model/session API. The requirement query
receives the same semantic operands as submission with only the output made
const and the workspace omitted. The `noexcept` facade returns a positive
accepted OID, or the established negative `InvalidArgument`, `Unsupported`,
`Overflow`, `ResourceExhausted`, `DeviceError`, and `InternalError` OIDs
described in
[TinyLlama forward layout — Workspace and execution](#tinyllama-forward-layout--workspace-and-execution).
Zero is never accepted, and every admission failure has no output effect,
registration, sequence consumption, or accepted token.

The table is rank-two `E[V,F]`, and its caller-selected leading plane offset is
honored. `indices[...,1,R]` and `out[...,R,F]` have exactly the same leading
tuple, and each has rank 2 through 8. `R`, `V`, `F`, and every leading extent
are runtime values, never checkpoint constants, and MUST be nonzero. With `b`
denoting the complete leading tuple, the operation is

```text
out[b,r,f] = table[indices[b,0,r],f].
```

Views supply leading-only maps. Each operand and output contributes its own
selected plane offset and its own leading offset and strides, so index and
output planes may use different offsets, steps, and permutations, and the table
plane may be an independently transformed rank-two selected view. The operation
adds no leading broadcast, no singleton output-rank inflation, no mid-plane or
final-axis transform, no transpose, and no implicit cache or model-state
broadcast. Padding is not logical data: `16x16` tile padding, subbyte remainder
bits, and uninitialized output bytes are outside the logical element set, and
valid logical output MUST NOT depend on padding or uninitialized storage.

Payload elements are copied as raw codes. Subbyte payloads are addressed at
their logical bit width, so a standard-storage implementation reads and writes
packed fields while a native-carrier implementation such as TTNN uses one whole
carrier cell per logical element and masks only its logical width — and, for a
64-bit element, two consecutive carrier cells holding the low and the high
`uint32` word. A 64-bit payload occupies paired low/high `uint32` words, never
floating arithmetic and never a shift by 64. Only `QuantizationFormat::NONE` is
in scope: every other recognized quantization format is `Unsupported`, and this
operation adds no quantization or storage format.

#### Payload and index classification

Table payload covers exactly the 23 existing `DataType` leaves:

`BOOL`, `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`,
`I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`,
`F8_E8M0`, `F16`, `BF16`, `F32`, and `F64`.

Every one of them is copied bit-exactly, including `F8_E8M0`, NaN payloads,
signed zero, infinities, and integer values above `2^53`. No arithmetic,
conversion, re-encoding, accumulation, rounding, or tolerance applies. `BF16`
is mandatory on every backend that implements the operation, and `F64` storage
does not imply FP64 arithmetic. `BOOL` is an 8-bit storage payload, and only
BOOL codes valid under the existing transfer contract are valid payload values.

Indices cover exactly the 12 integral leaves:

`I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and
`U64`.

`BOOL` and every floating leaf are semantically invalid IDs: an implicit
boolean or real-to-token conversion is never performed. A signed negative ID
and every ID greater than or equal to the vocabulary extent `V` are errors.

An unknown `DataType` or `QuantizationFormat` enum value is `InvalidArgument`. A
recognized inapplicable index leaf, a recognized unsupported payload leaf, a
recognized non-`NONE` quantization format, and a recognized unsupported backend
capability are `Unsupported`. Semantic classification is independent of native
storage: a backend limitation never narrows the classes above, it only decides
which leaves that backend implements.

#### Backend capability matrix

| Backend | Payload leaves | Index leaves | Quantization |
| --- | --- | --- | --- |
| CPU | all 23 | all 12 | `NONE` only |
| CUDA | all 23 | all 12 | `NONE` only |
| ROCm | all 23 | all 12 | `NONE` only |
| SYCL | all 23 | all 12 | `NONE` only |
| TTNN (native) | 22: all except `F8_E8M0` | all 12 | `NONE` only |
The CPU implementation covers all 23 payload and 12 integral index leaves for
`QuantizationFormat::NONE`. It reports `{0,1}`, rejects positive
`RawWorkspace` creation, executes gathers on the existing asynchronous FIFO
worker with raw bit helpers, and defers queued negative or out-of-vocabulary
IDs as repeatable `std::invalid_argument` failures after acceptance.

The TTNN native row is final. Its 22 payload leaves and all 12 integral index
leaves use the same carrier layout: one native `UInt32` column per logical
element for every leaf of at most 32 bits, and two consecutive native columns —
low word at column `2*f`, high word at `2*f+1` — for the `I64`, `U64`, and
`F64` payloads and the `I64` and `U64` indices. A wide row index is valid only
when its high word is zero and its low word is below `V`, which is the same
condition as "nonnegative and `< V`" for a signed leaf whose sign bit lives in
the high word. `F8_E8M0` remains an unsupported payload leaf, and the backend's
carrier table does not store it.

The TTNN native row is measured on the configured TTNN device by
`iom_ttnn_conformance_tests` in `test/ttnn/test_ttnn_conformance.cpp`. With the
final declaration above, `ctest --test-dir build/ttnn --output-on-failure
--timeout 300 -R '^iom_ttnn_conformance_tests$'` passed 1/1 in 55.66 s on the
TTNN host (`bv1`, mirror `l11-wide-final-a1`), and a direct run of the same
binary reported `test cases: 48 | 48 passed`, `assertions: 1863947 | 1863947
passed`, and `Status: SUCCESS!` with a warm JIT cache. Those runs instantiate
the 22 declared payload leaves with `U32` indices, all 12 index leaves with
`BF16` payload, the targeted mixed 64-bit pairs, every directed 64-bit
raw-code fixture, the high-word-only out-of-vocabulary fixtures, and the
focused double-carrier case whose gathering, deferred failure, and proven
status/workspace reuse all cross the doubled column space.

No backend may advertise a capability it has not implemented. An `Unsupported`
result, a storage-only observation, or a rejection-only probe is never
successful embedding conformance, and an unported backend keeps explicit
rejection expectations until its own port lands.

#### Admission, aliasing, and error precedence

Common admission validates in this order, and only afterwards may owner
registration, sequence consumption, token acceptance, metadata effects, or
backend work occur:

1. rank, nonzero, and structure: well-formed views, rank-two table, index and
   output rank `2..8`, identical leading tuples, exactly one index row axis, an
   output last axis equal to `F`, and nonzero `V`, `F`, `R`, and leading
   extents;
2. exact queue `Device`, stable live owner and native handle for table,
   indices, and output, and leading view bounds and strides;
3. every checked element, bit, tile, plane, address, byte, and stride product;
4. exact output and leading shape, identical table and output dtype,
   conservative output/input overlap rejection, recognized dtype and
   quantization, and backend capability; and
5. the caller's supplied workspace against the reported `{bytes, alignment}`,
   including liveness, exact-device identity, size, alignment, overlap, and
   lease availability.

Embedding uses the same bounded, allocation-free checked-view helper path as
binary for recognized encodings, rank and nonzero extents, exact live owner and
stable native handle, leading-only view metadata, selected-plane bounds, and
checked plane, tile, element, bit, and byte arithmetic. That path allocates,
snapshots, registers, and leases nothing; embedding's shaping, capability,
alias, workspace, and status policy stays in this section.

Input-input read aliases are allowed: two inputs may overlap or alias exactly
because both accesses are reads. Output MUST be disjoint from the table and from
the indices. Same-owner output/input is rejected even when the transformed
windows appear disjoint, and standard backends additionally reject actual
intersecting backing ranges and identical native handles. No operand or output
is allocated, relocated, replaced, or silently converted by admission.

Admission snapshots the exact live registered owners, native handles, view
metadata, leading maps, plane selections, and workspace range for table,
indices, and output. Read/read aliases deduplicate owners, while output stays
disjoint and separately registered. No borrowed view is retained beyond
submission: a temporary caller view may die once its submission is accepted,
because registered owners keep the storage alive, and caller-owned output and
workspace remain leased through proven completion.

Requirement queries are deterministic and pure. A successful query allocates
nothing, including host metadata; mutates no registration, lease, status, or
queue state; consumes no token; reads no index value; submits nothing; and
depends only on the supplied views and immutable backend capability. It uses
borrowed views, checked scalars, and fixed stack state, and MUST NOT materialize
a vector-backed snapshot, `TensorShape`, `CopyViewSnapshot`, or request. A
supplied workspace is validated only after a successful requirement query,
during submission.

The rejection results are normative:

| Condition | Result |
| --- | --- |
| malformed view, rank or nonzero violation, leading-tuple mismatch, wrong index row axis or output shape, table/output dtype mismatch, unstable or stale owner, invalid selected plane | `InvalidArgument` |
| signed negative ID or ID `>= V` known to the host | `InvalidArgument` |
| unknown `DataType` or `QuantizationFormat` enum value | `InvalidArgument` |
| `BOOL` or floating index leaf, recognized unsupported payload leaf, recognized non-`NONE` quantization, recognized unsupported backend capability | `Unsupported` |
| output/input overlap, including conservative same-owner rejection and standard-backend backing-range intersection or identical native handles | `InvalidArgument` |
| supplied workspace that is not a live exact-device range, too small, misaligned, out of range, or overlapping an operand or output | `InvalidArgument` |
| overlapping live workspace lease, or bounded-resource exhaustion | `ResourceExhausted` |
| checked element, plane, tile, bit, byte, address, or stride overflow | `Overflow` |
| pre-acceptance runtime or device failure | `DeviceError` |
| any other unclassified failure | `InternalError` |
| negative or `>= V` ID discoverable only in queued storage | accepted, then the deferred failure below |

#### Workspace and control-status protocol

Successful requirements are exactly `{0,1}` for CPU and `{32,32}` for a
supported accelerator (CUDA, ROCm, SYCL, and TTNN). CPU allocates no operation
workspace and no device scratch, continues to reject creation of a positive
`RawWorkspace`, and keeps the established zero-requirement semantics: a
supplied but unused range is neither validated nor leased.

A positive accelerator range is caller-owned control scratch. It MUST be a live
exact-device owner range, at least 32-byte aligned, sufficient for the reported
bytes, disjoint from every operand and output, and leased through proven
completion. Owner-absolute subranges are checked and 32-byte aligned; disjoint
aligned ranges may be used concurrently, while overlapping live leases reject
with `ResourceExhausted`.

Those 32 bytes carry one fixed control packet: the first `uint32` is a bounds
status (`0` valid, `1` invalid ID) and the remaining 28 bytes are reserved
control padding. Each accepted call resets that status in queue order, and the
caller neither initializes nor polls it; only deferred completion after native
proof interprets it. Status transfer is explicit, bounded control metadata: it
is never table, index, or output staging, never a hidden payload allocation, and
never a host round trip for operand data. Standard accelerators enqueue, on the
same in-order native queue, the status reset, the raw gather, exactly a
four-byte device-to-host transfer of the status word, and the completion event.
CUDA takes the status cell from queue-owned page-locked memory
(`cudaHostAlloc`/`cudaFreeHost`); ROCm uses `hipHostMalloc`/`hipHostFree`; SYCL
uses `sycl::malloc_host`/`sycl::free` in the exact queue context with explicit
event dependencies. Standard SYCL host USM is not claimed to be physically
pinned or portably DMA-nonblocking. SYCL uses no `host_task` for the gather and
adds no submission-side wait. Runtime conformance and configured Level Zero
status-only traffic remain execution evidence, not an API guarantee. Native
launch, copy, or event failure always takes precedence over the status word.

The CUDA implementation is the standard raw-word path in
`src/shared/standard_tiled_embedding.hpp`,
`src/shared/standard_tiled_embedding.inl`, and `src/shared/gpu_queue_operations.inl`.
`src/cuda/copy.cu` enqueues the fixed metadata upload, device status reset,
one bounded 256-thread grid-stride gather, exactly one four-byte
`cudaMemcpyAsync` status transfer, and the completion event on the queue
stream. `src/cuda/device.cpp` reserves one page-locked `cudaHostAlloc` status
cell per fixed queue metadata slot and releases it through `cudaFreeHost` only
when that queue-resource lease is proven safe. The worker interprets the cell
only after event proof and caches a queued `std::invalid_argument`; native
launch, transfer, and event failures retain precedence. These are inspected
source/API facts, and the closing five-backend gate executed them on the
configured CUDA device: `ctest --test-dir build --output-on-failure --timeout
300 -R '^iom_cuda_conformance_tests$'` passed `1/1`, the direct binary reported
`31/31` cases with `5,956,331/5,956,331` assertions, and its embedding cases
alone reported `2/2` cases with `117,047/117,047` assertions on
`NVIDIA GeForce RTX 5090` (compute capability `12.0`, driver `595.71.05`,
`nvcc` release `13.2` build `V13.2.78`). The status reset, bounded gather, and
single four-byte status transfer are exercised by the deferred invalid-ID,
repeated-wait, and status-reuse cases those runs contain; the exact commands,
mirror, and observed results are recorded in the same-revision evidence below.

The ROCm implementation uses the same shared raw-word metadata and gather
kernel. `src/rocm/copy.hip` enqueues the metadata upload, device status reset,
bounded 256-thread HIP gather, exactly one four-byte `hipMemcpyAsync` status
transfer, and the completion event on the queue stream. `src/rocm/device.cpp`
reserves one page-locked `hipHostMalloc` status cell per fixed queue metadata
slot and releases it through `hipHostFree` only after the queue-resource lease
has proven completion. These are inspected source/API facts, and the closing
five-backend gate executed them on the configured ROCm device: `ctest
--test-dir build --output-on-failure --timeout 300 -R
'^iom_rocm_conformance_tests$'` passed `1/1`, the direct binary reported `32/32`
cases with `5,947,095/5,947,095` assertions, and its embedding cases alone
reported `2/2` cases with `117,046/117,046` assertions on `gfx1201`
(`AMD Radeon AI PRO R9700`, HIP `7.15.26333-0000000`, AMD clang `23.0.0git`),
including the deferred invalid-ID and repeated-wait cases that require the
four-byte `hipMemcpyAsync` status transfer.

TTNN transfers the entire 32-byte control packet. Its positive workspace is one
real owning replicated DRAM `MeshBuffer` on the existing unit mesh, whose native
page size is the checked request rounded up to a multiple of 32 while
`byte_size()` remains exactly the caller-requested logical bytes. Each accepted
call resets exactly `BufferRegion{owner_offset, 32}` once, gathers its planes in
order, and then reads exactly that 32-byte region non-blockingly before host
completion; only its first `uint32` is interpreted. A four-byte transfer for
that region, a read of an oversized workspace owner, and any admission-side wait
are contract violations.

Proven native completion releases the status cell, packet, and workspace lease
exactly once and permits safe reuse of independent scratch. Unknown completion,
failed drain, or unknown native finish MUST retain or quarantine the actual
owners, status cells, and workspace range instead of releasing or reusing them
merely because a semantic error was observed.

#### Queued index data, deferred failure, and recovery

Every `TensorView` index payload is queued data, including CPU-host-visible
storage: a preceding queued producer may still write it, so no index value has
host-known provenance. Admission MUST NOT scan indices, add a host index span
API, or hide a host round trip, and the operation MUST preserve FIFO order so a
produced index tensor is observed at execution time.

The worker or kernel decodes each index's full raw width and signedness: mask
the unsigned bits, inspect a signed sign bit before conversion, compare the
full-width `uint64` value against `V` before any narrowing or addressing, and
never wrap. A 64-bit index is reconstructed from paired low/high `uint32` words
in the same way as its payload. Structural, range, device, owner, view, dtype,
quantization, alias, overflow, and workspace errors that are host-known reject
at admission with the categories above.

A negative or `>= V` ID that is discoverable only in device or queued storage
may be accepted and fail later. Device paths set the control status without
forming an out-of-bounds table address; the CPU worker reports
`std::invalid_argument`. The accepted OID stays positive and caches that
category and context, and every repeated wait for it MUST rethrow the same
failure; message wording is not part of the contract. Invalid output is not
usable, and failure has no rollback.

After an accepted execution failure the entire output is unspecified and MUST
NOT be consumed: no rollback, no failed-output-unchanged guarantee, and no
transactional semantics are promised. Already accepted following work may still
execute, but a consumer of a failed producer MUST NOT be submitted after that
producer's wait fails, and its dependent output MUST NOT be consumed. The
caller's model session is unusable and MUST fail and drain every accepted OID
before reset or destruction, continuing to drain even when an individual wait
throws. The operation itself adds no session state, tensor-validity flag, or
global queue cancellation.

Native or worker completion proof is independent of a data error. Proven
completion permits safe release and independent reuse of storage and scratch
after drain; an unknown or unprovable completion quarantines owners, status
cells, and workspace rather than allowing reuse.

#### Implementation recipes

The standard recipe (CPU, CUDA, ROCm, and SYCL) gathers one work item per
destination 32-bit word, mapping each output plane, row, and feature through
`16x16` tiles and independent leading strides. A destination word decodes each
complete integral ID for every fragment it covers, checks sign and range before
any source addressing, selects the source raw bits, and merges only the bits
that word owns. A 6-bit payload may straddle two words but has exactly one
writer per destination word. A 64-bit payload is copied as paired low/high
`uint32` words, never with floating arithmetic and never with a shift by 64.
Indices use widths 2, 4, 8, 16, 32, or 64, never 6. Output padding is
unspecified and is never used as an ID or a value. The CPU worker runs on the
existing asynchronous FIFO worker and moves raw bits with the existing
`load_bits`/`store_bits`/`copy_value` helpers; it never uses a numeric codec. No
implementation stages table, index, or output data on the host.

The TTNN recipe is a raw device-native gather over the existing native carriers
(`uint8`, `uint16`, `BF16`, `float32`, and `uint32` containers) and fixed
DRAM-interleaved geometry. One reusable raw `MeshWorkload`/`Program` with one
data-movement core and a fixed raw L1/circular-buffer capacity sized for the
largest native tile (4096 bytes) serves every call; there is no compute kernel,
type conversion, numeric decode, high-level gather, or shape-keyed program
cache, and `TensorAccessorArgs::create_dram_interleaved()` fixes the topology
without caller tensors. Native data is a 32x32 `TILE` of four row-major 16x16
faces. For a logical carrier cell `(r,c)` with native carrier width
`carrier_bytes`, the native tile and intra-tile byte offset are

```text
tile = (r / 32) * native_tile_columns + c / 32
byte = (((r % 32) / 16 * 2 + (c % 32) / 16) * 256
        + (r % 16) * 16 + (c % 16)) * carrier_bytes
```

and the second, wide carrier of a 64-bit element is the following logical
column, which stays inside the same 16x16 face whenever the pair starts there.
Each dispatch supplies runtime words for `V`, `R`, `F`, the native table, index,
and output addresses, the carrier widths, the payload and index widths, the
signed index marker, the native carrier factor of the table and index planes
(one column per logical element, two for a 64-bit leaf), the padded column
count, and the status owner's base, page size, and subrange; native data pages
stay 1024, 2048, or 4096 bytes and the status owner uses its own full page size.
Each logical feature copies its raw carrier cells from the table row's physical
tile position to the output position, while physical padding stays non-logical.
Index values are checked inside the data-movement core before any NoC page
arithmetic: a 64-bit index reads both carrier words, treats a nonzero high word
as out of range, and never narrows a rejected value to an addressable row. The
core accumulates a monotone invalid-ID flag that it publishes with the required
NoC barrier, so ordered plane launches preserve earlier flags without a
cross-core atomic protocol. Leading planes are iterated on the host from the
immutable submission snapshots, without reading values or allocating a plane
list, and native submission reuses the existing unit-mesh device mutex.

The public `ttnn::embedding` operator is not suitable for this contract: it is
`ROW_MAJOR`/`BF16`-only, performs layout conversion, and leaves out-of-range IDs
unchecked.

#### Independent reference and conformance obligations

The independent reference MUST separately encode the table as row-major raw
bytes and select raw logical bits through the index equation above. It MUST NOT
call production address, codec, gather, or CPU embedding helpers as its sole
oracle; it reuses only test-only bit readers and canonical physical mapping, and
only where those stay independent. Fixtures are deterministic synthetic
patterns with explicit expected raw-code rows, including a deliberate
wrong-row or permuted-oracle sanity check, so a transpose, a numeric
re-encoding, and a wrong leading-plane selection are observable. Comparisons
are raw-bit exact with zero numeric tolerance.

Shared coverage MUST include every one of the 23 payload leaves with `U32`
indices, every one of the 12 index leaves with `BF16` payload, and targeted
mixed and 64-bit combinations without a wasteful full Cartesian matrix. It MUST
include valid `BOOL` `0` and `1` codes; directed floating special values where
representable; raw subbyte and wide-carrier codes; signed zero, NaN payloads,
and integer values above `2^53`; repeated IDs with the first and last vocabulary
rows; runs `R=1`, `15`, `16`, and `17`; non-tile feature sizes `F=1`, `15`,
`17`, `31`, and `33`; table `V` boundaries; a selected rank-two table view;
independent leading transforms, offsets, steps, and permutations through rank
8; and poisoned native and output padding whose bytes never change valid
output. Every declared 64-bit payload also carries a directed raw-code fixture
(both carrier words set, sign bit, NaN payload, `2^53` crossing, high-word-only
and low-word-only values) with an odd non-tile feature count, because a salted
pattern of that width reaches those classes only by chance. Failure coverage
MUST include malformed structure, wrong device, stale registration, alias and
overlap, dtype and non-`NONE` quantization, checked overflow, workspace
liveness, size, alignment, device, overlap, and lease, a producer copy to
embedding to consumer FIFO case, negative, `>= V`, `U64_MAX`, and signed
sign-bit IDs without narrowing, repeated waits, and status and workspace reuse
after proven drain. A 64-bit index additionally carries a high-word-only
outlier whose low word alone is a valid small ID, so a truncated comparison
addresses a valid row instead of failing. Small index widths keep `V` inside
that representation's nonnegative domain, and a positive out-of-vocabulary
fixture chooses a representable ID code.

The cases live in the shared header
`test/backend/backend_conformance_embedding.hpp` and run through the existing
`iom_backend_conformance_cpu_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, `iom_sycl_conformance_tests`, and
`iom_ttnn_conformance_tests` drivers. No second test project, generic test
framework, model fixture, checkpoint, or network dependency is permitted. Each
driver supplies its own explicit expected payload and index spans, and common
code never switches on backend kind. An unported backend keeps an empty
embedding span and asserts `Unsupported` only; that rejection probe is not
gather conformance. The existing copy/add/mul/sub/div suites and the `silu`
and `sdpa` `Unsupported` probes remain unchanged until their own operation
leaves migrate them; the `linear` probes were migrated by the five linear
leaves, and RMSNorm now has backend-owned CUDA and ROCm
launch wrappers, while its shared conformance helper remains task-owned.

Each driver declares through `iom_conformance::EmbeddingDeclaration` the
payload and index matrix its port must reach, the leaves this revision
implements (both spans empty for an unported port), and its exact workspace
requirement (`{0, 1}` on CPU and `{32, 32}` on CUDA, ROCm, SYCL, and TTNN). The
matrix drives the shared fixtures, so a case exists exactly for a leaf the port
must reach; `iom_conformance::kEmbeddingPayloadSpan`,
`iom_conformance::kEmbeddingIdSpan`, and
`iom_conformance::kEmbeddingTtnnPayloadSpan` are the standard 23/12 matrix and
the final TTNN 22/12 native row, and `iom_conformance::kNoEmbeddingSpan` is the
empty implemented span of an unported port. Entry points are
`iom_conformance::run_embedding_conformance` (self-check, declared request
cases, and common admission/ownership/queue/failure cases),
`iom_conformance::run_embedding_oracle_self_check`,
`iom_conformance::run_embedding_reference_conformance`, and
`iom_conformance::run_embedding_common_conformance`, and the driver cases are
the `embedding lookup reference, admission, and lifetime` cases of
`test/cpu/test_cpu_conformance.cpp`, `test/cuda/test_cuda_conformance.cpp`,
`test/rocm/test_rocm_conformance.cpp`, `test/sycl/test_sycl_conformance.cpp`,
and `test/ttnn/test_ttnn_conformance.cpp`.

Every rule in this section MUST be observable through the shared conformance
suite or through a focused native lifetime and control-transfer scenario; tests
assert behavior rather than field forwarding, source text, or incidental message
wording.

#### Same-revision five-backend gate evidence

The closing Embedding gate ran every backend's existing conformance target and
its direct binary at one revision — `5f839f10`, prepared worktree
`.work/006-tinyllama/03-embedding-lookup/12-embedding-five-backend-gate` — and
recorded the device, runtime, and toolchain identity of each run. The CPU pair
ran locally; every accelerator pair used exact-worktree `csw-remote` sync/exec
with a fresh sync immediately before each execution, a unique per-profile
mirror, remote-side `timeout --kill-after=30s`, and a bounded hardware lock,
and the TTNN invocation kept its 300-second CTest bound. No `.cswd` metadata
was copied to any host. Every row below is a real run: nothing in it is
compile-only, storage-only, rejection-only, or skipped.

| Backend | Commands (mirror) | Device / runtime identity | Observed result |
| --- | --- | --- | --- |
| CPU | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_backend_conformance_cpu_tests$'` and `./build/test/iom_backend_conformance_cpu_tests` | Local host CPU, `g++` `15.2.0`, CMake `4.0.2`, release build | `1/1` CTest pass in `10.93 s`; `24/24` cases, `5,936,189/5,936,189` assertions; the embedding case alone `1/1` case, `118,307/118,307` assertions |
| CUDA | both commands on remote host `bv1` (mirror `csw03emb12-a1-cuda`) | `NVIDIA GeForce RTX 5090`, compute capability `12.0`, driver `595.71.05`, `nvcc` release `13.2` build `V13.2.78` | `1/1` CTest pass in `16.33 s`; `31/31` cases, `5,956,331/5,956,331` assertions; embedding cases `2/2` cases, `117,047/117,047` assertions |
| ROCm | both commands on remote host `bv2` (mirror `csw03emb12-a1-rocm`) | `gfx1201` (`AMD Radeon AI PRO R9700`), HIP `7.15.26333-0000000`, AMD clang `23.0.0git` | `1/1` CTest pass in `20.78 s`; `32/32` cases, `5,947,095/5,947,095` assertions; embedding cases `2/2` cases, `117,046/117,046` assertions |
| SYCL | both commands on remote host `bv2` through the outside-checkout profile override with an empty `REMOTE_SETUP`, `set +u; source /opt/intel/oneapi/setvars.sh; set -u` and a `sycl-ls` GPU enumeration in every remote call (mirror `csw03emb12-a1-sycl`) | Level Zero V2 `Intel(R) Arc(TM) Pro B60 Graphics`, oneAPI DPC++/C++ `2026.1.0`, `ocloc` `26.22.38646.7` | `1/1` CTest pass in `6.83 s`; `29/29` cases, `5,907,297/5,907,297` assertions; embedding cases `4/4` cases, `117,124/117,124` assertions |
| TTNN | `ctest --test-dir build/ttnn --output-on-failure --timeout 300 -R '^iom_ttnn_conformance_tests$'` and the direct binary on remote host `bv1` (mirror `csw03emb12-a1-ttnn`) | Blackhole device 0, UMD firmware bundle `19.13.1`, TT-Metalium `v0.76.0-dev20260801-268-g06994d4afda` (`06994d4afda`) | `1/1` CTest pass in `33.63 s`; `48/48` cases, `1,863,947/1,863,947` assertions; embedding cases `3/3` cases, `116,723/116,723` assertions |

Each sync and exec pair, with its exact argv and observed exit status, is
retained in
`.cswd/tasks/006-tinyllama/03-embedding-lookup/12-embedding-five-backend-gate/remote.log`;
the remote-side logs named there hold the full run output.

**Matrix closure.** The four standard drivers declare and reach the complete
23-payload and 12-index matrix through `kEmbeddingPayloadSpan` and
`kEmbeddingIdSpan`, and the TTNN driver declares and reaches its 22-payload,
12-index native row through `kEmbeddingTtnnPayloadSpan`. This gate adds no case,
no span, and no second test project: the runs above are the existing shared
cases plus each driver's own native cases. `BF16` and `R=1`, `15`, `16`, and
`17` execute on every backend inside those declared cases, because the derived
fixtures pair every declared payload leaf with `U32` indices, pair the declared
`BF16` payload with every index leaf, and vary `R`, the feature extent, the
vocabulary, and the leading planes with `BF16` as the default payload. TTNN's
only payload outside its row is `F8_E8M0`: the canonical TTNN storage span omits
it, the storage case asserts that `create_tensor` throws for it while all
twenty-two other leaves succeed, and the embedding declaration therefore holds
no `F8_E8M0` case. No other leaf, shape, or backend was excluded, and no
`Unsupported`-only, storage-only, or compile-only observation is counted
anywhere in the table.

**Lifetime, workspace, and queue obligations observed by the same runs.** The
shared `run_embedding_conformance` cases behind every row above cover the
independent raw-bit oracle and its deliberate wrong-row, transposed,
padded-table, numeric-re-encoding, and plane-blind negative variants; the
declared request matrix with repeated IDs, first and last vocabulary rows,
non-tile features, vocabulary boundaries, independent leading offsets, steps,
and permutations through rank eight, a selected rank-two table plane, and
poisoned native and output padding; accepted out-of-vocabulary and negative IDs
including the `U64_MAX`, `I64_MIN`, and high-word-only classes, together with
the directed 64-bit payload codes that cross `2^53` and carry both carrier
words; repeated waits caching the same `std::invalid_argument`; the pure
requirement query with no record, submission, registration, lease, or sequence
consumption; positive-workspace validation, disjoint-range coexistence,
overlapping live lease, stale and foreign range, and proven-completion reuse;
output/input alias and overlap rejection with input/input read aliasing allowed;
the producer copy to embedding to output consumer FIFO order; and
unknown-completion quarantine with release only after native proof. The declared
workspace rows are `{0, 1}` on CPU and `{32, 32}` on CUDA, ROCm, SYCL, and TTNN,
whose first `uint32` is the bounds status (`0` valid, `1` invalid ID) and whose
remaining 28 bytes are reserved control padding the caller neither initializes
nor polls.

**Status-transfer and execution-style evidence.** The standard accelerators
transfer exactly four status bytes: the shared
`src/shared/gpu_queue_operations.inl` path that CUDA and ROCm instantiate resets
the device status word, launches the bounded gather, copies back
`sizeof(std::uint32_t)` through `cudaMemcpyAsync`/`hipMemcpyAsync`, and records
the completion event in FIFO order on the in-order queue. On SYCL the run
executes the native gather declared in `src/sycl/embedding.cpp`: the status
reset and the four-byte status copy are queue submissions that explicitly
depend on the preceding event in `src/sycl/queue_embedding.cpp`, the built
conformance binary carries the `launch_embedding_words` kernel symbol
(31 occurrences) and no `host_task` occurrence, and the only `host_task` in the
whole source tree is the unrelated SYCL binary-operation host realization in
`src/sycl/queue_binary.cpp`. The deferred device-side invalid-ID failures the
run asserts are reachable only when the native kernel writes the status word
and the event-ordered copy delivers it, because no host scan of queued indices
exists on this path. TTNN transfers the complete 32-byte control packet, and
its run observed the raw Metal route rather than a high-level operator: with the
Metalium build logger enabled (`TT_METAL_LOGGER_LEVEL=debug
TT_METAL_LOGGER_TYPES=BuildKernels`) the focused TTNN embedding cases report
`JIT build cache hit:
.../kernels/embedding/16918367578600865223/ncrisc/ncrisck.o`, so the
data-movement (`ncrisc`) kernel built from `src/ttnn/kernels/embedding.cpp` is
what the device executes; `EmbeddingProgram` creates the program, circular
buffer, and reader kernel through `CreateProgram`/`CreateKernel` and dispatches
them with `EnqueueMeshWorkload`, and the status owner is read back as exactly
`BufferRegion{owner_offset, 32}` through `enqueue_read_shards`. The public
`ttnn::embedding` operator appears nowhere in the tree, so no high-level,
host-emulated, or BF16-only substitution took place.

**No defect.** Every recorded run passed with zero failures, so this gate
corrected no normative statement, weakened no comparison, and changed no case,
span, declaration, or implementation file: the section, the shared header, and
the five drivers already match the exercised behavior, and the previously
completed ports were revalidated by these same runs rather than by an unrun
obligation.

#### Implementation references and delivery prerequisites

Implementers need these existing sources and seams:

- public ABI, request snapshots, owner registration, and workspace views:
  `include/iom/iom.hpp`, `include/iom/tensor.hpp`, and `include/iom/device.hpp`;
- checked logical shapes, slots, and leading-plane arithmetic: `src/tensor.cpp`
  and `src/iom_internal.hpp`;
- shared allocation-free checked-view helpers and operation dispatch:
  `src/device_ops.cpp` and `src/device_ops_binary.cpp`;
- admission, FIFO acceptance, rollback, deferred completion, and repeated
  waits: `src/device_ops.cpp`;
- workspace owner bytes, 32-byte subranges, leases, proof, and quarantine:
  `src/workspace.cpp` and `include/iom/detail/workspace_registry.hpp`;
- standard tiled word ownership, launch mapping, and queue resources:
  `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_add.inl`,
  `src/shared/gpu_queue.hpp`, `src/shared/gpu_queue_operations.inl`, and
  `src/shared/event_ring.hpp`;
- CPU asynchronous worker and raw bit movement: `src/cpu/queue.cpp` and
  `src/cpu/transfer_helpers.hpp`;
- TTNN native carriers, storage, queue, and workspace:
  `src/ttnn/device_types.cpp`, `src/ttnn/device.cpp`,
  `src/ttnn/device_internal.hpp`, `src/ttnn/registry_state.hpp`,
  `src/ttnn/queue_internal.hpp`, and `src/ttnn/copy.cpp`;
- TT-Metal raw path: `mesh_buffer.hpp`, `buffer.hpp`,
  `tensor_accessor_args.hpp`, `tensor_accessor.h`, and `dataflow_api.h` under
  the installed `tt-metalium` API and hardware headers;
- accelerator status APIs: the CUDA Runtime memory interface (`cudaHostAlloc`,
  `cudaFreeHost`, and `cudaMemcpyAsync`), `hip/hip_runtime_api.h` page-locked
  allocation and asynchronous copy, and the SYCL USM, queue, and `atomic_ref`
  interface references;
- independent storage oracle and conformance drivers:
  `test/backend/backend_conformance_oracle.hpp`,
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_memory.hpp`, the five
  `test/<backend>/test_<backend>_conformance.cpp` drivers, and
  `test/CMakeLists.txt`.

The shared raw-word embedding sources, the TTNN carrier/queue/workspace
sources, and the five conformance drivers named above are the delivered
implementation and test surface of this section.

The genuine prerequisites and the final closure are producer/consumer
relationships, not a fixed serial backend order:

1. the shared allocation-free checked-view helpers plus the public embedding
   and requirement-query declarations with `Unsupported` defaults;
2. the independent raw-bit reference and the shared embedding conformance
   header, before any backend claims numerical conformance;
3. the per-backend port, which for TTNN required the caller-owned
   positive workspace and its 32-byte status owner before the native
   one-carrier port, and then the double-carrier completion; and
4. the shared five-backend gate, which closes Embedding lookup before linear
   implementation begins and revalidates already completed ports if a later
   backend exposes a contract defect.

CPU, CUDA, ROCm, SYCL, and TTNN appear in that order only as the task-list and
scheduling order. This contract imposes no serial CPU-to-TTNN execution
requirement: accelerator ports may proceed independently once the shared ABI,
admission, and reference prerequisites exist, and an unported backend names its
missing evidence instead of inventing capability. This section records the
frozen target and the obligations of implementers; it claims no build, test,
accelerator, hardware, or runtime validation on any backend.

### Linear projections

This is the operation-owned contract for `DeviceOps::linear`. The public
facade, its admission rules, and its pure requirement query are frozen here
before implementation begins; the default common hooks keep a well-formed
request `Unsupported` exactly as [section 9](#9-other-compute-capabilities)
states, and an unsupported port never counts as numerical conformance. This
section is the single normative source for the linear operation's ABI, layout
equations, validation precedence, ownership and lifetime, dtype and
quantization classification, integer and scalar arithmetic, nonfinite
behavior, workspace formulas, backend capability, fixture thresholds, and
native-matrix evidence.

#### Public ABI, layout, and view semantics

The complete public surface is exactly `DeviceOps::linear` and
`DeviceOps::linear_workspace_requirements`. The frozen declarations, in this
exact token order and with these exact defaults, are:

```cpp
enum class LinearOutputLayout { ordinary, head_planar };

oid linear(const TensorView& x, const TensorView& w, TensorView& out,
           std::size_t s, std::size_t R, LinearOutputLayout layout,
           std::size_t H, std::size_t D,
           RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements linear_workspace_requirements(
        const TensorView& x, const TensorView& w, const TensorView& out,
        std::size_t s, std::size_t R, LinearOutputLayout layout,
        std::size_t H, std::size_t D);
```

There is no second argument order, no overload of `linear`, no public status
or capability-registry API, and no transpose, head-pack, or row-extraction
API; the three-view `linear(x, w, y)` facade is not part of this contract. The
requirement query receives the same semantic arguments in the same order as
submission, with only the output made const and the workspace omitted. The
`noexcept` facade returns a positive accepted OID, or the established negative
`InvalidArgument`, `Unsupported`, `Overflow`, `ResourceExhausted`,
`DeviceError`, and `InternalError` OIDs described in
[TinyLlama forward layout — Workspace and execution](#tinyllama-forward-layout--workspace-and-execution).

`x` is `[...,T,I]`, the weight is exactly rank-two `w[O,I]` and is shared
unchanged across every independent leading plane, and `out` is the caller's
result leaf. `w` uses the Hugging Face `[out,in]` orientation: output
coordinate `o` selects weight row `w[o,*]`, and the operation never transposes
a checkpoint weight. `enum class LinearOutputLayout { ordinary, head_planar }`
is the fixed output-mode type, and output rank never selects the mode.
`ordinary` output is `out[...,R,O]` and requires exactly `H=1` and `D=O`.
`head_planar` output is `out[...,H,R,D]` and requires the checked equality
`O=H*D`, inserting exactly one head axis. Output rank is at most eight, so a
rank-eight ordinary output is structurally valid while an inserted head axis
that would reach rank nine is rejected as `InvalidArgument`. `x` and `out`
carry the exact same leading tuple after excluding the inserted head axis,
each with its own selected plane offset and its own transformed leading offsets
and strides; the rank-two weight's empty leading tuple is not matched and
implies no broadcast, no singleton expansion, and no model-state or cache
broadcast.

`T`, `I`, `O`, `H`, `D`, and `R` are runtime values, never checkpoint
constants. `s` is nonnegative, `R > 0`, and `R <= T-s`, so the selected source
window is exactly `x[b,s+r,i]` for `r` in `0..R-1`; `R` is never inferred from
`s` or `T`, and `O` equals `H*D` only after checked head-planar validation.
With `b` denoting the complete leading tuple, the semantic equations are

```text
out[b,r,o]   = sum(i=0..I-1) x[b,s+r,i] * w[o,i]
out[b,h,r,d] = sum(i=0..I-1) x[b,s+r,i] * w[h*D+d,i].
```

Leading planes, selected rows, heads, and features are independent: no result
crosses a plane or a row window, no result reads a logical element outside
these equations, and no result depends on `16x16` tile padding, sub-byte
remainder bits, or uninitialized storage.

The final untied LM head uses `ordinary` mode with `s = input_run - 1`,
`R = 1`, `H = 1`, and `D = V`, writing `[1,V]` logits directly into the
supplied output with no final-axis slice and no last-row extraction operation.
Projection rearranges only newly computed selected rows. It MUST NOT
materialize repeated KV heads, make a persistent host copy or transpose of
checkpoint weights, or expose a public transpose, head-pack, or extraction
operation.

#### Validation, admission, and failure precedence

Before owner registration, sequence consumption, token acceptance, metadata
effects, or backend work, a submission validates in exactly this order:

1. structural rank and extents: well-formed views, `x` of rank two through
   eight with nonzero extents, a rank-two `w`, an `out` of rank two through
   eight, and a recognized `LinearOutputLayout` value;
2. the mode and head relation: `ordinary` requires exactly `H=1` and `D=O`,
   while `head_planar` checks `H*D` for overflow before comparing it with `O`
   and then requires exactly `O=H*D`; either way a validated request has
   nonzero `H` and `D`, because `O` is a nonzero extent and `H=1` in ordinary
   mode;
3. the selected-row range `s <= T`, `R > 0`, and `R <= T-s`;
4. the exact leading tuple and transformed leading strides, including the
   inserted head-axis exclusion for `head_planar`, with no broadcast and no
   guessed mode;
5. exact queue `Device` identity for every view, stable live owner
   registration, and a live native handle;
6. liveness and bounds: selected plane bounds and leading view bounds;
7. every checked element, bit, byte, address, stride, plane, tile, and product
   arithmetic, including output rank growth, rejecting overflow before
   narrowing, size conversion, or pointer arithmetic;
8. operand, output, and workspace disjointness and alias rules, followed by a
   recognized `QuantizationFormat::NONE`, leaf classification, and immutable
   backend capability; and
9. the caller's supplied workspace against the reported `{bytes, alignment}`:
   liveness, exact-device identity, size, alignment, disjointness from every
   operand and from the output, and lease availability.

The rejection results are normative:

| Condition | Result |
| --- | --- |
| malformed view, rank outside `2..8`, zero extent, non-rank-two weight, invalid final shape, unmatched leading tuple or transformed strides, wrong mode/`H`/`D`/`O` relation, or invalid row window | `InvalidArgument` |
| unrecognized `LinearOutputLayout` value | `InvalidArgument` |
| rank growth beyond eight | `InvalidArgument` |
| output/operand overlap, including conservative same-owner rejection and standard-backend backing-range intersection or identical native handles | `InvalidArgument` |
| supplied workspace that is not a live exact-device range, too small, misaligned, out of range, or overlapping an operand or output | `InvalidArgument` |
| unknown `DataType` or `QuantizationFormat` enumeration value | `InvalidArgument` |
| recognized inapplicable leaf, recognized mixed-operand leaf request, recognized unsupported leaf or backend capability, or recognized non-`NONE` quantization format | `Unsupported`, only after structural validation |
| checked element, bit, byte, address, stride, plane, tile, product, or rank-growth overflow | `Overflow` |
| overlapping live workspace lease, or bounded-resource exhaustion | `ResourceExhausted` |
| pre-acceptance runtime or device failure | `DeviceError` |
| any other unclassified failure | `InternalError` |

Admission failures map through `noexcept` to these negative OIDs and consume
no token, register no owner, submit no work, and mutate no queue, token,
metadata, or output state; zero is never accepted. An accepted submission
returns a positive OID whose shapes, leaf types, quantization, plane offsets,
leading strides, owners, native handles, scalars, `layout`, and workspace range
are immutable. A failure after acceptance leaves output unusable and repeats
the same error on every wait; no rollback and no failed-output-unchanged
guarantee is made, and a consumer of a failed producer MUST NOT be submitted
after that producer's wait fails. The caller MUST fail and drain every accepted
OID before reset or destruction, continuing to drain even when an individual
wait throws.

#### Ownership, liveness, aliasing, and workspace

Output and workspace MUST each be disjoint from every operand and from each
other, so `out` is rejected when it shares an owner with `x` or `w`, including
the conservative case where the transformed windows appear disjoint; standard
backends additionally reject intersecting backing ranges and identical native
handles. Both inputs are reads, so `x` and `w` MAY overlap or alias exactly.
In-place operation is not promised, and admission allocates, relocates,
replaces, or silently converts no operand or output.

Submission snapshots by value the validated shapes, leaf types, quantization,
plane offsets, leading offsets and strides, scalar parameters, `layout`, and
workspace range, together with the exact live registered owner and native
handle of `x`, `w`, and `out`. It registers the distinct owners, deduplicating
read/read aliases, leases positive scratch through proven completion, and
retains no borrowed `TensorView`: a temporary caller view may die as soon as
its submission is accepted, while caller-owned output and workspace stay alive
through proven completion. No hidden allocation, hidden synchronization, host
roundtrip, persistent transpose, or output relocation is permitted, and no
checkpoint weight is copied, transposed, or staged on the host.

The requirement query is pure and deterministic. It allocates nothing,
including host metadata; constructs no request or vector snapshot; registers
or leases no owner; submits nothing; reads no operand or result data; mutates
no queue, token, lease, or state; and depends only on the supplied views and
immutable backend capability, so it stays valid while the queue is occupied. It
validates the same operands, scalars, `layout`, aliasing, capability, and
checked arithmetic as submission and returns `WorkspaceRequirements`, or
throws the established validation exception instead of an OID. The supplied
workspace is deliberately not a query operand: it is validated only by
submission, after a successful query.

#### Dtypes, quantization, arithmetic, and nonfinite behavior

Applicable leaves are exactly these 21:

`I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`,
`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
`F32`, and `F64`.

`BOOL` and `F8_E8M0` are inapplicable: a boolean is neither an integer nor a
signed numeric product, and an unsigned exponent-only encoding is not a general
signed numeric result. `x`, `w`, and `out` use one leaf type and
`QuantizationFormat::NONE`; only `NONE` is supported, and the operation adds no
quantization or storage format. An unknown `DataType` or `QuantizationFormat`
enumeration value is `InvalidArgument`. A recognized inapplicable leaf, a
recognized mixed-operand leaf request, a recognized unsupported leaf or
backend capability, and a recognized non-`NONE` quantization format are
`Unsupported`, and only after structural validation has already succeeded.
Semantic applicability is independent of native storage: a backend limitation
decides which leaves that backend implements and never narrows this
classification. `BOOL` and `F8_E8M0` are inapplicable on every backend, so no
capability record supersedes this classification.

**Integer arithmetic.** The 12 integer leaves use exact dot products in
unsigned modulo-`2^N` arithmetic. Every multiply and add reduces modulo `2^N`
at that step, which is equivalent to reducing only the final sum; no step
depends on signed overflow, and no value is ever cast to an out-of-range signed
type. A signed output leaf stores the two's-complement bit pattern of the
accumulated value, with no saturation, clamping, widening, or implicit
conversion to a floating leaf.

**Scalar floating recurrence.** The scalar path starts at `+0` and processes
increasing `i`: `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`,
`F16`, `BF16`, and `F32` decode to FP32 and accumulate with correctly rounded
FP32 fused multiply-add, `F64` accumulates with FP64 fused multiply-add, and
the accumulated sum is encoded exactly once to the output leaf with the
existing named-format encoding rules. Those existing rules govern infinity,
NaN, saturation, subnormal results, and signed zero; `F8_E4M3FN` has NaN but no
infinity, and `F64` is never narrowed to FP32. Implementations MUST NOT enable
flush-to-zero or fast-math, and no NaN-payload equality is promised.

**Native BF16.** A native BF16 specialization may reassociate the product sum
only when it accumulates in FP32 or in a demonstrably equivalent width whose
result matches the FP32 rule, and it MUST store the result once with
round-to-nearest, ties-to-even. Rounding through BF16 partial products, or
accumulating in BF16, is non-conforming.

**Nonfinite values.** Admission MUST NOT scan operands for nonfinite values and
MUST NOT introduce a host transfer or a linear-specific queued-data failure.
Nonfinite inputs follow the recurrence and the existing named-format rules
above, and no NaN payload is preserved or compared. A post-acceptance failure
is a runtime or device failure and follows the repeat-wait rule of the previous
subsection.

#### Backend capability matrix

Applicability is exactly the 21 leaves above, and the matrix below is the
complete leaf-by-backend capability record.

| Leaf | Contract | CPU | CUDA | ROCm | SYCL | TTNN |
| --- | --- | --- | --- | --- | --- | --- |
| `BOOL` | inapplicable | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
| `I2`, `U2` | applicable | supported | supported | supported | supported | `Unsupported` |
| `I4`, `U4` | applicable | supported | supported | supported | supported | `Unsupported` |
| `I8`, `U8` | applicable | supported | supported | supported | supported | `Unsupported` |
| `I16`, `U16` | applicable | supported | supported | supported | supported | `Unsupported` |
| `I32`, `U32` | applicable | supported | supported | supported | supported | `Unsupported` |
| `I64`, `U64` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F4_E2M1` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F6_E2M3` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F6_E3M2` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F8_E4M3FN` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F8_E5M2` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F8_E8M0` | inapplicable | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
| `F16` | applicable | supported | supported | supported | supported | `Unsupported` |
| `BF16` | applicable | supported | supported | supported | supported | supported |
| `F32` | applicable | supported | supported | supported | supported | `Unsupported` |
| `F64` | applicable | supported | supported | supported | `aspect::fp64` only | `Unsupported` |

CPU, CUDA, ROCm, and SYCL implement all 21 semantic leaves. On CUDA, ROCm, and
SYCL the scalar path covers the 20 non-BF16 leaves and a separate native BF16
specialization covers `BF16`; on CPU, `BF16` uses the scalar recurrence above.
SYCL reports `F64` `Unsupported` at runtime unless the device reports
`aspect::fp64`; that gate is a genuine device fact. TTNN implements mandatory
`BF16` only and explicitly rejects the other 20 applicable leaves, because its
native TILE compute cannot consume the encoded carriers of those leaves without
the forbidden host staging. `BF16` weights, activations, and caches remain
mandatory on all five backends.

**TTNN direct Metalium `BF16` route.** `src/ttnn/linear.hpp`,
`src/ttnn/linear.cpp`, and the `linear_reader`, `linear_compute`, and
`linear_writer` kernels in `src/ttnn/kernels/` implement the mandatory `BF16`
leaf as a direct per-plane Metalium program family on the existing unit mesh:
the reader gathers the selected `x[...,s:s+R,I]` rows and the Hugging Face
`w[O,I]` rows out of the caller's own native DRAM planes into the `in0`/`in1`
operand tiles, the matrix engine accumulates every inner tile in the FP32
destination register through the native matmul facility, and one output pack
performs the single BF16 round-to-nearest-ties-to-even store into the caller's
own output plane. The route allocates no tensor, stages nothing through host
memory, transposes no tensor, and consumes no workspace, so
`linear_workspace_requirements` stays `{0, 1}` for `BF16`, the queue keeps the
same FIFO, owner-registration, fence, and completion machinery, and every other
applicable leaf remains an explicit capability rejection.

**Observed TTNN nonfinite limitation.** That facility's native matrix multiply
does not reproduce this contract's nonfinite classes when an operand is itself
nonfinite or when the product's exponent leaves the accumulator's range.
Executed through this route with BF16 operands and `{0, 1}` workspace,
`inf*inf`, `huge_finite*inf`, `inf*huge_finite`, and `1.7e38*1.7e38` all
return `+0` where the contract's FP32 recurrence returns `+inf`, and a `NaN`
operand returns `+inf` where the recurrence returns `NaN`; finite products,
finite overflow (`max_finite*max_finite -> +inf`), `inf*finite(1) -> +inf`, and
signed zero agree with the recurrence. The behavior is independent of
`fp32_dest_acc_en`, so it is an unpack/multiply property of the facility rather
than a port or destination-accumulate defect. The shared `BF16` fixture
deliberately places `x = 3.38953e38` and `w = +inf` in the same inner row, so
`canonical I=3 O=10 H=2 D=5 T=19 s=2 R=1 ordinary` exercises both declared
limitations on this backend: the elements those declarations cover are
observed and counted rather than asserted, and every other element of that case
is still compared under the unchanged tolerances. Before the declarations
existed, a strict run of the same case failed exactly one of its assertions
with 1,505,991 of the TTNN conformance target's 1,505,992 assertions passing;
that pre-declaration measurement is the history that motivated the two `false`
flags, not the current state of the suite. A port whose facility cannot express these classes
records the measured limitation in its `LinearDeclaration`
(`nonfinite_classes_asserted = false`) instead of failing conformance: the
shared comparison then still checks every finite expectation under the
unchanged tolerances, still requires the observed element from accepted queued
work, and observes a nonfinite expected class rather than asserting it, and it
counts every element it observes that way in a `LinearComparisonRecord`. The
TTNN declaration sets that flag from the measured table below, and it also
sets `subnormal_operands_preserved = false` from the separate subnormal table
further below; CPU, CUDA, ROCm, and SYCL keep both defaults (`true`) and their
unchanged class and finite assertions, so these two exceptions are TTNN-only
and no other backend may carry them.
Until a developer decision changes the fixture, the obligation, or the
facility, the TTNN `BF16` linear leaf is implemented and exercised with both
measured limitations recorded, and it is not reported as supported numerical
conformance for the affected classes and elements. The residual
subnormal-operand deviation recorded below is the finite deviation the
`subnormal_operands_preserved = false` declaration observes rather than
asserts, so the port remains short of full numerical conformance for that
element until a developer decision addresses that second facility limitation.
At the five-backend gate revision the TTNN conformance target passes `47/47`
cases with `1,859,192/1,859,192` assertions, of which the linear projection
case contributes `1,449,065`; that case also requires both declarations to stay
live (`skips.nonfinite_elements > 0` and `skips.subnormal_elements > 0`), so
the two flags cannot rot into dead declarations.

**Facility class expressibility (measured).** Probing one `32x32x32` matrix
operation per case through this same route, with `MathFidelity::HiFi4` under
both `fp32_dest_acc_en` settings, the facility agrees with the contract's FP32
FMA rule for `inf*finite(1) -> ±inf`, `inf + inf -> +inf`,
`max_finite*max_finite -> +inf`, and subnormal underflow, and disagrees where
the rule requires an infinity from a nonfinite or extreme operand
(`inf*inf`, `max_finite*inf`, `inf*max_finite`, and `1.7e38*1.7e38` all return
`+0` instead of `+inf`), where it requires `NaN` (`inf*0`, `0*inf`,
`inf + (-inf)`, and `max*max + (-max*max)` return `+0` instead of `NaN` or
`-inf`), and for every `NaN` operand, which returns `+inf` instead of `NaN`.
No tested operand encoding produces a `NaN` output at all, so no selection over
the facility's own outputs can synthesize that class; an implementation that
wanted these classes would have to detect nonfinite operands and inject the
class outside the matrix facility, which is an elementwise substitute in the
sense of
[Native matrix evidence obligations](#native-matrix-evidence-obligations) and
is therefore not accepted as native TTNN evidence.

**Facility subnormal handling (measured).** The same facility flushes
subnormal operands to zero before the multiply: `1.0` times the largest BF16
subnormal (`0x007f` = `1.16631e-38`, whose FP32 product `1.16631e-38` is
normal) returns `+0`, and `max_finite` times that subnormal returns `+0` where
the rule gives `3.95325`; `min_normal` (`0x0080` = `1.17549e-38`) and larger
operands multiply correctly, and a subnormal-times-subnormal product
underflows to zero under both rules. A request whose inner sum contains such a
product is therefore a finite-value deviation, not a nonfinite class, and the
`nonfinite_classes_asserted` declaration does not cover it. Measured through
the same production route with `MathFidelity::HiFi4` and `fp32_dest_acc_en`
true.

Missing implementation is never unsupported hardware. An unported backend or
leaf reports `Unsupported` as missing capability and names its missing
evidence; no backend may advertise a capability it has not implemented, and a
rejection-only probe, a storage-only observation, or a host or elementwise
substitute is never linear conformance. The per-backend native feasibility
records remain the
[CUDA](#tinyllama-forward-layout--cuda-matrix-feasibility),
[ROCm](#tinyllama-forward-layout--rocm-matrix-feasibility),
[SYCL](#tinyllama-forward-layout--sycl-matrix-feasibility), and
[TTNN](#tinyllama-forward-layout--ttnn-matrix-feasibility) matrix records.

#### Workspace requirements and packed-writer ownership

The reported requirement is exact per path. `P` is the checked product of the
logical leading extents of `x`, equivalently of `out`; `pad16(n)` is the
checked round-up of `n` to a multiple of 16; and `A32(n)` is the checked
round-up of `n` to a multiple of 32, which is the alignment those paths report.

| Path | Requirement |
| --- | --- |
| CPU, every applicable leaf | `{0, 1}` |
| CUDA, every applicable leaf | `{0, 1}` |
| ROCm, the 20 scalar leaves | `{0, 1}` |
| SYCL, the 20 scalar leaves | `{0, 1}` |
| ROCm `BF16` | alignment 32, bytes `A32(P*pad16(R)*pad16(I)*2) + A32(P*pad16(R)*pad16(O)*2)` |
| SYCL `BF16` | alignment 32, bytes `A32(P*pad16(R)*pad16(O)*4)` |
| TTNN `BF16` | `{0, 1}` |

A `{0, 1}` requirement means only the empty `RawWorkspaceView{}` is
admissible, and a supplied owner is `InvalidArgument` before dispatch. For
TTNN `BF16` that is the frozen direct-reader and direct-writer route: the
operation advertises no packed operand or product region, and a route that
needed one would require changing this contract instead of silently reporting
a different query result. Where the requirement is positive, the range MUST be
a live exact-device owner range, 32-byte aligned, at least the reported bytes,
disjoint from every operand and from the output, and leased through proven
completion; owner-absolute subranges
are checked and 32-byte aligned, disjoint aligned ranges may be used
concurrently, and overlapping live leases reject with `ResourceExhausted`. The
positive range is caller-owned staging only: it carries no control packet, so
no status word is reset, transferred, or interpreted, and no operand, weight,
or result is staged on the host or inside a library's hidden internal
workspace. Proven completion releases the lease and permits safe reuse of
independent scratch; unknown completion retains or quarantines the range
instead of reusing it.

**Packed sub-byte ownership.** The `I2`, `U2`, `I4`, `U4`, `F4_E2M1`,
`F6_E2M3`, and `F6_E3M2` leaves are stored packed at their logical width. Every
packed writer MUST own whole physical write units race-safely: exactly one
writer per destination byte or word, no read-modify-write of a unit that a
concurrent writer also owns, and no two writers sharing one packed unit for
different logical cells. Every physical bit outside a writer's own logical
cells, including remainder bits, padding bits, and `16x16` tile padding, MUST
be preserved, and valid logical output MUST NOT depend on those bits.

#### Fixtures, reference, and tolerances

The fixture set is fixed before measurement and MUST include all of the
following:

- the non-square reference `I=3, O=10, H=2, D=5, T=19, s=2` across the exact
  runs `R=17` selecting `x[2..18]`, `R=16` selecting `x[2..17]`, `R=15`
  selecting `x[2..16]`, and `R=1` selecting `x[2]`. Ordinary output columns
  `o=0..9` consume `w[0]` through `w[9]`; head-planar `(h=0,d=0..4)` maps to
  `o=0..4` and `(h=1,d=0..4)` maps to `o=5..9`. A transposed checkpoint
  weight, an inferred or off-by-one row window, and an `h`/`d` shuffle are
  therefore observably wrong;
- the non-tile inner extent `I=65`, which spans five `16`-wide inner tiles and
  exercises inner-tail masking, accumulation order, and packed sub-byte inner
  tails;
- independent leading planes: rank two through eight, distinct per-operand
  plane offsets and leading transforms for `x` and `out`, a distinct final row
  so a wrong `s` is observable, and no result that crosses a plane;
- padding invariance: deliberately perturbed `16x16` tile padding, sub-byte
  remainder bits, and uninitialized storage whose change never alters valid
  output; and
- the LM-head case with `s = input_run - 1`, `R = 1`, ordinary mode, and
  `[1,V]` output.

Comparison policy is fixed before measurement, and the reference MUST be
independent: production code MUST NOT serve as its own oracle.

| Leaf class | Comparison |
| --- | --- |
| the 12 integer leaves | exact output bits |
| `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16` | exact encoded result of the scalar recurrence |
| `BF16` | the FP64 evaluation of the equation rounded once to BF16, with `abs(actual - q) <= max(ULP_BF16(q), 2^-7, 2^-6 * abs(q))` |
| `F32` | `abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)` |
| `F64` | `abs(actual - reference) <= 1e-12 + 1e-12 * abs(reference)` |

These are fixed fixture thresholds, not arbitrary-input accuracy promises.

The independent reference MUST evaluate the ordinary and head-planar equations
in exact integer arithmetic for the integer leaves and in FP64 for the floating
leaves, derive the row window, head, and weight-row mapping itself, and include
a deliberate permuted-oracle or transposed-weight sanity check. Shared coverage
MUST exercise every applicable leaf, both layouts, all four row runs, the
`I=65` inner tail, non-tile `O`, `H`, `D`, and `I`, independent leading planes
and transformed mappings, padding invariance, the LM-head case, malformed
structure, wrong device, stale registration, alias and overlap, mixed and
unsupported leaves, non-`NONE` quantization, every checked-overflow class,
invalid `layout` values, row-window violations, rank growth, workspace
liveness, size, alignment, device, overlap, and lease rejections, an accepted
failure with repeated waits, and scratch reuse after proven drain.

The cases live in the shared header
`test/backend/backend_conformance_linear.hpp` and run through the existing
`iom_backend_conformance_cpu_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, `iom_sycl_conformance_tests`, and
`iom_ttnn_conformance_tests` drivers. No second test project, generic test
framework, model fixture, checkpoint, or network dependency is permitted, and
common code never switches on backend kind. An unported backend keeps an empty
linear span and asserts `Unsupported` for every applicable leaf; that rejection
probe is not projection conformance. Each of the five ports migrated its own
`linear` `Unsupported` probe when it landed, so no driver keeps a blanket
linear rejection probe today, while the unrelated copy/add/mul/sub/div suites
remain unchanged.

The fixture provenance and case links of that header are:

- the independent raw/scalar reference decodes and encodes every named leaf
  itself, implements the unsigned modulo-`2^N` integer dot and the `+0`-started
  increasing-`i` correctly rounded FP32 fused-multiply-add recurrence (`FP64`
  for `F64`) with exactly one final encode, derives the row window, head
  coordinate, and Hugging Face weight row itself, and provides the independent
  FP64 evaluation behind the BF16, F32, and F64 thresholds; it calls no
  production codec, address mapper, admission, linear, CPU, or backend helper;
- the deliberate negative variants are `transposed_weight`, `head_order`,
  `head_repeat`, `wrong_row_window`, `mixed_plane`, `padding_dependent`,
  `tile_tail`, and `numeric_reencode`; the self-check asserts for every
  applicable leaf that each is detected — bit-exactly where the thresholds
  compare exact bits, and as a demonstrated model difference where a threshold
  admits a bounded deviation — and that each coincides with the reference on a
  fixture where the perturbation is genuinely invisible, so detection is
  selective rather than unconditional;
- the shared case matrix supplies the canonical `I=3, O=10, H=2, D=5, T=19,
  s=2` fixture across `R=1, 15, 16, 17` in both layouts, the `I=65` and `I=33`
  inner tails across 16-wide inner tiles and 32-wide inner groups, the LM-head
  `s = input_run - 1, R = 1` `[1, V]` request, the leading-plane profiles of
  rank two through eight with distinct per-operand plane offsets, leading
  transforms, and strides, deliberately poisoned native padding, and per-leaf
  special-class fixtures whose zero, signed, subnormal, infinity, and NaN
  classes match each leaf's own encoding;
- each driver passes an explicit declaration of its target leaf matrix, its
  scalar and native-BF16 paths, its exact scratch path (`{0, 1}`, the ROCm
  aligned BF16 sum, or the SYCL `A32(P*pad16(R)*pad16(O)*4)` range), the leaves
  this revision implements, and — on SYCL — the `sycl::aspect::fp64` device
  fact that gates `F64`, so a capability expectation is never an echo of an
  implementation report;
- ROCm implements all twenty-one applicable leaves: the twenty non-BF16
  leaves through the shared raw-word scalar kernel and `BF16` through the
  native specialization in `src/rocm/copy.hip` (checked caller workspace,
  direct GFX12 wave32 BF16 WMMA with FP32 accumulation, one RNE BF16
  scatter), and declares exactly that implemented span, so the suite runs the
  independent reference numerically on the real device for every applicable
  leaf;
- the ROCm per-leaf device evidence is complete for all twenty-one applicable
  leaves under the shared fixture set and comparison policy; the `F32` and
  `BF16` non-finite-class reconciliation authorized for the shared fixture is
  part of the gate this leaf ran, and no leaf is reported separately as
  unclaimed here; and
- the common admission, ownership, workspace, queue-order, and failure cases
  run against a declaration-driven common double next to the host-only
  reference self-check.

The executed ROCm native `BF16` record of this revision is the following
observation, collected from this leaf's task worktree (worktree base
`3421f67fb6817e97444353bb03343ac33b02b24d`) on the configured `rocm` host:

| Field | Observation |
| --- | --- |
| Backend, device | ROCm, `gfx1201` (Radeon AI PRO R9700), wave32, ordinal 0 |
| Toolchain | HIP `7.15.26333-0000000`, AMD clang `23.0.0git`; `hipRuntimeGetVersion` `71526333` |
| Profiler | `rocprofv3` version `1.3.5`, git revision `6b0e43f341195e203754e08f850e437ff2fc09f9` |
| Profiler command | `rocprofv3 --kernel-trace --output-format csv -d prof -o evi -- <evidence driver>` |
| Layout, dimensions | `x[...,T,I]` and HF `w[O,I]`: `P=1, T=19, I=3, O=10, s=2`; ordinary `H=1, D=10` and head-planar `H=2, D=5`; `R in {1,15,16,17}` |
| Padded dimensions | `pad16(R)/pad16(I)/pad16(O)`: `16/16/16` for `R in {1,15,16}` and `32/16/16` for `R=17` |
| Kernel symbols | `linear_bf16_pack_kernel`, `linear_bf16_wmma_kernel`, `linear_bf16_scatter_kernel` (`iom::detail`, one dispatch of each per accepted submission; `9/9/9` dispatches for the eight fixture submissions plus one wider submission) |
| Facility | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` executed in the same session as the submissions and verified against the exact `16x16x16` product (`facility_probe` `pass`, `worst_deviation=0`); the fixture products themselves are compared against the FP64-derived reference by the shared suite |

| `R` | Layout | `pad16(R)/pad16(I)/pad16(O)` | Workspace (bytes, alignment) | Accepted OID | Observed facility |
| ---: | --- | --- | --- | ---: | --- |
| 1 | ordinary | `16/16/16` | `1024, 32` | `36028797018963969` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 1 | head-planar | `16/16/16` | `1024, 32` | `36028797018963970` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 15 | ordinary | `16/16/16` | `1024, 32` | `36028797018963971` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 15 | head-planar | `16/16/16` | `1024, 32` | `36028797018963972` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 16 | ordinary | `16/16/16` | `1024, 32` | `36028797018963973` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 16 | head-planar | `16/16/16` | `1024, 32` | `36028797018963974` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 17 | ordinary | `32/16/16` | `2048, 32` | `36028797018963975` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 17 | head-planar | `32/16/16` | `2048, 32` | `36028797018963976` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |

The unsupported device is reported separately: the installed `gfx1036`
(ordinal 1) returns `Unsupported` for `BF16` from both the pure requirement
query and submission, and only the twenty scalar leaves stay available there.
`rocprofv3` 1.3.5 supports neither PC sampling nor SPM counter collection on
this `gfx1201` agent (`Given PC sampling configuration is not supported on
any of the agents`, `rocprofiler_iterate_agent_supported_counters failed ...
Agent HW architecture is not supported, no counter metrics found`), so the
facility statement above rests on the executed probe and the traced kernel
dispatches of the submitted OIDs rather than on a hardware instruction
counter.

The executed SYCL native `BF16` record of this revision is the following
observation, collected on the configured SYCL host from the SYCL native `BF16`
leaf's own task worktree at commit `ed92ff6`, which is this gate's revision,
and re-observed here by the gate's own runs of the same production route:

| Field | Observation |
| --- | --- |
| Backend, device | SYCL over Level Zero V2, `Intel(R) Arc(TM) Pro B60 Graphics`, architecture `intel_gpu_bmg_g21`, driver `1.15.38646+7`, subgroup sizes `16,32`, `ext_intel_matrix` present, PCI `8086:e211` |
| Toolchain | Intel oneAPI DPC++/C++ Compiler `2026.1.0`, `-ffp-model=precise`; `ocloc` `26.22.38646.7` |
| Evidence commands | The leaf's production-queue evidence driver (`record:` lines) invoked through `csw-remote-exec` with `csw-remote-sync` immediately before it and `flock -w 300` plus remote-side `timeout --kill-after=30s`; the gate then re-ran `./build/test/iom_sycl_conformance_tests` directly and through `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_(conformance\|smoke)_tests$'` |
| Layout, dimensions | `x[...,T,I]` and HF `w[O,I]`: `P=1, T=19, I=3, O=10, s=2`; ordinary `H=1, D=10`; head-planar `H=2, D=5`; `R in {1,15,16,17}` |
| Padded dimensions | `pad16(R)/pad16(I)/pad16(O)`: `16/16/16` for `R in {1,15,16}` and `32/16/16` for `R=17` |
| Scratch | alignment `32`; `1024` bytes for `R in {1,15,16}` and `2048` bytes for `R=17`, exactly the reported `A32(P*pad16(R)*pad16(O)*4)` |
| Kernel symbols | `LinearBf16MadKernel<16ul>`, `LinearBf16TailKernel<1ul>`, `LinearBf16TailKernel<16ul>`, and `LinearBf16PackKernel`, traced as `4/4/2/8` occurrences over the eight evidence submissions, one pack per output tile row |
| Facility | The selected device reports `ext_intel_matrix`, subgroup `16`, and the BF16/BF16/FP32 combination; the executed route is the subgroup-16 `joint_matrix` MAD path with an explicit RNE BF16 pack, not a host or elementwise substitute |
| ISA confirmation | Attempted and unavailable: `clang-offload-extract` on the built conformance binary yields ten `sycl-spir64` SPIR-V images, and `ocloc compile -spirv_input -device bmg` followed by `ocloc disasm` on each reports `dpas=0` and no `LinearBf16` symbol text (`ocloc 26.22.38646.7`), so the record rests on the traced dispatch and the device facts rather than on disassembly |

| `R` | Layout | `pad16(R)/pad16(I)/pad16(O)` | Scratch (bytes, alignment) | Accepted OID | Wait |
| ---: | --- | --- | --- | ---: | --- |
| 1 | ordinary | `16/16/16` | `1024, 32` | `36028797018963969` | `ok` |
| 1 | head-planar | `16/16/16` | `1024, 32` | `36028797018963970` | `ok` |
| 15 | ordinary | `16/16/16` | `1024, 32` | `36028797018963971` | `ok` |
| 15 | head-planar | `16/16/16` | `1024, 32` | `36028797018963972` | `ok` |
| 16 | ordinary | `16/16/16` | `1024, 32` | `36028797018963973` | `ok` |
| 16 | head-planar | `16/16/16` | `1024, 32` | `36028797018963974` | `ok` |
| 17 | ordinary | `32/16/16` | `2048, 32` | `36028797018963975` | `ok` |
| 17 | head-planar | `32/16/16` | `2048, 32` | `36028797018963976` | `ok` |

Per-`R` conclusion: `R=1`, `15`, `16`, and `17` are each supported on both
layouts; every submission returned its positive accepted OID and completed its
wait, and each is covered by the traced dispatch symbols above. The gate's own
re-run of the same production route at this revision holds the same result: the
filtered linear projection case passes `1/1` case with
`5,670,248/5,670,248` assertions including both crossing-factorization shapes,
the full target passes `29/29` cases with `5,905,415/5,905,415` assertions, and
`ctest` passes `2/2` including the smoke target, all with the Level Zero GPU
enumerated before the run.

#### Native matrix evidence obligations

Native matrix work is evidenced by execution, not by inspection or analogy.
Every native BF16 implementation records the backend, the exact device, the
toolchain and profiler versions, the exact profiler command, the layout, `P`,
`T`, `I`, `O`, `H`, `D`, `s`, and `R`, the padded dimensions, the exercised
submission and its accepted OID, the kernel symbol, and the observed native
matrix instruction or facility. The record MUST connect the executed submission
to the matrix kernel; compile success, disassembly presence, or a bounded
capability sample alone is insufficient, and an unrun production port is
reported as unimplemented rather than supported. Supported or unsupported
conclusions are recorded separately for `R=1`, `15`, `16`, and `17`, and each
of those conclusions must come from real execution. CPU scalar, host,
elementwise, and emulated substitutes are never native evidence for CUDA, ROCm,
SYCL, or TTNN. Accelerator verification follows the `csw-remote` procedure, and
TTNN verification uses a 300-second timeout.

**Observed TTNN `BF16` evidence.** Remote executions used profile `ttnn`
(mirror `cswrun-20260918-04lp-11-ttnn-v1`) with `csw-remote-sync` immediately
before every `csw-remote-exec`, remote-side `timeout --kill-after=30s`, and a
bounded `flock -w` hardware lock, against the installed Blackhole (device 0,
UMD firmware bundle `19.13.1`, TT-Metalium
`v0.76.0-dev20260801-268-g06994d4afda`, `TT_METAL_HOME` from the profile's
`ttnn_env.sh`). The gate command is
`ctest --test-dir build/ttnn --output-on-failure --timeout 300 -R "^iom_ttnn_conformance_tests$"`.
Layout: ordinary and head-planar planes, rank two through eight, `I=3`,
`O=10`, `H=2`, `D=5`, `T=19`, `s=2`, native padded tile grid `32 x 32` rows
and columns per plane with the `I=65` fixture spanning three inner tile
columns, workspace `{0, 1}`. Kernel symbols: `linear_reader`,
`linear_compute` (the matrix facility on one Tensix core, one `MeshWorkload`
per output plane), and `linear_writer`. Real accepted submissions: `R=1`,
`R=15`, `R=16`, and `R=17` each returned a positive OID (queue 1, sequence 1:
`36028797018963969`) and completed with zero mismatches against an independent
host FP64 reference under the contract's BF16 bound; the shared canonical
`R=1` case reaches the device with exactly the elements the two declared
facility limitations cover observed and counted rather than asserted, which
the TTNN driver pins with `skips.nonfinite_elements > 0` and
`skips.subnormal_elements > 0`.

#### Same-revision five-backend gate evidence

The closing Linear-projections gate ran every backend's conformance target and
its direct binary at one revision, from one prepared task worktree, and
recorded the device, runtime, and toolchain identity of each run. The CPU pair
ran locally; every accelerator pair used exact-worktree `csw-remote`
sync/exec with a fresh sync immediately before each execution, remote-side
`timeout --kill-after=30s`, and a bounded hardware lock, and the TTNN pair kept
its 300-second bound. The observed results are:

| Backend | Commands | Device / runtime identity | Observed result |
| --- | --- | --- | --- |
| CPU | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_backend_conformance_cpu_tests$'` and the direct `./build/test/iom_backend_conformance_cpu_tests` | Local host CPU, C++20 release build | `24/24` cases, `5,934,262/5,934,262` assertions, `1/1` CTest pass |
| CUDA | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_cuda_conformance_tests$'` and the direct binary on remote `bv1` | `NVIDIA GeForce RTX 5090`, compute capability `12.0`, driver `595.71.05`, `nvcc` release `13.2` build `V13.2.78` | `31/31` cases, `5,954,449/5,954,449` assertions, `1/1` CTest pass |
| ROCm | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_rocm_conformance_tests$'` and the direct binary on remote `bv2` | `gfx1201` (`AMD Radeon AI PRO R9700`), HIP `7.15.26333-0000000`, AMD clang `23.0.0git` | `32/32` cases, `5,945,213/5,945,213` assertions at the recorded run (one other identical run reported `5,945,212`; no case failed in any run), `1/1` CTest pass |
| SYCL | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_(conformance\|smoke)_tests$'` and the direct binary on remote `bh2` | Level Zero V2 `Intel(R) Arc(TM) Pro B60 Graphics`, oneAPI DPC++/C++ `2026.1.0`, `ocloc` `26.22.38646.7` | `29/29` cases, `5,905,415/5,905,415` assertions, `2/2` CTest pass including the smoke target |
| TTNN | `ctest --test-dir build/ttnn --output-on-failure --timeout 300 -R '^iom_ttnn_conformance_tests$'` and the direct binary on remote `bv1` | Blackhole device 0, UMD firmware bundle `19.13.1`, TT-Metalium `v0.76.0-dev20260801-268-g06994d4afda` | `47/47` cases, `1,859,192/1,859,192` assertions, `1/1` CTest pass |

CPU, CUDA, ROCm, and SYCL therefore demonstrate all twenty-one applicable
leaves at this revision, and TTNN demonstrates the mandatory `BF16` leaf with
the twenty other applicable leaves explicitly rejected after structural
validation. No planned, rejection-only, storage-only, compile-only, or
unsupported probe is counted as a pass anywhere in that table.

**Crossing-factorization obligation.** The two additive head-planar crossing
cases (`H=2 D=9 O=18` and `H=4 D=5 O=20`, `T=19`, `s=2`, `R=17`, every
declared leaf) exist because the earlier fixture set's only head stride,
`H=2 D=5`, never crossed a packed 16-column boundary and therefore left a real
head-planar scatter defect invisible. This gate re-verified every backend
against the independent FP64 oracle with those cases included; a filtered run
of the shared linear projection case with successful-assertion reporting
prints each case's label in its observed comparison contexts, so the cases are
observably executed rather than skipped:

| Backend | Crossing-case contexts (`H=2 D=9 O=18` / `H=4 D=5 O=20`) | Filtered linear case result |
| --- | --- | --- |
| CPU | `39,234` / `43,710` | `1/1` case, `5,676,263/5,676,263` assertions |
| CUDA | `39,045` / `43,521` | `1/1` case, `5,670,287/5,670,287` assertions |
| ROCm | `39,045` / `43,521` | `1/1` case, `5,670,297/5,670,297` assertions |
| SYCL | `39,045` / `43,521` | `1/1` case, `5,670,248/5,670,248` assertions |
| TTNN (`BF16`, its declared span) | `2,484` / `2,784` | `1/1` case, `1,449,065/1,449,065` assertions |

No backend reported a crossing-case mismatch, so no case, tolerance, or
declaration was weakened or re-scoped at this gate, and the TTNN row stays
inside TTNN's documented mandatory-`BF16` exception. The `R=1`, `15`, `16`,
and `17` native-facility records above remain the per-backend execution
records for those logical runs.

**TTNN per-leaf scope (observed at this revision).** A temporary `_local`
probe driven through `csw-remote` on the TTNN host submitted the canonical
ordinary `I=3, O=10, T=19, s=2, R=17, H=1, D=10` request for all twenty-one
applicable leaves through the real queue and OID path. `BF16` returned the pure
`{0, 1}` requirement, accepted the positive OID `36028797018963969`, completed
its wait, and wrote the output; each of the other twenty leaves returned
`Unsupported` from both the pure query and the submission (negative OID), and
no leaf returned any other category — `accepted=1 rejected=20 other=0`. With
`R=0`, the implemented `BF16` leaf and a rejected non-`BF16` leaf both return
`InvalidArgument` rather than the capability rejection, so structural
validation precedes the dtype/backend capability decision exactly as this
contract requires. The device stores `22` leaves (`F8_E8M0` excluded), so the
twenty rejections are capability statements about an implemented port, not
storage-only observations; the probe was removed after the run.

#### Implementation references and delivery prerequisites

Implementers need these existing sources and seams:

- public facade, declarations, and default hooks: `include/iom/iom.hpp` and
  `src/device_ops_neural.cpp`;
- admission, request snapshots, owner registration, FIFO acceptance, deferred
  completion, repeated waits, and quarantine: `src/device_ops.cpp` and
  `src/device_ops_binary.cpp`;
- checked logical shapes, slots, and leading-plane arithmetic: `src/tensor.cpp`
  and `src/iom_internal.hpp`;
- workspace owner bytes, 32-byte subranges, leases, proof, and quarantine:
  `src/workspace.cpp` and `include/iom/detail/workspace_registry.hpp`;
- standard tiled word ownership, packed sub-byte writers, and launch mapping:
  `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_add.inl`,
  `src/shared/gpu_queue.hpp`, and `src/shared/gpu_queue_operations.inl`;
- CPU asynchronous worker and raw codecs: `src/cpu/queue.cpp` and
  `src/cpu/transfer_helpers.hpp`;
- backend capability classification, queues, and native storage:
  `src/cuda/copy.hpp`, `src/cuda/copy.cu`, `src/rocm/copy.hpp`,
  `src/rocm/copy.hip`, `src/sycl/queue_internal.hpp`, `src/sycl/queue.cpp`,
  `src/sycl/queue_binary.cpp`, `src/ttnn/device_types.cpp`,
  `src/ttnn/queue.cpp`, and `src/ttnn/queue_internal.hpp`; and
- independent reference and conformance harness:
  `test/backend/backend_conformance_oracle.hpp`,
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_other.hpp`, and the five
  `test/<backend>/test_<backend>_conformance.cpp` drivers.

`test/backend/backend_conformance_linear.hpp` is the shared conformance header
described above. The per-backend linear kernels, launch wrappers, and
capability predicates are owned by their own ports, which replace only their
own hooks and migrate their own `Unsupported` probe as each declared leaf is
actually implemented. The executed records this section carries — the ROCm and
SYCL native `BF16` observations, the TTNN `BF16` evidence, and the
same-revision five-backend gate table — are observations at their named
revisions; the CUDA and TTNN native records referenced above live in their own
matrix-feasibility subsections, and nothing here extends a recorded result to
an unexercised shape, device, leaf, or revision.

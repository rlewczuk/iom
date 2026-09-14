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
   build/test execution follows the repository remote-development procedure.

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

Other compute hooks (`silu`, `linear`, `rmsnorm`, and `sdpa`) remain unsupported
and return negative `Unsupported` before submission, mutation, or token
acceptance.

#### TinyLlama forward layout — Embedding and projection boundaries

This subsection freezes planned backend-neutral boundaries; it does not declare
or implement them. No embedding overload is declared by the current neural
facade, and the existing three-view `linear` signature remains unchanged and
returns `Unsupported` as specified above. Both target operations remain
unavailable until the operation-owned
[Embedding lookup](#embedding-lookup) and
[Linear projections](#linear-projections) sections and their backend gates
land. In the target ABI, `LinearOutputLayout` and the methods are exactly:

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
tolerances, and backend support are owned by
[Embedding lookup](#embedding-lookup).

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
element, stride, byte-range, output-size, and other shape products. Linear
additionally validates a recognized `layout`, `s <= T`, `R > 0`,
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
before a kernel belong to the linear contract; this table does not infer
integer normalization or silently mean “all floats.” Only
`QuantizationFormat::NONE` is applicable; every other quantization format is
rejected.

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

The following interfaces are planned ABIs. This documentation does not make
them available:

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

The RMS normalization implementation MUST remove the current redundant `dim`
argument in the same declaration/caller/test cutover. The last logical axis is
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
backend limitations remain solely owned by the planned **RMS normalization**
and **SiLU** operation sections. Until each operation sibling lands, the
current `rmsnorm` and `silu` calls continue to return negative `Unsupported`
before submission, mutation, or token acceptance; no declaration or kernel is
changed by this documentation contract.

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

The following RoPE and cache-append interfaces are planned ABIs, not currently
declared or supported facades:

```cpp
oid rope(
        const TensorView& x, TensorView& out, size_t a, double theta,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rope_workspace_requirements(
        const TensorView& x, const TensorView& out, size_t a, double theta);

oid cache_append(
        const TensorView& source, TensorView& destination, size_t a,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements cache_append_workspace_requirements(
        const TensorView& source, const TensorView& destination, size_t a);
```

Their operation siblings MUST add these declarations and implementations before
reporting support. Each requirements query mirrors all semantic arguments,
accepts no workspace argument, and MUST be pure: it performs no allocation,
registration, submission, sequence reservation, or state mutation.

For RoPE, `x` and `out` MUST have identical logical shape `[...,H,R,D]` and
rank three through eight. `H` and `R` are nonzero, `D` is positive and even,
and the views have identical leading tuples, dtype, device, quantization, and
mapping-compatible final axes. Every logical output element MUST be written
from its corresponding input element, honoring the independent leading offset
and strides of both views and the logical mapping of partial final tiles. If
`b` denotes the complete leading-coordinate tuple, then for every `j < D/2`

```text
angle(a,r,j)       = (a+r) * theta^(-2*j/D)
y[b,h,r,j]         = x[b,h,r,j] * cos(angle(a,r,j))
                     - x[b,h,r,j+D/2] * sin(angle(a,r,j))
y[b,h,r,j+D/2]     = x[b,h,r,j] * sin(angle(a,r,j))
                     + x[b,h,r,j+D/2] * cos(angle(a,r,j))
```

These are exactly first-half/second-half pairs, never adjacent pairs. The
operation MUST use the explicit absolute position `a+r`; it has no hidden
cursor, state broadcast, subtraction artifact, scaling parameter, or alternate
angle convention. `theta` MUST be finite and greater than zero. Admission MUST
compute `a+(R-1)` with checked arithmetic and enforce the representable
position/conversion bound fixed by the RoPE numerical contract before any
kernel effect. The session's bounded per-layer capacity `C` supplies the model
positions and model-position bound; RoPE does not own `C`. The ABI's `double`
scalar does not require FP64 device trigonometry. The RoPE operation's
numerical sibling MUST define applicable dtypes, reference and fixture
provenance, special-value behavior, tolerances, BF16 operation-boundary
rounding, and backend feasibility.

Q and K MUST be submitted to RoPE independently, so `Hq != Hkv` is valid.
Independent Q and K calls are each admitted under their own operation alias
contract; an exact alias relationship between operands read by those separate
calls MUST NOT be rejected merely because it is a Q/K alias category.
Repeated one-row calls use continuous absolute positions: calls with `R=1`
and `a=1`, `a=15`, `a=16`, or `a=17` rotate positions 1, 15, 16, and 17
respectively, with no reset or discontinuity at the 16-row tile boundary. A
multi-row call starting at any such `a` uses the consecutive positions
`a` through `a+R-1`.
Consecutive calls MUST pass the next `a` as the preceding `a+R`; for example,
successive one-row calls at 15, 16, and 17 cross the tile boundary without
reusing or skipping a position.

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

This subsection specifies a planned facade boundary. It does not declare or
implement a neural operation. The seven target `DeviceOps` facades return
`oid`, are `noexcept`, and have exactly these signatures:

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
when the query reports zero bytes. CPU and TTNN currently reject creation of a
positive `RawWorkspace`; the first operation that actually requires positive
scratch on either backend owns the minimal factory support and conformance
tests. A capability-blocked backend MUST name the missing evidence instead of
inventing a byte requirement.

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

The planned signatures and queries above are not current API declarations.
The existing neural hooks remain `Unsupported` until their real operation
ports land; this subsection changes no facade, kernel, queue, session, or
selector implementation.

#### TinyLlama forward layout — Final-logits selection and ownership

This subsection freezes the planned synchronous selection boundary for the
final untied LM-head result described by
[Embedding and projection boundaries](#tinyllama-forward-layout--embedding-and-projection-boundaries).
It does not declare or implement the interface. The exact target ABI is:

```cpp
class TokenSelector {
public:
    virtual ~TokenSelector() = default;
    virtual std::size_t select(
            DeviceOps& queue,
            const TensorView& logits,
            std::size_t valid_vocabulary,
            oid producer,
            std::span<const std::size_t> history) = 0;
};
```

`select` is deliberately not `noexcept`. Invalid input, readiness, runtime,
and device-data failures are delivered with standard exceptions. The method
returns one token ID only after the producing work and all selector work have
completed successfully; it exposes no asynchronous selection result or
selector OID.

`logits` MUST be a borrowed const BF16 view with exact logical rank-two shape
`[1,V]`, where both dimensions are nonzero. The view and its live storage
owner MUST belong to the exact device served by `queue`. `valid_vocabulary`
MUST be nonzero and exactly equal the logical `V`; the only selectable IDs are
`[0,V)`. Physical 16x16 tile padding, and any physical row or feature outside
that logical extent, MUST NOT be read as a candidate or included in validation
of logit values.

`producer` MUST be positive and MUST identify work actually submitted by this
same live `DeviceOps` queue. An OID from a different queue is invalid even when
both queues serve the same backend device; zero, negative, foreign, future,
skipped, reserved-but-never-submitted, and otherwise unsubmitted values remain
invalid under the existing queue contract. The session MUST successfully wait
for every direct prerequisite before submitting a dependent final projection.
Selection itself MUST then successfully observe `queue.wait(producer)` before
using the logits. Positive admission alone is not readiness. An already
successful wait does not remove this requirement because successful waits are
repeatable. If the producer has a retained completion failure, this call
rethrows it, and every later wait for that OID rethrows the same failure.

The queue, logits view, logits storage owner, and the storage owner's exact
device identity MUST remain live and unchanged throughout the call. `history`
is likewise borrowed only for the call. The selector MUST NOT retain the
queue, either borrowed view, either span, or any referenced storage after
return. Selection MUST NOT mutate logits, history, KV storage, or any owner
identity. A concrete selector that needs scratch MAY write only
caller-provisioned reusable setup storage; that implementation sibling owns
the scratch size, alignment, placement, and lifetime contract.

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
  state. Session poisoning and draining of already accepted OIDs remain the
  session's responsibility.
- A deterministic injected selector may replace greedy behavior only in
  tests. If it returns an ID outside `[0,V)`, the session rejects that result
  before history, generated count, or cache mutation. Injection cannot bypass
  producing-OID readiness, ID-range validation, history ownership, or cache
  rules.

Thus a failed or out-of-range selection consumes no invalid token. Selector
scratch may contain partial private work after a failure, but logits, history,
KV contents, owner identities, committed-token count, and initialized cache
length remain unmodified by selection.

This is a planned-only contract. It adds no public selector header, source,
test target, session implementation, transfer strategy, allocation, facade,
kernel, or current-support claim, and it does not migrate the existing neural
hooks from `Unsupported`. The future selector sibling owns the concrete
greedy implementation, deterministic oracle, and any exact reusable scratch
requirement.

### 10. Backend integration and conformance obligations

The following source map is executable contract coverage. Shared scalar,
common validation, rank, queue, workspace, and memory-boundary scenarios live
in `test/backend/backend_conformance_common.hpp`,
`test/backend/backend_conformance_memory.hpp`, and
`test/backend/backend_conformance_add.hpp`; storage, transforms, tails, padding,
and copies live in `test/backend/backend_conformance_copy_storage.hpp` and the
independent `AcceleratorStorageOracle`. Backend-local targets are
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

### 11. Contract source map

Use these sources when changing or extending the contract:

- public device/tensor/queue API: `include/iom/device.hpp`,
  `include/iom/tensor.hpp`, `include/iom/iom.hpp`;
- standard capability and transfer interface:
  `src/shared/standard_tiled_copy.hpp`;
- common binary validation, operation dispatch, and independent scalar oracle:
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_add.hpp`;
- storage, transfer, copy, and physical oracle:
  `test/backend/backend_conformance_copy_storage.hpp`,
  `test/backend/backend_conformance_oracle.hpp`;
- native setup/allocation seams: `test/cuda`, `test/rocm`, and `test/sycl`
  smoke/conformance drivers;
- backend-local full suites: `test/cpu/test_cpu_conformance.cpp`,
  `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`,
  `test/sycl/test_sycl_conformance.cpp`, and
  `test/ttnn/test_ttnn_conformance.cpp`;
- coexistence integration: `test/backend/test_backend_coexistence.cpp` and
  `test/CMakeLists.txt`.

If implementation and this document differ, update implementation, conformance,
`ARCHITECTURE.md`, and this contract together. Do not weaken shared tests.

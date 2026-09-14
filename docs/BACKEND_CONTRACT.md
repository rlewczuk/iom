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
behavior, and passes its native gates.

| Product | Logical `R=1` | `R=15` | `R=16` | `R=17` | Required native mapping and present blocker |
| --- | --- | --- | --- | --- | --- |
| ordinary linear | blocked/native route present | blocked/native route present | blocked/native route present | blocked/native route present | Per `b`, multiply selected `x[b,s:s+R,I]` by transposed shared `w[O,I]` and write exactly `out[b,R,O]`. Non-tile I/O and every leading plane are valid only with tail-masked readers. High-level transpose/no-temporary behavior and full accumulation fidelity are unproved. |
| head-planar linear | blocked/native route present | blocked/native route present | blocked/native route present | blocked/native route present | The same product writes `out[b,h,R,d]` from weight row `h*D+d`, after checked `O=H*D`; no padded-column head shuffle is permitted. A custom writer or caller-owned checked rearrangement is required because IOM stores each head plane separately. |
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

Current TTNN `create_workspace(bytes)` rejects every positive size, so the
score/P route is a concrete blocker rather than permission for a hidden tensor
allocation. The first TTNN operation that actually requires positive scratch
owns minimal TTNN-private `RawWorkspace` buffer support: the TTNN leaf under
`.cswd/tasks/006-tinyllama/04-linear-projections` if its selected linear route
packs, otherwise the TTNN SDPA leaf under
`.cswd/tasks/006-tinyllama/08-causal-grouped-attention`. Creation, rebind,
subrange addressability, exact-device checks, and destruction/reset must be
implemented before queued use. Workspace and output are disjoint from all
operands and each other; new output/read and scratch/operand overlap are
rejected, while valid read/read overlap, including exact Q/K/V aliases, is
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
| linear | Native matrix hardware and lower-level destination buffers make the route feasible as specified above. Current support is blocked on implementation, proven accumulation/RNE, optional-output or direct-writer ownership, and any required positive workspace. |
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
CUDA neural implementation or a support claim. The current CUDA neural probes
remain `Unsupported` before submission. On the inventoried device the native
CUDA Toolkit route is **supported for implementation feasibility** for
ordinary and head-planar linear, QK, and PV at every required row count.
Production status is nevertheless **blocked** until the operation-owning CUDA
ports supply runtime numerical, conformance, and profiler evidence. A device
below compute capability 8.0 is **unsupported** for this BF16 WMMA route; it
does not earn a fallback pass.

##### Evidence boundary and installed capability

The following evidence was collected on 2026-09-14 through the configured
`cuda` remote-development profile in the unique
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
| reduction | Contract-compatible FP32 CUDA block/warp reduction for linear accumulation helpers, RMSNorm, and stable softmax. Reductions must exclude masked/padded cells and retain the specified wide intermediates; WMMA does not replace max, sum, or normalization. Current kernels remain blocked. |
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
.omp/skills/remote-development/scripts/remote-sync cuda forward-layout-cuda-feasibility
.omp/skills/remote-development/scripts/remote-exec cuda forward-layout-cuda-feasibility 'nvcc --version && nvidia-smi'
.omp/skills/remote-development/scripts/remote-exec cuda forward-layout-cuda-feasibility 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader'
.omp/skills/remote-development/scripts/remote-exec cuda forward-layout-cuda-feasibility "dpkg-query -W 'cuda-*'"
.omp/skills/remote-development/scripts/remote-exec cuda forward-layout-cuda-feasibility 'cmake --build build --target iom_cuda_conformance_tests'
.omp/skills/remote-development/scripts/remote-exec cuda forward-layout-cuda-feasibility "ctest --test-dir build --output-on-failure -R '^iom_cuda_conformance_tests$'"
.omp/skills/remote-development/scripts/remote-exec cuda forward-layout-cuda-feasibility "ncu --set full --target-processes all --kernel-name regex:'.*(linear|qk|pv).*' ./build/test/iom_cuda_conformance_tests --test-case='CUDA TinyLlama BF16 matrix paths cover R=1,15,16,17'"
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
.omp/skills/remote-development/scripts/remote-exec rocm \
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
| ordinary linear | 1 | `M=16`, `K=Ip`, `N=Op`; store only row 0 and `o<O` | Feasible on checked gfx1201 WMMA; current facade remains `Unsupported`. |
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

This is a bounded feasibility record for the planned seven-operation ABI, not
a declaration that those operations are implemented. The current SYCL queue
still exposes only copy and binary execution, its neural conformance probes
still return `Unsupported`, and no project conformance or profiler run is
claimed here. A future port MUST preserve the signatures, pure requirement
queries, validation precedence, owner rules, and producer-wait schedule above;
SYCL types remain backend-private.

The evidence layers are deliberately separate:

| Evidence layer | Observation on 2026-09-14 | What it establishes |
| --- | --- | --- |
| Repository declaration | `CMakeLists.txt` selects `icpx`/`dpcpp`, `sycl/sycl.hpp`, and `libsycl` with `-fsycl`, but has no compiler-version or matrix-extension check. Standard BF16 storage uses 16x16 row-major physical tiles in device USM from the exact context; the data arena and every suballocation are 32-byte aligned. | Storage and build wiring only; neither is native matrix evidence. |
| Installed compiler and headers | `icpx --version` reported Intel oneAPI DPC++/C++ Compiler `2026.1.0 (2026.1.0.20260617)`, package `intel-oneapi-compiler-dpcpp-cpp-2026.1 2026.1.0-235`, and `SYCL_EXT_ONEAPI_MATRIX=1`. The 2026.1 installation contains `sycl/ext/oneapi/matrix/{query-types.hpp,matrix.hpp,matrix-unified.hpp,matrix-intel.hpp}` dated 2026-06-17; the API remains in `sycl::ext::oneapi::experimental`. | The experimental API and runtime query compile in the installed SDK. It is not a stability promise. |
| Installed runtime and device | Required `sycl-ls` enumeration reported two Level Zero V2 devices, each `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, driver `1.15.38646+7`, PCI device `8086:e211`, architecture `intel_gpu_bmg_g21`, and subgroup sizes `16,32`. PCI inventory reported the in-kernel `xe` driver on Linux `7.0.0-31-generic`. The selected device reports `aspect::ext_intel_matrix=true` and maximum work-group size 1024. | An eligible Level Zero XMX device and subgroup 16 are installed. The eventual queue MUST bind one exact enumerated device, not an OpenCL or other-device fallback. |
| Documented native capability | The installed `matrix_combinations` query returned 53 combinations per B60. For `A=BF16,B=BF16,C=FP32,D=FP32`, it returned continuous `M<=8,N=16,K=16`, exact `16x16x16`, and exact `1x64x16`, `32x64x16`, `1x64x32`, and `32x64x32`. It also returned BF16-output variants, but this contract does not rely on their conversion rounding. | `M=1`, `M=16`, and a `16+1` decomposition are legal with BF16 inputs and FP32 accumulation; N/K tails require physical padding or tile decomposition. |
| Bounded sample | A removed standalone sample allocated device USM, required subgroup 16, invoked `joint_matrix_load`, `joint_matrix_mad`, and `joint_matrix_store`, and checked all-one BF16 products. Separate Level Zero executions printed `BF16xBF16->FP32 1x16x16 PASS` and `16x16x16 PASS`. | Representative native XMX execution exists for the row shapes used below. It does not implement IOM linear, QK, or PV and is not conformance or profiler evidence. |
| Production evidence | Not run: IOM linear, QK, PV, masking, softmax, RNE packing, head merge, all shape cases, conformance, tuning, and profiling. | The SYCL port remains blocked/unimplemented. Inventory or the sample cannot close a native-operation gate. |

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
entry below is *native-shape feasible on the enumerated B60, production port
blocked*: the device query supports the decomposition and the two primitive M
shapes executed, but the complete operation has not. The blocked result must
remain `Unsupported` until that product's real SYCL implementation and native
tests land.

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
| Embedding gather | A tile-aware `parallel_for` can copy selected BF16 payload bits on device and preserve independent planes. Integral indices and device-discovered OOB values need checked device-side handling and an accepted retained error; a host scan/roundtrip is forbidden. No SYCL gather kernel exists today. |
| Linear | Direct backend-private `joint_matrix` is feasible as tabulated. FP32 accumulator storage, explicit RNE pack, row/window predicates, head scatter, alias checks, and all non-tile/plane cases are unimplemented. |
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

A future SYCL linear-projection leaf is the first consumer and should share the
same minimal backend-private matrix facility with the later SYCL SDPA leaf.
That consumer must make enabled-SYCL configuration fail clearly if the pinned
compiler/header contract is absent, and runtime queries must return
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

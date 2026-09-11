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

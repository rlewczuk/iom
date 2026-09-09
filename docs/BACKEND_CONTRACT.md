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
4. `supported_data_types()` MUST return immutable, nonempty storage and have
   no side effects. It is the exact set accepted with
   `QuantizationFormat::NONE`, not an aspirational list. A standard-layout
   backend MUST return the shared 23-entry sequence: `BOOL`, `I2`, `U2`, `I4`,
   `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`,
   `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F8_E8M0`, `F16`, `BF16`,
   `F32`, `F64`.
5. A native backend MAY expose a smaller set, but every advertised type MUST
   pass every shared scenario. It MUST reject a type not in that set before
   allocating native storage.
6. `create_tensor` MUST call/observe `TensorSpec::validate()`. Today that
   means rank at least two, nonzero dimensions, checked size arithmetic, a
   declared leaf type, and `QuantizationFormat::NONE`; every other declared
   quantization format is currently invalid. It MUST reject capability,
   allocation, context/device, and overflow errors before beginning work.
7. `create_ops` MUST return an independent queue associated with that exact
   device. A device may create multiple queues. A queue created by one device
   MUST reject views from another device instance, even when both instances
   target the same hardware ordinal.
8. A backend factory MUST leave no partially usable device/context behind when
   initialization fails. An enabled backend with unavailable SDK/hardware is a
   configuration/runtime failure, never an implicit fallback to CPU.

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

### 6. Copy contract

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
   Waiting on the last token must make all earlier queued writes visible.
### 7. Compute capability contract

1. `copy` is the only operation every current backend is required to
   implement. The compute hooks `add`, `mul`, `silu`, `linear`, `rmsnorm`, and
   `sdpa` remain opt-in; current hooks other than the later ADD implementation
   report `Unsupported`. A backend MUST NOT claim a capability until it
   supplies its real semantic, validation, asynchronous, and numerical
   contract.
2. An unsupported compute request MUST return the common negative
   `Unsupported` OID result before submission, sequence consumption, effects,
   or output mutation. The same rule applies to transformed operands and
   outputs. It is not a synchronous exception crossing the OID facade.
3. A future implementation of a compute hook MUST preserve caller-provided
   operand/output storage, exact device identity, queue ordering, wait-token
   behavior, and the no-hidden-allocation rule. It requires behavioral tests
   before it becomes a reported capability.

### 8. Backend integration and conformance obligations

The following suite files are the executable contract. A new backend driver
must consume the shared harness rather than duplicate a weaker subset.

| Contract area | Required harness/scenario | What it proves |
| --- | --- | --- |
| Logical bytes and API signatures | `test/backend/backend_conformance_common.hpp` | Independent LSB-first encoding, sentinel readback, token decoding, and all compute view signatures. |
| Storage and transformed transfers | `run_storage_and_transfer_conformance` | Every advertised type; tile boundaries/padding; rank 2 through 6; full, sliced, selected, permuted, nested, stepped, and reshaped views. |
| Physical layout | `run_storage_oracle_conformance` | Independent canonical 16x16 layout, padded bytes, and untouched owner planes; a perturbed map must fail. |
| Asynchronous copy | `run_async_copy_conformance` | Queue order, tokens, idempotent wait, full/offset/stepped/permuted/nested/no-op copies, and bit-for-bit CPU-reference parity. |
| Copy validation | `run_copy_error_conformance` | Invalid metadata/device rejection before writes/sequence consumption and a waitable identical-window no-op. |
| Transfer failures | `run_transfer_error_conformance` | Exact-span and canonical-BOOL validation with stable view metadata and handle. |
| Tokens and lifetime | `run_lifetime_conformance` | Stable owner/view/handle addresses, in-order completion, repeatable wait/failure, foreign tokens, and base queue-ID release behavior. |
| Compute gaps | `run_compute_capability_conformance` | Unsupported methods fail before output mutation or submission. |
| Aggregate gate | `run_backend_conformance` | Executes the storage, copy, error, lifetime, and capability scenarios in dependency order. |
| Multi-backend process | `test/backend/test_backend_coexistence.cpp` | Enabled backend headers/libraries link together; multiple devices and queues execute without a registry or cross-backend interference. |

CMake registration uses `add_iom_backend_tests` to create both
`iom_<backend>_smoke_tests` and `iom_<backend>_conformance_tests`. Link the
backend library, `libiom`, required vendor runtime libraries, include paths,
and compile/link options into the appropriate targets. Add the backend to the
combined coexistence target with a distinct `IOM_COEXIST_<BACKEND>` macro and a
participant factory that creates two queues plus matching tensors.

The conformance driver supplies backend-specific mechanics only: a valid
allocator/context setup, CPU reference, foreign-device construction (or the
nearest independent device that still tests identity rejection when a runtime
forbids two live contexts), hardware gating, and a native storage oracle. It
MUST use the candidate's advertised capability span, and it MUST run enabled
hardware rather than skip it. The shared scenarios define the behavior.

### 9. Contract source map

Use these sources when changing or extending the contract:

- public device/tensor/queue API: `include/iom/device.hpp`,
  `include/iom/tensor.hpp`, `include/iom/iom.hpp`;
- standard capability and transfer interface:
  `src/shared/standard_tiled_copy.hpp`;
- common conformance definitions and independent encoding:
  `test/backend/backend_conformance_common.hpp`;
- storage, transfer, copy, and error scenarios:
  `test/backend/backend_conformance_copy_storage.hpp`;
- independent physical oracle: `test/backend/backend_conformance_oracle.hpp`;
- lifetime/capability/full-suite scenarios:
  `test/backend/backend_conformance_other.hpp`;
- backend target registration and coexistence:
  `test/CMakeLists.txt` and `test/backend/test_backend_coexistence.cpp`.

If implementation and this document differ, update the implementation,
conformance suite, `ARCHITECTURE.md`, and this contract together. Do not solve a
conformance failure by weakening the shared test or special-casing an input.

# Expand TTNN storage to all required numeric leaves

**Order:** 15
**Priority:** P1 — required TTNN storage and transfer capability for the ADD matrix; it is independently runnable and must complete before TTNN ADD execution is verified.
**Blocked by:** `09-document-oid-contract`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

TTNN storage is a real capability for `BOOL` and all 21 required numeric `QuantizationFormat::NONE` leaves: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. Every leaf can be created, transferred, and copied through TTNN, including the 13 non-native and/or sub-byte leaves through internal backing, emulation, and conversion. The implementation retains TTNN-owned stable per-leading-plane native storage and handles while exposing the existing logical standard-byte view. This task has no ADD arithmetic.

## Scope

- Expand the TTNN supported-type table and `Device::supported_data_types()` to exactly `BOOL` plus the 21 numeric leaves above. `F8_E8M0` storage is not required; non-`NONE` quantization remains outside this task.
- Make `create_tensor` construct a valid `TensorView` for every required leaf instead of rejecting a leaf solely because tt-metal lacks a native dtype. Native leaves may keep their current carrier mapping; `I2`, `U2`, `I4`, `U4`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, and `F64` require an actual internal carrier/emulation and conversion path, not an advertised-only entry or an unconditional rejection.
- Preserve TTNN ownership and the current one-native-tensor-per-leading-plane organization. Native planes use the TTNN 32x32 tiled/padded geometry, and their native storage/handles remain stable for the lifetime of the tensor and across host transfers and copies. Do not relocate or replace caller-visible tensor owners or handles.
- Extend logical host upload/download and TTNN copy to convert between each leaf's standard tiled bytes and its internal native/emulated representation. Logical transfers must be exact and must never expose native tile ordering, carrier bytes, or native-only padding.
- Preserve transformed leading-view behavior: independently apply each view's plane offset and leading-plane strides for uploads, downloads, and copies, without assuming that a view addresses contiguous planes or that all views share one offset.
- Preserve serialized TTNN staging and synchronization. Host staging and native operations remain protected by the device API mutex; a staging buffer or lease must not be reused while a TTNN operation may still read or write it.
- Preserve failure and lifetime cleanup for native and emulated paths. A failed transfer/copy must drain when possible, discard poisoned staging, retire staging when completion is unknown, and leave tensor owners, stable handles, and subsequent operations usable. In-flight native planes must remain protected through tensor destruction/quarantine.

## Implementation references

- **Modify:** `include/iom/ttnn/device.hpp` — `ttnn_supported_data_types()` declaration and its contract comment; keep this as the single public TTNN storage-type table surface.
- **Modify:** `src/ttnn/device.cpp:94-421,851-861` — supported-to-native mapping, `TtnnDevice::supported_data_types()`, `TtnnDevice::create_tensor()`, `TtnnTensor` construction/backing, stable `storage_handle()`, and per-plane cleanup. Add internal carrier/emulation state only as needed to make all required leaves real tensors while retaining one stable TTNN-owned plane allocation per leading plane.
- **Modify:** `src/ttnn/copy.cpp:84-243,384-520` — element-width/conversion dispatch, logical plane upload/download, transformed-plane mapping, native copy submission, and staging/failure cleanup. Reuse the existing serialized retained staging and drain/discard/retire rules rather than exposing a new public fallback control.
- **Modify:** `src/ttnn/staging.hpp` — retained upload/download slot bookkeeping if additional internal carrier formats or conversion buffers require it; preserve lease ownership, no-reuse-before-drain, and device-lifetime retirement semantics.
- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — expand the supported-type assertions and storage/copy conformance. Reuse `TtnnStorageOracle`, `TtnnPlaneLayout`, the independent padded physical readback, existing transformed-view cases, and existing transfer/lifetime/failure seams; extend the oracle where carrier conversion makes native bytes differ from standard bytes.
- **Read:** `test/CMakeLists.txt:65-158,226-235` — the existing `iom_ttnn_smoke_tests` and `iom_ttnn_conformance_tests` targets and their TTNN hardware-gated setup.

## Requirements

- The reported supported span and device query must contain `BOOL` and each of the 21 numeric leaves exactly once. `F8_E8M0` need not be stored, and no public emulation knob, capability query, generic capability framework, or backend-specific fallback API may be added.
- For every required leaf, with `QuantizationFormat::NONE`, `create_tensor` must allocate usable TTNN-owned backing and return a `TensorView` whose logical dimensions, element count, standard storage sizing, owner identity, and native handle are coherent. Internal emulation may use a native carrier, host/device temporary, widening, unpacking, repacking, or conversion, but it must support creation, upload, download, and copy rather than merely listing the leaf.
- Keep one stable native TTNN plane/handle for each leading-plane allocation. Do not substitute a standard caller allocator, expose carrier dtype or native padding through `TensorView`, or relocate/replace the plane handles during conversion or transfer.
- `copy_from_host` followed by `copy_to_host` must reproduce the exact logical standard bytes for every required numeric leaf and preserved `BOOL` storage, including packed sub-byte values. Standard logical bytes are the contract; native tile-major ordering and carrier representation are implementation details.
- Native physical readback must retain the existing 32x32 tile geometry and zero native-only padding policy. The independent storage oracle must detect a physical tile-slot permutation or nonzero native-only padding while separately confirming that the logical projection remains correct. For emulated leaves, compare the decoded logical model and carrier-aware physical geometry without treating carrier bytes as the public leaf encoding.
- Copies must preserve source bytes and write the expected destination bytes for every required leaf, including copies through nested transformed leading views. Plane offsets and strides must be applied independently to source and destination, untouched owner planes must remain untouched, and no padding cell may be read as logical data or written as a logical element.
- Host transfer and copy paths must obey serialized staging and queue ordering. Retained staging leases, typed/native conversion buffers, and any temporary planes must remain alive until the operation has drained; completed buffers may be reused only after completion is proven.
- Injected submission, staging-allocation, drain/finish, and cleanup failures must follow existing TTNN behavior: pre-completion failures clean up or quarantine safely, undrainable work retires referenced storage, and a later operation can complete without use-after-free, double release, stale poisoned bytes, or handle replacement. Tensor and device destruction must clean both native and internal emulation resources.
- Do not implement ADD arithmetic, `DeviceOps::add`, broadcasting, ADD validation, ADD support signaling, or any other operation in this task.

## Non-goals

- TTNN `F8_E8M0` storage, non-`NONE` quantization, or a public mixed-type conversion/promotion API.
- Any ADD arithmetic or ADD-specific queue behavior; task 16 owns TTNN ADD execution and task 17 owns cross-backend coexistence.
- Replacing TTNN-owned storage with caller-allocator storage, relocating caller-visible owners or handles, or exposing public broadcast views/zero strides.
- A generic capability system, public `add_support`/emulation selector, options object, signature knob, or generic fallback subsystem.
- Changes to CPU, CUDA/ROCm, or SYCL storage and copy implementations, OID migration, final ADD documentation, or unrelated TTNN operations.

## Acceptance criteria

- [ ] On the configured TTNN host, the supported-type table and `Device::supported_data_types()` contain exactly `BOOL` plus all 21 numeric leaves, and each required leaf creates a usable tensor; `F8_E8M0` is not required.
- [ ] For every one of the 21 numeric leaves and preserved `BOOL`, conformance creates a tensor, performs an exact raw logical host round-trip, and proves that the implementation is real for non-native/sub-byte leaves rather than an advertise-only or rejection path.
- [ ] For every required leaf and `BOOL`, the independent physical/padding oracle checks the TTNN 32x32 per-plane layout, native-only zero padding, logical standard-byte projection, untouched planes, and stable per-plane native handles; it catches a deliberate physical permutation or padding mutation.
- [ ] For every required leaf and `BOOL`, transformed leading-view copies apply independent offsets/strides, preserve source and unrelated planes, and produce exact logical destination bytes without reading or exposing padding.
- [ ] Focused fault cases cover host staging allocation/submission, partial-plane submission, drain/finish, and cleanup/quarantine for the expanded carrier paths. After each applicable failure, staging is discarded or retired safely, owners and handles remain valid, and a subsequent transfer/copy succeeds; destruction with in-flight work does not leak or double-release resources.
- [ ] The focused TTNN storage/conformance coverage contains no ADD arithmetic or ADD support assertion; it verifies storage, logical transfers, copies, physical layout, lifetime, and failure cleanup only.

## Verification

- Proposed focused TTNN hardware smoke gate (not run per assignment): `cmake --build <configured-build-dir> --target iom_ttnn_smoke_tests && <configured-build-dir>/test/iom_ttnn_smoke_tests`.
- Proposed focused TTNN conformance gate on the configured TTNN host (not run per assignment): `cmake --build <configured-build-dir> --target iom_ttnn_conformance_tests && <configured-build-dir>/test/iom_ttnn_conformance_tests`.
- No gates are run for this specification-writing task.

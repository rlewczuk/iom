# Implement TTNN ADD

**Order:** 16
**Priority:** P1 — required TTNN ADD execution after common validation and expanded TTNN storage are available.
**Blocked by:** `12-implement-cpu-add`, `15-expand-ttnn-storage`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

TTNN implements the complete backend-specific `DeviceOps::add(lhs, rhs, out)` path behind the migrated common non-throwing OID facade. Every required numeric `QuantizationFormat::NONE` leaf executes through a native TTNN operation or a serialized internal staging/emulation path selected before acceptance, including broadcasts, transformed views, tails, aliases, and asynchronous ownership. TTNN returns `Unsupported` for ADD on `BOOL` and `F8_E8M0` while retaining `BOOL` storage support, and its hardware conformance driver proves the full TTNN behavior rather than deferring it to cross-backend integration.

## Scope

- Extend the TTNN device and queue implementation to execute ADD for exactly these 21 numeric leaves: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`, all with `QuantizationFormat::NONE`.
- Ensure TTNN ADD does not reject a required leaf merely because tt-metal lacks a matching native dtype. Select the native or emulated path before any effects or positive-token acceptance. Internal conversion, widening, host/device staging, broadcast materialization, and workspace are permitted; no public path selector or fallback API is added.
- Integrate the TTNN queue task with the common task-10 validation, view-metadata snapshots, three-owner lifetime registration, and exact-alias deduplication. The backend must receive stable snapshots rather than retaining caller `TensorView` objects for deferred execution.
- Map logical broadcast coordinates independently for each input and output after right-aligning ranks and prepending conceptual singleton axes. Cover equal shapes, opposite-direction singleton axes, rank promotion, `[1,1]` scalar broadcasting, singleton final tiled axes, and 32x32 native tiles with row/column tails. Never read or write padding or an unrelated leading plane.
- Apply independent plane offsets and strides for nested transformed leading views. Capture both input values before each output store so permitted exact in-place aliases are correct.
- Preserve the task-11 production ADD semantics: exact two's-complement modulo-$2^w$ integer results and the reference RNE floating encoding or a finite result within one ULP, including the required subnormal, saturation, NaN, infinity, and signed-zero classes. Reuse the task-11 scalar arithmetic/codec contract; do not create a competing public numeric or quantization API.
- Keep TTNN caller-created owners, native handles, and storage locations stable. ADD allocates neither operands nor the result and never replaces or relocates their storage; all internal temporary resources are TTNN-owned and cleaned up internally.
- Extend `TtnnHostStaging` for any serialized ADD staging leases and temporary byte/typed buffers needed by the emulated path. A lease must not be reused until the queue finish proves completion; failed or undrainable work must discard or retire poisoned buffers safely.
- Add TTNN-specific exhaustive and representative ADD cases to the configured-hardware conformance driver, reusing task-12 shared cases and task-11's independent oracle. This task owns the complete TTNN matrix; task 17 must not be used to defer TTNN leaf, arithmetic, alias, lifetime, or fault coverage.

## Implementation references

- **Modify:** `src/ttnn/device.cpp` — `kSupportedToNative`, `TtnnDevice`, `TtnnTensor`, `TtnnQueue`, and the TTNN task/fence/finish paths; expand TTNN ADD dispatch, path selection, ownership registration, asynchronous completion, and error retention without changing the public signature.
- **Modify:** `src/ttnn/copy.cpp` and, if needed for an internal declaration, `src/ttnn/copy.hpp` — `ttnn_detail` plane mapping and transfer helpers; add the concrete internal plane-wise ADD operation (for example `ttnn_detail::add_planes`) and logical/native 32x32 tile mapping used by native and serialized emulation paths. Keep all TTNN types private to the backend target.
- **Modify:** `src/ttnn/staging.hpp` — `TtnnHostStaging`, `UploadLease`, and `DownloadLease`; provide serialized ADD staging/workspace ownership, completion-safe reuse, and discard/retire cleanup for failures. Preserve existing host-transfer staging behavior and `BOOL` storage.
- **Planned internal helper:** `ttnn_detail::add_planes` in `src/ttnn/copy.cpp` (declared in `src/ttnn/copy.hpp`, or kept private to the implementation if no declaration is needed); it must consume immutable view snapshots and caller-owned plane handles, capture both operands before stores, report whether native work was submitted, and never expose a backend-specific public operation or capability query.
- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — the TTNN hardware driver and its independent native storage oracle setup; add focused TTNN ADD cases using the candidate TTNN device and CPU reference, while retaining the independent storage/transfer observation of 32x32 tile padding. Use the existing `TtnnDevices`, `TtnnStorageOracle`, `require_hardware()`, and doctest conventions.
- **Read:** `test/backend/backend_conformance_common.hpp`, `test/backend/backend_conformance_copy_storage.hpp`, and the task-12 shared ADD cases — reuse the backend-neutral device setup, logical byte observation, transformed-view/broadcast matrices, token helpers, and caller-allocator traffic checks rather than duplicating common behavior.
- **Read:** task-11's scalar ADD semantics and independent test oracle — use the exact leaf decoding/encoding and reference comparison rules; the production path must not call or mirror the test oracle in a way that makes a shared arithmetic defect self-confirming.
- **Read:** `docs/BACKEND_CONTRACT.md` and `include/iom/tensor.hpp` — preserve asynchronous ordering, repeatable waits, retained failures, view transforms, owner identity, and caller-owned storage contracts.

## Requirements

- The common `add` facade remains the sole support signal. Do not add `add_support`, a capability query, `EltwiseOpOptions`, an options parameter, a signature knob, or a public backend fallback/path-selection API. Matching `BOOL`, matching `F8_E8M0`, and matching recognized non-`NONE` quantization return negative `OidError::Unsupported`; mismatched or malformed recognized specs remain `InvalidArgument` under common validation. Required numeric leaves return a positive token when accepted.
- The TTNN supported-type/storage path must advertise, create, logically round-trip, and copy all 21 required ADD leaves before ADD execution. Preserve `BOOL` storage and do not add TTNN `F8_E8M0` storage. Native dtype absence alone is never a reason to return `Unsupported` for a required numeric ADD leaf.
- The backend must choose native versus serialized emulation before token acceptance and before mutating output. A pre-acceptance temporary, metadata, staging, conversion, workspace, or other bounded-resource failure maps through the OID boundary to `ResourceExhausted`; a pre-acceptance TTNN/runtime failure maps to `DeviceError` or the established uniform category. Do not retry a failed accepted native operation.
- Use the task-10 validation and snapshot contract: exact queue `Device` identity, recognized specs and matching leaves/quantization, valid checked shape/view arithmetic, exact broadcast output shape, and alias rules are enforced before effects. For same-owner input/output, permit only an exact in-place alias with identical `TensorSpec`, plane offset, plane strides, and logical mapping; reject every other same-owner relationship, including disjoint windows and broadcast input windows, with no output mutation and no sequence/token side effect. `lhs`/`rhs` read overlap is allowed.
- Register all three caller storages through completion and deduplicate exact aliases. Keep owners, handles, and captured metadata alive through wait; derived temporary views may die immediately after submission. Do not retain caller view references in a deferred worker.
- For each logical output coordinate, map each operand independently with right-aligned broadcast rules, then map to its owner plane and native 32x32 tile slot. Final-axis singleton and tail coordinates use logical extents, not padding. Preserve untouched output padding and all untouched planes.
- Capture both input values before every output store. Exact `lhs`, exact `rhs`, and all-three exact aliases must produce the same observable result as the independent oracle; source values remain unchanged except for the permitted exact in-place output window.
- Integer arithmetic is exact modulo $2^w$ for each required width, with signed two's-complement interpretation and no saturation or undefined overflow. Floating arithmetic uses task-11's named-format decode, exact-real sum, one RNE encoding, gradual underflow/no FTZ or DAZ, required special-value classes and canonical reference NaNs, and the allowed finite one-ULP envelope.
- Maintain in-order queue behavior, positive token encoding, queue serialization, visibility, repeat waits, and retained asynchronous failures. A deterministic post-acceptance TTNN failure must be retained and thrown again by every repeated `wait(token)`; it must not be converted to a synchronous result or retried. A synchronous rejection or resource failure must not consume a sequence or leave pending native work.
- Make staging allocations and cleanup deterministic at the test seam. Prove successful completion returns reusable buffers, a failed drained transfer discards poisoned storage, and a failure whose drain cannot complete retires storage until safe device teardown. Internal resources must not outlive or race their native readers.
- Exercise every required leaf individually in the TTNN driver. Exhaustively check all ordered raw-encoding pairs for `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `I2`, `U2`, `I4`, and `U4`; use representative boundary, wide-value, subnormal, overflow, NaN, infinity, zero-sign, and cancellation vectors for the remaining required leaves. Compare against task-11's independent scalar oracle and require exact integer bits or the permitted floating result.
- The TTNN driver must cover equal-shape and compact broadcast cases, opposite singleton directions, rank 2/3/6 promotion, `[1,1]` scalar behavior, final-axis tails, and one nested transformed leading view. Include incompatible and wrong-output-shape rejection, forbidden alias windows, read/read overlap, source preservation, untouched padding/planes, stable owner/handle checks, and no caller allocator traffic or storage relocation during ADD.
- Add deterministic TTNN fault seams for applicable pre-acceptance native/staging/resource failures and post-acceptance runtime failure. Verify cleanup, no effects on rejection, sequence/token behavior, and repeated retained failure. Include three-owner lifetime, temporary derived views, ordering, repeated waits, and exact-alias deduplication in the focused TTNN suite.
- Keep this task's acceptance self-contained on configured TTNN hardware. It must establish TTNN completeness for all required leaves and matrices; task 17 covers only genuinely cross-backend integration and cannot substitute for these checks.

## Non-goals

- Do not modify the public `DeviceOps` signature, common OID facade, public path selection, generic capability framework, or generic fallback subsystem.
- Do not implement ADD for `BOOL` or `F8_E8M0`; only preserve `BOOL` storage and return `Unsupported` for those ADD leaves. Do not add F8_E8M0 storage.
- Do not implement public mixed-type promotion, caller-visible casts, public broadcast views, public zero strides, or non-`NONE` quantization arithmetic.
- Do not add a CPU `Device`, CPU tensor, CPU queue, or a generic CPU fallback. Backend-internal serialized staging/emulation is permitted only as TTNN implementation policy.
- Do not change or add support for `sub`, `mul`, `cmp`, or other compute operations.
- Do not implement coexistence coverage, cross-backend integration owned by task 17, or ADD documentation owned by task 18.
- Do not redesign task-10 validation, task-11 scalar semantics/oracle, task-12 shared cases, or task-15 TTNN storage expansion; consume their completed contracts and extend only TTNN ADD behavior and its driver.

## Acceptance criteria

- [ ] On configured TTNN hardware, `supported_data_types()`, tensor creation, logical host transfer, copy, and `add` accept every one of the 21 required numeric `NONE` leaves and return a positive token for accepted ADD. No required leaf returns `Unsupported` because of missing native TTNN dtype.
- [ ] TTNN `add` on matching `BOOL` or `F8_E8M0` returns `OidError::Unsupported` with no effect or token acceptance, while `BOOL` tensors still create and round-trip; TTNN does not claim F8_E8M0 storage.
- [ ] TTNN conformance exhaustively validates all ordered low-width integer and F4/F6/FP8 encoding pairs and validates representative wide/special vectors against the independent task-11 oracle, with exact modulo integer results and the specified floating one-ULP/special-value contract.
- [ ] The focused driver passes the complete TTNN broadcast matrix (including `[1,1]`, rank promotion through rank 6, final 32x32 tiled tails, and nested transformed leading views), rejects incompatible/wrong-output shapes before effects, and never observes padding or untouched-plane mutation.
- [ ] Exact `lhs`, exact `rhs`, and all-three exact aliases are correct; forbidden same-owner windows are rejected before submission; read/read overlap is supported; source bytes, caller owners, native handles, and storage locations remain stable.
- [ ] Three-owner registration, derived-view lifetime, in-order execution, positive tokens, repeat waits, and retained post-acceptance failures are demonstrated. Repeated waits throw the same retained failure, and accepted native failures are never retried.
- [ ] Deterministic TTNN pre-acceptance resource/runtime faults map to the required OID category without effects or sequence side effects, and staging/native resources are released, discarded, or retired safely according to completion proof.
- [ ] The implementation contains no public fallback/capability/path knob and no common-code vendor dependency; all TTNN-specific native/emulated selection and staging remain internal.

## Verification

- `ctest --test-dir <configured-build> -R '^iom_ttnn_conformance_tests$'` — proposed focused hardware conformance gate; **not run for this specification-writing task**.
- The task-12 shared ADD conformance cases, invoked through `test/ttnn/test_ttnn_conformance.cpp`, must be run on the configured TTNN device after implementation; no project-wide gates are prescribed here.

# Implement SYCL ADD

**Order:** 14
**Priority:** P1 — required SYCL production ADD behavior after the CPU contract and before cross-backend integration.
**Blocked by:** `12-implement-cpu-add`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The SYCL backend implements the production `DeviceOps::add(lhs, rhs, out) -> oid` path for every required numeric `NONE` leaf. It executes accepted work on the owned in-order SYCL queue using native arithmetic where available and an internal emulated path otherwise, while preserving the common scalar/broadcast policy, caller-owned storage, queue token/lifetime rules, and retained asynchronous failures. SYCL conformance proves the complete ADD behavior rather than deferring any leaf or scenario to a later integration task.

## Scope

- Extend the SYCL queue behind the common non-throwing `add` facade. `add` is the only support signal; do not add a public capability query, `add_support`, options object, or fallback control.
- Implement the 21 required numeric `QuantizationFormat::NONE` leaves individually: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. Matching `BOOL`, matching `F8_E8M0`, and matching recognized non-`NONE` quantization return `Unsupported` through the common facade; mismatched recognized specs and malformed/unknown requests remain `InvalidArgument`.
- Snapshot all needed view metadata and owner handles before returning from `add`; never retain caller `TensorView` pointers. Capture shape, leaf/quantization metadata, plane offset/strides, logical extents, and owner addresses needed by the queued operation.
- Map right-aligned multidirectional broadcasts, rank promotion, `[1,1]` scalar broadcasts, transformed leading views, and final-axis tails in logical coordinates before selecting standard 16x16 tiled slots. Singleton coordinates map to zero and padding is never read or written.
- Track all three input/output owners until completion, deduplicating exact owner aliases. Preserve exact in-place aliasing only when the common validation has established identical spec, plane offset, plane strides, and logical mapping; reject all other input/output same-owner relationships before effects.
- Reuse the existing SYCL in-order queue, event/fence state, staging and metadata pools, and outstanding-work registry. Do not replace or relocate caller allocations or native handles. Preserve ordering, repeated waits, temporary derived-view lifetime, allocator ownership, and cleanup/quarantine behavior.
- Add deterministic SYCL ADD fault seams for pre-acceptance resource and runtime failures and for a post-acceptance failure. Pre-acceptance failures must produce the mapped negative OID without output mutation or token/sequence acceptance; post-acceptance failures must remain attached to the positive token and be rethrown identically by every `wait`.
- Extend `test/sycl/test_sycl_conformance.cpp` with the complete SYCL ADD oracle and shared behavior matrix, including exhaustive low-width checks, representative wide/special checks, broadcasts, aliases, lifetimes, errors, ordering, repeat waits, and retained failures.

## Implementation references

- **Modify:** `src/sycl/copy.cpp` — `SyclQueue`, queued task snapshots, fence completion, and `make_queue`; integrate ADD submission with the existing in-order `sycl::queue`, metadata/staging pools, worker, owner registry, and retained-failure path rather than retaining view references.
- **Modify:** `src/sycl/copy.hpp` — SYCL queue/fault seam declarations and the backend-private ADD submission boundary; keep vendor types behind the SYCL implementation boundary.
- **Add (planned):** `src/sycl/add.cpp` — SYCL ADD kernel/emulation helpers, logical broadcast/transformed-view-to-tile mapping, encoded-leaf arithmetic dispatch, and metadata preparation. Register this source in the existing `iom_sycl` target; it is an internal file, not a public API.
- **Modify:** `CMakeLists.txt` — the `iom_sycl` source list; register planned `src/sycl/add.cpp` so the helper is compiled only when SYCL is enabled.
- **Read/reuse:** `src/shared/standard_tiled_copy.inl` and `src/shared/standard_tiled_copy.hpp` — canonical 16x16 word/slot mapping and checked metadata conventions; ADD must map logical coordinates before using these physical slots.
- **Read/reuse:** `include/iom/detail/outstanding_work_registry.hpp` — `RegistryState`, fence capture, owner registration/release/invalidation, and quarantine conventions. Use the common three-owner registration/deduplication facility from the completed ADD prerequisite; do not create a SYCL-local registry.
- **Read/reuse:** `src/sycl/staging_pool.hpp`, `src/sycl/staging_pool.cpp`, and the `SyclMetadataSlotPool`/`SyclFenceState` implementation in `src/sycl/copy.cpp` — bounded internal staging, metadata capacity, event waits, poisoning, and cleanup after runtime failure.
- **Read/reuse:** `test/sycl/test_sycl_conformance.cpp` — `SyclDevices`, `SyclStorageOracle`, `SubmissionFault`, allocator traffic gates, and existing queue failure/lifetime tests; extend these fixtures instead of creating a parallel driver.
- **Tests:** `test/sycl/test_sycl_conformance.cpp` — add SYCL-specific ADD cases and invoke the shared broadcast/alias/lifetime/error/async matrix with the independent scalar oracle. `test/sycl/test_sycl_smoke.cpp` remains the backend smoke surface and must continue to exercise a real accelerator.

## Requirements

- The common facade must validate device identity, recognized matching specs, output broadcast shape, offsets/strides, checked arithmetic, and alias rules before SYCL submission. SYCL must not bypass that boundary or expose a second support/fallback policy.
- For each required numeric leaf, accepted work returns a positive OID and performs the exact integer result modulo `2^w` using two's-complement signed interpretation or ordinary unsigned interpretation. A missing native oneAPI dtype is never a reason to return `Unsupported`; use internal unpack/widen/convert/repack or another contract-compliant emulation path.
- Floating leaves decode their named scalar format, compute the exact real sum, and encode once with the specified RNE/reference behavior. Outputs must be the reference encoding or a finite value within one ULP, with correct NaN, infinity, cancellation, and accepted-zero-sign classes; no FTZ/DAZ path may violate the contract. F4/F6 use the OCP MX scalar tables, E4M3FN uses its saturation/NaN rules, E5M2 uses infinity-capable overflow, and F16/BF16/F32/F64 use their specified IEEE-style encodings.
- Capture both input values before each output store so arbitrary read/read overlap is safe. Exact in-place aliasing must not be treated as broadcast. Broadcast and transformed views must read only logical elements, preserve untouched planes/padding, and never write a caller allocation outside `out`'s logical mapping.
- Register source, rhs, and output owners before native work is accepted, deduplicating identical owner handles while retaining each distinct owner through the fence. Release successful registrations only after a completion proof; invalidate/quarantine them on runtime failure so storage cannot be reused while SYCL may still access it.
- Keep all operation metadata and derived view state owned by the queued task/fence. Destroying temporary `TensorView` objects, releasing caller references only after the registry permits it, and repeated waits must remain safe. Caller allocator counts, owner addresses, and native handles must remain stable; `add` must not allocate, replace, or relocate operand/output storage.
- Use the existing in-order queue/event chain and staging/metadata pools. A pre-acceptance metadata, staging, workspace, event, or runtime submission failure maps through the common OID error boundary (`ResourceExhausted`, `DeviceError`, `Overflow`, or `InternalError` as applicable) and leaves no accepted token or output effect. A failure after positive-token acceptance is retained and repeatably thrown by `wait`; never retry an accepted native failure.
- Make fault injection deterministic and test-only: cover at least a pre-acceptance internal-resource allocation failure, a pre-acceptance SYCL/runtime submission failure, and a post-launch/post-acceptance failure. Fault seams must not alter normal public API behavior or add a production fallback knob.
- In `test/sycl/test_sycl_conformance.cpp`, exhaustively enumerate all ordered raw-encoding pairs for `I2`, `U2`, `I4`, `U4`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, and `F8_E5M2` on SYCL and compare with the independent scalar oracle. Exercise representative boundary vectors for every wider integer and floating leaf, including NaN/infinity/overflow/underflow/cancellation cases, and verify the integer exactness and floating one-ULP envelope.
- The same SYCL conformance file must cover equal shapes, both directions of singleton-axis broadcasting, rank promotion and `[1,1]`, final tiled-axis broadcasts with tails, ranks 2/3/6, and a nested transformed leading view; incompatible/output-shape-invalid requests must be rejected before writes and token acceptance.
- The SYCL matrix must cover exact lhs/rhs/all-three alias cases, forbidden same-owner windows, read/read overlap, source preservation, untouched padding and owner planes, temporary derived views, stable handles/allocator ownership, queue ordering, repeated successful waits, pre-acceptance error timing, and repeated retained post-acceptance failures.
- Keep common code backend-neutral and do not modify unrelated compute hooks. The SYCL implementation may use internal device/host staging and implicit conversions, but the public operands/result remain caller-owned standard tiled tensors.

## Non-goals

- No public `add_support`, capability-query API, options parameter, mixed-type promotion, caller-visible cast, or public fallback control.
- No TTNN storage or ADD work, CUDA/ROCm/CPU implementation changes, coexistence integration, or cross-backend completeness work.
- No relocation/reallocation/replacement of caller tensor storage or handles, and no new generic fallback subsystem, CPU device/tensor/queue, or public broadcast/zero-stride view.
- No changes to `sub`, `mul`, `silu`, `linear`, `rmsnorm`, `sdpa`, or other compute hooks.
- No documentation or conformance changes outside the SYCL test file and the internal SYCL implementation needed for this task.

## Acceptance criteria

- [ ] SYCL `add` accepts each of the 21 required numeric `NONE` leaves, returns a positive token, and produces oracle-compliant results through native or internal emulated execution; matching `BOOL`, `F8_E8M0`, and recognized non-`NONE` requests remain negative `Unsupported` with no effects.
- [ ] SYCL conformance exhaustively passes all ordered low-width raw-encoding pairs and passes representative wide integer/floating and special-value oracle vectors, including required NaN, infinity, zero-sign, overflow, and one-ULP behavior.
- [ ] Broadcast, rank, transformed-leading-view, tiled-tail, alias, overlap, untouched-padding/plane, and invalid-request cases pass in `test/sycl/test_sycl_conformance.cpp` without caller storage relocation or handle changes.
- [ ] Derived views may be destroyed immediately after submission; three-owner registration is deduplicated and keeps every distinct owner alive until completion, and allocator/handle/lifetime assertions pass.
- [ ] In-order queue visibility, submission ordering, repeated waits, deterministic pre-acceptance resource/runtime errors, and repeatable retained post-acceptance failures pass without stray token/sequence acceptance or unsafe resource reuse.
- [ ] The SYCL smoke and conformance targets remain real-device tests and the ADD cases are executed in the SYCL conformance target rather than deferred to task 17.

## Verification

Use the `remote-development` workflow on the configured `sycl` host; do not build or execute SYCL locally:

- `.agents/skills/remote-development/scripts/remote-sync sycl <unique-task-id>` — synchronize the exact implementation workspace.
- `.agents/skills/remote-development/scripts/remote-exec sycl <unique-task-id> 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls'` — require the Level Zero GPU devices before testing.
- `.agents/skills/remote-development/scripts/remote-exec sycl <unique-task-id> 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; cmake --build <build-dir> --target iom_sycl_smoke_tests iom_sycl_conformance_tests && ctest --test-dir <build-dir> --output-on-failure -R "^(iom_sycl_smoke_tests|iom_sycl_conformance_tests)$"'` — build and run the focused real-device targets.

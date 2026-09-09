# Implement shared CUDA/ROCm ADD

**Order:** 13
**Priority:** P1 — required production ADD execution for the shared CUDA and ROCm queue path after the CPU contract is established.
**Blocked by:** `12-implement-cpu-add`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

CUDA and ROCm execute the complete public ADD contract through one policy-templated shared GPU queue path. Each backend accepts every required numeric `NONE` leaf, including leaves that require internal emulation, and produces contract-compliant results for broadcasting, transformed views, tails, aliases, asynchronous ordering, lifetimes, and failures without replacing caller-owned storage or exposing backend fallback controls.

## Scope

- Extend the shared `GpuQueue<Policy>` path in `src/shared/gpu_queue.hpp` with ADD submission, metadata capture, execution, completion, and three-owner outstanding-work tracking while preserving the existing `EventRing`, metadata/resource pools, worker ordering, context activation, queue identity, token sequence, repeat-wait, and retained-failure protocol.
- Add planned standard-tiled ADD metadata and kernel support under `src/shared` in `standard_tiled_add.hpp` and `standard_tiled_add.inl`. The shared implementation must contain no CUDA/ROCm vendor types, vendor-kind switch, global backend state, public fallback API, or caller-selectable option.
- Capture snapshots of all three views and their owner/handle identities at submission. Register lhs, rhs, and output through the existing outstanding-work registry, deduplicating exact aliases, and keep every owner alive through completion. Do not retain caller `TensorView` objects as mutable operation state.
- Implement logical coordinate mapping before physical tile-slot mapping: right-align ranks for broadcasting, singleton axes (including final tiled rows and columns), transformed leading offsets/strides, and final row/column tails must never read or write padding. Capture both input values before each output store.
- Implement the exact integer contract for `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and `U64`: two's-complement signed interpretation or ordinary unsigned interpretation, with the low `w` result bits of the exact sum modulo `2^w`.
- Implement the bounded floating contract for `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64` using the scalar semantics and independent oracle established by the preceding ADD work. Output must be the reference encoding or a finite encoding within one ULP, with required NaN, infinity, zero-sign, gradual-underflow, no-FTZ/DAZ, saturation, and special-value classes.
- Extend the CUDA policy and queue integration in `src/cuda/copy.hpp` and `src/cuda/copy.cu`, and the ROCm policy and queue integration in `src/rocm/copy.hpp` and `src/rocm/copy.hip`, so each backend activates its context/ordinal before allocation, metadata transfer, kernel launch, event operations, and cleanup. Use the existing stream, event, staging, transfer, and metadata pool abstractions rather than a second queue implementation.
- Make backend path selection before effects or positive-token acceptance. Internal unpack/widen/repack, staging, broadcast materialization, workspace, conversions, and native or emulated device paths are permitted, but ADD must not allocate or return operands/results and must not replace, relocate, or change caller-visible owners or handles.
- Preserve the common OID boundary: accepted work returns a positive token; synchronous validation, arithmetic/size overflow, resource, runtime, and internal failures use the already-defined negative OID mapping; accepted failures are retained and repeatably rethrown by `wait`. Never retry a native/runtime failure after a positive token has been accepted.
- Extend both backend conformance drivers, `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp`, to exercise this backend's complete behavior. Reuse task 10's common validation, view/owner snapshots, and three-owner deduplication, task 11's scalar semantics and independent oracle, and task 12's reusable shared ADD cases; these are inputs to this task, not deferred coverage. Keep deterministic pre-acceptance allocation/runtime fault seams and backend-specific native access only in the drivers.

## Implementation references

- **Modify:** `src/shared/gpu_queue.hpp` — `iom::detail::GpuQueue<Policy>`, its task state, submission, execution, and completion paths; add ADD without duplicating the CUDA and ROCm queue lifecycle.
- **Add (planned):** `src/shared/standard_tiled_add.hpp` — host-side ADD metadata layout, checked metadata sizing, snapshot serialization, and shared launch declarations following `standard_tiled_copy.inl` conventions.
- **Add (planned):** `src/shared/standard_tiled_add.inl` — standard 16x16 tiled ADD kernel body and policy-independent logical/broadcast/tail mapping; the backend translation units provide only launch and device-language macros.
- **Modify:** `src/cuda/copy.hpp`, `src/cuda/copy.cu` — CUDA `gpu_policy` launch/event/error hooks and translation-unit expansion/instantiation for the shared ADD path, while retaining `CUcontext` activation and CUDA diagnostics.
- **Modify:** `src/rocm/copy.hpp`, `src/rocm/copy.hip` — ROCm `gpu_policy` launch/event/error hooks and translation-unit expansion/instantiation for the shared ADD path, while retaining ordinal activation and HIP diagnostics.
- **Read/reuse:** `src/shared/standard_tiled_copy.inl` and `src/shared/gpu_queue.hpp` — existing metadata-slot, grid-stride, EventRing, worker, registry, cleanup, and deferred-error conventions; do not create a parallel queue or pool protocol.
- **Read/reuse:** `test/backend/backend_conformance_common.hpp`, `test/backend/backend_conformance_oracle.hpp`, `test/backend/backend_conformance_other.hpp`, and the shared ADD cases supplied by tasks 10–12 — independent physical-storage/oracle observation, owner lifetime checks, token checks, deterministic fakes, and reusable broadcast/alias/lifetime/fault scenarios.
- **Tests:** `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp` — backend allocators, native storage observers, fault injection, and the complete CUDA/ROCm ADD matrix; smoke executables remain the focused runtime entry points for configured accelerator hosts.

## Requirements

- For each backend, `add(lhs, rhs, out)` must accept all 21 required numeric `NONE` leaves individually: `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. Native SDK dtype support is not a reason to return `Unsupported`; use a contract-compliant internal emulation path when needed.
- A matching `BOOL`, matching `F8_E8M0`, or matching recognized non-`NONE` quantization request must return negative `OidError::Unsupported` with no effect or token acceptance. Mismatched recognized specs and malformed/unknown/device/shape/view/alias requests remain handled by the common ADD validation boundary as `InvalidArgument`; checked arithmetic overflow maps to `Overflow`; pre-acceptance temporary or runtime allocation failure maps deterministically to its prescribed negative OID. Other operations remain `Unsupported`.
- Do not add `add_support`, capability queries, public fallback methods, options objects, mixed-type promotion, public broadcast views, public zero strides, quantization codecs, or a CPU fallback device/queue. Backend path selection is internal to ADD.
- Implement full multidirectional broadcasting for ranks at least two, including rank promotion, `[1,1]` scalar convention, opposite-direction singleton axes, singleton final tiled axes, odd row/column tails, ranks 2, 3, and 6, and transformed leading views with independent plane offsets and strides. Require the output shape to equal the broadcast result and reject incompatible or incorrectly shaped outputs before effects.
- Enforce alias policy before submission: lhs/rhs may overlap; each input may alias output only for exact same-owner, same-spec, same-plane-offset, same-plane-strides, identical logical mapping in-place operation. Reject all other same-owner input/output relationships, including broadcast windows and non-identical or merely disjoint windows. Preserve source values and capture both inputs before each store.
- Preserve caller-created storage, owners, native handles, view metadata, untouched padding, and untouched planes. Track all three owners through completion and deduplicate exact aliases in registry entries. Derived temporary metadata may be retained, but caller view objects must not be retained as operation snapshots.
- Accepted work must follow in-order queue semantics and return a positive token. Repeated waits on a valid token must preserve visibility and repeat the same retained asynchronous failure. Negative, zero, foreign, future, skipped, or unsubmitted values are not waitable under the migrated common OID contract. A failure discovered after acceptance must be retained and rethrown by `wait`; it must not trigger a retry.
- Preserve `GpuQueue<Policy>` context/ordinal activation, queue serialization, EventRing leases, metadata/resource pool cleanup, stable queue IDs and handles, and bounded cleanup after event/launch failures. A pre-acceptance allocation/runtime fault must have no output, owner, registry, sequence, or token side effect; an accepted fault may leave unspecified partial output but must remain observable through repeat waits.
- The backend conformance cases must compare logical results and physical storage against the independent oracle, not a production mapper or arithmetic helper. Include exhaustive ordered raw-encoding pairs for every low-width `I2/U2/I4/U4/F4/F6/F8` leaf, representative wide-format integer and floating boundaries, NaN/infinity/zero classes, full broadcast/transformed/tail mappings, exact and forbidden aliases, read/read overlap, owner/handle stability, temporary-derived-view lifetime, ordering, repeat waits, deduplication, deterministic pre-acceptance allocation/runtime failures, and retained post-acceptance failures.

## Non-goals

- Do not implement CPU, SYCL, or TTNN ADD, TTNN storage expansion, cross-backend coexistence, public documentation, or the independent scalar oracle; those are separate tasks.
- Do not defer low-width exhaustive behavior, wide/special values, broadcasting, aliasing, lifetime, or fault coverage to the cross-backend integration task. This task must prove the complete CUDA and ROCm behavior in their own conformance drivers.
- Do not change the public `DeviceOps::add` signature, introduce a caller-visible fallback/native-selection knob, replace caller storage, or add generic capability/fallback infrastructure.
- Do not alter non-ADD operation behavior beyond preserving their migrated negative `Unsupported` results.

## Acceptance criteria

- [ ] On each configured CUDA and ROCm device, every one of the 21 numeric `NONE` leaves accepts ADD with a positive token and produces exact integer modulo results or floating results satisfying the independent reference/one-ULP and special-value contract; no required numeric leaf is rejected for lacking native SDK support.
- [ ] The exhaustive low-width raw-encoding matrix covers all ordered pairs for `I2`, `U2`, `I4`, `U4`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, and `F8_E5M2` on both backends, and wider integer/floating boundary vectors cover all remaining leaves.
- [ ] Equal-shape, opposite-direction singleton, rank-promoted, `[1,1]`, final-axis tail, rank-2/rank-3/rank-6, and nested transformed-view cases map logical coordinates before tile slots and leave padding/other planes untouched; incompatible shapes and incorrect output shapes are rejected before effects.
- [ ] Exact lhs, exact rhs, and all-three exact aliases succeed where permitted; every other same-owner input/output relationship is rejected before effects. Source bytes, owners, handles, and caller storage placement remain stable.
- [ ] Positive-token work is ordered and visible, repeated waits are valid and repeat retained failures, all three owners survive until completion, exact aliases are registered once, and no accepted native/runtime failure is retried.
- [ ] Deterministic pre-acceptance allocation/runtime faults return the prescribed negative OID without output, sequence, token, or registry effects; accepted faults are retained and rethrown by `wait`. Matching `BOOL`, `F8_E8M0`, and recognized non-`NONE` requests return `Unsupported`, while non-ADD operations retain `Unsupported`.
- [ ] The focused CUDA and ROCm smoke/conformance commands exercise the actual configured accelerator binaries: `ctest --test-dir <configured-cuda-build> -R '^iom_cuda_(smoke|conformance)_tests$' --output-on-failure` and `ctest --test-dir <configured-rocm-build> -R '^iom_rocm_(smoke|conformance)_tests$' --output-on-failure`.

## Verification

- No gates are run while creating this frozen mini-specification, as required by the assignment.
- After implementation on configured accelerator hosts, run the two focused `ctest` commands listed in Acceptance criteria against the CUDA and ROCm builds; do not substitute CPU-only execution for either backend.

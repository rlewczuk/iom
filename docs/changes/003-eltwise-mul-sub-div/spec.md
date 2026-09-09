# Elementwise multiplication, subtraction, and division

## Purpose

Implement backend-neutral elementwise multiplication, subtraction, and division on CPU, CUDA, ROCm, SYCL, and TTNN. The public operations are imperative, asynchronous three-view `DeviceOps` facades over caller-owned tensors:

```cpp
oid mul(const TensorView& lhs, const TensorView& rhs, TensorView& out) noexcept;
oid sub(const TensorView& lhs, const TensorView& rhs, TensorView& out) noexcept;
oid div(const TensorView& lhs, const TensorView& rhs, TensorView& out) noexcept;
```

`mul` already exists in the public interface but is unsupported by real backends; `sub` and `div` are new public methods. All three must use the same validation, view mapping, queue, lifetime, and backend execution structure as `add`. Arithmetic selection must not produce copied traversal or kernel bodies.

## Verified current state

- `DeviceOps::add(lhs, rhs, out)` is a common non-virtual `noexcept` facade with validated immutable request snapshots, right-aligned broadcasting, exact-alias policy, owner registration, positive OID acceptance, and retained asynchronous failures.
- `DeviceOps::mul(a, b, c)` exists, but its facade validates only queue-device membership and every real backend currently returns `OidError::Unsupported`. The current protected hook receives raw views rather than the validated ADD request.
- No public or protected `sub` or `div` API exists.
- CPU, CUDA, ROCm, SYCL, and TTNN implement ADD for 21 unquantized numeric leaves. `BOOL` and `F8_E8M0` storage may exist but ADD rejects them.
- CUDA and ROCm include one shared vendor-neutral standard-tiled ADD kernel and use one policy-based GPU queue. CPU, SYCL, and TTNN each have an ADD traversal around the shared scalar arithmetic helper; their storage, staging, and synchronization remain backend-specific.
- Current ADD request, owner-registration, scalar, GPU metadata, queue-task, and completion symbols are ADD-specific. Extending them independently for three operations would duplicate load-bearing validation and lifetime behavior.
- Existing documentation and conformance tests state that MUL is unsupported and contain no SUB or DIV contract or arithmetic oracle.

## Scope

### In scope

- Implement `mul` and `sub` for all 21 unquantized numeric leaves supported by ADD:
  `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`.
- Implement `div` for the nine floating leaves:
  `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`.
- Apply the existing ADD shape, broadcasting, transformed-view, aliasing, error, queue, ownership, and storage contracts to all three operations.
- Refactor ADD and the existing MUL facade into one operation-neutral internal binary-operation boundary shared by ADD, MUL, SUB, and DIV.
- Implement every enabled backend and update shared/backend conformance tests and public architecture/backend-contract documentation.

### Non-goals

- Integer DIV. Integer leaves passed to `div` are unsupported; no integer divide-by-zero or signed `MIN / -1` policy is introduced.
- `BOOL`, `F8_E8M0`, or non-`NONE` quantization arithmetic.
- Mixed-type promotion, caller-visible casts, options objects, public support/capability queries, fallback selectors, or signature knobs.
- Public broadcast views, public zero strides, rank-0, rank-1, or empty tensors.
- A generic operation registry, graph/autograd machinery, a new fallback framework, or a universal traversal shared across incompatible storage backends.
- New public tensor ownership or allocation behavior, or changes to `silu`, `linear`, `rmsnorm`, or `sdpa`.
- Unrelated changes to ADD numerical semantics or backend path policy.

## Public and validation contract

Each operation is exactly a three-view common non-virtual `noexcept` facade. The method itself is the sole support signal: accepted work returns a positive OID token; synchronous rejection returns one existing negative `OidError`. No synchronous exception crosses the facade.

Validation occurs before effects, sequence consumption, owner registration, or token acceptance and follows the current ADD order and precedence:

1. Validate recognized tensor specs, rank at least two, nonzero dimensions, exact queue `Device` identity, stable owner/native handle identity, view offsets and strides, storage bounds, and checked size/address arithmetic.
2. Require identical leaf types and quantization formats for `lhs`, `rhs`, and `out`. A recognized mismatch is `InvalidArgument` even when one operand's leaf would be unsupported by the selected operation.
3. Compute right-aligned broadcasting by conceptually prepending leading ones. Each operand axis must equal the other or be one; the output shape must exactly equal the axis-wise maximum. `[1,1]` is the scalar convention. Singleton final tiled axes map to logical coordinate zero and padding is never read.
4. Snapshot each view's logical mapping. Existing transformed leading views retain independent plane offsets and strides.
5. Permit arbitrary read/read overlap. Permit an input/output same-owner relationship only for an exact, unbroadcasted in-place alias with identical spec, plane offset, plane strides, and logical mapping. Reject every other input/output same-owner relationship before submission. Load both logical operands before storing an aliased output element.
6. Apply operation-specific dtype support only after the preceding checks. Matching `BOOL`, `F8_E8M0`, or recognized non-`NONE` quantization returns `Unsupported` for every operation. Matching integer leaves additionally return `Unsupported` for DIV. A required MUL/SUB leaf or floating DIV leaf must not return `Unsupported` merely because a backend lacks a native SDK dtype.

Malformed specs and device/shape/view/alias errors map to `InvalidArgument`; checked arithmetic maps to `Overflow`; pre-acceptance bounded-resource allocation failure maps to `ResourceExhausted`; backend/runtime failure detected before acceptance maps to `DeviceError`; otherwise unclassified failure maps to `InternalError`.

Operand order is observable: `sub(lhs, rhs, out)` computes `lhs - rhs`, and `div(lhs, rhs, out)` computes `lhs / rhs`. Tests and implementations must not use commutativity assumptions for SUB or DIV.

## Numerical contract

### Integers

MUL and SUB interpret signed leaves as two's-complement bit patterns and unsigned leaves as ordinary binary values. For a leaf width `w`:

- MUL returns the low `w` bits of the exact product, modulo `2^w`.
- SUB returns the low `w` bits of the exact difference, modulo `2^w`.

The implementation must not invoke signed-overflow undefined behavior and must not saturate. DIV does not accept integer leaves.

### Floating point

Use the same scalar format definitions and encoding policy as ADD. Decode both operands according to the destination leaf, compute the selected operation in the extended mathematical/IEEE domain, and encode once to the same leaf using round-to-nearest, ties-to-even. There is no intermediate destination-width rounding.

The operation-specific reference rules are:

- MUL: a NaN operand or zero multiplied by infinity produces NaN; otherwise signs and finite/infinite products follow IEEE multiplication.
- SUB: a NaN operand or subtraction of same-sign infinities produces NaN; otherwise compute the IEEE difference `lhs - rhs`.
- DIV: a NaN operand, zero divided by zero, or infinity divided by infinity produces NaN. A nonzero finite numerator divided by signed zero, or an infinite numerator divided by a finite denominator, produces a same-XOR-sign infinity. Signed zero divided by a nonzero finite value, and a finite value divided by signed infinity, produces a same-XOR-sign zero. Other finite quotients use exact division before the one encoding step.

Destination encoding remains format-specific:

- `F4_E2M1`, `F6_E2M3`, and `F6_E3M2` have no NaN or infinity encoding. Finite overflow and an exact infinity saturate to the same-sign maximum finite value; an exact NaN encodes as the canonical positive maximum finite value used by the shared scalar codec.
- `F8_E4M3FN` has a NaN encoding but no infinity; infinity and finite overflow saturate to same-sign maximum finite, and NaN uses the format's NaN class.
- `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64` preserve NaN and signed infinity classes. Finite overflow follows the format's RNE threshold to infinity.
- Gradual underflow is required; implementations may not rely on FTZ or DAZ when that would violate the output envelope.

For a finite nonzero reference result, a backend output may be the reference encoding or an adjacent finite encoding no more than one ULP away in that leaf's signed numerical order. A reference NaN requires a NaN where the format has NaN; payload and sign need not be canonical. A reference infinity requires the same infinity sign where representable, otherwise the saturation rule above. A reference zero requires the reference sign. These rules apply uniformly to every backend, including emulated paths.

## Shared implementation boundary

The implementation must make a clean internal cutover from ADD-specific binary request/submission machinery to an operation-neutral binary-operation path used by all four operations. Exact internal names and whether the backend execution boundary uses one virtual hook or trivial per-operation adapters are implementation choices, but the following properties are required:

- One common validator and immutable request snapshot own dtype equality, operation-specific eligibility, broadcasting, view mapping, aliasing, and error precedence for ADD, MUL, SUB, and DIV.
- One common submission/owner-registration path snapshots metadata before acceptance, registers the three distinct owners, rolls registration back on rejected submission, and carries the selected operation through asynchronous work and completion.
- Superseded ADD-only request/submission/registration paths and the old raw-view MUL validation path are removed; do not retain aliases or a parallel second binary-operation system.
- Within each backend family, one traversal/staging/completion structure handles all four operations. Arithmetic is selected once per request outside logical element loops. Avoid per-element operation switches, allocation, or metadata reconstruction.
- Operation-specialized template/functor instantiations and distinct compiled GPU entry points are permitted. Copied scalar codec, packed-word mapping, coordinate traversal, metadata serialization, or kernel bodies are not.
- CUDA and ROCm continue to share one vendor-neutral standard-tiled source and the policy-based GPU queue; no CUDA-only or HIP-only arithmetic copy is added.
- CPU, SYCL, and TTNN reuse their respective existing ADD mapping and backend-specific storage/staging boundaries. The requirement does not force one traversal abstraction across standard tiled storage and TTNN native per-plane storage.
- Common code remains backend-neutral: no runtime headers/types, backend-kind switch, or global active-backend registry.

The ADD cutover must preserve its existing accepted dtype set, arithmetic results, validation/error precedence, snapshot timing, queue ordering, owner lifetime, repeat-wait behavior, and retained failure behavior. It must not add transfers, synchronization, or caller-visible allocation to the ADD path.

## Execution, ownership, and failure behavior

MUL, SUB, and DIV are in-order asynchronous operations; CPU may complete inline. Every accepted request returns one positive token. Waiting on a valid token is repeatable, makes completed writes visible, and rethrows the same retained post-acceptance failure on every wait. A pre-acceptance error returns a negative OID, leaves output unchanged, consumes no sequence, and registers no owner.

Track all three caller tensor owners through terminal completion and deduplicate exact aliases. Retain immutable metadata snapshots rather than caller `TensorView` objects. A backend may use internal temporary buffers, workspace, staging, conversions, broadcast materialization, or native tensors, but it must not allocate operands or output, replace or relocate caller storage, or retarget owners/handles. Partial output after an accepted asynchronous failure remains unspecified. Never retry accepted failed work.

Backend setup that can fail due to metadata size or bounded resources must complete before token acceptance. No backend-local rank or dtype limitation may be deferred into a positive token followed by a wait-time failure when the common contract requires support.

## Implementation touchpoints

- `include/iom/iom.hpp` and `src/iom.cpp`: add the SUB/DIV facades; migrate ADD/MUL to the operation-neutral request, validation, hook, submission, and lifetime boundary; update public comments and default unsupported behavior.
- `include/iom/detail/outstanding_work_registry.hpp`: generalize ADD-only registration names/records without changing registry ownership, deduplication, cleanup, or quarantine behavior.
- `src/shared/scalar_add.hpp`: refactor the shared scalar codec and expose operation-specialized ADD/MUL/SUB/DIV arithmetic without duplicating format decode/encode logic.
- `src/cpu/device.cpp`: reuse one standard-tiled coordinate/packed-bit traversal and select arithmetic once per request.
- `src/shared/standard_tiled_add.inl`, `src/shared/gpu_queue.hpp`, and CUDA/ROCm policy translation units: generalize metadata, packed mapping, queue outcome/completion, and the shared kernel source while preserving backend policy separation and checked pre-acceptance metadata acquisition.
- SYCL implementation files: reuse one validated request, staging/mapping traversal, queue/fence, and cleanup path for all four operations; a new device kernel is not required by this specification.
- `src/ttnn/device.cpp` and `src/ttnn/copy.cpp`: reuse one native per-plane mapping/staging traversal and queue/lifetime path; preserve TTNN's existing storage carrier behavior and public supported-data-type span.
- Shared scalar, common request, backend conformance, backend capability, and coexistence tests; all CPU/CUDA/ROCm/SYCL/TTNN conformance drivers.
- `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and public API comments: document the three new supported operations, their differing dtype domains, arithmetic, validation, errors, ownership, and compatibility cutover from unsupported MUL.

File names may change if the clean operation-neutral cutover makes an ADD-specific name misleading. Do not create backend-specific copies merely to preserve an old file name.

## Compatibility

This is an additive source API change for SUB and DIV and a behavioral cutover for the existing MUL facade: a valid supported MUL request that formerly returned negative `Unsupported` now returns a positive token. Callers must continue to test OIDs and wait only on positive tokens. Existing ADD callers and results remain compatible. Consumers must rebuild against the changed C++ headers; no mixed-version ABI behavior is promised.

## Acceptance and automated test strategy

All caller-visible behavior above must be observable through automated tests:

- Compile-time checks pin the exact three-view `noexcept` signatures of ADD, MUL, SUB, and DIV. Common fake-queue tests cover every facade's validation order, error mapping, result-shape snapshot, operation identity, owner registration/deduplication, rollback, sequence behavior, repeated waits, and retained failures.
- One independent test oracle, separate from production scalar arithmetic and backend kernels, defines ADD/MUL/SUB/DIV raw integer and floating results. Generalize shared packing, format classification, ULP comparison, and diagnostics instead of creating three copied ADD test frameworks.
- Direct scalar tests exhaust all ordered raw operand pairs for I2/U2/I4/U4 MUL and SUB and for every applicable compact floating operation. Include reversed unequal pairs so SUB/DIV operand order is proven. Wide integer vectors cover zero, extrema, high bits, wraparound products, and underflowing differences. Wide floating vectors cover ordinary rounding boundaries, subnormals, overflow/underflow, both zero signs, infinities, NaNs, MUL zero-times-infinity, SUB same-sign infinities, DIV signed zero, zero/zero, finite/infinity, and infinity/infinity.
- Every enabled backend executes every required leaf for each supported operation through its real queue, waits, reads logical output, and compares every addressed element with the independent oracle. Integer MUL/SUB results are exact; floating finite results are within one ULP and special/zero classes follow the contract.
- Shared mapping cases cover equal shapes, opposite-direction singleton axes, rank promotion, `[1,1]`, ranks 2, 3, 6, 8, 9, 16, and 17, final tiled row/column tails, nested transformed leading inputs, a nonzero transformed output offset/stride, and untouched padding/other planes. Use representative U8 MUL/SUB and F32 DIV cases without duplicating the full dtype matrix for every shape.
- Alias cases cover exact `lhs`/`out`, exact `rhs`/`out`, and all-three aliases with noncommutative operands chosen to expose reversal. Representative partial, shifted, and broadcast input/output same-owner windows return `InvalidArgument` before effects. Read/read overlap remains valid.
- Capability/error cases verify MUL/SUB reject matching `BOOL`, `F8_E8M0`, and non-`NONE` quantization as `Unsupported`; DIV additionally rejects every matching integer leaf as `Unsupported`. Mismatched leaves/quantization, wrong device, incompatible shapes, wrong output shape, malformed views, and checked overflow preserve the specified earlier error precedence and have no output/token/sequence effect.
- Backend fault seams cover pre-acceptance resource/runtime rejection and post-acceptance execution/fence failure for each new dispatch path. Post-acceptance failures retain owners, repeat on every wait, clean staging/metadata exactly once, and leave the queue usable for later work.
- Existing ADD numerical, mapping, rank, alias, lifetime, fault, and coexistence tests remain green after the shared refactor. Add operation-neutral coexistence coverage that interleaves ADD, MUL, SUB, DIV, and COPY across every enabled backend without introducing global dispatch state.
- Each enabled backend's conformance target is the completeness gate; coexistence is integration proof, not a replacement. Accelerator gates run on their configured CUDA, ROCm, SYCL, and TTNN hosts and fail rather than skip when enabled hardware is unavailable.

The no-duplication requirement is a structural acceptance condition: source review must find one operation-neutral request/submission path, one traversal per backend family, one CUDA/ROCm kernel source body, and one shared scalar codec. A deliberate wrong-operation dispatch, swapped SUB/DIV operands, two-ULP floating perturbation, invalid special-value class, or transformed-output mapping error must cause the focused tests to fail.

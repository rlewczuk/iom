# Elementwise addition

## Purpose

Implement backend-neutral raw `DeviceOps::add(lhs, rhs, out) -> oid` on CPU, CUDA, ROCm, SYCL, and TTNN. The caller owns every input and the output tensor. ADD is imperative and follows the asynchronous queue contract. CPU is the normative reference implementation, but the arithmetic contract—not incidental CPU hardware behavior—defines correctness.

## Scope and non-goals

### In scope

- ADD.
- Full multidirectional broadcasting for valid tensor ranks.
- Existing transformed leading views.
- Exact numerical semantics.
- Per-device, per-spec ADD capability reporting.
- Every backend implementation and the corresponding conformance documentation and tests.

### Out of scope

- Graphs, autograd, tensor allocation, and tensor return.
- Mixed-type promotion and implicit casts.
- Public broadcast views and public zero strides.
- Other operations, including `sub`, `mul`, and `cmp`.
- A generic capability framework or queries for other operations.
- Rank-0, rank-1, and empty tensors.
- Every non-`NONE` quantization format. Quantization is not part of ADD and remains invalid under the existing `TensorSpec` validation contract.
- Any replacement for removed quantization support.

## Verified current state

- The ADD declaration is a three-view `DeviceOps` virtual method, and the backends currently inherit unsupported behavior. No ADD implementation, numeric codec, fallback, options object, or per-operation capability query currently exists.
- Shapes have rank at least two and nonzero dimensions, and the final two axes are tiled. `TensorView` transforms are leading-only; the relevant symbols are in `include/iom/tensor.hpp` and `src/iom.cpp`.
- Standard backends advertise 23 storage leaves in `src/shared/standard_tiled_copy.hpp`: 21 numeric leaves plus `BOOL` and `F8_E8M0`. TTNN advertises nine `NONE` storage leaves in `src/ttnn/device.cpp`: `BOOL` plus eight numeric leaves.
- Current code has bit-width and copy semantics but no ADD numeric codec. Quantization formats other than `NONE` are rejected by existing tensor-spec validation in `src/iom.cpp`.
- Queue, token, and lifetime rules are documented in `docs/BACKEND_CONTRACT.md`. The repository-wide OID migration below is proposed prerequisite behavior, not current behavior.

## Preliminary repository-wide OID migration

ADD cannot begin until this proposed migration is complete and its tests and documentation pass. The migration applies to every public `DeviceOps` operation returning `oid`: `copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`.

### Public type and exact token encoding

Create the dedicated public OID definition at `include/iom/oid.hpp`:

```text
using oid = std::int64_t;
```

Define exactly these stable negative error categories and values:

```text
enum class OidError : oid {
    InvalidArgument = -1,
    Unsupported = -2,
    Overflow = -3,
    ResourceExhausted = -4,
    DeviceError = -5,
    InternalError = -6,
};
```

All negative `oid` values are reserved for errors; these six values are the complete defined set for this change. The same header provides:

```text
[[nodiscard]] constexpr oid to_oid(OidError error) noexcept;
[[nodiscard]] constexpr bool oid_is_error(oid value) noexcept;
[[nodiscard]] constexpr bool oid_is_token(oid value) noexcept;
```

`to_oid` returns the enum's underlying negative value, `oid_is_error` is `value < 0`, and `oid_is_token` is `value > 0`. `0` is invalid and is never a token or error code.

Preserve all 255 eight-bit queue IDs by encoding each queue ID `q` in `[1, 255]` in bits 55 through 62 and using a 55-bit sequence:

```text
token = static_cast<oid>((std::uint64_t{q} << 55) | sequence)
```

The sequence is in `[1, 2^55 - 1]`; sequence zero is not submitted. The encoding produces every valid positive token through `INT64_MAX` without losing a queue ID. Sequence exhaustion is a synchronous `Overflow` result before effects or token acceptance.

### Error boundary and waiting

Every public OID-returning operation is a common non-throwing `noexcept` facade. Backend extension points sit behind this facade and cannot bypass its validation, error mapping, token encoding, or lifetime registration. Synchronous failures return one negative `OidError` code before any effect or token acceptance; no synchronous exception crosses an OID-returning public method. Accepted work always returns one positive token.

The mapping is uniform across `copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`: invalid caller input maps to `InvalidArgument`; an unavailable operation or unsupported specification maps to `Unsupported`; checked arithmetic or sequence exhaustion maps to `Overflow`; an allocation or bounded-resource failure maps to `ResourceExhausted`; a backend/runtime failure detected before acceptance maps to `DeviceError`; and an otherwise unclassified internal failure maps to `InternalError`. Negative OIDs are terminal synchronous results, not waitable tokens.

Failures that remain asynchronous after acceptance are retained and are repeatably thrown by the existing `void wait(token)` operation. `wait` immediately throws `std::invalid_argument` for a negative, zero, foreign, future, skipped, or otherwise unsubmitted value. A value that was skipped or reserved but never submitted remains invalid even if later submissions complete. Waiting a valid token preserves queue ordering, visibility, repeated-wait behavior, and retained-failure semantics. Capability queries do not return OID values and retain their specified exception behavior.

### Separately taskable migration ordering

Planning must create a separate preliminary task for every numbered migration below. Complete items 1 and 2 first; items 3 through 6 are then independent backend migrations; items 7 through 9 finish the repository-wide cutover. ADD implementation starts only after all nine pass.

1. Add `include/iom/oid.hpp`, move the public OID definition into it, define the signed error categories and helpers, and migrate the common 55-bit sequence/token encoding and validation.
2. Replace the public virtual OID operations with common non-throwing facades and protected backend extension points; migrate `submit`, sequence exhaustion, default unimplemented compute hooks, and synchronous exception-to-error mapping.
3. Migrate the CPU `copy` implementation and CPU-specific queue fakes to the new extension boundary and error results.
4. Migrate the shared CUDA/ROCm `copy` queue and both backend drivers to the new extension boundary and error results.
5. Migrate the SYCL `copy` queue and driver to the new extension boundary and error results.
6. Migrate the TTNN `copy` queue and driver to the new extension boundary and error results.
7. Migrate core tests, shared conformance fakes, token-exhaustion/error tests, and unsupported-hook expectations.
8. Migrate backend conformance and coexistence tests, including negative/zero/foreign/future/skipped/unsubmitted waits, error timing, and retained asynchronous failures. Include a focused migrated test for the current `seek_next_sequence` behavior: after it leaves a skipped value and a later submission completes, verify that the skipped value is still rejected.
9. Update `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, public headers, and any other public API documentation for the type, encoding, errors, waiting, and compatibility impact.

All tests and documentation in these tasks must be migrated before ADD work starts. The migration is a clean public compatibility cutover: callers must handle negative OID results and must not expect synchronous exceptions from OID-returning operations. It does not change capability-query exception behavior.

## Public ADD API after the prerequisite

Keep exactly three `TensorView` parameters; no `EltwiseOpOptions`, options parameter, or signature knob exists or may be added. As with every public OID-returning operation after the prerequisite, `add` is the non-virtual facade; backend-specific work is implemented behind it:

```text
oid add(const TensorView& lhs, const TensorView& rhs, TensorView& out) noexcept;
```

The common OID facade owns the non-throwing boundary. Broadcasting, arithmetic, emulation eligibility, memory and staging policy, and optimization are fixed semantics or backend-internal policy; none is caller-selectable.

Define the enum in the backend-neutral public header `include/iom/operation.hpp`:

```text
enum class OperationSupport : std::uint8_t {
    Unsupported,
    Emulated,
    Native,
};
```

Add only this operation-specific virtual query to `Device` in this change:

```text
[[nodiscard]] virtual OperationSupport add_support(const TensorSpec& lhs,
                                                   const TensorSpec& rhs,
                                                   const TensorSpec& out) const = 0;
```

`OperationSupport` is reusable vocabulary for future `DeviceOps` operations, but this change adds no generic capability framework and no query or implementation for another operation. `Native` describes the selected ADD implementation path; exactness is an ADD semantic requirement, not a promise implied by the enumerator's name. `Emulated` and `Native` are reported only when the exact ADD contract can be met. `Unsupported` means that a well-formed request has no ADD implementation.

`add_support` applies `TensorSpec::validate` to each spec, checks that each spec is storable by the `Device` being queried, and then validates matching leaf types and quantization formats, broadcast compatibility, and the exact output shape. It performs no allocation, submission, warning, or runtime mutation and can run before tensor or queue creation. A recognized non-`NONE` quantization retains the existing `TensorSpec` `std::runtime_error`; an unknown leaf or quantization value, a mismatch among individually valid specs, malformed specs, broadcast incompatibility, or an incorrect output shape throws `std::invalid_argument`. Device storage absence and well-formed excluded leaves return `Unsupported`. `add` independently validates its three views.

## Supported unquantized specs

With `QuantizationFormat::NONE`, I32 and F32 are the portable required floor on every enabled backend. For a concrete `Device`, every additional unquantized numeric storage leaf for which its runtime or SDK provides an ADD path that can write the caller-provided output, avoid temporary backend tensor or device-workspace allocation, and satisfy this specification's exact semantics must be reported and implemented as `Native`. Other valid numeric storage leaves may be `Emulated` or `Unsupported`. Storage support alone is insufficient.

The standard numeric storage leaves are `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`. This list describes possible storage leaves, not a mandatory support matrix. `BOOL` and `F8_E8M0` remain `Unsupported`. Every non-`NONE` quantization remains invalid and out of scope. A native but approximate BF16 path does not qualify as `Native`.

TTNN follows the same rule over its current eight numeric `NONE` storage leaves: `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `BF16`, and `F32`; it must at least support I32 and F32. `BOOL` remains storable but `Unsupported` for ADD.

## Broadcasting and validation

Rank less than two is invalid. `[1,1]` is the only scalar convention: it broadcasts across every output axis, and two `[1,1]` inputs produce `[1,1]`.

For ranks at least two, broadcasting right-aligns ranks and conceptually prepends ones. Two operand axes are compatible if they are equal or either is one; each result axis is their maximum. The output shape must be exactly the resulting shape. Singleton axes, including final tiled rows and columns, map their operand coordinate to zero. Resolve logical coordinates before mapping to tile or native slots, and never read padding. Do not materialize broadcast copies or expose a public zero-stride view. Final tiled axes, including tails, follow the same logical rule.

Existing transformed leading views retain independent plane offsets and strides. ADD must apply those offsets and strides independently for each input and output while preserving logical broadcast mapping.

Before submission, metadata or staging allocation, write, or token creation, `add` validates exact queue `Device` identity for all three views, identical leaf data type and quantization format, exact output broadcast shape, valid offsets and strides, checked size, offset, stride, and shape arithmetic, ADD support, and aliasing. Malformed specs and device, type, shape, view, or alias errors return `InvalidArgument`; a recognized non-`NONE` quantization or well-formed unsupported leaf returns `Unsupported`; checked arithmetic overflow returns `Overflow`; an allowed metadata or staging allocation failure returns `ResourceExhausted`; a backend failure detected before acceptance returns `DeviceError`; and an unexpected internal failure returns `InternalError`. No synchronous exception crosses the public OID-returning method. `add_support` instead follows the query-specific exception and `Unsupported` behavior above.

## Numerical semantics

Inputs and output have the same leaf type; there is no promotion. Signed integer values use two's-complement interpretation and unsigned values use ordinary binary interpretation. For every integer width, the output is the low `w` bits of the exact sum modulo `2^w`; arithmetic is neither saturating nor undefined.

Scalar floating types decode according to their named format, compute an exact real sum, and encode once to the same format using round-to-nearest, ties-to-even (RNE). Internal widening is permitted only as an implementation technique.

`F4` and `F6` follow the OCP Microscaling Formats v1.0 scalar-element tables, including subnormals, RNE, and saturation to signed maximum finite. `F8_E4M3FN` follows OCP E4M3 with saturation and canonical positive NaN `0x7f`. `F8_E5M2` uses OCP overflow mode and infinity. `F16`, `F32`, and `F64` are IEEE binary16, binary32, and binary64. `BF16` has a sign bit, 8 exponent bits, and 7 fraction bits with IEEE-style specials and RNE.

Arithmetic has gradual underflow and must not use FTZ or DAZ. Zero handling is exact: `-0 + -0 = -0`; opposite zero signs and exact nonzero cancellation produce `+0`; a nonzero value rounded to zero retains its mathematical sign. For infinity-capable formats, any NaN or opposite infinities produce canonical positive quiet NaN; otherwise an infinity operand determines the result.

Canonical NaN encodings are E4M3FN `0x7f`, E5M2 `0x7e`, F16 `0x7e00`, BF16 `0x7fc0`, F32 `0x7fc00000`, and F64 `0x7ff8000000000000`. Finite overflow follows the RNE threshold to infinity for formats capable of representing infinity. The OCP Microscaling Formats v1.0, September 2023, is normative for F4, F6, and FP8 encodings: <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>. The explicit ADD policies here take precedence wherever that standard provides options.

Every leaf reported `Native` or `Emulated` must implement these exact semantics. This specification does not prescribe accumulator widths, lookup tables, software arithmetic, or other implementation mechanics.

## Aliasing

`lhs` and `rhs` may overlap arbitrarily because both are read-only. Validate only each input versus `out`. For an input and output with the same owner, exact in-place alias is permitted only when `TensorSpec`, plane offset, plane strides, and logical element mapping are identical; therefore the aliased input is not broadcast. Reject every other same-owner input/output relationship, including provably disjoint windows, broadcast input windows, and other non-identical relationships, with `InvalidArgument` before submission.

Capture both input values before each output store. Track all three storages through completion and deduplicate exact aliases. Snapshot view metadata rather than retaining caller view objects.

## Execution, allocation, and performance

Preserve in-order asynchronous tokens, repeat waits and failures, sequence behavior, caller serialization, owner lifetimes, derived temporary metadata snapshots, and exact Device identity. A pre-submit error produces a negative OID and does not mutate output; partial output after an accepted failure is unspecified. CPU may complete inline. Common code contains no vendor types, vendor-kind switch, or global backend state.

ADD never creates, allocates, frees, or relocates operands, output, temporary backend tensors, broadcast copies, or operation-specific device workspace. It may allocate host-side control or metadata and may reuse the same host/device metadata, events, and staging-buffer mechanisms already used by that backend's copy/transfer path, including staging growth when necessary. There is no new generic fallback subsystem, CPU `Device`, CPU tensor, or CPU queue.

Backend-internal emulation may unpack, widen, repack, or stage data when needed for an exact `Emulated` path. Any staged or emulated path is bounded by `O(lhs logical bytes + rhs logical bytes + out logical bytes)`, independent of expanded broadcast sizes, and never materializes expanded operands. It must not retry an accepted native runtime failure. There is no required uniform host-reference fallback; backend policy remains internal while preserving queue order, visibility, lifetimes, and retained asynchronous failures.

## Verified backend constraints

These constraints affect implementation but expose no public knobs or alternate semantics:

- Standard storage uses 16x16 caller-allocator tiles.
- TTNN has nine `NONE` storage leaves including `BOOL`, with eight numeric leaves in ADD scope.
- TTNN native storage is organized per leading plane with 32x32 storage tiles.
- TTNN owns its allocations and uses serialized staging where required, unlike standard caller-allocator storage.

## Implementation touchpoints

### Preliminary OID phase

Implement the public OID type and helpers; common no-throw facades and token/error boundary; all backend `copy` implementations, queue submission paths, and fakes; default unimplemented compute hooks and the protected extension boundary; core/shared/backend coexistence and token/error tests; and updates to `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and public API documentation. Complete and verify this proposed phase before touching ADD.

### ADD phase

After the OID phase, add `include/iom/operation.hpp`, update the public and core headers and `src/iom.cpp`; implement ADD validation, shape mapping, exact numeric arithmetic, and CPU reference behavior; shared queue metadata, kernels, and the multi-owner registry; backend queues and kernels where an ADD path exists; TTNN storage-aware ADD and serialized staging; shared conformance, drivers, fakes, and coexistence; and applicable architecture/contract documentation. Do not add an options object, a signature knob, a numeric codec for quantization, or a backend-specific public fallback API.

The preliminary OID phase migrates unsupported assertions for every OID-returning hook from exceptions to `OidError::Unsupported`. The ADD phase then replaces only ADD's unsupported expectations with its required positive support; all other unimplemented hooks remain negative `Unsupported` results.

## Acceptance and test strategy

All of the following are observable automated acceptance requirements:

- Capability queries run before allocation and verify no side effects. Every enabled backend reports and implements I32 and F32. Each backend driver independently derives the qualifying native set from its build and runtime capabilities and checks that every such leaf reports `Native`; it must not derive the expected set from `add_support` itself. Each reported `Native` or `Emulated` leaf is tested individually, never through an aggregate subspan or partial assertion. Query calls return `Unsupported` for `BOOL`, `F8_E8M0`, and other well-formed unsupported leaves and retain the specified validation exceptions for malformed or non-`NONE` specs; `add` returns the corresponding negative OID without effects.
- Use one independent scalar oracle. For each distinct arithmetic implementation, exhaustively test all ordered raw-encoding pairs for every supported F4, F6, `F8_E4M3FN`, and `F8_E5M2` leaf and for I2, U2, I4, and U4. Use representative exact boundary vectors for wider formats and every distinct arithmetic path.
- Use a compact broadcast matrix covering equal shapes, opposite-direction singleton axes, rank promotion and `[1,1]`, final tiled-axis broadcasts with tails, ranks 2, 3, and 6, and one nested transformed leading view. Test incompatible shapes and incorrect output shapes.
- Test exact `lhs`, exact `rhs`, and all-three exact alias cases, plus representative forbidden windows and read/read overlap. Reject every other same-owner input/output relationship before effects.
- Verify canonical NaNs and zeros, untouched padding and other planes, source preservation, caller allocator silence, no temporary device tensors, and staged-path bounds only when staging is used.
- Use deterministic common/fake seams for each OID synchronous error category and targeted backend fault seams only where an implementation path exists. Verify rejection has no effects or sequence/token side effects; separately test `add_support` exception categories.
- Keep all-255-ID, sequence-exhaustion, skipped-wait, and other OID migration tests in the preliminary migration. ADD-specific asynchronous tests cover ordering, repeated wait, three-owner lifetime, temporary derived views, alias deduplication, retained failure, and coexistence.
- Retain non-ADD `Unsupported` expectations after migration. Do not claim support, codecs, or queries for backends or operations where no implementation path exists.

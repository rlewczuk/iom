# Elementwise addition

## Purpose

Implement backend-neutral raw `DeviceOps::add(lhs, rhs, out) -> oid` on CPU, CUDA, ROCm, SYCL, and TTNN. The caller owns every input and the output tensor. ADD is imperative and follows the asynchronous queue contract. CPU is the normative reference implementation, but the arithmetic contract—not incidental CPU hardware behavior—defines correctness.

## Scope and non-goals

### In scope

- ADD.
- Full multidirectional broadcasting for valid tensor ranks.
- Existing transformed leading views.
- Exact numerical semantics.
- Per-invocation ADD capability reporting.
- Every backend implementation and the corresponding conformance documentation and tests.

### Out of scope

- Graphs, autograd, tensor allocation, and tensor return.
- Mixed-type promotion and implicit casts.
- Public broadcast views and public zero strides.
- Other operations, including `sub`, `mul`, and `cmp`.
- A generic capability framework.
- Rank-0, rank-1, and empty tensors.
- Every non-`NONE` quantization format. Quantization is not part of ADD and remains invalid under the existing `TensorSpec` validation contract.
- Any replacement for removed quantization support.

## Verified current state

- The ADD declaration is a three-view `DeviceOps` virtual method, and all backends currently inherit unsupported behavior. No ADD implementation, numeric codec, fallback, options object, or per-operation capability query currently exists.
- Shapes have rank at least two and nonzero dimensions, and the final two axes are tiled. `TensorView` transforms are leading-only; the relevant symbols are in `include/iom/tensor.hpp` and `src/iom.cpp`.
- Standard backends advertise 23 storage leaves in `src/shared/standard_tiled_copy.hpp`: the 21 numeric leaves listed below plus `BOOL` and `F8_E8M0`. TTNN advertises nine `NONE` storage leaves in `src/ttnn/device.cpp`: `BOOL` plus the eight numeric leaves listed below.
- Current code has bit-width and copy semantics but no ADD numeric codec. Quantization formats other than `NONE` are rejected by the existing tensor-spec validation in `src/iom.cpp`.
- Queue, token, and lifetime rules are documented in `docs/BACKEND_CONTRACT.md`. The repository-wide OID migration below is prerequisite work for ADD.

## Preliminary repository-wide OID migration

ADD cannot begin until this migration is complete and its tests and documentation pass. The migration applies to every public `DeviceOps` operation returning `oid`: `copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`.

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

All negative `oid` values are reserved for errors; the six values above are the complete defined set for this change. The same header provides the exact public helpers:

```text
[[nodiscard]] constexpr oid to_oid(OidError error) noexcept;
[[nodiscard]] constexpr bool oid_is_error(oid value) noexcept;
[[nodiscard]] constexpr bool oid_is_token(oid value) noexcept;
```

`to_oid` returns the enum's underlying negative value, `oid_is_error` is `value < 0`, and `oid_is_token` is `value > 0`. Adding another defined error requires a later public-contract change.

`0` is invalid and is never a token or error code. Positive values are waitable tokens. Preserve all 255 eight-bit queue IDs by encoding each queue ID `q` in `[1, 255]` in bits 55 through 62 and using a 55-bit sequence:

```text
token = static_cast<oid>((std::uint64_t{q} << 55) | sequence)
```

The sequence is in `[1, 2^55 - 1]`; sequence zero is not submitted. The encoding therefore produces every valid positive token through `INT64_MAX` without losing a queue ID. Sequence exhaustion is a synchronous `Overflow` result before effects or token acceptance.

### Error boundary and waiting

Every public OID-returning operation is a common non-throwing `noexcept` facade. Backend extension points sit behind this facade and cannot bypass its validation, error mapping, token encoding, or lifetime registration. Synchronous failures return one negative `OidError` code before any effect or token acceptance; no synchronous exception may cross an OID-returning public method. Accepted work always returns one positive token.

The mapping is uniform across `copy`, `add`, `mul`, `silu`, `linear`, `rmsnorm`, and `sdpa`: invalid caller input maps to `InvalidArgument`; an unavailable operation or unsupported specification maps to `Unsupported`; checked arithmetic or sequence exhaustion maps to `Overflow`; an allocation or bounded-resource failure maps to `ResourceExhausted`; a backend/runtime failure detected before acceptance maps to `DeviceError`; and an otherwise unclassified internal failure maps to `InternalError`. Negative OIDs are terminal synchronous results, not waitable tokens.

Failures that remain asynchronous after acceptance are retained and are repeatably thrown by the existing `void wait(token)` operation. `wait` throws `std::invalid_argument` for a negative, zero, foreign, or unsubmitted value. Waiting a valid token preserves queue ordering, visibility, repeated-wait behavior, and the existing retained-failure semantics. Capability queries do not return OID values and retain their specified exception behavior.

### Separately taskable migration ordering

Planning must create a separate preliminary task for every numbered migration below. Complete items 1 and 2 first; items 3 through 6 are then independent backend migrations; items 7 through 9 finish the repository-wide cutover. ADD implementation starts only after all nine pass.

1. Add `include/iom/oid.hpp`, move the public OID definition into it, define the signed error categories and helpers, and migrate the common 55-bit sequence/token encoding and validation.
2. Replace the public virtual OID operations with common non-throwing facades and protected backend extension points; migrate `submit`, sequence exhaustion, default unimplemented compute hooks, and synchronous exception-to-error mapping.
3. Migrate the CPU `copy` implementation and CPU-specific queue fakes to the new extension boundary and error results.
4. Migrate the shared CUDA/ROCm `copy` queue and both backend drivers to the new extension boundary and error results.
5. Migrate the SYCL `copy` queue and driver to the new extension boundary and error results.
6. Migrate the TTNN `copy` queue and driver to the new extension boundary and error results.
7. Migrate core tests, shared conformance fakes, token-exhaustion/error tests, and unsupported-hook expectations.
8. Migrate backend conformance and coexistence tests, including negative/zero/foreign/unsubmitted waits, error timing, and retained asynchronous failures.
9. Update `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, public headers, and any other public API documentation for the type, encoding, errors, waiting, and compatibility impact.

All tests and documentation in these tasks must be migrated before ADD work starts. The migration is a clean public compatibility cutover: callers must handle negative OID results and must not expect synchronous exceptions from OID-returning operations. It does not change capability-query exception behavior.

## Public ADD API after the prerequisite

Keep exactly three `TensorView` parameters; no `EltwiseOpOptions`, options parameter, or signature knob exists or may be added. As with every public OID-returning operation after the prerequisite, `add` is the non-virtual facade; backend-specific work is implemented behind it:

```text
oid add(const TensorView& lhs, const TensorView& rhs, TensorView& out) noexcept;
```

The common OID facade owns the non-throwing boundary. Broadcasting, arithmetic, fallback eligibility, memory and staging policy, and optimization are fixed semantics or backend-internal policy; none is caller-selectable.

Add the backend-neutral capability enum:

```text
enum class AddSupport : uint8_t {
    Unsupported,
    Emulated,
    NativeExact,
};
```

Add this side-effect-free query, which is not `noexcept` and does not return an OID:

```text
[[nodiscard]] AddSupport Device::add_support(const TensorSpec& lhs,
                                             const TensorSpec& rhs,
                                             const TensorSpec& out) const;
```

`NativeExact` means that this exact same-format `TensorSpec` triple uses a device-native ADD path and meets every exact arithmetic requirement here. Native storage alone is insufficient. `Emulated` means that an exact implementation exists through explicit widening or repacking, software device code, or the uniform host-reference fallback. `Unsupported` means that a well-formed unquantized request has no ADD implementation, including an excluded leaf or a leaf the device cannot store. There is no latency promise.

`add_support` applies `TensorSpec::validate` to each spec, performs no allocation, submission, warning, or runtime mutation, and can run before tensor or queue creation. A recognized non-`NONE` quantization retains the existing `TensorSpec` `std::runtime_error`; an unknown leaf or quantization value, a mismatch among individually valid specs, malformed specs, broadcast incompatibility, or an incorrect output shape throws `std::invalid_argument`. Device storage absence and well-formed excluded unquantized leaves return `Unsupported`. `add` independently validates its three views.

## Supported unquantized specs

With `QuantizationFormat::NONE`, CPU, CUDA, ROCm, and SYCL must implement every currently advertised leaf except `BOOL` and `F8_E8M0`:

- `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and `U64`.
- `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`.

These backends must widen or emulate a leaf when necessary to meet the exact contract. TTNN must implement its eight numeric `NONE` leaves: `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `BF16`, and `F32`. `BOOL` remains storable but ADD is `Unsupported`. Every non-`NONE` quantization is outside ADD for every backend.

## Broadcasting and validation

Rank less than two is invalid. `[1,1]` is the only scalar convention: it broadcasts across every output axis, and two `[1,1]` inputs produce `[1,1]`.

For ranks at least two, broadcasting right-aligns ranks and conceptually prepends ones. Two operand axes are compatible if they are equal or either is one; each result axis is their maximum. The output shape must be exactly the resulting shape. Singleton axes, including final tiled rows and columns, map their operand coordinate to zero. Resolve logical coordinates before mapping to tile or native slots, and never read padding. Do not materialize broadcast copies or expose a public zero-stride view.

Existing transformed leading views retain independent plane offsets and strides. ADD must apply those offsets and strides independently for each input and output while preserving the logical broadcast mapping.

Before submission, allocation, write, or token creation, validate:

- All three views use the exact queue `Device`.
- All three views have identical leaf data type and quantization format.
- The operand pair is supported by the ADD specification.
- The output has the same type and quantization as the inputs.
- The broadcasted result shape is exactly the output shape.
- No invalid aliasing relationship exists.
- All view offsets, strides, and checked size computations are valid.

For `add`, apply the repository-wide mapping: malformed specs and device, type, shape, view, or alias errors return `OidError::InvalidArgument`; a recognized non-`NONE` quantization or well-formed unsupported leaf returns `OidError::Unsupported`; checked sequence, size, offset, stride, and shape arithmetic overflow returns `OidError::Overflow`; scratch allocation failure returns `OidError::ResourceExhausted`; a backend failure detected before work is accepted returns `OidError::DeviceError`; and an unexpected internal failure returns `OidError::InternalError`. These are negative results before effects or token acceptance. No synchronous exception crosses the public OID-returning method. Quantization never receives an ADD implementation or replacement.

## Numerical semantics

Inputs and output have the same leaf type; there is no promotion. Signed integer values use two's-complement interpretation and unsigned values use ordinary binary interpretation. For every integer width, the output is the low `w` bits of the exact sum modulo `2^w`; arithmetic is neither saturating nor undefined.

Scalar floating types decode according to their named format, compute an exact real sum, and encode once to the same format using round-to-nearest, ties-to-even (RNE). Internal widening is permitted only as an implementation technique.

`F4` and `F6` follow the OCP Microscaling Formats v1.0 scalar-element tables, including subnormals, RNE, and saturation to signed maximum finite. `F8_E4M3FN` follows OCP E4M3 with saturation and canonical positive NaN `0x7f`. `F8_E5M2` uses OCP overflow mode and infinity. `F16`, `F32`, and `F64` are IEEE binary16, binary32, and binary64. `BF16` has a sign bit, 8 exponent bits, and 7 fraction bits with IEEE-style specials and RNE.

Arithmetic has gradual underflow and must not use FTZ or DAZ. Zero handling is exact: `-0 + -0 = -0`; opposite zero signs and exact nonzero cancellation produce `+0`; a nonzero value rounded to zero retains its mathematical sign.

For infinity-capable formats, any NaN or opposite infinities produce canonical positive quiet NaN; otherwise an infinity operand determines the result. Canonical NaN encodings are:

- E4M3FN: `0x7f`.
- E5M2: `0x7e`.
- F16: `0x7e00`.
- BF16: `0x7fc0`.
- F32: `0x7fc00000`.
- F64: `0x7ff8000000000000`.

Finite overflow follows the RNE threshold to infinity for formats capable of representing infinity. The OCP Microscaling Formats v1.0, September 2023, is normative for F4, F6, and FP8 encodings: <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>. The explicit ADD policies in this specification take precedence wherever that standard provides options.

## Aliasing

`lhs` and `rhs` may alias arbitrarily. For each input whose `native_handle()` is the same as the output's `native_handle()`, require the input and output to have exactly the same `TensorSpec`, plane offset, and plane strides. Such an exact in-place relationship is allowed. Reject every non-identical same-owner relationship, including relationships that are provably disjoint and broadcast input windows, with `OidError::InvalidArgument` before submission.

Load both logical operands before storing output. Make no restrict assumptions. Validate both input/output relationships before submission.

## Execution, fallback, and performance

Preserve in-order asynchronous tokens, repeat waits and failures, sequence behavior, caller serialization, owner lifetimes, derived temporary metadata snapshots, and exact `Device` identity. Track every unique operand and output storage until completion. Deduplicate exact aliases and extend the registry beyond a source/destination pair. Retain accepted failures. A pre-submit error produces a negative OID and does not mutate output; partial output after an accepted failure is unspecified.

Do not allocate, free, or relocate caller tensors. Do not create an output or copy the full broadcasted tensors. Reuse queue, event, metadata, and staging mechanisms. Bounded internal metadata and scratch are allowed. CPU may complete inline. CUDA and ROCm should share neutral queue logic where practical; SYCL remains equivalent. Common code must contain no vendor types, vendor-kind switch, or global backend state.

Every supported request is either `NativeExact` or `Emulated`. `NativeExact` paths use native same-format ADD. `Emulated` paths may unpack, widen, and repack, but must avoid whole-tensor copies. Exactness takes precedence over performance.

### Uniform host-reference fallback

Host-reference fallback is an internal, uniform per-backend contract. A supported request that cannot use an exact native path is reported `Emulated` and may execute the same CPU reference arithmetic inside the original backend's `DeviceOps`. Fallback never creates a CPU `Device`, CPU tensors, or a CPU queue. It preserves the original positive OID, queue order, visibility, owner lifetimes, derived metadata, and failure behavior; completion covers all staging, host arithmetic, uploads, downloads, and device accesses involved in the accepted operation.

Fallback scratch must be bounded by `O(lhs.spec().logical_nbytes() + rhs.spec().logical_nbytes() + out.spec().logical_nbytes())`, independent of the broadcasted result's expanded operand sizes; it never fully materializes either broadcasted input. Backend memory, staging, serialization, and synchronization choices are internal. Fallback is selected only for a request known before acceptance to require emulation; it never retries an arbitrary accepted native runtime failure. It emits no warning or other user-visible side effect. A synchronous failure before acceptance follows the OID error mapping above, and an accepted asynchronous failure remains repeatably observable through `wait`.

## Verified backend constraints

These constraints affect implementation but expose no public knobs or alternate semantics:

- Standard storage uses 16x16 caller-allocator tiles.
- TTNN has nine `NONE` storage leaves including `BOOL`, but only its eight numeric leaves are in ADD scope.
- TTNN native storage is organized per leading plane with 32x32 storage tiles.
- TTNN owns its allocations and uses serialized staging where required, unlike standard caller-allocator storage.

## Implementation touchpoints

### Preliminary OID phase

Implement the public OID type and helpers; common no-throw facade and token/error boundary; all backend `copy` implementations, queue submission paths, and fakes; default unimplemented compute hooks and the extension boundary; core/shared/backend coexistence and token/error tests; and updates to `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and public API documentation. Complete and verify this phase before touching ADD.

### ADD phase

After the OID phase, implement the public and core headers and `src/iom.cpp`; ADD validation, shape mapping, numeric arithmetic, and CPU reference behavior; shared queue metadata, kernels, and the multi-owner registry; CPU, CUDA, ROCm, and SYCL queues and kernels; TTNN storage-aware ADD and serialized staging; shared conformance, drivers, fakes, and coexistence; and the applicable updates to `docs/ARCHITECTURE.md` and `docs/BACKEND_CONTRACT.md`. Do not add an options object, a signature knob, a numeric codec for quantization, or a backend-specific public fallback API.

The preliminary OID phase migrates unsupported assertions for every OID-returning hook from exceptions to `OidError::Unsupported`. The ADD phase then replaces only ADD's unsupported expectations with its required positive support; all other unimplemented hooks remain negative `Unsupported` results.

## Acceptance and test strategy

All of the following are observable automated acceptance requirements:

- A shared CPU-reference ADD suite is invoked by every driver. Capability queries run before allocation and verify no side effects. CPU, CUDA, ROCm, and SYCL report non-`Unsupported` for every required unquantized leaf; TTNN reports non-`Unsupported` for each of its eight required numeric leaves. Every required type is checked independently, not through an aggregate `subspan(0, 1)` or equivalent partial assertion. Excluded valid unquantized leaves return `Unsupported`, while non-`NONE` specs retain the query's `TensorSpec` validation error.
- Use an independent spec-derived oracle. Test exhaustive ordered pairs for F4, F6, `F8_E4M3FN`, and `F8_E5M2` scalar formats (excluding `F8_E8M0`) and for I2/U2/I4/U4; test wider integer wrap boundaries; and test wider-float ties, subnormal transitions, zeros, cancellations, infinities, NaNs, and overflow with exact bytes.
- Test equal shapes, compatible and incompatible unequal ranks, bidirectional singleton broadcasting, `[1,1]` scalar broadcasting, final row and column broadcasting such as `[B,S,H] + [1,H]`, tile tails, ranks 2 through 6, and transformed and nested leading views with independent offsets and strides. Reject incompatible inputs and incorrect output shapes.
- Test exact `lhs`, exact `rhs`, and both-input alias parity. Reject every non-identical same-owner output relationship before effects.
- Verify canonical NaNs and zeros, untouched padding and owner planes, source windows not identical to output remain unchanged, exact in-place sources become output, caller allocator traffic is absent, instrumented fallback scratch obeys the stated bound, and no broadcast-expanded operand is materialized.
- Use deterministic fault-injection seams to test negative error codes and error timing for malformed/device/type/shape/alias input, unsupported leaves, recognized non-`NONE` quantization reaching an OID-returning operation, checked overflow, scratch exhaustion, pre-accept backend failure, and unexpected internal failure. Verify rejection has no effects and no sequence/token side effects. Separately test that `add_support` retains the specified exception categories.
- Test positive token encoding across all 255 queue IDs, sequence exhaustion, queue ordering, repeated waits, temporary views, three-owner lifetimes, retained asynchronous failure, and foreign/zero/negative/unsubmitted wait errors. Test coexistence without a global backend registry or cross-backend interference.
- Run the numerical suite for every request reported `Emulated` and test native parity for every request reported `NativeExact`. For each backend that implements host-reference fallback, force its seam for representative required types and verify `Emulated` classification, output parity, token visibility, lifetime, and failure behavior. A backend needs no fallback implementation when every supported request has another exact path.
- Retain unsupported assertions for non-ADD hooks after OID migration and expect `OidError::Unsupported`, not synchronous exceptions. Ensure all accelerator suites fail rather than skip; use the remote-development procedures for accelerator execution.

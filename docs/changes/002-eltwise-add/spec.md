# Elementwise addition

## Purpose

Implement backend-neutral raw `DeviceOps::add(lhs, rhs, out) -> oid` on CPU, CUDA, ROCm, SYCL, and TTNN. The caller owns every input and the output tensor. ADD is imperative and follows the asynchronous queue contract. `add(...)` is the sole ADD support signal: accepted supported work returns a positive token, while a well-formed request whose three recognized specs match and use an excluded leaf or recognized non-`NONE` quantization returns negative `OidError::Unsupported`; backend path selection is internal and is not exposed. An independent exact scalar oracle defines the reference; every backend, including CPU, obeys exact integer semantics and the bounded floating contract below.

## Scope and non-goals

### In scope

- ADD.
- Full multidirectional broadcasting for valid tensor ranks.
- Existing transformed leading views.
- Exact integer semantics and bounded floating-point semantics.
- Every backend implementation and the corresponding conformance documentation and tests.

### Out of scope

- Graphs, autograd, and allocation or return of operands or the result by `add` are out of scope.
- Public mixed-type promotion and caller-visible casts. Backend-internal conversions remain permitted while public input and output leaf types are identical.
- Public broadcast views and public zero strides.
- Other operations, including `sub`, `mul`, and `cmp`.
- A generic capability framework or capability queries for operations.
- Rank-0, rank-1, and empty tensors.
- Every non-`NONE` quantization format. Quantization is not part of ADD and remains invalid/out of scope under the existing `TensorSpec` validation contract.
- Any replacement for removed quantization support. TTNN storage expansion and backend-internal temporary buffers, tensors, workspace, staging, and conversions are in scope where required by ADD.

## Verified current state

- The ADD declaration is a three-view `DeviceOps` virtual method, and the backends currently inherit unsupported behavior. No ADD implementation, numeric codec, fallback, or options object currently exists.
- Shapes have rank at least two and nonzero dimensions, and the final two axes are tiled. `TensorView` transforms are leading-only; the relevant symbols are in `include/iom/tensor.hpp` and `src/iom.cpp`.
- The standard table has 23 storage entries in `src/shared/standard_tiled_copy.hpp`: `BOOL` plus 22 numeric leaves. Excluding `F8_E8M0` leaves exactly 21 required ADD numeric leaves.
- TTNN currently advertises `BOOL` plus eight numeric `NONE` storage leaves in `src/ttnn/device.cpp` and rejects other leaves before allocation.
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

Failures that remain asynchronous after acceptance are retained and are repeatably thrown by the existing `void wait(token)` operation. The migrated contract requires `wait` to immediately throw `std::invalid_argument` for a negative, zero, foreign, future, skipped, or otherwise unsubmitted value. A value that was skipped or reserved but never submitted remains invalid even if later submissions complete. Waiting a valid token preserves queue ordering, visibility, repeated-wait behavior, and retained-failure semantics.

### Separately taskable migration ordering

Planning must create a separate preliminary task for every numbered migration below. Complete items 1 and 2 first; items 3 through 6 are then independent backend migrations; items 7 through 9 finish the repository-wide cutover. ADD implementation starts only after all nine pass.

1. Add `include/iom/oid.hpp`, add it to the root CMake explicit public-header list, move the public OID definition into it, define the signed error categories and helpers, and migrate the common 55-bit sequence/token encoding and validation.
2. Replace the public virtual OID operations with common non-throwing facades and protected backend extension points; migrate `submit`, sequence exhaustion, default unimplemented compute hooks, and synchronous exception-to-error mapping.
3. Migrate the CPU `copy` implementation and CPU-specific queue fakes to the new extension boundary and error results.
4. Migrate the shared CUDA/ROCm `copy` queue and both backend drivers to the new extension boundary and error results.
5. Migrate the SYCL `copy` queue and driver to the new extension boundary and error results.
6. Migrate the TTNN `copy` queue and driver to the new extension boundary and error results.
7. Migrate core tests, shared conformance fakes, token-exhaustion/error tests, and unsupported-hook expectations.
8. Migrate backend conformance and coexistence tests, including negative/zero/foreign/future/skipped/unsubmitted waits, error timing, and retained asynchronous failures. Include a focused migrated test for the current `seek_next_sequence` behavior: after it leaves a skipped value and a later submission completes, verify that the skipped value is still rejected.
9. Update `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, public headers, and any other public API documentation for the type, encoding, errors, waiting, and compatibility impact.

All tests and documentation in these tasks must be migrated before ADD work starts. The migration is a clean public compatibility cutover: callers must handle negative OID results and must not expect synchronous exceptions from OID-returning operations. Generic non-OID `Device` queries remain outside the OID result contract.

## Public ADD API after the prerequisite

Keep exactly three `TensorView` parameters; no `EltwiseOpOptions`, options parameter, or signature knob exists or may be added. As with every public OID-returning operation after the prerequisite, `add` is the non-virtual facade; backend-specific work is implemented behind it:

```text
oid add(const TensorView& lhs, const TensorView& rhs, TensorView& out) noexcept;
```

The common OID facade owns the non-throwing boundary and `add(...)` is the sole ADD support signal. A well-formed request whose three recognized specs match and uses a required matrix leaf returns a positive token; matching `BOOL` or `F8_E8M0`, or matching recognized non-`NONE` quantization, returns negative `OidError::Unsupported`. Broadcasting, arithmetic, backend path selection, memory and staging policy, and optimization are fixed semantics or backend-internal policy; none is caller-selectable.

## Supported unquantized specs

With `QuantizationFormat::NONE`, every enabled CPU, CUDA, ROCm, SYCL, and TTNN backend must accept exactly these 21 numeric leaves for ADD, individually:

`I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`.

`BOOL` and `F8_E8M0` are excluded and return `Unsupported` for ADD. Every non-`NONE` quantization remains invalid and out of scope. A backend must not return `Unsupported` for a required numeric leaf merely because its native SDK lacks that dtype; it selects an internal path satisfying the integer and floating reference contract instead. Storage support alone is insufficient unless the backend can create, transfer, copy, and add the leaf.

TTNN currently advertises `BOOL` plus eight numeric `NONE` storage leaves and rejects other leaves before allocation. The target must extend TTNN's `supported_data_types()`, tensor creation and backing, logical host transfers, copy, and ADD so all 21 required numeric leaves produce real `TensorView`s and pass the same matrix. Preserve existing `BOOL` storage, but TTNN need not store `F8_E8M0`.

## Broadcasting and validation

Rank less than two is invalid. `[1,1]` is the only scalar convention: it broadcasts across every output axis, and two `[1,1]` inputs produce `[1,1]`.

For ranks at least two, broadcasting right-aligns ranks and conceptually prepends ones. Two operand axes are compatible if they are equal or either is one; each result axis is their maximum. The output shape must be exactly the resulting shape. Singleton axes, including final tiled rows and columns, map their operand coordinate to zero. Resolve logical coordinates before mapping to tile or native slots, and never read padding. Broadcast copies are backend-internal only; no public zero-stride view is exposed. Final tiled axes, including tails, follow the same logical rule.

Existing transformed leading views retain independent plane offsets and strides. ADD must apply those offsets and strides independently for each input and output while preserving logical broadcast mapping.

Before effects or token acceptance, `add` validates exact queue `Device` identity for all three views, required owner identity relationships, well-formed recognized specs, exact output broadcast shape, valid offsets and strides, checked size, offset, stride, and shape arithmetic, and aliasing. Any mismatch among the three recognized leaf types or quantization formats returns `InvalidArgument`. Only after the three recognized specs match does a common excluded leaf (`BOOL` or `F8_E8M0`) or a common recognized non-`NONE` quantization return `Unsupported`. Other malformed or unknown specs and device, shape, view, or alias errors return `InvalidArgument`; checked arithmetic overflow returns `Overflow`; a pre-acceptance temporary, metadata, staging, or other allocation failure returns `ResourceExhausted`; a backend failure detected before acceptance returns `DeviceError`; and an unexpected internal failure returns `InternalError`.

## Numerical semantics

Inputs and output have the same leaf type; there is no promotion. Signed integer values use two's-complement interpretation and unsigned values use ordinary binary interpretation. For every integer width, the output is the low `w` bits of the exact sum modulo `2^w`; arithmetic is neither saturating nor undefined.

Scalar floating types decode according to their named format, compute an exact real sum, and encode once to the same format using round-to-nearest, ties-to-even (RNE); that encoding is the floating reference. Every backend, including CPU, may return either the reference encoding or a finite encoding at most one ULP away.

The floating reference uses `F4` and `F6` OCP Microscaling Formats v1.0 scalar-element tables, including subnormals, RNE, and saturation to signed maximum finite. `F8_E4M3FN` follows OCP E4M3 with saturation and reference NaN `0x7f`. `F8_E5M2` uses OCP overflow mode and infinity. `F16`, `F32`, and `F64` are IEEE binary16, binary32, and binary64. `BF16` has a sign bit, 8 exponent bits, and 7 fraction bits with IEEE-style specials and RNE.

The floating reference has gradual underflow and does not use FTZ or DAZ. Reference zero handling is exact: `-0 + -0 = -0`; opposite zero signs and exact nonzero cancellation produce `+0`; a nonzero value rounded to zero retains its mathematical sign. For infinity-capable formats, any NaN or opposite infinities produce a reference NaN; otherwise an infinity operand determines the reference result.

Reference NaN encodings are E4M3FN `0x7f`, E5M2 `0x7e`, F16 `0x7e00`, BF16 `0x7fc0`, F32 `0x7fc00000`, and F64 `0x7ff8000000000000`. Finite reference overflow follows the RNE threshold to infinity for formats capable of representing infinity. The OCP Microscaling Formats v1.0, September 2023, is normative for F4, F6, and FP8 encodings: <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>. The explicit ADD policies here take precedence wherever that standard provides options.

One ULP is one adjacency step in the format's ordered finite representable values; `-0` and `+0` are treated as the same value. If the reference is NaN, output must be a NaN but its payload and sign need not be canonical. If the reference is infinity, output must be the same-sign infinity. If the reference is zero, either zero sign is accepted and the general one-ULP envelope applies. Gradual underflow and no-FTZ/DAZ are required insofar as the output meets this envelope.

Every required ADD numeric leaf and every backend path must implement the exact integer contract and this bounded floating contract. This specification does not prescribe accumulator widths, lookup tables, software arithmetic, or other implementation mechanics.

## Aliasing

`lhs` and `rhs` may overlap arbitrarily because both are read-only. Validate only each input versus `out`. For an input and output with the same owner, exact in-place alias is permitted only when `TensorSpec`, plane offset, plane strides, and logical element mapping are identical; therefore the aliased input is not broadcast. Reject every other same-owner input/output relationship, including provably disjoint windows, broadcast input windows, and other non-identical relationships, with `InvalidArgument` before submission.

Capture both input values before each output store. Track all three storages through completion and deduplicate exact aliases. Snapshot view metadata rather than retaining caller view objects.

## Execution, allocation, and performance

Preserve in-order asynchronous tokens, repeat waits and failures, sequence behavior, caller serialization, caller owner lifetimes, stable caller owners and handles, derived temporary metadata snapshots, and exact `Device` identity. A pre-submit error produces a negative OID and does not mutate output; partial output after an accepted failure is unspecified. CPU may complete inline. Common code contains no vendor types, vendor-kind switch, or global backend state.

`add` does not allocate operands or the result and never replaces or relocates caller-created operand/output storage or handles. Backend-internal temporary host/device buffers, backend tensors, broadcast materialization, operation workspace, staging, and implicit conversions are permitted when needed, including TTNN storage expansion; their allocation and cleanup remain internal. There is no new generic fallback subsystem, CPU `Device`, CPU tensor, or CPU queue.

The backend selects its path before effects or token acceptance. Internal fallback or emulation may unpack, widen, repack, stage, convert, or materialize expanded broadcasts for a contract-compliant result. A pre-acceptance temporary or workspace allocation failure maps to `ResourceExhausted`; a failure after positive-token acceptance is retained and repeatably rethrown by `wait`. Never retry an accepted native runtime failure. Backend policy remains internal while preserving queue order, visibility, caller lifetimes, and retained asynchronous failures.

## Verified backend constraints

These constraints affect implementation but expose no public knobs or alternate semantics:

- Standard storage uses 16x16 caller-allocator tiles.
- TTNN currently has nine `NONE` storage leaves including `BOOL`, with eight numeric leaves; the target expands creation, backing, logical host transfers, copy, and ADD to all 21 required numeric leaves while preserving `BOOL` storage.
- TTNN native storage is organized per leading plane with 32x32 storage tiles.
- TTNN owns its allocations and uses serialized staging where required, unlike standard caller-allocator storage; internal backing changes must not replace or relocate caller-visible owners or handles.

## Implementation touchpoints

### Preliminary OID phase

Implement the public OID type and helpers, including the root CMake explicit public-header list entry; common no-throw facades and token/error boundary; base `Device` identity and owner-identity plumbing needed by the common facade and alias checks (without prescribing a concrete accessor); all backend `copy` implementations, queue submission paths, and fakes; default unimplemented compute hooks and the protected extension boundary; core/shared/backend coexistence and token/error tests; and updates to `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and public API documentation. Complete and verify this proposed phase before touching ADD.

### ADD phase

After the OID phase, update the public and core headers and `src/iom.cpp`; implement ADD validation, shape mapping, exact integer arithmetic, the bounded floating contract, the CPU implementation, and the independent exact scalar oracle; shared queue metadata, kernels, and the multi-owner registry; backend queues and kernels for all five required backends; TTNN `supported_data_types()`, storage/backing, logical transfers, copy, and ADD expansion with internal staging; shared conformance, drivers, fakes, and coexistence; and applicable `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and conformance documentation. Do not add an options object, a signature knob, a numeric codec for quantization, or a backend-specific public fallback API.

The preliminary OID phase migrates unsupported assertions for every OID-returning hook from exceptions to `OidError::Unsupported`. The ADD phase then replaces only ADD's unsupported expectations with its required positive support; all other unimplemented hooks remain negative `Unsupported` results.

## Acceptance and test strategy

All of the following are observable automated acceptance requirements:

- Every enabled backend executes every one of the 21 required numeric leaves individually through `add`, with `QuantizationFormat::NONE`, against one independent exact scalar oracle: integer outputs are exact modulo `2^w`, and floating outputs are the reference encoding or a finite encoding at most one ULP away, with the required NaN, infinity, and zero classes. `add` returns a positive token for accepted work.
- Common/fake facade checks send `BOOL`, `F8_E8M0`, and recognized non-`NONE` quantization through `add` and verify negative `OidError::Unsupported`, no effects, and no token acceptance. Required numeric leaves never become `Unsupported` solely because a backend lacks a native SDK dtype.
- TTNN advertises, creates, logically round-trips, and copies all 21 required numeric leaves before its ADD checks. Preserve existing `BOOL` storage; no TTNN `F8_E8M0` storage is required.
- On every enabled backend, exhaustively test all ordered raw-encoding pairs for every required F4, F6, `F8_E4M3FN`, and `F8_E5M2` leaf and for I2, U2, I4, and U4. On every enabled backend, use representative boundary vectors for wider formats. Require exact integer results, floating results within one ULP of the exact reference, and correct special-value classes.
- Use a compact broadcast matrix covering equal shapes, opposite-direction singleton axes, rank promotion and `[1,1]`, final tiled-axis broadcasts with tails, ranks 2, 3, and 6, and one nested transformed leading view. Test incompatible shapes and incorrect output shapes.
- Test exact `lhs`, exact `rhs`, and all-three exact alias cases, plus representative forbidden windows and read/read overlap. Reject every other same-owner input/output relationship before effects. Assert stable caller owners and handles, no replacement or relocation, source preservation, untouched padding and other planes, floating NaN and infinity classes, accepted zero signs, and the one-ULP floating envelope.
- Use deterministic common/fake seams for each OID synchronous error category and applicable deterministic backend fault seams. Verify validation and rejection have no effects or sequence/token side effects; verify deterministic temporary allocation failure and cleanup for allocations that occur; and verify a post-acceptance failure is retained and rethrown by `wait`.
- Keep all-255-ID, sequence-exhaustion, skipped-wait, and other OID migration tests in the preliminary migration. ADD-specific asynchronous tests cover ordering, repeated wait, three-owner lifetime, temporary derived views, alias deduplication, retained failure, and coexistence. The migrated skipped-sequence test must show that a skipped value remains rejected after a later submission completes.
- Retain non-ADD `Unsupported` expectations after migration. Do not claim support or codecs for operations where no implementation path exists.

# Elementwise addition

## Purpose

Implement backend-neutral raw `DeviceOps::add(lhs, rhs, out) -> oid` on CPU, CUDA, ROCm, SYCL, and TTNN. The caller owns all input tensors and the output tensor. The operation is imperative and follows the asynchronous queue contract. CPU is the normative reference implementation, but the arithmetic contract—not incidental CPU hardware behavior—defines correctness.

## Scope and non-goals

### In scope

- ADD.
- Full multidirectional broadcasting.
- Existing transformed views.
- Exact numerical semantics.
- Per-invocation ADD capability reporting.
- TTNN `BFLOAT4_B`/`BFLOAT8_B` support through the specified quantization formats.
- Every backend implementation and the corresponding conformance documentation and tests.

### Out of scope

- Graphs, autograd, tensor allocation, and tensor return.
- Mixed-type promotion and implicit casts.
- Public broadcast views and public zero strides.
- Other operations, including `sub`, `mul`, and `cmp`.
- A generic capability framework.
- Math modes.
- Rank-0 tensors and empty tensors.
- All quantization formats except TT BFP4-B and TT BFP8-B.
- TT BFP2/A variants.
- Aliases between TT block formats and generic F4/F8 formats.

## Current state

- The ADD stub signature is at `include/iom/iom.hpp:231-236`; all backends currently inherit unsupported behavior.
- Shapes are rank >= 2 and nonzero, and the final two axes are tiled. `TensorView` transforms are leading-only; see `include/iom/tensor.hpp:140-179` and `src/iom.cpp:43-53,290-437`.
- Standard backends support storage and copy for 23 leaves at `src/shared/standard_tiled_copy.hpp:11-39`; TTNN currently supports nine at `src/ttnn/device.cpp:107-135`.
- Current code has bit-width and copy semantics but no numeric codec; quantization formats other than `NONE` are rejected at `src/iom.cpp:75-85,115-145`.
- Queue, token, and lifetime rules remain as documented in `docs/BACKEND_CONTRACT.md:267-338`.

## Public API

Keep the existing virtual method unchanged:

```text
virtual oid add(const TensorView& lhs, const TensorView& rhs, TensorView& out)
```

Add the backend-neutral capability enum:

```text
enum class AddSupport : uint8_t {
    Unsupported,
    Emulated,
    NativeExact,
};
```

Add the following `Device` query, which is not `noexcept`:

```text
[[nodiscard]] AddSupport Device::add_support(const TensorSpec& lhs,
                                             const TensorSpec& rhs,
                                             const TensorSpec& out) const;
```

`NativeExact` means that this exact `TensorSpec` triple uses a device-native path on the same formats and meets the exact arithmetic semantics in this specification. Native storage alone is insufficient for this classification.

`Emulated` means that an exact implementation exists through explicit widening or repacking, software device code, staging, or host fallback.

`Unsupported` means that a well-formed combination has no ADD implementation or uses excluded `BOOL` or `F8_E8M0`. For valid specs that are not storable on a device, return `Unsupported` so callers can compare backends.

Before classification, `add_support` applies `TensorSpec::validate` to each spec. A globally unsupported quantization retains `TensorSpec`'s `std::runtime_error`; an invalid leaf/quantization pairing, such as TT_BFP with a non-`BF16` leaf, is `std::invalid_argument`; and a mismatch among individually valid specs is `std::invalid_argument`. Device storage absence returns `Unsupported`. Malformed specs, broadcast incompatibility, and an incorrect output shape throw the same category of exception as `add`. The capability query is side-effect free: it performs no allocation, submission, warning, or runtime mutation and is usable before tensor or queue creation. `add` validates independently. The classification is not a latency promise.

## Supported specs

With `QuantizationFormat::NONE`, CPU, CUDA, ROCm, and SYCL must implement every currently advertised leaf except `BOOL` and `F8_E8M0`:

- `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and `U64`.
- `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`, `F32`, and `F64`.

These backends must widen or emulate a leaf when necessary to meet the exact contract.

TTNN must implement its accepted unquantized numeric set: `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `BF16`, and `F32`. `BOOL` remains storable but ADD is `Unsupported`.

Add valid `TensorSpec` combinations `{DataType::BF16, QuantizationFormat::TT_BFP4}` and `{DataType::BF16, QuantizationFormat::TT_BFP8}`. These map explicitly to TTNN `BFLOAT4_B` and `BFLOAT8_B` B formats. TTNN must store, copy, and add these formats. Other backends may return `Unsupported` or implement exact emulation. `TT_BFP4A`, `TT_BFP8A`, BFP2 variants, and every other non-`NONE` quantization remain rejected. `supported_data_types` remains the `NONE` storage-leaf list.

## Block format contract

The public host representation is canonical packed bytes. Upload and download are bit-preserving and perform no implicit BF16 or F32 conversion.

For each leading plane and logical row, columns are grouped as `0-15`, `16-31`, and so on; groups never cross rows. Group order is lexicographic by leading plane, row, and group. The full tail group is always present.

A TT_BFP8 group consists of one exponent byte followed by 16 lane bytes. In each lane byte, the sign is bit 7 and the magnitude is bits 0-6. A TT_BFP4 group consists of one exponent byte followed by eight bytes. The even lane occupies the low nibble and the odd lane the high nibble; the sign is bit 3 and the magnitude is bits 0-2.

Decode values as follows:

- BFP8: `(-1)^s * (m/64) * 2^(e-127)`.
- BFP4: `(-1)^s * (m/4) * 2^(e-127)`.

A zero magnitude is canonical positive zero and requires a zero sign. NaN, infinity, and signed-zero input encodings are not permitted.

For 16 exact finite values, the canonical encoder is:

1. An all-zero group is encoded as all zero, including its exponent byte.
2. Otherwise choose `e = clamp(floor(log2(max_abs)) + 127, 0, 255)`.
3. For BFP8, scale each magnitude as `abs(x) * 2^(6 - (e - 127))`; for BFP4, scale it as `abs(x) * 2^(2 - (e - 127))`. Round each scaled magnitude to nearest, ties to even (RNE), then clamp it to `127` for BFP8 or `7` for BFP4. Set each nonzero lane's sign to the sign of its input value; zero lanes have sign zero.
4. Do not recompute the exponent after a rounding carry.
5. If every lane rounds to zero, emit an all-zero group.

This defines saturation and underflow. An input group is canonical if and only if decoding it and canonically re-encoding it returns the same bytes. Reject a noncanonical payload before mutating any tensor. Tail-absent lanes must be zero; they participate in neither `max_abs` nor ADD.

The logical byte count is:

`logical_nbytes = leading_plane_count * rows * ceil(columns/16) * group_bytes`,

where `group_bytes` is 17 for TT_BFP8 and 9 for TT_BFP4. Group encoding bytes are intrinsic block-format bytes, not logical-value bytes. Standard padded storage uses the checked size formula `leading_plane_count * padded_rows * (padded_columns / 16) * group_bytes`; all padding groups and lanes are canonical zero. All size and offset arithmetic is checked.

Broadcasting expands decoded represented input values before destination grouping. ADD computes exact value sums and performs one destination-group canonical encode; there is no intermediate BF16 or F32 rounding.

TTNN may translate canonical bytes to native exponent/data sections and 32x32 tiles, but it must preserve the public groups and bits. Pin the format to Tenstorrent BFP-B semantics, citing the official TT data-format report at <https://github.com/tenstorrent/tt-metal/blob/76030e1e24e3fd27d898990c69ea7013483eedfa/tech_reports/data_formats/data_formats.md> and TT ISA `FloatBitPatterns` at <https://github.com/tenstorrent/tt-isa-documentation/blob/5287a62727350bcef35f7b411d1b8a706172ec4c/WormholeB0/TensixTile/TensixCoprocessor/FloatBitPatterns.md>. The public wire order is IOM-defined and is not claimed to be native byte order.

## Broadcasting and validation

Broadcasting right-aligns ranks and conceptually prepends ones. Two operand axes are compatible if they are equal or either is 1; the result axis is their maximum. The output shape must be exactly the resulting shape.

All logical axes, including the final two tiled axes, may broadcast. This is ADD-local coordinate mapping, not a `TensorView` transform. A singleton dimension maps to coordinate 0. Resolve operand logical coordinates before mapping to tile or native slots, and never read padding. Do not materialize broadcast copies.

Before submission, warning, allocation, write, or token creation, validate that:

- All three views use the exact queue `Device`.
- All three views have identical leaf data type and quantization format.
- The operand pair is supported by the ADD specification.
- The output has the same type as the inputs.
- The broadcasted result shape is exactly the output shape.
- No invalid aliasing relationship exists.
- Existing views honor independent plane offsets and plane strides.

An incompatible device, type, shape, or output, a mismatch among individually valid quantizations, and every invalid aliasing relationship, throws `std::invalid_argument` synchronously before any warning, allocation, write, or token. A globally unsupported quantization retains the `TensorSpec` validation `std::runtime_error`. A storable type that is unsupported for ADD throws `std::runtime_error` naming the backend, `add`, and the leaf or quantization. It also throws before effects. Established overflow, `bad_alloc`, and other runtime failures retain their existing behavior.

## Numerical semantics

Inputs and output have the same type; there is no promotion. Signed integer values use two's-complement interpretation; unsigned integer values use ordinary binary interpretation. For every integer width, the output is the low `w` bits of the exact sum modulo `2^w`; arithmetic is neither saturating nor undefined.

Scalar floating types decode according to their named format, compute an exact real sum, and encode once to the same format using RNE. Internal widening is permitted only as an implementation technique.

`F4` and `F6` follow the OCP Microscaling Formats v1.0 scalar-element tables, including subnormals, RNE, and saturation to signed maximum finite. `F8_E4M3FN` follows OCP E4M3 with saturation and canonical positive NaN `0x7f`. `F8_E5M2` uses OCP overflow mode and infinity. `F16`, `F32`, and `F64` are IEEE binary16, binary32, and binary64. `BF16` has a sign bit, 8 exponent bits, and 7 fraction bits with IEEE-style specials and RNE.

Arithmetic has gradual underflow and must not use FTZ or DAZ. Zero handling is exact: `-0 + -0 = -0`; opposite zero signs and exact nonzero cancellation produce `+0`; a nonzero value rounded to zero retains its mathematical sign.

For infinity-capable formats, any NaN or opposite infinities produce canonical positive quiet NaN; otherwise an infinity operand determines the result. Canonical NaN encodings are:

- E4M3FN: `0x7f`.
- E5M2: `0x7e`.
- F16: `0x7e00`.
- BF16: `0x7fc0`.
- F32: `0x7fc00000`.
- F64: `0x7ff8000000000000`.

Finite overflow follows the RNE threshold to infinity for formats capable of representing infinity.

The OCP Microscaling Formats v1.0, September 2023, is normative for F4, F6, and FP8 encodings: <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>. The explicit ADD policies in this specification take precedence wherever that standard provides options.

## Aliasing

`lhs` and `rhs` may alias arbitrarily. For each input whose `native_handle()` is the same as the output's `native_handle()`, require the input and output to have exactly the same `TensorSpec`, plane offset, and plane strides. Such an exact in-place relationship is allowed. Reject every non-identical same-owner relationship, including relationships that are provably disjoint and broadcast input windows.

Load both logical operands before storing output. Make no restrict assumptions. Validate both input/output relationships before submission.

## Execution and performance

Preserve in-order asynchronous tokens, repeat waits and failures, sequence behavior, caller serialization, owner lifetimes, derived temporary metadata snapshots, and exact `Device` identity.

Track every unique operand and output storage until completion. Deduplicate exact aliases and extend the registry beyond a source/destination pair. Retain accepted failures. A pre-submit error produces no token and does not mutate output. Partial output after an accepted failure is unspecified.

Do not allocate, free, or relocate caller tensors. Do not create an output or copy the full broadcasted tensors. Reuse queue, event, metadata, and staging mechanisms. Bounded internal metadata and scratch are allowed. CPU may complete inline. CUDA and ROCm should share neutral queue logic where practical; SYCL remains equivalent. Common code must contain no vendor types, vendor-kind switch, or global backend state.

`NativeExact` paths use native same-format ADD. Exactness takes precedence over performance. Emulated paths may unpack, widen, and repack, but must avoid whole-tensor copies. TTNN fallback is governed by the separate rules below.

## TTNN fallback

Determine native eligibility before submitting native arithmetic, including whether a native path can meet the exact semantics. For a valid request known to be unsupported by the software stack or for which the native path cannot meet those semantics, execute internally on the host using the same CPU reference arithmetic. Never instantiate a CPU `Device`, CPU tensors, or a CPU queue, and never retry an arbitrary native runtime failure through fallback.

TTNN ADD support in this specification targets Blackhole and newer devices. Older TTNN hardware is outside the supported target and must be rejected as unavailable rather than routed through the software-gap fallback.

Fallback returns one TTNN `oid` and preserves the original identity, order, lifetimes, and failure behavior. It sees writes that preceded it; later work sees its upload. Completion covers downloads, CPU arithmetic, upload, and all device accesses. Do not compose public host transfers inside the active queue.

Allow TTNN-only host scratch of at most `O(lhs + rhs + out)` logical bytes. Do not allocate caller tensors. Retain or quarantine resources until completion is proven.

After validation and acceptance, emit exactly once per fallback submission to stderr only after `submit` has committed a waitable token and before `add` returns:

```text
IOM warning: TTNN add is using CPU fallback
```

A synchronous pre-enqueue failure returns no token and emits no warning. Emit no warning for native submissions, rejected requests, or repeated waits. Ignore stderr failure. An accepted submission that later fails is still warned exactly once.

## Implementation touchpoints

Implementation must cover the public and core headers and `src/iom.cpp`; numeric and block codecs; the CPU queue; shared CUDA/ROCm queue metadata, kernels, and the three-handle registry; the SYCL queue and kernel; TTNN storage, copy, queue, and staging; shared conformance, drivers, and coexistence; and `docs/ARCHITECTURE.md` and `docs/BACKEND_CONTRACT.md`.

This is a clean cutover: replace ADD-unsupported assertions only. Retain unsupported tests for all other hooks.

## Acceptance and test strategy

All of the following are automated acceptance requirements:

- A shared CPU-reference ADD suite is invoked by every driver. Positive types are selected using `add_support`. Capability-query tests run before allocation and verify no side effects. CPU, CUDA, ROCm, and SYCL must report non-`Unsupported` for every required unquantized leaf; TTNN must report non-`Unsupported` for every required unquantized and block-format spec. The query must not hide missing support by selecting only the types it reports as positive.
- Test exhaustive ordered pairs for F4, F6, `F8_E4M3FN`, and `F8_E5M2` scalar formats (excluding `F8_E8M0`) and for I2/U2/I4/U4; test wider integer wrap boundaries; and test wider-float ties, subnormal transitions, zeros, cancellations, infinities, NaNs, and overflow with exact bytes. Include independent spec-derived codec cases so production and reference codecs cannot self-validate a shared bug.
- Test equal shapes; right-rank mismatch; bidirectional singleton broadcasting; final row and column broadcasting such as `[B,S,H] + [1,H]`; tile tails; ranks 2 through 6; and transformed and nested leading views. Reject incompatible inputs and incorrect output shapes.
- Test exact `lhs`, `rhs`, and both-input alias parity. Reject every non-identical same-owner output relationship.
- Verify output padding and untouched owner planes are preserved, source windows not identical to output remain unchanged (exact in-place sources necessarily become output), caller allocator traffic is absent, and broadcast replication is absent.
- Test tokens, ordering, repeated waits, temporary views, three-owner lifetimes, fault retention, and absence of sequence effects on rejection. Test coexistence without a global backend registry or cross-backend interference.
- For block formats, test exhaustive individual exponent and lane-field codec coverage plus representative multi-lane canonical groups; malformed and tail rejection; exponent selection; RNE, carry clamp, saturation, and underflow; logical and padded sizes; broadcast-before-repack; and a native physical oracle. Test every reported `NativeExact` path, and use a seam to force fallback for representative supported types and block formats, checking parity, warning, error, and lifetime behavior.
- Enabled accelerator suites must fail rather than skip. Use the remote-development procedures for accelerator execution. Run all backend conformance suites after every nontrivial implementation.

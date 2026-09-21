# Linear projections

This is the operation-owned contract for `DeviceOps::linear`. The public
facade, its admission rules, and its pure requirement query are frozen here
before implementation begins; the default common hooks keep a well-formed
request `Unsupported` exactly as [section 9](other-compute-capabilities.md#9-other-compute-capabilities)
states, and an unsupported port never counts as numerical conformance. This
section is the single normative source for the linear operation's ABI, layout
equations, validation precedence, ownership and lifetime, dtype and
quantization classification, integer and scalar arithmetic, nonfinite
behavior, workspace formulas, backend capability, fixture thresholds, and
native-matrix evidence.

## Public ABI, layout, and view semantics

The complete public surface is exactly `DeviceOps::linear` and
`DeviceOps::linear_workspace_requirements`. The frozen declarations, in this
exact token order and with these exact defaults, are:

```cpp
enum class LinearOutputLayout { ordinary, head_planar };

oid linear(const TensorView& x, const TensorView& w, TensorView& out,
           std::size_t s, std::size_t R, LinearOutputLayout layout,
           std::size_t H, std::size_t D,
           RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements linear_workspace_requirements(
        const TensorView& x, const TensorView& w, const TensorView& out,
        std::size_t s, std::size_t R, LinearOutputLayout layout,
        std::size_t H, std::size_t D);
```

There is no second argument order, no overload of `linear`, no public status
or capability-registry API, and no transpose, head-pack, or row-extraction
API; the three-view `linear(x, w, y)` facade is not part of this contract. The
requirement query receives the same semantic arguments in the same order as
submission, with only the output made const and the workspace omitted. The
`noexcept` facade returns a positive accepted OID, or the established negative
`InvalidArgument`, `Unsupported`, `Overflow`, `ResourceExhausted`,
`DeviceError`, and `InternalError` OIDs described in
[TinyLlama forward layout — Workspace and execution](tinyllama-forward-layout-workspace-and-execution.md#tinyllama-forward-layout--workspace-and-execution).

`x` is `[...,T,I]`, the weight is exactly rank-two `w[O,I]` and is shared
unchanged across every independent leading plane, and `out` is the caller's
result leaf. `w` uses the Hugging Face `[out,in]` orientation: output
coordinate `o` selects weight row `w[o,*]`, and the operation never transposes
a checkpoint weight. `enum class LinearOutputLayout { ordinary, head_planar }`
is the fixed output-mode type, and output rank never selects the mode.
`ordinary` output is `out[...,R,O]` and requires exactly `H=1` and `D=O`.
`head_planar` output is `out[...,H,R,D]` and requires the checked equality
`O=H*D`, inserting exactly one head axis. Output rank is at most eight, so a
rank-eight ordinary output is structurally valid while an inserted head axis
that would reach rank nine is rejected as `InvalidArgument`. `x` and `out`
carry the exact same leading tuple after excluding the inserted head axis,
each with its own selected plane offset and its own transformed leading offsets
and strides; the rank-two weight's empty leading tuple is not matched and
implies no broadcast, no singleton expansion, and no model-state or cache
broadcast.

`T`, `I`, `O`, `H`, `D`, and `R` are runtime values, never checkpoint
constants. `s` is nonnegative, `R > 0`, and `R <= T-s`, so the selected source
window is exactly `x[b,s+r,i]` for `r` in `0..R-1`; `R` is never inferred from
`s` or `T`, and `O` equals `H*D` only after checked head-planar validation.
With `b` denoting the complete leading tuple, the semantic equations are

```text
out[b,r,o]   = sum(i=0..I-1) x[b,s+r,i] * w[o,i]
out[b,h,r,d] = sum(i=0..I-1) x[b,s+r,i] * w[h*D+d,i].
```

Leading planes, selected rows, heads, and features are independent: no result
crosses a plane or a row window, no result reads a logical element outside
these equations, and no result depends on `16x16` tile padding, sub-byte
remainder bits, or uninitialized storage.

The final untied LM head uses `ordinary` mode with `s = input_run - 1`,
`R = 1`, `H = 1`, and `D = V`, writing `[1,V]` logits directly into the
supplied output with no final-axis slice and no last-row extraction operation.
Projection rearranges only newly computed selected rows. It MUST NOT
materialize repeated KV heads, make a persistent host copy or transpose of
checkpoint weights, or expose a public transpose, head-pack, or extraction
operation.

## Validation, admission, and failure precedence

Before owner registration, sequence consumption, token acceptance, metadata
effects, or backend work, a submission validates in exactly this order:

1. structural rank and extents: well-formed views, `x` of rank two through
   eight with nonzero extents, a rank-two `w`, an `out` of rank two through
   eight, and a recognized `LinearOutputLayout` value;
2. the mode and head relation: `ordinary` requires exactly `H=1` and `D=O`,
   while `head_planar` checks `H*D` for overflow before comparing it with `O`
   and then requires exactly `O=H*D`; either way a validated request has
   nonzero `H` and `D`, because `O` is a nonzero extent and `H=1` in ordinary
   mode;
3. the selected-row range `s <= T`, `R > 0`, and `R <= T-s`;
4. the exact leading tuple and transformed leading strides, including the
   inserted head-axis exclusion for `head_planar`, with no broadcast and no
   guessed mode;
5. exact queue `Device` identity for every view, stable live owner
   registration, and a live native handle;
6. liveness and bounds: selected plane bounds and leading view bounds;
7. every checked element, bit, byte, address, stride, plane, tile, and product
   arithmetic, including output rank growth, rejecting overflow before
   narrowing, size conversion, or pointer arithmetic;
8. operand, output, and workspace disjointness and alias rules, followed by a
   recognized `QuantizationFormat::NONE`, leaf classification, and immutable
   backend capability; and
9. the caller's supplied workspace against the reported `{bytes, alignment}`:
   liveness, exact-device identity, size, alignment, disjointness from every
   operand and from the output, and lease availability.

The rejection results are normative:

| Condition | Result |
| --- | --- |
| malformed view, rank outside `2..8`, zero extent, non-rank-two weight, invalid final shape, unmatched leading tuple or transformed strides, wrong mode/`H`/`D`/`O` relation, or invalid row window | `InvalidArgument` |
| unrecognized `LinearOutputLayout` value | `InvalidArgument` |
| rank growth beyond eight | `InvalidArgument` |
| output/operand overlap, including conservative same-owner rejection and standard-backend backing-range intersection or identical native handles | `InvalidArgument` |
| supplied workspace that is not a live exact-device range, too small, misaligned, out of range, or overlapping an operand or output | `InvalidArgument` |
| unknown `DataType` or `QuantizationFormat` enumeration value | `InvalidArgument` |
| recognized inapplicable leaf, recognized mixed-operand leaf request, recognized unsupported leaf or backend capability, or recognized non-`NONE` quantization format | `Unsupported`, only after structural validation |
| checked element, bit, byte, address, stride, plane, tile, product, or rank-growth overflow | `Overflow` |
| overlapping live workspace lease, or bounded-resource exhaustion | `ResourceExhausted` |
| pre-acceptance runtime or device failure | `DeviceError` |
| any other unclassified failure | `InternalError` |

Admission failures map through `noexcept` to these negative OIDs and consume
no token, register no owner, submit no work, and mutate no queue, token,
metadata, or output state; zero is never accepted. An accepted submission
returns a positive OID whose shapes, leaf types, quantization, plane offsets,
leading strides, owners, native handles, scalars, `layout`, and workspace range
are immutable. A failure after acceptance leaves output unusable and repeats
the same error on every wait; no rollback and no failed-output-unchanged
guarantee is made, and a consumer of a failed producer MUST NOT be submitted
after that producer's wait fails. The caller MUST fail and drain every accepted
OID before reset or destruction, continuing to drain even when an individual
wait throws.

## Ownership, liveness, aliasing, and workspace

Output and workspace MUST each be disjoint from every operand and from each
other, so `out` is rejected when it shares an owner with `x` or `w`, including
the conservative case where the transformed windows appear disjoint; standard
backends additionally reject intersecting backing ranges and identical native
handles. Both inputs are reads, so `x` and `w` MAY overlap or alias exactly.
In-place operation is not promised, and admission allocates, relocates,
replaces, or silently converts no operand or output.

Submission snapshots by value the validated shapes, leaf types, quantization,
plane offsets, leading offsets and strides, scalar parameters, `layout`, and
workspace range, together with the exact live registered owner and native
handle of `x`, `w`, and `out`. It registers the distinct owners, deduplicating
read/read aliases, leases positive scratch through proven completion, and
retains no borrowed `TensorView`: a temporary caller view may die as soon as
its submission is accepted, while caller-owned output and workspace stay alive
through proven completion. No hidden allocation, hidden synchronization, host
roundtrip, persistent transpose, or output relocation is permitted, and no
checkpoint weight is copied, transposed, or staged on the host.

The requirement query is pure and deterministic. It allocates nothing,
including host metadata; constructs no request or vector snapshot; registers
or leases no owner; submits nothing; reads no operand or result data; mutates
no queue, token, lease, or state; and depends only on the supplied views and
immutable backend capability, so it stays valid while the queue is occupied. It
validates the same operands, scalars, `layout`, aliasing, capability, and
checked arithmetic as submission and returns `WorkspaceRequirements`, or
throws the established validation exception instead of an OID. The supplied
workspace is deliberately not a query operand: it is validated only by
submission, after a successful query.

## Dtypes, quantization, arithmetic, and nonfinite behavior

Applicable leaves are exactly these 21:

`I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, `U64`,
`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
`F32`, and `F64`.

`BOOL` and `F8_E8M0` are inapplicable: a boolean is neither an integer nor a
signed numeric product, and an unsigned exponent-only encoding is not a general
signed numeric result. `x`, `w`, and `out` use one leaf type and
`QuantizationFormat::NONE`; only `NONE` is supported, and the operation adds no
quantization or storage format. An unknown `DataType` or `QuantizationFormat`
enumeration value is `InvalidArgument`. A recognized inapplicable leaf, a
recognized mixed-operand leaf request, a recognized unsupported leaf or
backend capability, and a recognized non-`NONE` quantization format are
`Unsupported`, and only after structural validation has already succeeded.
Semantic applicability is independent of native storage: a backend limitation
decides which leaves that backend implements and never narrows this
classification. `BOOL` and `F8_E8M0` are inapplicable on every backend, so no
capability record supersedes this classification.

**Integer arithmetic.** The 12 integer leaves use exact dot products in
unsigned modulo-`2^N` arithmetic. Every multiply and add reduces modulo `2^N`
at that step, which is equivalent to reducing only the final sum; no step
depends on signed overflow, and no value is ever cast to an out-of-range signed
type. A signed output leaf stores the two's-complement bit pattern of the
accumulated value, with no saturation, clamping, widening, or implicit
conversion to a floating leaf.

**Scalar floating recurrence.** The scalar path starts at `+0` and processes
increasing `i`: `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`,
`F16`, `BF16`, and `F32` decode to FP32 and accumulate with correctly rounded
FP32 fused multiply-add, `F64` accumulates with FP64 fused multiply-add, and
the accumulated sum is encoded exactly once to the output leaf with the
existing named-format encoding rules. Those existing rules govern infinity,
NaN, saturation, subnormal results, and signed zero; `F8_E4M3FN` has NaN but no
infinity, and `F64` is never narrowed to FP32. Implementations MUST NOT enable
flush-to-zero or fast-math, and no NaN-payload equality is promised.

**Native BF16.** A native BF16 specialization may reassociate the product sum
only when it accumulates in FP32 or in a demonstrably equivalent width whose
result matches the FP32 rule, and it MUST store the result once with
round-to-nearest, ties-to-even. Rounding through BF16 partial products, or
accumulating in BF16, is non-conforming.

**Nonfinite values.** Admission MUST NOT scan operands for nonfinite values and
MUST NOT introduce a host transfer or a linear-specific queued-data failure.
Nonfinite inputs follow the recurrence and the existing named-format rules
above, and no NaN payload is preserved or compared. A post-acceptance failure
is a runtime or device failure and follows the repeat-wait rule of the previous
subsection.

## Backend capability matrix

Applicability is exactly the 21 leaves above, and the matrix below is the
complete leaf-by-backend capability record.

| Leaf | Contract | CPU | CUDA | ROCm | SYCL |
| --- | --- | --- | --- | --- | --- |
| `BOOL` | inapplicable | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
| `I2`, `U2` | applicable | supported | supported | supported | supported |
| `I4`, `U4` | applicable | supported | supported | supported | supported |
| `I8`, `U8` | applicable | supported | supported | supported | supported |
| `I16`, `U16` | applicable | supported | supported | supported | supported |
| `I32`, `U32` | applicable | supported | supported | supported | supported |
| `I64`, `U64` | applicable | supported | supported | supported | supported |
| `F4_E2M1` | applicable | supported | supported | supported | supported |
| `F6_E2M3` | applicable | supported | supported | supported | supported |
| `F6_E3M2` | applicable | supported | supported | supported | supported |
| `F8_E4M3FN` | applicable | supported | supported | supported | supported |
| `F8_E5M2` | applicable | supported | supported | supported | supported |
| `F8_E8M0` | inapplicable | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` |
| `F16` | applicable | supported | supported | supported | supported |
| `BF16` | applicable | supported | supported | supported | supported |
| `F32` | applicable | supported | supported | supported | supported |
| `F64` | applicable | supported | supported | supported | `aspect::fp64` only |

CPU, CUDA, ROCm, and SYCL implement all 21 semantic leaves. On CUDA, ROCm,
and SYCL the scalar path covers the 20 non-BF16 leaves and a separate native
BF16 specialization covers `BF16`; on CPU, `BF16` uses the scalar recurrence
above. SYCL reports `F64` `Unsupported` at runtime unless the device reports
`aspect::fp64`; that gate is a genuine device fact. `BF16` weights, activations,
and caches remain mandatory on all four retained backends.

Missing implementation is never unsupported hardware. An unported backend or
leaf reports `Unsupported` as missing capability and names its missing
evidence; no backend may advertise a capability it has not implemented, and a
rejection-only probe, a storage-only observation, or a host or elementwise
substitute is never linear conformance. The per-backend native feasibility
records remain the [CUDA](tinyllama-forward-layout-cuda-matrix-feasibility.md#tinyllama-forward-layout--cuda-matrix-feasibility),
[ROCm](tinyllama-forward-layout-rocm-matrix-feasibility.md#tinyllama-forward-layout--rocm-matrix-feasibility), and
[SYCL](tinyllama-forward-layout-sycl-matrix-feasibility.md#tinyllama-forward-layout--sycl-matrix-feasibility) matrix records.

## Workspace requirements and packed-writer ownership

The reported requirement is exact per path. `P` is the checked product of the
logical leading extents of `x`, equivalently of `out`; `pad16(n)` is the
checked round-up of `n` to a multiple of 16; and `A32(n)` is the checked
round-up of `n` to a multiple of 32, which is the alignment those paths report.

| Path | Requirement |
| --- | --- |
| CPU, every applicable leaf | `{0, 1}` |
| CUDA, every applicable leaf | `{0, 1}` |
| ROCm, the 20 scalar leaves | `{0, 1}` |
| ROCm `BF16` | alignment 32, bytes `A32(P*pad16(R)*pad16(I)*2) + A32(P*pad16(R)*pad16(O)*2)` |
| SYCL, the 20 scalar leaves | `{0, 1}` |
| SYCL `BF16` | alignment 32, bytes `A32(P*pad16(R)*pad16(O)*4)` |

A `{0, 1}` requirement means only the empty `RawWorkspaceView{}` is
admissible, and a supplied owner is `InvalidArgument` before dispatch. Where
the requirement is positive, the range MUST be a live exact-device owner
range, 32-byte aligned, at least the reported bytes, disjoint from every
operand and from the output, and leased through proven completion. Owner-
absolute subranges are checked and 32-byte aligned, disjoint aligned ranges
may be used concurrently, and overlapping live leases reject with
`ResourceExhausted`. The positive range is caller-owned staging only: it
carries no control packet, so no status word is reset, transferred, or
interpreted, and no operand, weight, or result is staged on the host or inside
a library's hidden internal workspace. Proven completion releases the lease
and permits safe reuse of independent scratch; unknown completion retains or
quarantines the range instead of reusing it.

**Packed sub-byte ownership.** The `I2`, `U2`, `I4`, `U4`, `F4_E2M1`,
`F6_E2M3`, and `F6_E3M2` leaves are stored packed at their logical width. Every
packed writer MUST own whole physical write units race-safely: exactly one
writer per destination byte or word, no read-modify-write of a unit that a
concurrent writer also owns, and no two writers sharing one packed unit for
different logical cells. Every physical bit outside a writer's own logical
cells, including remainder bits, padding bits, and `16x16` tile padding, MUST
be preserved, and valid logical output MUST NOT depend on those bits.

## Fixtures, reference, and tolerances

The fixture set is fixed before measurement and MUST include all of the
following:

- the non-square reference `I=3, O=10, H=2, D=5, T=19, s=2` across the exact
  runs `R=17` selecting `x[2..18]`, `R=16` selecting `x[2..17]`, `R=15`
  selecting `x[2..16]`, and `R=1` selecting `x[2]`. Ordinary output columns
  `o=0..9` consume `w[0]` through `w[9]`; head-planar `(h=0,d=0..4)` maps to
  `o=0..4` and `(h=1,d=0..4)` maps to `o=5..9`. A transposed checkpoint
  weight, an inferred or off-by-one row window, and an `h`/`d` shuffle are
  therefore observably wrong;
- the non-tile inner extent `I=65`, which spans five `16`-wide inner tiles and
  exercises inner-tail masking, accumulation order, and packed sub-byte inner
  tails;
- independent leading planes: rank two through eight, distinct per-operand
  plane offsets and leading transforms for `x` and `out`, a distinct final row
  so a wrong `s` is observable, and no result that crosses a plane;
- padding invariance: deliberately perturbed `16x16` tile padding, sub-byte
  remainder bits, and uninitialized storage whose change never alters valid
  output; and
- the LM-head case with `s = input_run - 1`, `R = 1`, ordinary mode, and
  `[1,V]` output.

Comparison policy is fixed before measurement, and the reference MUST be
independent: production code MUST NOT serve as its own oracle.

| Leaf class | Comparison |
| --- | --- |
| the 12 integer leaves | exact output bits |
| `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16` | exact encoded result of the scalar recurrence |
| `BF16` | the FP64 evaluation of the equation rounded once to BF16, with `abs(actual - q) <= max(ULP_BF16(q), 2^-7, 2^-6 * abs(q))` |
| `F32` | `abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)` |
| `F64` | `abs(actual - reference) <= 1e-12 + 1e-12 * abs(reference)` |

These are fixed fixture thresholds, not arbitrary-input accuracy promises.

The independent reference MUST evaluate the ordinary and head-planar equations
in exact integer arithmetic for the integer leaves and in FP64 for the floating
leaves, derive the row window, head, and weight-row mapping itself, and include
a deliberate permuted-oracle or transposed-weight sanity check. Shared coverage
MUST exercise every applicable leaf, both layouts, all four row runs, the
`I=65` inner tail, non-tile `O`, `H`, `D`, and `I`, independent leading planes
and transformed mappings, padding invariance, the LM-head case, malformed
structure, wrong device, stale registration, alias and overlap, mixed and
unsupported leaves, non-`NONE` quantization, every checked-overflow class,
invalid `layout` values, row-window violations, rank growth, workspace
liveness, size, alignment, device, overlap, and lease rejections, an accepted
failure with repeated waits, and scratch reuse after proven drain.

The cases live in the shared header
`test/backend/backend_conformance_linear.hpp` and run through the existing
`iom_backend_conformance_cpu_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, and `iom_sycl_conformance_tests` drivers. No
second test project, generic test framework, model fixture, checkpoint, or
network dependency is permitted, and common code never switches on backend kind.
An unported backend keeps an empty linear span and asserts `Unsupported` for
every applicable leaf; that rejection probe is not projection conformance.
Each retained port owns its own `linear` capability probe, while the unrelated
copy/add/mul/sub/div suites remain unchanged.

The fixture provenance and case links of that header are:

- the independent raw/scalar reference decodes and encodes every named leaf
  itself, implements the unsigned modulo-`2^N` integer dot and the `+0`-started
  increasing-`i` correctly rounded FP32 fused-multiply-add recurrence (`FP64`
  for `F64`) with exactly one final encode, derives the row window, head
  coordinate, and Hugging Face weight row itself, and provides the independent
  FP64 evaluation behind the BF16, F32, and F64 thresholds; it calls no
  production codec, address mapper, admission, linear, CPU, or backend helper;
- the deliberate negative variants are `transposed_weight`, `head_order`,
  `head_repeat`, `wrong_row_window`, `mixed_plane`, `padding_dependent`,
  `tile_tail`, and `numeric_reencode`; the self-check asserts for every
  applicable leaf that each is detected — bit-exactly where the thresholds
  compare exact bits, and as a demonstrated model difference where a threshold
  admits a bounded deviation — and that each coincides with the reference on a
  fixture where the perturbation is genuinely invisible, so detection is
  selective rather than unconditional;
- the shared case matrix supplies the canonical `I=3, O=10, H=2, D=5, T=19,
  s=2` fixture across `R=1, 15, 16, 17` in both layouts, the `I=65` and `I=33`
  inner tails across 16-wide inner tiles and 32-wide inner groups, the LM-head
  `s = input_run - 1, R = 1` `[1, V]` request, the leading-plane profiles of
  rank two through eight with distinct per-operand plane offsets, leading
  transforms, and strides, deliberately poisoned native padding, and per-leaf
  special-class fixtures whose zero, signed, subnormal, infinity, and NaN
  classes match each leaf's own encoding;
- each driver passes an explicit declaration of its target leaf matrix, its
  scalar and native-BF16 paths, its exact scratch path (`{0, 1}`, the ROCm
  aligned BF16 sum, or the SYCL `A32(P*pad16(R)*pad16(O)*4)` range), the leaves
  this revision implements, and — on SYCL — the `sycl::aspect::fp64` device
  fact that gates `F64`, so a capability expectation is never an echo of an
  implementation report;
- ROCm implements all twenty-one applicable leaves: the twenty non-BF16
  leaves through the shared raw-word scalar kernel and `BF16` through the
  native specialization in `src/rocm/copy.hip` (checked caller workspace,
  direct GFX12 wave32 BF16 WMMA with FP32 accumulation, one RNE BF16
  scatter), and declares exactly that implemented span, so the suite runs the
  independent reference numerically on the real device for every applicable
  leaf;
- the ROCm per-leaf device evidence is complete for all twenty-one applicable
  leaves under the shared fixture set and comparison policy; the `F32` and
  `BF16` non-finite-class reconciliation authorized for the shared fixture is
  part of the gate this leaf ran, and no leaf is reported separately as
  unclaimed here; and
- the common admission, ownership, workspace, queue-order, and failure cases
  run against a declaration-driven common double next to the host-only
  reference self-check.

The executed ROCm native `BF16` record of this revision is the following
observation, collected from this leaf's task worktree (worktree base
`3421f67fb6817e97444353bb03343ac33b02b24d`) on the configured `rocm` host:

| Field | Observation |
| --- | --- |
| Backend, device | ROCm, `gfx1201` (Radeon AI PRO R9700), wave32, ordinal 0 |
| Toolchain | HIP `7.15.26333-0000000`, AMD clang `23.0.0git`; `hipRuntimeGetVersion` `71526333` |
| Profiler | `rocprofv3` version `1.3.5`, git revision `6b0e43f341195e203754e08f850e437ff2fc09f9` |
| Profiler command | `rocprofv3 --kernel-trace --output-format csv -d prof -o evi -- <evidence driver>` |
| Layout, dimensions | `x[...,T,I]` and HF `w[O,I]`: `P=1, T=19, I=3, O=10, s=2`; ordinary `H=1, D=10` and head-planar `H=2, D=5`; `R in {1,15,16,17}` |
| Padded dimensions | `pad16(R)/pad16(I)/pad16(O)`: `16/16/16` for `R in {1,15,16}` and `32/16/16` for `R=17` |
| Kernel symbols | `linear_bf16_pack_kernel`, `linear_bf16_wmma_kernel`, `linear_bf16_scatter_kernel` (`iom::detail`, one dispatch of each per accepted submission; `9/9/9` dispatches for the eight fixture submissions plus one wider submission) |
| Facility | `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` executed in the same session as the submissions and verified against the exact `16x16x16` product (`facility_probe` `pass`, `worst_deviation=0`); the fixture products themselves are compared against the FP64-derived reference by the shared suite |

| `R` | Layout | `pad16(R)/pad16(I)/pad16(O)` | Workspace (bytes, alignment) | Accepted OID | Observed facility |
| ---: | --- | --- | --- | ---: | --- |
| 1 | ordinary | `16/16/16` | `1024, 32` | `36028797018963969` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 1 | head-planar | `16/16/16` | `1024, 32` | `36028797018963970` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 15 | ordinary | `16/16/16` | `1024, 32` | `36028797018963971` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 15 | head-planar | `16/16/16` | `1024, 32` | `36028797018963972` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 16 | ordinary | `16/16/16` | `1024, 32` | `36028797018963973` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 16 | head-planar | `16/16/16` | `1024, 32` | `36028797018963974` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 17 | ordinary | `32/16/16` | `2048, 32` | `36028797018963975` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |
| 17 | head-planar | `32/16/16` | `2048, 32` | `36028797018963976` | native WMMA kernel + sampled-facility probe, `facility_probe` `pass` |

The unsupported device is reported separately: the installed `gfx1036`
(ordinal 1) returns `Unsupported` for `BF16` from both the pure requirement
query and submission, and only the twenty scalar leaves stay available there.
`rocprofv3` 1.3.5 supports neither PC sampling nor SPM counter collection on
this `gfx1201` agent (`Given PC sampling configuration is not supported on
any of the agents`, `rocprofiler_iterate_agent_supported_counters failed ...
Agent HW architecture is not supported, no counter metrics found`), so the
facility statement above rests on the executed probe and the traced kernel
dispatches of the submitted OIDs rather than on a hardware instruction
counter.

The executed SYCL native `BF16` record of this revision is the following
observation, collected on the configured SYCL host from the SYCL native `BF16`
leaf's own task worktree at commit `ed92ff6`, which is this gate's revision,
and re-observed here by the gate's own runs of the same production route:

| Field | Observation |
| --- | --- |
| Backend, device | SYCL over Level Zero V2, `Intel(R) Arc(TM) Pro B60 Graphics`, architecture `intel_gpu_bmg_g21`, driver `1.15.38646+7`, subgroup sizes `16,32`, `ext_intel_matrix` present, PCI `8086:e211` |
| Toolchain | Intel oneAPI DPC++/C++ Compiler `2026.1.0`, `-ffp-model=precise`; `ocloc` `26.22.38646.7` |
| Evidence commands | The leaf's production-queue evidence driver (`record:` lines) invoked through `csw-remote-exec` with `csw-remote-sync` immediately before it and `flock -w 300` plus remote-side `timeout --kill-after=30s`; the gate then re-ran `./build/test/iom_sycl_conformance_tests` directly and through `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_(conformance\|smoke)_tests$'` |
| Layout, dimensions | `x[...,T,I]` and HF `w[O,I]`: `P=1, T=19, I=3, O=10, s=2`; ordinary `H=1, D=10`; head-planar `H=2, D=5`; `R in {1,15,16,17}` |
| Padded dimensions | `pad16(R)/pad16(I)/pad16(O)`: `16/16/16` for `R in {1,15,16}` and `32/16/16` for `R=17` |
| Scratch | alignment `32`; `1024` bytes for `R in {1,15,16}` and `2048` bytes for `R=17`, exactly the reported `A32(P*pad16(R)*pad16(O)*4)` |
| Kernel symbols | `LinearBf16MadKernel<16ul>`, `LinearBf16TailKernel<1ul>`, `LinearBf16TailKernel<16ul>`, and `LinearBf16PackKernel`, traced as `4/4/2/8` occurrences over the eight evidence submissions, one pack per output tile row |
| Facility | The selected device reports `ext_intel_matrix`, subgroup `16`, and the BF16/BF16/FP32 combination; the executed route is the subgroup-16 `joint_matrix` MAD path with an explicit RNE BF16 pack, not a host or elementwise substitute |
| ISA confirmation | Attempted and unavailable: `clang-offload-extract` on the built conformance binary yields ten `sycl-spir64` SPIR-V images, and `ocloc compile -spirv_input -device bmg` followed by `ocloc disasm` on each reports `dpas=0` and no `LinearBf16` symbol text (`ocloc 26.22.38646.7`), so the record rests on the traced dispatch and the device facts rather than on disassembly |

| `R` | Layout | `pad16(R)/pad16(I)/pad16(O)` | Scratch (bytes, alignment) | Accepted OID | Wait |
| ---: | --- | --- | --- | ---: | --- |
| 1 | ordinary | `16/16/16` | `1024, 32` | `36028797018963969` | `ok` |
| 1 | head-planar | `16/16/16` | `1024, 32` | `36028797018963970` | `ok` |
| 15 | ordinary | `16/16/16` | `1024, 32` | `36028797018963971` | `ok` |
| 15 | head-planar | `16/16/16` | `1024, 32` | `36028797018963972` | `ok` |
| 16 | ordinary | `16/16/16` | `1024, 32` | `36028797018963973` | `ok` |
| 16 | head-planar | `16/16/16` | `1024, 32` | `36028797018963974` | `ok` |
| 17 | ordinary | `32/16/16` | `2048, 32` | `36028797018963975` | `ok` |
| 17 | head-planar | `32/16/16` | `2048, 32` | `36028797018963976` | `ok` |

Per-`R` conclusion: `R=1`, `15`, `16`, and `17` are each supported on both
layouts; every submission returned its positive accepted OID and completed its
wait, and each is covered by the traced dispatch symbols above. The gate's own
re-run of the same production route at this revision holds the same result: the
filtered linear projection case passes `1/1` case with
`5,670,248/5,670,248` assertions including both crossing-factorization shapes,
the full target passes `29/29` cases with `5,905,415/5,905,415` assertions, and
`ctest` passes `2/2` including the smoke target, all with the Level Zero GPU
enumerated before the run.

## Native matrix evidence obligations

Native matrix work is evidenced by execution, not by inspection or analogy.
Every native BF16 implementation records the backend, the exact device, the
toolchain and profiler versions, the exact profiler command, the layout, `P`,
`T`, `I`, `O`, `H`, `D`, `s`, and `R`, the padded dimensions, the exercised
submission and its accepted OID, the kernel symbol, and the observed native
matrix instruction or facility. The record MUST connect the executed submission
to the matrix kernel; compile success, disassembly presence, or a bounded
capability sample alone is insufficient, and an unrun production port is
reported as unimplemented rather than supported. Supported or unsupported
conclusions are recorded separately for `R=1`, `15`, `16`, and `17`, and each
of those conclusions must come from real execution. CPU scalar, host,
elementwise, and emulated substitutes are never native evidence for CUDA, ROCm,
or SYCL. Accelerator verification follows the `csw-remote` procedure with
finite deadlines.

## Same-revision retained-backend gate evidence

The closing Linear-projections gate ran every retained backend's conformance
target and its direct binary at one revision, from one prepared task worktree,
and recorded the device, runtime, and toolchain identity of each run. The CPU
pair ran locally; every accelerator pair used exact-worktree `csw-remote`
sync/exec with a fresh sync immediately before each execution, remote-side
`timeout --kill-after=30s`, and a bounded hardware lock. The observed results
are:

| Backend | Commands | Device / runtime identity | Observed result |
| --- | --- | --- | --- |
| CPU | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_backend_conformance_cpu_tests$'` and the direct `./build/test/iom_backend_conformance_cpu_tests` | Local host CPU, C++20 release build | `24/24` cases, `5,934,262/5,934,262` assertions, `1/1` CTest pass |
| CUDA | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_cuda_conformance_tests$'` and the direct binary on remote `bv1` | `NVIDIA GeForce RTX 5090`, compute capability `12.0`, driver `595.71.05`, `nvcc` release `13.2` build `V13.2.78` | `31/31` cases, `5,954,449/5,954,449` assertions, `1/1` CTest pass |
| ROCm | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_rocm_conformance_tests$'` and the direct binary on remote `bv2` | `gfx1201` (`AMD Radeon AI PRO R9700`), HIP `7.15.26333-0000000`, AMD clang `23.0.0git` | `32/32` cases, `5,945,213/5,945,213` assertions at the recorded run (one other identical run reported `5,945,212`; no case failed in any run), `1/1` CTest pass |
| SYCL | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_(conformance\|smoke)_tests$'` and the direct binary on remote `bh2` | Level Zero V2 `Intel(R) Arc(TM) Pro B60 Graphics`, oneAPI DPC++/C++ `2026.1.0`, `ocloc` `26.22.38646.7` | `29/29` cases, `5,905,415/5,905,415` assertions, `2/2` CTest pass including the smoke target |

CPU, CUDA, ROCm, and SYCL therefore demonstrate all twenty-one applicable
leaves at this revision. No planned, rejection-only, storage-only,
compile-only, or unsupported probe is counted as a pass anywhere in that
table.

**Crossing-factorization obligation.** The two additive head-planar crossing
cases (`H=2 D=9 O=18` and `H=4 D=5 O=20`, `T=19`, `s=2`, `R=17`, every
declared leaf) exist because the earlier fixture set's only head stride,
`H=2 D=5`, never crossed a packed 16-column boundary and therefore left a real
head-planar scatter defect invisible. This gate re-verified every backend
against the independent FP64 oracle with those cases included; a filtered run
of the shared linear projection case with successful-assertion reporting
prints each case's label in its observed comparison contexts, so the cases are
observably executed rather than skipped:

| Backend | Crossing-case contexts (`H=2 D=9 O=18` / `H=4 D=5 O=20`) | Filtered linear case result |
| --- | --- | --- |
| CPU | `39,234` / `43,710` | `1/1` case, `5,676,263/5,676,263` assertions |
| CUDA | `39,045` / `43,521` | `1/1` case, `5,670,287/5,670,287` assertions |
| ROCm | `39,045` / `43,521` | `1/1` case, `5,670,297/5,670,297` assertions |
| SYCL | `39,045` / `43,521` | `1/1` case, `5,670,248/5,670,248` assertions |

No backend reported a crossing-case mismatch, so no case, tolerance, or
declaration was weakened or re-scoped at this gate. The `R=1`, `15`, `16`,
and `17` native-facility records above remain the per-backend execution
records for those logical runs.
## Implementation references and delivery prerequisites

Implementers need these existing sources and seams:

- public facade, declarations, and default hooks: `include/iom/iom.hpp` and
  `src/device_ops_neural.cpp`;
- admission, request snapshots, owner registration, FIFO acceptance, deferred
  completion, repeated waits, and quarantine: `src/device_ops.cpp` and
  `src/device_ops_binary.cpp`;
- checked logical shapes, slots, and leading-plane arithmetic: `src/tensor.cpp`
  and `src/iom_internal.hpp`;
- workspace owner bytes, 32-byte subranges, leases, proof, and quarantine:
  `src/workspace.cpp` and `include/iom/detail/workspace_registry.hpp`;
- standard tiled word ownership, packed sub-byte writers, and launch mapping:
  `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_add.inl`,
  `src/shared/gpu_queue.hpp`, and `src/shared/gpu_queue_operations.inl`;
- CPU asynchronous worker and raw codecs: `src/cpu/queue.cpp` and
  `src/cpu/transfer_helpers.hpp`;
- backend capability classification and queues:
  `src/cuda/copy.hpp`, `src/cuda/copy.cu`, `src/rocm/copy.hpp`,
  `src/rocm/copy.hip`, `src/sycl/queue_internal.hpp`, and `src/sycl/queue.cpp`;
- independent reference and conformance harness:
  `test/backend/backend_conformance_oracle.hpp`,
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_other.hpp`, and the four retained
  `test/<backend>/test_<backend>_conformance.cpp` drivers.

`test/backend/backend_conformance_linear.hpp` is the shared conformance header
described above. The per-backend linear kernels, launch wrappers, and
capability predicates are owned by their own ports, which replace only their
own hooks and migrate their own `Unsupported` probe as each declared leaf is
actually implemented. The executed ROCm and SYCL native `BF16` records and
the retained-backend gate are observations at their named revisions; nothing
here extends a recorded result to an unexercised shape, device, leaf, or
revision.

# TinyLlama forward layout — Positions and cache boundaries

The common layer now declares the RoPE facade and freezes its
backend-neutral admission contract. It does not add a kernel or claim positive
support: until a backend leaf replaces the protected hooks, a valid request
returns `Unsupported` and the pure query throws the same category.

```cpp
oid rope(
        const TensorView& x, TensorView& out, std::size_t a, double theta,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rope_workspace_requirements(
        const TensorView& x, const TensorView& out, std::size_t a,
        double theta);
```

The RoPE facade and its query have exactly the same semantic arguments in
the stated order. `a` is an explicit absolute position and `theta` is an
explicit runtime scalar; neither is inferred from a session, cursor, cache,
or model state.
`x` and `out` MUST have identical logical shape `[...,H,R,D]`, rank three
through eight, identical nonzero leading dimensions, `H > 0`, `R > 0`, and
positive even `D`. In particular, an odd `D` for RoPE is `InvalidArgument`.
Every logical output element is written from its corresponding input under the
two views' independent valid offset and leading-stride mappings. Tile padding
and uninitialized physical slots are not logical values. Q and K are separate
calls, so unequal Q/K head counts are valid and there is no cross-request Q/K
alias category.


For every leading coordinate tuple `b`, head `h`, run index `r`, and
`0 <= j < D/2`, the frozen split-half equation is:

```text
angle = (a + r) * theta^(-2*j/D)
y[b,h,r,j]       = x[b,h,r,j]       * cos(angle)
                   - x[b,h,r,j+D/2] * sin(angle)
y[b,h,r,j+D/2]   = x[b,h,r,j+D/2] * cos(angle)
                   + x[b,h,r,j]       * sin(angle)
```

The implementation MUST pair the first and second halves, never adjacent
elements, and MUST use the explicit `a+r`. It adds no scaling, position
broadcast, reset, cursor, cache update, transpose, head packing, or alternate
angle convention. Admission accepts only finite `theta` in
`[1, std::numeric_limits<float>::max()]`. Checked arithmetic MUST compute
`a + R - 1` and require the result to be no greater than `2^24 - 1` before
any queue, owner, output, or backend effect. `a` is `std::size_t`, so a
position-range overflow is `std::overflow_error`; a representable position
above the bound is `std::invalid_argument`.

At position zero, the encoding is an exact bitwise identity for every
admitted leaf, including signed-zero representations and all nonfinite
payloads. At nonzero positions, the eight non-F64 leaves use binary32
exponent, frequency, angle, sine, cosine, and pair intermediates. Each pair
uses separate noncontracted multiply operations followed by the add/subtract;
accidental FMA/contraction and fast-math are forbidden. Exactly one existing
named-format round-to-nearest-even/saturation encode stores each destination
value. `F64` uses binary64 exponent, frequency, angle, trigonometry, and
intermediates throughout and is never narrowed.

The nine semantically applicable ordinary signed floating leaves are exactly
`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
`F32`, and `F64`. The fourteen recognized but semantically inapplicable leaves
are exactly `BOOL`, `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`,
`U32`, `I64`, `U64`, and `F8_E8M0`. RoPE has no boolean/integer trigonometric
result, and exponent-only `F8_E8M0` cannot represent a general signed
rotation. Unknown enum values are `InvalidArgument`; only
`QuantizationFormat::NONE` is admitted. Recognized non-`NONE` formats and the
fourteen inapplicable leaves return `Unsupported` only after every earlier
shape, device, alias, parameter, range, overflow, and metadata check.

Admission MUST NOT scan tensor values. IEEE nonfinite classes follow the
written expression, including infinity-times-zero becoming NaN; nonzero
position NaN payload equality is not promised, and NaN comparisons are by
class. Finite acceptance uses an independent high-precision reference and
fixed destination tolerances: each of the eight non-F64 leaves permits at
most one destination ULP plus
`2^-9 * (abs(x_first) + abs(x_second))`; `F64` permits at most one
destination ULP plus `2^-44 * pair_norm`, where `pair_norm` is the
corresponding pair magnitude. Analytic and golden coefficients are pinned to
`mpmath 1.3.0` at 100 decimal digits in generated/static fixtures only;
production code has no runtime Python dependency.

Both facades perform the same host-side checks before capability dispatch,
snapshot allocation with externally visible effects, sequence reservation,
owner registration, workspace lease, queue submission, native metadata
transfer, or data access, in this order:

1. Validate both full specs, rank three through eight, all nonzero extents,
   positive even `D`, and identical `[...,H,R,D]` shape.
2. Validate exact queue-device identity, non-null/live owner registrations and
   native handles, value-consistent owner metadata, and checked plane,
   leading-stride, and view bounds.
3. Validate identical leading tuple/final axes and every checked shape, tile,
   element, byte, stride, address, and `a + R - 1` arithmetic, including
   the position conversion bound.
4. Reject any input/output owner identity overlap and any conservatively
   overlapping complete owner storage or transformed view range. Input and
   output owners MUST be distinct; in-place encoding is not promised.
5. Validate finite `theta`, its `[1,float_max]` bound, and all checked
   exponent/frequency/angle and narrowing intermediates.
6. Validate matching dtype and quantization, then classify the recognized
   dtype and quantization capability.

`rope_workspace_requirements` is pure and deterministic. After the same
admission validation it returns exactly `{0,1}` for an admitted supported
capability and performs no allocation, retained request construction, owner
registration, lease acquisition, token/sequence reservation, queue
submission, native metadata upload, data read/write, or queue/session
mutation. Its result is independent of queue occupancy, allocator state,
registration state, and prior submissions. Malformed and overflow input
propagates as `std::invalid_argument` or `std::overflow_error`; unsupported
capability propagates as `UnsupportedOperation`, never as an OID.

Submission repeats admission, obtains fixed-capacity value-copied
`RopeViewSnapshot`s and an immutable `RopeRequest`, and accepts only an empty
`RawWorkspaceView`. A supplied nonempty workspace is `InvalidArgument` and is
never silently ignored on an admitted supported path. Because the common
requirement is zero, no positive workspace, angle cache, caller scratch,
hidden tensor allocation, host round trip, or cursor exists. The request
retains exact owner/native identities and queue metadata through the existing
prepared registration and rollback mechanics until proven in-order
completion. Any setup failure rolls back every earlier registration and lease.
Admission failures map through `invoke_failure` to established negative OIDs,
consume no sequence/token, submit no work, and leave output bytes, owner
registry, workspace state, and queue/session state unchanged.

A default `UnsupportedOperation` hook is a pre-acceptance failure after common
admission: `rope` returns negative `OidError::Unsupported` with no accepted
token, registration, output mutation, or retained request. A positive OID is
never replaced by a later negative result. An accepted runtime/device failure
may leave output indeterminate or partial, is retained and rethrown
identically on every repeat wait, and prohibits output/session reuse until the
failure is drained under existing queue rules; no rollback of already-written
device data is promised.

The eventual backend matrix is explicit but is not a support claim in this
common leaf:

| Backend | Applicable leaves | Explicitly rejected leaves |
| --- | --- | --- |
| CPU | all nine ordinary signed floating leaves | the fourteen inapplicable leaves |
| CUDA | all nine ordinary signed floating leaves | the fourteen inapplicable leaves |
| ROCm | all nine ordinary signed floating leaves | the fourteen inapplicable leaves |
| SYCL | the eight non-F64 applicable leaves | `F64` and the fourteen inapplicable leaves |

Backend branches consume only their own frozen producer outputs and add no
common backend-kind switch or capability registry. Every later port preserves
this ABI, validation order, zero-workspace result, aliases, absolute
positions, arithmetic, nonfinite behavior, and OID/lifetime contract.
Calls with `R=1` and `a=1,15,16,17`, and multi-row calls spanning those
positions, use consecutive absolute positions with no tile-boundary reset,
skipped position, or reused cursor. The common leaf owns no cache/session
initialized length.

## Cache append

The complete cache-append public surface is exactly:

```cpp
oid cache_append(
        const TensorView& source, TensorView& destination, std::size_t a,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements cache_append_workspace_requirements(
        const TensorView& source, const TensorView& destination,
        std::size_t a);
```
The repository header uses the `std::size_t` spelling for the required
`size_t` parameter type; the declarations above match that exact public ABI.

A successful cache-append requirements query is not a runtime support result;
it reports only the deterministic requirements of the validated request.

For cache append, `source` MUST have shape `[...,H,R,D]` and `destination`
MUST have shape `[...,H,C,D]`. Both operands MUST have operation rank `3..8`
and every extent MUST be nonzero. Their complete leading tuples, `H`, `D`,
dtype, quantization, and exact device MUST match; `R` is the independent
source length and `C` is cache capacity. Leading planes are complete
coordinates, never broadcast or implicitly shared.

This operation-specific rank rule is narrower than the general
`TensorView` and other-facade rank rule of `2..8`, which remains unchanged
elsewhere in this document. Cache append does not inflate a singleton output
rank or broadcast any leading plane.

Admission first validates structurally readable views, operation rank and
extents, live owner/device identity, transformed leading mappings, and the
checked view, stride, and plane-address metadata needed to obtain `C` and
`R`. These structural checks precede the cache-window checks and every
observable effect. Within the operation-specific window check, check `a <= C`
first; only then check `R <= C-a`. After those checks, perform append-specific
checked element, byte, storage-range, and workspace arithmetic. All malformed
view, range, and arithmetic checks MUST finish before owner registration,
sequence consumption, output mutation, or submission. The subtraction form is
required so that an out-of-range `a` is rejected without forming an
overflowing `a+R`.

The only logical write is exactly

```text
destination[b,h,a+r,d] = source[b,h,r,d]
```

for every complete leading coordinate `b`, head `h`, `0 <= r < R`, and
`0 <= d < D`. Every other logical destination row and every physical padding
cell in every affected or unaffected plane MUST remain unchanged. The copy is
opaque and bit-preserving for all 23 existing payload leaves: it performs no
arithmetic, conversion, re-encoding, or rounding, and it does not read
uninitialized logical rows or padding as initialized data. Independent
transformed leading offsets and strides MUST be honored for both operands,
including partial final tiles.

The boundary cases are normative and MUST be observable with independent
offsets and row lengths `1`, `15`, `16`, and `17`. With exact-end capacity
`C=a+R`, the cases `(a,R,C)=(1,1,2)`, `(15,15,30)`, `(16,16,32)`, and
`(17,17,34)` write exactly rows `[a,C)` and preserve every row `[0,a)`.
There is no special behavior at any of those row lengths or at positions
`15`, `16`, and `17`. With capacity larger than `a+R`, rows `[0,a)` and
`[a+R,C)` and all physical padding remain untouched. `a>C`, `R>C-a`, or
overflow in any position or storage range is rejected before mutation.
Exact-end append is valid, while padding after a partial tile and every
uninitialized logical row remain excluded from initialized length.

Source/destination overlap MUST be rejected, including overlap discovered
through transformed views, intersecting backing ranges, or identical native
handles. Read/read overlap elsewhere is harmless when all shape rules hold.
No operand, output, or temporary tensor may be allocated, relocated, silently
converted, or moved through hidden or unaccounted host staging. Standard tiled
paths write directly and require no hidden or unaccounted staging.

## Cache append feasibility

The following concise table records implementation feasibility, not current
runtime support. A port MUST advertise only the leaves it has implemented and
MUST return `Unsupported` for a recognized but unported capability.

| Backend path | Storage leaves | Query workspace and required route |
| --- | --- | --- |
| CPU / CUDA / ROCm / SYCL (standard tiled) | support all 23 storage leaves | zero workspace (`{0, 1}`); direct existing tiled writes through the 16x16 mapping |

The SYCL standard-tiled path uses the exact-device in-order queue and one
native `parallel_for` over destination words after the fixed immutable
descriptor upload. It advertises all 23 opaque leaves, including `BF16` and
`F8_E8M0`, with `QuantizationFormat::NONE`, reports `{0, 1}`, and performs no
payload staging, caller-workspace lease, or host payload round trip. A missing
Level Zero GPU or another unsupported SYCL runtime is reported through the
common unsupported/resource-exhaustion categories; this path is never emulated.

The ROCm cache-append port is implemented for all 23 standard opaque storage
leaves, including `BOOL`, packed 2/4/6/8-bit fields, `BF16`, and raw `F64`.
Its policy is a direct HIP raw-word kernel on the queue's existing nonblocking
stream with the exact `{0, 1}` requirement: it performs no host staging,
conversion, hidden allocation, or numerical/matrix operation. Installed HIP
compiler and device properties are checked facts; hardware/runtime behavior is
claimed only from the configured `csw-remote` ROCm build and conformance
evidence, not inferred from this capability statement.

BF16 is mandatory on CPU, CUDA, ROCm, and SYCL, and `QuantizationFormat::NONE`
is the only applicable quantization format. No route may zero storage, convert
or re-encode payloads, allocate hidden workspace, or use native partial-row or
partial-matrix evidence in place of the complete logical mapping. Workspace
synchronization MUST be proven before its lease is released. Hidden or
unaccounted staging remains forbidden.

## CUDA cache-row append implementation boundary
CUDA's cache-row append leaf is implemented in `src/cuda/copy.cu` through the
common `DeviceOps::cache_append` admission and queue path. It launches one
direct device kernel over destination packed words, with one writer per word,
the existing 16x16 tiled plane mapping, transformed leading-plane offsets and
strides, and checked 64-bit metadata. Logical rows `[a,a+R)` are copied
without conversion; untouched rows, tile padding, and unrelated leading
planes remain read-modify-write preserved. The CUDA policy advertises all 23
storage leaves, including BF16, for `QuantizationFormat::NONE` and requires
the frozen zero-workspace query `{0,1}`.

The CUDA conformance driver exercises this leaf through the independent cache
append reference, transformed-view, ordering, workspace, owner-lifetime, and
accepted-failure cases, including launch and event-record failure retention.
Those runtime results require the configured CUDA profile; this implementation
statement is not a substitute for execution-connected hardware evidence.


## Cache append queue, session, and lifetime boundaries
K and V MUST use separate `cache_append` submissions with distinct cache
owners; there is no atomic two-cache call. Cache append owns no initialized
length, reset, clearing, growth, or session failure policy. The session MAY
publish `initializedL = a + R` only after both append OIDs have successfully
completed their waits. Positive admission alone is insufficient. If either
accepted append fails, the session MUST publish no new initialized length and
MUST submit no dependent consumer; it MUST NOT expose uninitialized cache
capacity or physical padding as a readable prefix. Attention receives the
initialized length explicitly and MUST NOT infer it from capacity or padding.

All cache destination and workspace storage MUST be disjoint from every input
and from each other. Cache source/destination overlap is rejected under the
transformed-view and native-handle rules above. Workspace is caller-owned
through proven completion; the supplied range is checked at submission for the
exact device, the query's exact required size and alignment, overlap,
freshness, and lease availability. The pure query itself does not inspect the
supplied workspace or mutate queue state. No operand, output, or temporary
tensor may be allocated, relocated, or silently converted, and cache append
introduces no hidden or unaccounted host round trip.

## Cache append admission, errors, and snapshots

The cache query and submission MUST validate malformed views, operation rank
and dimensions, complete leading tuples, dtype, exact device, quantization,
aliases, ranges, workspace, and checked arithmetic before effects. A malformed
request maps to `InvalidArgument=-1`; a recognized unsupported dtype or
capability maps to `Unsupported=-2` only after those earlier checks;
checked arithmetic or sequence exhaustion maps to `Overflow=-3`; bounded
resources or lease failure map to `ResourceExhausted=-4`; a
pre-acceptance runtime failure maps to `DeviceError=-5`; and any other
failure maps to `InternalError=-6`. A pre-acceptance failure mutates no
output, registers no owner, and consumes no sequence. Once accepted, a
positive OID remains positive; an accepted completion failure is retained
and rethrown by every repeated wait.

Cache append performs no device-data out-of-vocabulary or nonfinite-policy
scan because every payload leaf is opaque. If an accepted device/runtime
failure is reported, the affected cache output is unusable, the session is
poisoned, and every wait for that OID MUST retain and rethrow the same
failure; the session MUST NOT publish a new initialized length or submit a
dependent consumer.

Submission snapshots tensor specs, exact device/owner/native-handle identity,
independent leading offsets and strides, scalar values including `a`, and the
workspace range. It retains the operand owners and workspace lease through
proven completion, but never retains borrowed `TensorView` objects beyond
submission.

## Cache append delivery and coverage
Future delivery order is contract, independent reference, and four-backend
feasibility, followed by CPU, CUDA, ROCm, and SYCL closure for RoPE; only then
does cache append follow the same sequence. Each operation's four-backend gate
MUST close before the next operation begins. CPU feasibility is scalar or wide
arithmetic over existing tiled storage; the table above makes no current
support, native accelerator, or profiler claim. The operation-specific
cache-row append implementation owns detailed capability, numerical,
snapshot/hook, and native evidence.

Its shared regression coverage MUST extend the existing byte-level
copy/storage suite and independent physical oracle for applicable payload
leaves, transformed leading planes, partial tiles and padding, aliases,
overflow, no-side-effect admission failures, accepted failures and repeat
waits, and the exact `1/15/16/17` offsets and row-length cases.

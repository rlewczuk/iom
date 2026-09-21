# TinyLlama forward layout — Embedding and projection boundaries

This subsection freezes the backend-neutral boundaries and records the
implemented embedding ports. The embedding methods below are declared and
admitted by the current neural facade: common structural, device, view, shape,
alias, checked-arithmetic validation, and the pure requirement query are
implemented, while CPU, CUDA, ROCm, and SYCL provide their operation hooks.
Each retained backend remains explicitly capability-gated according to its own
port state.
The normative `linear` surface is exactly the frozen `LinearOutputLayout` form
below, owned by [Linear projections](linear-projections.md#linear-projections): there is no second
`linear` declaration, and the earlier three-view `linear(x, w, y)` facade is
not retained as an overload, alias, shim, or re-export. Until a backend port
lands, a well-formed request for that frozen ABI returns `Unsupported` as
specified above.
The operation-owned [Embedding lookup](embedding-lookup.md#embedding-lookup) and
[Linear projections](linear-projections.md#linear-projections) sections and their backend gates
publish the exact operation ABI and remain the normative source for per-backend
capability, workspace, status, and failure policy. The embedding declarations
and the frozen `LinearOutputLayout` and `linear` methods are exactly:

```cpp
enum class LinearOutputLayout { ordinary, head_planar };

oid embedding(
        const TensorView& table, const TensorView& indices, TensorView& out,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements embedding_workspace_requirements(
        const TensorView& table, const TensorView& indices,
        const TensorView& out);

oid linear(
        const TensorView& x, const TensorView& w, TensorView& out,
        size_t s, size_t R, LinearOutputLayout layout, size_t H, size_t D,
        RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements linear_workspace_requirements(
        const TensorView& x, const TensorView& w, const TensorView& out,
        size_t s, size_t R, LinearOutputLayout layout, size_t H, size_t D);
```

Both requirement queries receive the same semantic arguments as submission,
including a const output view, but no workspace. They are pure: they allocate
nothing and perform no registration, submission, queue/OID consumption, or
state change. They validate operands, shapes, scalar parameters, aliases,
capability, and checked arithmetic, then return `WorkspaceRequirements` or
throw the established validation exception. Exact-device, size, alignment,
overlap, staleness, and lease checks on a supplied workspace occur only during
submission. The `noexcept` submission facades instead map synchronous failures
to OID errors and otherwise return a positive accepted token.
Synchronous rejection has no output effect, registration, sequence
consumption, or accepted token. A failure after acceptance remains attached to
its positive token and is rethrown on every repeat wait.

For embedding, the table is rank-two `[V,F]` and is shared unchanged by every
independent leading plane. `indices[...,1,R]` and `out[...,R,F]` have the same
leading tuple and each has rank 2 through 8. There is no implicit
index-to-output plane broadcast and no singleton output-rank inflation. With
`b` denoting the complete leading tuple, the mapping is

```text
out[b,r,f] = table[indices[b,0,r],f].
```

Runtime shapes, never checkpoint constants, determine `R`, `V`, and `F`.
Indices use the caller-selected integral tensor index type. A signed negative
index or an index at least `V` is invalid. A value discoverable only in device
storage may instead fail after acceptance as a device-data error; an
implementation MUST NOT introduce a host round trip or hidden host-side index
scan. The selected table payload bits are copied unchanged, with no numeric
conversion or rounding. Index representation, applicable leaves, fixtures,
tolerances, workspace and status protocol, ownership and failure rules, and
backend support are owned by [Embedding lookup](embedding-lookup.md#embedding-lookup).

For linear projection, `x` is `[...,T,I]` and rank-two `w[O,I]` is shared
unchanged across all independent leading planes. This is the Hugging Face
`[out,in]` weight orientation: output coordinate `o` selects weight row
`w[o,*]`; the boundary never transposes a checkpoint weight. Ordinary layout
produces `out[...,R,O]` and requires `H=1,D=O`. Head-planar layout produces
`out[...,H,R,D]` and requires the checked equality `O=H*D`. Input and output
leading tuples must match after excluding the inserted head axis; the
rank-two weight's empty leading tuple is not matched and does not imply state
broadcast. The final output rank cannot exceed 8. Thus an ordinary rank-8
output is structurally valid, while adding a head axis that would make a
rank-9 head-planar output is `InvalidArgument`.

The ordinary and head-planar equations are, respectively,

```text
out[b,r,o]   = sum_i x[b,s+r,i] * w[o,i]
out[b,h,r,d] = sum_i x[b,s+r,i] * w[h*D+d,i].
```

Here `T` is the source row extent, `s` is its starting row, and `R` is the
independently requested row count; `R` is neither inferred from `s` nor
confused with `T` or `O`. Likewise `O` is the weight's output-row extent and
equals `H*D` only after checked validation in head-planar mode.

The normative non-square case is
`I=3,O=10,H=2,D=5,T=19,s=2,R=17`. It selects exactly `x[2]` through
`x[18]`. Ordinary output columns `o=0..9` consume `w[0]` through `w[9]`.
Head-planar `(h=0,d=0..4)` maps to `o=0..4`, and
`(h=1,d=0..4)` maps to `o=5..9`. With all other values unchanged, `R=1`
selects `x[2]`, `R=15` selects `x[2..16]`, and `R=16` selects
`x[2..17]`. These cases make a checkpoint transpose, an inferred or
off-by-one row window, and an `h`/`d` shuffle observably wrong.

The final untied LM head uses ordinary mode with `s=T-1`, `R=1`, `H=1`, and
`D=V`. For one sequence it writes `[1,V]` logits directly into the supplied
output; it requires no final-axis `TensorView` slice. Projection rearranges
only newly computed selected rows. It MUST NOT materialize repeated KV heads,
make a persistent host copy or transpose of checkpoint weights, or expose a
public transpose, head-pack, or row-extraction operation.

All ranks remain in 2 through 8 and all dimensions are nonzero. Embedding and
linear validate, in the common precedence and before effects: well-formed
views, ranks, shapes, leading mappings, exact device ownership, leaf type and
quantization; scalar and range rules; aliasing and capability; then checked
element, stride, byte-range, output-size, and other shape products. Binary and
embedding share one bounded, allocation-free checked-view admission path for
the common structural facts: recognized encodings, rank and nonzero extents,
exact live owner and stable native handle, leading-only view metadata,
selected-plane bounds, and checked plane, tile, element, bit, and byte
arithmetic. It allocates, snapshots, registers, and leases nothing, and
shaping, capability, alias, and workspace policy stay with each operation.
Linear additionally validates a recognized `layout`, `s <= T`, `R > 0`,
`R <= T-s`, `H > 0`, and `D > 0`; head-planar mode checks `H*D` for overflow
before comparing it with `O`, while ordinary mode requires `H=1,D=O`.
Logical mapping excludes all 16x16 tile padding and honors every transformed
leading offset and stride independently.

The rejection results are normative:

| Condition | Result |
| --- | --- |
| malformed view, rank outside 2..8, zero dimension, mismatched leading plane, invalid final shape, or invalid row window | `InvalidArgument` |
| signed negative index or index `>= V` | `InvalidArgument`, or an accepted device-data failure when the value is discoverable only on device |
| unsupported recognized dtype, device, or quantization | `Unsupported` after earlier validation |
| invalid `LinearOutputLayout` value | `InvalidArgument` |
| output/input or workspace overlap, including forbidden transformed-owner overlap | `InvalidArgument` |
| rank growth beyond 8 | `InvalidArgument` |
| checked element, stride, byte-range, output-size, or `H*D` overflow | `Overflow` |
| bounded-resource, runtime/device, or internal submission failure | `ResourceExhausted`, `DeviceError`, or `InternalError` |

Inputs may overlap other inputs: valid read/read overlap and exact aliases are
not rejected merely because both accesses are reads. New output and workspace
storage must each be disjoint from every input and from each other; these
gather/projection operations reject every output/input overlap. Workspaces are
caller-owned reusable scratch subject to the shared exact-device, nonoverlap,
size, alignment, and lifetime rules. The operations allocate, relocate, or
silently convert no operand or output. Submission snapshots metadata,
registers distinct owners through the existing queue protocol, and retains
the workspace lease through proven completion; it never retains borrowed
`TensorView` objects beyond submission.

Every result is stored in the caller-provided output leaf. Embedding preserves
payload bits; each distinct projection result is rounded and stored at the
operation boundary according to [Linear projections](linear-projections.md#linear-projections).
BF16 is mandatory for CPU, CUDA, ROCm, and SYCL when those ports are
implemented. The operation-owned sections classify datatype leaves, special
values, numerical references, tolerances, fixture provenance, and backend
feasibility. Delivery order is contract/reference/four-backend feasibility,
then CPU, CUDA, ROCm, and SYCL, closing the four-backend gate before the next
operation. This planned boundary makes no kernel, profiler, or native-backend
conformance claim.

Conformance records must distinguish shape failures, plane/leading mismatch,
invalid row windows, dtype/device/quantization support, output/workspace
overlap, rank growth, and checked arithmetic overflow. They cover independent
leading planes, transformed mappings, logical padding exclusion, and the
`1/15/16/17` row cases. Unsupported probes migrate only when their operation
is implemented; an unsupported port never counts as numerical conformance.

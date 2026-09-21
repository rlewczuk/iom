# TinyLlama forward layout — Causal grouped-query attention

This subsection freezes the TinyLlama caller boundary for the operation-owned
[Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention) section. It does
not add declarations or implementation. CPU, CUDA, ROCm, and SYCL all accept
the current BF16 leaf through this facade at this revision; the other eight
floating leaves and every other backend/leaf pair remain `Unsupported` before
submission, output mutation, or token acceptance. The four-backend closure
record is in [Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention).

The clean-cutover target has exactly the following submission and pure
workspace-query signatures:

```cpp
oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
         TensorView& out, size_t a, size_t L,
         RawWorkspaceView workspace={}) noexcept;
WorkspaceRequirements sdpa_workspace_requirements(
        const TensorView& q, const TensorView& k, const TensorView& v,
        const TensorView& out, size_t a, size_t L);
```

There are no head-count or head-dimension scalar arguments. The query has the
same semantic arguments as submission, makes `out` const, accepts no
workspace, returns only deterministic capacity/alignment requirements, and
throws the standard validation exceptions. It is pure: it allocates nothing,
registers no owner or workspace, consumes no queue sequence or native credit,
submits no work, and does not depend on queue occupancy or free capacity.

For one common, possibly empty, leading tuple `B`, shapes are
`q[B...,Hq,R,D]`, `k[B...,Hkv,C,D]`, `v[B...,Hkv,C,D]`, and
`out[B...,R,Hq*D]`. Derive `Hq`, `R`, and `D` from `q`; `Hkv` and `C` from
the matching `k`/`v` axes. The Q/K/V ranks are 3 through 8 and the merged
output rank is 2 through 7. Every dimension is nonzero; in particular
`Hq>0`, `Hkv>0`, `D>0`, and `C>0`. The K and V shapes must match, every
leading tuple must be identical, `Hq % Hkv == 0`, `Hq*D` must be checked,
and the output shape must match exactly. There is no cache, state, or
leading-plane broadcast. For each leading coordinate `b`, the head merge is
exactly

```text
out[b,r,h*D+d]
```

for `0<=r<R`, `0<=h<Hq`, and `0<=d<D`. Logical elements use the shared
16x16 tiled mapping and independent leading-plane mappings; every backend
physical representation must preserve these logical coordinates and tail rules.

The scalar range contract is `0<L<=C`, `a<C`, and `R<=C-a`; implementations
must perform the last check by subtraction rather than by first forming
`a+R`. For query row `r`, only the nonempty initialized causal set

```text
T(r) = { t : 0 <= t < L and t <= a+r }
```

is logical input. Generic SDPA deliberately permits `L<a+R`: rows whose
absolute positions reach or pass `L` attend to all and only the shorter
initialized prefix. The session specialization calls SDPA only after both
successful cache appends and sets `L=a+R`. Thus a cached/chunked row `r`
uses absolute position `a+r` and the same keys `0..a+r` as that row in a
full-sequence causal evaluation; full prefill is the `a=0`, `L=R` case.

Let `G=Hq/Hkv` and map query head `h` to KV head
`g(h)=floor(h/G)`. Independently for every leading coordinate, compute the
following target equations:

```text
S[h,r,t] = (sum(d=0..D-1, q[h,r,d] * k[g(h),t,d])) / sqrt(D)
m[h,r]   = max(t in T(r), S[h,r,t])
Pfp[h,r,t] =
    exp(S[h,r,t] - m[h,r])
    / sum(u in T(r), exp(S[h,r,u] - m[h,r]))
Pfp[h,r,t] = 0 exactly when t is masked
Pbf[h,r,t] = BF16_RNE(Pfp[h,r,t])
out[r,h*D+d] =
    BF16_RNE(sum(t in T(r), Pbf[h,r,t] * v[g(h),t,d]))
```

Q and K operands are BF16, their products accumulate in FP32, and the
`1/sqrt(D)` scale and stable max-subtracted softmax are FP32. Probabilities
are rounded RNE to BF16 before the BF16-probability/BF16-V PV product and
FP32 accumulation. That probability store and the final RNE BF16 merged
store are observable numerical boundaries; fusion may not erase either one.
The Scaled dot-product attention operation owns detailed special-value policy,
datatype applicability, tolerances, reference implementation, fixture
provenance, snapshots/hooks, backend support, and kernels, using these same
equations and rounding boundaries.

Excluded logical K or V values must not be read or participate in output,
reductions, or nonfinite checks. In particular, an implementation must not
turn an excluded V row into `0 * NaN`. Physical tensor padding is masked
separately from the causal and initialized-prefix masks. A native kernel may
read explicitly initialized neutral padding in caller-owned packed scratch,
but that padding is not a logical token and cannot change a logical result.

For the independent worked case
`Hq=4,Hkv=2,D=3,a=2,R=2,L=4,C=7`, `G=2`: query heads 0 and 1 map to KV
head 0, while heads 2 and 3 map to KV head 1. The exact permitted sets are
`T(0)={0,1,2}` and `T(1)={0,1,2,3}`. Independently perturbing K row 3 or V
row 3 cannot change output row 0, because that initialized future row is
causally masked there. Independently perturbing any K or V row in the
uninitialized capacity tail `4..6` cannot change either output row. The same
invariances apply per leading plane without cross-plane broadcast.

Output and workspace storage must each be disjoint from Q, K, and V and from
one another. Exact read-only aliases, and harmless read/read overlap, among
Q/K/V are allowed when every aliased view independently satisfies its shape
contract; validation must not ban them. The supplied workspace must have a
live owner on the exact device, meet the query's byte capacity and alignment,
not overlap operand or output storage, and not conflict with an outstanding
lease. Output and scratch remain caller-owned. Device-local packing and head
merge may use only that scratch; repeated KV-head materialization is not
required. The operation allocates or relocates no operand, output, or
temporary tensor and performs no hidden host round trip. It introduces no
public softmax, transpose, packing, cache, or session-mutation helper.

Both query and submission validate operand/view metadata, nonzero dimensions,
ranks, exact shapes and leading tuples, exact device identity, dtype and
quantization, read/write alias rules, mode-independent `a` and `L`, workspace
device/capacity/alignment/overlap, and checked `Hq*D`, range, element, byte,
stride, and address arithmetic before owner registration, sequence
consumption, output mutation, or submission. Malformed shape, device, alias,
or range is `InvalidArgument`; checked arithmetic is `Overflow`; a recognized
but unsupported dtype or capability is `Unsupported` only after those earlier
checks. Bounded-resource, runtime/device, and otherwise unclassified failures
map to `ResourceExhausted`, `DeviceError`, and `InternalError`. The `noexcept`
facade returns negative OIDs for synchronous failures. Accepted failures stay
observable on every repeated wait, while a pre-submit failure changes no
output and consumes no token. Submission snapshots metadata and never retains
a borrowed `TensorView` beyond the call.

Delivery order is contract and independent reference, four-backend native
feasibility, then CPU, CUDA, ROCm, and SYCL closure; SDPA is final among the
seven missing TinyLlama operations. The shared CPU scalar/wide baseline over
existing tiled storage is correctness coverage, not an accelerator claim.
Native evidence must cover both QK and PV for prefill and logical
`run=1`, not merely one GEMM, and excludes host computation or round trips,
elementwise substitutes, and padded extra logical tokens. Unsupported
hardware must be reported, and this design subsection makes no native kernel
or profiler-conformance claim.

Future shared conformance covers runs `1/15/16/17`, non-tile dimensions and
features, independent leading planes, padding/tail isolation, causal prefill,
cached decode, generic shorter initialized prefixes, every applicable
dtype/backend combination, wrong rank/empty-key/shape/GQA ratio, alias,
device, workspace, range and overflow rejection, and accepted-failure
repeated waits. An unsupported port is not numerical conformance. That shared
suite now runs on all four retained backends and closes the four-backend SDPA
gate in [Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention); SDPA
closure was the last missing operation prerequisite, so the session component
may claim complete decoder-layer assembly and verification against these
published boundaries without this design subsection adding session work.

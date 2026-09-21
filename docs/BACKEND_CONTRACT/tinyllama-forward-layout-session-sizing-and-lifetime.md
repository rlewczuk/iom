# TinyLlama forward layout — Session sizing and lifetime

This subsection is the normative bounded-storage plan for the planned
single-sequence session. It does not add a model, session, loader, selector,
facade, kernel, or current support claim. The existing neural hooks remain
`Unsupported`; the operation-owning sections and four-backend gates above remain
authoritative.

## Parameters and two-stage setup

`Nlayers,F,M,Hq,Hkv,D,V,C,R` are runtime values. Each is nonzero when used,
and the session validates checked `F=Hq*D`. `C` is the context and per-cache
logical row capacity. The separately configured nonzero `Qcap` is queue
admission capacity only: it bounds accepted in-flight work but neither changes
`C` nor multiplies weights, caches, banks, or scratch. All tensor ranks remain
2 through 8; an inserted head axis or additional leading axes must still fit
that interval. Plane counts and transformed leading offsets remain independent
and are never state broadcast.

Setup occurs in two stages:

1. **Session setup, before requests.** Validate configuration and checkpoint
   shapes; establish mapped source -> `SafeTensors` -> caller-created device
   ownership; create persistent BF16 weights; allocate each layer's bounded K
   and V cache owners; and allocate immutable-final-axis logical-`R1=1`
   decode banks. These resources are not recreated for a different prompt
   length.
2. **Request setup, after tokenization.** Validate exact `R>0` and `R<=C`
   before submission, reject overflow or excessive prompts without
   truncation, allocate exact-`R` prefill banks, create every operand and
   output needed to query actual operation and transfer requirements, and
   provision reusable prefill/decode workspace plus separate synchronous
   selector scratch. Replacing a request's banks or workspace requires a
   complete drain of the prior request first.

There is no one-bank-per-possible-`R` family, capacity-row logical-padding
trick, final-axis slicing, retargeted owner or view, per-token allocation, or
exposure of physical capacity as initialized logical length. An owner has one
shape and stable identity for its entire lifetime. Exact-R prefill and
fixed-R1 decode banks are reused only after all readers and workspace leases
that reference them complete.

The loader validates model configuration, tensor names, exact logical shapes,
and checked storage before copying mapped bytes into caller-created device
tensors. It alone adapts rank-one normalization scales to `[1,F]`. Persistent
BF16 weight shapes are:

| Scope | Weight shape |
| --- | --- |
| model | embedding `[V,F]`, final norm `[1,F]`, untied LM head `[V,F]` |
| each layer | attention norm `[1,F]`, post-attention norm `[1,F]` |
| each layer | Q `[F,F]`, K `[Hkv*D,F]`, V `[Hkv*D,F]`, O `[F,F]` |
| each layer | gate `[M,F]`, up `[M,F]`, down `[F,M]` |

The mapped source and SafeTensor views remain source ownership, not a
persistent second checkpoint bank. Device/arena/queue metadata overhead and
host transfer staging are accounted separately from tensor bytes. This BF16
session plan adds no additional F64 implementation or precision obligation.

## Checked cache and activation storage

For one sequence, each layer owns distinct K and V BF16 tensors with exact
logical shape `[Hkv,C,D]`; no implicit leading-plane or session-state
broadcast is allowed. All products are evaluated one factor at a time with
checked arithmetic:

```text
cache_logical_bytes =
    2 (K and V) * Nlayers * Hkv * C * D * 2 (BF16 bytes)

per_cache_standard_bytes =
    Hkv * ceil(C / 16) * ceil(D / 16) * 256 * 2
all_standard_cache_bytes =
    2 * Nlayers * per_cache_standard_bytes.
```

The two final axes are padded independently; `256` is the number of BF16
elements in a 16x16 tile. A checked ceiling division must not first perform an
unchecked `extent+tile-1`. Standard storage is the result of the existing
`TensorSpec::tiled_storage_nbytes()` contract, not its logical byte count.

For the independent checked case
`Nlayers=2,F=8,M=12,Hq=4,Hkv=2,D=2,V=19,C=17`, `F=Hq*D`.
Logical K+V storage is `2*2*2*17*2*2 = 544` bytes. Standard storage is
`2*ceil(17/16)*ceil(2/16)*256*2 = 2048` bytes per cache, so both caches in
both layers require `2*2*2048 = 8192` bytes. Native storage is not part of
this standard sizing contract.

`F=8` and `M=12` deliberately exercise non-tile final axes. Exact BF16
activation accounting is:

| Exact logical `R` | `[R,8]` logical bytes | `[R,8]` standard bytes | `[R,12]` standard bytes |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 512 | 512 |
| 15 | 240 | 512 | 512 |
| 16 | 256 | 512 | 512 |
| 17 | 272 | 1024 | 1024 |

The logical expression is `R*8*2`. Standard storage is
`ceil(R/16)*ceil(width/16)*256*2`; both widths occupy one feature tile in this
case. A decode bank is the fixed `R1=1` row. `R=17` is exact capacity and is
accepted without truncation; `R=18` is rejected before allocation. The
physical padded rows in the table are neither initialized tokens nor
permission to allocate a logical `[C,width]` prefill bank.

Every rank, element, plane, tile, byte, alignment, subrange-end, native
conversion, and address-offset calculation is checked stepwise before
allocation or submission. This includes `Hkv*D`, `Hq*D`, `a+(R-1)`,
`C-a`, weight products, all leading-plane products, tile ceiling/round-up,
and conversion of bits to bytes. Overflow is an error; it never wraps into a
smaller allocation.

## Forward stores and live ranges

The session embeds the input once. Each configured decoder layer performs, in
order, RMSNorm; Q/K/V projections; separate Q and K RoPE; separate K and V
append; causal GQA SDPA; output projection; residual; RMSNorm; independent
gate/up projections; `SiLU(gate)`; the existing `mul(SiLU(gate),up)`; down
projection; and residual. After all layers, final RMSNorm and the untied
ordinary LM head's final-row window produce `[1,V]` logits.

Ordinary and head-planar projections retain the equations fixed above:

```text
Y[b,r,o]   = sum_i X[b,s+r,i] * W[o,i]
Y[b,h,r,d] = sum_i X[b,s+r,i] * W[h*D+d,i].
```

Q is `[Hq,R,D]`; K and V are `[Hkv,R,D]`. RoPE uses the split-half pair at
absolute `a+r`, separate K/V append writes
`cache[b,h,a+r,d]=new[b,h,r,d]`, and GQA selects
`g(h)=floor(h/(Hq/Hkv))`. SDPA reads only initialized `0<=t<L` that also
satisfies `t<=a+r`, applies its stable masked softmax and explicit BF16
probability boundary, and stores merged `[b,R,Hq*D]`. The final LM head uses
ordinary mode `s=run-1,R=1` and writes `[1,V]`; no extraction operation or
final-axis view transform is involved.

All operands and outputs below are BF16 except integer token indices and wide
caller scratch. Outputs are disjoint from inputs/readers. Read/read weight
reuse is allowed, but no owner moves and no physical bank or scratch range is
reused before every direct reader and lease completes.

| Value/resource | Shape/storage | Lifetime and reuse rule |
| --- | --- | --- |
| `X`, `X2`, next residual | exact prefill `[R,F]`, decode `[1,F]` | Each bank remains live through all direct readers; reuse follows their terminal OIDs. |
| Norm outputs | `[R,F]` or `[1,F]` | Separate output and read owners; these paths do not use an in-place alias. |
| Q/K/V projections | `[Hq,R,D]` / `[Hkv,R,D]` | Q/K remain through RoPE; K/V remain through their respective appends. |
| Rotated Q/K | matching head-planar shapes | Rotated Q remains through SDPA and rotated K through append; neither view is retargeted. |
| K/V caches | per layer `[Hkv,C,D]` | Persistent through the request; append alone initializes rows, only the published prefix is readable, and reset/destruction follows a safe drain. |
| Attention merged/output projection | `[R,Hq*D]`, then `[R,F]` | Output/read storage is disjoint and scratch reuse waits for all readers. |
| Gate/up, SiLU, product, down | `[R,M]`, `[R,M]`, `[R,M]`, `[R,F]` | Gate and up may enqueue independently; subsequent consumers wait for every direct producer. |
| Final norm/logits | `[R,F]`, then `[1,V]` | Logits are borrowed by the selector only for its synchronous call. |
| Token-index input | integer `[1,R]` or explicit independent planes | Caller-owned, validated before embedding, retained through embedding completion. |
| Host transfer/staging | caller-owned wide scratch and backend staging | Sized from actual transfer requirements; never hidden persistent checkpoint duplication. |
| Selector resources | caller-owned synchronous scratch | Provisioned separately from its actual requirement; no per-selection allocation or async task. |

The table defines logical values, not one simultaneously resident owner per
row of the table or per layer. Storage sharing is allowed only when these
actual nonoverlapping lifetimes prove it; no memory saving may be claimed from
an assumed final OID, queue order alone, or an aliased output.

## Actual requirement maximum and backend staging

Create all tensor operands and outputs first, then call the pure workspace
queries and host-transfer requirement queries with the actual exact-R
prefill and fixed-R1 decode views. The reusable operation/transfer owner is
sized as:

```text
scratch_bytes =
    max(each actual prefill operation requirement,
        each actual decode operation requirement,
        each actual host-transfer requirement)
scratch_alignment =
    max(32, each actual required power-of-two alignment).
```

Both calculations use checked alignment and address arithmetic. The selector's
synchronous scratch is separately provisioned from its own requirement.
Mutually exclusive live ranges are not added; simultaneous independent
submissions use checked disjoint subranges or the existing lease
serialization. A changed request cannot replace this owner until every
accepted use is terminal.

The four feasibility records below constrain, but do not replace, those actual
queries:

| Backend | Matrix staging that request setup must include when selected |
| --- | --- |
| CUDA | Direct linear may query zero global scratch because its assessed route uses fixed kernel-local tiles. The SDPA caller range is the exact two-segment layout of `src/cuda/sdpa.hpp`: `alignment 32`, `bytes = A32(P*Hq*pad16(R)*pad16(L)*4) + A32(P*Hq*pad16(R)*pad16(L)*2)`, with the BF16 probability segment beginning at the checked offset `A32(P*Hq*pad16(R)*pad16(L)*4)`. |
| ROCm | The conservative assessed linear range contains checked aligned `x_pack` and `y_pack`. Its SDPA range contains `q_pack`, sequentially reused per-`Hkv` K/V pack, FP32 scores, BF16 probabilities, BF16 PV, and merged staging. |
| SYCL | The assessed linear range contains checked FP32 product staging. Its conservative SDPA range contains FP32 scores, BF16 probabilities, FP32 PV, and BF16 head staging, with reuse only after the producing stage completes. |
Standard CUDA/ROCm/SYCL tensor data and raw workspace subranges retain the
existing 32-byte arena guarantees. CPU now provides the positive aligned
caller-owned workspace required by BF16 SDPA; its other zero-workspace
operations retain their `{0, 1}` requirement.
Unknown capability-dependent requirements are recorded as missing evidence,
not filled with a speculative constant.

## Producer schedule, cache publication, and abort

A positive OID proves admission only. Consumers are submitted only after all
their direct producer waits succeed:

1. wait for embedding before attention RMSNorm, and wait for that norm;
2. enqueue Q, K, and V projections as independent branches, then attempt all
   three waits and require all three successes;
3. enqueue Q and K RoPE independently, then attempt and require both waits;
4. enqueue separate K and V appends, attempt both waits, and only after both
   succeed publish checked `initializedL=a+R` and submit SDPA;
5. wait successively for SDPA, output projection, first residual, and MLP
   RMSNorm;
6. enqueue gate and up together, attempt and require both waits, then wait in
   turn for SiLU, product, down projection, and the second residual; and
7. after the final layer, wait for final norm and the one-row LM head before
   synchronous selection and logical token commit.

The two append waits are independent correctness barriers. If K append fails
and V append succeeds, the V write may be physical but no new initialized
prefix is published and SDPA is not submitted. Every cache row is initialized
only by its append. No SDPA call may read a row outside the last successfully
published prefix.

Queues are in order but do not propagate predecessor errors. A later
successful OID, including a final OID, says nothing about an earlier failed
OID. On admission or completion failure, the session stops new dependent work,
becomes poisoned, and attempts to wait/drain every accepted OID; one thrown
wait does not stop later drain attempts. Token commit is separate from
physical cache initialization, and there is no rollback, retry, or reuse of a
possibly updated failed-session cache.

Storage whose terminality is unknown remains retained or quarantined. A
request reset may publish valid length zero, reset an allocator, destroy
owners, or replace exact-R banks/workspace only after a safe drain and after
all live owners and leases permit it. A replacement request never reads the
prior request's cache and never installs new storage before that boundary.

## Attribution ownership seam

Attribution is correlation metadata owned by the caller, not a tracing or
timing implementation. For every phase—`load`, `tokenization`, `prefill`,
`decode`, and `selection`—the caller may associate:

- phase and operation name;
- decoder layer where applicable;
- absolute row/token position;
- the operation's positive OID where one exists; and
- explicit host-enqueue and completion-observation boundaries.

Host enqueue elapsed time, completion-observed elapsed time, and genuine
device timestamps are different measurements and must not be conflated.
Selection is synchronous and has no selector OID. This seam allocates no
telemetry event, adds no timing API, retains no view, creates no async selector
task, and does not force a per-operation wait merely because tracing is
disabled. The correctness waits above remain mandatory with or without
attribution.

Complete mathematical layer assembly was gated on the four-backend SDPA gate,
which the revision recorded in
[Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention) closes.
The first session-side checkpoint is now implemented as one private,
allocation-free exactly-one-layer composition: it consumes caller-owned
attention/QKV, cache/attention, and MLP stores, preserves the fixed
`X -> attention residual -> MLP residual` equations, and publishes the cache
prefix only after both append completions succeed.  This checkpoint records
producer waits, poison/drain behavior, absolute positions, GQA, and exact
capacity without claiming N-layer orchestration, final normalization or the
untied LM head, selection, generation, CLI behavior, or official-corpus
validation.  Incremental model/session/selector integration remains owned by
later siblings.

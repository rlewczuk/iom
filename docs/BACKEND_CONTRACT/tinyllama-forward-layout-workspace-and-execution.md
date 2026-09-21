# TinyLlama forward layout — Workspace and execution

This subsection records the shared facade boundary. The common layer now
declares embedding, linear, RMSNorm, RoPE, and SiLU; cache append remains
planned, and all four retained backends implement the BF16 GQA SDPA leaf,
closed in [Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention). The
target `DeviceOps` facades return `oid`, are `noexcept`, and have exactly
these signatures:

```cpp
oid embedding(const TensorView& table, const TensorView& indices,
              TensorView& out, RawWorkspaceView workspace = {}) noexcept;
oid linear(const TensorView& x, const TensorView& w, TensorView& out,
           std::size_t s, std::size_t R, LinearOutputLayout layout,
           std::size_t H, std::size_t D,
           RawWorkspaceView workspace = {}) noexcept;
oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace = {}) noexcept;
oid rope(const TensorView& x, TensorView& out, std::size_t a, double theta,
         RawWorkspaceView workspace = {}) noexcept;
oid cache_append(const TensorView& source, TensorView& destination,
                 std::size_t a, RawWorkspaceView workspace = {}) noexcept;
oid silu(const TensorView& x, TensorView& out,
         RawWorkspaceView workspace = {}) noexcept;
oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
         TensorView& out, std::size_t a, std::size_t L,
         RawWorkspaceView workspace = {}) noexcept;
```

Every dimension, row, head, position, and extent is `std::size_t`. Tensor
inputs are const, the submitted output (or cache destination) is mutable, and
the optional caller workspace is last. `enum class LinearOutputLayout {
ordinary, head_planar }` is the fixed output-mode type; output rank does not
select the mode. `ordinary` uses `H=1,D=O`. `head_planar` requires checked
`H*D=O` and inserts a head axis without exceeding rank eight. Embedding derives
the vocabulary extent from its rank-two table. RMSNorm derives feature width
from its last axis and has no `dim` argument. SDPA derives head counts and head
width from validated tensor shapes and receives only the explicit positions
`a,L`.

Each facade has one pure workspace query. The query has the same semantic
arguments in the same order, changes only its output/destination to const,
omits only the workspace argument, returns `WorkspaceRequirements`, and is
allowed to throw the established validation exceptions:

```cpp
WorkspaceRequirements embedding_workspace_requirements(
        const TensorView& table, const TensorView& indices,
        const TensorView& out);
WorkspaceRequirements linear_workspace_requirements(
        const TensorView& x, const TensorView& w, const TensorView& out,
        std::size_t s, std::size_t R, LinearOutputLayout layout,
        std::size_t H, std::size_t D);
WorkspaceRequirements rmsnorm_workspace_requirements(
        const TensorView& x, const TensorView& scale,
        const TensorView& out, float eps);
WorkspaceRequirements rope_workspace_requirements(
        const TensorView& x, const TensorView& out, std::size_t a,
        double theta);
WorkspaceRequirements cache_append_workspace_requirements(
        const TensorView& source, const TensorView& destination,
        std::size_t a);
WorkspaceRequirements silu_workspace_requirements(
        const TensorView& x, const TensorView& out);
WorkspaceRequirements sdpa_workspace_requirements(
        const TensorView& q, const TensorView& k, const TensorView& v,
        const TensorView& out, std::size_t a, std::size_t L);
```

A query MUST be deterministic for the supplied values and current backend
capability. It MUST validate operation support; every operand and output spec;
the general rank rule `2..8`; the cache_append exception rank `3..8`;
nonzero dimensions; exact queue-device identity; live owners; leading-view
bounds and strides; scalar, range, and mode values; output shape; aliases;
checked element, byte, address, plane, tile, and stride arithmetic; and
backend capability. It MUST NOT allocate, register an owner, acquire a
lease, reserve a queue credit or token, submit backend work, inspect queue
occupancy or free-arena capacity, or depend on prior completion. A successful
query reserves nothing. The actual supplied workspace is deliberately not a
query operand and is validated only by facade admission after the requirement
is known.

Before owner registration, sequence consumption, token acceptance, metadata
effects, or backend work, a facade MUST validate, in order:

1. host-known rank, nonzero shape, mode, position, and range constraints;
2. exact device identity, live owner identity, native handles, and leading
   view bounds and strides;
3. all checked multiplication, byte, address, plane, tile, and stride
   arithmetic;
4. operand/output shape and alias rules, followed by supported
   `QuantizationFormat::NONE`, dtype, and backend capability; and
5. the caller's actual workspace against the query's `{bytes, alignment}`,
   including liveness, exact-device identity, size, alignment, overlap, and
   lease availability.

For `cache_append`, structural validation of the views, owners, and checked
plane/stride/address metadata precedes the operation-specific window checks.
Once `C` and `R` are safely known, those checks are ordered as `a <= C` first
and then `R <= C-a`; append-specific checked element, byte, storage-range,
and workspace arithmetic follows them and completes before any owner
registration, sequence consumption, mutation, or submission.

A malformed request returns `InvalidArgument=-1`; a well-formed but
unsupported operation or matching input returns `Unsupported=-2`; checked
arithmetic or sequence exhaustion returns `Overflow=-3`; bounded resource or
lease failure returns `ResourceExhausted=-4`; a pre-acceptance runtime failure
returns `DeviceError=-5`; and any other unclassified failure returns
`InternalError=-6`. An admission failure accepts no OID and causes no backend
effect. Zero is never accepted. Every positive result is an accepted
submitted token; it proves admission only, not successful initialization of
output or cache data. A completion failure accepted after submission remains
observable and every wait for that OID MUST retain and rethrow the same
failure. Device-data out-of-vocabulary or nonfinite-policy failures MAY be
accepted without a hidden host round trip. Their output is unusable, their
session is poisoned, and every wait for that OID MUST retain and rethrow the
same failure.

The caller owns every output and all scratch. Nonzero scratch MUST be a live
`RawWorkspace` owner created by the queue's exact `Device`, with sufficient
bytes, the queried alignment, and no overlap with any operand or output.
Workspace subrange offsets are checked and 32-byte aligned, and owner
subranges are at least 32-byte aligned. An empty workspace is valid exactly
when the query reports zero bytes. A backend that requires positive scratch owns
the minimal factory support and conformance tests for that requirement.
A capability-blocked backend MUST name the missing evidence instead of
inventing a byte requirement.

The reusable session scratch capacity is the maximum of the actual
prefill/decode operation and host-transfer requirements, not the sum of
mutually exclusive live ranges. Independent submissions MAY use disjoint
aligned subranges of one owner. Overlapping use requires the existing lease
serialization. Outputs, operands, and scratch MUST remain alive until every
accepted reader and writer completes, and a workspace range is reusable only
after its covering completion is proved. When terminality cannot be proved,
the entire affected range and storage remain retained or quarantined; they
MUST NOT be released or reused by an unrelated queue or device destruction.

At each facade call, common code MUST snapshot every required tensor spec,
exact device and owner identity, native handle, leading offset and strides,
scalar, mode, and workspace range. It MUST NOT retain a borrowed `TensorView`
or other borrowed view reference beyond the call. Submission follows the
existing snapshot, owner-registration, and exact-alias owner-deduplication
pattern. This contract adds no global active-backend registry, duplicate
validation framework, or no-allocation guarantee beyond the existing queue
machinery.

A queue is in order, but it does not propagate predecessor errors: current
queues can execute a successor after a predecessor fails, and `wait(last)`
does not report an earlier failed OID. A future TinyLlama session therefore
MUST successfully wait for every direct producer before submitting a consumer
of that producer. It MUST NOT submit that consumer after a producer wait
fails. Independent branches MAY be submitted together and then waited
individually; Q, K, and V projections are independent after their shared input
succeeds, as are gate and up projections. These are explicit correctness
boundaries, not hidden operator round trips or waits introduced only for
disabled tracing.

For one token row or prefill block, the required producer-success schedule is:

1. submit embedding and wait successfully before submitting its RMSNorm
   consumer; submit that attention RMSNorm and wait successfully;
2. submit Q, K, and V linear projections together, then wait successfully for
   all three OIDs individually;
3. submit the Q and K RoPE branches together and wait successfully for each;
4. submit separate K and V `cache_append` operations and wait successfully for
   both append OIDs; only then publish checked `initializedL = a + R` and
   submit SDPA with that initialized prefix;
5. wait successfully for SDPA before its output projection, wait for that
   projection before the residual add, and wait for the residual before the
   MLP RMSNorm;
6. after the MLP RMSNorm succeeds, submit gate and up linear projections
   together and wait successfully for both; wait for SiLU of gate before the
   elementwise gate/up product, then wait in turn for the down projection and
   final residual add; and
7. commit the logical token/request only after the final producer succeeds.

For a concrete failure boundary, suppose accepted `token1` completes,
accepted `token2` fails, and independently accepted `token3` completes.
Successful `wait(token3)` MUST NOT be interpreted as success of `token2` or of
the whole chain. The session waits all three separately, submits no consumer
of `token2`, marks itself failed, and drains every already accepted OID. The
same rule applies to cache append: if accepted K append `tokenK` fails while
accepted V append `tokenV` succeeds, the session still waits/drains both,
publishes no new `initializedL`, and submits no SDPA. Physical cache writes
that did complete remain distinct from the uncommitted logical token; a later
failure may therefore leave updated cache storage while poisoning the session.

For an accepted OID, `wait(oid)` blocks through completion and rethrows its
retained failure. Every repeated wait for that OID MUST rethrow the same
failure; successful waits and completion visibility are likewise repeatable.
`wait` immediately throws `std::invalid_argument` for negative, zero, foreign,
future, skipped, reserved-but-never-submitted, or otherwise unsubmitted
values. There is no cumulative wait, fail-fast cancellation, KV rollback,
retry, or reuse of poisoned state.

After any admission failure or accepted completion failure, the session MUST
stop new dependent submissions, become poisoned, and observe or drain every
accepted OID. Each wait is attempted independently, and draining MUST continue
after an individual wait throws. Storage can be released or reused only where
terminality is proved; otherwise it remains retained or quarantined. Token
commit is separate from physical cache initialization, and no failed cache
prefix may be published.

The interfaces in this planning subsection that are not declared by an
operation-owned section are not current API declarations. Existing neural hooks
not covered by a retained operation port remain `Unsupported`; the implemented
SiLU hook is governed by the [SiLU activation](silu-activation.md#silu-activation) contract and
its four-backend gate receipt. This subsection changes no other facade, kernel,
queue, session, or selector implementation.

The first session-side composition of these facades is implemented as a
private, allocation-free exactly-one-layer checkpoint in `src/session.cpp`.
It composes attention RMSNorm, the three independent head-planar Q/K/V
projections, independent Q/K split-half rotations, separate K/V cache
publication, causal grouped-query SDPA, the ordinary attention projection and
first residual, then post-attention RMSNorm, independent gate/up projections,
stored SiLU, multiply, down projection, and the second residual.  It waits
every direct producer successfully, waits both append OIDs before publishing
`initializedL` or submitting SDPA, preserves absolute positions and the
GQA mapping, and drains accepted work after failure.  All stores, views, queue
identity, and scratch remain caller-owned; this adds no public stage API,
graph, kernel, capability, or contract change.  Focused synthetic CPU
coverage lives in `test/test_model_session.cpp`.  N-layer orchestration,
final normalization and LM-head logits, selection, generation, CLI behavior,
and official-corpus validation remain later work.

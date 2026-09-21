# TinyLlama forward layout — Normalization and MLP boundaries

The following interfaces are the frozen ABI of this subsection. The RMS
normalization declarations and their admission contract are frozen and
declared by [RMS normalization](rms-normalization.md#rms-normalization); the SiLU declarations are
also frozen, declared, and implemented:

```cpp
oid rmsnorm(const TensorView& x, const TensorView& scale, TensorView& out,
            float eps, RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements rmsnorm_workspace_requirements(
        const TensorView& x, const TensorView& scale,
        const TensorView& out, float eps);

oid silu(const TensorView& x, TensorView& out,
         RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements silu_workspace_requirements(
        const TensorView& x, const TensorView& out);
```

The RMS normalization cutover removed the redundant `dim` argument from
declarations, definitions, callers, tests, and documentation; no overload,
alias, compatibility shim, or re-export survives. The last logical axis is
the feature width. For `x` and `out` shaped `[...,R,F]`, `scale` MUST have
exactly `[1,F]`; it is shared explicitly across every row and independent
leading plane, without introducing a general hidden-state or leading-plane
broadcast. The loader alone adapts a checkpoint normalization vector `[H]` to
the public rank-two `[1,H]` scale. RMSNorm MUST NOT accept a rank-one scale.

For every logical leading-plane index `b`, row `r`, and feature `f`, RMSNorm is
row-local:

```text
out[b,r,f] = round(
    x[b,r,f]
    * rsqrt(sum(i=0..F-1, x[b,r,i] * x[b,r,i]) / F + eps)
    * scale[0,f])
```

The reduction includes exactly the `F` logical features: tiled padding, other
rows, and other planes MUST NOT contribute. `eps` MUST be finite and
nonnegative; the model configuration supplies a finite positive value. With
`eps == 0`, an all-zero row produces a quiet NaN, with no NaN-payload promise.
For the model's BF16 boundary, squares, reduction, reciprocal square root, and
scale multiplication use FP32 or a demonstrated sufficiently wide equivalent,
then the result is rounded once to the output leaf.

The operation-specific [SiLU activation](silu-activation.md#silu-activation) contract requires
rank 2 through 8, identical input/output shapes, independent leading planes,
and the element equation `y[b,r,f] = x[b,r,f] / (1 + exp(-x[b,r,f]))`.
It evaluates that equation stably in one wide computation and rounds once to
the output leaf. The MLP MUST compose that stored result with the existing
binary operation as `mul(SiLU(gate), up, product)`—in that operand order—and
MUST NOT add a fused SiLU-times-multiply operation.

Each requirement query returns `WorkspaceRequirements`, is deterministic and
pure, and has the same semantic arguments as its operation except for a const
output and no workspace argument. It performs operand, shape, scalar, alias,
capability, and checked-arithmetic validation only. It MUST NOT allocate,
register owners, acquire a lease, reserve or consume a sequence, submit work,
read queue occupancy or arena availability, mutate output, or otherwise change
state. Query validation errors are standard exceptions. The eventual
`noexcept` operation facades instead map pre-acceptance errors to the common
negative OID categories.

Admission MUST finish before work, owner registration, sequence consumption,
or output mutation. It validates ranks 2 through 8, nonzero dimensions,
leading tuples, exact device identity, dtype and quantization, aliases,
workspace device/size/alignment/nonoverlap, scalar finiteness, and checked
shape/range/element/byte/address arithmetic. Neural outputs are disjoint from
their inputs and scratch. Submission snapshots metadata rather than retaining
borrowed views, registers all required owners and workspace through completion,
and preserves repeated-wait observation of an accepted device-data failure.

Within each decoder layer, normalization and MLP boundaries are:

1. store RMSNorm of residual `X [R,F]` in distinct `N [R,F]`;
2. after attention output projection stores `B [R,F]`, use existing
   `add(X,B,X2)` to store the first residual `X2 [R,F]`;
3. store RMSNorm of `X2` in distinct `N2 [R,F]`;
4. independently store gate and up linear outputs `Gate [R,M]` and
   `Up [R,M]`;
5. store `SiLU(Gate)` in `ActivatedGate [R,M]`;
6. use existing `mul(ActivatedGate,Up,Product)` to store `Product [R,M]`;
7. store the down projection in `Down [R,F]`; and
8. use existing `add(X2,Down,NextX)` to store the second residual and next
   layer input `NextX [R,F]`.

Every numbered boundary has its own BF16 result store and operation-boundary
rounding; fusion MUST NOT erase either residual, either normalization boundary,
or the SiLU/multiply boundary. Existing `add` and `mul` broadcasting, alias,
workspace, queue, and numerical rules remain unchanged. Names denote logical
values, not a requirement to retain one persistent activation bank per layer:
storage MAY be reused only after all readers and accepted work have completed.
Each layer's K and V caches remain distinct persistent owners.

For the worked non-tile case `F=8`, `M=12`, `Hq=4`, `Hkv=2`, and `D=2`,
`F=Hq*D`. Each of `R=1,15,16,17` therefore has `X/N/N2/B/Down/NextX [R,8]`,
Q `[4,R,2]`, K/V `[2,R,2]`, merged attention `[R,8]`, and
`Gate/Up/ActivatedGate/Product [R,12]`. Consumers match those logical shapes
for all four runs; padding is not a logical row or feature. A conformance
example with leading extent `P` applies the same mapping independently:
`[P,R,8]` produces Q `[P,4,R,2]`, K/V `[P,2,R,2]`, and MLP intermediates
`[P,R,12]`. It does not broadcast state between planes or transform either
final tensor axis.

Setup MUST use checked arithmetic and a deterministic sizing order covering
embedding output, per-layer normalization and Q/K/V projections, RoPE,
separate bounded K/V caches, SDPA and maximum operation workspace, attention
projection, both residual stores, all MLP intermediates, final normalization,
and one-row logits. It keeps cache capacity and maximum workspace explicit,
does not duplicate persistent checkpoint weights, and never allocates or
relocates an operand or result inside an operation. Actual allocation,
initialized cache length, reset/growth/failure policy, and selector invocation
belong to the session component; operation-local minimum scratch belongs to
the corresponding operation contract.

The detailed datatype applicability, special-value behavior, references,
tolerances, fixture provenance, snapshots/hooks, kernels, and evidenced
backend limitations remain solely owned by the
[RMS normalization](rms-normalization.md#rms-normalization) and the implemented **SiLU** operation
sections. RMSNorm's four retained-backend ports are closed by the focused
conformance targets named in its section; SiLU's four retained-backend ports
are closed by the gate receipt in its operation section. This plan subsection
declares no additional ABI.

Delivery is operation-first, not mathematical-forward order: embedding,
linear, RMSNorm, RoPE, cache append, SiLU, and finally SDPA. For each operation,
finish its contract, independent reference, and four-backend feasibility record,
then close CPU, CUDA, ROCm, and SYCL in that order before starting the next
operation. The CPU baseline is scalar/wide arithmetic over existing tiled
storage, not native-accelerator evidence. CUDA/ROCm/SYCL native BF16 matrix
feasibility for linear and SDPA prefill and logical `R=1` MUST be evidenced
before those implementations; host computation, round trips, elementwise
substitutes, and padded extra logical tokens do not count.
Unavailable ports remain reported as unsupported and do not count as numerical
conformance. Future four-backend conformance covers success, runs
`1/15/16/17`, non-tile features, independent planes, padding isolation,
rejections, and repeatable accepted failures. Only after SDPA closes may the
session assemble and verify the complete layer from these boundaries.

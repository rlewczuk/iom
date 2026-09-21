# SiLU activation

This is the operation-owned contract for the implemented `DeviceOps::silu`.
The common facade admits a well-formed SiLU request on each backend according to
the capability matrix below; unsupported semantic or backend leaves still
return `Unsupported` before submission, mutation, or token acceptance. The
exact public surface is:

```cpp
oid silu(const TensorView& x, TensorView& y,
         RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements silu_workspace_requirements(
        const TensorView& x, const TensorView& y);
```

The requirement query has the same semantic tensor arguments as submission,
with no workspace argument. It is a pure requirement query, not a support
probe: it performs no allocation, owner registration, lease acquisition,
sequence reservation, queue submission, queue or arena inspection, data read,
or output mutation. It may perform only the operation's throwing structural,
dtype, capability, alias, and checked-arithmetic validation, and it is
deterministic for the same complete tensor arguments and immutable device
capability.

1. **Logical mapping and equation.** `x` and `y` MUST have the same complete
   rank-two through rank-eight shape `[...,R,F]`, with every extent nonzero.
   `R` is the logical run extent and `F` is the logical feature extent. For
   every complete leading-plane tuple `b`, every `0 <= r < R`, and every
   `0 <= f < F`, SiLU evaluates each element exactly once:

   ```text
   y[b,r,f] = x[b,r,f] / (1 + exp(-x[b,r,f]))
   ```

   Every leading plane is independent. There is no leading-plane or feature
   broadcast, reduction, transposition, row extraction, hidden state,
   session state, or cross-element dependence.

2. **Tensor, view, and padding boundaries.** Both views MUST use the supported
   TILE layout, the same dtype, and `QuantizationFormat::NONE`, and they MUST
   belong to the exact same device identity. `y` is caller-owned output with
   no owner or storage in common with `x`; exact aliases and every partial
   overlap, including overlap exposed by transformed view ranges, are
   rejected. Only logical elements are read from `x` and written to `y`.
   Physical `16x16` TILE padding and every unused storage cell are outside
   the equation and MUST NOT be read as inputs or modified. Poisoning input
   padding MUST NOT change logical output, and poisoning output padding MUST
   remain unobservable.

3. **Semantic dtype classification.** The complete SiLU classification is
   the following 23-leaf table. Applicability is semantic and is not a claim
   that every backend stores or computes every applicable leaf.

   | Leaf | Semantic SiLU class | CPU | CUDA | ROCm | SYCL |
   | --- | --- | --- | --- | --- |
   | `BOOL` | Unsupported/inapplicable: not a signed real-valued transcendental operand or result | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `I2` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `U2` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `I4` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `U4` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `I8` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `U8` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `I16` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `U16` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `I32` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `U32` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `I64` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `U64` | Unsupported/inapplicable: no implicit integer-to-real SiLU | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `F4_E2M1` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F6_E2M3` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F6_E3M2` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F8_E4M3FN` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F8_E5M2` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F8_E8M0` | Unsupported/inapplicable: unsigned exponent-only encoding is not a signed real value | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability | `Unsupported` — semantic inapplicability |
   | `F16` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `BF16` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F32` | Applicable signed real format | Supported | Supported | Supported | Supported |
   | `F64` | Applicable signed real format; never silently narrowed | Supported | Supported | Supported | `Unsupported` — no device-independent FP64 guarantee |

CPU, CUDA, and ROCm therefore support all nine applicable leaves. SYCL supports
exactly the eight applicable leaves other than `F64` and rejects `F64`
deterministically with the named `Unsupported` capability reason that this path
has no device-independent FP64 guarantee; it does not lower or emulate `F64`.
`BOOL`, all 12 integer leaves, and `F8_E8M0` remain `Unsupported` on every
backend because they are semantically inapplicable. An unknown `DataType` enum
is `InvalidArgument`, not a capability result. A recognized but unsupported
applicable leaf returns `Unsupported` with the named capability reason only
after structural admission checks. Only `QuantizationFormat::NONE` is in scope;
an unknown quantization enum is `InvalidArgument`, and a recognized non-`NONE`
format is `Unsupported`.

4. **Zero workspace and pure requirements.** Every backend's
   `silu_workspace_requirements(x, y)` returns exactly `{0, 1}`. Since the
   required capacity is zero, a supplied nonempty `RawWorkspaceView` is
   ignored: its owner, device, address, size, alignment, overlap, and
   liveness are not validated, and it is neither leased nor retained. This
   is the zero-capacity path of `WorkspaceValidation::validated` in
   `src/device_ops.cpp`. No implementation may hide an allocation, scratch
   buffer, staging transfer, or host round trip behind this contract.

5. **Admission and errors.** Before sequence consumption, owner
   registration, allocation, output mutation, or submission, admission MUST
   validate every host-known fact: rank and nonzero dimensions; complete
   shape equality and leading-plane tuples; TILE layout and `NONE`
   quantization; known dtype enum and backend capability; exact device
   identity; live owner and native-view identity; input/output disjoint
   storage including transformed-view ranges; and checked element, tile,
   byte, stride, plane, address, and other size arithmetic. The zero-capacity
   workspace rule above is the only workspace exception. No pre-acceptance
   failure may enqueue work, mutate output, register an owner, allocate, or
   consume a sequence.

   The `noexcept` facade maps pre-acceptance failures through the common
   categories `InvalidArgument=-1`, `Unsupported=-2`, `Overflow=-3`,
   `ResourceExhausted=-4`, `DeviceError=-5`, and `InternalError=-6`. A
   positive OID denotes accepted work. The request copies all needed tensor
   metadata—shape, dtype, quantization, offsets, strides, device identity,
   owner identity, native-view identity, and validated bounds—rather than
   retaining borrowed `TensorView` objects. Caller-owned operand and output
   resources remain retained until proven completion, and temporary views may
   be destroyed immediately after submission.

6. **Queue, ownership, and accepted failures.** SiLU follows the established
   in-order asynchronous queue protocol. An accepted device-side failure
   remains attached to its positive OID, and every repeated wait observes the
   same failure; callers follow the common drain/reset rules before destroying
   resources. The contract does not promise rollback or unchanged output
   after an accepted failure. SiLU owns no session or model state and performs
   no operation-level synchronization beyond its queue submission.

7. **Stable finite arithmetic.** Mathematical SiLU is
   `x / (1 + exp(-x))`. For negative finite `x`, the independent reference
   and every conforming implementation use the underflow-safe form
   `t = exp(x/2); y = ((x*t)*t)/(1+t*t)`, with the numerator evaluated in
   the written left-associated order so `t*t` is not rounded to zero before
   multiplication by `x`. A nonnegative finite branch may use
   `x/(1+exp(-x))`. `x*exp(x)/(1+exp(x))` is permitted only where it preserves
   every required representable tail; naive `inf/inf` behavior MUST NOT select
   policy.

8. **Special values and named scalar formats.** Representable special values
   are handled before finite arithmetic: `+infinity` maps to `+infinity`,
   `-infinity` maps to negative zero, signed `+0` and `-0` are preserved,
   and NaN maps to NaN without a payload or sign-equality promise. `F4_E2M1`,
   `F6_E2M3`, and `F6_E3M2` represent no nonfinite values;
   `F8_E4M3FN` represents NaN but no infinity. Finite inputs never produce
   NaN. If a negative finite result rounds to zero, its negative sign is
   retained. Saturation, subnormal, signed-zero, and representable special
   classes reuse the named-format codec rules; finite-only formats never gain
   invented nonfinite encodings.

9. **Intermediates, rounding, and tolerance.** Applicable leaves through
   `F32` use at least FP32 intermediates, while `F64` uses FP64 intermediates.
   FTZ and fast-math behavior that erases required tails are non-conforming.
   Exactly one RNE output encoding/store occurs at the operation boundary; the
   target format is not used as an intermediate. Supported `F32` input
   `x=-104` MUST retain a nonzero negative subnormal tail, and supported
   `F64` input `x=-746` MUST retain a nonzero negative subnormal tail.

   The independent reference uses one target-format ULP as the ceiling for
   `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, and
   `BF16`; four ULP for `F32`; and eight ULP for `F64`. Special-value classes,
   signed-zero signs, exact zero behavior, and the two underflow-tail fixtures
   require exact class/sign checks in addition to those ceilings. The
   independent scalar/raw reference is linked to
   `src/shared/scalar_binary_codec.hpp` and MUST NOT use production SiLU as
   its sole oracle.

10. **Shared SiLU conformance map.** The implemented shared header
    `test/backend/backend_conformance_silu.hpp` owns the following named
    cases; every rule above MUST be observable through one of them, and an
    unsupported backend keeps an explicit `Unsupported` probe rather than
    counting rejection as conformance:

    | Shared case | Required coverage |
    | --- | --- |
    | `api_and_query_purity_zero_workspace` | Exact ABI and pure query; deterministic `{0, 1}` requirements; nonempty workspace ignored without validation, leasing, retention, or state effects. |
    | `shape_rank_planes_runs_and_padding` | Rank 2..8, complete `[...,R,F]` mapping, independent leading planes, logical runs `R=1,15,16,17`, non-tile feature sizes, TILE padding poisoning, and output-padding isolation. |
    | `dtype_classification_and_backend_matrix` | All 23 leaves, semantic inapplicability versus capability, `NONE` quantization, CPU/CUDA/ROCm nine-leaf support, and SYCL eight-leaf support plus deterministic `F64` rejection. |
    | `admission_alias_device_owner_and_overflow` | Shape/layout/dtype/device/owner/native-view admission, exact aliases and partial overlaps, transformed-view disjointness, checked element/tile/byte/stride/plane/address overflow, and negative OID categories. |
    | `stable_finite_reference_and_tails` | Independent scalar/raw reference, stable finite branches, FP32/FP64 intermediates, one output encode, no FTZ, `F32 -104` and `F64 -746` tails, and 1/4/8-ULP ceilings. |
    | `special_values_signed_zero_and_rounding` | Infinity and NaN classes, signed zeros, negative-zero underflow, finite-only format limits, saturation/subnormals, exact class/sign checks, and RNE encoding. |
    | `accepted_failure_repeat_wait_and_temporary_view` | Accepted asynchronous failures retained on positive OIDs, repeated identical waits, owner retention, temporary-view destruction safety, in-order queue behavior, and drain/reset boundaries. |
    | `stored_result_composes_with_mul_and_add` | Distinct `[R,M]` or independent-plane `[P,R,M]` `ActivatedGate` storage, then existing `mul(ActivatedGate, Up, Product)` and `add(Product, Residual, Summed)` with stored rounding and ownership boundaries intact. |

    The shared cases cover exact output ownership, logical runs and features,
    independent planes, poisoned input/output padding, every applicable
    backend/dtype expectation, admission and overflow failures, special
    classes and rounding, accepted failures and repeated waits, and the
    stored-result composition boundary through existing `mul` and `add`. They
    do not add a generic unary, SwiGLU, or MLP API.

11. **MLP composition boundary.** The caller stores `SiLU(Gate)` in a
    distinct `[R,M]` result, or `[P,R,M]` for independent leading planes,
    then calls existing `mul(ActivatedGate, Up, Product)` followed by
    `add(Product, Residual, Summed)`. SiLU MUST NOT fuse with multiplication or
    addition, alter `mul` or `add`, erase the stored-result rounding boundary,
    or add a generic unary or fused SwiGLU/MLP operation. Existing residual,
    session, and model boundaries remain outside this section.

The operation-specific conformance header and the four backend drivers are the
only test surface for this contract. The declaration, implementation, kernel,
test registration, and queue integration are all present in the retained
backends; the gate receipt below records the exact same-worktree evidence.

## CUDA SiLU implementation boundary and evidence

CUDA's SiLU leaf is implemented in `src/cuda/copy.cu` behind the common
`DeviceOps::silu` admission path, the shared queue seam of
`src/shared/gpu_queue.hpp`, and the CUDA `gpu_policy` hooks declared in
`src/cuda/copy.hpp`. This subsection records the CUDA boundary only: it makes
no claim about the other backends and no native-matrix claim of any kind.

- **Capability.** CUDA queues exactly the nine applicable signed floating
  leaves (`F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`,
  `BF16`, `F32`, and `F64`) and no other. `BOOL`, the twelve integer leaves,
  `F8_E8M0`, and every non-`NONE` quantization format stay `Unsupported` from
  common admission before the CUDA capability predicate is consulted, and the
  pure `silu_workspace_requirements` query keeps returning exactly `{0, 1}`
  while a supplied nonempty workspace stays ignored.
- **Device operation.** Each accepted submission uploads its immutable
  descriptor into the queue's fixed metadata slot and then launches exactly one
  kernel on the queue's own nonblocking stream: no staging, hidden allocation,
  host round trip, second queue, or device-side detour through another
  operation. One thread owns one 16-feature tile-row packet and every word
  touched by that packet, so every packet word is read, merged, and stored once,
  each logical element is evaluated exactly once, and neither physical tile
  padding nor any packed padding bit outside the logical element set is written.
  Independent leading planes, logical runs, and features keep their own
  transformed plane offsets and strides.
- **Numerics.** `src/shared/scalar_binary_codec.hpp` remains the sole
  named-format codec and `src/shared/scalar_silu.hpp` the sole stable scalar
  evaluator; the CUDA translation unit instantiates both with an FP64 carrier
  for all nine leaves and performs exactly one RNE destination encode per
  element, with no FTZ or fast-math behavior on this path, so the required
  `F32` `-104` and `F64` `-746` negative subnormal tails survive.
- **Queue, ownership, and errors.** Owner registration, metadata leasing, the
  completion event, retained failures, drain, and quarantine remain properties
  of the shared queue. An accepted launch or event-record failure keeps its
  positive OID, consumes no further sequence, and reports the identical error
  on every wait; operands and temporary views may be destroyed immediately
  after submission and the accepted work still completes from retained
  storage; an unknown completion whose covering drain fails quarantines the
  queue lease until a covering proof reclaims its partition.
- **Observed evidence.** The CUDA-only configuration (`-DCUDA_ENABLED=ON`, no
  other optional backend) built `iom_cuda_conformance_tests`; `nvcc` emitted
  pre-existing unused-function warnings in `src/cuda/linear.cu`, so this was
  not a diagnostics-free build. On NVIDIA GeForce RTX 5090 (compute
  capability 12.0, driver 595.71.05, CUDA `13.2`), the anchored CUDA CTest
  target passed 1/1 and the focused `*SiLU*` selection passed 5/5 with 4,127
  assertions. The run covers all nine leaves, the ULP, class, and sign checks,
  zero-workspace query, logical-only write and tile-padding invariants across
  full, transformed, and selected leading views, `runs` 1/15/16/17,
  non-tile feature sizes, the stored `SiLU -> mul -> add` boundary, and the
  CUDA fault/lifetime cases named in the queue clause above.

## ROCm SiLU implementation boundary and evidence

ROCm's SiLU leaf is implemented in `src/rocm/copy.hip` behind the common
`DeviceOps::silu` admission path, the shared queue seam, and the ROCm
`gpu_policy` hooks. ROCm advertises the same nine applicable signed floating
leaves as CPU and CUDA; all semantic integer, `BOOL`, `F8_E8M0`, and non-`NONE`
quantization probes remain rejected by common admission.

The HIP kernel assigns one thread to a 16-feature tile-row packet and all words
touched by that packet. Each logical field is decoded, evaluated with the
stable shared scalar SiLU evaluator, RNE-encoded, and stored exactly once;
cross-word F6 fields therefore have one packet owner rather than two adjacent
word owners. Physical tile padding is not written. The byte-wise word access
helpers remain a conservative alignment-safe implementation detail; their
cost is performance-only and does not change the logical contract.

The final ROCm gate ran on AMD Radeon AI PRO R9700 (`gfx1201`) with HIP
7.15.26333 / ROCm core 10.0. The anchored `iom_rocm_conformance_tests` CTest
target passed 1/1, and the focused `*SiLU*` selection passed 5/5 with 4,172
assertions, including the stored `SiLU -> mul -> add` boundary and native
failure/lifetime checks.

## SYCL SiLU implementation boundary and evidence

The native SYCL SiLU port currently advertises exactly the eight applicable
leaves `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`, `F16`, `BF16`,
and `F32`. `F64` remains deterministically `Unsupported` with the named
limitation that this path has no device-independent FP64 guarantee; the port
does not lower or emulate `F64` through FP32. The semantically inapplicable
`BOOL`, integer, and `F8_E8M0` leaves remain `Unsupported`.

The implementation runs the accepted request through one native in-order SYCL
device kernel. Each work item owns one physical output word for the 4-, 8-,
16-, and 32-bit leaves, or one aligned three-word (96-bit) packet for the
six-bit leaves; it reads those words, decodes, evaluates, and RNE-encodes each
logical field in the packet exactly once, merges only logical bits, and stores
each owned word once. The stable evaluator uses FP32 intermediates and the
mandated negative half-exponential branch. Logical tiles, cross-word fields,
partial final tile rows, and physical padding are covered by the shared
storage observer.

This boundary is evidenced by the SYCL-only oneAPI run on mirror
`006-tinyllama-09-silu-gate-sycl`: `sycl-ls` enumerated two enabled Level Zero
Intel(R) Arc(TM) Pro B60 Graphics GPUs; configuration and
`cmake --build build --target iom_sycl_conformance_tests` both succeeded; the
anchored `iom_sycl_conformance_tests` CTest target passed 1/1; and the focused
`*SiLU*` selection passed 2/2 with 2,310 assertions. The focused driver also
observed the explicit `F64` rejection without output mutation or admission
side effects. The final shared scenario covered the stored `SiLU -> mul -> add`
boundary without fusion.


## Five-backend SiLU gate receipt (same prepared worktree snapshot)

The four backend rows below were run from the same prepared worktree snapshot:
branch `run-task/006-tinyllama--09-silu-activation--13-silu-five-backend-gate`,
base `HEAD` `185caa78771639054a7a332d4c9291995b670cba`, and worktree
`/home/rlew/iom/src/iom/.work/006-tinyllama/09-silu-activation/13-silu-five-backend-gate`.
The accelerator rows used a fresh `csw-remote-sync` immediately before each
`csw-remote-exec`, the required bounded GPU lock and timeout, and the
profile-specific mirrors `006-tinyllama-09-silu-gate-cuda`,
`006-tinyllama-09-silu-gate-rocm`, and `006-tinyllama-09-silu-gate-sycl`.
The retained helper transcript is
`/home/rlew/iom/src/iom/.cswd/tasks/006-tinyllama/09-silu-activation/13-silu-five-backend-gate/remote.log`.

- **CPU (local AMD Ryzen AI 9 HX 370 / Radeon 890M host).** Build:
  `timeout --kill-after=30s 1800s cmake --build build --target iom_backend_conformance_cpu_tests -j2`.
  Focused:
  `timeout --kill-after=30s 900s ./build/test/iom_backend_conformance_cpu_tests --test-case='*SiLU*'`
  — 2/2 test cases and 3,303/3,303 assertions passed. Complete target:
  `timeout --kill-after=30s 900s ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_backend_conformance_cpu_tests$'`
  — 1/1 passed.
- **CUDA (NVIDIA GeForce RTX 5090, compute 12.0, driver 595.71.05,
  CUDA 13.2).** Complete target:
  `flock -w 600 /tmp/iom-cuda-gpu.lock timeout --kill-after=30s 900s ctest --test-dir build --output-on-failure --timeout 300 -R ^iom_cuda_conformance_tests$`
  — 1/1 passed. Focused:
  `flock -w 600 /tmp/iom-cuda-gpu.lock timeout --kill-after=30s 900s ./build/test/iom_cuda_conformance_tests --test-case="*SiLU*"`
  — 5/5 and 4,127/4,127 assertions passed.
- **ROCm (AMD Radeon AI PRO R9700, `gfx1201`, HIP 7.15.26333,
  ROCm core 10.0).** Complete target:
  `flock -w 600 /tmp/iom-rocm-gpu.lock timeout --kill-after=30s 900s ctest --test-dir build --output-on-failure --timeout 300 -R ^iom_rocm_conformance_tests$`
  — 1/1 passed. Focused:
  `flock -w 600 /tmp/iom-rocm-gpu.lock timeout --kill-after=30s 900s ./build/test/iom_rocm_conformance_tests --test-case="*SiLU*"`
  — 5/5 and 4,172/4,172 assertions passed.
- **SYCL (two Intel Arc Pro B60 Level Zero GPUs, runtime
  `1.15.38646+7`, oneAPI DPC++ 2026.1.0).** Complete target:
  `set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-silu-setvars.log 2>&1; set -u; sycl-ls; flock -w 600 /tmp/iom-sycl-gpu.lock timeout --kill-after=30s 900s ctest --test-dir build --output-on-failure --timeout 300 -R ^iom_sycl_conformance_tests$`
  — 1/1 passed. Focused:
  `set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-silu-setvars.log 2>&1; set -u; sycl-ls; flock -w 600 /tmp/iom-sycl-gpu.lock timeout --kill-after=30s 900s ./build/test/iom_sycl_conformance_tests --test-case="*SiLU*"`
  — 2/2 and 2,310/2,310 assertions passed.

The shared capability assertions cover the 23 stored leaves: CPU/CUDA/ROCm
support all nine applicable signed floating leaves; SYCL supports exactly eight
and explicitly rejects `F64` with its named capability reason; semantic
inapplicable leaves stay `Unsupported`; and unknown enums stay
`InvalidArgument`. The shared stored-boundary case submits SiLU, existing
`mul`, and existing `add` in order, then verifies values plus distinct
owner/native/storage identities without fusion or operation-specific
synchronization.

Reconciliation dispositions are explicit. CUDA and ROCm now use exclusive
16-feature packet ownership, eliminating the former cross-word F6 duplicate
evaluation/RNE-store path; both focused and complete suites pass after that
change. The CUDA documentation now records the observed `linear.cu` compiler
warnings instead of claiming a diagnostics-free build. The shared-suite
sequence/failure/lifetime observations, the leaf-06 zero-class/tail/latch
observations, and the SYCL/ROCm helper-reuse and whole-byte observations do not
change this gate's cross-backend result; they remain covered by the existing
driver-specific controls or are performance/test-hygiene findings, so no
behavioral weakening or unrelated cleanup was introduced here.

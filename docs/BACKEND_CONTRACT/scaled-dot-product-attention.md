# Scaled dot-product attention

This is the operation-owned numerical and reference contract for causal
grouped-query scaled dot-product attention (`SDPA`).  The common queue/OID,
owner-registration, validation-precedence, workspace-liveness, asynchronous
failure, and no-hidden-host-roundtrip rules remain inherited from sections
5–6 and the TinyLlama caller boundary above.  This section fixes the logical
equations and the observable numerical result before any backend result is
measured.  It does not permit an implementation to add a public transpose,
packing, repeated-KV-head, softmax, cache, or session-mutation operation.

## Public ABI, logical layout, and admission

The complete operation surface is exactly the following facade and pure
workspace query:

```cpp
oid sdpa(const TensorView& q, const TensorView& k, const TensorView& v,
         TensorView& out, std::size_t a, std::size_t L,
         RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements sdpa_workspace_requirements(
        const TensorView& q, const TensorView& k, const TensorView& v,
        const TensorView& out, std::size_t a, std::size_t L);
```

For one common, possibly empty, leading tuple `B`, the views have these exact
logical shapes:

```text
q   [B..., Hq, R, D]
k   [B..., Hkv, C, D]
v   [B..., Hkv, C, D]
out [B..., R, Hq*D]
```

The leading tuple is identical for all four views and is never broadcast.
The final axes are logical axes; physical 16x16 tile padding is not a token,
head, feature, or cache element.  `Hq`, `Hkv`, `R`, `D`, and `C` are nonzero,
`Hq % Hkv == 0`, and `Hq*D` is checked before it is compared with the output
feature extent.  All ranks remain in the common interval `2..8`, and every
view independently satisfies its exact shape, device, leaf, quantization,
owner, leading-plane bounds, and stride contract.

Admission requires, in checked arithmetic:

```text
0 < L <= C
a < C
R <= C-a
```

The subtraction is checked before comparison; no wrapping comparison is
permitted.  For every query row `r` in `[0,R)`, the nonempty visible set is

```text
T(r) = { t | 0 <= t < L and t <= a+r }.
```

`L < a+R` is valid and is the ordinary incremental/prefix case.  Attention
uses the explicit initialized length `L`; it never infers length from
capacity, physical allocation, padding, or a session cursor.  The logical
output index is exactly

```text
out[b, r, h*D+d],  0 <= h < Hq, 0 <= d < D.
```

For `h`, the grouped KV head is

```text
g(h) = floor(h / (Hq/Hkv)).
```

There is no repeated KV-head materialization and no public transpose, pack,
head-merge, or extract operation.  A backend may perform device-local
addressing and merge work only inside this operation and only into caller
owned storage or the queried caller workspace.

Q, K, and V are read-only operands.  Exact aliases and harmless read/read
overlap among them are permitted when each aliased view independently
satisfies the shape and view contract.  `out` and workspace are each disjoint
from Q, K, V, and one another, including conservative transformed-owner
overlap.  Admission validates all of these relationships before owner
registration, sequence consumption, output mutation, or backend work.
Only `QuantizationFormat::NONE` is in scope.  Excluded cache tail
`[L,C)`, causal future positions, and physical padding are never read,
validated as logical data, or allowed to change the arithmetic plan.

The requirement query is pure.  It performs no allocation, owner
registration, workspace lease, queue/OID consumption, submission, operand
read, result mutation, or queue-occupancy inspection.  It validates the same
operands, ranges, aliases, capability, quantization, and checked arithmetic
as submission and returns the deterministic backend requirement or throws
the established validation exception.  The supplied workspace is checked
only by submission for exact-device identity, liveness, capacity, alignment,
disjointness, and lease availability.  An accepted request snapshots view
metadata, owner identities, native handles, `a`, `L`, and the workspace range
by value; it retains no borrowed `TensorView` beyond the call.

## Normative equations and logical read boundary

For every independent leading plane `b`, query head `h`, row `r`, and
included token `t in T(r)`, the scalar reference evaluates:

```text
S[h,r,t] =
    (sum_{d=0}^{D-1} Q[h,r,d] * K[g(h),t,d]) / sqrt(D)

m[h,r] = max_{t in T(r)} S[h,r,t]

P[h,r,t] =
    exp(S[h,r,t] - m[h,r])
    / sum_{u in T(r)} exp(S[h,r,u] - m[h,r])
```

`P[h,r,t]` is `+0` when `t` is outside `T(r)`.  The output is:

```text
O[r,h*D+d] =
    sum_{t in T(r)} P[h,r,t] * V[g(h),t,d]
```

Only included `t` participates in PV.  The reference and a conforming
backend MUST NOT scan, sanitize, prefetch, or multiply an excluded V value,
an uninitialized cache tail, or a physical padding cell.  Sizing and launch
semantics use `L`, not `C`.  Leading planes are independent and no result
may cross a plane boundary.

## Nine-leaf semantic table

The semantic table is complete and is not narrowed by a backend storage
capability:

| Leaf | QK/scale/softmax accumulator | P representation before PV | PV accumulator | Destination |
| --- | --- | --- | --- | --- |
| `F4_E2M1` | FP32 | FP32 | FP32 | one final destination RNE |
| `F6_E2M3` | FP32 | FP32 | FP32 | one final destination RNE |
| `F6_E3M2` | FP32 | FP32 | FP32 | one final destination RNE |
| `F8_E4M3FN` | FP32 | FP32 | FP32 | one final destination RNE |
| `F8_E5M2` | FP32 | FP32 | FP32 | one final destination RNE |
| `F16` | FP32 | FP32 | FP32 | one final destination RNE |
| `BF16` | FP32 | BF16 RNE | FP32 | BF16 RNE once at output |
| `F32` | FP32 | FP32 | FP32 | one final destination RNE |
| `F64` | F64 throughout | F64 | F64 | never narrowed |

The nine rows are semantically applicable.  `BOOL`, `I2/U2/I4/U4/I8/U8`,
`I16/U16/I32/U32/I64/U64`, and `F8_E8M0` are inapplicable and remain
explicit rejection classifications.  No inapplicable or unsupported leaf is
converted to BF16 to obtain an apparent success.

## Current capability matrix

The current operation matrix is BF16-only on the four retained backends:

| Backend | `F4_E2M1` | `F6_E2M3` | `F6_E3M2` | `F8_E4M3FN` | `F8_E5M2` | `F16` | `BF16` | `F32` | `F64` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Supported` | `Unsupported` | `Unsupported` |
| CUDA | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Supported` | `Unsupported` | `Unsupported` |
| ROCm | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Supported` | `Unsupported` | `Unsupported` |
| SYCL | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Unsupported` | `Supported` | `Unsupported` | `Unsupported` |

`Supported` in this matrix means that the operation/type pair is the current
target capability and may be advertised only after the backend has supplied
the real device-local implementation and conformance evidence.  Compiler
flags, a storage capability, a rejection-only probe, a host implementation,
or a compile-only sample is not evidence.  A backend or device without that
evidence returns `Unsupported` after structural validation, never a silent
conversion or fallback.  The reference header records the same split through
`sdpa_data_type_classification(DataType)`: `current_supported` is BF16,
`unsupported` is the other eight floating leaves, and `inapplicable` is every
BOOL/integer/`F8_E8M0` leaf.  Every `Supported` cell above is backed by the
executed retained-backend record in
[Same-revision retained-backend SDPA gate evidence](#same-revision-retained-backend-sdpa-gate-evidence);
the eight floating and all inapplicable leaves stay explicit rejection classes
on all four backends.

## BF16 arithmetic, DAZ, underflow, and rounding

For current BF16, Q/K products accumulate in FP32.  Scale, stable masked
softmax, and all sums remain FP32.  The logical FP32 probability is converted
exactly once:

```text
Pbf = BF16_RNE(Pfp)
```

PV uses native BF16 P/V operands with an FP32 accumulator, and output is
rounded once to BF16 at the destination.  Every masked probability and every
stored numerical zero is canonical `+0`.

The matrix-operand DAZ function is exact:

```text
D(x) = copysign(+0, x)  when 0 < abs(x) < 2^-126
       x                 otherwise
```

For BF16, apply `D` only to participating Q/K values before QK and to `Pbf`
and V values before PV, device-locally.  Normal values, infinities, and NaNs
are preserved.  Do not scan or clean excluded capacity.  Do not flush `Pfp`
before BF16 conversion: a subnormal probability that rounds to the smallest
BF16 normal survives the conversion.  Do not apply `D` to final output
storage; BF16 RNE may produce a representable subnormal output.

FP32 products, scale, softmax, and accumulators use round-to-nearest with
gradual underflow.  The scalar reference reduces in increasing logical order
and materializes each declared FP32 step.  Native association or fused
accumulation is allowed only within the declared accumulator domain.  It MUST
NOT add an operand/result flush, saturation, or BF16 partial-product round.
Native QK/PV association may change a cancellation-sensitive formed score's
finite/nonfinite class; no portable arbitrary-case class equality is promised
for such an unpinned case.  The predeclared stable fixtures below avoid
using that as an excuse to weaken their checks.

For F4/F6/F8/F16/F32, QK, scale, softmax, P, and PV use FP32 as shown in the
table, with exactly one final destination RNE.  F64 uses F64 for every
operation, including scale, softmax, and PV, and is never narrowed to FP32.
Finite-only encoding overflow and NaN behavior follows the existing named
scalar-format rules only when a future backend enables that pair; the
reference keeps those eight pairs explicitly unsupported today.

## Formed-score special values and V inclusion

Special policy applies to actually formed scaled FP32 score rows, not to
excluded values:

1. If any included score is NaN, the row's P and output are quiet NaN.
2. Otherwise, if one or more included scores are `+inf`, probability is equal
   over the `+inf` positions and `+0` elsewhere.
3. Otherwise, if every included score is `-inf`, the row's P and output are
   quiet NaN.
4. Otherwise stable max-subtracted softmax is used; `-inf` receives `+0`.

NaN payload and sign are unspecified and ignored by comparison.  Only
included V values participate in PV.  IEEE multiplication therefore makes
an included `+0 * infinity` or `+0 * NaN` poison the affected output
component, while an excluded nonfinite V is never read or multiplied.
Masked and stored zeros are `+0`; explicit signed-zero input fixtures retain
their raw input signs, while destination zero signs follow the canonical
stored-zero rule.

## Workspace, ownership, and inherited queue boundaries

SDPA does not own a hidden allocation or a backend-private result tensor.
Output and workspace are caller-owned, remain disjoint from every operand and
from one another, and stay live through proven completion.  Device-local QK,
softmax, explicit BF16 P preparation, PV, and head merge may use only the
caller workspace reported by the pure operation query.  A positive range is
validated for the exact device, capacity, alignment, owner-absolute
subrange, overlap, and outstanding lease before acceptance; an unknown
completion retains or quarantines it rather than reusing it.  An empty range
is valid only when the backend query reports zero bytes.

The operation inherits common owner registration, exact-alias
deduplication for read/read inputs, FIFO queue admission, sequence/OID
encoding, repeatable waits, retained accepted failures, and no hidden
host synchronization or round trip.  Common admission owns malformed views,
rank/shape/range/device/quantization errors, checked overflow, output and
workspace overlap, resource errors, and negative OID mapping.  The SDPA
reference produces only logical expected values and classifications; it does
not perform admission, workspace allocation, tensor creation, queue work, or
backend setup.

## Independent reference API and fixed comparison policy

The reusable test-side API is
`test/backend/backend_conformance_sdpa_reference.hpp`.  It exposes a stable
`SdpaReferenceCase` containing `DataType`, the exact leading tuple, `Hq`,
`Hkv`, `R`, `D`, `C`, `a`, `L`, and contiguous row-major raw Q/K/V bits; a
`SdpaReferenceValue` containing destination raw bits and a special class;
`evaluate(case)`; `matches(DataType, actual, expected)`;
`sdpa_reference_cases(DataType)`; and
`sdpa_data_type_classification(DataType)`.  The current deterministic fixture
corpus and successful expectations are BF16-only; the table, independent
codec, and accumulator-domain implementation preserve all nine future
precision rules without converting an unsupported leaf to BF16.  The header
independently decodes and encodes F4/F6/F8/F16/BF16/F32/F64, performs FP32 or
FP64 accumulator arithmetic, and derives GQA and causal indexing itself.  It
calls no production SDPA, scalar codec, tile mapper, address helper, CPU
operation, backend queue, or capability declaration.

Finite acceptance is fixed before backend measurement.  For stable
predeclared fixtures proven across QK, softmax, BF16-P rounding, and PV,
`matches` accepts:

```text
max(2 BF16 ULP at the reference value, 2^-10 absolute)
```

This is not a universal cancellation bound.  Explicit special-value and
zero fixtures require exact class and stored-zero sign, with NaN payload and
sign ignored.  Future leaves retain the semantic table's precision and
destination boundary; they are never made current by widening this
tolerance or by converting their operands to BF16.  A deliberately
perturbed expected value outside the fixed margin MUST fail.

## Fixtures and conformance provenance

Before any backend result is measured, the deterministic reference cases pin:

- all-ones and mixed-sign/magnitude values, with non-unit values where
  applicable;
- distinct GQA mappings, exact leading-plane tuples with no broadcast, and
  multiple independent planes;
- nonzero `a`, `L<C`, the valid generic `L<a+R` case, exact `L=C`, and the
  logical row boundaries `R=1,15,16,17`;
- non-tile `D` and `L`, causal future-token perturbation, capacity-tail
  perturbation, and physical-padding perturbation, all requiring bitwise-equal
  outputs when logical inputs and the arithmetic plan are unchanged;
- cached incremental one-row evaluation versus full causal recomputation;
- special formed scores (NaN, equal `+inf`, and all `-inf`), included zero
  probability multiplied by nonfinite V, signed zeros, and canonical `+0`;
- exact BF16 DAZ boundaries `0`, `+/-2^-127`, `+/-2^-126`, normals,
  infinities, and NaNs; and

The focused backend-free analytic test is
`test/backend/backend_conformance_sdpa_reference_test.cpp`, registered as the
single target `iom_backend_conformance_sdpa_reference_tests`.  It checks
equation/indexing identities, GQA groups, causal visibility, exact
special/zero behavior, DAZ and gradual underflow, final RNE, tail/future/
padding independence, cached/full identity, the nine-leaf semantic
classification/codec table, the explicit current matrix classification, and
rejection of a perturbed expectation.  It creates no device, queue, workspace,
backend, or production operation.  The future shared suite consumes this same
header and fixed BF16 fixtures; it MUST NOT duplicate arithmetic, special
policy, expected-value generation, or a second backend matrix.

The focused target is run with:

```text
cmake --build build --target iom_backend_conformance_sdpa_reference_tests
ctest --test-dir build --output-on-failure -R '^iom_backend_conformance_sdpa_reference_tests$'
```

The backend conformance drivers are the vehicles for operation verification:
`iom_backend_conformance_cpu_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, and `iom_sycl_conformance_tests`. This reference
leaf claims no backend result, kernel, queue, workspace-size, or native matrix
evidence. Accelerator execution remains remote-only under the configured
`csw-remote` profiles, with SYCL setup rules inherited from the repository
guidance. Every SDPA port runs the shared oracle on its own backend and retains
explicit `Unsupported` results for the other eight leaves.

## Same-revision retained-backend SDPA gate evidence

The closing four-backend SDPA gate ran the one shared conformance harness
through every retained backend's conformance target and its direct binary at
one revision, the prepared integration base
`668cd13d74dcdd8df87e1b56606d7191dfd980f7`, from one prepared task worktree,
and recorded the device, runtime, toolchain, and profiler identity of each run.
The CPU pair ran locally; every accelerator pair used exact-worktree
`csw-remote` sync/exec with a fresh sync immediately before each execution,
remote-side `timeout --kill-after=30s`, a bounded hardware lock, and
`ctest --timeout 300`. Every row below was then re-run unchanged on the
delivered tree after the documentation edits of this gate, which change no
source file; the recorded timings are those delivered-tree re-runs. The
observed results are:

| Backend | Commands | Device / runtime identity | Observed result |
| --- | --- | --- | --- |
| CPU | `ctest --test-dir build --output-on-failure --timeout 300 -R '^(iom_backend_conformance_cpu_tests\|iom_backend_conformance_sdpa_reference_tests)$'`, the direct `./build/test/iom_backend_conformance_cpu_tests`, its `--test-case='*SDPA*'` selection, and the backend-free reference binary | Local `x86_64` host, AMD Ryzen AI 9 HX 370 w/ Radeon 890M, GCC `15.2.0`, CMake `4.0.2` | `2/2` CTest pass (`13.47 s`); direct binary `34/34` cases and `6,174,802/6,174,802` assertions; SDPA case `1/1` with `2,761/2,761` assertions; reference target `2/2` cases |
| CUDA | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_cuda_conformance_tests$'`, the direct binary, and `--test-case='*TinyLlama*SDPA*decode*'` on remote `bv1` | `NVIDIA GeForce RTX 5090`, compute capability `12.0`, driver `595.71.05`, CUDA Toolkit `13.2` (`nvcc` `V13.2.78`), runtime and driver API `13020`, executed image architecture `1200`, `bf16_wmma=supported` | `1/1` CTest pass (`20.64 s`); `46/46` cases and `6,214,549/6,214,549` assertions; native record `cases=decode,prefill kernel=sdpa_qk_kernel,sdpa_pv_kernel facility=bf16-wmma` |
| ROCm | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_rocm_conformance_tests$'`, the direct binary, and `--test-case='*TinyLlama*SDPA*'` on remote `bv2` | `gfx1201` (AMD Radeon AI PRO R9700; the installed `gfx1036` remains the unproved ordinal), HIP `7.15.26333-0000000`, AMD clang `23.0.0git` | `1/1` CTest pass (`26.39 s`); `48/48` cases and `6,186,178/6,186,178` assertions; native record `facility=gfx1201-wave32-bf16-wmma` with the complete eight-kernel chain |
| SYCL | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_sycl_conformance_tests$'`, the direct binary, and `--test-case='*SDPA*native*'` on remote `bv2` through the nounset-safe no-setup profile override with `sycl-ls` Level Zero enumeration first | Level Zero V2 `Intel(R) Arc(TM) Pro B60 Graphics` (two enumerated), architecture `intel_gpu_bmg_g21`, driver `1.15.38646+7`, subgroup sizes `16,32`, `ext_intel_matrix` present; Intel oneAPI DPC++/C++ `2026.1.0` | `1/1` CTest pass (`22.75 s`); `40/40` cases and `6,116,692/6,116,692` assertions; native-stage record for `shape=decode rows=1 head_dim=3` and `shape=prefill rows=16 head_dim=7` |

The four drivers consume the single shared
`test/backend/backend_conformance_sdpa.hpp` matrix over the independent
`test/backend/backend_conformance_sdpa_reference.hpp` oracle; no backend runs
a private copy of the arithmetic, special policy, expected values, or
capability table, no case was skipped on an enabled device, and no
`Unsupported` leaf is counted as conformance. The automated checks are
`test/cpu/test_cpu_conformance.cpp`, `test/cuda/test_cuda_conformance.cpp`,
`test/rocm/test_rocm_conformance.cpp`, and `test/sycl/test_sycl_conformance.cpp`.

Native QK/PV evidence at prefill and logical `R=1` decode, in addition to the
passing values above:

| Backend | Native facility | Prefill and decode records | Trace or profiler result |
| --- | --- | --- | --- |
| CUDA | Direct BF16/FP32 `wmma` QK and PV stages | `cuda-sdpa-integration-record backend=CUDA cases=decode,prefill kernel=sdpa_qk_kernel,sdpa_pv_kernel facility=bf16-wmma image_arch=1200` — `1/1` case, `72` assertions | `nsys profile --force-overwrite=true -o /tmp/csw-17-cuda-sdpa ./build/test/iom_cuda_conformance_tests --test-case=*TinyLlama*SDPA*decode*`, then `nsys stats --report cuda_gpu_kern_sum`: `sdpa_pv_kernel` with `2` instances and `sdpa_qk_kernel` with `2` instances beside two scale, softmax, and canonicalization instances each (Nsight Systems `2026.4.1.191`). One instance per product at decode and one at prefill, with no duplicate or corrective pass. |
| ROCm | GFX12 wave32 BF16 WMMA QK and PV stages | `rocm-sdpa-integration-record backend=ROCm facility=gfx1201-wave32-bf16-wmma kernels=sdpa_q_pack_kernel,sdpa_k_pack_kernel,sdpa_qk_wmma_kernel,sdpa_softmax_kernel,sdpa_v_pack_kernel,sdpa_pv_wmma_kernel,sdpa_merge_kernel,sdpa_output_store_kernel prefill_rows=17 prefill_position=0 prefill_head_dim=3 decode_rows=1 decode_position=16 decode_head_dim=3` — `1/1` case, `152` assertions | `rocprofv3 --kernel-trace --hip-trace --sys-trace -f csv -d /tmp/csw-17-rocm-prof -- ./build/test/iom_rocm_conformance_tests --test-case=*TinyLlama*SDPA*` exited `0`; the `*_kernel_trace.csv` of that capture contains `4` `sdpa_qk_wmma_kernel` and `4` `sdpa_pv_wmma_kernel` dispatches (`rocprofv3` `1.3.5`). |
| SYCL | Subgroup-16 `ext_intel_matrix` joint-matrix QK and PV stages | `sdpa-native-stage shape=decode rows=1 position=3 length=4 capacity=4 head_dim=3` and `sdpa-native-stage shape=prefill rows=16 position=2 length=17 capacity=19 head_dim=7` — `1/1` case, `254` assertions | `sycl-trace --print-format=verbose --ur.call ./build/test/iom_sycl_conformance_tests --test-case=*SDPA*native*` exited `0`; the Unified Runtime trace creates `...SdpaQkJointMatrixTailKernelTagE` (handle `0x38b21f0`) and `...SdpaPvJointMatrixTailKernelTagE` (handle `0x38b2490`), and each handle appears in exactly two `urEnqueueKernelLaunchWithArgsExp` calls, one per recorded shape. |

CPU's scalar port is the host baseline and supplies no accelerator claim. The
recorded values are the logical fixtures the shared harness submits; the
native records connect them to the executed device-local stages rather than to
host computation, a storage round trip, a padded extra logical token, or an
elementwise substitute. Three stated limitations accompany this record:

1. CUDA hardware-counter collection is denied to this account
   (`ERR_NVGPUCTRPERM`, no passwordless root), so the CUDA observation is a
   kernel summary and claims no unavailable counter.
2. `rocprof` does not exist on the ROCm 7.x host; `rocprofv3`, its supported
   successor with the same kernel-identification surface, produced the kernel
   trace. PC sampling and SPM counters are unsupported on `gfx1201`, so no
   hardware instruction counter is claimed there either.
3. No SYCL ISA-level confirmation was obtained: `clang-offload-extract` on the
   built conformance binary yields the `sycl-spir64` images (the one carrying
   both SDPA tail-kernel symbols compiles with
   `ocloc compile -spirv_input -device bmg`), but `ocloc disasm -dump`
   (`ocloc` `26.22.38646.7`) emits only `sections.txt` with no instruction
   listing, so no `dpas` count is reported. The SYCL record rests on the traced
   dispatch, the created kernel identities, the device's queried
   `ext_intel_matrix`/subgroup-16 facts, and the numerical result.

The supported/unsupported matrix published in
[Current capability matrix](#current-capability-matrix) is the state these runs
observed: `BF16` is `Supported` on CPU, CUDA, ROCm, and SYCL with
`QuantizationFormat::NONE` only, while `F4_E2M1`, `F6_E2M3`, `F6_E3M2`,
`F8_E4M3FN`, `F8_E5M2`, `F16`, `F32`, and `F64` remain explicit
`Unsupported` rejection classes on every backend, as do BOOL, every integer
leaf, `F8_E8M0`, and every non-`NONE` quantization. No failed or unavailable
native path in this record is relabelled as supported, and no case, fixture,
or tolerance was weakened to reach it.

Publishing this gate closes the last missing operation prerequisite of the
mathematical forward sequence: the session and decoder-layer components may
now assemble and verify the complete layer against these operation boundaries.
This publication is evidence and documentation only — it adds no session,
model, kernel, fixture, workspace abstraction, or scheduler code.

## Implementation references and delivery boundary

The common facade and default unsupported hooks are
`include/iom/iom.hpp`, `src/device_ops_neural.cpp`, and the operation's
common admission/request machinery.  Queue ownership, snapshot, completion,
workspace lease, and repeated-wait behavior remain in
`src/device_ops.cpp`, `src/workspace.cpp`, and the shared queue/runtime
helpers.  Standard tiled storage and packed writers remain backend-owned.
This leaf changes only the normative operation section, the independent
reference header, its focused backend-free test, and the one CMake target;
it adds no production API, dispatch, kernel, queue, workspace implementation,
backend capability declaration, model/session/cache/RoPE logic, or host
fallback.

# TinyLlama forward layout — CUDA matrix feasibility

This is a bounded capability record for the planned interfaces above, not a
CUDA neural implementation or a support claim; each interface-owning port
carries its own evidence. On the inventoried device the native CUDA Toolkit
route is **supported for implementation feasibility** for ordinary and
head-planar linear, QK, and PV at every required row count. Production status
is **closed for linear, the CUDA SDPA matrix QK/PV stages, and the end-to-end
CUDA SDPA port** — the operation-owning ports supply the runtime numerical,
conformance, and execution-connected evidence recorded in `[Linear native evidence](#linear-native-evidence)`,
`[SDPA matrix native evidence](#sdpa-matrix-native-evidence)`, and
`[SDPA integration native evidence](#sdpa-integration-native-evidence)` — and
the ROCm and SYCL end-to-end SDPA ports have since supplied the same evidence,
so the four-backend SDPA gate is closed by the
`[Same-revision retained-backend SDPA gate evidence](scaled-dot-product-attention.md#same-revision-retained-backend-sdpa-gate-evidence)`
record; RoPE retains its own status. The implemented SiLU port is closed
separately by its operation-owned gate receipt.
A device below compute capability 8.0 is **unsupported** for this BF16 WMMA
route; it does not earn a fallback pass.

## Linear native evidence

Recorded on 2026-09-18 through the configured `cuda` `csw-remote` profile from
the exact `run-task/006-tinyllama--04-linear-projections--06-cuda-native-bf16`
worktree, mirror `lp-cuda-bf16-a1`, host `bv1`. This record covers the native
BF16 linear specialization (`src/cuda/linear.cu`) only; it makes no QK, PV,
RoPE, SiLU, or SDPA claim.

- **Backend, device, and toolchain.** CUDA; NVIDIA GeForce RTX 5090, compute
  capability 12.0, driver 595.71.05; `nvcc` release 13.2 (`V13.2.78`); runtime
  and driver API 13020 as reported by `cudaRuntimeGetVersion` and
  `cudaDriverGetVersion` inside the conformance binary. The CUDA target pins
  `CUDA_ARCHITECTURES=75;80;90;120`, and the loaded image of the specialization
  reports its own architecture at runtime through the arch-dependent device
  constant of `src/cuda/linear.cu` (`linear_bf16_wmma_image_arch() = 1200`), so
  the executed image is the sm_120 one and not a pre-Ampere JIT image.
- **Kernel and exercised submissions.** Symbol
  `standard_tiled_linear_bf16_kernel` (CUDA-local `__global__`, eight warps of
  32 lanes, one warp per `(plane, head, row tile, column tile)` unit, launched
  by `launch_linear_bf16` on the queue's own in-order stream). The
  conformance binary emitted one record per required row run from the real
  queue/OID path (`cuda-linear-bf16-record ...` lines of
  `test/cuda/test_cuda_conformance.cpp`), each with `P=2`, `T=19`, `I=3`,
  `O=10`, `s=2`, padded `Rp/Op/Ip = 16/16/16` for `R=1,15,16` and `32/16/16`
  for `R=17`, accepted OID `36028797018963969` at sequence 1 of its fresh
  queue, and launch geometry `blocks=2/2/2/4` (ordinary, `H=1,D=10`) and
  `blocks=4/4/4/8` (head-planar, `H=2,D=5`) for `R=1,15,16,17`; every run was
  compared element-by-element against the shared independent reference and
  passed. A device below compute capability 8.0, or a build whose loaded image
  carries no BF16 WMMA statements, reports the leaf `Unsupported` instead of
  producing a record, and the difference is verified through the injected
  capability fact in
  `test/cuda/test_cuda_conformance.cpp` (`CUDA BF16 linear is Unsupported on a
  device without the WMMA facility`).
- **Observed native facility.** The executed `wmma` BF16/FP32 operation behaves
  as the tensor-core datapath and not as any per-step FP32 accumulation: for
  the discriminative product set `{+4.014e38, +4.014e38, -7.603e38}` (every
  product beyond `FLT_MAX`, exact sum `4.256e37`) the device returns
  `0x1p+125 = 4.2535e37`, while every left-to-right or tree FP32 accumulation
  order of the same products overflows to `±Inf`; BF16-subnormal products are
  preserved (`2^-133` exactly, no flush-to-zero), `NaN` multiplicands admit
  `NaN`, and `(+Inf, -Inf)` products give `NaN` while `+Inf` plus a finite
  product stays `+Inf`. The compiled image of that kernel contains
  `HMMA.16816.F32.BF16` (`cuobjdump -sass build/libiom_cuda.a`, six
  occurrences across the three architectures at or above the floor and none in
  the sm_75 image). The disassembly is supplementary here: capability is
  decided at runtime from the device attribute and the loaded image, never from
  compilation or disassembly presence.
- **Profiler.** `ncu` 2026.1.1.0 is installed, but counter collection is denied
  to this account (`ERR_NVGPUCTRPERM`; no passwordless root), so no
  counter-based instruction-level observation was obtained. `nsys` 2026.4.1
  does run: `nsys profile --force-overwrite=true -o /tmp/lp-nsys` around the
  native-evidence test case, read back with
  `nsys stats --report cuda_gpu_kern_sum`, reports the executed symbol
  `iom::cuda_detail::<unnamed>::standard_tiled_linear_bf16_kernel(iom::detail::LinearMetadata)`
  with eight instances — one per recorded submission, in the same order as the
  eight emitted records — beside the shared copy and gather kernels. The record
  therefore reports the executed kernel, its loaded image architecture, and the
  observed facility behaviour above, and claims no counter-based instruction
  observation; the missing counters are a host-permission limitation, not a
  device or implementation result.
- **Fixture class agreement.** The complete BF16 case matrix (33 cases, 45023
  reference elements) is scanned for the invariant that the mandated FP32
  fused-multiply-add recurrence and the FP64 equation rounded once to BF16
  agree in class and satisfy the frozen threshold:
  `class_disagreements=0`, `threshold_failures=0`. The scan is what motivated
  the fixture scale constraint recorded on `linear_fixture_special_code` in
  `test/backend/backend_conformance_linear.hpp`, which scales the BF16
  saturation-magnitude ladder position to `2^48`; before that constraint
  exactly one element (the LM-head case, element 11) disagreed, `+inf` against
  `-inf`, because no reassociating native reduction reproduces per-step FP32
  overflow.
- **Conclusion.** Supported on this device and this loaded image for `R=1`,
  `15`, `16`, and `17` in ordinary and head-planar mode, each established by an
  executed submission whose output was compared against the independent
  reference; a device without the facility reports `Unsupported`.

## SDPA matrix native evidence

Recorded on 2026-09-20 through the configured `cuda` `csw-remote` profile
from the exact
`run-task/006-tinyllama--08-causal-grouped-attention--05-cuda-matrix-products`
worktree, mirror `05-cuda-matrix-a1`, host `bv1`. This record covers the
backend-private CUDA matrix stages (`src/cuda/sdpa_matrix.cu`) only; it makes
no claim for CUDA scale, causal masking, softmax, probability preparation,
RoPE, or end-to-end SDPA integration.

- **Backend, device, and toolchain.** CUDA; NVIDIA GeForce RTX 5090,
  compute capability 12.0, driver 595.71.05; CUDA Toolkit 13.2
  (`nvcc` `V13.2.78`), with the direct BF16/FP32 WMMA route selected at
  runtime for CC >= 8.0.
- **Kernel and exercised submissions.** The private native symbols
  `sdpa_qk_kernel` and `sdpa_pv_kernel` were both emitted by the focused
  conformance record for logical `R=1,15,16,17`, `L=19,15,16,19`,
  respectively, with `P=2`, `Hq=4`, `Hkv=2`, `C=24`, and non-tile `D=7`.
  The fixtures use transformed leading-plane offsets and strides, grouped
  KV heads, logical/padded tails, BF16 DAZ probes, and direct merged-output
  checks. The focused selection passed with `19196/19196` assertions, and the
  shared `*SDPA*` selection passed with `19533/19533` assertions.
- **Conformance gate.** The configured target
  `ctest --test-dir build --output-on-failure --timeout 300 -R
  ^iom_cuda_conformance_tests$` passed on the same mirror (`1/1` test,
  `78.72s`).
- **Profiler.** Nsight Compute exercised the focused matrix selection and
  reported `ERR_NVGPUCTRPERM`; counter collection is denied to this account,
  so no counter-based instruction observation was obtained. Nsight Systems
  successfully captured `/tmp/sdpa-matrix-a1.nsys-rep`; its
  `cuda_gpu_kern_sum` reports `sdpa_pv_kernel` with 4 instances and
  `sdpa_qk_kernel` with 4 instances. This proves both native matrix stages
  ran for the decode and prefill fixture submissions without claiming
  unavailable performance counters.
- **Conclusion.** The private CUDA QK/PV matrix stages are closed on this
  device for the exercised decode and prefill rows; at that record full CUDA
  SDPA was still gated on its nonmatrix and integration ports, which the
  following record closes.

## SDPA integration native evidence

Recorded on 2026-09-20 through the configured `cuda` `csw-remote` profile from
the exact
`run-task/006-tinyllama--08-causal-grouped-attention--07-cuda-integration`
worktree, mirror `csw-run-006-08-07-cuda-integration`, host `bv1`. This record
closes the end-to-end CUDA SDPA port — the exact `sdpa` facade and pure query,
the CUDA policy seam of `src/cuda/copy.hpp`, the stage chain of
`src/cuda/sdpa.hpp`/`src/cuda/sdpa.cpp`, and the shared SDPA branch of
`src/shared/gpu_queue.hpp`, `src/shared/gpu_queue_operations.inl`, and
`src/shared/gpu_queue_lifecycle.inl`. It makes no ROCm, SYCL, CPU, RoPE,
session, or four-backend claim.

- **Backend, device, and toolchain.** CUDA; NVIDIA GeForce RTX 5090, compute
  capability 12.0, driver 595.71.05; CUDA Toolkit 13.2 (`nvcc` `V13.2.78`),
  with runtime and driver API 13020 as reported by `cudaRuntimeGetVersion` and
  `cudaDriverGetVersion` inside the conformance binary, and the executed image
  reporting its own architecture through `linear_bf16_wmma_image_arch()` =
  `1200`. The port is selected only on the queue policy of a device that
  proved the BF16 WMMA facility; a queue whose construction observed the
  injected facility absence reports the operation `Unsupported` for both the
  pure query and the submission and leaves the output byte-exact.
- **Public path, workspace, and exercised submissions.** `DeviceOps::sdpa`
  and `DeviceOps::sdpa_workspace_requirements` through the real queue/OID
  path: one full causal prefill (`Hq=2,Hkv=1,R=4,D=3,C=4,a=0,L=4`) and one
  logical `R=1` decode (its last row, `a=3`) were each accepted with a
  caller-owned `alignment 32` workspace and waited twice; every output element
  was compared against the shared independent reference, and each decoded row
  matched the corresponding prefill row bit for bit. The queried layout is
  exactly `A32(P*Hq*pad16(R)*pad16(L)*4)` for the FP32 score segment plus
  `A32(P*Hq*pad16(R)*pad16(L)*2)` for the BF16 probability segment that begins
  at the checked score offset, at the fixed 32-byte alignment; the same value
  is returned by the pure query and validated by the accepted stage chain.
- **Native stages and resource use.** The single in-order queue stream runs
  `sdpa_qk_kernel` then the device-local `scale_mask_kernel`,
  `softmax_kernel`, `sdpa_pv_kernel`, and `canonicalize_output_kernel`; an
  accepted prefill/decode pair creates no additional stream and no additional
  completion resource (asserted around that pair with the CUDA resource
  counters), so the port adds no hidden stream, event, or allocation.
- **Conformance gate.** `ctest --test-dir build --output-on-failure --timeout
  300 -R '^iom_cuda_conformance_tests$'` passed on that mirror (`1/1`,
  `79.49s`), and the direct binary reports `46/46` cases and
  `6,214,411/6,214,411` assertions with no skipped case. That selection covers
  the shared SDPA matrix (grouped KV heads, causal/initialized-prefix and
  capacity-tail isolation, physical-padding perturbation, `R=1,15,16,17`,
  non-tile `D`, multiple leading planes, transformed head strides, cached
  incremental rows, the negative admission matrix, workspace/alias/overflow
  rejection, and accepted-failure lifetime), the semantic probes — an included
  `+0 * infinity` probability poisons its output component with a quiet NaN,
  and `2^-127` survives the final BF16 RNE store — and the CUDA BF16 native
  evidence cases. `iom_cuda_smoke_tests` additionally passed with `23/23` cases
  and `462/462` assertions on the same mirror.
- **Accepted failure and workspace lifetime.** An injected native stage launch
  fault after acceptance keeps one positive OID, leaves the output unusable,
  rethrows the same error on every repeated wait, and the next submission
  recovers. A second accepted submission of the same live workspace range is
  bounded-resource exhaustion until the caller observes the first token, whose
  observation releases the range again; the shared workspace case was repeated
  twelve times on that mirror without a single failure, and the earlier
  proof-based release of the same branch failed ten of twelve runs, which is
  why a successful accepted submission now retains its lease until its token is
  observed and queue teardown releases whatever no caller observed.
- **Profiler.** Nsight Compute is installed but counter collection is denied
  to this account (`ERR_NVGPUCTRPERM`; no passwordless root), so no
  counter-based instruction observation was obtained. Nsight Systems captured
  `/tmp/cuda-sdpa-integration.nsys-rep`; its `cuda_gpu_kern_sum` reports
  `sdpa_qk_kernel` with 2 instances and `sdpa_pv_kernel` with 2 instances —
  one per product at decode and one at prefill — beside the two scale, softmax,
  and canonicalization instances. This proves both native matrix stages ran for
  decode and prefill through the public request and that the chain contains no
  duplicate or corrective pass; it claims no unavailable performance counters.
- **Conclusion.** The CUDA BF16 SDPA port is closed on this device for the
  exercised decode and prefill rows and for the shared conformance matrix. The
  ROCm and SYCL ports have since supplied the same evidence, so the
  four-backend SDPA gate is closed by the same-revision gate record below.

## Evidence boundary and installed capability

The following evidence was collected on 2026-09-14 through the configured
`cuda` `csw-remote` profile in the unique
`forward-layout-cuda-feasibility` workspace:

- `nvcc --version` reported CUDA compilation tools 13.2,
  `V13.2.78`; `/usr/local/cuda` resolved to `/usr/local/cuda-13.2`.
  Installed package inventory reported `cuda-compiler-13-2` 13.2.1-1,
  `cuda-nvcc-13-2` 13.2.78-1, and `cuda-cudart-dev-13-2` 13.2.75-1.
  `dpkg-query -S` assigned both `mma.h` and `cuda_bf16.h` under the
  13.2 target include directory to `cuda-cudart-dev-13-2`.
- `nvidia-smi` reported driver 595.71.05, driver-supported CUDA 13.2,
  and one NVIDIA GeForce RTX 5090. A bounded Runtime API inventory executable
  reported header `CUDA_VERSION=13020`, `CUDART_VERSION=13020`, runtime
  13020, driver API 13020, compute capability 12.0, warp size 32, and
  33,670,758,400 bytes of global memory.
- A temporary installed-header translation unit instantiated
  `<mma.h>` BF16/FP32 fragments for `16x16x16` with all four
  row-major/column-major A/B combinations, and `32x8x16` and `8x32x16`
  with row-major A and column-major B. It used `__nv_bfloat16`
  multiplicands, `float` accumulators, `load_matrix_sync`, `mma_sync`, and
  row-major `store_matrix_sync`. It compiled successfully with
  `nvcc -std=c++20 -arch=sm_120`; its executable performed only the device
  inventory above. The translation unit was removed after the bounded sample.
  This is installed SDK/header and target-code-generation evidence, **not**
  a matrix launch, numerical result, throughput result, or profiler result.

The [CUDA 12.8 Warp Matrix Functions documentation][cuda-wmma] defines WMMA
as warp-cooperative `D=A*B+C`, permits `row_major` and `col_major`
multiplicands, and permits a row- or column-major accumulator store. Its
[alternate-floating-point section][cuda-wmma-alt] requires compute capability
8.0 or newer for BF16 Tensor Core use and requires `__nv_bfloat16` fragments
to use `float` accumulators. Its
[element-type and matrix-size table][cuda-wmma-sizes] lists BF16/FP32
`16x16x16`, `32x8x16`, and `8x32x16`. The canonical route below uses only
`16x16x16`; the other shapes are optional tuning choices, not semantic
requirements.

`load_matrix_sync` and `store_matrix_sync` require a 256-bit (32-byte)
aligned base. `ldm` is in elements and describes the distance between
successive rows or columns. The guide requires a 16-byte stride for the
listed 16-bit and FP32 cases, and states that BF16 has the same shapes and
operations as FP16. This route therefore uses BF16 leading dimensions
divisible by 8 and FP32 leading dimensions divisible by 4; padding every
native dimension to 16 satisfies both. Each warp sees identical fragment
parameters and participates unconditionally. Fragments never cross a
translation-unit or external ABI boundary; the guide explicitly calls their
register layout architecture-specific.

These installed and documented facts establish that the primitives needed by
a direct port exist on the sampled host. They do not establish current IOM
support. The repository currently links the CUDA target only to
`CUDA::cuda_driver` and `CUDA::cudart`, allocates exact-context standard-tiled
storage, uses one in-order CUDA stream per queue, and leaves the neural
capability probes unsupported. No CUDA linear, QK, PV, end-to-end
conformance, sanitizer, or profiler command was run for this assessment.
The repository facts above are bounded by `CMakeLists.txt`'s CUDA target,
`src/cuda/device_tensor.cpp`'s exact-context standard-tiled allocation,
`src/cuda/copy.cu`'s shared queue factory, device enumeration in
`test/cuda/test_cuda_smoke.cpp`, and the still-unsupported neural probes in
`test/cuda/test_cuda_conformance.cpp`.

## Fixed ABI and native lowering

This route changes none of the preceding ABI. Calls return `oid`, are
`noexcept`, take const input views, a mutable caller output, `std::size_t`
positions and extents, and an optional final `RawWorkspaceView`. Their
workspace queries have the same semantic arguments, a const output, and no
workspace. A query validates shapes, aliases, capability, and every checked
size but never allocates, registers, submits, consumes an OID, inspects a
handle, or depends on arena capacity. Submission additionally validates the
workspace's exact device, capacity, 32-byte alignment, nonoverlap, and lease.
Negative OIDs report pre-acceptance failure; an accepted device failure
remains attached to its positive OID on every repeated wait.
Only `QuantizationFormat::NONE` is admitted. Valid read/read aliases remain
explicitly supported; output/read and workspace/operand/output overlap remain
invalid.

Let `P` be the checked product of the complete leading tuple (one when it is
empty), and define, with checked addition and multiplication,

```text
pad16(x) = checked_add(x, 15) / 16 * 16
A32(n)   = checked_align_up(n, 32)
Rp=pad16(R), Ip=pad16(I), Op=pad16(O),
Dp=pad16(D), Lp=pad16(L).
```

All runtime ranks remain 2 through 8 and dimensions are nonzero. A CUDA kernel
maps each transformed leading coordinate through its snapshotted owner offset
and strides; it never collapses or broadcasts independent planes. Standard
IOM storage is tile-major, with row-major values inside each 16x16 tile.
That is not one globally contiguous WMMA matrix, and an arbitrary `s` or
transformed plane is not a legal aligned WMMA base. Each warp therefore
copies only its current logical tile into an `alignas(32)` kernel-local shared
tile, writes zero to out-of-range physical cells, synchronizes, and loads the
WMMA fragment with leading dimension 16. This bounded on-chip tile is not an
allocation, persistent weight copy, or caller workspace. It also prevents
perturbed owner padding, uninitialized cache capacity, and excluded keys from
being read as multiplicands.

For linear, A is selected `x[b,s+r,i]` in row-major order. A rank-two
HF `w[O,I]` tile is staged without transposing checkpoint storage and is
interpreted as column-major B, so `B[i,o]=w[o,i]`. The result fragment is
stored to an aligned FP32 shared tile, converted once with RNE, and scattered
directly to ordinary `out[b,r,o]` or head-planar
`out[b,h,r,d]`, `o=h*D+d`. There is no bias, hidden row extraction, complete
weight pack, duplicate checkpoint allocation, or host work.

For QK, A is row-major Q `[R,D]`; the selected
`K[g(h),0:L,D]` tile is column-major B representing `K` transposed, where
`g(h)=floor(h/(Hq/Hkv))`. Only `t<L` and `t<=a+r` survives the separate
FP32 scale/mask step. For PV, rounded BF16 P is row-major A and the selected
V `[L,D]` tile is row-major B. K and V are addressed at their one owning
`Hkv` head for every query head; no buffer materializes `Hq/Hkv` copies.
QK and PV accumulate FP32. Stable max-subtracted softmax, exact zero for
masked P, RNE FP32-to-BF16 probability preparation, final RNE conversion, and
head merge all remain CUDA kernels on the same queue stream.

The logical/physical row mapping and status are:

| Logical `R` | Physical WMMA M | Ordinary linear | Head-planar linear | QK | PV |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |
| 15 | 16 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |
| 16 | 16 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |
| 17 | 32 | supported feasibility | supported feasibility | supported feasibility | supported feasibility |

The same 16-padding applies independently to non-tile `I`, `O`, `D`, and
`L`. In linear the native problem is `(Rp,Op,Ip)`; in QK it is
`(Rp,Lp,Dp)`; in PV it is `(Rp,Dp,Lp)`. For `R=1/15`, physical rows through
15 are neutral staging only; for `R=17`, rows 17 through 31 are neutral.
They are never logical token rows, never enter a softmax reduction, and never
reach output. The analogous column and reduction tails are zero
multiplicands or discarded FP32 results. Public K/V owners remain
`[P,Hkv,C,D]`; packing never changes them to public `[P,Hkv,L,D]` views.
All four modes still require `Hq%Hkv=0`, `0<L<=C`, `a<C`, and
`R<=C-a`, and only `0<=t<L,t<=a+r` contributes.

## Checked data flow and scratch

Every expression in this table is evaluated with checked products before
bytes are formed. `logical bytes` accounts for the required payload even when
the data stays in an existing owner. `segment` names caller workspace only;
kernel-local 16x16 shared tiles have a fixed launch-time size and lifetime.

| Product | Existing owner and logical bytes | Caller-owned temporary / checked segment | Native invocation and caller-owned result |
| --- | --- | --- | --- |
| Ordinary linear | selected BF16 X is `P*R*I*2`; persistent BF16 W is `O*I*2` logical bytes and is never scratch | none; tile-local staging handles `s`, strides, and neutral tails; query may return zero bytes | BF16/FP32 WMMA `(Rp,Op,Ip)`; RNE directly to standard-tiled `out[P,R,O]`, `P*R*O*2` logical bytes |
| Head-planar linear | the same X and persistent W; `O=H*D` is checked | none; no complete weight or output pack | the same WMMA product; RNE/scatter to `out[P,H,R,D]`, checked `P*H*R*D*2 = P*R*O*2` |
| QK | BF16 Q logical bytes `P*Hq*R*D*2`; K owner remains `P*Hkv*C*D*2`, while only its initialized causal prefix is staged tile-by-tile | `scores`: `A32(P*Hq*Rp*Lp*4)` bytes, 32-byte aligned; FP32 QK/scale/mask, alive from QK through probability preparation | BF16/FP32 WMMA `(Rp,Lp,Dp)` per query head and `g(h)`; logical FP32 scores account for `P*Hq*R*L*4`; excluded/padded cells are masked and are not tokens |
| Softmax / P preparation | logical FP32 scores are `P*Hq*R*L*4` | `probability`: `A32(P*Hq*Rp*Lp*2)` bytes after `scores`; stable FP32 row reduction writes exact-zero masked, RNE BF16 P; alive until PV completion | no matrix substitute: a device reduction/conversion kernel produces the observable BF16 boundary, whose logical payload is `P*Hq*R*L*2` |
| PV and merge | probability segment plus BF16 V owner `P*Hkv*C*D*2`; only mapped initialized V tiles are read | no KV-head expansion and no full PV buffer; an FP32 accumulator tile is kernel-local | BF16/FP32 WMMA `(Rp,Dp,Lp)`; the logical planar PV payload is `P*Hq*R*D*2` after RNE and is scattered directly, without duplicating it, to merged `out[P,R,Hq*D]`, also `P*R*Hq*D*2` |

The deterministic SDPA query therefore returns

```text
alignment = 32
score_bytes = A32(P*Hq*Rp*Lp*4)
probability_bytes = A32(P*Hq*Rp*Lp*2)
bytes = checked_add(score_bytes, probability_bytes)
```

with `probability` beginning at checked offset `score_bytes`. Both segments
belong to one live `RawWorkspace` on the queue's exact CUDA device and are
disjoint from Q, K, V, and output. Valid Q/K/V read/read overlap, including
exact aliases that independently satisfy their shape contracts, remains
allowed; output/read, scratch/read, scratch/output, and conflicting scratch
lease overlap is rejected. Linear needs no global temporary in this route,
so an empty workspace is valid exactly when its query reports zero.

Scores and P are per query head because those values differ under GQA; K and V
are not copied per query head. The score and probability segments coexist
while P is prepared. Once QK scores are no longer needed the score segment
may be reused only after the queued phase proves that lifetime transition;
the probability segment and all owners remain live through PV and its
completion event. Across submissions, neither segment is reusable before the
accepted OID completes successfully or fails terminally. Unknown completion
retains or quarantines the lease. Queue submission uses the existing stream,
snapshots metadata instead of retaining borrowed views, records completion
after all packing/matrix/reduction/merge kernels, preserves in-order OIDs,
and does not hide a stream synchronization. Sessions must still wait every
producer individually before submitting its consumer and drain every accepted
OID after a failure.

## Seven-operation fit and incompatibilities

| Interface operation | CUDA assessment |
| --- | --- |
| gather | Contract-compatible device kernel for integral index payloads and bit-preserving BF16 table values; a device-discovered bad index must become an accepted retained failure. WMMA is irrelevant. A host index scan or round trip is forbidden. Current implementation remains blocked. |
| matmul | Supported feasibility on the sampled CC 12.0 device by direct BF16/FP32 WMMA with tile-local standard-layout staging. Global owner layout, arbitrary `s`, transformed leading strides, and physical tails are incompatible with direct unguarded `load_matrix_sync`; the bounded staging above is required. |
| reduction | Contract-compatible device-local FP32 reduction for linear accumulation helpers, RMSNorm, and stable softmax. Reductions exclude masked/padded cells and retain the specified wide intermediates; WMMA does not replace max, sum, or normalization. The CUDA RMSNorm wrapper is source-inspected, the linear accumulation helpers are implemented through the landed port (see [Linear projections](linear-projections.md#linear-projections)), and the CUDA SDPA nonmatrix scale/mask, stable softmax, BF16 probability preparation, and merged-output canonicalization are implemented by `src/cuda/sdpa_nonmatrix.cu` and independently passed the remote `iom_cuda_sdpa_nonmatrix_tests` device smoke; QK, PV, and the complete attention operation are closed by their own CUDA records above. |
| trig | Contract-compatible CUDA device FP32/wide-domain sine and cosine with finite positive `theta`; it neither uses nor is evidenced by WMMA. A host math substitute is forbidden. Current RoPE kernel remains blocked. |
| partial-tile copy | Existing CUDA standard-tiled copy machinery establishes device-local tile addressing, but not neural matrix support. A native port must use guarded logical loads/stores and neutral shared cells so owner padding, cache capacity tail, and rows outside `R` are unobservable. |
| unary | Contract-compatible CUDA device FP32/wide-domain SiLU and BF16 RNE output. It is elementwise by design and is not an invalid matrix substitute. Current kernel remains blocked. |
| attention | Supported matrix feasibility for both QK and PV, including GQA and every required `R`, only as the complete device-local flow above. FP32 masking/softmax and explicit P-to-BF16 preparation are mandatory. Elementwise QK/PV, repeated KV heads, host work, hidden allocation, or treating physical rows as tokens is incompatible. The CUDA SDPA port and its prefill/decode evidence are closed by its operation-owned records above. |

This matrix result says nothing about the other applicable dtype leaves; their
kernel selection, numerics, and tolerances remain with the operation-owned
specifications. In particular this record neither adds nor narrows any F64
path.

## Dependency decision and remaining gates

**Decision: no added matrix library.** The installed `<mma.h>`,
`cuda_bf16.h`, compiler, driver, and runtime provide the required direct
primitive, orientation combinations, target code generation, and queue-local
kernel launch route. The project already discovers the CUDA Toolkit and links
the CUDA backend privately to the driver and runtime. The only identified
layout mismatch is solved by bounded device shared-memory tiles and checked
caller scratch; it is not a reason to add a BLAS dependency. Absence of
cuBLAS linkage is therefore not a gap, cuBLAS is not a default fallback, and
the first CUDA linear/SDPA ports must reuse this backend-private direct route.

This decision also avoids a concrete lifecycle conflict. The
[cuBLAS `cublasCreate`/`cublasDestroy` documentation][cublas-create] says
creation allocates host and device resources and destruction implicitly calls
`cudaDeviceSynchronize`. [`cublasSetStream`][cublas-stream] unconditionally
resets a user workspace to the default pool.
[`cublasSetWorkspace`][cublas-workspace] binds user-owned device memory only
for the current stream, requires 256-byte alignment, and warns that calls must
be serialized while kernels retain it. Those handle allocations, default-pool
fallback, stream rebinding, and destruction synchronization are not
allocation-neutral conveniences and cannot override caller-owned scratch,
exact-stream lifetime, or asynchronous teardown rules. If a future consuming
port finds a direct-WMMA blocker, it must first document that concrete blocker
and reject any library route that cannot write caller output, bind
deterministic caller scratch, initialize per-context bookkeeping before
submission, rebind after every stream change, and retain resources through
completion. No such blocker is evidenced here.

The following are exact reproducibility and future production commands, run
from the assigned local worktree through the same unique remote workspace:

```sh
.omp/csw/bin/csw-remote-sync cuda forward-layout-cuda-feasibility
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility 'nvcc --version && nvidia-smi'
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader'
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility "dpkg-query -W 'cuda-*'"
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility 'cmake --build build --target iom_cuda_conformance_tests'
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility "ctest --test-dir build --output-on-failure -R '^iom_cuda_conformance_tests$'"
.omp/csw/bin/csw-remote-exec cuda forward-layout-cuda-feasibility "ncu --set full --target-processes all --kernel-name regex:'.*(linear|qk|pv).*' ./build/test/iom_cuda_conformance_tests --test-case='CUDA TinyLlama BF16 matrix paths cover R=1,15,16,17'"
```

The first four inventory/capability steps were run for this assessment; the
last three are deliberately unrun production-port gates. The future
conformance case must cover ordinary and head-planar linear plus both QK and
PV at logical `R=1,15,16,17`, non-tile `I/O/D/L`, independent transformed
planes, GQA, perturbed physical padding and excluded K/V rows, exact
BF16/FP32/RNE boundaries, every scratch rejection, accepted failure with
repeat waits, and no host traffic or hidden allocation. The profiler must
show the direct matrix kernels for both decode and prefill; inventory,
successful compilation, or one GEMM cannot substitute for that evidence.

[cuda-wmma]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-matrix-functions
[cuda-wmma-alt]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#alternate-floating-point
[cuda-wmma-sizes]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes
[cublas-create]: https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
[cublas-stream]: https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream
[cublas-workspace]: https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace

# TinyLlama forward layout — ROCm matrix feasibility

This is a bounded feasibility record for the planned ABI above, not a ROCm
implementation or a support advertisement for the operations covered below.
The existing ROCm neural probes not covered by a retained operation port
continue to return negative `Unsupported`; the implemented SiLU port is
covered by its operation-owned gate receipt. The observations below do not
replace operation conformance, a production kernel run, or profiler evidence.
They establish that compiler-native BF16 matrix instructions are a viable
implementation route on a checked supporting target without adding a matrix
library.

## Evidence boundary and installed target

The evidence was collected on the configured `rocm` host from the exact
task worktree, synchronized to the unique remote workspace
`forward-layout-rocm-feasibility`. The following evidence classes are kept
separate:

| Evidence class | Observation | What it does not prove |
| --- | --- | --- |
| Project source declaration | `CMakeLists.txt` requires HIP at least 7.2 and privately links `hip::host`; ROCm storage uses the standard tiled allocation and the existing in-order GPU queue. | A version constraint, storage support, or link target does not prove a native matrix kernel. |
| Installed compiler/SDK | `hipcc --version` reported HIP `7.15.26333-0000000`, AMD Clang `23.0.0git`, commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`; `hipconfig --full` reported `HIP_PATH=ROCM_PATH=/opt/rocm/core-10.0`; `.info/version` reported ROCm `10.0.0`; `hip_version.h` reports `7.15.26333`; Clang's resource directory is `/opt/rocm/core-10.0/lib/llvm/lib/clang/23`. The installation is not represented by the queried Debian `hip-*`/`rocm-*` package names, so those names are not used as version evidence. | Upstream-main or ROCm-7.2.4 source is not assumed to equal this installed compiler. |
| Installed target definitions | Installed generated `clang/Basic/BuiltinsAMDGPU.inc` exposes the BF16-to-FP32 GFX11 WMMA `_w32`/`_w64`, GFX12 `_w32_gfx12`/`_w64_gfx12`, and BF16 MFMA names. Its guards are respectively `wmma-256b-insts` plus the selected wave size, `wmma-128b-insts` plus the selected wave size, and `mai-insts`. | A name in a compiler table is not proof that it is invocable on every target. |
| Installed runtime/device | `rocminfo` reported HSA runtime `1.21`, runtime extension `1.30`, a Radeon AI PRO R9700 `gfx1201` agent with wavefront size 32, and a separate `gfx1036` agent. `hipGetDeviceProperties` enumerated ordinal 0 as the R9700 (`gfx1201`, `warpSize=32`, 32 reported multiprocessors, 1024 maximum threads per block, 34,208,743,424 bytes global memory) and ordinal 1 as `gfx1036`; `hipRuntimeGetVersion` and `hipDriverGetVersion` both returned `70152801`. AMD's [GPU specification table](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html) independently identifies the R9700 as RDNA4/gfx1201 with wave32 or wave64. | The inventory does not make the older `gfx1036` matrix-capable and does not benchmark the R9700. |
| Bounded capability samples | `hipcc --offload-arch=gfx1201 -O2` compiled a 32-lane kernel calling `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`; execution on ordinal 0 completed and preserved an FP32 accumulator for a zero BF16 product (`gfx12_w32_bf16_wmma_zero_product=pass`). The same installed compiler compiled, but did not run, a `--offload-arch=gfx1100` kernel calling the GFX11 `_w32` builtin with its replicated input vector shape. Both temporary `_local` sources and the remote workspace were removed. | The zero-product sample proves target compilation, dispatch, all-wave participation, and FP32 result transport only. It is not a layout, tail, numerical, throughput, or production conformance test; the GFX11 result is compile-only. |
| Deliberately unrun production gates | No `iom_rocm_conformance_tests` build or CTest run, production native linear/QK/PV kernel, numerical matrix, profiler, tuning run, or four-backend gate was executed. | No current ROCm neural support or performance conclusion follows. |

The exact inventory command was:

```text
.omp/csw/bin/csw-remote-exec rocm \
  forward-layout-rocm-feasibility 'hipcc --version && rocminfo'
```

The bounded runtime probe was compiled with
`hipcc --offload-arch=gfx1201 -O2` and used
`hipRuntimeGetVersion`, `hipDriverGetVersion`,
`hipGetDeviceProperties`, a 32-thread launch, and
`hipDeviceSynchronize`. A separate compile-only probe used
`hipcc --offload-arch=gfx1100 -O2 -c`. These observations are reproducible
installed facts, while the eventual production commands remain:

```text
cmake --build build --target iom_rocm_conformance_tests
ctest --test-dir build --output-on-failure \
  -R '^iom_rocm_conformance_tests$'
```

Those commands must run remotely after a port exists and must then cover
native linear, QK, and PV at logical prefill and `R=1`; they were not run for
this assessment.

## Native instruction route

The [ROCm 7.2.4 release-pinned builtin definitions](https://raw.githubusercontent.com/ROCm/llvm-project/rocm-7.2.4/clang/include/clang/Basic/BuiltinsAMDGPU.def)
define BF16-to-FP32 WMMA `_w32` and `_w64`, BF16 MFMA including
`__builtin_amdgcn_mfma_f32_16x16x8bf16`, and the GFX12-suffixed WMMA
variants. The current [upstream definitions](https://raw.githubusercontent.com/llvm/llvm-project/main/clang/include/clang/Basic/BuiltinsAMDGPU.td)
and [builtin documentation](https://raw.githubusercontent.com/llvm/llvm-project/main/clang/include/clang/Basic/BuiltinsAMDGPUDocs.td)
give the target-feature and operand details. AMD's
[RDNA3 WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/) independently
documents the cooperative `D=A*B+C` 16x16x16 operation and GFX11
replication. The general
[HIP C++ language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html)
describe kernels, waves, and stream launches but expose no CUDA-style
`nvcuda::wmma` facade; that omission is not evidence that AMDGPU compiler
builtins are absent.

For the installed, selected `gfx1201` wave32 target the route is exactly
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12`. Each lane supplies
two `_ExtVector<8, short>` BF16-bit fragments and one
`_ExtVector<8, float>` accumulator and receives an
`_ExtVector<8, float>` result. The operation is a 16x16x16 `A*B+C`;
all 32 lanes must participate with a converged execution mask. The `short`
elements carry the 16-bit BF16 representation, not integer numerical input,
and accumulation/output registers are FP32. Installed HIP declares raw BF16
as a two-byte-aligned `unsigned short` payload and provides `__bf16`/
`__hip_bfloat16` conversion support; operation stores must still demonstrate
the contract's explicit RNE boundaries.

GFX11 wave32 instead uses
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` with
`_ExtVector<16, short>` A and B fragments and `_ExtVector<8, float>` C/D.
The full 256-bit A/B operands encode the required two-copy replication
between half-waves. Its wave64 variant has four-copy GFX11 replication and
four FP32 C/D elements per lane. GFX12 removes explicit A/B replication:
the `_w32_gfx12` operands are 128 bits and `_w64_gfx12` operands are 64 bits.
The suffixes and vector widths are ABI-significant and may not be selected by
a runtime branch inside one incorrectly compiled device image.

BF16 MFMA remains a distinct CDNA/`mai-insts` route. For example,
`__builtin_amdgcn_mfma_f32_16x16x8bf16` consumes two two-short vectors plus
a four-float accumulator and control immediates and returns four floats per
lane. It is suitable only in an image compiled for a matching MFMA target and
wave64 execution. The installed `gfx1201` target does not satisfy `mai-insts`;
its checked route is GFX12 WMMA, not MFMA. No installed CDNA device was
available, so MFMA runtime feasibility is **blocked on a matching device
sample**, rather than inferred from compiler-table presence. Conversely,
`gfx1036` has neither the checked GFX11/GFX12 WMMA route nor checked BF16 MFMA
and remains unsupported for these matrix products.

Direct builtins take registers, not pointers, strides, alignments, or layout
tags. The backend-private loader therefore owns every memory-layout decision.
It loads logical A row-major and emits logical C/D row-major. For linear and
QK it presents B as column-major without changing the public owner:
`B[k,o]=w[o,k]` preserves HF `[out,in]`, and
`B[d,t]=k[g(h),t,d]` is K-transpose at fragment load. PV presents
`A[r,t]=P[r,t]` and row-major `B[t,d]=V[g(h),t,d]`. Standard tiled owner
offsets are resolved element-wise; transformed leading offsets and strides are
honored per plane. A scalar BF16 loader needs only the owner's guaranteed
two-byte element alignment. Packed scratch rows begin at 32-byte-aligned
subranges with multiples-of-16 leading dimensions. Any vectorized load must
prove its stronger alignment or fall back to scalar/coalesced loads; the
builtin itself justifies no pointer-alignment assumption. The
[rocWMMA API reference](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html)
confirms that row/column combinations are representable, but rocWMMA is not
required for this direct route.

## Logical products and tail mapping

Let `P` be the checked product of all independent leading/batch extents and
let `pad16(n)` be checked round-up to a multiple of 16. Define
`Mp=pad16(R)`, `Ip=pad16(I)`, `Op=pad16(O)`, `Lp=pad16(L)`, and
`Dp=pad16(D)`. Every physical tile is 16x16; K loops advance by 16. The
following matrix is the feasibility result on the installed `gfx1201`.
“Feasible” means the compiler/device route and a caller-scratch mapping exist,
not that the operation is currently implemented.

| Product/mode | Logical `R` | Physical mapping | Result |
| --- | ---: | --- | --- |
| ordinary linear | 1 | `M=16`, `K=Ip`, `N=Op`; store only row 0 and `o<O` | Feasible on checked gfx1201 WMMA and implemented by the ROCm native `BF16` specialization on that device (see [Linear projections](linear-projections.md#linear-projections)); the twenty scalar leaves use the shared raw-word kernel. |
| ordinary linear | 15 | `M=16`; row 15 is neutral padding and is never logical | Feasible, including non-tile `I/O` and independent planes. |
| ordinary linear | 16 | one logical M tile | Feasible, including non-tile `I/O`. |
| ordinary linear | 17 | `M=32`; rows 17..31 are neutral padding | Feasible; padded rows are discarded, not extra tokens. |
| head-planar linear | 1 | same product, then `o=h*D+d` into `[H,1,D]` | Feasible with checked `O=H*D`; no head broadcast. |
| head-planar linear | 15 | `M=16`, `N=Op`; logical head scatter only | Feasible for non-tile `D/O` and independent planes. |
| head-planar linear | 16 | one logical M tile; logical `h,d` mapping retained | Feasible. |
| head-planar linear | 17 | two M tiles; padded rows discarded before head scatter | Feasible. |
| QK | 1 | `M=16`, `K=Dp`, `N=Lp`; cached `L` remains independent of `R` | Feasible with GQA head map and separate initialized/causal masks. |
| QK | 15 | `M=16`; full prefill has `Lp=16`, generic cached use has `pad16(L)` | Feasible; physical K/L padding is masked separately. |
| QK | 16 | one M tile; full prefill has one N tile | Feasible. |
| QK | 17 | `M=32`; full prefill has `Lp=32` | Feasible; padded rows/columns never enter softmax. |
| PV | 1 | `M=16`, `K=Lp`, `N=Dp`; masked P entries are exact BF16 zero | Feasible with native BF16 P/V and FP32 accumulation. |
| PV | 15 | `M=16`; non-tile `L/D` use neutral K/N tails | Feasible. |
| PV | 16 | one logical M tile | Feasible. |
| PV | 17 | `M=32`; padded output rows and `d>=D` are discarded | Feasible; output then merges to `[R,Hq*D]`. |

For every row of this table, linear computes the selected source window
`x[b,s+r,*]` and never infers `R` from `s`. Q owns `[P,Hq,R,D]`; K and V
owners remain `[P,Hkv,C,D]` and only their initialized prefix `0..L-1` is
packed. `G=Hq/Hkv` and `g(h)=floor(h/G)` select a KV head at load time; no
full KV-head replication is stored. QK accumulates BF16 products in FP32,
applies `1/sqrt(D)` and both initialized-prefix and `t<=a+r` masks in FP32,
then uses stable FP32 softmax. Masked probabilities are exactly zero. Every
logical probability is RNE-rounded to BF16 before the BF16 P/V WMMA loop,
whose FP32 result is RNE-rounded to BF16 before the device-local head merge.
Excluded K/V rows are not read; initialized neutral packed padding is not a
logical token.

## Caller-owned flow and checked scratch

Every segment below starts at `align_up(previous_end,32)` in one exact-device
`RawWorkspaceView`; every multiplication, `pad16`, byte conversion, alignment
round-up, and segment addition is checked. `A32(x)` below means checked
`align_up(x,32)`. The query reports the aligned sum and alignment 32 without
allocating, inspecting allocator capacity, creating a handle, or submitting.

| Segment | Checked payload bytes before `A32` | Purpose and lifetime |
| --- | --- | --- |
| linear `x_pack` | `P*Mp*Ip*2` | Gather exactly `x[b,s+r,i]`, zero physical M/K tails, and retain through the last linear WMMA read. This includes at least `P*R*I*2` logical bytes. |
| linear `y_pack` | `P*Mp*Op*2` | RNE BF16 matrix result before ordinary copy or `o=h*D+d` head-planar scatter to caller output; retain through that copy. This includes at least `P*R*O*2` logical bytes. |
| SDPA `q_pack` | `P*Hq*Mp*Dp*2` | Head-planar Q with neutral M/K tails; retain through QK. |
| SDPA `kv_pack` | `P*Hkv*Lp*Dp*2` | Pack only K rows `t<L` for QK, then reuse the same non-overlapping-in-time segment for only V rows `t<L` before PV. No `Hq` KV replication and no capacity tail. |
| SDPA `scores` | `P*Hq*Mp*Lp*4` | FP32 QK output, scale, mask, max, and sum storage; includes at least `P*Hq*R*L*4` logical score bytes and retains through probability production. |
| SDPA `p_bf16` | `P*Hq*Mp*Lp*2` | Explicit RNE BF16 probability boundary; masked and physical-tail entries are zero; retain through PV. Includes at least `P*Hq*R*L*2` logical bytes. |
| SDPA `pv_bf16` | `P*Hq*Mp*Dp*2` | Per-head RNE BF16 PV result; retain through merge. Includes at least `P*Hq*R*D*2` logical bytes. |
| SDPA `merged` | `P*R*(Hq*D)*2` | Checked `Hq*D` and device-local `[P,R,Hq*D]` merge before copy to the caller-owned standard-tiled output. |

The conservative linear requirement is
`A32(P*Mp*Ip*2)+A32(P*Mp*Op*2)`. The conservative SDPA requirement is the
aligned sum of the six SDPA rows, with `kv_pack` reused sequentially for K and
V. An implementation may prove a smaller direct-load/store requirement, but
it may not omit a logically required value, use hidden allocation, or make
the pure query depend on queue or allocator state. Persistent checkpoint
weights are read in place and are never scratch, transposed persistently, or
duplicated. Cache capacity `C` is not charged as prefix scratch, and neither
K nor V is materialized per query head.

The product flow is consequently:

| Product | Existing owner to temporary | Native invocation | Caller-owned result |
| --- | --- | --- | --- |
| ordinary linear | standard-tiled `x[...,T,I]` selected rows to `x_pack`; standard-tiled HF `w[O,I]` loaded in place as conceptual column-major B | one full wave per 16x16 C tile, K-loop over `Ip`, GFX12 BF16 WMMA to FP32 registers | RNE to `y_pack`, then logical `[P,R,O]` copy |
| head-planar linear | same input/weight path; no persistent weight packing | same product | RNE to `y_pack`, then `o=h*D+d` scatter to `[P,H,R,D]` |
| QK | standard-tiled Q to `q_pack`; only K prefix `L` to per-`Hkv` `kv_pack` | per `P,h` QK with `g(h)`, FP32 WMMA accumulation | FP32 `scores`, device-local scale/mask/stable softmax, RNE to `p_bf16` |
| PV | reuse `kv_pack` for only the V prefix; read `p_bf16` | per `P,h` BF16 P/V WMMA with FP32 accumulation | RNE to `pv_bf16`, device-local head merge to `merged`, logical copy to caller output |

Submission validates ranks 2..8 as applicable, nonzero runtime
`B,R,I,O,Hq,Hkv,D,L,C`, mode/window and GQA rules, transformed leading
mappings, `QuantizationFormat::NONE`, aliases, exact device, checked
arithmetic, and the actual workspace before registration or OID consumption.
Output/read, scratch/read, and scratch/output overlap are rejected. Valid
read/read overlap, including exact Q/K/V aliases that independently satisfy
their shapes, remains allowed. The HIP stream is the existing queue's stream;
there is no default-stream detour. The queue retains every operand/output owner
and the scratch lease through terminal completion. A full wave stays active
at every builtin, while tail predicates apply only to loads/stores and
device-local mask stages.

Every accepted call returns its positive OID before asynchronous completion.
Negative admission errors, producer-success waits, repeated observation of an
accepted failure, and exact-device destruction/quarantine follow the existing
queue contract unchanged. No consumer is submitted after a failed producer,
and K/V initialized length is published only after both append OIDs have
completed successfully.

## Seven-operation fit and dependency decision

| Planned operation | ROCm feasibility and concrete incompatibility |
| --- | --- |
| embedding/gather | A device-local index-and-copy kernel can preserve BF16 payload bits and independent planes under the same queue/alias rules. WMMA is irrelevant. Device-resident invalid indices may fail after acceptance; a host scan or round trip is forbidden. |
| linear | The checked GFX12 WMMA route covers ordinary and head-planar BF16 products above. GFX11 may use its replicated-input variant; matching CDNA may use MFMA after device proof. `gfx1036` is unsupported. No host or elementwise matrix substitute is acceptable. |
| RMSNorm/reduction | Device-local FP32 accumulation/reduction, reciprocal square root, scaling, and one BF16 RNE store fit the queue. WMMA does not perform the reduction; tiled padding and other planes must be excluded. Detailed kernel and tolerances remain operation-owned. |
| RoPE/trig | Device-local FP32 angle/sine/cosine and split-half rotation fit; the `double theta` ABI does not require FP64 device trig. WMMA is irrelevant. Host trig tables/round trips, adjacent-pair rotation, and padded features are incompatible. |
| cache append/partial-tile copy | Existing device-local tiled copy machinery is a suitable basis for a bit-preserving logical row append, but it must preserve all other logical rows and physical padding. WMMA is irrelevant; K and V remain separate owners/submissions. |
| SiLU/unary | Device-local stable FP32 unary evaluation followed by one BF16 RNE store fits. It remains a distinct operation before existing `mul`; WMMA does not justify fusion or hidden scratch. |
| SDPA/attention | WMMA covers QK and PV only. Separate device-local FP32 scale, causal/prefix mask, max/sum/exp softmax, explicit BF16 probability preparation, and head merge are required. Missing either native product, reading excluded K/V, full KV-head replication, exact-FP32 P in PV, host work, or hidden allocation is incompatible. |

**Decision: no added matrix library.** The installed compiler, HIP runtime,
and checked `gfx1201` device provide the direct GFX12 wave32 BF16-to-FP32 WMMA
route. The release-pinned and installed builtin tables also define the GFX11
and MFMA alternatives. Backend-private fragment loaders are enough to preserve
the fixed layouts and caller workspace; a missing CUDA-style HIP WMMA facade
is not a reason to add a dependency.

If direct builtins later prove insufficient on a required supporting target,
the first facility to assess is rocWMMA, not rocBLAS. The
[rocWMMA programming guide](https://rocwmma.readthedocs.io/en/latest/conceptual/programmers-guide.html)
documents a header-only C++17 wrapper around AMDGCN intrinsics, all-wave
participation, and no external kernel invocation. No rocWMMA package, minimum
version, CMake target, redistribution footprint, or fallback is prescribed
here because the checked direct route needs none. rocBLAS is not selected:
its [programming guide](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/Programmers_Guide.html)
requires a device/stream-bound handle, documents synchronizing stream changes,
and documents default temporary allocation as a synchronizing event. Those
handle and allocation semantics would require separate proof of pure
requirements, deterministic caller scratch, stream rebinding, reset/destruction
synchronization, and retained lifetime, with no demonstrated benefit for this
fixed route.

The first production consumer is the ROCm leaf under
`.cswd/tasks/006-tinyllama/04-linear-projections`; its private loader/kernel is
then reused by the ROCm SDPA leaf under
`.cswd/tasks/006-tinyllama/08-causal-grouped-attention`. Enabled ROCm builds
need no new discovery or private link. At runtime, a compiler image/device
without a matching checked WMMA/MFMA feature fails capability explicitly as
`Unsupported`; it never turns an unsupported device into a dependency pass.
Any future need for a helper must be established by those consuming leaves
with exact package/header/imported-target, minimum-version, license,
compile/link/runtime, handle, stream, workspace, and lifetime evidence before
changing enabled-ROCm-only CMake. CPU-only and ROCm-disabled builds acquire no
such dependency.

# TinyLlama forward layout — SYCL matrix feasibility

This is the bounded feasibility record for the seven-operation ABI. Embedding
lookup has its device-native raw-word `parallel_for` port, linear projections
are implemented as the twenty non-BF16 leaves on the operation-local in-order
path plus the native BF16 `joint_matrix` route with explicit RNE packing,
recorded in [Linear projections](linear-projections.md#linear-projections), SiLU is implemented by
the native packet path recorded in its operation-owned section, and SDPA is
implemented by the native subgroup-16 `joint_matrix` QK/PV route with
device-local masking, stable softmax, RNE BF16 P preparation, and head merge,
closed in [Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention). The
status of every other operation remains as stated in its own operation-owned
section. A future port MUST preserve
the signatures, pure requirement queries, validation precedence, owner rules,
and producer-wait schedule above; SYCL types remain backend-private.

The evidence layers are deliberately separate:

| Evidence layer | Observation on 2026-09-14 | What it establishes |
| --- | --- | --- |
| Repository declaration | `CMakeLists.txt` selects `icpx`/`dpcpp`, `sycl/sycl.hpp`, and `libsycl` with `-fsycl`, but has no compiler-version or matrix-extension check. Standard BF16 storage uses 16x16 row-major physical tiles in device USM from the exact context; the data arena and every suballocation are 32-byte aligned. | Storage and build wiring only; neither is native matrix evidence. |
| Installed compiler and headers | `icpx --version` reported Intel oneAPI DPC++/C++ Compiler `2026.1.0 (2026.1.0.20260617)`, package `intel-oneapi-compiler-dpcpp-cpp-2026.1 2026.1.0-235`, and `SYCL_EXT_ONEAPI_MATRIX=1`. The 2026.1 installation contains `sycl/ext/oneapi/matrix/{query-types.hpp,matrix.hpp,matrix-unified.hpp,matrix-intel.hpp}` dated 2026-06-17; the API remains in `sycl::ext::oneapi::experimental`. | The experimental API and runtime query compile in the installed SDK. It is not a stability promise. |
| Installed runtime and device | Required `sycl-ls` enumeration reported two Level Zero V2 devices, each `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, driver `1.15.38646+7`, PCI device `8086:e211`, architecture `intel_gpu_bmg_g21`, and subgroup sizes `16,32`. PCI inventory reported the in-kernel `xe` driver on Linux `7.0.0-31-generic`. The selected device reports `aspect::ext_intel_matrix=true` and maximum work-group size 1024. | An eligible Level Zero XMX device and subgroup 16 are installed. The eventual queue MUST bind one exact enumerated device, not an OpenCL or other-device fallback. |
| Documented native capability | The installed `matrix_combinations` query returned 53 combinations per B60. For `A=BF16,B=BF16,C=FP32,D=FP32`, it returned continuous `M<=8,N=16,K=16`, exact `16x16x16`, and exact `1x64x16`, `32x64x16`, `1x64x32`, and `32x64x32`. It also returned BF16-output variants, but this contract does not rely on their conversion rounding. | `M=1`, `M=16`, and a `16+1` decomposition are legal with BF16 inputs and FP32 accumulation; N/K tails require physical padding or tile decomposition. |
| Bounded sample | A removed standalone sample allocated device USM, required subgroup 16, invoked `joint_matrix_load`, `joint_matrix_mad`, and `joint_matrix_store`, and checked all-one BF16 products. Separate Level Zero executions printed `BF16xBF16->FP32 1x16x16 PASS` and `16x16x16 PASS`. | Representative native XMX execution exists for the row shapes used below. It does not implement IOM linear, QK, or PV and is not conformance or profiler evidence. |
| Production evidence (2026-09-14 snapshot) | Not run at that date: IOM linear, QK, PV, masking, softmax, RNE packing, head merge, all shape cases, conformance, tuning, and profiling. | At that date the SYCL port was blocked/unimplemented, and inventory or the sample cannot close a native-operation gate. The linear portion is since superseded by the implemented port and its executed record in [Linear projections](linear-projections.md#linear-projections), and the QK, PV, masking, softmax, RNE packing, and head merge portions by the implemented SDPA port and its executed record in [Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention); the remaining operations remain unrun. |

These conclusions use the
[experimental matrix extension at revision `cf12c378`](https://github.com/intel/llvm/blob/cf12c3783cc6a7adaf76e54c6a4f11f81ec8599b/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc),
the [Intel compiler 2026.1 release record](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html),
and the
[oneMath usage model at revision `0feb864`](https://github.com/uxlfoundation/oneMath/blob/0feb864ddaf49d12aa35e5492f6899fa823d9d9b/README.md).
The extension identifies itself as experimental, supports subgroup scope only,
requires A/B to be row- or column-major and accumulators to have dynamic
layout, and says an unsupported type/shape submission fails synchronously with
`kernel_not_supported`. For `intel_gpu_bmg_g21`, its XMX table contains the
queried BF16/FP32 combinations. Its installed-device restrictions require
`stride*sizeof(element)` to be a multiple of 8 and at most `2^24`, and a
4-byte-aligned load/store base. IOM's 32-byte arena, 16-element tile strides,
and 512-byte BF16 or 1024-byte FP32 tile steps satisfy the alignment minima,
but every offset, stride product, and byte address still requires checked
validation. A future kernel MUST require subgroup 16 explicitly; subgroup 32
was enumerated but was not sampled as a matrix execution shape.

**Product/row decision.** Let `p16(x)` mean checked round-up to 16. Every
entry below is *native-shape feasible on the enumerated B60*: the device query
supports the decomposition and the two primitive M shapes executed. The two
linear rows are implemented by the SYCL linear port and evidenced in
[Linear projections](linear-projections.md#linear-projections); the QK and PV rows are implemented
by the SYCL SDPA port and evidenced in
[Scaled dot-product attention](scaled-dot-product-attention.md#scaled-dot-product-attention). A device or
queue whose queried facilities do not cover the submitted shape keeps
returning `Unsupported`, never host attention, host staging, or an elementwise
substitute.

| Product | logical `R=1` | logical `R=15` | logical `R=16` | logical `R=17` |
| --- | --- | --- | --- | --- |
| Ordinary linear, `X[R,I] * W[O,I]^T` | One queried/sampled `1x16x16` M/N/K tile family. | One queried `16x16x16` family with physical row 15 zero and never stored as a logical row. | One queried/sampled `16x16x16` family. | One `16x16x16` family plus one queried/sampled `1x16x16` M tail. |
| Head-planar linear, same product followed by `o=h*D+d` | Same M decomposition; a device-local RNE/scatter writes only `[h,0,d]`. | Same 16-row physical tile; the neutral row is discarded before head-planar output. | Same exact 16-row tile. | Same 16+1 decomposition; scatter honors each head's independent output plane. |
| QK, `Q[h,R,D] * K[g(h),L,D]^T` | Same M=1 family; N walks 16-key tiles and K walks 16-wide D tiles. | Same padded M=16 family; causal masking excludes the neutral row and key tails. | Same exact M=16 family. | Same 16+1 family; each query head directly selects `g(h)` without expanding a KV head. |
| PV, `P[h,R,L] * V[g(h),L,D]` | Same M=1 family after FP32 softmax and explicit RNE BF16 P preparation. | Same padded M=16 family; padded probabilities and V/D tails are neutral. | Same exact M=16 family. | Same 16+1 family; FP32 results are explicitly RNE-packed before merge. |

The standard physical tile order makes full 16x16 X/Q/P and W/K/V tiles
directly loadable. X, Q, P, and V are row-major A/B operands. HF weights remain
the sole persistent `[out,in]` owner: a row-major W tile is interpreted as the
column-major B tile of `W^T`, so there is no persistent transpose. Likewise a
row-major K cache tile is the column-major B tile of QK. No checkpoint copy and
no repeated `Hq/Hkv` KV materialization are permitted. If an installed
compiler/device cannot accept those column-major loads, the capability query
returns `Unsupported`; a host transpose is not a fallback.

For non-tile `I`, `O`, `D`, or `L`, matrix hardware sees neutral physical K/N
tails only. `R=15` uses a physical M=16 tile, but row 15 is not a token;
`R=17` uses logical rows 0..15 and a distinct M=1 tail. Output conversion,
masking, and merge predicates exclude all padded coordinates. Leading-only
views select checked whole physical planes; every rank-2..8 leading tuple is
walked independently with its transformed offset and strides and is never
broadcast. Ordinary output writes `Y[b,r,o]`; head-planar scatter writes
`Y[b,h,r,d]` from global `o=h*D+d`. Q owns `[b,Hq,R,D]`, K/V owners remain
`[b,Hkv,C,D]`, and the kernel reads only `t<L` from `g(h)` while also enforcing
`t<=a+r`. The public cache shape is never narrowed to L.

**Checked flow and scratch.** `P` below is the checked product of actual
linear leading-plane extents and `B` is the checked attention batch/leading
product. Every displayed product, `p16`, addition, alignment, and conversion to
bytes is checked in `size_t` before admission. “Logical” is payload accounting;
“physical” is the conservative deterministic caller-workspace requirement.
All scratch subranges start at 32-byte boundaries in a live `RawWorkspace`
from the queue's exact device. They are disjoint from operands/output and from
simultaneously live scratch, and the lease and all operand/output registrations
remain retained until the accepted OID is terminal. Reuse is allowed only
after the in-order predecessor that last touches a subrange completes.

| Segment | Checked elements/bytes | Device-local flow and purpose |
| --- | --- | --- |
| Linear selected X | logical read `P*R*I*2`; existing owner physical tiles | Existing BF16 tiled owner -> row-major A loads. Window starts at `s`; no relocation or scratch copy. |
| Persistent W | logical read `O*I*2`; existing shared owner | Existing BF16 `[O,I]` tiles -> column-major B loads. It is never scratch, transposed checkpoint state, or duplicated persistent storage. |
| Linear FP32 products | physical scratch `P*p16(R)*p16(O)*4` | `joint_matrix` FP32 C/D tiles -> caller scratch -> explicit device-local RNE pack. Ordinary pack writes caller `[P,R,O]`; head-planar pack/scatter writes caller `[P,H,R,D]`. Logical input/output accounting is respectively `P*R*I*2` and `P*R*O*2`. |
| Q and K | logical reads `B*Hq*R*D*2` and `B*Hkv*L*D*2`; existing owners retain physical R/D and C/D tiles | Q A tiles plus direct K B tiles selected by `g(h)` -> QK MAD. Only the initialized prefix is addressed; K remains capacity C. |
| QK scores | logical `B*Hq*R*L*4`; physical scratch `B*Hq*p16(R)*p16(L)*4` | FP32 QK accumulator/store -> tiled caller scratch -> device-local scale, causal mask, max/sum reductions, and stable FP32 softmax. Padded positions are never reduction members. |
| BF16 probabilities | logical `B*Hq*R*L*2`; physical scratch `B*Hq*p16(R)*p16(L)*2` | FP32 probabilities -> explicit device-local RNE BF16 preparation -> PV row-major A tiles. This buffer remains live through PV; elementwise Q*K is not an alternative to MAD. |
| V and FP32 PV | V logical read `B*Hkv*L*D*2`; FP32 physical scratch `B*Hq*p16(R)*p16(D)*4` | Existing V row-major B tiles selected by `g(h)` plus BF16 P -> PV MAD -> FP32 caller scratch. V is not expanded per query head. |
| BF16 PV/head staging | logical `B*Hq*R*D*2`; physical scratch `B*Hq*p16(R)*p16(D)*2` | FP32 PV -> explicit RNE BF16 head tiles. A device-local merge then writes the caller-owned `[B,R,Hq*D]` output, whose logical payload is `B*R*(Hq*D)*2`; the checked `Hq*D` precedes byte sizing. |

An implementation may prove a smaller tile-streamed requirement or reuse
non-overlapping lifetimes (for example QK score storage after probability
preparation), but its pure query and submission MUST use the same documented
schedule. It may not replace deterministic caller scratch with USM allocation,
a library handle's hidden workspace, host staging, or allocator-capacity
inspection. The conservative linear requirement is the aligned FP32-product
segment. The conservative SDPA requirement is the checked, aligned sum of the
score, probability, FP32-PV, and BF16-head segments above; the reusable session
capacity remains the maximum of mutually exclusive operation/transfer
requirements, not their sum.

**Seven-operation fit.**

| Planned operation | SYCL feasibility and concrete present blocker |
| --- | --- |
| Embedding gather | Implemented by `src/sycl/embedding.cpp` and `src/sycl/queue_embedding.cpp`: one native `parallel_for` work item owns each destination word, with queued status reset, gather, and four-byte host-USM status copy. Runtime conformance and Level Zero scheduling evidence remain execution obligations; no host scan, host task, or operand staging is permitted. |
| Linear | Implemented by `src/sycl/queue_linear.cpp` and `src/sycl/queue_internal.hpp`: the twenty non-BF16 leaves queue an operation-local in-order `parallel_for` with the pure `{0, 1}` requirement query and an `aspect::fp64` device gate on `F64`, and `BF16` uses the backend-private subgroup-16 `joint_matrix` BF16/BF16/FP32 route with one explicit RNE BF16 pack/scatter kernel per output tile row over the caller-owned alignment-32 `A32(P*pad16(R)*pad16(O)*4)` product scratch, covering ordinary and head-planar output over every transformed leading plane and both `M=16` and single-row tails. It returns `Unsupported` before submission when `ext_intel_matrix`, subgroup 16, or the BF16/FP32 matrix combination is absent. Executed device facts and per-`R` conclusions are recorded in [Linear projections](linear-projections.md#linear-projections). |
| RMSNorm reduction | BF16 loads, FP32 squares/reduction, normalization, and explicit BF16 result packing fit subgroup/work-group kernels and caller scratch. It is nonmatrix work; a matrix MAD substitute is unnecessary, and no native SYCL reduction port exists. |
| RoPE trig | Device-local FP32 range reduction/trig and BF16 output packing fit an ordinary kernel over the tiled owner. It must preserve position `a+r`, pair boundaries, tails, aliases, and the queue error model. No native SYCL RoPE port exists. |
| Cache partial-tile copy | A device-local predicated tile copy can append only `[a,a+R)` while preserving untouched/padded cache bytes. Source/destination overlap, capacity C, initialized-prefix publication, and separate K/V OIDs remain as specified. Existing generic copy is not this cache operation. |
| SiLU unary | Device-local FP32 evaluation followed by BF16 packing fits an elementwise kernel with no matrix claim. The current SYCL queue has no SiLU kernel or neural facade implementation. |
| Causal grouped-query attention | The two MAD products are feasible as tabulated; masking, stable softmax, RNE P preparation, GQA head selection, caller scratch, and merged output remain device-local nonmatrix kernels. Missing either native QK or PV is `Unsupported`; host attention, host staging, and elementwise multiplication do not qualify. |

Every requirement query remains pure and has the same semantic arguments and
const output as its facade, with no workspace argument. It validates runtime
`B,R,I,O,Hq,Hkv,D,L,C`, nonzero extents, rank 2..8, modes/windows, shapes,
leading offsets/strides, `QuantizationFormat::NONE`, BF16 capability, aliases,
matrix combinations, subgroup 16, stride/alignment, and all checked arithmetic.
It allocates, registers, submits, or consumes no OID and does not inspect queue
or arena capacity. Submission additionally validates the actual workspace's
exact device, liveness, capacity, 32-byte alignment, lease, and nonoverlap.
Valid read/read overlap, including exact SDPA input aliases, remains allowed;
new output/read and scratch/operand/output overlap is rejected.

The backend-private kernels submit to the existing in-order queue and retain
the exact context/device, tensor owners, workspace lease, and native events
through terminal completion. They introduce no host wait inside an operation.
The caller still waits every direct producer before submitting a consumer;
negative OIDs report `noexcept` admission failure, and accepted asynchronous
failures stay visible on every repeated wait. Workspace reset/destruction must
therefore synchronize through the existing lease/quarantine path rather than
freeing or rebinding live USM.

**Dependency decision: no added matrix library.** The installed compiler,
Level Zero runtime, extension headers, positive device query, and native M=1
and M=16 samples establish that small direct kernels can cover linear, QK, and
PV. Direct kernels also expose tile padding, column-major W/K interpretation,
explicit RNE boundaries, exact queue, and caller workspace without a handle or
hidden allocator. By contrast, oneMath is a selector plus backend-wrapper and
third-party-library stack; its host BLAS usage models add discovery/link/runtime
surfaces and do not establish the IOM-specific no-hidden-allocation,
head-planar, GQA, or logical-row contracts. Convenience or broad GEMM tuning
does not justify that dependency, and a library cannot create matrix hardware
when the required combination is absent.

The landed SYCL linear-projection leaf is the first consumer of this facility
and shares the same minimal backend-private matrix route with the later SYCL
SDPA leaf. That consumer makes enabled-SYCL configuration fail clearly if the
pinned compiler/header contract is absent, and its runtime queries return
`Unsupported` before submission when the feature macro is not 1,
`ext_intel_matrix` or subgroup 16 is absent, no BF16/BF16/FP32 combination
covers the chosen tile, or stride/alignment cannot be satisfied. Disabled
SYCL/CPU-only builds gain no dependency. If future measured evidence proves
direct kernels cannot meet this contract, that consuming leaf—not this
assessment—must justify a minimal replacement with exact package, minimum
version/features, CMake target, private link and runtime footprint,
redistribution terms, deterministic caller-workspace/handle lifecycle, and
configuration/runtime failure behavior before adoption. Unsupported hardware
never becomes a dependency pass.

The installed inventory is reproducible remotely after the required oneAPI
initialization with:

```text
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u;
icpx --version && sycl-ls
sycl-ls --verbose
```

The future production port gates, intentionally not run by this assessment,
are the remote initialized-environment commands
`cmake --build build --target iom_sycl_conformance_tests` and
`ctest --test-dir build --output-on-failure -R
'^iom_sycl_conformance_tests$'`, plus native profiler evidence for logical
`R=1` and prefill linear, QK, and PV. They must cover all `1/15/16/17`,
non-tile dimensions, planes, GQA, padding exclusion, numerics, aliases,
workspace failures, device rejection, accepted failures, and repeated waits.

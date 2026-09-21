# Embedding lookup

Embedding lookup is a bit-preserving row gather. It selects whole table rows by
integral IDs and copies their raw payload bits into caller-owned output. It
performs no arithmetic, numeric conversion, re-encoding, accumulation,
rounding, or tolerance comparison, and it never reads or writes a logical
element outside the equations below.

## Public ABI, layout, and view semantics

The complete public surface is exactly `DeviceOps::embedding` and
`DeviceOps::embedding_workspace_requirements`:

```cpp
oid embedding(const TensorView& table, const TensorView& indices,
              TensorView& out, RawWorkspaceView workspace = {}) noexcept;
WorkspaceRequirements embedding_workspace_requirements(
        const TensorView& table, const TensorView& indices,
        const TensorView& out);
```

There is no host-index overload, public status handle, capability registry,
generic gather/pack/transpose API, or model/session API. The requirement query
receives the same semantic operands as submission with only the output made
const and the workspace omitted. The `noexcept` facade returns a positive
accepted OID, or the established negative `InvalidArgument`, `Unsupported`,
`Overflow`, `ResourceExhausted`, `DeviceError`, and `InternalError` OIDs
described in
[TinyLlama forward layout — Workspace and execution](tinyllama-forward-layout-workspace-and-execution.md#tinyllama-forward-layout--workspace-and-execution).
Zero is never accepted, and every admission failure has no output effect,
registration, sequence consumption, or accepted token.

The table is rank-two `E[V,F]`, and its caller-selected leading plane offset is
honored. `indices[...,1,R]` and `out[...,R,F]` have exactly the same leading
tuple, and each has rank 2 through 8. `R`, `V`, `F`, and every leading extent
are runtime values, never checkpoint constants, and MUST be nonzero. With `b`
denoting the complete leading tuple, the operation is

```text
out[b,r,f] = table[indices[b,0,r],f].
```

Views supply leading-only maps. Each operand and output contributes its own
selected plane offset and its own leading offset and strides, so index and
output planes may use different offsets, steps, and permutations, and the table
plane may be an independently transformed rank-two selected view. The operation
adds no leading broadcast, no singleton output-rank inflation, no mid-plane or
final-axis transform, no transpose, and no implicit cache or model-state
broadcast. Padding is not logical data: `16x16` tile padding, subbyte remainder
bits, and uninitialized output bytes are outside the logical element set, and
valid logical output MUST NOT depend on padding or uninitialized storage.

Payload elements are copied as raw codes. Subbyte payloads are addressed at
their logical bit width, so standard storage reads and writes packed fields
while preserving each logical cell's untouched carrier bits. A 64-bit element
uses paired low/high `uint32` words, never floating arithmetic and never a
shift by 64. Only `QuantizationFormat::NONE` is in scope: every other
recognized quantization format is `Unsupported`, and this operation adds no
quantization or storage format.

## Payload and index classification

Table payload covers exactly the 23 existing `DataType` leaves:

`BOOL`, `I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`,
`I64`, `U64`, `F4_E2M1`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3FN`, `F8_E5M2`,
`F8_E8M0`, `F16`, `BF16`, `F32`, and `F64`.

Every one of them is copied bit-exactly, including `F8_E8M0`, NaN payloads,
signed zero, infinities, and integer values above `2^53`. No arithmetic,
conversion, re-encoding, accumulation, rounding, or tolerance applies. `BF16`
is mandatory on every backend that implements the operation, and `F64` storage
does not imply FP64 arithmetic. `BOOL` is an 8-bit storage payload, and only
BOOL codes valid under the existing transfer contract are valid payload values.

Indices cover exactly the 12 integral leaves:

`I2`, `U2`, `I4`, `U4`, `I8`, `U8`, `I16`, `U16`, `I32`, `U32`, `I64`, and
`U64`.

`BOOL` and every floating leaf are semantically invalid IDs: an implicit
boolean or real-to-token conversion is never performed. A signed negative ID
and every ID greater than or equal to the vocabulary extent `V` are errors.

An unknown `DataType` or `QuantizationFormat` enum value is `InvalidArgument`. A
recognized inapplicable index leaf, a recognized unsupported payload leaf, a
recognized non-`NONE` quantization format, and a recognized unsupported backend
capability are `Unsupported`. Semantic classification is independent of native
storage: a backend limitation never narrows the classes above, it only decides
which leaves that backend implements.

## Backend capability matrix

| Backend | Payload leaves | Index leaves | Quantization |
| --- | --- | --- | --- |
| CPU | all 23 | all 12 | `NONE` only |
| CUDA | all 23 | all 12 | `NONE` only |
| ROCm | all 23 | all 12 | `NONE` only |
| SYCL | all 23 | all 12 | `NONE` only |
The CPU implementation covers all 23 payload and 12 integral index leaves for
`QuantizationFormat::NONE`. It reports `{0,1}`, rejects positive
`RawWorkspace` creation, executes gathers on the existing asynchronous FIFO
worker with raw bit helpers, and defers queued negative or out-of-vocabulary
IDs as repeatable `std::invalid_argument` failures after acceptance.

No backend may advertise a capability it has not implemented. An `Unsupported`
result, a storage-only observation, or a rejection-only probe is never
successful embedding conformance, and an unported backend keeps explicit
rejection expectations until its own port lands.

## Admission, aliasing, and error precedence

Common admission validates in this order, and only afterwards may owner
registration, sequence consumption, token acceptance, metadata effects, or
backend work occur:

1. rank, nonzero, and structure: well-formed views, rank-two table, index and
   output rank `2..8`, identical leading tuples, exactly one index row axis, an
   output last axis equal to `F`, and nonzero `V`, `F`, `R`, and leading
   extents;
2. exact queue `Device`, stable live owner and native handle for table,
   indices, and output, and leading view bounds and strides;
3. every checked element, bit, tile, plane, address, byte, and stride product;
4. exact output and leading shape, identical table and output dtype,
   conservative output/input overlap rejection, recognized dtype and
   quantization, and backend capability; and
5. the caller's supplied workspace against the reported `{bytes, alignment}`,
   including liveness, exact-device identity, size, alignment, overlap, and
   lease availability.

Embedding uses the same bounded, allocation-free checked-view helper path as
binary for recognized encodings, rank and nonzero extents, exact live owner and
stable native handle, leading-only view metadata, selected-plane bounds, and
checked plane, tile, element, bit, and byte arithmetic. That path allocates,
snapshots, registers, and leases nothing; embedding's shaping, capability,
alias, workspace, and status policy stays in this section.

Input-input read aliases are allowed: two inputs may overlap or alias exactly
because both accesses are reads. Output MUST be disjoint from the table and from
the indices. Same-owner output/input is rejected even when the transformed
windows appear disjoint, and standard backends additionally reject actual
intersecting backing ranges and identical native handles. No operand or output
is allocated, relocated, replaced, or silently converted by admission.

Admission snapshots the exact live registered owners, native handles, view
metadata, leading maps, plane selections, and workspace range for table,
indices, and output. Read/read aliases deduplicate owners, while output stays
disjoint and separately registered. No borrowed view is retained beyond
submission: a temporary caller view may die once its submission is accepted,
because registered owners keep the storage alive, and caller-owned output and
workspace remain leased through proven completion.

Requirement queries are deterministic and pure. A successful query allocates
nothing, including host metadata; mutates no registration, lease, status, or
queue state; consumes no token; reads no index value; submits nothing; and
depends only on the supplied views and immutable backend capability. It uses
borrowed views, checked scalars, and fixed stack state, and MUST NOT materialize
a vector-backed snapshot, `TensorShape`, `CopyViewSnapshot`, or request. A
supplied workspace is validated only after a successful requirement query,
during submission.

The rejection results are normative:

| Condition | Result |
| --- | --- |
| malformed view, rank or nonzero violation, leading-tuple mismatch, wrong index row axis or output shape, table/output dtype mismatch, unstable or stale owner, invalid selected plane | `InvalidArgument` |
| signed negative ID or ID `>= V` known to the host | `InvalidArgument` |
| unknown `DataType` or `QuantizationFormat` enum value | `InvalidArgument` |
| `BOOL` or floating index leaf, recognized unsupported payload leaf, recognized non-`NONE` quantization, recognized unsupported backend capability | `Unsupported` |
| output/input overlap, including conservative same-owner rejection and standard-backend backing-range intersection or identical native handles | `InvalidArgument` |
| supplied workspace that is not a live exact-device range, too small, misaligned, out of range, or overlapping an operand or output | `InvalidArgument` |
| overlapping live workspace lease, or bounded-resource exhaustion | `ResourceExhausted` |
| checked element, plane, tile, bit, byte, address, or stride overflow | `Overflow` |
| pre-acceptance runtime or device failure | `DeviceError` |
| any other unclassified failure | `InternalError` |
| negative or `>= V` ID discoverable only in queued storage | accepted, then the deferred failure below |

## Workspace and control-status protocol

Successful requirements are exactly `{0,1}` for CPU and `{32,32}` for a
supported accelerator (CUDA, ROCm, and SYCL). CPU allocates no operation
workspace and no device scratch, continues to reject creation of a positive
`RawWorkspace`, and keeps the established zero-requirement semantics: a
supplied but unused range is neither validated nor leased.

A positive accelerator range is caller-owned control scratch. It MUST be a live
exact-device owner range, at least 32-byte aligned, sufficient for the reported
bytes, disjoint from every operand and output, and leased through proven
completion. Owner-absolute subranges are checked and 32-byte aligned; disjoint
aligned ranges may be used concurrently, while overlapping live leases reject
with `ResourceExhausted`.

Those 32 bytes carry one fixed control packet: the first `uint32` is a bounds
status (`0` valid, `1` invalid ID) and the remaining 28 bytes are reserved
control padding. Each accepted call resets that status in queue order, and the
caller neither initializes nor polls it; only deferred completion after native
proof interprets it. Status transfer is explicit, bounded control metadata: it
is never table, index, or output staging, never a hidden payload allocation, and
never a host round trip for operand data. Standard accelerators enqueue, on the
same in-order native queue, the status reset, the raw gather, exactly a
four-byte device-to-host transfer of the status word, and the completion event.
CUDA takes the status cell from queue-owned page-locked memory
(`cudaHostAlloc`/`cudaFreeHost`); ROCm uses `hipHostMalloc`/`hipHostFree`; SYCL
uses `sycl::malloc_host`/`sycl::free` in the exact queue context with explicit
event dependencies. Standard SYCL host USM is not claimed to be physically
pinned or portably DMA-nonblocking. SYCL uses no `host_task` for the gather and
adds no submission-side wait. Runtime conformance and configured Level Zero
status-only traffic remain execution evidence, not an API guarantee. Native
launch, copy, or event failure always takes precedence over the status word.

The CUDA implementation is the standard raw-word path in
`src/shared/standard_tiled_embedding.hpp`,
`src/shared/standard_tiled_embedding.inl`, and `src/shared/gpu_queue_operations.inl`.
`src/cuda/copy.cu` enqueues the fixed metadata upload, device status reset,
one bounded 256-thread grid-stride gather, exactly one four-byte
`cudaMemcpyAsync` status transfer, and the completion event on the queue
stream. `src/cuda/device.cpp` reserves one page-locked `cudaHostAlloc` status
cell per fixed queue metadata slot and releases it through `cudaFreeHost` only
when that queue-resource lease is proven safe. The worker interprets the cell
only after event proof and caches a queued `std::invalid_argument`; native
launch, transfer, and event failures retain precedence. These are inspected
source/API facts, and the retained-backend gate executed them on the
configured CUDA device: `ctest --test-dir build --output-on-failure --timeout
300 -R '^iom_cuda_conformance_tests$'` passed `1/1`, the direct binary reported
`31/31` cases with `5,956,331/5,956,331` assertions, and its embedding cases
alone reported `2/2` cases with `117,047/117,047` assertions on
`NVIDIA GeForce RTX 5090` (compute capability `12.0`, driver `595.71.05`,
`nvcc` release `13.2` build `V13.2.78`). The status reset, bounded gather, and
single four-byte status transfer are exercised by the deferred invalid-ID,
repeated-wait, and status-reuse cases those runs contain; the exact commands,
mirror, and observed results are recorded in the same-revision evidence below.

The ROCm implementation uses the same shared raw-word metadata and gather
kernel. `src/rocm/copy.hip` enqueues the metadata upload, device status reset,
bounded 256-thread HIP gather, exactly one four-byte `hipMemcpyAsync` status
transfer, and the completion event on the queue stream. `src/rocm/device.cpp`
reserves one page-locked `hipHostMalloc` status cell per fixed queue metadata
slot and releases it through `hipHostFree` only after the queue-resource lease
has proven completion. These are inspected source/API facts, and the
retained-backend gate executed them on the configured ROCm device: `ctest
--test-dir build --output-on-failure --timeout 300 -R
'^iom_rocm_conformance_tests$'` passed `1/1`, the direct binary reported `32/32`
cases with `5,947,095/5,947,095` assertions, and its embedding cases alone
reported `2/2` cases with `117,046/117,046` assertions on `gfx1201`
(`AMD Radeon AI PRO R9700`, HIP `7.15.26333-0000000`, AMD clang `23.0.0git`),
including the deferred invalid-ID and repeated-wait cases that require the
four-byte `hipMemcpyAsync` status transfer.

Proven native completion releases the status cell, packet, and workspace lease
exactly once and permits safe reuse of independent scratch. Unknown completion,
failed drain, or unknown native finish MUST retain or quarantine the actual
owners, status cells, and workspace range instead of releasing or reusing them
merely because a semantic error was observed.

## Queued index data, deferred failure, and recovery

Every `TensorView` index payload is queued data, including CPU-host-visible
storage: a preceding queued producer may still write it, so no index value has
host-known provenance. Admission MUST NOT scan indices, add a host index span
API, or hide a host round trip, and the operation MUST preserve FIFO order so a
produced index tensor is observed at execution time.

The worker or kernel decodes each index's full raw width and signedness: mask
the unsigned bits, inspect a signed sign bit before conversion, compare the
full-width `uint64` value against `V` before any narrowing or addressing, and
never wrap. A 64-bit index is reconstructed from paired low/high `uint32` words
in the same way as its payload. Structural, range, device, owner, view, dtype,
quantization, alias, overflow, and workspace errors that are host-known reject
at admission with the categories above.

A negative or `>= V` ID that is discoverable only in device or queued storage
may be accepted and fail later. Device paths set the control status without
forming an out-of-bounds table address; the CPU worker reports
`std::invalid_argument`. The accepted OID stays positive and caches that
category and context, and every repeated wait for it MUST rethrow the same
failure; message wording is not part of the contract. Invalid output is not
usable, and failure has no rollback.

After an accepted execution failure the entire output is unspecified and MUST
NOT be consumed: no rollback, no failed-output-unchanged guarantee, and no
transactional semantics are promised. Already accepted following work may still
execute, but a consumer of a failed producer MUST NOT be submitted after that
producer's wait fails, and its dependent output MUST NOT be consumed. The
caller's model session is unusable and MUST fail and drain every accepted OID
before reset or destruction, continuing to drain even when an individual wait
throws. The operation itself adds no session state, tensor-validity flag, or
global queue cancellation.

Native or worker completion proof is independent of a data error. Proven
completion permits safe release and independent reuse of storage and scratch
after drain; an unknown or unprovable completion quarantines owners, status
cells, and workspace rather than allowing reuse.

## Implementation recipes

The standard recipe (CPU, CUDA, ROCm, and SYCL) gathers one work item per
destination 32-bit word, mapping each output plane, row, and feature through
`16x16` tiles and independent leading strides. A destination word decodes each
complete integral ID for every fragment it covers, checks sign and range before
any source addressing, selects the source raw bits, and merges only the bits
that word owns. A 6-bit payload may straddle two words but has exactly one
writer per destination word. A 64-bit payload is copied as paired low/high
`uint32` words, never with floating arithmetic and never with a shift by 64.
Indices use widths 2, 4, 8, 16, 32, or 64, never 6. Output padding is
unspecified and is never used as an ID or a value. The CPU worker runs on the
existing asynchronous FIFO worker and moves raw bits with the existing
`load_bits`/`store_bits`/`copy_value` helpers; it never uses a numeric codec. No
implementation stages table, index, or output data on the host.

## Independent reference and conformance obligations

The independent reference MUST separately encode the table as row-major raw
bytes and select raw logical bits through the index equation above. It MUST NOT
call production address, codec, gather, or CPU embedding helpers as its sole
oracle; it reuses only test-only bit readers and canonical physical mapping, and
only where those stay independent. Fixtures are deterministic synthetic
patterns with explicit expected raw-code rows, including a deliberate
wrong-row or permuted-oracle sanity check, so a transpose, a numeric
re-encoding, and a wrong leading-plane selection are observable. Comparisons
are raw-bit exact with zero numeric tolerance.

Shared coverage MUST include every one of the 23 payload leaves with `U32`
indices, every one of the 12 index leaves with `BF16` payload, and targeted
mixed and 64-bit combinations without a wasteful full Cartesian matrix. It MUST
include valid `BOOL` `0` and `1` codes; directed floating special values where
representable; raw subbyte and wide-carrier codes; signed zero, NaN payloads,
and integer values above `2^53`; repeated IDs with the first and last vocabulary
rows; runs `R=1`, `15`, `16`, and `17`; non-tile feature sizes `F=1`, `15`,
`17`, `31`, and `33`; table `V` boundaries; a selected rank-two table view;
independent leading transforms, offsets, steps, and permutations through rank
8; and poisoned native and output padding whose bytes never change valid
output. Every declared 64-bit payload also carries a directed raw-code fixture
(both carrier words set, sign bit, NaN payload, `2^53` crossing, high-word-only
and low-word-only values) with an odd non-tile feature count, because a salted
pattern of that width reaches those classes only by chance. Failure coverage
MUST include malformed structure, wrong device, stale registration, alias and
overlap, dtype and non-`NONE` quantization, checked overflow, workspace
liveness, size, alignment, device, overlap, and lease, a producer copy to
embedding to consumer FIFO case, negative, `>= V`, `U64_MAX`, and signed
sign-bit IDs without narrowing, repeated waits, and status and workspace reuse
after proven drain. A 64-bit index additionally carries a high-word-only
outlier whose low word alone is a valid small ID, so a truncated comparison
addresses a valid row instead of failing. Small index widths keep `V` inside
that representation's nonnegative domain, and a positive out-of-vocabulary
fixture chooses a representable ID code.

The cases live in the shared header
`test/backend/backend_conformance_embedding.hpp` and run through the existing
`iom_backend_conformance_cpu_tests`, `iom_cuda_conformance_tests`,
`iom_rocm_conformance_tests`, and `iom_sycl_conformance_tests` drivers. No
second test project, generic test framework, model fixture, checkpoint, or
network dependency is permitted. Each driver supplies its own explicit expected
payload and index spans, and common code never switches on backend kind. An
unported backend keeps an empty embedding span and asserts `Unsupported` only;
that rejection probe is not gather conformance.

Each retained driver declares through `iom_conformance::EmbeddingDeclaration`
the payload and index matrix its port must reach, the leaves this revision
implements (both spans empty for an unported port), and its exact workspace
requirement. The matrix drives the shared fixtures, so a case exists exactly
for a leaf the port must reach; `iom_conformance::kEmbeddingPayloadSpan`,
`iom_conformance::kEmbeddingIdSpan`, and `iom_conformance::kNoEmbeddingSpan`
are the standard matrix and empty-port span. Entry points are
`iom_conformance::run_embedding_conformance`,
`iom_conformance::run_embedding_oracle_self_check`,
`iom_conformance::run_embedding_reference_conformance`, and
`iom_conformance::run_embedding_common_conformance`, and the driver cases are
the `embedding lookup reference, admission, and lifetime` cases of
`test/cpu/test_cpu_conformance.cpp`, `test/cuda/test_cuda_conformance.cpp`,
`test/rocm/test_rocm_conformance.cpp`, and
`test/sycl/test_sycl_conformance.cpp`.

Every rule in this section MUST be observable through the shared conformance
suite or through a focused native lifetime and control-transfer scenario; tests
assert behavior rather than field forwarding, source text, or incidental
wording.
## Same-revision retained-backend gate evidence

The closing Embedding gate ran every retained backend's existing conformance
target and its direct binary at one revision and recorded the device, runtime,
and toolchain identity of each run. The CPU pair ran locally; every
accelerator pair used exact-worktree `csw-remote` sync/exec with a fresh sync
immediately before each execution, a unique per-profile mirror, remote-side
`timeout --kill-after=30s`, and a bounded hardware lock. No `.cswd` metadata
was copied to any host. Every row below is a real run: nothing in it is
compile-only, storage-only, rejection-only, or skipped.

| Backend | Commands (mirror) | Device / runtime identity | Observed result |
| --- | --- | --- | --- |
| CPU | `ctest --test-dir build --output-on-failure --timeout 300 -R '^iom_backend_conformance_cpu_tests$'` and `./build/test/iom_backend_conformance_cpu_tests` | Local host CPU, `g++` `15.2.0`, CMake `4.0.2`, release build | `1/1` CTest pass in `10.93 s`; `24/24` cases, `5,936,189/5,936,189` assertions; the embedding case alone `1/1` case, `118,307/118,307` assertions |
| CUDA | both commands on remote host `bv1` (mirror `csw03emb12-a1-cuda`) | `NVIDIA GeForce RTX 5090`, compute capability `12.0`, driver `595.71.05`, `nvcc` release `13.2` build `V13.2.78` | `1/1` CTest pass in `16.33 s`; `31/31` cases, `5,956,331/5,956,331` assertions; embedding cases `2/2` cases, `117,047/117,047` assertions |
| ROCm | both commands on remote host `bv2` (mirror `csw03emb12-a1-rocm`) | `gfx1201` (`AMD Radeon AI PRO R9700`), HIP `7.15.26333-0000000`, AMD clang `23.0.0git` | `1/1` CTest pass in `20.78 s`; `32/32` cases, `5,947,095/5,947,095` assertions; embedding cases `2/2` cases, `117,046/117,046` assertions |
| SYCL | both commands on remote host `bv2` through the outside-checkout profile override with an empty `REMOTE_SETUP`, `set +u; source /opt/intel/oneapi/setvars.sh; set -u` and a `sycl-ls` GPU enumeration in every remote call (mirror `csw03emb12-a1-sycl`) | Level Zero V2 `Intel(R) Arc(TM) Pro B60 Graphics`, oneAPI DPC++/C++ `2026.1.0`, `ocloc` `26.22.38646.7` | `1/1` CTest pass in `6.83 s`; `29/29` cases, `5,907,297/5,907,297` assertions; embedding cases `4/4` cases, `117,124/117,124` assertions |

Each sync and exec pair, with its exact argv and observed exit status, is
retained in the gate task's `remote.log` beside its specification in the
shared task store; the remote-side logs named there hold the full run output.

**Matrix closure.** The four retained drivers declare and reach the complete
23-payload and 12-index matrix through `kEmbeddingPayloadSpan` and
`kEmbeddingIdSpan`. `BF16` and `R=1`, `15`, `16`, and `17` execute inside those
declared cases, with independent planes, padding isolation, and checked
failure behavior. Unsupported-only, storage-only, and compile-only
observations are not counted.

**Lifetime and queue obligations.** The shared cases cover the independent
raw-bit oracle, transformed leading planes through rank eight, padded-table
invariance, accepted out-of-vocabulary and negative IDs, repeated waits,
pure requirement queries, workspace validation and reuse, output/input alias
rejection, producer-to-consumer FIFO order, and unknown-completion quarantine.
The retained-backend control/status and workspace implementations own their
backend-specific transfer details while preserving the common contract.

**No defect.** The retained-backend gate changes no normative statement or
implementation file; it records only the existing shared cases and each
driver's declared matrix.

## Implementation references and delivery prerequisites

Implementers need these existing sources and seams:

- public ABI, request snapshots, owner registration, and workspace views:
  `include/iom/iom.hpp`, `include/iom/tensor.hpp`, and `include/iom/device.hpp`;
- checked logical shapes, slots, and leading-plane arithmetic: `src/tensor.cpp`
  and `src/iom_internal.hpp`;
- shared allocation-free checked-view helpers and operation dispatch:
  `src/device_ops.cpp` and `src/device_ops_binary.cpp`;
- admission, FIFO acceptance, rollback, deferred completion, and repeated
  waits: `src/device_ops.cpp`;
- workspace owner bytes, 32-byte subranges, leases, proof, and quarantine:
  `src/workspace.cpp` and `include/iom/detail/workspace_registry.hpp`;
- standard tiled word ownership, launch mapping, and queue resources:
  `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_add.inl`,
  `src/shared/gpu_queue.hpp`, `src/shared/gpu_queue_operations.inl`, and
  `src/shared/event_ring.hpp`;
- CPU asynchronous worker and raw bit movement: `src/cpu/queue.cpp` and
  `src/cpu/transfer_helpers.hpp`;
- accelerator capability classification, queues, and storage:
  `src/cuda/copy.hpp`, `src/cuda/copy.cu`, `src/rocm/copy.hpp`,
  `src/rocm/copy.hip`, `src/sycl/queue_internal.hpp`, and `src/sycl/queue.cpp`;
- accelerator status APIs: the CUDA Runtime memory interface (`cudaHostAlloc`,
  `cudaFreeHost`, and `cudaMemcpyAsync`), `hip/hip_runtime_api.h` page-locked
  allocation and asynchronous copy, and the SYCL USM, queue, and `atomic_ref`
  interface references;
- independent storage oracle and conformance drivers:
  `test/backend/backend_conformance_oracle.hpp`,
  `test/backend/backend_conformance_common.hpp`,
  `test/backend/backend_conformance_memory.hpp`, the four retained
  `test/<backend>/test_<backend>_conformance.cpp` drivers, and
  `test/CMakeLists.txt`.

The shared raw-word embedding sources and the four retained conformance
drivers are the delivered implementation and test surface of this section.

The genuine prerequisites and the final closure are producer/consumer
relationships, not a fixed serial backend order:

1. the shared allocation-free checked-view helpers plus the public embedding
   and requirement-query declarations with `Unsupported` defaults;
2. the independent raw-bit reference and the shared embedding conformance
   header, before any backend claims numerical conformance;
3. the per-backend port, which owns its caller-provided workspace and status
   transfer requirements; and
4. the shared retained-backend gate, which closes Embedding lookup before
   linear implementation begins and revalidates already completed ports if a
   later backend exposes a contract defect.

CPU, CUDA, ROCm, and SYCL appear in the task-list and scheduling order only.
This contract imposes no serial backend execution requirement: accelerator ports
may proceed independently once the shared ABI, admission, and reference
prerequisites exist, and an unported backend names its missing evidence instead
of inventing capability. This section records the frozen target and obligations
of implementers; it claims no build, test, accelerator, hardware, or runtime
validation on any backend.

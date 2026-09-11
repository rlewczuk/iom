# 004 — Bounded, separate device-memory arenas

## Status and intent

Replace independently growing accelerator allocation pools with two separate allocation domains: **tensor data** and **device-visible metadata**. Each domain has its own backing allocation and its own allocator. Do not interpret “metadata arena for both” as one mixed arena or as metadata prepended to tensor payloads.

The caller chooses memory capacity and prepares tensors/workspaces above `DeviceOps`. The device factory performs explicit reservation after establishing the native context. `DeviceOps` consumes prepared storage and fixed metadata slots; it never reserves, grows, or frees native device memory.

This specification supersedes the current permission for internal operation-time staging/metadata allocations in `docs/BACKEND_CONTRACT.md`. Numerical behavior, view addressing, owner identity, and repeatable completion/failure semantics remain unchanged except for the explicit rank and resource limits below.

## 1. Scope

### Required

- Separate, fixed-backed data and metadata allocators on CUDA, ROCm, and SYCL.
- Maximum tensor rank eight, including the two tiled matrix axes (so, six dimensions above tiled axes).
- Bounded queues, outstanding submissions, descriptor capacity, and completion resources.
- Removal of lazy device allocation from metadata, host-transfer staging, and SYCL binary staging paths.
- Explicit caller preparation of payload scratch, outside operations.
- Thread-safe allocator access and completion-proven reuse, including exceptional paths.
- Clean migration of factories, public workspace contracts, callers, tests, examples, and architecture documentation.

### Backend boundary

The two raw device-allocation requirement applies to **CUDA, ROCm, and SYCL**, whose storage is a standard tiled byte allocation. CPU retains its existing caller-supplied allocator and has no device metadata arena. TTNN retains its native, per-plane tensor-storage ownership: it cannot be made a raw `ListAllocator` client without a separate native-storage redesign. Do not add an unused arena or pretend native TTNN tensors are raw pointers.

Rank and queue admission limits apply to all backends. TTNN operations must continue to use existing destination tensors and host-only scratch without adding IOM device allocations. Vendor-internal TTNN runtime allocations remain an explicitly unproven boundary, not a claimed two-allocation guarantee.

### Non-goals

- Changing codecs, arithmetic accuracy, broadcasting, alias support, or tile representation.
- Implementing currently unsupported `silu`, `linear`, `rmsnorm`, or `sdpa`.
- Implementing a SYCL binary device kernel as part of this allocation change; the existing exact host fallback may remain with explicit device scratch.
- Eliminating host metadata allocations or bounding host token-history memory.
- Eliminating all possible SDK-internal allocation, kernel-module loading, or runtime compilation.
- Compaction, paging, eviction, unified-memory oversubscription, peer allocation, or a global device registry.

## 2. Terminology and invariant

A **native allocation** reserves storage through the accelerator runtime (`cuMemAlloc`, `hipMalloc`, `sycl::malloc_device`, or equivalent). An **arena suballocation** reserves a range within an already allocated backing buffer and changes only host allocator bookkeeping.

After successful standard-GPU device setup:

1. IOM owns exactly two live native arena backing allocations for that device: data and metadata.
2. Tensor creation/destruction and explicit raw-workspace creation/destruction may suballocate/free **data-arena ranges**, but must not reserve or release native backing memory.
3. A submission may acquire/release one fixed metadata slot; it must not suballocate tensor data or payload scratch.
4. Operations, waits, view transforms, and host transfers must issue **zero IOM native device allocation/free calls**, including first use, saturation, and failure recovery.
5. Neither arena grows, spills, or falls back to another allocator. No operation triggers implicit tensor creation.
6. Arena backing is released only at safe device teardown. Unknown completion must never be treated as permission to reuse or free memory.

The existing owner-storage invariant still applies: user tensors and views never relocate or retarget storage.

## 3. Fixed limits and sizing

These are fixed limits for this change, not a new family of tuning knobs.

| Resource | Limit | Rationale |
| --- | ---: | --- |
| Full tensor/view/result rank | 2–8 inclusive | Six leading axes cover batch, head, layer/expert and grouped inference layouts while bounding descriptors. |
| Live `DeviceOps` queues per `Device` | 4 | Allows compute plus several independent inference pipelines without an unbounded number of worker threads and streams. |
| Outstanding submissions per queue | 16 | Enough enqueue depth for a sequence of kernels without tying the limit to batch size or token count. Preserves the current GPU event-ring scale. |
| Outstanding submissions per device | 64 | Four queues times sixteen credits. |
| Metadata slot payload/stride | 512 bytes | Fits all current rank-eight copy and binary descriptors with headroom. |
| Metadata slot alignment | 32 bytes | Meets existing engine storage alignment and descriptor alignment requirements. |
| Metadata slots per standard GPU device | 64 | One descriptor per potentially live submission. |
| Metadata backing per standard GPU device | 32 KiB | `4 × 16 × 512`. No growth. |
| Supported deployment target | Up to 8 physical GPUs per host | Up to 32 GPU queues, 512 outstanding submissions and 256 KiB of GPU metadata backing for one `Device` per physical GPU. |

Eight GPUs is a supported/tested topology, **not** a new process-global registry or ordinal restriction. Do not reject a valid device solely because its runtime ordinal is eight or larger. Multiple independent `Device` objects for one physical GPU have independent reservations and limits; memory budgeting must count each object. Existing process-wide OID queue-ID limits remain unchanged and include CPU/reference queues.

Four queues is an upper bound, not a recommendation to create four queues. One queue is sufficient for a serialized inference pipeline. These limits are engineering choices, not benchmark claims.

### Descriptor bound

Current copy metadata is `48 + 24 × leading_rank`, hence at most 192 bytes for rank eight. Current binary metadata requires `sizeof(BinaryMetadata) + 4 × 8 × sizeof(uint64_t)`; on the current 64-bit representation this is 400 bytes.

The implementation must check actual layouts with compile-time assertions against the 512-byte slot and its alignment. Do not rely solely on the numerical ABI estimate. A future descriptor that exceeds the slot must require a deliberate specification/ABI update, not automatic growth.

### Rank validation

Reject full tensor specs, views produced by transforms, and binary result shapes outside rank 2–8 before allocation, registration, token acceptance, metadata upload, or kernel launch. Include `reshape_leading`, since it can increase rank while preserving element count. Never silently flatten leading axes to accept rank nine.

Apply the bound to full tensor/view shapes, not indiscriminately to helper spans representing only leading dimensions. Throw `std::invalid_argument` in throwing APIs; return `OidError::InvalidArgument` through OID facades. Preserve existing validation precedence for other errors.

## 4. Explicit setup and allocator ownership

### Standard-GPU factory contract

Introduce a common, backend-neutral configuration value named `DeviceMemoryConfig`, containing a required `tensor_arena_bytes` capacity. No default percentage of free VRAM, automatic growth, or memory-discovery heuristic is permitted. The caller selects the capacity after budgeting weights, activations, KV storage, and prepared scratch, leaving SDK/other-process headroom.

CUDA/ROCm/SYCL factory signatures take the ordinal and this configuration instead of the existing borrowed `Allocator&`. The concrete backend must:

1. Validate the configuration and checked address/size arithmetic.
2. Establish its native device/context.
3. Reserve one device-memory buffer of `tensor_arena_bytes` and one distinct 32-KiB metadata buffer in that exact context/device.
4. Construct a `ListAllocator` over the data buffer, alignment 32.
5. Construct a separate `FixedSizeAllocator` over the metadata buffer, alignment 32, payload size 512, exactly 64 blocks.
6. Publish the fully initialized device only after both reservations and allocator construction succeed.

Zero capacity and a capacity not divisible by 32 are invalid; do not round a caller budget upward silently. Backing bases must satisfy 32-byte alignment. Usable capacities must equal the requested data capacity and 32 KiB respectively. Metadata capacity is additional to `tensor_arena_bytes`, not subtracted from it.

This factory-owned reservation is the explicit setup layer, **not `DeviceOps` allocation policy**. It also avoids the SYCL bootstrap problem of asking a caller to allocate memory in a private context that does not exist yet. Preserve backend-neutral common code: native allocation and context handling stay in concrete backends.

Initialization failure must release any acquired backing and context resources safely; no partially usable device escapes. Native OOM maps to `std::bad_alloc` at the factory. Runtime/context errors retain their established categories.

Use factory-local RAII owners for each native backing and the context, including failures in host allocator construction and transfer-stream setup. Roll back in reverse acquisition order while the context is valid and preserve the original error. Normal teardown first closes admission and proves outstanding use safe, then destroys allocator bookkeeping, releases safe arena backings, and finally releases the context. The unknown-completion retention rules below override ordinary backing release.

### Allocator implementation

Reuse the existing `ListAllocator` and `FixedSizeAllocator` in `include/iom/alloc.hpp` / `src/alloc.cpp`; do not introduce competing best-fit or fixed-slot allocator implementations. They are separate allocator objects over separate, non-overlapping native allocations. `LinearAllocator` is not the default: tensor/workspace destruction must make ranges reusable without resetting all live storage.

The device owns both backing buffers and both allocators. Tensors, raw workspaces, and queues borrow the device and must not outlive it. Native handles remain payload addresses, even when they are interior pointers into an arena; validate context/device membership without requiring each payload to be the base of an independent native allocation.

The existing allocators are not thread-safe. Serialize each allocator's host bookkeeping at its owning device boundary. Never hold an allocator lock across a kernel submission, completion wait, registry callback, or worker drain. A lock for bookkeeping is permitted; waiting for capacity is not.

Suballocation failures, including host bookkeeping exceptions, must preserve allocator ownership/free-space state. Free/coalescing and failed construction must not lose or duplicate a range. No public allocator `reset()` may invalidate live owners, leased metadata, or quarantined ranges. Arena teardown is not an allocator reset while work is live.

### Fragmentation guarantee

Fixed 512-byte metadata slots have no external fragmentation; slot slack is bounded internal waste. Small descriptor allocations never split data-arena free ranges. The data arena remains a variable-size best-fit/coalescing allocator: it can still fragment when differently sized payloads have different lifetimes. Report `std::bad_alloc` if no contiguous range fits even when total free bytes are sufficient. Do not promise compaction or relocate existing tensors.

The high layer should create long-lived weights/KV/workspaces during preparation and reuse them. This is usage guidance, not another internal allocation heuristic.

## 5. What belongs in the metadata arena

The metadata arena holds **immutable per-submission kernel descriptors**, including captured view offsets/strides and binary broadcast/result mappings. It does not allocate one persistent device record for every tensor or view.

Authoritative tensor/view metadata stays on the host. Different views of one owner and different operations on the same tensor must have independent descriptors while simultaneously live. No mutable “current descriptor” may be attached to a tensor owner and overwritten by the next submission.

Preserve CUDA/ROCm's existing inline-copy descriptor path. With rank at most eight, all their supported copies fit it; remove the unreachable arbitrary-rank device-copy-metadata branch. Do not force a metadata allocation/upload where an existing inline parameter suffices.

CUDA/ROCm binary operations take at most one fixed metadata slot. SYCL copy may keep its pointer-based descriptor using the same fixed arena; adding a new inline optimization is not required. TTNN and CPU must not allocate dummy device slots for host-only descriptors. No implemented operation may require more than one metadata slot.

Replace `MetadataSlotPool::ensure_slot_capacity` and `SyclMetadataSlotPool` device-buffer ownership/growth with the device's fixed allocator and a small completion lease. Remove duplicated storage-capacity, replacement-allocation, and per-slot native-free machinery. Preserve lifecycle state that is necessary to prevent reuse before completion.

Host upload mirrors may remain host allocations. Each submitted descriptor's host bytes must also remain unchanged until its asynchronous upload is complete. Do not reuse one mutable host staging area across live transfers without a completion proof.

## 6. Admission, completion and failure behavior

### Admission

A queue has sixteen credits shared by **all** accepted operation types, including no-op copy. A credit covers admitted, queued, executing, and device-completion-pending work. Completed token history does not consume credits.

Admission must reserve the necessary queue credit, metadata slot if required, and existing completion resources before the first native effect or accepted token. A preacceptance failure must roll back all those reservations and owner registrations. Invalid/unsupported input must not consume capacity.

When a queue has sixteen live credits, a descriptor cannot be acquired, or required workspace is unavailable, return `OidError::ResourceExhausted` synchronously. Do not block waiting for completion, spin/retry, synchronize a stream for space, or create another buffer. The caller may wait for previously accepted work and retry explicitly. Request snapshotting may still allocate host metadata.

Creating a fifth live queue throws `std::bad_alloc` before starting another stream/worker; destruction of a safely drained queue releases its queue-count reservation. Limits are local to the exact device identity.

### Retirement

A credit and its metadata slot become reusable when completion has been proved and all device/host accesses using that slot are over—not when `copy`/binary returns, not when the task is merely enqueued, and not only when the user calls `wait`.

Completion processing must make forward progress and return safe slots even when the user has not waited. Preserve token results independently of reusable native event/descriptor state: repeating a wait on an old token must not observe a later submission that reused its slot.

An operation with a retained error may release its slot after a successful covering drain proves that no access remains. If completion is unknown, quarantine the metadata slot and referenced payload/workspace ranges. Do not return them to either allocator. Quarantine must be device-owned so queue destruction cannot drop its final lifetime protection. Unknown slots remain unavailable to new queues and may reduce usable capacity; do not compensate by growing an arena.

Quarantine the whole unresolved submission lease: queue credit, native event generation, descriptor, host upload backing where needed, and payload/workspace references. Destroying a queue with unresolved work retains its queue-count reservation until that work is proved complete; repeated destruction/recreation must not exceed the four-queue/sixteen-credit-per-generation bound. Sixteen credits is a ceiling, not a promise of admission when resources are quarantined. A proof must cover the originating stream/queue's work; successfully draining a different queue is not sufficient.

Queue destruction requires the caller to stop concurrent member calls. Stop admission, finish host preparation, drain accepted work, and transfer unresolved leases to device quarantine before releasing host descriptors or queue resources. Ordinary completed token history must retain only durable outcomes, not scarce descriptor/event/workspace leases.

A later successful covering proof can reclaim quarantined ranges without erasing the original token's failure. On unrecoverable teardown, retain affected backing memory rather than freeing an arena still referenced by native work. In a monolithic arena, one unproven range can require retaining the entire backing allocation; this is an explicit safety-versus-reclamation cost of the design.

### Native queue resources

CUDA/ROCm `create_ops` eagerly creates the queue stream and its sixteen completion events. Submission reuses those events and never lazily creates replacements. Retire unsafe events rather than recycling them without proof. Setup failure rolls back resources; admission fails if the remaining safe resources cannot satisfy it.

Preserve per-queue concurrency: no device-wide synchronization on normal metadata reuse, no global mutex across devices, and no hidden serialization through a shared transfer stream for compute operations. SDK-internal allocations associated with SYCL event objects are outside the explicit IOM-native-allocation guarantee and must be reported separately if observed.

## 7. Payload scratch is explicit and separate

A metadata arena alone is insufficient: `src/sycl/copy.cpp` currently allocates three temporary device buffers for every binary, and the standard host-transfer staging pools grow their own device buffers. Those allocations must not survive this change or be redirected into the metadata arena.

### Raw workspace contract

Add a backend-neutral, non-copyable/non-movable raw workspace owner created explicitly through `Device`, with copyable non-owning byte-range views. It is not a tiled `Tensor`. It exposes checked capacity/range and exact device/owner identity; common code does not expose or interpret backend runtime types.

Standard GPU workspace creation suballocates the **data arena**, just like tensor payloads. It occurs only at caller request, outside a submission/transfer. Allocation exhaustion throws `std::bad_alloc`; no extra native buffer is reserved. CPU/TTNN paths that require no device scratch accept an empty workspace and need not manufacture a raw native allocation.

Add pure requirements queries for binary operations and host transfers. A query validates the request and returns required device-workspace bytes and alignment without allocating device memory, reserving a slot, or submitting work. Results are deterministic for the selected backend, operation and views. Extend all four binary facades and both host-transfer APIs consistently with a borrowed workspace argument, empty by default for zero-workspace paths. A positive requirement must never cause an automatic allocation when the argument is absent.

Validate required capacity, alignment, checked offsets, exact device identity, and live owner before effects. Missing, undersized, misaligned or foreign workspace is invalid input (`InvalidArgument` OID / `std::invalid_argument`); a valid workspace already exclusively leased is resource exhaustion (`ResourceExhausted` / `std::bad_alloc`). Reject overlap with operands/output and overlapping concurrently leased workspace ranges. Disjoint ranges of one owner may be used concurrently. Raw workspaces must participate in outstanding-work lifetime registration/quarantine just like tensors.

The caller retains workspace ownership through completion. A runtime error does not release its exclusivity until completion is proved. Existing scalar semantics and padding preservation remain unchanged.

### SYCL binary fallback

Retain the present host scalar algorithm, but replace the three `malloc_device`/`free` pairs with three aligned slices of the supplied raw workspace. For each operand/output, preserve `binary_view_staging_bytes` semantics: padded storage through the highest addressed plane, including untouched gaps, not merely logical element bytes.

With `A=32` and checked `align_up`, slice offsets are `0`, `align_up(lhs_bytes,A)`, and `align_up(lhs_bytes,A)+align_up(rhs_bytes,A)`. Required capacity is the third offset plus `out_bytes`. No overlap is permitted between slices. Host staging allocation remains out of scope. A success or failure must never native-free a slice or return it to the data allocator; release only its completed operation lease.

The device arena is not host-dereferenceable. Retain the existing separate host-USM `lhs_stage`, `rhs_stage`, and `out_stage` allocations and their transfer ordering; only the three device-USM allocations are replaced. Host allocations may remain internal and must survive until their asynchronous readers/writers have completed. Do not reinterpret a device-workspace slice as a host span or switch the arena to shared USM to evade this distinction.

CUDA/ROCm binary and TTNN/CPU binary currently require zero device payload scratch. Preserve that property. A future direct SYCL binary kernel could reduce its query result to zero, but is not an acceptance dependency here.

### Synchronous host transfers

CUDA/ROCm/SYCL host-transfer requirements are `gpu_algorithm::compute_staging_size(logical_nbytes)` bytes, with 32-byte workspace-base alignment. Preserve tail-word initialization and logical-to-tiled conversion. Acquire no internal device staging buffer; use the supplied workspace, and retain it through the existing synchronous completion boundary.

Remove the device-owned growing `StagingSlotPool` implementations once all callers use explicit workspace. Host mirrors may be retained separately; they must not own device backing. Remove the CUDA/ROCm elastic `TransferStreamPool`. Create **one dedicated host-transfer stream/queue per device during setup**, matching the existing SYCL model; serialize synchronous host transfers on that device. This permits one active public host transfer per device and avoids lazy stream creation. It must not serialize independent `DeviceOps` compute streams or host transfers on other devices. Host transfers retain the existing rule that callers must explicitly wait for conflicting operation-queue work.

Serial host transfers are a deliberate concurrency trade-off for predictable setup, not an assertion of equal parallel-transfer throughput. A higher-level asynchronous transfer scheduler is outside this change.

## 8. Compatibility and removal

This is a clean cutover, not an opt-in alternate backend mode.

- Migrate standard-GPU factories from arbitrary borrowed allocators to explicit `DeviceMemoryConfig`; retain CPU and TTNN's appropriate factory models.
- Remove dynamically growing metadata/staging device-buffer owners, poisoned-buffer replacement allocation, and elastic transfer-stream creation once replaced.
- Keep `ListAllocator` and `FixedSizeAllocator` as the canonical implementations. Do not remove standalone allocator APIs used by CPU or other callers.
- Update all callers to reserve explicit data capacity, create needed raw workspaces and pass their views. No legacy factory overload may bypass the two-arena invariant; no missing-workspace fallback may allocate.
- Replace tests that assume arbitrary rank with valid rank-eight coverage and explicit rank-nine rejection tests. Do not narrow dtype or broadcast support.
- Update `include/iom` contracts, `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, `docs/MEMORY.md`, and examples/driver setup to distinguish native backing allocation, suballocation, slot leasing, and host allocation.

Principal implementation touchpoints: `include/iom/alloc.hpp`, `src/alloc.cpp`, `include/iom/device.hpp`, `include/iom/tensor.hpp`, `include/iom/iom.hpp`, `src/iom.cpp`, backend factory headers/device implementations, `src/shared/gpu_queue.hpp`, `src/shared/metadata_slot_pool.hpp`, `src/shared/event_ring.hpp`, `src/shared/staging_pool.hpp`, `src/shared/transfer_pool.hpp`, `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_add.inl`, `src/sycl/copy.cpp`, and `src/sycl/staging_pool.*`. Shared behavior belongs in `test/backend`; native allocation instrumentation and setup belong in backend drivers.

## 9. Acceptance and verification

### Allocation and data preservation

- Instrument actual IOM native allocation/free boundaries, not only `iom::Allocator`: current hidden metadata allocations bypass that interface.
- Standard-GPU device setup has exactly two arena allocation calls on success. Creating/destroying tensors and raw workspaces subsequently performs zero native alloc/free calls and preserves the backing allocation count.
- From the first valid operation—not only after warm-up—copy, all four binaries, waits, view transforms and host transfers issue zero native alloc/free calls. Exercise changing ranks/sizes within prepared capacity, failures, queue recreation and arena exhaustion.
- Descriptor addresses are within metadata backing; all tensor/raw-workspace addresses are within data backing. No domain crosses into the other. Independently observe tensor results, untouched planes and padding.
- Exhaust and free fragmented data ranges; verify correct `bad_alloc`, safe reuse/coalescing, stable live addresses and transactional failure behavior. Do not require allocations to succeed merely because aggregate free bytes suffice.
- Prove rank-eight owner/view/broadcast correctness; rank-nine creation and rank-increasing transforms fail before side effects. Assert compiled descriptor sizes/alignment fit the slot.

### Lifetime and admission

- Hold device completion deterministically so sixteen accepted submissions remain live. The seventeenth returns `ResourceExhausted` without token acceptance or output changes. Release completion and verify admission resumes without native allocation.
- Exercise four queues independently on one device and the fifth-queue limit. Verify metadata slots never overlap while live; an inline/no-metadata operation still consumes a queue credit, but not a dummy descriptor slot.
- Reclaim completed slots without requiring an explicit user wait. Reuse slots/events, then repeatedly wait on earlier success and failure tokens and observe their original result.
- Submit different views of the same owner on multiple queues, plus binary broadcasts with different mappings. Verify descriptors cannot be overwritten by later submissions.
- Force preacceptance failures and prove reservations roll back. Force postlaunch/unknown-completion failures and prove metadata, event, tensor and workspace ranges are not reused; prove a covering drain can reclaim safe ranges without changing retained token failures.
- Destroy a failed queue with spare queue-count capacity, create another, and verify device-owned quarantine still prevents aliasing with unproven work. Retire all four queue generations without completion proof and verify that another queue cannot be created. A drain of an unrelated queue must not reclaim their leases. Teardown must not free any backing that remains in use.
- Exercise missing/undersized/foreign workspace, overlapping leased ranges, disjoint concurrent workspace slices and workspace lifetime through failure.
- Exhaust or quarantine resources on GPU 0 while GPU 1 continues admitting its independent quota. Keep more than sixteen completed tokens alive and verify that they do not consume credits or descriptor/event leases.

### Backend and performance evidence

Run shared backend conformance on CPU, CUDA, ROCm, SYCL and TTNN as available under the project's enabled-backend rules; enabled hardware tests must fail rather than skip. Accelerator builds/tests/profiling must follow the remote-development workflow. Report unavailable hardware explicitly; do not claim an eight-GPU run from single-device tests. Test isolation on two or more GPUs when available and the full eight-GPU topology when available; verify limits/accounting without introducing a global registry.

Compare first-use and repeated copy/binary/host-transfer paths on identical hardware/shapes against the pre-change baseline. Record native allocation counts, peak reserved bytes, latency distribution and throughput. Include small descriptors, rank-eight broadcasts and large payload transfers. Allocation counts and fixed capacity are hard gates; latency improvements are not presumed. Investigate a repeatable warmed compute-path regression above 5%; do not accept one silently. Report serialized-host-transfer throughput separately from compute throughput.

Expected mechanism: removal of native allocator churn and metadata external fragmentation, bounded reservation and no grow-time spikes. Caller workspace retains the SYCL host fallback's transfer/CPU costs; this change must not claim to remove them. The 32-KiB metadata budget is negligible relative to consumer VRAM, but the explicitly chosen data arena reserves its full capacity even when only partially used.

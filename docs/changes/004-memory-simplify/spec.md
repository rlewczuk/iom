# 004 — Bounded, separate device-memory arenas

## Status and intent

Replace independently growing accelerator allocation pools with two separate allocation domains: **tensor data** and **device-visible metadata**. Each domain has its own backing allocation and its own allocator. Do not interpret “metadata arena for both” as one mixed arena or as metadata prepended to tensor payloads.

The caller chooses memory capacity and prepares tensors/workspaces above `DeviceOps`. The device factory accepts an immutable backend-neutral `QueueConfig` and performs explicit reservation after establishing the native context. `Device::create_ops()` remains argument-free; it consumes prepared storage and fixed metadata partitions/resources, and it never reserves, grows, frees, or resizes native device resources.

This specification supersedes the current permission for internal operation-time staging/metadata allocations in `docs/BACKEND_CONTRACT.md`. Numerical behavior, view addressing, owner identity, and repeatable completion/failure semantics remain unchanged except for the explicit rank, queue, and resource limits below.

## 1. Scope

### Required

- Separate, fixed-backed data and metadata allocators on CUDA, ROCm, and SYCL.
- A backend-neutral `QueueConfig` with immutable `max_in_flight_per_queue` (default 16; zero invalid) selected at device construction and applied to every queue on that device.
- Maximum tensor rank eight, including the two tiled matrix axes (so, six dimensions above tiled axes); current core/copy/binary paths and tests accept larger ranks and must migrate to this deliberate new cutover.
- A four-live-queue-per-`Device` cap, host-side parking for accepted excess work, and native in-flight credits, metadata partitions, and completion resources bounded by the configured capacity.
- Removal of lazy device allocation from metadata, host-transfer staging, and SYCL binary staging paths.
- Explicit caller preparation of payload scratch, outside operations.
- Thread-safe allocator access and completion-proven reuse, including exceptional paths.
- Clean migration of factories, public workspace contracts, callers, tests, examples, and architecture documentation.

### Backend boundary

The two-raw-device-allocation requirement applies to **CUDA, ROCm, and SYCL**, whose storage is a standard tiled byte allocation. Every backend factory accepts `QueueConfig`; standard-GPU factories additionally accept the `DeviceMemoryConfig` introduced by this change. CPU retains its borrowed allocator and has no device metadata arena. TTNN retains its native, per-plane tensor-storage ownership and native storage model: it cannot be made a raw `ListAllocator` client without a separate native-storage redesign. Do not add an unused arena or pretend native TTNN tensors are raw pointers.

Rank and queue admission limits apply to all backends. TTNN operations must continue to use existing destination tensors and host-only scratch without adding IOM device allocations. Vendor-internal TTNN runtime allocations remain an explicitly unproven boundary, not a claimed two-allocation guarantee.

### Non-goals

- Changing codecs, arithmetic accuracy, broadcasting, alias support, or tile representation.
- Implementing currently unsupported `silu`, `linear`, `rmsnorm`, or `sdpa`.
- Implementing a SYCL binary device kernel as part of this allocation change; the existing exact host fallback may remain with explicit device scratch.
- Eliminating host metadata allocations or bounding host token-history memory.
- Eliminating all possible SDK-internal allocation, kernel-module loading, or runtime compilation.
- Compaction, paging, eviction, unified-memory oversubscription, peer allocation, or a global device registry.

## 2. Terminology and invariant

A **native allocation** reserves storage through the accelerator runtime (`cuMemAlloc`, `hipMalloc`, `sycl::malloc_device`, or equivalent). An **arena suballocation** reserves a range within an already allocated backing buffer and changes only host allocator bookkeeping. Let **C** denote the Device's immutable, nonzero `max_in_flight_per_queue`; its default is 16.

**Accepted/outstanding work** includes every positively tokenized request, including parked requests, and is bounded by host memory and the 55-bit sequence space rather than by a configured host-backlog count. **Native in-flight work** is the subset currently holding a queue credit, metadata slot when required, and completion resource; it is bounded by C per queue and `4 × C` per device. A parked request holds no native credit, metadata slot, completion event/resource, or native effect.

After successful standard-GPU device setup:

1. IOM owns exactly two live native arena backing allocations for that device: data and metadata.
2. Tensor creation/destruction and explicit raw-workspace creation/destruction may suballocate/free **data-arena ranges**, but must not reserve or release native backing memory.
3. A dispatched submission may acquire/release one fixed metadata slot from its queue's disjoint C-slot partition; acceptance of parked work acquires no slot and must not suballocate tensor data or payload scratch.
4. Operations, waits, view transforms, and host transfers must issue **zero IOM native device allocation/free calls**, including first use, saturation, parking, dispatch, retirement, and failure recovery.
5. Neither arena grows, spills, or falls back to another allocator. No operation triggers implicit tensor creation.
6. Arena backing is released only at safe device teardown. Unknown completion must never be treated as permission to reuse or free memory.

The existing owner-storage invariant still applies: user tensors and views never relocate or retarget storage.

## 3. Fixed limits and sizing

The four-queue cap and slot geometry are fixed structural limits. `C` is selected only before Device construction; there is no live reconfiguration or per-queue override.

| Resource | Limit | Rationale |
| --- | ---: | --- |
| Full tensor/view/result rank | 2–8 inclusive | Six leading axes cover batch, head, layer/expert and grouped inference layouts while bounding descriptors. |
| Live `DeviceOps` queues per `Device` | 4 | Allows compute plus several independent inference pipelines without an unbounded number of worker threads and streams. |
| Configured native in-flight credits per queue | `C ≥ 1` (default 16) | Chosen at Device construction and shared by every queue on that Device. |
| Native in-flight credits per device | `4 × C` | Four queues times the configured per-queue credit capacity; parked work does not consume this bound. |
| Metadata slot payload/stride | 512 bytes | Fits all current rank-eight copy and binary descriptors with headroom. |
| Metadata slot alignment | 32 bytes | Meets existing engine storage alignment and descriptor alignment requirements. |
| Metadata slots per standard GPU device | exactly `4 × C` | Four disjoint queue partitions of exactly C slots each. |
| Metadata backing per standard GPU device | checked `4 × C × 512` bytes | Default C=16 gives 64 slots and 32 KiB; those are default-derived values, not universal constants. |
| Supported deployment target | Up to 8 physical GPUs per host | Up to 32 GPU queues and `8 × 4 × C` native credits across one Device per physical GPU; metadata sizing is `8 × 4 × C × 512` bytes. |

Eight GPUs is a supported/tested topology, **not** a new process-global registry or ordinal restriction. Do not reject a valid device solely because its runtime ordinal is eight or larger. Multiple independent `Device` objects for one physical GPU have independent reservations and limits; memory budgeting must count each object. Existing process-wide OID queue-ID limits remain unchanged and include CPU/reference queues.

Four queues is an upper bound, not a recommendation to create four queues. One queue is sufficient for a serialized inference pipeline. These limits are engineering choices, not benchmark claims.

### Descriptor bound

Current copy metadata is `48 + 24 × leading_rank`, hence at most 192 bytes for rank eight. Current binary metadata requires `sizeof(BinaryMetadata) + 4 × 8 × sizeof(uint64_t)`; on the current 64-bit representation this is 400 bytes.

The implementation must check actual layouts with compile-time assertions against the 512-byte slot and its alignment. Do not rely solely on the numerical ABI estimate. A future descriptor that exceeds the slot must require a deliberate specification/ABI update, not automatic growth.

### Rank validation

Current core/copy/binary paths and tests accept ranks greater than eight. This change deliberately cuts over to rank 2–8, so callers and tests must migrate; rank greater than eight is not already unreachable.

Reject full tensor specs, views produced by transforms, and binary result shapes outside rank 2–8 before allocation, registration, token acceptance, metadata upload, or kernel launch. Include `reshape_leading`, since it can increase rank while preserving element count. Never silently flatten leading axes to accept rank nine.

Apply the bound to full tensor/view shapes, not indiscriminately to helper spans representing only leading dimensions. Throw `std::invalid_argument` in throwing APIs; return `OidError::InvalidArgument` through OID facades. Preserve existing validation precedence for other errors.

## 4. Explicit setup and allocator ownership

### Standard-GPU factory contract

Introduce a common, backend-neutral `QueueConfig` whose sole field is `max_in_flight_per_queue`, default 16. Zero is invalid. The value is immutable for the Device lifetime and applies to every queue it creates; `Device::create_ops()` remains argument-free. Standard-GPU factories accept `QueueConfig` and a `DeviceMemoryConfig` containing the required `tensor_arena_bytes` capacity. CPU factories retain their borrowed allocator and accept `QueueConfig`; TTNN factories accept `QueueConfig` and retain their native storage model.

No default percentage of free VRAM, automatic growth, or memory-discovery heuristic is permitted. The caller selects tensor capacity after budgeting weights, activations, KV storage, and prepared scratch, leaving SDK/other-process headroom. Validate checked sizing, backend resource limits, overflow, and whether the requested configuration can be provisioned before publishing the Device. A backend need not accept every nonzero C: zero or a value outside a documented backend limit throws `std::invalid_argument`, checked sizing overflow throws `std::overflow_error`, inability to provision storage/resources throws `std::bad_alloc`, and runtime/context errors retain their established categories.

The concrete standard-GPU backend must:

1. Validate both configurations and checked address/size arithmetic.
2. Establish its native device/context.
3. Reserve one device-memory buffer of `tensor_arena_bytes` and one distinct metadata buffer of checked capacity `4 × C × 512` bytes in that exact context/device.
4. Construct a `ListAllocator` over the data buffer, alignment 32.
5. Construct one `FixedSizeAllocator` across the entire metadata buffer, alignment 32, payload size 512, exactly `4 × C` blocks.
6. Publish the fully initialized device only after both reservations and allocator construction succeed.

Zero tensor capacity and a capacity not divisible by 32 are invalid; do not round a caller budget upward silently. Zero C is invalid. Check `4 × C × 512` for overflow and backend limits; backing bases must satisfy 32-byte alignment. Usable capacities must equal the requested data capacity and checked metadata capacity; metadata capacity is additional to `tensor_arena_bytes`, not subtracted from it.

This factory-owned reservation is the explicit setup layer, **not `DeviceOps` allocation policy**. It also avoids the SYCL bootstrap problem of asking a caller to allocate memory in a private context that does not exist yet. Preserve backend-neutral common code: native allocation and context handling stay in concrete backends.

Initialization failure must release any acquired backing and context resources safely; no partially usable device escapes. Native OOM maps to `std::bad_alloc` at the factory. Runtime/context errors retain their established categories.

Use factory-local RAII owners for each native backing and the context, including failures in host allocator construction and transfer-stream setup. Roll back in reverse acquisition order while the context is valid and preserve the original error. Normal teardown first closes admission and proves outstanding use safe, then destroys allocator bookkeeping, releases safe arena backings, and finally releases the context. The unknown-completion retention rules below override ordinary backing release.

### Queue setup and partitions

On standard-GPU queue construction, atomically reserve one disjoint partition of exactly C metadata slots from the device's single `FixedSizeAllocator`, and eagerly construct all C completion resources plus fixed host metadata mirrors and queue-native bookkeeping required by that queue. Publish the queue only after every reservation and construction succeeds; rollback all reservations and resources on failure. Queue construction may allocate or provision these resources. Submission, dispatch, retirement, waits, and operations may not native-allocate, native-free, resize, or replace them.

A queue never borrows another queue's partition. Safely drained queue destruction returns its partition and queue-count reservation. Unknown native use retains/quarantines both the partition and queue reservation until a covering proof; live slots are never reassigned.

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

Preserve CUDA/ROCm's existing inline-copy descriptor path. After the deliberate rank-2–8 cutover, all supported copies fit it; current larger-rank paths and tests must migrate before the arbitrary-rank device-copy-metadata branch is removed. Do not force a metadata allocation/upload where an existing inline parameter suffices.

CUDA/ROCm binary operations take at most one fixed metadata slot. SYCL copy may keep its pointer-based descriptor using the same fixed arena; adding a new inline optimization is not required. TTNN and CPU must not allocate dummy device slots for host-only descriptors. No implemented operation may require more than one metadata slot.

Current metadata and staging paths grow or lazily allocate device storage, including `MetadataSlotPool::ensure_slot_capacity`, `SyclMetadataSlotPool`, and staging-buffer replacement machinery. The fixed-slot design removes that growth, replacement-allocation, and per-slot native-free code while preserving lifecycle state needed to prevent reuse before completion.

Host upload mirrors and fixed host metadata mirrors may remain host allocations. Each submitted descriptor's host bytes must remain unchanged until its asynchronous upload is complete. Do not reuse one mutable host staging area across live transfers without a completion proof.

## 6. Admission, completion and failure behavior

### Admission

Every accepted operation type, including no-op copy, uses the same per-queue FIFO and native-credit rule. The caller serializes calls on one queue, so positive-token/sequence acceptance order is FIFO. Acceptance completes validation, an immutable host request/view snapshot, owner/tensor/raw-workspace lifetime registration and exclusivity lease, sequence/token reservation, and append to that queue's FIFO before returning a positive token.

Host snapshot and queue-node allocation are permitted at acceptance. If either fails, return synchronous `OidError::ResourceExhausted` / `std::bad_alloc` with no token, native effect, or output mutation and roll back every reservation and registration transactionally. Sequence exhaustion remains synchronous `Overflow`. Invalid or unsupported input, including invalid workspace, must not consume capacity.

At most C operations per queue hold native in-flight credits. A dispatchable request acquires one credit, its preconstructed completion resource, and one metadata slot when required only when it reaches the FIFO head. Accepted excess work is parked host-side: it holds no native credit, metadata slot, completion event/resource, or native effect, and is not subject to a configured host-backlog count. It retains its host snapshot, token outcome state, owner registrations, and workspace exclusivity lease from acceptance through completion.

When a credit becomes safe, autonomous completion processing dispatches the oldest parked request; this requires neither an explicit wait nor another submission. No later request, including no-op, inline, or no-metadata work, may bypass an earlier parked request. There is no cross-queue ordering. If no safe resource is available, the FIFO head remains parked until a covering proof; never grow, borrow, replace, or reuse an unsafe resource.

Failures discovered after acceptance become retained terminal failures observed repeatably by `wait`; they are never converted into a synchronous negative OID. Resource quarantine reduces the safely dispatchable credit count. A workspace range already leased by another accepted request is synchronous resource exhaustion (`ResourceExhausted` / `std::bad_alloc`), with rollback if acceptance cannot finish.

Creating a fifth live queue throws `std::bad_alloc` before starting another stream/worker; construction rollback releases its partition and queue-count reservation. Destruction of a safely drained queue releases both. The cap and all reservations are local to the exact Device identity.

### Retirement

A native credit, metadata slot, and completion resource become reusable when completion has been proved and all device/host accesses using them are over—not when `copy`/binary returns, not when the task is merely enqueued, and not only when the user calls `wait`. Retained token outcomes do not keep scarce native resources.

Completion processing must make forward progress and return safe credits, slots, and events even when the user has not waited, then dispatch the oldest parked request. Preserve token results independently of reusable native event/descriptor state: repeating a wait on an old token must not observe a later submission that reused its resources.

An operation with a retained error may release its native resources after a successful covering drain proves that no access remains. If completion is unknown, quarantine the whole unresolved submission lease: queue credit, native event generation, metadata partition/slot, host upload backing where needed, and payload/workspace references. Do not return any part to an allocator or queue partition. Quarantine is device-owned so queue destruction cannot drop its final lifetime protection; unknown resources remain unavailable and may reduce usable capacity without arena growth.

Queue destruction requires the caller to stop concurrent member calls. Stop acceptance, finish host preparation, drain every previously accepted request—including all parked requests—in FIFO to a terminal outcome, and transfer unresolved native leases to device-owned quarantine before releasing host descriptors or queue resources. Accepted work is never silently cancelled, discarded, or omitted from token history. A queue with unresolved native use retains its queue-count reservation and partition until a covering proof; a drain of another queue is insufficient, and live slots are never reassigned. Metadata backing outlives all possible native use.

A later successful covering proof can reclaim quarantined ranges and queue reservations without erasing the original token's failure. On unrecoverable teardown, retain affected backing memory rather than freeing an arena still referenced by native work. In a monolithic arena, one unproven range can require retaining the entire backing allocation; this is an explicit safety-versus-reclamation cost of the design.

### Native queue resources

The current CUDA/ROCm `EventRing` lazily creates up to 16 events and waits at saturation. This is a current implementation fact, not eager-event behavior. The proposed queue construction change eagerly creates all C completion resources and fixed host mirrors/bookkeeping for that queue; submission reuses them and never lazily creates replacements or resizes arrays. Retire unsafe events rather than recycling them without proof. Setup failure rolls back resources; dispatch uses only safely provisioned resources.

Preserve per-queue concurrency: no device-wide synchronization on normal metadata reuse, no global mutex across devices, and no hidden serialization through a shared transfer stream for compute operations. SDK-internal allocations associated with SYCL event objects are outside the explicit IOM-native-allocation guarantee and must be reported separately if observed.

## 7. Payload scratch is explicit and separate

A metadata arena alone is insufficient: `src/sycl/copy.cpp` currently allocates three temporary device buffers for every binary, and the standard host-transfer staging pools grow their own device buffers. Those allocations must not survive this change or be redirected into the metadata arena.

### Raw workspace contract

Add a backend-neutral, non-copyable/non-movable raw workspace owner created explicitly through `Device`, with copyable non-owning byte-range views. It is not a tiled `Tensor`. It exposes checked capacity/range and exact device/owner identity; common code does not expose or interpret backend runtime types.

Standard GPU workspace creation suballocates the **data arena**, just like tensor payloads. It occurs only at caller request, outside a submission/transfer. Allocation exhaustion throws `std::bad_alloc`; no extra native buffer is reserved. CPU/TTNN paths that require no device scratch accept an empty workspace and need not manufacture a raw native allocation.

Add pure requirements queries for binary operations and host transfers. A query validates the request and returns required device-workspace bytes and alignment without allocating device memory, reserving a slot, or submitting work. Results are deterministic for the selected backend, operation and views. Extend all four binary facades and both host-transfer APIs consistently with a borrowed workspace argument, empty by default for zero-workspace paths. A positive requirement must never cause an automatic allocation when the argument is absent.

Validate required capacity, alignment, checked offsets, exact device identity, and live owner before acceptance. Missing, undersized, misaligned, or foreign workspace is invalid input (`InvalidArgument` OID / `std::invalid_argument`). Overlap with operands or output, and invalid range/alignment/device identity, is also `InvalidArgument`. A workspace range already exclusively leased by another accepted request is resource exhaustion (`ResourceExhausted` / `std::bad_alloc`). Acceptance leases the supplied workspace range even while the request is parked; disjoint ranges of one owner may be used concurrently. Raw workspaces must participate in outstanding-work lifetime registration/quarantine just like tensors. Dispatch allocates no workspace.

The caller retains workspace ownership through completion. A runtime error does not release its exclusivity until completion is proved. Existing scalar semantics and padding preservation remain unchanged.

### SYCL binary fallback

Retain the present host scalar algorithm, but replace the three `malloc_device`/`free` pairs with three aligned slices of the supplied raw workspace. For each operand/output, preserve `binary_view_staging_bytes` semantics: padded storage through the highest addressed plane, including untouched gaps, not merely logical element bytes.

With `A=32` and checked `align_up`, slice offsets are `0`, `align_up(lhs_bytes,A)`, and `align_up(lhs_bytes,A)+align_up(rhs_bytes,A)`. Required capacity is the third offset plus `out_bytes`. No overlap is permitted between slices. Host staging allocation remains out of scope. A success or failure must never native-free a slice or return it to the data allocator; release only its completed operation lease.

The device arena is not host-dereferenceable. Retain the existing separate host-USM `lhs_stage`, `rhs_stage`, and `out_stage` allocations and their transfer ordering; only the three device-USM allocations are replaced. Host allocations may remain internal and must survive until their asynchronous readers/writers have completed. Do not reinterpret a device-workspace slice as a host span or switch the arena to shared USM to evade this distinction.

CUDA/ROCm binary and TTNN/CPU binary currently require zero device payload scratch. Preserve that property. A future direct SYCL binary kernel could reduce its query result to zero, but is not an acceptance dependency here.

### Synchronous host transfers

CUDA/ROCm/SYCL host-transfer requirements are `gpu_algorithm::compute_staging_size(logical_nbytes)` bytes, with 32-byte workspace-base alignment. Preserve tail-word initialization and logical-to-tiled conversion. Acquire no internal device staging buffer; validate and lease the supplied workspace before the first native effect, and retain it through the existing synchronous completion boundary.

Remove the device-owned growing `StagingSlotPool` implementations once all callers use explicit workspace. Host mirrors may be retained separately; they must not own device backing. Remove the CUDA/ROCm elastic `TransferStreamPool`. Create **one dedicated host-transfer stream/queue per device during setup**, matching the existing SYCL model; serialize synchronous host transfers on that device. This permits one active public host transfer per device and avoids lazy stream creation. It must not serialize independent `DeviceOps` compute streams or host transfers on other devices. Host transfers retain the existing rule that callers must explicitly wait for conflicting operation-queue work.

Serial host transfers are a deliberate concurrency trade-off for predictable setup, not an assertion of equal parallel-transfer throughput. A higher-level asynchronous transfer scheduler is outside this change.

## 8. Compatibility and removal

This is a clean cutover, not an opt-in alternate backend mode.

- Add `QueueConfig` to every backend factory contract; standard-GPU factories also take explicit `DeviceMemoryConfig`, while CPU retains its borrowed allocator and TTNN retains native storage. Keep `Device::create_ops()` argument-free and remove any mutable or per-queue configuration path.
- Remove dynamically growing metadata/staging device-buffer owners, poisoned-buffer replacement allocation, and elastic transfer-stream creation once replaced.
- Keep `ListAllocator` and `FixedSizeAllocator` as the canonical implementations. Do not remove standalone allocator APIs used by CPU or other callers.
- Update all callers to reserve explicit data capacity, create needed raw workspaces, pass their views, and migrate factory/device setup to `QueueConfig`. No legacy factory overload may bypass the two-arena invariant; no missing-workspace fallback may allocate.
- Current core/copy/binary callers and tests that accept ranks above eight must migrate to rank-eight coverage and explicit rank-nine rejection tests. Do not narrow dtype or broadcast support.
- Current queue coexistence/stress tests create six or more queues in some paths; migrate every such test to the deliberate four-live-queue cap and fifth-queue rejection.
- Update `include/iom` contracts, `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and examples/driver setup to distinguish native backing allocation, suballocation, slot leasing, and host allocation.

Principal implementation touchpoints: `include/iom/alloc.hpp`, `src/alloc.cpp`, `include/iom/device.hpp`, `include/iom/tensor.hpp`, `include/iom/iom.hpp`, `src/iom.cpp`, backend factory headers/device implementations, `src/shared/gpu_queue.hpp`, `src/shared/metadata_slot_pool.hpp`, `src/shared/event_ring.hpp`, `src/shared/staging_pool.hpp`, `src/shared/transfer_pool.hpp`, `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_add.inl`, `src/sycl/copy.cpp`, and `src/sycl/staging_pool.*`. Shared behavior belongs in `test/backend`; native allocation instrumentation and setup belong in backend drivers.

## 9. Acceptance and verification

### Allocation and data preservation

- Instrument actual IOM native allocation/free boundaries, not only `iom::Allocator`: current hidden metadata allocations bypass that interface.
- For standard-GPU Devices with default and custom C (including 1 and 17), verify exactly two setup backing-allocation calls on success, checked `4 × C × 512` metadata sizing, and exactly one device-wide `FixedSizeAllocator` spanning those slots. Verify zero/overflow/unprovisionable QueueConfig failures before publication.
- Creating/destroying tensors and raw workspaces after setup, and all submission, dispatch, retirement, waits, view transforms, and host transfers from first use onward, perform zero native alloc/free calls and no resource-array resizing. Preserve the two backing allocations through queue recreation and arena exhaustion.
- Descriptor addresses are within metadata backing; all tensor/raw-workspace addresses are within data backing. No domain crosses into the other. Independently observe tensor results, untouched planes and padding.
- Exhaust and free fragmented data ranges; verify correct `bad_alloc`, safe reuse/coalescing, stable live addresses and transactional failure behavior. Do not require allocations to succeed merely because aggregate free bytes suffice.
- Prove rank-eight owner/view/broadcast correctness; migrate current larger-rank coverage and verify rank-nine creation, binary result shapes, and rank-increasing transforms fail before side effects. Assert compiled rank-eight descriptor sizes/alignment fit the slot.

### Lifetime and admission

- Hold C in-flight completions, submit at least two additional valid operations, and verify both calls promptly return positive tokens without native resource acquisition or output mutation. Release credits one at a time and observe autonomous dispatch of parked requests strictly FIFO without wait or new submission; repeatedly wait on earlier and parked tokens and observe their original results.
- Verify default C=16 and custom capacities below and above 16, including C=1 and C=17. Exercise four queues with four disjoint metadata partitions, reject a fifth queue, roll back failed queue construction, safely reuse a drained partition, and never reuse a partition retained by unknown completion/quarantine.
- Reclaim completed credits, slots, and events without requiring explicit user wait. Reuse resources, then repeatedly wait on earlier success and failure tokens; completed token history must not consume native credits.
- Submit different views of the same owner on multiple queues, plus binary broadcasts with different mappings. Verify immutable snapshots, owner/tensor lifetime retention, and descriptors cannot be overwritten by later submissions.
- Force preacceptance host snapshot/queue-node allocation failure and prove all reservations, owner registrations, and workspace leases roll back with no token/effects. Exercise temporary transformed views, raw-workspace retention while parked, workspace overlap/exclusivity, and invalid range/alignment/device/operand-overlap precedence.
- Verify no-op/inline/no-metadata operations follow the same FIFO without a dummy metadata slot and cannot bypass a parked head. Force failures after acceptance and verify repeatable terminal waits, autonomous retirement, and quarantine until covering proof.
- Destroy queues with parked and executing work and verify every accepted request drains FIFO to a terminal outcome; no accepted work is silently cancelled or discarded. Verify cross-device isolation, including independent quotas and quarantine on one device.

### Backend conformance and environment-dependent evidence

- Run shared backend conformance for CPU, CUDA, ROCm, SYCL, and TTNN under the project's enabled-backend rules; enabled hardware tests must fail rather than skip. Keep backend-driver/native instrumentation for allocation boundaries and shared conformance for normative behavior. Accelerator builds/tests/profiling must follow the remote-development workflow.
- Allocation counts, checked fixed capacity, correctness, FIFO parking/dispatch, partition disjointness/reuse/quarantine, and no-growth/no-post-construction-native-allocation are hard automated gates.
- Report two/eight-GPU topology isolation and accounting when matching hardware is available; these are environment-dependent evidence, not universal CI blockers. Do not claim an eight-GPU run from single-device tests, and do not introduce a global registry.
- Compare first-use and repeated copy/binary/host-transfer paths on identical hardware/shapes against the pre-change baseline when that baseline and matching hardware are available. Record native allocation counts, peak reserved bytes, latency distribution and throughput, including small descriptors, rank-eight broadcasts, and large payload transfers. Treat this as performance evidence, not a universal deterministic pass/fail gate. Investigate and report any repeatable warmed compute-path regression above 5%; report serialized-host-transfer throughput separately from compute throughput.

Expected mechanism: removal of native allocator churn and metadata external fragmentation, bounded reservation and no grow-time spikes. Caller workspace retains the SYCL host fallback's transfer/CPU costs; this change must not claim to remove them. The metadata budget is `4 × C × 512` bytes and is negligible relative to consumer VRAM; the explicitly chosen data arena reserves its full capacity even when only partially used.


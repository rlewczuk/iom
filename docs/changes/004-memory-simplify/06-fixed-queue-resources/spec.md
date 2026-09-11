# Preallocate fixed GPU queue resources

**Order:** 06
**Priority:** P0 — bounded admission depends on fixed queue credits, disjoint metadata partitions, and completion resources being provisioned before work is accepted.
**Blocked by:** `01-native-allocation-instrumentation`, `03-rank-eight-contract`, `04-device-memory-arenas`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

CUDA, ROCm, and SYCL Devices enforce at most four live `DeviceOps` queues per exact Device identity. Each queue is published only after it owns one disjoint partition of exactly `C` slots from the Device's single metadata-arena `FixedSizeAllocator`, exactly `C` completion resources, and fixed host metadata mirrors and bookkeeping. Queue construction is transactional; queue destruction returns resources only after a covering completion proof, and unknown native use retains the partition and queue reservation.

## Scope

- Update the CUDA, ROCm, and SYCL Device/queue implementations and their shared queue/resource machinery to use the Device-owned fixed metadata allocator established during Device setup.
- Replace per-queue lazy/growing event and metadata behavior with fixed per-queue ownership. `C` is the immutable, nonzero `QueueConfig::max_in_flight_per_queue` selected at Device construction (default 16); every queue on that Device uses the same `C`.
- Preserve the existing CUDA/ROCm inline copy-descriptor path for rank 2–8. Inline copies, no-op copies, and other no-metadata paths consume no metadata slot; CUDA/ROCm binary operations and SYCL pointer-copy descriptors consume at most one slot.
- Migrate focused queue/resource tests and callers, including obsolete direct lazy-pool tests, without changing CPU or TTNN queue caps, host-transfer resource ownership, or admission/parking policy owned by later tasks.

## Implementation references

- `src/shared/gpu_queue.hpp`: `iom::detail::GpuQueue`, its constructor/destructor, `make_worker_callbacks`, `copy_impl`, and `binary_impl`; queue-count/partition reservation must precede `Policy::create_queue_stream()` and `worker_.start()`.
- `src/shared/event_ring.hpp:126-236`: `EventRingState::acquire`, completion/retirement, and queue-drain handling; replace the current lazy `created_count_` event ring with exactly `C` eagerly created completion resources and proof-based reuse.
- `src/shared/metadata_slot_pool.hpp:83-107`: `MetadataSlotPool::ensure_slot_capacity`; remove lazy replacement allocation/free and make leases address fixed 512-byte slots in the Device-wide pool, partitioned exclusively per queue.
- `src/cuda/copy.hpp` (`gpu_policy::create_queue_stream`, `create_event`, destruction/synchronization policy) and `src/rocm/copy.hpp` (the corresponding `gpu_policy`); `src/cuda/copy.cu:make_queue` and `src/rocm/copy.hip:make_queue` pass the Device resource owner into `GpuQueue`.
- `src/sycl/copy.cpp:54-184`: `SyclMetadataSlotPool` and its `ensure_slot_capacity`/native free paths; `src/sycl/copy.cpp:411-465`: `SyclQueue` construction/destruction; `src/sycl/copy.cpp:863-903`: pointer-copy metadata submission. Remove this duplicate pool and use the same fixed partition/resource protocol as CUDA/ROCm.
- `src/sycl/device.cpp:SyclDevice::create_ops` and `src/cuda/device.cpp:CudaDevice::create_ops` / `src/rocm/device.cpp:RocmDevice::create_ops`: enforce the four-live-queue reservation and publish only fully initialized queues.
- `src/shared/standard_tiled_copy.inl`: `InlineCopyMetadata`, `CopyMetadataLayout`, `copy_metadata_layout`, and `write_copy_metadata`; `src/shared/standard_tiled_add.inl:35-85`: `BinaryMetadata`, `binary_metadata_storage_bytes`, and `make_binary_metadata`. Use actual compiled layouts for assertions: rank-eight copy is `48 + 24 * 6` leading-rank bytes and binary storage is `sizeof(BinaryMetadata) + 4 * 8 * sizeof(std::uint64_t)`; both must fit 512 bytes at alignment 32.
- Focused coverage belongs in `test/cuda/test_cuda_smoke.cpp`, `test/rocm/test_rocm_smoke.cpp`, `test/sycl/test_sycl_smoke.cpp`, their conformance drivers, and shared queue/coexistence coverage where applicable. Replace the current CUDA event-ring tests at `test/cuda/test_cuda_smoke.cpp:645-795` and ROCm event-ring tests at `test/rocm/test_rocm_smoke.cpp:238-427` rather than preserving tests of lazy growth.

## Requirements

1. Define the queue resource geometry in terms of `C`: valid `C` values include `1`, the default `16`, and `17`; a Device has exactly four live queue reservations and exactly `4 * C` metadata slots split into four disjoint `C`-slot partitions. Each live queue eagerly owns exactly `C` completion resources, so the per-Device native in-flight and completion-resource upper bound is `4 * C`. Do not add a per-queue override or live reconfiguration path.
2. Reserve the queue-count credit and one disjoint `C`-slot range atomically at the owning Device boundary from the one Device-wide `FixedSizeAllocator`. Serialize metadata-allocator bookkeeping at that Device boundary, but never hold its lock across native stream/event creation, waits, submissions, callbacks, or worker drains. Never borrow, split, resize, or reassign another queue's partition. Reservation and all queue-local host bookkeeping must roll back if any later setup step fails.
3. Eagerly create exactly `C` completion resources through the backend policy (`cudaEventCreateWithFlags`, `hipEventCreateWithFlags`, or the SYCL equivalent) and exactly `C` fixed host metadata mirrors/bookkeeping records. Setup failure destroys only resources already created and returns the partition/count reservation while the native context remains valid. No fifth queue may create a stream, worker, event, or other queue resource: it throws `std::bad_alloc` before stream/worker creation.
4. Publish a queue only after its partition, all completion resources, host mirrors/bookkeeping, native stream/queue, worker, and registry state have succeeded. Preserve C++20 non-copyable/non-movable ownership and existing backend policy boundaries; do not add competing allocator implementations, public operation/direction enums, or duplicate owner query APIs.
5. Expose fixed queue-resource leases that provide exactly one completion resource and at most one metadata slot to a later dispatch. Inline CUDA/ROCm rank-2–8 copies and no-op/no-metadata operations must remain resource-correct without a dummy slot. CUDA/ROCm binary and SYCL pointer-copy metadata must use no more than one slot; no implemented operation may require more than one. FIFO-head eligibility and parked admission are owned by tasks 08–10.
6. Keep every submitted descriptor immutable for its whole asynchronous use. A descriptor's device address must lie within the Device metadata backing and its host mirror must not be overwritten or reused until upload and all device readers have completed. Simultaneous views of one owner and simultaneous submissions must have independent snapshots.
7. Remove `MetadataSlotPool::ensure_slot_capacity`, `SyclMetadataSlotPool` duplicate/growing behavior, lazy event creation, poisoned-buffer replacement, metadata native allocation/free, and any resource-array resizing. After Device setup, queue submission, dispatch, retirement, waits, and queued operations perform no IOM native device memory allocation/free and no metadata growth; only fixed-slot bookkeeping and proven leases change. Synchronous host-transfer allocation removal remains task 11.
8. On normal completion, release a slot/event/credit only after completion and all upload/device/host accesses are proved safe, independent of whether the caller has called `wait`. On queue destruction, stop new acceptance and drain accepted work. A safely drained queue returns its partition and queue-count reservation for reuse.
9. If completion or native use is unknown, quarantine the entire unresolved queue lease: queue credit, completion generation/resource, metadata partition/slot, host mirror where needed, and captured references. Retain both partition and queue reservation until a covering proof for that queue; a drain of another queue is insufficient, and live slots must never be reassigned. Later reclamation may return the resources without changing the original token outcome.
10. Add compile-time checks against the actual compiled descriptor types and alignment, not estimates: the rank-eight `InlineCopyMetadata`/copy descriptor representation corresponding to `48 + 24 * 6` and the rank-eight `BinaryMetadata` representation corresponding to `sizeof(BinaryMetadata) + 4 * 8 * sizeof(std::uint64_t)` must each be `<= 512` bytes and satisfy 32-byte alignment. A future overflow requires an intentional ABI/spec update, never automatic slot growth.
11. Add meaningful focused regression coverage for `C = 1`, `16`, and `17`; four disjoint partitions; fifth-queue rejection before native stream/worker creation; failed construction rollback; safe drained partition reuse; and unknown-completion non-reuse followed by reclamation only after a covering proof. Cover descriptor backing addresses, immutable simultaneous view descriptors, no-dummy-slot inline/no-metadata paths, no post-setup native memory calls, and no resizing.
12. Migrate callers and tests that assumed six or more live GPU queues or a lazily growing 16-entry pool. Keep queue parking/admission semantics and CPU/TTNN queue caps in task 08, CUDA/ROCm adoption in task 09, SYCL adoption in task 10, and host-transfer resources in task 11; this task must not implement those contracts.

## Non-goals

- Queue parking, native-credit admission, FIFO dispatch policy, or the host backlog; those are task 08.
- CPU or TTNN queue caps/resources; those belong to task 08.
- Host-transfer staging pools, transfer streams/queues, or host-transfer workspace APIs; those are task 11.
- Tensor/data-arena sizing, `DeviceMemoryConfig`, allocator instrumentation, raw workspaces, or rank validation beyond descriptor layout checks; those are specified by the blocked tasks and other mini-specifications.
- Changes to arithmetic, broadcasting, tile representation, codecs, owner/view addressing, or unsupported operations.
- Cross-queue borrowing, metadata compaction, lazy allocation, automatic growth, or adding public operation/direction enums and duplicate query APIs.

## Acceptance criteria

- For `C = 1`, `16`, and `17`, one Device provisions four disjoint queue partitions of exactly `C` slots and exactly `C` completion resources per successfully created queue; a fifth `create_ops()` throws `std::bad_alloc` before stream/worker creation.
- A fault at each queue setup stage leaves no published queue and returns every acquired partition/count/resource; a safely drained queue can be recreated and receives its own returned partition. A queue with unknown completion retains its reservation and partition, cannot be reused by a new queue, and is reclaimed only after a covering proof.
- Runtime instrumentation observes descriptor addresses inside metadata backing, no metadata native allocation/free or array resizing after setup, no metadata/data-domain crossover, and no host mirror reuse before asynchronous upload completion. Actual compile-time layout/alignment assertions pass for the rank-eight copy and binary descriptors.
- Inline rank-2–8 CUDA/ROCm copies and no-op/no-metadata operations follow the same resource lifecycle without dummy slots; CUDA/ROCm binary and SYCL pointer-copy paths use at most one immutable fixed slot, and simultaneous view descriptors remain independent.
- The focused CUDA, ROCm, and SYCL smoke/conformance coverage exercises construction rollback, disjointness, rejection, safe reuse, quarantine, and descriptor/resource invariants. Existing obsolete direct lazy event/metadata pool tests are removed or replaced by these queue-owned behavior tests.

## Verification

The following are proposed gates and are not run by this specification writer. Accelerator commands MUST run through the `remote-development` workflow:

- `ctest --test-dir <build> --output-on-failure -R '^iom_cuda_smoke_tests$'` (remote-development).
- `ctest --test-dir <build> --output-on-failure -R '^iom_cuda_conformance_tests$'` (remote-development).
- `ctest --test-dir <build> --output-on-failure -R '^iom_rocm_smoke_tests$'` (remote-development).
- `ctest --test-dir <build> --output-on-failure -R '^iom_rocm_conformance_tests$'` (remote-development).
- `ctest --test-dir <build> --output-on-failure -R '^iom_sycl_smoke_tests$'` (remote-development).
- `ctest --test-dir <build> --output-on-failure -R '^iom_sycl_conformance_tests$'` (remote-development).
- Run the focused queue/resource instrumentation scenarios for `C = 1`, `16`, and `17` and inspect native allocation/event/stream traces for the no-post-setup-allocation and rollback invariants; accelerator runs use `remote-development`.

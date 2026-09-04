# Fence queued tensor storage before destruction and quarantine failed resources

**Order:** 49
**Priority:** P0 — high, strongly-supported lifetime defect. Tensor destructors free CPU, CUDA, ROCm, and TTNN storage without proving that queued work has stopped using it.
**Blocked by:** `47-ST-001-fence-backend-queue-teardown`, `48-ST-002-own-cpu-ttnn-task-views`
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-003`
**Review severity:** high
**Review verification:** strongly-supported, confidence 88

## Outcome

Every backend tensor destructor consults device-owned outstanding-work state before freeing storage. A live entry is fenced exactly once; successful completion removes that entry and permits normal cleanup. A failed or queue-invalidated fence never frees or reuses the storage: it moves the cleanup action into a device-owned quarantine drained only after all device tensors and queues are gone.

Registration occurs inside each AR-002 `Execute` callback before the helper publishes the staged task, so a worker cannot complete a task before its source and destination entries exist. Completion releases the exact entry IDs recorded for that sequence, never an entire address bucket. Queue teardown invalidates its entries before destroying native queue resources; invalidated entries remain indexed by address so a later tensor destructor quarantines them instead of freeing them.

The public `Device`, `Tensor`, `TensorView`, `Allocator`, and `DeviceOps` interfaces remain unchanged. The implementation uses private shared registry code plus typed per-backend state headers; it does not expose backend implementation classes through `include/iom/device.hpp`.

## Current failure

The destructors at `src/cuda/device.cpp:141-154`, `src/rocm/device.cpp:101-114`, `src/cpu/device.cpp:183-185`, and `src/ttnn/device.cpp:211-214` release storage directly. A queued copy can still read or write that storage. CUDA/HIP allocator behavior may be asynchronous; a CPU free can return an address to a recycling allocator while `copy_elements` still runs; TTNN native plane values can be destroyed while a queued command references them.

The review evidence is the absence of any connection from tensor destruction to queue tokens, events, streams, or the device-wide TTNN command queue. The existing `GatedAllocator` and `ReusingCudaAllocator` fixtures demonstrate the expected no-reuse invariant but do not enforce it. The impact is silent corruption, a driver fault, or a use-after-free.

## Scope

- **Create:** `include/iom/outstanding_work_registry.hpp` with private generic `OutstandingWorkRegistry`, `EntryId`, `EntryState`, `FenceResult`, `CleanupAction`, `AllocatorCleanupAction`, and `Quarantine` declarations/definitions. It includes no TTNN header.
- **Create:** `src/cuda/registry_state.hpp`, `src/rocm/registry_state.hpp`, `src/cpu/registry_state.hpp`, and `src/ttnn/registry_state.hpp` with typed state and backend-local bridge functions. The TTNN header owns the concrete native-plane cleanup action.
- **Modify:** each backend `Device` to own the registry, quarantine, entry-id counter, and a stable backend state object. Each tensor receives a pointer to that state; each queue receives the same state's queue identity.
- **Modify:** each backend queue's AR-002 `Execute` and `Complete` callback wiring. Register source and destination handles inside Execute before it returns; record the two entry IDs by task sequence; release or invalidate those IDs from Complete.
- **Modify:** `~CpuTensor`, `~CudaTensor`, `~RocmTensor`, and `~TtnnTensor` to snapshot entries for their storage, invoke fences outside registry locks, and free or quarantine only after all entries are classified.
- **Modify:** queue destructors to invalidate their registry entries before ST-001 destroys stream/event/mesh resources. Invalidation must keep address indexes and replace each callable with a no-op failed fence.
- **Modify:** backend device destructors to drain quarantine after all tensors and queues have been destroyed.
- **Tests:** add deterministic CPU recycling/poisoning coverage and CUDA/ROCm/TTNN hardware regressions where the backend test conventions support them.

## Implementation references

- **Create:** `include/iom/outstanding_work_registry.hpp`. `OutstandingWorkRegistry::Entry` stores an `EntryId`, address, sequence, owning queue ID, `EntryState`, and a `std::function<FenceResult()>`. `snapshot_for(address)` returns copies of IDs, states, and fence callables. `register_entry`, `try_release_entry`, `invalidate_entries_for_sequence`, `invalidate_entries_for_queue`, and `remove_entries` are mutex-protected. `invalidate_entries_for_queue` marks entries `Invalidated`, replaces their fence with a no-op failed callable, clears only sequence indexing, and retains address indexing until explicit removal.
- **Create:** `src/cuda/registry_state.hpp`. Define `CudaRegistryState`, typed source/destination handle accessors for AR-002's raw `TensorView*` task fields, and bridges to the generic registry. CUDA fence callables accept only the opaque `Task::fence` resource and a stable CUDA event synchronization function; they capture no `CudaQueue*` or stream member.
- **Create:** `src/rocm/registry_state.hpp`. Mirror the CUDA state with HIP event synchronization and AR-002's raw view pointers.
- **Create:** `src/cpu/registry_state.hpp`. Define CPU typed state and value-view accessors for ST-002's `Task::source` and `Task::destination`. A CPU fence is a no-op success for work that completed, and queue invalidation replaces it with failed status.
- **Create:** `src/ttnn/registry_state.hpp`. Define TTNN typed state and value-view accessors. Add `TtnnNativeCleanupAction`, which owns the moved `std::vector<ttnn::Tensor>` and its mesh/device cleanup context and runs native destruction only from quarantine drain.
- **Modify:** `src/cuda/copy.cu`, `src/rocm/copy.hip`, `src/cpu/device.cpp`, and `src/ttnn/device.cpp`. In each Execute callback, enqueue native work, set the post-AR-002 `Task::fence`, register source and destination entries before Execute returns, and insert a `SequenceOutcome { source_entry_id, destination_entry_id, fence_succeeded, retained_failure }` under the queue mutex. In Complete, call the existing `DeviceOps::complete` and then release exact IDs on success or mark them invalidated on fence/retained failure.
- **Modify:** the four queue destructors. Invoke `invalidate_entries_for_queue` before native stream/mesh destruction and before clearing queue-local sequence state. ST-001 remains responsible for queue event/stream fencing.
- **Modify:** the four tensor destructors. CPU/CUDA/ROCm create `AllocatorCleanupAction` for the address and byte count; TTNN moves its native-plane vector into `TtnnNativeCleanupAction` when any entry fails. A destructor never calls an allocator free for an address with a failed or invalidated entry.
- **Read/consume:** `docs/changes/0001-tensor-view/30-AR-002-share-backend-queue-scaffold/spec.md` for Execute publication ordering, exact four callbacks, `Task::fence`, and completion behavior.
- **Read/consume:** `docs/changes/0001-tensor-view/47-ST-001-fence-backend-queue-teardown/spec.md` and `48-ST-002-own-cpu-ttnn-task-views/spec.md` for queue teardown and CPU/TTNN task field contracts.
- **Read:** `include/iom/tensor.hpp`, `include/iom/alloc.hpp`, backend device implementations, and existing allocator fixtures for stable storage addresses, byte counts, native plane ownership, and exception categories.

### Registration and completion protocol

For each task that can reference storage, Execute obtains backend handles from the task's actual final shape:

- CUDA/ROCm: `task.source->native_handle()` and `task.destination->native_handle()`.
- CPU/TTNN: `task.source.native_handle()` and `task.destination.native_handle()`.

It creates two entries with the same sequence and queue ID. The fence callable captures only the per-task opaque fence and stable backend synchronization function. It does not capture a queue object, stream member, mesh reference, tensor view reference, or stack address. Registration is sequenced before AR-002 publishes the task to its worker.

The queue's sequence map is protected by its existing submission mutex. On successful fence completion and `failure == nullptr`, Complete removes both entries. On a retained post-link failure or a fence exception, Complete marks both IDs invalidated and leaves their address index. If registration throws before Execute returns, AR-002 removes the staged task and ST-007 rolls back the sequence; partially registered IDs are removed in the Execute rollback guard.

### Tensor destructor protocol

The destructor snapshots all entries indexed by its storage address, then releases the registry lock before invoking fences. For each snapshot, a live entry invokes its callable and records success/failure; an invalidated entry is failed without invoking its no-op callable. If every entry succeeds, the destructor performs the normal allocator/native-plane cleanup and removes all snapshot IDs. If any entry fails, it adds exactly one cleanup action to quarantine and removes all snapshot IDs without freeing the storage directly. Cleanup-action insertion and registry removal are exception-safe; an unexpected destructor exception is converted to quarantine rather than escaping.

The device destructor drains quarantine after its tensors and queues. `AllocatorCleanupAction::run()` calls the original allocator exactly once and records a cleanup failure without throwing. `TtnnNativeCleanupAction::run()` finishes the required TTNN command queue under the device API mutex and then destroys the moved native plane vector, recording failures without throwing.

## Requirements

1. Registry state is device-owned and outlives every tensor and queue created by that device.
2. Registration happens inside Execute before AR-002 staged publication; registering after `submit` returns is prohibited.
3. Entries are keyed by unique `EntryId`; completion releases exactly the source and destination IDs for one sequence, not all entries for an address.
4. Registry fences capture only stable per-task resources. No fence callable captures a queue pointer, stream, mesh reference, tensor view, or caller stack object.
5. Queue invalidation marks entries failed and retains address indexes until tensor cleanup removes them. It must not erase an invalidated entry from `by_address_` early.
6. A tensor destructor never frees storage after a failed or invalidated fence. Failed storage enters quarantine and is not reused before device teardown.
7. CPU allocator cleanup preserves the allocator API and exact one-free behavior. CUDA/ROCm cleanup preserves caller allocator ownership. TTNN quarantine owns the native plane values themselves.
8. Existing AR-002, ST-001, ST-002, ST-006, and ST-007 completion, failure, no-op, and teardown contracts remain observable.
9. Registry and quarantine code in common headers is free of CUDA, HIP, SYCL, and TTNN types; backend-specific native cleanup remains behind typed private headers.
10. Registry operations validate duplicate IDs, missing IDs, address mismatches, and EntryId overflow before mutating state; invalid operations use established standard exception categories.
11. Destructors and quarantine drain are no-throw effective. Cleanup errors remain recorded for diagnostics and do not permit unsafe direct reuse.
12. A task that references the same storage in both source and destination creates two independent entries and removes both exactly once.
13. No public header or public API gains registry, quarantine, queue, or native-state accessors.

## Non-goals

- Changing `Device`, `Tensor`, `TensorView`, `Allocator`, or `DeviceOps` public signatures.
- Replacing ST-001's queue event/stream teardown fence or ST-002's owned CPU/TTNN task views.
- Adding implicit global synchronization to every device operation, a public cancel API, or cross-queue dependency tracking.
- Changing allocator implementations, TTNN layout, supported data types, or native plane representation.
- Making a destroyed queue waitable or preserving its private failure map after destruction.
- Adding a backend switch or global active-backend registry to common code.

## Acceptance criteria

- [ ] A CPU poisoning/recycling allocator test submits a copy, destroys the source or destination tensor without waiting, and proves no freed address is reused until the queued operation is complete; a failed fence quarantines instead of freeing.
- [ ] CUDA and ROCm tests with `ReusingCudaAllocator`/the HIP equivalent submit a copy, destroy an operand before wait, and pass Compute Sanitizer/ROCm memory diagnostics without use-after-free or invalid access.
- [ ] TTNN coverage destroys an operand with queued native planes and proves the moved plane vector remains owned by quarantine until safe drain.
- [ ] A two-copy same-address test proves completion removes only the exact EntryIds for each sequence and never releases a concurrent entry early.
- [ ] A queue-destruction test proves invalidated entries remain address-indexed and the later tensor destructor quarantines them rather than freeing them.
- [ ] A post-link retained-failure test proves Complete marks both entries invalidated, repeated `wait` still observes the original failure, and no direct free occurs.
- [ ] Source inspection confirms registration is inside Execute before AR-002 publication, fences capture no queue/tensor/view lifetime, and common registry headers include no backend SDK.
- [ ] Device teardown drains quarantine after all backend tensors/queues and invokes each cleanup action at most once.
- [ ] Existing CPU, CUDA, ROCm, and TTNN conformance suites retain their previous observable case coverage and pass on supported configurations.

## Verification

Run CPU verification locally with the repository CMake/CTest workflow, including the allocator poisoning/reuse and same-address multi-entry cases. Run CUDA, ROCm, and TTNN builds and hardware tests only through `.agents/skills/remote-development`; use exclusive accelerator access and the backend's memory diagnostics. Exercise queue teardown, retained failures, temporary/owned task views from ST-002, and tensor destruction in both source and destination roles. Use source audits to verify registration ordering, exact EntryId release, retained address indexes after invalidation, absence of backend types in the shared header, and absence of queue captures in fence callables. Clean remote mirrors after verification.

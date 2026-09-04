# Replace per-plane GPU copies with one grid-stride launch and bounded metadata slots

**Order:** 42
**Priority:** P1 — medium, strongly-supported performance defect. The current CUDA and ROCm copy paths allocate an O(plane-count) host vector and launch once per plane; the fix must make submission and launch count independent of the leading-dimension product.
**Blocked by:** `28-ST-007-commit-sequence-before-queue-work`, `29-AR-001-share-cuda-rocm-copy-queue`, `30-AR-002-share-backend-queue-scaffold`
**Source:** `docs/changes/0001-tensor-view/review.md` — `PF-002`
**Review severity:** medium
**Review verification:** strongly-supported (mechanism code-deterministic; magnitude unmeasured), confidence 85

## Outcome

For CUDA and ROCm, one non-identical `DeviceOps::copy` submission enqueues one grid-stride copy kernel and one native completion event, regardless of the number of leading-dimension planes. An identical source/destination window remains a no-op with no kernel, metadata slot, or native event. Plane coordinates and view strides are encoded in a queue-owned metadata slot and decoded by the kernel. The caller thread performs no plane enumeration and constructs no per-submit `std::vector`.

The metadata slot is acquired inside the AR-002 `Execute` callback, after the no-op check, and remains owned until the task's existing `fence_destroy` callback runs. A backend-private fence resource bundles the native event and metadata-slot lease. Failed pre-link work releases both resources before propagating; the existing post-link retained-failure contract calls `DeviceOps::commit_failure`, clears `task.fence`, and returns the valid token. Public headers and `DeviceOps::copy` signatures remain unchanged.

## Current failure

`src/cuda/copy.cu` and `src/rocm/copy.hip` currently enumerate `P = product(leading dimensions)` `PlanePair` values on the submitting thread. They then launch once per pair, so a view such as `[1, 4096, 32, 128]` causes thousands of launches even though each plane contains only a small amount of work. The vector allocation and repeated launch/error-check path are on the token submission hot path. The same implementation also creates one per-task event directly instead of reusing bounded queue-owned metadata and event resources.

The violated invariant is that asynchronous copy submission cost and launch count must scale with logical bytes, not with the number of leading planes. The review evidence is the explicit `PlanePair` vector and `launch_copy_plane` loop at `src/cuda/copy.cu:319-352,450-499`, mirrored at `src/rocm/copy.hip:392-425,482-509`.

## Scope

- Replace the CUDA and ROCm device-to-device `PlanePair` enumeration and per-plane launch loop with one `grid_stride_copy_kernel` launch per non-identical copy.
- Add one private metadata-slot pool per queue. Each pool has a bounded number of reusable slots, a host mirror, and a device mirror. A slot is not reused until its task fence is destroyed.
- Fill one slot with source/destination plane offsets, plane strides, leading dimensions, rows, columns, plane count, and leaf bit width. The kernel decodes a logical plane index from its global work index and performs the existing logical copy semantics.
- Keep AR-002's exact `StagedWorker<Task>` and four callbacks. PF-002 adds no worker, mutex, callback, public header, `Task`, or `DeviceOps` abstraction.
- Preserve CUDA and ROCm fault injection, pre-link rollback, post-link retained failures, null-fence behavior, event ordering, and no-op sequence consumption.

Out of scope: CPU, TTNN, SYCL, host-transfer staging, word-oriented kernel arithmetic, allocator APIs, and public ABI changes.

## Implementation references

- **Modify:** `src/cuda/copy.cu` — remove `PlanePair`, `view_planes`, the per-plane `launch_copy_plane` loop, and caller-side vector construction. Add `CudaCopyMetadataHeader`, `CudaMetadataSlotPool`, `CudaFenceResource`, the grid-stride kernel, and the CUDA `Execute`, `fence_complete`, and `fence_destroy` callback bodies consumed by AR-002.
- **Modify:** `src/rocm/copy.hip` — mirror the CUDA metadata header, slot pool, fence resource, kernel, and callbacks with HIP types and launch syntax.
- **Read/consume:** `include/iom/iom.hpp` and `docs/changes/0001-tensor-view/30-AR-002-share-backend-queue-scaffold/spec.md` — `Task { sequence, source, destination, event, no_op, fence }`, `StagedWorker<Task>`, `Execute`, `fence_complete`, `fence_destroy`, and `complete`. The four callback members remain exactly four.
- **Read:** `include/iom/tensor.hpp` — `TensorSpec::TILE`, logical dimensions, leaf bit widths, plane offsets, and plane strides.
- **Read:** `src/cuda/copy.cu` and `src/rocm/copy.hip` — existing `plane_slot`, launch error helpers, context/device guards, and `SubmissionFault` seams. Preserve error categories and test names.
- **Modify:** `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp` only if an existing or focused regression needs an assertion for one launch and zero launches for a no-op. Do not weaken existing byte-exact conformance.

### Metadata format and bounds

Use a trivially-copyable fixed header followed by three variable-length `std::uint64_t` arrays in one contiguous allocation:

```cpp
struct CudaCopyMetadataHeader {
    std::uint64_t source_plane_offset;
    std::uint64_t destination_plane_offset;
    std::uint64_t rows;
    std::uint64_t columns;
    std::uint64_t plane_count;
    std::uint32_t bits;
    std::uint32_t leading_rank;
};
```

The arrays are, in order, source plane strides, destination plane strides, and leading dimensions. Their total byte size is checked before multiplication or allocation. `plane_count`, `rows`, `columns`, and `bits` are derived from validated `TensorSpec` and view metadata. Every multiplication and byte-size addition uses checked arithmetic and throws `std::overflow_error` before any enqueue on overflow.

`grid_stride_copy_kernel` receives source storage, destination storage, and one device metadata pointer. It computes `words_per_plane` from padded rows, padded columns, tile size 16, and `bits`; `total_words = plane_count * words_per_plane`; and uses a grid-stride loop over `total_words`. The logical plane index is decoded before applying the leading-coordinate stride walk. The kernel writes only the destination window and never reads a host vector.

### Slot and fence ownership

`CudaMetadataSlotPool` and `HipMetadataSlotPool` are private queue members with `kMetadataSlotCount = 16`. `acquire()` blocks until a slot is free. `ensure_slot_capacity(required_bytes, stream)` grows host and device mirrors geometrically, after synchronizing the queue stream, and commits both mirrors only after replacement allocation succeeds. A failed replacement allocation leaves the old mirrors and slot accounting intact. `release(slot)` is idempotence-protected by the pool's slot state and notifies a waiter.

`CudaFenceResource` contains `cudaEvent_t event`, the metadata slot index, and a pointer to the owning pool; the HIP resource is symmetric. On successful execution, `task.event` is set to the resource event and `task.fence` is the resource pointer. `fence_complete(void*)` synchronizes the resource event. `fence_destroy(void*)` destroys the event, releases the slot, and deletes the resource, in that order. No callback captures a queue-local reference beyond the opaque resource lifetime.

The Execute callback sequence is:

1. Return immediately for `task.no_op` with `task.fence == nullptr`.
2. Acquire one metadata slot.
3. Checked-build and write the metadata header and arrays.
4. Asynchronously copy the metadata to the slot's device mirror.
5. Enqueue exactly one grid-stride kernel with `total_words` work.
6. Check the launch and preserve `SubmissionFault::third_plane_launch` at the post-PF-002 wrapper boundary.
7. Create and record one event; publish the resource through `task.event` and `task.fence`.
8. On any failure before successful event publication, destroy any created event, release the slot, clear `task.fence`, and rethrow. If kernels were already enqueued, route the exception through `commit_failure(sequence, current_exception())` and return normally per AR-002/ST-006.

The slot is released only by the fence callback on the success path or by Execute's failure cleanup on the failure path. It is never released by the caller-side `copy` override.

## Requirements

1. No `PlanePair`, `view_planes`, `plane_pairs`, or per-submit vector remains in the CUDA or ROCm copy submission path.
2. A non-identical copy issues exactly one grid-stride kernel launch and one event record. An identical-window copy issues zero kernel launches and does not acquire a slot or event.
3. `total_words` is checked before launch and is based on destination storage words, not `plane_count * rows * columns` element count.
4. Metadata supports arbitrary validated leading rank; no fixed rank cap is introduced.
5. Metadata host and device mirrors have equal slot capacity. Growth is geometric, stream-safe, checked, and strongly exception-safe; allocation failure preserves the prior mirror and in-flight slot accounting.
6. The submitting thread performs no plane decomposition and no per-submit `std::vector` allocation. A single backend-private fence resource allocation is permitted for a non-identical task; pool allocations happen only during pool construction or capacity growth.
7. The event and metadata slot remain live until `fence_destroy`; failed event creation/recording and post-link retained failures release resources exactly once.
8. CUDA uses `cudaEvent_t`, CUDA launch syntax, and `ContextGuard`; ROCm uses `hipEvent_t`, HIP launch syntax, and `DeviceGuard`. Runtime failures retain established `std::runtime_error` categories and messages.
9. AR-002's helper remains the only worker and has exactly four callbacks. PF-002 does not add a `Task` field, callback, public API, queue worker, or second queue mutex.
10. Existing CUDA and ROCm transactional-failure tests continue to observe valid tokens and repeated `wait` rethrows for post-link failures. Pre-link failures still roll back the ST-007 sequence reservation.
11. All existing shared copy conformance cases remain byte-exact for full views, transformed views, high ranks, sub-byte widths, and identical windows.

## Non-goals

- CPU, TTNN, or SYCL copy changes.
- PF-003's synchronous host-transfer stream and staging pools.
- PF-006's word-oriented bit extraction and non-atomic destination stores.
- Refactoring AR-002's worker or changing ST-001/ST-002/ST-003/ST-006/ST-007 contracts.
- Public headers, allocator ownership, tensor layout, tile size, or `DeviceOps::copy` signatures.
- Unbounded metadata allocation, a fixed maximum leading rank, or a second metadata representation.

## Acceptance criteria

- [ ] Source inspection finds no `PlanePair`, `view_planes`, `plane_pairs`, or per-submit `std::vector` in the CUDA/ROCm copy submission path.
- [ ] A focused CUDA and ROCm launch-count regression observes one kernel for a non-identical many-plane view and zero kernels for an identical window.
- [ ] Shared CUDA and ROCm conformance passes for every backend-supported leaf type and every existing owner/view shape.
- [ ] Fault injection at event creation, the retained third-launch wrapper, and event record preserves the valid-token/repeated-wait contract and leaves no slot or event leak.
- [ ] A metadata-growth test exercises a rank beyond the initial slot capacity, verifies checked capacity growth, and proves an injected replacement-allocation failure leaves the prior slot usable.
- [ ] Concurrent submissions on one queue do not reuse a metadata slot before its fence is destroyed; the test reports a bounded slot count and no invalid device access.
- [ ] `CudaFenceResource` and `HipFenceResource` release event and slot exactly once on worker completion and shutdown drain.
- [ ] No public header or public operation signature changes.

## Verification

Accelerator verification follows `.agents/skills/remote-development`. Configure and build CUDA and ROCm separately on their configured hosts, then run the backend conformance and focused launch-count tests under exclusive GPU access. Run the existing transactional-failure cases and the full shared copy conformance suite. Use Compute Sanitizer memcheck/racecheck for CUDA and the repository's ROCm memory diagnostics for HIP on the metadata-growth, queue-teardown, and many-plane cases. Confirm with source inspection that the caller path contains no plane-vector allocation and that the only slot release sites are Execute failure cleanup and `fence_destroy`. Clean each remote mirror after verification.

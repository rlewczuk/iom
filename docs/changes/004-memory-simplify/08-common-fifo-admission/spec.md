# Implement common FIFO admission for CPU and TTNN

**Order:** 08
**Priority:** P1 — all backends must share accepted-versus-native-in-flight semantics before GPU adoption.
**Blocked by:** `04-device-memory-arenas`, `05-raw-workspace-contract`, `07-binary-workspace-cutover`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

CPU and TTNN use one bounded, backend-neutral FIFO admission protocol. Every positive token is accepted only after its immutable request/view/workspace snapshots, owner registrations, exclusivity leases, sequence/token history, and FIFO node are committed. Only `C = QueueConfig::max_in_flight_per_queue` FIFO heads may execute or hold native completion credit; valid excess is promptly accepted and parked without native effect or native resource use. Safe retirement autonomously dispatches parked work in strict FIFO order, and repeatable token outcomes, owner/workspace lifetime, queue teardown, quarantine, and four-queue Device limits remain correct on both backends.

## Scope

- Refactor the common admission path around the existing `detail::StagedWorker`, `DeviceOps` sequence/token history (`submit`, `complete`, `commit_failure`, and `wait`), and `detail::OutstandingWorkRegistry`. The planned admission/controller state is private/backend-neutral; do not add public operation or direction enums or duplicate owner query APIs.
- At acceptance, validate the already-migrated borrowed workspace argument and all request/view contracts, snapshot every host request and `TensorView` descriptor by value, register every tensor/native owner, acquire every required exclusivity lease (including workspace ranges), reserve the 55-bit sequence/token, and append the FIFO node before returning a positive token. The transaction must roll back all partial registrations, leases, sequence state, and nodes on failure.
- A host snapshot or FIFO-node allocation failure is synchronous `std::bad_alloc` and `OidError::ResourceExhausted`, with no positive token, output mutation, native submission, or retained owner/workspace effect. Invalid input remains synchronous and does not consume a sequence or admission capacity.
- For each queue, permit at most `C` executing/credit-holding FIFO heads. Valid excess is accepted and parked with no native credit, completion event/resource, metadata slot, device workspace allocation, native effect, or output mutation; parked work retains snapshots, owner registrations, and workspace leases. The host backlog is not bounded by a new configured limit.
- Retirement must autonomously return safe credit and dispatch the oldest parked node without a user `wait` or a new submission. No-op copies, inline work, binary operations, and operations requiring no metadata may bypass a parked head or obtain a dummy native resource. Preserve strict same-queue FIFO and no cross-queue ordering.
- Preserve success and post-acceptance failure outcomes independently from reusable credits, completion resources, descriptors, and workspace leases. A post-acceptance failure remains a terminal, repeatably waitable failure; a later submission must never replace an earlier token result.
- Quarantine every associated owner registration, workspace lease, native completion reference, credit, and queue/device reservation when completion is unknown. A covering proof may reclaim the quarantined resources without erasing the original token result. Queue destruction stops acceptance, drains all accepted executing and parked nodes to terminal outcomes in FIFO order, and transfers unresolved leases to device-owned quarantine rather than cancelling or forgetting tokens.
- Apply the exact four-live-`DeviceOps`-queue cap and transactional construction rollback/reuse to each CPU and TTNN `Device` identity. Reject the fifth queue before starting another worker/native stream; safely reuse only a proven-drained queue's reservation. Preserve process-global 255 OID queue IDs and their existing token encoding/allocation behavior.
- Adopt the common protocol in CPU and TTNN queue operations. CPU may use host execution credits but remains asynchronous at the public queue contract. TTNN retains `TtnnDevice::api_mutex` serialization and native per-plane storage; the admission layer must not redesign TTNN allocation or serialize independent queues through a new global lock.
- Define one immutable common copy-request snapshot and use it for CPU and TTNN admission; migrate any TTNN task fields that retain borrowed view objects, and make CPU construct the same snapshot before asynchronous host dispatch. Temporary transformed view objects must be allowed to die after the submitting call returns while owner identity and native handles remain protected. CUDA/ROCm/SYCL adopt this snapshot contract in tasks 09–10.

## Implementation references

- **Modify:** `include/iom/iom.hpp` — `detail::StagedWorker` (lines 35–203), `DeviceOps` facades and protected `submit`/`submit_binary` (lines 223–388); host the planned private/common admission controller and keep `Device::create_ops()` argument-free.
- **Modify:** `src/iom.cpp` — `DeviceOps::wait`, `complete`, `commit_failure`, `encode_token`, `lease_queue_id`, and `release_queue_id` (lines 484–489 and 1006–1160); retain repeatable history and the process-global 255-ID allocator while integrating credit retirement and queue admission.
- **Modify:** `include/iom/detail/outstanding_work_registry.hpp` — `OutstandingWorkRegistry`, `RegistryState`, `register_entry`, invalidation/removal, `release_or_quarantine`, and `allocate_queue_id`; use existing registration and quarantine semantics rather than parallel owner tracking.
- **Modify:** `src/cpu/device.cpp` — `CpuDevice` (lines 363–388), `CpuQueue` (lines 605–897), `copy_impl`, `binary_impl`, and queue destruction; route all CPU accepted operations through the common FIFO controller while retaining host execution and asynchronous token behavior.
- **Modify:** `src/ttnn/device.cpp` — `TtnnDevice::api_mutex` (lines 287–309), `TtnnQueue::Task`/constructors/destructor (lines 522–604), `execute`, `execute_copy`, `complete_task`, and `fence_through_sequence` (lines 627–1021); preserve per-plane native work and mutex-protected completion while removing borrowed-view task pointers.
- **Read:** `test/backend/backend_conformance_other.hpp` — `DeferredCopyQueue`, `InstrumentedQueue`, and `run_lifetime_conformance` (lines 43–485); reuse the existing deferred-failure, repeatable-wait, owner-lifetime, temporary-view, and teardown patterns for common admission coverage.
- **Modify:** `test/backend/backend_conformance_other.hpp` and `test/backend/backend_conformance_common.hpp` — add shared behavioral cases for FIFO parking/retirement, no-op fairness, snapshot/lease retention, rollback, unknown completion, and queue-cap identity behavior.
- **Modify:** `test/test_iom.cpp` — common `DeviceOps`/OID tests around `FakeQueue`, `InlineQueue`, sequence history, and queue-ID boundaries; preserve the 55-bit sequence and process-global 255-ID assertions.
- **Modify:** `test/cpu/test_cpu_conformance.cpp` — real CPU queue admission, temporary-view, failure, teardown, and four-queue/fifth-queue cases; `test/ttnn/test_ttnn_smoke.cpp` and `test/ttnn/test_ttnn_conformance.cpp` — TTNN smoke/conformance equivalents through the existing hardware fixtures.

## Requirements

- `QueueConfig` is backend-neutral, passed by value, immutable for a Device, has `max_in_flight_per_queue` default `16`, and rejects zero. Use the Device's one configured `C` for every queue; do not add per-queue overrides or live reconfiguration.
- Treat every accepted operation type, including copy no-ops, binary/no-metadata operations, and host-work execution represented by these queue paths, identically for token acceptance, FIFO order, credit accounting, owner retention, failure history, retirement, and teardown.
- Use `RawWorkspaceView{}` as the empty default. Zero-byte requirements accept the empty view with alignment 1; a positive requirement with an empty view is invalid. Validate capacity, alignment, checked offsets, exact Device identity, live owner identity, operand/output overlap, and workspace exclusivity before acceptance. A conflicting accepted workspace lease maps to `std::bad_alloc`/`OidError::ResourceExhausted` and rolls back the whole transaction.
- Do not allocate, grow, replace, or free native resources while accepting or dispatching parked work. A parked node owns only host snapshots, token outcome state, owner registrations, and workspace/exclusivity leases until it reaches the head.
- Dispatch only the FIFO head when a safe credit and all operation-specific resources are available. A later no-op, copy, or binary request must wait behind an earlier parked request even if it would require no metadata or native workspace.
- On safe completion, release native credit and completion/metadata state only after all device and host accesses are proved complete, then wake/dispatch the oldest parked node. User waits are observers and must not be required for progress or resource reuse.
- Keep token outcome history separate from reusable execution state. Repeated `wait` on old successes returns successfully; repeated `wait` on old failures rethrows the same retained exception after credits/resources have been reused. Unknown completion must not release any associated lease or reservation merely because a token was waited.
- Queue construction/destruction must be transactional and Device-local: reserve one of four live queue slots before worker/native stream publication, roll back on any setup failure, stop acceptance before destruction drain, drain every accepted node including parked work, and retain unresolved queue reservation/resources until a covering proof. A different queue's drain must not release them.
- Do not add hidden internal locks to canonical standalone allocators. Synchronization for allocator bookkeeping remains at the owning Device boundary; TTNN's existing `api_mutex` remains the native serialization boundary.
- Keep public API compatibility with the frozen contracts: no public admission/operation/direction enum, no duplicate owner query API, no token-ID redesign, no cross-queue ordering, and no alternate opt-in protocol.

## Non-goals

- GPU (CUDA, ROCm, or SYCL) admission/resource adoption; those backends consume the shared protocol in a later task.
- Device arena reservation, raw-workspace API design, or binary/host-transfer workspace migration; those are prerequisites and must be consumed rather than redesigned here.
- Cross-queue ordering, configured host-backlog limits, cancellation, token sequence/ID redesign, or a process-global Device registry.
- TTNN allocator/storage redesign, replacement of native per-plane tensors, removal of `api_mutex`, or claims about vendor-internal allocations.
- New arithmetic, shape, dtype, broadcast, tile, or operation capabilities unrelated to admission.
- Broad allocator implementation changes or hidden synchronization inside `ListAllocator`, `FixedSizeAllocator`, or standalone CPU allocators.

## Acceptance criteria

- [ ] With `C=1`, `C=16`, and `C=17`, submit `C+2` valid operations on one CPU queue and one TTNN queue. Every call promptly returns a positive token; only `C` heads can execute/hold native credit, and parked requests produce no native effect, output mutation, metadata/resource acquisition, or workspace allocation.
- [ ] Prove one-credit retirement autonomously dispatches exactly the oldest parked request in FIFO order without a user wait and without another submit. A parked earlier operation cannot be bypassed by a later no-op, copy, binary, or no-metadata operation.
- [ ] Reuse credits/resources and repeatedly wait on earlier successful and failed tokens; old successes remain successful and old failures rethrow the original exception after later submissions reuse completion state.
- [ ] Submit copies using temporary sliced/permuted views whose stack objects die at return. Completion uses immutable snapshots and retained owners/handles, not dangling `TensorView*` pointers; later view mutation cannot rewrite an accepted descriptor.
- [ ] Hold tensors, views, and a borrowed workspace alive through parked completion. Verify workspace overlap/exclusivity rejects synchronously, positive requirements reject an empty workspace, and disjoint ranges of one owner can proceed according to the frozen contract.
- [ ] Inject host snapshot, FIFO-node, owner-registration, and outcome/lease setup failures before acceptance. Verify `bad_alloc`/`ResourceExhausted` has no token, output/native effect, sequence consumption, owner registration, workspace lease, or allocator-state leak; the next successful token starts at the expected sequence.
- [ ] Force post-acceptance operation and completion failures. Verify terminal failures are waitable and repeatable, native/owner/workspace leases remain protected until completion proof, unknown completion quarantines every associated reference/credit/reservation, and a covering proof reclaims without changing token history.
- [ ] Destroy CPU and TTNN queues with executing and parked accepted work. Acceptance stops, every token drains to a terminal success/failure in FIFO order, no accepted work is silently cancelled, and unresolved leases move to Device-owned quarantine.
- [ ] Create four live queues on each CPU and TTNN Device identity, reject the fifth with `std::bad_alloc` before worker/native-stream publication, roll back a failed construction, and safely reuse a drained queue reservation. Never reuse a queue partition/reservation retained by unknown completion; a separate Device has independent capacity.
- [ ] Keep process-global OID queue IDs through 255 unchanged, preserve 55-bit sequence encoding, and keep a released queue ID reusable only after its own queue's safe drain.

## Verification

- `ctest --test-dir <cpu-build> --output-on-failure -R '^iom_tests$'` — proposed focused common/OID/admission gate; not run while writing this specification.
- `ctest --test-dir <cpu-build> --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — proposed CPU conformance gate covering real queue behavior; not run while writing this specification.
- `ctest --test-dir <ttnn-build> --output-on-failure -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests)$'` — proposed TTNN smoke/conformance gate, run through the `remote-development` workflow; not run while writing this specification.
- The focused tests must exercise all acceptance criteria at `C=1/16/17`, without relying on explicit waits or later submissions to trigger retirement, and must fail rather than skip when TTNN is enabled hardware.

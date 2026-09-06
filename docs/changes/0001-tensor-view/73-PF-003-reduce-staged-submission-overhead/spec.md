# Cut per-submission fixed cost: fixed-capacity inline type-erased fence with intrusive-refcounted CUDA/HIP leases (cached `FenceResult`), pointer-view TTNN tasks, and inline CPU completion

**Order:** 73
**Priority:** P1 — reduces the deterministic allocation/lock/handoff cost inherited by every queued operation on every `StagedWorker` backend (the future 0002 decode loop issues many ops per token); required normal-path work that gates no correctness remediation.
**Blocked by:** `62-AR-002-share-release-quarantine-protocol` — the shared lifetime machinery (registry header at `include/iom/detail/outstanding_work_registry.hpp`, `detail::RegistryState`, shared `release_or_quarantine`/`SequenceOutcome` glue) must exist in its final shape before the fence representation inside `Entry` is changed; transitively includes `61-AR-003-simplify-outstanding-work-registry`. (`67-NT-001` is **not** a blocker of this task — see Requirement 13 and the "Bench re-calibration" non-goal below for the clean dependency on a separate follow-up.)
**Source:** `docs/changes/0001-tensor-view/review.md` — `PF-003`
**Review severity:** medium
**Review verification:** strongly-supported, confidence 78

## Outcome

After 61-AR-003 (two-index registry, CPU registration opt-out) and 62-AR-002 (shared release/quarantine protocol and state glue), the remaining per-submission fixed cost is removed at three points. (1) `iom::detail::Fence` stops being a type-erasing `std::function<FenceResult()>` and becomes a **fixed-capacity inline type-erased value** with the constants pinned: `inline constexpr std::size_t kFenceStorageBytes = 32;` and `inline constexpr std::size_t kFenceStorageAlign = alignof(std::max_align_t);`. Each `Fence` carries one backend capture inside that aligned 32-byte buffer plus four function pointers (`invoke`, `copy_construct`, `move_construct`, `destroy`); every backend's capture size and alignment are pinned by `static_assert`s at the top of each backend TU; registration and `snapshot_for` produce independent copies whose inline state survives any later invocation. (2) `TtnnQueue::Task` stores the caller's two `TensorView` arguments by pointer (the shape CUDA/ROCm already use) instead of by value, eliminating four small-vector heap allocations per submission; CPU has no task at all after (3). (3) `CpuQueue` completes every submission inline on the submitting thread — `execute` already ran the work there inside `StagedWorker::submit_copy` — so `CpuQueue` drops its `StagedWorker`, task list node, worker thread, and both condvar handoffs entirely, while tokens stay waitable and repeatable and every failure category is preserved.

The CUDA/ROCm per-task resource is extended from a raw `void*` fence handle into an **intrusive-refcounted** state (`CudaFenceResource`/`HipFenceResource` gain `std::atomic<std::size_t> refcount` plus `cached_result` plus `cached_result_mu` plus `retained_failure`, and already hold the metadata slot/pool/context/event). Each registry inline capture is a one-pointer RAII `CudaFenceLease` / `HipFenceLease`; copy bumps refcount, move transfers, destroy decrements. The lease **caches the fence outcome per resource** under the resource's mutex — `invoke` and the worker's `cuda_fence_destroy` synchronize once, store the result, and every later `invoke` (including a destructor that snapshotted the fence after worker completion) returns the cached value without touching the event. The worker step releases the metadata slot then drops the Task refcount; only the very last outstanding lease destroys the now-already-synchronized event and frees the resource. `StagedWorker::process` is **not reordered**: today's `fence_complete → fence_destroy → callbacks_.complete` ordering is required because `complete()` is non-noexcept, wakes waiters on the completion CV, and stores the failure in `failures_`/`pending_failures_` under `completion_mutex_`. PF-004 (`74-PF-004-pool-gpu-copy-events/spec.md`) retains its scope to pool events; the lease protocol here is the seam PF-004 plugs a pooled slot into.

Per-copy allocation, lock-operation, and event counts are recorded before/after. The CPU bench (post-inline-completion) is left to a separate explicit follow-up task that owns the bench re-calibration (see Requirement 13 and the "Bench re-calibration" non-goal). The review's estimated magnitudes are never reported as measured.

## Current failure

The review's invariant: submission overhead stays small relative to op cost, because the future decode loop issues many operations per token. PF-003 traced the per-copy bookkeeping stack on the pre-61 tree at ~14 small heap allocations and ~10–16 lock operations on the caller thread plus two thread handoffs per submit+wait, anchored to the measured 7.1 µs 16×16 round trip whose data movement is sub-microsecond. Tasks 61/62 removed the registry index over-build and CPU's registration; the *remaining* fixed cost per queued 16×16 non-no-op copy is:

- **CPU** (`src/cpu/device.cpp` `CpuQueue`): `Task` (`:735-752`) value-copies two `TensorView`s — each copy heap-allocates its `TensorSpec::shape.dimensions_` vector and `plane_strides_` vector (`include/iom/tensor.hpp:186-189`) — 4 allocations; `StagedWorker::submit_copy` (`include/iom/iom.hpp:66-113`) allocates one `std::list` node; the caller thread takes `completion_mutex_` once and the worker mutex twice, the worker thread takes the worker mutex and `completion_mutex_`, and the waiter takes `completion_mutex_` — with two condvar handoffs (submit→worker, worker→waiter) even though `execute` already ran `copy_elements` to completion on the submitting thread before publication.
- **TTNN** (`src/ttnn/device.cpp` `TtnnQueue`): same 4 view-vector allocations + list node (Task at `:373-390`), plus 2 registry nodes and the `outcomes_` map node; `make_fence` (`:346-359`) builds a capturing `std::function` that `register_copy_entries` copies twice per submission.
- **CUDA/ROCm** (`src/cuda/copy.cu` / `src/rocm/copy.hip`): `make_fence` (`copy.cu:208-215`, `copy.hip:214-221`) captures `{context, event, retained_failure}` (~24 bytes — beyond libstdc++'s 16-byte small-buffer) so `register_copy_entries` performs two heap-allocating `std::function` constructions per submission (`copy.cu:814-821`, `copy.hip:815-822`).

A first-draft primitive `{invoke, void* context}` design was rejected because every reachable fence copy shared ownership of a resource whose lifetime was governed by the worker thread (or, alternatively, by a destructor): a concurrent `snapshot_for` followed by the worker's `fence_destroy` would either hand the destructor a fence pointing at freed storage or require a `StagedWorker::process` reorder — both violate the established invariant that `complete()` may wake waiters while task-owned state is still live. A second-draft design that inlined `{context, event, retained_failure}` by value into the Fence and kept the raw event handle was rejected because the raw event copied into the registry fence did **not** own/outlive `fence_destroy`: copying the registry fence copies the raw event handle; the worker later destroys the underlying event via the task's resource, leaving every registry/snapshot copy holding a stale handle. A third-draft design extended the per-task resource with an intrusive refcount, but the worker's `cuda_fence_destroy` step synchronized + released the metadata slot + dropped the Task refcount without recording a cached result; a concurrent destructor snapshot that invoked the fence would find the resource refcount still non-zero (other leases alive) but the `worker_cleared` flag set and the metadata slot already released — the snapshot would then take a refcount-to-1 branch that ran the destroy-event-and-free path on an event the worker had not synchronized a second time, or worse, the snapshot saw `worker_cleared` and the resource refcount drop to 0 from the Task decrement alone (because the snapshot's own refcount had been moved out by the worker's path) and the event/resource leaked. The design below eliminates the per-copy `std::function` heap allocations, gives every copy independent lifetime via intrusive refcounting, **caches the fence outcome per resource** (so every `invoke` after the first returns the cached result, exactly like SYCL's `SyclFenceState::result()`), and leaves `process` alone.

The impact is per-op latency that every future compute op inherits, multiplied across the 0002 op inventory. The review's "plausibly 3–6 µs of the 7.1 µs" attribution is an estimate — this task's evidence must be measured counts and medians, not that estimate.

## Scope

- **Fence representation (shared):** in `include/iom/detail/outstanding_work_registry.hpp` (the post-62 location), replace `using Fence = std::function<FenceResult()>` with a fixed-capacity inline type-erased value built around the pinned constants `kFenceStorageBytes = 32` and `kFenceStorageAlign = alignof(std::max_align_t)`. Each `Entry`'s fence, each `snapshot_for` copy's fence, and the `failed_invalidated_fence` value all store their own independent capture inside the inline buffer; copy/move/destroy are routed through four explicit function pointers invoked by `Fence`'s five special members (`Fence()`, `~Fence()`, `Fence(const Fence&)`, `Fence(Fence&&)`, `Fence& operator=(const Fence&)`, `Fence& operator=(Fence&&)`), so each captured resource is constructed and destroyed the correct number of times. `Entry`'s fence member, `register_entry` validation, `snapshot_for`, release/remove paths, 61's two-index shape, 57's rollback, and the body of 62's `release_or_quarantine` helper keep their existing code — the helper's `!snapshot.fence` and `snapshot.fence()` expressions compile unchanged against the new type.
- **Worker ordering untouched:** `include/iom/iom.hpp` `StagedWorker::process` (`:136-155`), `drain_list` (`:157-163`), `shutdown_and_drain` (`:115-133`), `submit_copy`, `~StagedWorker`, and the `Callbacks` shape are unchanged. Today's `fence_complete → fence_destroy → callbacks_.complete` ordering is preserved verbatim.
- **CUDA/ROCm per-task resource becomes intrusive-refcounted with cached fence outcome:** the existing `CudaFenceResource` (`src/cuda/copy.cu:600-605`) and `HipFenceResource` (`src/rocm/copy.hip:604-609`) gain `std::atomic<std::size_t> refcount{1};`, `std::optional<FenceResult> cached_result;`, `std::mutex cached_result_mu;`, `std::exception_ptr retained_failure;`, and already hold the metadata slot/pool/context/event. The Task holds the raw pointer (one owner refcount at construction, decremented in `fence_destroy`/`~Task`). Registry inline captures become one-pointer RAII leases (`CudaFenceLease` / `HipFenceLease`): copy ctor `++refcount;`, move ctor transfers (source ptr becomes null), dtor `if (--refcount == 0) destroy;` where the destroy path is the only one that destroys the event and frees the resource — and only after the cached result has been populated. No per-copy allocation.
- **Backend fences (cuda, rocm, ttnn, and the post-55/60 sycl participant):** the inline capture is the lease (CUDA/HIP) or a small POD pointer/double-pointer wrapper (TTNN: `TtnnDevice*`; SYCL post-55: `std::shared_ptr<SyclFenceState>`); every CUDA/ROCm/SYCL `invoke` returns a cached, race-safe result — identical behavior to SYCL's pre-existing `SyclFenceState::result()`; the worker's `cuda_fence_destroy` synchronizes once, caches the result, releases the metadata slot exactly once, then drops the Task refcount; only the last outstanding lease destroys the (already-synchronized) event and frees the resource.
- **TTNN task views:** `TtnnQueue::Task` stores `const TensorView*` / `TensorView*` instead of two value-copied `TensorView`s.
- **CPU inline completion:** `CpuQueue` completes each submission on the submitting thread immediately after the synchronous work and no longer instantiates `detail::StagedWorker` at all.
- **Preserved contracts:** the public asynchronous contract (token-based `copy` → repeatable `wait`, retained asynchronous failures, sequence leasing/reclaim, `commit_failure`, the `process` ordering that lets `complete()` wake waiters safely) is untouched; `DeviceOps` (`include/iom/iom.hpp:212-368`, `src/iom.cpp:515-619`) is not modified. The per-copy event create/destroy count on CUDA/ROCm is **unchanged at 1/1** — the intrusive resource still creates one event per non-no-op copy and destroys it once; only the timing of the destruction moves to the last-lease point. The Device-outlives-allocations-and-queues precondition (which already covers every existing CUDA/ROCm allocation) is the only lifetime contract — **no new device bookkeeping** is added.
- **Backends in scope:** cpu, cuda, rocm, ttnn (the finding's scope); the SYCL queue delivered by 55/60 consumes the shared `Fence` type and gets the mechanical representation adaptation.

## Implementation references

- **Modify:** `include/iom/detail/outstanding_work_registry.hpp` — replace `using Fence = std::function<FenceResult()>` with the fixed-capacity inline type-erased layout pinned at `kFenceStorageBytes = 32` and `kFenceStorageAlign = alignof(std::max_align_t)` (Requirements 1–4); the common captureless `failed_invalidated_fence_invoke` / `make_invalidated_fence` (Requirement 5); `OutstandingWorkRegistry::register_entry` validation, `invalidate_entry_locked`, `register_registry_entries` (post-62 location) adapt to the new type. Do not touch `release_or_quarantine`/`SequenceOutcome`/`release_or_invalidate_entries` bodies.
- **Modify:** `src/cuda/copy.cu` — `CudaFenceResource` (`:600-605`) gains `std::atomic<std::size_t> refcount{1};`, `std::optional<FenceResult> cached_result;`, `std::mutex cached_result_mu;`, `std::exception_ptr retained_failure;`; the existing `CudaMetadataSlotPool* pool` / `std::size_t metadata_slot` / `CUcontext context` / `cudaEvent_t event` stay. The TU gains an inline `CudaFenceLease` RAII struct wrapping `CudaFenceResource*` with `acquire_lease(resource)` that bumps the refcount, copy ctor that bumps, move ctor that transfers, and a destructor that on the last refcount (`fetch_sub == 1`) runs the activate-then-destroy path. `cuda_fence_complete` (`:616-620`) and `cuda_fence_destroy` (`:622-632`) are refactored.
- **Modify:** `src/rocm/copy.hip` — exact mirror: `HipFenceResource` (`:604-609`) gains `std::atomic<std::size_t> refcount{1};`, `std::optional<FenceResult> cached_result;`, `std::mutex cached_result_mu;`, `std::exception_ptr retained_failure;`; `HipFenceLease` RAII struct; `hip_fence_complete` and `hip_fence_destroy` refactored to mirror CUDA.

- **Split of `destroy_cuda_resource_noexcept` / `destroy_hip_resource_noexcept`:** today's `src/cuda/copy.cu:607-614` `destroy_cuda_resource_noexcept(CudaFenceResource*)` (and the HIP twin at `:611-618`) does three things in sequence: `destroy_event_noexcept(event)` + `pool->release(metadata_slot)` + `delete resource`. Under this task that single helper is **deleted** — its three responsibilities split across two new sites and one existing one:
   1. **`pool->release(metadata_slot)`** moves into the worker callback `cuda_fence_destroy(void*)` (called once per task, before the Task refcount drop). The HIP twin mirrors.
   2. **`destroy_event_noexcept(event)` + `delete resource`** move into `~CudaFenceLease`'s last-lease branch (activate-then-destroy; called exactly once per task lifetime, on whichever thread drops the last refcount). The HIP twin mirrors in `~HipFenceLease`.
   3. The **pre-link `execute` failure branch** (the path that destroys a never-recorded event because no registry entries were committed and no `cuda_fence_destroy(void*)` will run) keeps its **direct `destroy_event_noexcept(event)` call** — this is the 56-ST-002 Requirement 4 preserved pre-link path. The HIP twin mirrors. The pre-link branch's `metadata_slot` was already released (the pool acquire/rollback path) and no resource `new`/`delete` is involved (the resource is only allocated once `acquire_lease` runs after a successful pre-link; if acquire fails, there is no resource to delete).
   The split is observable in source: `grep -n "destroy_cuda_resource_noexcept\|destroy_hip_resource_noexcept" src/cuda/copy.cu src/rocm/copy.hip` returns no match in the post-73 tree. The pre-link direct `destroy_event_noexcept` call remains visible in the pre-link failure branches of `execute` and the HIP twin.
- **Modify:** `src/ttnn/device.cpp` — `make_fence` (`:346-359`) becomes a builder that fills the inline fence via the TTNN `copy_construct`/`move_construct`/`destroy` triple, capturing `TtnnDevice*`. Registration (`:488-494`) populates the inline fence and passes it by `const&`. TTNN's `complete_task` keeps calling `finish_native` directly.
- **Modify:** `src/sycl/copy.cpp` — the post-55/60 `make_fence` populates the inline fence via the SYCL `copy_construct`/`move_construct`/`destroy` triple with `std::shared_ptr<SyclFenceState>` only (55-ST-001 requirements 15 and 16). `SyclFenceState::result()` is the cached idempotent fence outcome; the fence's `invoke` trampoline calls `state->result()` and returns it. No separate `sycl::event` or `exception_ptr` is captured into the fence — both live inside `SyclFenceState`, behind the state's mutex, accessed through `result()` (55-ST-001 requirements 15 and 16). The shared_ptr is constructed once in `execute` step (b) of 55-ST-001, captured into the registry `Fence` in step (c), and released when both registry entries are gone and the worker has run `fence_destroy`. `<memory>` is the only new include the SYCL backend adds.
- **Modify:** `src/cpu/device.cpp` — `CpuQueue` (`:734-962`): `copy` (`:791-802`) executes inline and completes inline; delete `Task` (`:735-752`), `execute` (`:831-863`), `complete_task` (`:865-890`), the `worker_` member and its `Callbacks` construction (`:762-782`), and `~CpuQueue`'s `shutdown_and_drain` (`:784-788`). Post-61/62 positions shift; apply against the delivered post-62 shape.
- **Read:** `include/iom/iom.hpp:66-113` — `StagedWorker::submit_copy`: `callbacks_.execute(*staged)` runs on the submitting thread before publication; `process` (`:136-155`) is the only fence callback path, unchanged by this task. `CompleteOnThrow` remains TTNN/SYCL's publication policy.
- **Read:** `include/iom/iom.hpp:136-155` — `StagedWorker::process` is the only place the three backend callbacks run; the order is `fence_complete → fence_destroy → callbacks_.complete`. `complete()` (`src/iom.cpp:560-582`) is non-noexcept, mutates `failures_`/`pending_failures_`/`completed_` under `completion_mutex_`, and `notify_all`s the completion CV; waiters can observe `completed_ ≥ sequence` after `complete` returns and before any subsequent cleanup, so this task must not reorder the steps.
- **Read:** `include/iom/iom.hpp:295-329` — `DeviceOps::submit`'s documented inline-completion seam.
- **Read:** `docs/changes/0001-tensor-view/62-AR-002-share-release-quarantine-protocol/spec.md` — the fence loop contract the new type must keep compiling against.
- **Read:** `docs/changes/0001-tensor-view/56-ST-002-restore-gpu-event-fencing/spec.md` — the no-throw synchronize-then-destroy primitive that `cuda_fence_destroy`/`hip_fence_destroy` restore. **Supersession note (record so the 56 audit is not read as still binding):** 56-ST-002's source-audit acceptance criterion "`destroy_event_noexcept` is reached only from the two `destroy_*_resource_noexcept` bodies and the two pre-link `execute` failure branches" is **superseded** by this task. Post-73, `destroy_event_noexcept` is reached from three new sites: (a) the last-lease destructor `~CudaFenceLease` / `~HipFenceLease` activate-then-destroy branch (the new home of the per-task event destroy); (b) the worker callback's defensive last-ref branch inside `cuda_fence_destroy(void*)` / `hip_fence_destroy(void*)` (Task-was-the-only-lease edge case); (c) the retained pre-link `execute` failure branches (unchanged from 56). The two `destroy_*_resource_noexcept` helpers are deleted — see the Implementation References "Split of `destroy_cuda_resource_noexcept` / `destroy_hip_resource_noexcept`" block above for the full mapping. The 56 invariant (every event synchronized exactly once before exactly-once destroy, on all four caller classes — worker success path, worker shutdown drain, staged-list drain, execute post-link rollback) is preserved verbatim; only its mechanism relocates. PF-004 must read its source-audit criterion against the post-73 destroy sites, not the pre-73 ones.
- **Read:** `docs/changes/0001-tensor-view/74-PF-004-pool-gpu-copy-events/spec.md` (when present) — PF-004 owns pooling events; the `CudaFenceResource`/`HipFenceResource` refcount + cached-result protocol here is the seam PF-004 plugs a pooled slot into.
- **Read:** `docs/changes/0001-tensor-view/55-ST-001-harden-sycl-queue-lifetimes/spec.md` — the SYCL fence-owner shape (`std::shared_ptr<SyclFenceState>`, `SyclFenceState::result()` idempotent cached outcome, requirements 15–16) that this task consumes verbatim.
- **Read:** `docs/changes/0001-tensor-view/48-ST-002-own-cpu-ttnn-task-views/spec.md` — the temporary-view hazard and its regression tests that must keep passing.
- **Tests:** `test/test_iom.cpp:2097-2135` — the `OutstandingWorkRegistry` case constructs `detail::Fence` from a capturing lambda (`:2101-2104`) and invokes `invalidated[0].fence()` (`:2128`); update constructions to the inline type-erased form. The post-57 fault sweep cases adapt the same way.
- **Tests:** `test/backend/backend_conformance_copy_storage.hpp` — the shared async-copy conformance is the behavioral safety net; unchanged.
- **Tests:** `test/cpu/test_cpu_bench.cpp` — **NOT in PF-003's touched-files list** (see Requirement 13 / "Bench re-calibration" non-goal). The 16×16 bench medians are recorded by PF-003 only as evidence in the change log; the gate-form update and `kAllowanceSeconds` re-derivation are owned by a separate explicit follow-up task.

## Requirements

1. **Pinned capacity, alignment, and nothrow contracts.** In `include/iom/detail/outstanding_work_registry.hpp`:
   ```cpp
   namespace iom::detail {
   inline constexpr std::size_t kFenceStorageBytes  = 32;
   inline constexpr std::size_t kFenceStorageAlign  = alignof(std::max_align_t);
   }
   ```
   The buffer fits every backend's `FenceCapture` declared below. `kFenceStorageAlign` is the C++ standard's "largest scalar alignment" and is the correct floor for any of these captures. `kFenceStorageBytes = 32` is the chosen value covering every backend's `FenceCapture`. **Each backend TU's first declaration of `FenceCapture` immediately precedes `static_assert(sizeof(Capture) <= kFenceStorageBytes);` and `static_assert(alignof(Capture) <= kFenceStorageAlign);`** — a build failure at either assert is the explicit signal to grow the buffer before the backend can ship.
2. **`Fence` layout, all five special members, `is_nothrow_*` traits.** In `include/iom/detail/outstanding_work_registry.hpp`:
   ```cpp
   namespace iom::detail {
   struct Fence {
       alignas(kFenceStorageAlign) unsigned char storage[kFenceStorageBytes]{};
       FenceResult (*invoke)(const Fence&) noexcept = nullptr;
       void (*copy_construct)(Fence* dst, const Fence& src) noexcept = nullptr;
       void (*move_construct)(Fence* dst, Fence* src) noexcept = nullptr;
       void (*destroy)(Fence*) noexcept = nullptr;

       Fence() noexcept = default;
       ~Fence() noexcept;
       Fence(const Fence& other) noexcept;
       Fence(Fence&& other) noexcept;
       Fence& operator=(const Fence& other) noexcept;
       Fence& operator=(Fence&& other) noexcept;

       FenceResult operator()() const { return invoke(*this); }
       explicit operator bool() const noexcept { return invoke != nullptr; }
   };

   static_assert(std::is_nothrow_default_constructible_v<Fence>);
   static_assert(std::is_nothrow_copy_constructible_v<Fence>);
   static_assert(std::is_nothrow_move_constructible_v<Fence>);
   static_assert(std::is_nothrow_destructible_v<Fence>);
   static_assert(std::is_nothrow_copy_assignable_v<Fence>);
   static_assert(std::is_nothrow_move_assignable_v<Fence>);
   }
   ```
   Every backend's `copy_construct` / `move_construct` / `destroy` / `invoke` is declared `noexcept`; each backend TU adds the matching `static_assert(noexcept(<Backend>_fence_copy_construct));` (and the three siblings) immediately after the helper declaration. A build failure at any of these asserts is the explicit signal to fix the helper or grow the buffer.
3. **All five special members, placement-new + source-destroy invariant for nontrivial moves.** The bodies below are normative; `src` may alias `*dst` (self-assignment-safe assignments handle this explicitly). The storage buffer is **uninitialized raw bytes** in copy/move-construct paths; the backend helper is responsible for placement-newing the destination, never assigning into a default-constructed sub-object.
   For **nontrivial moves**, the rule is: the backend's `move_construct` helper must, after placement-newing the destination, **destroy the source's live capture** before returning (so the source's move-out is observable only as `destroy_at(src_capture)` — and the source's destroy pointer must remain non-null so the moved-from `Fence::~Fence` cleans up correctly via the standard route). In other words, `cuda_fence_move_construct` placement-news the destination lease from `std::move(*src)`, then calls `std::destroy_at(reinterpret_cast<CudaFenceLease*>(src->storage));` (or its semantic equivalent) so the source's refcount is decremented exactly once. The SYCL `shared_ptr` capture follows the same pattern via `shared_ptr`'s move (which zeros the source); the TTNN trivial `memcpy` path is exempt (no live capture). After every Fence-level move, **all four of the source's function pointers are nulled** so the moved-from fence is empty:
   - `Fence::Fence() noexcept = default` — leaves `invoke = nullptr`, `copy_construct = nullptr`, `move_construct = nullptr`, `destroy = nullptr`, `storage` zero-initialized. An empty fence.
   - `Fence::~Fence() noexcept` — if `destroy != nullptr`, call `destroy(this)`. If `destroy == nullptr`, no-op.
   - `Fence::Fence(const Fence& other) noexcept` — if `other.copy_construct != nullptr`, call `other.copy_construct(this, other)` (placement-new the capture into `this->storage`); else `std::memcpy(this->storage, other.storage, kFenceStorageBytes)`. Then copy the four function pointers verbatim from `other`. (Source is unchanged.)
   - `Fence::Fence(Fence&& other) noexcept` — if `other.move_construct != nullptr`, call `other.move_construct(this, &other)` (placement-new dst, then destroy source capture per the rule above); else `std::memcpy(this->storage, other.storage, kFenceStorageBytes)`. Then copy the four function pointers verbatim from `other`, and null all four of `other`'s function pointers (`other.invoke = other.copy_construct = other.move_construct = other.destroy = nullptr;`). The moved-from fence is empty and `~Fence` on the source is a no-op.
   - `Fence& Fence::operator=(const Fence& other) noexcept` — if `this == &other`, return `*this`. Otherwise: if `this->destroy != nullptr`, call `this->destroy(this)`; if `other.copy_construct != nullptr`, call `other.copy_construct(this, other)`; else `std::memcpy(this->storage, other.storage, kFenceStorageBytes)`. Copy the four function pointers verbatim from `other`. Return `*this`.
   - `Fence& Fence::operator=(Fence&& other) noexcept` — if `this == &other`, return `*this`. Otherwise: if `this->destroy != nullptr`, call `this->destroy(this)`; if `other.move_construct != nullptr`, call `other.move_construct(this, &other)` (placement-new dst, then destroy source capture); else `std::memcpy(this->storage, other.storage, kFenceStorageBytes)`. Copy the four function pointers verbatim from `other`, then null all four of `other`'s function pointers. Return `*this`.
   `register_entry`'s `if (!fence)` validation works via the explicit conversion to bool (function pointers null iff the fence is empty). `<functional>` stays included (`AllocatorCleanupAction::pre_release_` still uses `std::function`).
4. **Capture destruction uses `std::destroy_at`.** Every backend's `destroy` helper that needs to drop a non-trivial capture uses `std::destroy_at(reinterpret_cast<Capture*>(f->storage));` (or `std::launder`-then-`destroy_at`) rather than manual member-level destructors. The exception is the trivial memcpy fallback (TTNN's pointer-only capture) which has null `destroy` and the special members fall back to no-op. The CUDA/ROCm `destroy` for the `CudaFenceLease`/`HipFenceLease` one-pointer leases is `std::destroy_at(reinterpret_cast<Lease*>(f->storage));` which decrements the resource's refcount and, if zero, runs the resource's last-lease destruction (which runs only after the cached `FenceResult` is populated; the path is `gpu_policy::destroy_event_noexcept(event); delete resource;` and does **not** synchronize again, does **not** call the metadata pool, and does **not** touch the device context — all of those happened in `cuda_fence_destroy` exactly once).
5. **The captureless invalidated fence.** In `include/iom/detail/outstanding_work_registry.hpp` define `failed_invalidated_fence_invoke` first, then `make_invalidated_fence` second (the inline factory references the invoke function — no anonymous-namespace or TU-static function; both are `inline` in the header):
   ```cpp
   namespace iom::detail {

   inline FenceResult failed_invalidated_fence_invoke(const Fence&) noexcept {
       try {
           return FenceResult::failed(std::make_exception_ptr(
                   std::runtime_error(
                           "outstanding-work entry was invalidated")));
       } catch (...) {
           return FenceResult::failed(std::current_exception());
       }
   }

   inline Fence make_invalidated_fence() noexcept {
       Fence f;
       f.invoke = &failed_invalidated_fence_invoke;
       // copy_construct, move_construct, destroy stay null:
       // a copy of an invalidated fence must remain a valid empty-storage fence,
       // which the trivially-copyable path handles via std::memcpy.
       return f;
   }

   }  // namespace iom::detail
   ```
   `invalidate_entry_locked` assigns `entry.fence = make_invalidated_fence();`. The invalidated fence is **common, captureless, allocation-free**, with empty inline storage and null construct/destroy helpers; the pre-change `failed_invalidated_fence` is replaced by `failed_invalidated_fence_invoke` taking `const Fence&` and ignoring its argument. The same value is used across all backends.
6. **CUDA intrusive-refcounted fence resource + cached `FenceResult` + one-pointer RAII lease.** Replace the raw `void*` fence handle pattern with an intrusive-refcounted resource, a per-resource cached outcome, and a one-pointer RAII lease. Sketch (the field layout grows from today's `CudaFenceResource`):
   ```cpp
   // In src/cuda/copy.cu (or the post-63 shared header).
   struct CudaFenceResource {
       CUcontext context = nullptr;
       cudaEvent_t event = nullptr;
       std::size_t metadata_slot = 0;
       CudaMetadataSlotPool* pool = nullptr;
       std::exception_ptr retained_failure;
       std::atomic<std::size_t> refcount{1};
       // Cached fence outcome, populated exactly once by the first synchronizing
       // caller (the worker's fence_destroy step or a registry/snapshot invoke).
       // Subsequent invokes return this without touching the event.
       std::optional<iom::detail::FenceResult> cached_result;
       std::mutex cached_result_mu;
   };

   // One-pointer RAII lease the Fence storage holds.
   //
   // Lifetime invariant (refcount accounting, exact, in order):
   //   1. Resource created with `refcount == 1` (the Task owner's initial reference).
   //   2. `acquire_lease(r)` (the factory used by builders) **bumps** the refcount:
   //      a new lease from `acquire_lease` takes refcount 1 -> 2.
   //      Owners at this step: Task, builder lease.
   //   3. `register_copy_entries` calls `register_entry` for the source registry
   //      entry — copy ctor bumps 2 -> 3. Then for the destination registry
   //      entry — copy ctor bumps 3 -> 4. Owners at this step: Task, builder,
   //      source-entry lease, destination-entry lease (4 leases alive).
   //   4. The builder fence value (its local lease) is dropped at the end of
   //      `execute`'s scope — destructor decrements 4 -> 3. Owners at this
   //      step: Task, source-entry lease, destination-entry lease (3 leases
   //      alive; this is the steady-state between execute returning and the
   //      worker running).
   //   5. The worker calls `cuda_fence_destroy(void*)` which decrements the
   //      Task refcount — 3 -> 2. Owners at this step: source-entry lease,
   //      destination-entry lease (2 leases alive).
   //   6. `complete_task` calls `try_release_entry` for the source registry
   //      entry — its fence destructor decrements 2 -> 1. Then for the
   //      destination registry entry — its fence destructor decrements 1 -> 0.
   //      The last decrement runs the last-lease destruction path
   //      (activate -> `destroy_event_noexcept` -> `delete resource`).
   //
   // The expected count sequence at every observable step is recorded in
   // Requirement 14's race regression test; the registration paragraph (the
   // numbered list immediately above this sketch) repeats the same sequence.
   //
   // Assignment is `= delete`: the Fence type's special members destroy-then-
   // construct, never assign a live capture. Implementing `operator=` with the
   // same `fetch_sub == 1 -> destroy/delete` tail as the destructor is also
   // acceptable, but the chosen design is `= delete` because no backend helper
   // invokes `operator=` on a populated lease (copy ctor + move ctor + dtor
   // are the only paths that mutate refcount).
   struct CudaFenceLease {
       CudaFenceResource* resource = nullptr;
       CudaFenceLease() noexcept = default;
       // Bumps the resource's refcount by 1. Null resource is a no-op (bumps nothing).
       static CudaFenceLease acquire_lease(CudaFenceResource* r) noexcept {
           CudaFenceLease lease;
           lease.resource = r;
           if (r) r->refcount.fetch_add(1, std::memory_order_relaxed);
           return lease;
       }
       CudaFenceLease(const CudaFenceLease& other) noexcept
               : resource(other.resource) {
           if (resource) resource->refcount.fetch_add(1, std::memory_order_relaxed);
       }
       CudaFenceLease(CudaFenceLease&& other) noexcept
               : resource(other.resource) { other.resource = nullptr; }
       CudaFenceLease& operator=(const CudaFenceLease&) = delete;
       CudaFenceLease& operator=(CudaFenceLease&&) = delete;
       ~CudaFenceLease() noexcept {
           if (resource && resource->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
               // Last lease out: the cached_result must already be populated by
               // the worker's fence_destroy step or by an earlier registry/snapshot
               // invoke. Destroy the (already-synchronized) event and free the
               // resource. Activate-then-destroy mirrors 56-ST-002: the last lease
               // may drop on a tensor-destructor thread whose current context differs
               // from the resource's context, so we activate before destroying the
               // event. The activate is wrapped in try/catch because the last-lease
               // drop can run after queue destruction (the device is still alive
               // per the Device-outlives precondition, but context activation may
               // still race with other device work).
               try { gpu_policy::activate(resource->context); } catch (...) {}
               gpu_policy::destroy_event_noexcept(resource->event);
               delete resource;
           }
       }
   };
   static_assert(sizeof(CudaFenceLease) <= iom::detail::kFenceStorageBytes);
   static_assert(noexcept(CudaFenceLease(std::declval<const CudaFenceLease&>())));
   static_assert(!std::is_copy_assignable_v<CudaFenceLease>);
   static_assert(!std::is_move_assignable_v<CudaFenceLease>);
   static_assert(std::is_nothrow_destructible_v<CudaFenceLease>);

   // The synchronize-once-cache body. noexcept in fact (see the noexcept guard
   // below). Called by cuda_fence_destroy (the worker's void* callback) AND by
   // cuda_fence_invoke (the registry trampoline) — whichever runs first populates
   // cached_result under the cached_result_mu; the other returns the cached value
   // without touching the event.
   //
   // Noexcept contract + lock-failure tradeoff (documented, accepted):
   // `cached_result_mu` is `std::mutex`. A `std::mutex::lock()` call from inside
   // a `noexcept` body that throws `std::system_error` would terminate the
   // process (`std::terminate`). PF-003 accepts this tradeoff explicitly — the
   // pattern mirrors task 55's `SyclFenceState` state mutex, which has the same
   // shape: a state mutex guarded by a noexcept body. Lock acquisition is a
   // wait-on-mutex operation; in practice it cannot throw on a healthy host
   // unless the implementation's mutex is poisoned (a separate failure mode
   // the engine does not recover from anyway). The noexcept contract is
   // preserved by relying on this runtime invariant; the tradeoff is documented
   // here, not silently relied on.
   //
   // Storage write/reset is wrapped in try/catch so that a throwing
   // FenceResult::failed(std::make_exception_ptr(...)) can never escape the Fence
   // invoke. A throwing std::optional<T>::operator=() or copy/move assignment
   // would violate the noexcept contract; the catch falls back to constructing a
   // fresh FenceResult::failed(std::current_exception()) (noexcept as long as
   // std::current_exception() does not throw, which it does not) and stores it
   // via std::optional::emplace (which is noexcept when T is nothrow-move-or-
   // copy-constructible and the held type does not throw on assignment). The
   // fence is fully populated by the time the function returns — no other path
   // writes cached_result, so the lock-and-check is the only writer.
   iom::detail::FenceResult cuda_synchronized_fence_event(CudaFenceResource& r) noexcept {
       std::lock_guard<std::mutex> lock(r.cached_result_mu);
       if (r.cached_result.has_value()) {
           return *r.cached_result;
       }
       try {
           gpu_policy::activate(r.context);
           gpu_policy::synchronize_event(r.event);
           if (r.retained_failure) {
               r.cached_result.emplace(iom::detail::FenceResult::failed(std::move(r.retained_failure)));
           } else {
               r.cached_result.emplace(iom::detail::FenceResult::success());
           }
           return *r.cached_result;
       } catch (...) {
           // Storage-write fallback: never let an exception escape the Fence invoke.
           // emplace is preferred over operator= because it can replace the held
           // value without invoking the prior value's assignment; it is noexcept
           // for nothrow-move-or-copy-constructible FenceResult (verified at
           // compile time below).
           r.cached_result.emplace(iom::detail::FenceResult::failed(std::current_exception()));
           return *r.cached_result;
       }
   }
   static_assert(noexcept(cuda_synchronized_fence_event(std::declval<CudaFenceResource&>())));

   // StagedWorker callback — exact name/signature the existing code and tests
   // already rely on (see src/cuda/copy.cu:622-632 cuda_fence_destroy and the
   // ReusingCudaAllocator regression). Casts the Task's raw resource alias,
   // runs the once-only cached synchronize path, releases the metadata slot
   // once, drops the Task refcount, and clears the Task alias. The Fence helper
   // above has a different signature and a different name (cuda_fence_storage_destroy).
   void cuda_fence_destroy(void* opaque) noexcept {
       if (opaque == nullptr) return;
       auto* resource = static_cast<CudaFenceResource*>(opaque);
       // 1. Once-only cached synchronize (populates cached_result if not yet set;
       //    returns the cached value either way; never touches the event twice).
       (void)cuda_synchronized_fence_event(*resource);
       // 2. Release the metadata slot exactly once.
       if (resource->pool != nullptr) {
           resource->pool->release(resource->metadata_slot);
       }
       // 3. Drop the Task refcount. The resource outlives this call (other
       //    registry/snapshot leases still hold it); the last lease to drop
       //    its refcount runs the destroy-event-and-free path.
       if (resource->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
           // Defensive: if the Task was the last outstanding lease (no registry
           // entries, no snapshot), run the destruction here under the same
           // shape the CudaFenceLease destructor would run. Mirrors the lease
           // last-lease path: activate-then-destroy mirrors 56-ST-002.
           try { gpu_policy::activate(resource->context); } catch (...) {}
           gpu_policy::destroy_event_noexcept(resource->event);
           delete resource;
       }
       // 4. The Task clears its alias (task.fence = nullptr; task.event = nullptr)
       //    immediately after cuda_fence_destroy returns, so the next destructor
       //    inspection of the Task sees an empty fence and cannot re-enter this
       //    path. This is a property of the existing CUDA task shape (today's
       //    task.event/task.fence are assigned in execute and may be cleared by
       //    the worker; see copy.cu:743-747 for the no-op path).
   }

   void cuda_fence_copy_construct(iom::detail::Fence* dst, const iom::detail::Fence& src) noexcept {
       // dst->storage is uninitialized raw bytes; placement-new the lease.
       ::new (dst->storage) CudaFenceLease{
               *std::launder(reinterpret_cast<const CudaFenceLease*>(src.storage))};
   }
   void cuda_fence_move_construct(iom::detail::Fence* dst, iom::detail::Fence* src) noexcept {
       // Placement-new dst from std::move(src), then destroy the source's live
       // capture so the source's refcount is decremented exactly once. The Fence
       // move ctor will then null all four of src's function pointers.
       ::new (dst->storage) CudaFenceLease{
               std::move(*std::launder(reinterpret_cast<CudaFenceLease*>(src->storage)))};
       std::destroy_at(std::launder(reinterpret_cast<CudaFenceLease*>(src->storage)));
   }
   void cuda_fence_storage_destroy(iom::detail::Fence* f) noexcept {
       std::destroy_at(std::launder(reinterpret_cast<CudaFenceLease*>(f->storage)));
   }
   iom::detail::FenceResult cuda_fence_invoke(const iom::detail::Fence& f) noexcept {
       const auto& lease = *std::launder(reinterpret_cast<const CudaFenceLease*>(f.storage));
       if (lease.resource == nullptr) {
           return iom::detail::FenceResult::success();
       }
       return cuda_synchronized_fence_event(*lease.resource);
   }
   ```

   Registration (`copy.cu:814-821`) creates `CudaFenceResource` with **refcount 1** (Task's initial reference) **before any registry insertion**. The builder fence value is then constructed via `CudaFenceLease::acquire_lease(resource)`, which **bumps the refcount** to 2 — without this bump the explicit ctor would steal the Task's only reference. From this point the expected refcount sequence is:

   1. **`refcount = 1`** — after resource creation (Task only).
   2. **`refcount = 2`** — after `acquire_lease(resource)` builds the builder fence (Task + builder).
   3. **`refcount = 3`** — after `register_copy_entries` copies the fence into the source registry entry (Task + builder + source entry).
   4. **`refcount = 4`** — after `register_copy_entries` copies the fence into the destination registry entry (Task + builder + source entry + destination entry).
   5. **`refcount = 3`** — after the builder fence is dropped (Task + 2 entries).
   6. **`refcount = 2`** — after `cuda_fence_destroy(void*)` runs in `StagedWorker::process` and decrements the Task reference (2 entries).
   7. **`refcount = 1`** — after `complete_task` calls `try_release_entry` for the source entry.
   8. **`refcount = 0`** — after `complete_task` calls `try_release_entry` for the destination entry; the last release runs the last-lease destruction path (activate + `destroy_event_noexcept` + `delete resource`).

   Each step is observable by `std::atomic<std::size_t> refcount::load(std::memory_order_relaxed)` in the race regression test (Requirement 14). The Task itself stores the raw `CudaFenceResource*` pointer that the worker passes to `cuda_fence_destroy(void*)` in `StagedWorker::process`.

   **Pre-registration rollback** (registry insert throws after the resource is created but before any registry entry is committed): the rollback path runs the worker-style cleanup explicitly — call `cuda_synchronized_fence_event(*resource)` (idempotent if already populated), release the metadata slot once, then drop the Task refcount (the last-lease destroy path runs if the Task was the sole lease) — before rethrowing. The rollback does **not** depend on `~Task` or on `~StagedWorker` to run any cleanup: the raw pointer is not an owning handle from `StagedWorker`'s point of view (`StagedWorker::erase` destructs the `std::list<Task>` node but does not know that `task.fence` aliases a `CudaFenceResource*`); the rollback must explicitly call the worker-style cleanup. After rollback, `task.fence` is set to `nullptr` so subsequent Task destruction is a no-op for the resource.

   **Post-registration rollback** (an entry was committed and later rolled back via `try_release_entry`/`remove_entries`): the existing rollback path runs (entries removed under registry lock; subsequent `try_release_entry` decrements refcount through the entry's release). The Task's `cuda_fence_destroy(void*)` callback still runs in `StagedWorker::process` and runs the cached-synchronize path and the metadata-slot release — by the time the callback runs, the registry entries have already been removed, so the Task's refcount drop is the lease drop that takes the refcount to 0 (assuming no snapshot copies), and the resource is destroyed cleanly. If a snapshot copy is alive, the last remaining lease (the snapshot) destroys the resource when it drops.

   **Success path**: the Task aliases the raw pointer; the worker runs `cuda_fence_destroy(void*)` in `process` exactly once, exactly as described; the Task clears its `task.fence`/`task.event` aliases after the callback returns.
7. **ROCm mirror.** Identical structure through `HipFenceResource`, `HipFenceLease`, `hip_fence_copy_construct`, `hip_fence_move_construct`, `hip_fence_storage_destroy(Fence*)`, `hip_fence_destroy(void*) noexcept` StagedWorker callback, `hip_fence_invoke`, and `hip_synchronized_fence_event`. The HIP lease `~HipFenceLease` last-lease branch and the worker's defensive last-lease branch inside `hip_fence_destroy(void*)` mirror the CUDA pattern: `try { gpu_policy::activate(resource->device); } catch (...) {}` then `gpu_policy::destroy_event_noexcept(resource->event); delete resource;` — the activate-then-destroy sequence mirrors 56-ST-002 because the last lease can land on a thread whose current device context differs. The HIP `acquire_lease` factory and the exact refcount sequence (1 → 2 → 3 → 4 → 3 → 2 → 1 → 0) match the CUDA spec verbatim, with `device_` (an `int` device ordinal) substituted for `context_` and `hipEvent_t` for `cudaEvent_t`. `HipFenceLease::operator=(const HipFenceLease&) = delete; HipFenceLease::operator=(HipFenceLease&&) = delete;` with the matching `static_assert`s. Registration at `copy.hip:815-822` constructs one `detail::Fence` carrying a `HipFenceLease`. **No new device bookkeeping**: the public Device-outlives-allocations-and-queues precondition already covers every lease's lifetime (the resource is owned by the device's metadata pool and the device outlives every queue and tensor on it).
8. **TTNN pointer-only capture.** `struct TtnnFenceCapture { TtnnDevice* device; };` is trivially-copyable and trivially-destructible; `copy_construct`, `move_construct`, `destroy` stay `nullptr` and the special members fall back to `std::memcpy` (copy/move) and no-op (destroy). The `invoke` trampoline (`ttnn_fence_invoke(const Fence&) noexcept`) locks `static_cast<TtnnDevice*>(capture.device)->api_mutex()`, calls `capture.device->mesh().mesh_command_queue(0).finish()`, and returns `FenceResult::success()` or `FenceResult::failed(std::current_exception())`. The device pointer lifetime is governed by the registry (entries invalidated at queue teardown; destructors run against the device's own machinery while the device is alive). Registration (`src/ttnn/device.cpp:488-494`) populates the inline fence and passes it by `const&`.
9. **SYCL `shared_ptr<SyclFenceState>` capture.** `struct SyclFenceCapture { std::shared_ptr<SyclFenceState> state; };`. `static_assert(sizeof(SyclFenceCapture) <= kFenceStorageBytes);` and `static_assert(alignof(SyclFenceCapture) <= kFenceStorageAlign);` — `std::shared_ptr`'s size is implementation-defined (commonly two pointers; the buffer may need to grow if the implementer observes a sizeof exceeding 32 on the target standard library — that's the explicit `static_assert` gate). `copy_construct` (`sycl_fence_copy_construct`) placement-new-constructs the destination's `shared_ptr` from the source's (refcount bump, no heap). `move_construct` placement-news from `std::move(*src)` (the source `shared_ptr` is left empty by move-construct's contract); `cuda_fence_move_construct`-equivalent — `sycl_fence_move_construct` does **not** call `std::destroy_at` on the source because `std::shared_ptr`'s move constructor already zeroes the source — but `Fence`'s move ctor still nulls all four of the source's function pointers after the helper returns. `destroy` (`sycl_fence_destroy`) calls `std::destroy_at(reinterpret_cast<SyclFenceCapture*>(f->storage));`. `invoke` (`sycl_fence_invoke(const Fence&) noexcept`) calls `capture.state->result()` and returns the cached `FenceResult`. **No separate `sycl::event` or `exception_ptr` is captured into the fence** — both live inside `SyclFenceState`, behind the state's mutex, accessed through `result()` (55-ST-001 requirements 15 and 16). The shared_ptr is constructed once in `execute` step (b) of 55-ST-001, captured into the registry `Fence` in step (c), and released when both registry entries are gone and the worker has run `fence_destroy`. `<memory>` is the only new include the SYCL backend adds.
10. **Construction/destruction accounting.** For every backend, the inline type-erased fence performs one construction in `make_fence` (or its replacement), one `copy_construct` per registered entry (two per submission), one `copy_construct` per `snapshot_for` entry stored into the result vector, one `destroy` whenever an entry is removed (`try_release_entry`, `remove_entries`, `remove_entry_if_present`), `erase_entry_locked`, or invalidated (`invalidate_entry_locked`), and one `destroy` at `~Fence` for transient stack/return values. `move_construct` is used when the implementation moves a fence value (vector reallocation during `snapshot_for`); under the current registry implementation, `snapshot_for` populates its result vector by push_back/copy — one `copy_construct` per snapshot element. CUDA/ROCm additionally: the lease's destructor decrements an intrusive refcount; the last lease to drop the refcount destroys the (already-synchronized) event and frees the resource — and the cached result is the only thing every invoke observes. The static count assertion in Requirement 14 includes the per-submission construction/destruction accounting and the per-lease atomic refcount activity (no heap allocations; the atomic fetch_add/fetch_sub are lock-free).
11. **TTNN pointer-view task.** `TtnnQueue::Task` replaces `TensorView source; TensorView destination;` with `const TensorView* source; TensorView* destination;` (CUDA/ROCm shape, `src/cuda/copy.cu:639-648`), constructor takes the parameter references and stores their addresses, and `execute`/`copy_planes`/registration read through the pointers. Document at the type: the views are dereferenced only inside `execute`, which `StagedWorker::submit_copy` invokes on the submitting thread before publication — i.e., inside the caller's `copy(...)` frame, where the arguments (including 48's temporary derived views) are alive; the publication, completion, and fence paths never dereference them. This supersedes 48-ST-002's mechanism (value copies) and its "no `TensorView*` in any `Task` field" source-text acceptance; 48's observable regression tests are the surviving contract and must keep passing.
12. **CPU inline completion.** `CpuQueue::copy` becomes:
    ```cpp
    return submit(
            [this, &source, &destination, no_op](
                    std::uint64_t sequence) {
                if (!no_op) {
                    copy_elements(source, destination);
                }
                complete(sequence, nullptr);
            });
    ```
    Delete `CpuQueue::Task`, `execute`, `complete_task`, the `worker_` member and its `Callbacks`/`PublishPolicy::CompleteOnThrow` construction, `worker_.start()`, and `worker_.shutdown_and_drain()`; `~CpuQueue` is defaulted. `submission_order_mutex_` stays and still covers the whole submission-plus-execution. `validate_copy`, `identical_window` no-op classification, and `copy_elements` are unchanged.
13. **Bench re-calibration is NOT in this task.** PF-003 removes the CPU `StagedWorker` path that 67-NT-001's `kAllowanceSeconds` was calibrated against; the `copy_median` vs `no_op_median` differential that drives `kAllowanceSeconds` changes shape, and a separate explicit follow-up task (a "67-b" or a later NT) must re-run 67's calibration on the post-PF-003 tree and update the test's `kAllowanceSeconds` and its provenance comment. PF-003 does **not** edit `test/cpu/test_cpu_bench.cpp`; PF-003 records the bench medians before and after in the change evidence (`MESSAGE`-reported copy median and no-op median), and acceptance only requires that the recorded post-PF-003 medians are observed to be not slower than the pre-PF-003 medians within run-to-run jitter. The gate re-calibration is the follow-up task's responsibility. The clean dependency direction is: `67` establishes the gate on the pre-PF-003 CPU path; PF-003 removes that path; the follow-up recalibrates against the new path. (Reverting the blocked-by direction — i.e. making PF-003 a blocker of `67` — would also be clean; the chosen direction here is "PF-003 acceptance does not modify or rely on the gate value, only records medians.")
14. **Race regression coverage.** Add an `iom_tests` regression test under `test/test_iom.cpp` named `*"snapshot copy remains invocable after worker drops its Task refcount"*` (or the closest matching doctest case name) that walks the exact refcount sequence documented in Requirement 6's registration paragraph and asserts every observable step:
    1. construct a `CudaFenceResource` with `refcount == 1` (the Task's initial reference); assert `refcount.load() == 1`.
    2. call `CudaFenceLease::acquire_lease(resource)` to build the builder fence value; assert `refcount.load() == 2` (Task + builder).
    3. copy-construct the builder into a fake registry `Entry` (the source registry entry); assert `refcount.load() == 3` (Task + builder + source entry).
    4. copy-construct the registry entry into the destination registry `Entry`; assert `refcount.load() == 4` (Task + builder + 2 entries).
    5. drop the builder fence value (let it go out of scope); assert `refcount.load() == 3` (Task + 2 entries).
    6. invoke the worker callback path explicitly: call `cuda_synchronized_fence_event(*resource)` (populates the cached result) then release the metadata slot and decrement the Task refcount via the same body as `cuda_fence_destroy(void*)`; assert `refcount.load() == 2` (the 2 entries).
    7. drop the source registry entry's fence value (let it go out of scope); assert `refcount.load() == 1`. At this point the cached `FenceResult` is populated; the event is **not** yet destroyed.
    8. instantiate a snapshot copy of the registry entry (`snapshot_for`'s copy_construct path); assert `refcount.load() == 2` (1 entry + 1 snapshot).
    9. drop the registry entry (the last remaining entry lease); assert `refcount.load() == 1` (snapshot only). The event is still alive (the snapshot keeps the resource pinned).
    10. invoke the snapshot copy's fence (`cuda_fence_invoke`); assert the cached `FenceResult` is returned and the cached result was populated before this invoke (the worker step populated it; the snapshot sees the same value).
    11. drop the snapshot lease; assert `refcount.load() == 0`, the event is destroyed **exactly once** at this last-lease point, and the cached result was populated before destruction.
    12. additionally assert that `static_assert(!std::is_copy_assignable_v<CudaFenceLease>)` and `static_assert(!std::is_move_assignable_v<CudaFenceLease>)` hold at compile time; the lease cannot be assigned.
    Add a second test that holds the last outstanding lease and asserts the destroy path runs only once (instrumented `std::atomic<bool> destroy_called` on the resource). Add a third test that asserts the `cuda_synchronized_fence_event` body populates the cache exactly once even when invoked concurrently from two threads (`std::thread` x 2 calling `cuda_synchronized_fence_event(*resource)` simultaneously; both calls return the same `FenceResult`; `std::atomic<std::size_t> synchronize_count` instruments `gpu_policy::synchronize_event` and asserts `synchronize_count.load() == 1`). The test uses a minimal CUDA-driver stub if no hardware is present (the lease protocol is backend-neutral; the test exercises the intrusive-refcount + cached-result machinery independent of `cudaEvent_t` itself).
15. **Preserved semantics.** A CPU token returned from `copy` is already complete; `wait` on it takes the no-block completed-sequence path (`src/iom.cpp:528-546` predicate true immediately) and remains repeatable. An exception from `copy_elements` propagates out of `copy` with no token and hits `DeviceOps::submit`'s reclaim rule unchanged; CPU retains no asynchronous failures. The base `DeviceOps` code is not edited. CUDA/ROCm post-enqueue failures still yield waitable tokens whose repeated `wait` rethrows the same retained failure; `commit_failure` and `fence_through_sequence` are unchanged. The destroy-before-wait guarantee for 49-ST-003/55-ST-001 is preserved: the destructor reads the fence's inline state (the lease pointer), the resource's cached result was populated by either the worker's step or an earlier `invoke`, and the lease's invoke returns the cached result without re-synchronizing; the last-lease destroy only runs the (no-synchronize) `destroy_event_noexcept` + `delete` path, by which point the event is already synchronized exactly once.

## Non-goals

- Registry redesign: index shape, `Entry` layout beyond the fence member's type, quarantine representation, header placement, CPU registration opt-out — 61-AR-003's delivered output, preserved as-is.
- The `release_or_quarantine` protocol, shared `RegistryState`, `SequenceOutcome` sharing, and per-queue `outcome_mutex_`/`outcomes_` maps — 62-AR-002 keeps them; eliminating the outcome map is not in PF-003's fix list.
- Reordering `StagedWorker::process` or any change to the `fence_complete`/`fence_destroy`/`complete` callback order — the established invariant that `complete()` wakes waiters on the completion CV requires that all task-owned state cleanup happens *before* `complete`, not the other way around. This task leaves `process` (`:136-155`), `drain_list` (`:157-163`), `shutdown_and_drain` (`:115-133`), `~StagedWorker`, and the `Callbacks` shape untouched.
- Changing `cuda_fence_complete`/`hip_fence_complete`/`sycl_fence_complete` semantics. The 56-ST-002 invariant is **preserved**: every native event is synchronized exactly once before it is destroyed exactly once, on all four caller classes — worker success path (worker `cuda_fence_destroy(void*)` synchronizes via `cuda_synchronized_fence_event` and the snapshot/registry `cuda_fence_invoke` returns the cached result without re-synchronizing), worker shutdown drain (`shutdown_and_drain` invokes the worker's `cuda_fence_destroy(void*)` per drained task), staged-list drain (the post-erase task list drains via the same callback), and execute post-link rollback (the rollback path runs `cuda_synchronized_fence_event` + `metadata_pool_->release` + the same refcount drop before rethrow). The 56-ST-002 **mechanism** relocates: the synchronize step moves into the cached `cuda_synchronized_fence_event` body (called exactly once per task, on whichever path runs first); the `destroy_event_noexcept` + `delete resource` step moves into `~CudaFenceLease`'s last-lease branch (activate-then-destroy, run on whichever thread drops the last refcount). The pool's `metadata_slot` release moves from the destroy_*_resource_noexcept helper into the worker callback (CUDA/ROCm twin at Requirement 7). **PF-004's scope is reduced**: 74 now replaces the intrusive per-task `CudaFenceResource`/`HipFenceResource` with a pooled `EventRing` lease slot that integrates with the intrusive refcount + cached-result protocol — PF-004 owns the pool, the slot acquisition/release, and the `retained_failure` propagation; this task owns the intrusive refcount, the cached-result protocol, the one-pointer RAII lease protocol, and the relocated 56-ST-002 mechanism. **PF-004 must preserve**: (a) the worker-callback names and signatures — `void cuda_fence_destroy(void*) noexcept`, `void cuda_fence_complete(void*)`, `void hip_fence_destroy(void*) noexcept`, `void hip_fence_complete(void*)`, `void sycl_fence_complete(void*)`, and the `void sycl_fence_destroy(void*) noexcept` 56-ST-002 synchronize-then-destroy primitive (`src/sycl/copy.cpp`'s 55-ST-001 surface); (b) the **cached once-only** protocol — exactly one event synchronize per task lifetime, exactly one metadata-slot release per task, exactly one refcount drop per Task; every `invoke` (registry, snapshot, or follow-up) goes through the cached result, never re-synchronizing the event. PF-004's `reusing-cuda-allocator`-style fault-injection regressions must keep passing with the new lease shape; PF-004's task description is updated to reflect the reduced scope and the preserved worker-callback surface.
- **Bench re-calibration** (`kAllowanceSeconds` re-derivation, the 20-run dataset, the `N/R/D/Q/diff_max/M/A_seconds/G` provenance comment in `test/cpu/test_cpu_bench.cpp`) — a separate explicit follow-up task. PF-003 records the medians only; it does not touch the bench TU.
- `StagedWorker` mechanics for TTNN/SYCL (staging lock windows, publication policy, worker loop) — 30-AR-002's scaffold is consumed, not redesigned; CPU leaving it does not change it.
- SYCL kernel, staging, and launch-count performance and its benchmark/launch-count closure — tasks 60 and PF-001 (71); TTNN per-plane download serialization — PF-005 (75); CPU/TTNN download pre-zero — PF-002 (72).
- The CPU/TTNN bench gate form and throughput floors — 67-NT-001 owns them; PF-003 records medians only.
- Dead-field cleanup on CUDA/ROCm tasks and pools (`fence_succeeded`, staging-pool members) — AR-006 (65).
- Any public API change: `DeviceOps`, `TensorView`, `Tensor`, `Device`, allocator surfaces are untouched.
- Any new device bookkeeping (`~CudaDevice`/`~HipDevice` waiting on leases, registry-level "shutting down" flags, etc.) — the public Device-outlives-allocations-and-queues precondition already covers every lease's lifetime.

## Acceptance criteria

- [ ] Source audit: `grep -rn "std::function<FenceResult" include/iom src` returns nothing; `grep -rn "make_fence" src` returns nothing; `grep -n "StagedWorker\|submit_copy\|shutdown_and_drain" src/cpu/device.cpp` returns nothing; each backend's registration site builds one `detail::Fence` value and passes it to both entries via `register_copy_entries`; `Fence` holds an aligned 32-byte buffer plus four function pointers.
- [ ] Pinned constants: `grep -n "kFenceStorageBytes\|kFenceStorageAlign" include/iom/detail/outstanding_work_registry.hpp` returns exactly two definitions (`kFenceStorageBytes = 32` and `kFenceStorageAlign = alignof(std::max_align_t)`); no other file defines either constant.
- [ ] Static asserts: every backend TU that constructs a fence has `static_assert(sizeof(<Backend>FenceCapture) <= iom::detail::kFenceStorageBytes);` and `static_assert(alignof(<Backend>FenceCapture) <= iom::detail::kFenceStorageAlign);` immediately before the first factory call; `static_assert(noexcept(<Backend>_fence_copy_construct));` (and the three siblings) follow each helper declaration; the header itself has the six `std::is_nothrow_*_v<Fence>` static_asserts. There is **no** `static_assert` comparing `sizeof(CudaFenceResource)` against `kFenceStorageBytes` (the resource is not an inline capture — only the lease is).
- [ ] All five special members + move lifecycle invariant: `grep -n "~Fence()\|Fence(const Fence\|Fence(Fence&&\|Fence& operator=" include/iom/detail/outstanding_work_registry.hpp` returns all five special members; the move ctor and move assignment **null all four source function pointers after the move** (`other.invoke = other.copy_construct = other.move_construct = other.destroy = nullptr;` after the storage move); for nontrivial captures (CUDA/ROCm/SYCL), the backend's `move_construct` placement-news the destination and **destroys the source's live capture** (`std::destroy_at` on the source's lease/capture) before returning so the source's refcount is decremented exactly once; the trivial TTNN `memcpy` path falls through the no-op destroy and is exempt. Assignments are self-safe (`this == &other` early return). `std::destroy_at` is the destroy idiom for nontrivial captures.
- [ ] Captureless invalidated fence: `grep -n "failed_invalidated_fence" include/iom src` shows `failed_invalidated_fence_invoke` defined **before** `make_invalidated_fence`; both are `inline` in the header; `make_invalidated_fence` produces an empty-storage fence with `invoke = &failed_invalidated_fence_invoke` and null `copy_construct`/`move_construct`/`destroy`; `invalidate_entry_locked` assigns `entry.fence = make_invalidated_fence();`.
- [ ] Worker ordering: `grep -n "fence_complete\|fence_destroy\|callbacks_.complete" include/iom/iom.hpp` shows the order `fence_complete → fence_destroy → callbacks_.complete` inside `process` (`:136-155`) and the same order in `drain_list`/`shutdown_and_drain` paths. The post-change `git diff include/iom/iom.hpp` for the `process`/`drain_list`/`shutdown_and_drain` bodies is empty.
- [ ] Intrusive-refcounted CUDA/ROCm leases with cached `FenceResult`: `grep -n "CudaFenceLease\|HipFenceLease" src/cuda/copy.cu src/rocm/copy.hip` returns the lease definition and every registration/invoke/destroy site; `CudaFenceResource::refcount` and `HipFenceResource::refcount` are `std::atomic<std::size_t>` initialised to `1`; `cuda_synchronized_fence_event` / `hip_synchronized_fence_event` populates `cached_result` exactly once and is the only path that synchronizes the event; `cuda_fence_destroy` / `hip_fence_destroy` call the cached-synchronize body, release the metadata slot, then decrement the Task refcount; the raw `task.event` is never reused after the worker step; `cuda_fence_invoke` / `hip_fence_invoke` read the cached result; the last lease to drop the refcount runs the (no-synchronize) `destroy_event_noexcept` + `delete resource;` path. **No `~CudaDevice`/`~HipDevice` waiting on leases is added** — the Device-outlives precondition is the only lifetime contract.
- [ ] SYCL capture: `grep -n "shared_ptr<SyclFenceState>" src/sycl/copy.cpp` shows the inline fence captures exactly `std::shared_ptr<SyclFenceState>` (one member) and `invoke` calls `state->result()`; there is no separate `sycl::event` or `exception_ptr` member inside the SYCL fence capture.
- [ ] `grep -n "TensorView source;\|TensorView destination;" src/ttnn/device.cpp` returns nothing inside `TtnnQueue::Task`, and the pointer members + lifetime-rule comment are present.
- [ ] CPU cost: a throwaway counting-`operator new` probe linked against `libiom.a` reports 0 global heap allocations per `copy`+`wait` cycle over ≥1000 cycles (pre-change: 5), and 0 for the identical-window no-op cycle; the before/after count table of Requirements 8 is recorded including the unchanged CUDA/ROCm event create/destroy counts (1/1), the intrusive refcount atomic activity (no heap), the cached-result mutex activity (one lock per first-invoke per resource, lock-free thereafter), and the inline type-erased fence's per-submission construction/destruction accounting.
- [ ] Latency evidence: under ordinary developer load (no isolation tooling, no `taskset`, no quiet-machine protocol — per task67's ordinary-load convention), record the 5-run `iom_cpu_bench` copy median and no-op baseline median from the post-change tree as **`MESSAGE` output only** and compare against the pre-change tree's medians under the same load on the same machine; the recorded comparison is "not slower than the pre-change tree within run-to-run jitter" (no absolute microsecond pass/fail; no `kAllowanceSeconds` re-derivation in this task). **PF-003 does not edit `test/cpu/test_cpu_bench.cpp`**; `git diff test/cpu/test_cpu_bench.cpp` is empty.
- [ ] Registry behavior unchanged: the updated `iom_tests` registry/quarantine cases pass, including 57-ST-003's fault sweep and the `fence()`-on-invalidated-entry assertion. The new lease race test (Requirement 14) passes: snapshot copy's invoke returns the cached `FenceResult` after the worker drops the Task refcount; last-lease destroy runs exactly once; the event is synchronized exactly once across the entire lifetime of the resource.
- [ ] Lifetime regressions unchanged on every exercising backend: 48-ST-002's `"CPU copy survives derived-view temporaries"` and the TTNN twin pass; on CUDA/ROCm hardware the `ReusingCudaAllocator`-style transactional-failure and queue-destruction-fences-pending-copies cases pass with the intrusive lease + cached-result protocol; on TTNN hardware the deferred-lifetime (destroy-before-wait) and post-58 fault-injection cases pass; on SYCL hardware the 55-ST-001 destroy-before-wait/transactional/teardown cases pass — all through `.agents/skills/remote-development` with exclusive device access.
- [ ] Full CPU-only suites **excluding the bench** exit 0: `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`. The `iom_cpu_bench` executable itself is run as a smoke check and its `MESSAGE`-reported copy median and no-op median are recorded in the change evidence; **the bench's pass/fail gate (67's `kAllowanceSeconds` self-relative gate) is explicitly not part of PF-003's acceptance**, because the pre-PF-003 gate was calibrated against the CPU `StagedWorker` path that PF-003 removes. Re-calibration against the new path is the explicit follow-up task, not PF-003.
- [ ] No retained-failure, token, teardown-ordering, or quarantine behavior differs: CPU tokens wait/rethrow identically; CUDA/ROCm post-enqueue failures still yield waitable tokens whose repeated `wait` rethrows the same retained failure; the registry's snapshot-for-during-pending-work path hands the destructor a fence copy whose lease holds a refcount that keeps the resource alive until `cached_result` is populated, and the destructor's invoke observes the cached result without touching the event.

## Verification

Local CPU-only build, tests, bench, and probe (no accelerator required):

```bash
cmake -S . -B build/pf003 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/pf003 -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench
ctest --test-dir build/pf003 --output-on-failure -R "^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$|^iom_cpu_bench$"
./build/pf003/test/iom_cpu_tests -tc="*derived-view temporaries*"
./build/pf003/test/iom_tests -tc="*snapshot copy remains invocable*"
```

- Before/after bench (recording-only, not a gate): configure and build the same targets from the pre-change tree into a second build dir (e.g. `build/pf003-before`); under ordinary developer load (no `taskset`, no pinning, no quiet machine — per task67's ordinary-load convention), run `./build/<dir>/test/iom_cpu_bench` five times per tree on the same machine and record the `MESSAGE`-reported copy median and no-op median per run. PF-003's evidence is the recorded pair of medians per tree (both trees, same machine, ordinary developer load); PF-003's acceptance is "post-PF-003 medians not slower than pre-PF-003 medians within run-to-run jitter". The bench TU is not modified by this task; re-calibration of `kAllowanceSeconds` is a separate follow-up. **Do not interpret a ctest exit non-zero on the pre-PF-003 gate as a PF-003 defect**; if it occurs, note it in the change evidence and route it to the bench-re-calibration follow-up task.
- Allocation probe: a throwaway TU replacing global `operator new`/`delete` with atomic counters, linked against the built `libiom.a`; create a CPU device/queue and two 16×16 tensors, reset counters, run 1000 `copy`+`wait` cycles and 1000 identical-window no-op cycles, print allocations per cycle. Before-change expectation 5/5, after-change expectation 0/0. Rebuild and rerun once under `-fsanitize=address` for the inline path.
- Accelerator verification must use `.agents/skills/remote-development` per profile (`cuda`, `rocm`, `ttnn`, `sycl` from `.remote-hosts.conf`), with a unique task id and exclusive device access (e.g. `flock /tmp/agent-gpu0.lock`):
  1. `.agents/skills/remote-development/scripts/remote-sync <profile> <task-id>`
  2. Configure and build the backend's conformance and smoke targets (e.g. CUDA: `remote-exec cuda <task-id> 'cmake -S . -B build/cuda -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF' && remote-exec cuda <task-id> 'cmake --build build/cuda --target iom_cuda_conformance_tests iom_cuda_smoke_tests -j'`; ROCm/TTNN/SYCL mirror).
  3. `remote-exec <profile> <task-id> 'ctest --test-dir <build-dir> --output-on-failure -R "<backend>"'` — full backend suite green.
  4. Focused lifetime cases: CUDA/ROCm `*remains transactional across post-enqueue failures*` and `*fences pending copies*`; TTNN deferred-lifetime/destroy-before-wait plus the 58 fault-injection case; SYCL the three 55-ST-001 focused cases. The CUDA/ROCm regressions prove the intrusive lease + cached-result protocol preserves the destroy-before-wait guarantee under the new lease plumbing.
  5. `.agents/skills/remote-development/scripts/remote-clean <profile> <task-id>`
- Static audits close the review's verification method (allocation/mutex/event counting before/after, registry conformance unchanged): the Requirements-10 count table, the probe output, the bench medians (recorded-only), and the acceptance-criteria greps.

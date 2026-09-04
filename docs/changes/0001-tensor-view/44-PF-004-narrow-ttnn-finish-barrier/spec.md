# Narrow TTNN finish barrier to wait boundaries and confine api_mutex_ to SDK calls

**Order:** 44
**Priority:** P1 — perf defect on every TTNN queued op; does not gate other work but ships only with TTNN.
**Blocked by:** `30-AR-002-share-backend-queue-scaffold` (the TTNN queue scaffold is refactored to compose `iom::detail::StagedWorker<Task>`; PF-004's `fence_through_sequence` override and `fence_mutex_` member live on `TtnnQueue` after that refactor, and the worker-shape references below describe the post-AR-002 form), `32-AR-004-unify-ttnn-type-mapping` (the supported-type / native-dtype tables are collapsed into one `constexpr` array; PF-004 does not touch the type-mapping tables, but AR-004 edits the same TU so PF-004's read of the existing lines assumes the post-AR-004 source).
**Source:** `docs/changes/0001-tensor-view/review.md` — `PF-004`
**Review severity:** medium
**Review verification:** strongly-supported (mechanism code-deterministic; latency unmeasured), confidence 80

## Outcome

`TtnnQueue::fence_complete` (the per-task fence callback supplied to the AR-002 `StagedWorker<Task>` helper) no longer calls `mesh().mesh_command_queue(0).finish()`. The single full-device barrier is issued exactly once per unique highest sequence submitted to the device's command queue, at the moment the caller asks for the fence — namely inside `DeviceOps::wait()`'s new fence hook — and is suppressed on every later `wait()` whose sequence is already at-or-below the recorded high-water mark. `api_mutex_` no longer spans a multi-statement "execute task body"; it is taken only around the exact native TTNN API calls that require exclusive device access (the mesh command-queue submission and the `TtnnTensor` native construction/teardown). Independent task progress, `commit_failure` retention, queue lifetime, and the `wait()` error contract are preserved unchanged. `DeviceOps::wait` releases `completion_mutex_` before invoking the fence hook and reacquires it before returning, so the hook's storage of a fence-time failure does not deadlock against the wait predicate, and the triggering `wait()` rethrows any fence-time failure before returning. Concurrent fence attempts are serialized by a TTNN-specific fence mutex; a failed `finish()` for one sequence never causes a later higher-sequence wait to skip its own fence.

## Current failure

`src/ttnn/device.cpp:406-420` (pre-AR-002 worker shape) shows the worker draining `tasks_` serially, taking `api_mutex_` (`device_->api_mutex()`), invoking `ttnn_detail::copy_planes(...)` and `device_->mesh().mesh_command_queue(0).finish()` under that lock, then calling `complete(task.sequence, failure)`. Two distinct defects:

1. **Per-task full-device `finish()`.** The TTNN mesh command queue accepts back-to-back submissions without an interposed barrier. The worker inserts a full-device `finish()` per task; for K independent copies followed by a single `wait()` on the last token, the device is drained K times instead of once. The barrier is unavoidable before `complete()` flips `completed_` (otherwise `DeviceOps::wait` would return prematurely and read stale host data), but its placement per task is what the review pinpoints.
2. **`api_mutex_` is taken around the whole native call pair** (`ttnn_detail::copy_planes` + `finish()`) inside the worker loop. The same mutex is held by `TtnnDevice::create_tensor` (`src/ttnn/device.cpp:454`), by `TtnnTensor::region_from_host` (`src/ttnn/device.cpp:224`), by `TtnnTensor::region_to_host` (`src/ttnn/device.cpp:232`), and by `~TtnnTensor` (`src/ttnn/device.cpp:212`). Any concurrent host-side tensor construction or host transfer blocks behind the worker's per-task native block, including its `finish()`.

`~TtnnQueue` (pre-AR-002 `src/ttnn/device.cpp:256-263`) explicitly does **not** wait on submitted work and relies on a contract-obeying caller to have already waited; the in-file comment at lines 394-401 documents this. No implicit destruction wait is to be added by this task.

A third structural hazard arises from relocating `finish()` into the wait path: the existing `DeviceOps::wait` (`src/iom.cpp:546-570`) holds `completion_mutex_` from line 559 through the function's return. If the fence hook runs while the lock is held and the hook records a failure via `record_post_completion_failure` (which itself takes `completion_mutex_`), the hook deadlocks on a non-recursive `std::mutex`. The fix in this task addresses all three defects.

## Scope

- Modify `src/ttnn/device.cpp` (post-AR-002 `TtnnQueue` member: `fence_mutex_`, `last_finished_seq_`, override of `fence_through_sequence`, and the body of one of AR-002's four `Callbacks` slots — `fence_complete` becomes a no-op for TTNN), the file-local `g_finish_call_count` counter, and the `ttnn_finish_call_count_*` free functions, and `include/iom/iom.hpp` (one new virtual on `DeviceOps`, one new protected helper, three new `protected static` helpers from AR-002 — `identical_window`/`validate_copy`/`unsupported`).
- Modify `src/iom.cpp` (the non-virtual `DeviceOps::wait` path) to release `completion_mutex_` before invoking the fence hook, reacquire it, and re-check `failures_[sequence]` before returning.
- Add a temporary test-only header `include/iom/ttnn/test_counter.hpp` (Step 3 only) and a TU-local free function in `src/ttnn/device.cpp` that exposes the file-local counter to the verification smoke program.
- CPU, CUDA, and ROCm backends (`src/cpu/device.cpp`, `src/cuda/copy.cu`, `src/rocm/copy.hip`) inherit the default no-op implementation of the new hook; no change to their source required.
- Preserve `create_tensor`'s exception categories, capability rejection, `TtnnTensor` construction contract, `commit_failure` retention, `publish_staged` ordering, the identical-window no-op, and the `DeviceOps::wait`/`complete` error contract verbatim.
- Preserve the post-AR-002 `TtnnQueue` destructor contract (`StagedWorker::shutdown_and_drain` is called first, no `finish()` is added).
- Preserve per-queue in-order serialization and per-device independent-queue progress.

## Implementation references

- **Post-AR-002 worker shape (the form PF-004 edits).** After `30-AR-002-share-backend-queue-scaffold` lands, the four backend queue classes compose `iom::detail::StagedWorker<Task>` as a value member. The TTNN-side shape is fixed by AR-002 lines 334-342. PF-004 quotes it verbatim here:
  ```cpp
  class TtnnQueue final : public DeviceOps {
  public:
      explicit TtnnQueue(TtnnDevice& device);
      ~TtnnQueue() override;
      oid copy(const TensorView& source, TensorView& destination) override;
      // six compute-op stubs from AR-002's DeviceOps::unsupported("TTNN", "<op>")
  private:
      // Post-AR-002 Task struct (AR-002 line 339): existing fields preserved verbatim + void* fence.
      // The four raw-pointer/raw-vector fields (source, destination, source_planes,
      // destination_planes) are the pre-AR-002 shape AR-002 preserves; ST-002 later
      // replaces them with value-owned view descriptors. PF-004 does not edit them
      // and does not assume the value-owned form.
      struct Task {
          std::uint64_t sequence;
          const TensorView* source;
          TensorView* destination;
          std::vector<ttnn::Tensor*> source_planes;
          std::vector<ttnn::Tensor*> destination_planes;
          bool no_op;
          void* fence = nullptr;                 // AR-002 adds this; TTNN always leaves it nullptr
      };
      // Post-AR-002 callback set: EXACTLY four Callbacks members (AR-002 line 126).
      // PF-004 does NOT add a fifth callback. PF-004 changes the body of one
      // (fence_complete becomes a no-op for TTNN) and adds the new DeviceOps
      // virtual fence_through_sequence, which is separate from the four-arg
      // StagedWorker::Callbacks struct.
      static void execute(Task& t) noexcept;             // post-PF-004: NO finish() here
      static void fence_complete(void* fence) noexcept;  // post-PF-004: no-op for TTNN (was the per-task finish() callback under AR-002)
      static void fence_destroy(void* fence) noexcept;   // no-op for TTNN under AR-002 already
      std::mutex submission_order_mutex_;                // AR-002
      TtnnDevice* device_;
      iom::detail::StagedWorker<Task> worker_;           // AR-002 value member; PublishPolicy::CompleteOnThrow
      // PF-004 adds the two members below (no other TTNN-side member changes):
      std::mutex fence_mutex_;
      std::atomic<std::uint64_t> last_finished_seq_{0};
  };
  ```
  **AR-002 line 126 pins the callback count:** `StagedWorker<Task>::Callbacks` has exactly four `std::function` members (`execute`, `fence_complete`, `fence_destroy`, `complete`). PF-004 does NOT add a fifth; it modifies one existing callback's body and adds a new `DeviceOps` virtual (`fence_through_sequence`) that lives on `DeviceOps`, not on `StagedWorker<Task>::Callbacks`.

  **AR-002's TTNN callback contents (lines 338, 403):** `TtnnQueue::fence_complete` is `std::lock_guard<std::mutex> api_lock(device_->api_mutex()); device_->mesh().mesh_command_queue(0).finish();` — the per-task `finish()` is supplied through `Callbacks::fence_complete`, not through `Callbacks::execute`. PF-004 replaces this `fence_complete` body with `return;`. The wait-boundary fence (`fence_through_sequence`) takes ownership of the device-wide barrier. `TtnnQueue::fence_destroy` is unchanged (already a no-op for TTNN under AR-002). `TtnnQueue::execute` is unchanged by PF-004 (it never called `finish()` under AR-002).

  **ST-002 note (out of scope for PF-004):** the four raw-pointer/raw-vector fields (`source`, `destination`, `source_planes`, `destination_planes`) are the pre-AR-002 shape that AR-002 preserves verbatim (see `30-AR-002-share-backend-queue-scaffold/spec.md:339`). `48-ST-002-own-cpu-ttnn-task-views` later replaces these fields with value-owned view descriptors. PF-004 does not edit them and does not assume the value-owned form. PF-004's `execute` only reads `source` / `destination` / `source_planes` / `destination_planes` to call `ttnn_detail::copy_planes`; it adds no reads or writes to those fields.

  PF-004's `TtnnQueue` delta vs the post-AR-002 baseline is:
  1. `TtnnQueue::fence_complete` body: replace the existing `mesh().mesh_command_queue(0).finish()` body with `return;` (the fence is owned by the wait boundary).
  2. Add the two private members `std::mutex fence_mutex_` and `std::atomic<std::uint64_t> last_finished_seq_{0}`.
  3. Add the `fence_through_sequence` override (body below) as a new `DeviceOps` virtual on `TtnnQueue`, separate from the AR-002 `StagedWorker<Task>::Callbacks` struct.
  4. Nothing else: no change to `TtnnQueue::execute` (which already did not call `finish()` under AR-002), no change to `submit`, `copy`, `~TtnnQueue`, the six compute-op stubs, `DeviceOps::complete`/`commit_failure`, or any other backend TU. PF-004 does NOT add a fifth `Callbacks` slot.

- **Modify:** `src/ttnn/device.cpp` `TtnnQueue::fence_complete`. New body (post-PF-004):
  ```cpp
  void TtnnQueue::fence_complete(void* /*fence*/) noexcept {
      // AR-002 called mesh().mesh_command_queue(0).finish() here under api_mutex_.
      // PF-004 suppressed it: the wait-boundary fence in fence_through_sequence
      // owns the device-wide barrier; this callback is a no-op for TTNN.
  }
  ```
- **Modify:** `include/iom/iom.hpp:28-141` (`class DeviceOps`). Add one new protected virtual and one new protected helper:
  - Virtual symbol: `virtual void fence_through_sequence(std::uint64_t sequence) noexcept;`
  - Helper symbol: `void record_post_completion_failure(std::uint64_t sequence, std::exception_ptr failure);` declared `protected`, defined in `src/iom.cpp`. Implementation: take `completion_mutex_`, look up `failures_.find(sequence)`; if absent, insert `std::move(failure)` and `notify_all`; if present, do nothing. The helper is idempotent: a worker-recorded failure for the same sequence is never overwritten.
  - Default `fence_through_sequence` implementation in `src/iom.cpp`: does nothing (CPU/CUDA/ROCm don't need it; TTNN overrides).
  - Visibility: `protected`. The override is called only by the base class; the helper is called only by overrides.
  ```cpp
  void DeviceOps::wait(oid token) {
      const std::uint64_t id = token >> kSequenceBits;
      const std::uint64_t sequence = token & kSequenceMask;
      if (id == 0) {
          throw std::invalid_argument("oid queue id is zero");
      }
      std::unique_lock<std::mutex> lock(completion_mutex_);
      if (id != queue_id_) {
          throw std::invalid_argument("oid belongs to another queue");
      }
      if (sequence == 0 || sequence >= next_sequence_) {
          throw std::invalid_argument("oid sequence was never submitted");
      }
      completion_cv_.wait(lock, [&] {
          return completed_ >= sequence || failures_.count(sequence) != 0;
      });
      if (const auto it = failures_.find(sequence); it != failures_.end()) {
          std::rethrow_exception(it->second);
      }
      lock.unlock();
      fence_through_sequence(sequence);
      lock.lock();
      if (const auto it = failures_.find(sequence); it != failures_.end()) {
          std::rethrow_exception(it->second);
      }
  }

  // The implementation retains the existing queue-id, sequence, and
  // completion-range checks; the hook is the only operation outside the
  // completion mutex.
  ```

  Phase 1 waits for completion or a worker-recorded failure. Phase 2 calls the
  no-throw fence hook after releasing `completion_mutex_`. Phase 3 rechecks
  `failures_` after reacquiring the lock, so a fence-time failure is retained
  and observable by the triggering and every subsequent wait.
- **Modify:** `src/ttnn/device.cpp` `class TtnnQueue` (post-AR-002). Add `fence_mutex_` and `last_finished_seq_` as private members (declared alongside the existing `submission_order_mutex_` and `worker_`), add the file-local atomic counter, and override `fence_through_sequence`:
  ```cpp
  // TU-local counter used by the verification seam. Counts actual
  // mesh().mesh_command_queue(0).finish() invocations (attempted, including
  // any that threw). Removed in Step 5.
  namespace ttnn_test_internal {
      std::atomic<std::uint64_t> g_finish_call_count{0};
  }

  void TtnnQueue::fence_through_sequence(std::uint64_t sequence) noexcept override {
      std::lock_guard<std::mutex> fence_lock(fence_mutex_);
      const std::uint64_t now =
              last_finished_seq_.load(std::memory_order_acquire);
      if (sequence <= now) {
          // Already fenced past this sequence. NO finish() invocation,
          // NO counter increment.
          return;
      }
      // Advance the high-water mark only after finish() returns
      // successfully. On failure, last_finished_seq_ stays at `now`;
      // the next wait() for a strictly higher sequence fences again.
      try {
          std::lock_guard<std::mutex> api_lock(device_->api_mutex());
          device_->mesh().mesh_command_queue(0).finish();
          // Count the attempt (including failed ones) BEFORE bumping.
          ttnn_test_internal::g_finish_call_count.fetch_add(
                  1, std::memory_order_relaxed);
          last_finished_seq_.store(sequence, std::memory_order_release);
      } catch (...) {
          // Count the failed attempt too: the SDK call was made.
          ttnn_test_internal::g_finish_call_count.fetch_add(
                  1, std::memory_order_relaxed);
          // last_finished_seq_ stays at `now`.
          record_post_completion_failure(
                  sequence, std::current_exception());
      }
  }
  ```
  The exact policy the spec pins:

  1. `fence_mutex_` is a private `std::mutex` member on `TtnnQueue`. `fence_through_sequence` takes exactly one `std::lock_guard<std::mutex> fence_lock(fence_mutex_)` for the entire hook body (one acquisition, one release). It serializes concurrent hook invocations on the same `TtnnQueue`. There is no second `fence_lock` site in the source.
  2. The `sequence <= now` skip branch does NOT issue `finish()` and does NOT increment the counter. The counter is incremented ONLY when an actual `mesh().mesh_command_queue(0).finish()` call site is entered — both the success and the failure paths increment by exactly one each.
  3. Bump-on-success-only: the `last_finished_seq_.store(sequence, ...)` call happens only inside the `try` block after `finish()` returns successfully. On exception, the high-water mark stays at `now`. The next `wait()` for a strictly higher sequence therefore fences again (its `sequence > last_finished_seq_ = now`).
  4. Each strictly-advancing `wait(sequence)` issues its own `finish()`. A `wait(k)` does NOT skip its fence just because a `wait(j)` with `j > k` already fenced — different SDK barrier call. The TTNN `mesh_command_queue(0).finish()` is a device-wide barrier whose scope is the state of the mesh command queue at the moment of the call.
  5. A fence-time failure is recorded by `record_post_completion_failure(k, ...)` against the very sequence whose `finish()` failed. The triggering `wait(k)` then rethrows via Phase 3's reacquired-lock re-check. Every subsequent `wait(k)` rethrows the same `exception_ptr` from `failures_[k]` (the entry is never erased on read; see `src/iom.cpp:566-569`).
  6. A `wait(k)` does NOT inherit a `finish()` failure from `wait(k - 1)` via Phase 1 or Phase 3. Each sequence has its own `failures_` entry; the caller observes a lower-sequence failure only by calling `wait(k - 1)` themselves. This matches the existing per-token semantics: `failures_[sequence]` is recorded per sequence and `wait(token)` rethrows only that token's failure.
  7. If both the worker and the fence recorded a failure for the same sequence, the worker's failure wins: `record_post_completion_failure` is a no-op when `failures_[sequence]` already exists. The triggering `wait()` still rethrows via Phase 1's early rethrow (worker-recorded failure path), not Phase 3.

- **Modify:** `src/ttnn/device.cpp` anonymous namespace. Add the file-local counter and the `extern`-visible free function pair that the temporary test header declares:
  ```cpp
  // Verification-only seam. Removed in Step 5 along with the test header.
  std::uint64_t ttnn_finish_call_count_load() noexcept {
      return ttnn_test_internal::g_finish_call_count.load(
              std::memory_order_relaxed);
  }
  void ttnn_finish_call_count_reset() noexcept {
      ttnn_test_internal::g_finish_call_count.store(
              0, std::memory_order_relaxed);
  }
  ```
- **Add (Step 3 only):** `include/iom/ttnn/test_counter.hpp`:
  ```cpp
  // Verification-only header. Removed in Step 5.
  #pragma once
  #include <cstdint>
  namespace iom::ttnn_test {
      // Counts actual mesh().mesh_command_queue(0).finish() invocations,
      // including any that threw. NOT incremented on the high-water skip
      // path; NOT incremented when the hook returns without calling finish.
      [[nodiscard]] std::uint64_t ttnn_finish_call_count_load() noexcept;
      void ttnn_finish_call_count_reset() noexcept;
  }
  ```
  This header is the verification-only seam. `TtnnQueue` itself is in `src/ttnn/device.cpp`'s anonymous namespace and is NOT visible to the smoke program; the smoke program reaches the counter through the named free functions in `iom::ttnn_test`, which read the TU-local atomic. No `dynamic_cast` is required, and no public type name is invented for `TtnnQueue`.
- **Read:** `src/ttnn/device.cpp` (post-AR-002 / post-AR-004 form) — `TtnnQueue` constructor, `submit`, the `execute` static callback, the `complete` lambda (`Callbacks::complete` calling `DeviceOps::complete`), `~TtnnQueue` calling `worker_.shutdown_and_drain()`. PF-004 does not touch any of these except `fence_complete` (replaces the body with a no-op).
- **Read:** `include/iom/iom.hpp:74-140` (`submit`, `complete`, `commit_failure`, `failures_`, `pending_failures_`) to confirm the per-sequence uniqueness invariant the hook relies on and the storage location for post-completion failures.
- **Read:** `src/iom.cpp:546-610` (`DeviceOps::wait`, `DeviceOps::complete`, `DeviceOps::commit_failure`) for the exact insertion point of the hook call and for the shape of `record_post_completion_failure`.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` (existing capability and copy-conformance cases) must remain green. No new test files. The existing `SubmissionFault` style conformance case (compare against `src/cuda/copy.cu:481-495` analogue) exercises the failure-capture path and must continue to rethrow on `wait()` post-fix.

## Requirements

- One new symbol `iom::DeviceOps::fence_through_sequence(std::uint64_t sequence) noexcept` is added. Its default implementation is a no-op. It is invoked exactly once per `DeviceOps::wait(token)` invocation whose Phase 1 predicate confirmed completion and whose Phase 1 early-rethrow did not fire. The invocation happens with `completion_mutex_` released at the call site.
- One new symbol `iom::DeviceOps::record_post_completion_failure(std::uint64_t sequence, std::exception_ptr failure)` is added. It writes the failure into `failures_[sequence]` only if no entry for that sequence already exists (worker-recorded failures are preserved). The write happens under `completion_mutex_` and `notify_all` is called so any pending `wait()` re-checks the predicate.
- `DeviceOps::wait(token)` is rewritten in three phases:
  - Phase 1: under `completion_mutex_`, wait for `completed_ >= sequence || failures_.count(sequence) != 0`; if a worker-recorded failure is present, rethrow and return (no hook call).
  - Phase 2: with `completion_mutex_` released, call `fence_through_sequence(sequence)`. The hook is `noexcept`; nothing it does escapes the call. A fence-time failure is stored into `failures_[sequence]` by `record_post_completion_failure` before the hook returns.
  - Phase 3: reacquire `completion_mutex_` and re-check `failures_[sequence]`. If the hook recorded a fence-time failure, **the triggering `wait(token)` rethrows via Phase 3 before returning**. Repeated `wait()` on the same token also rethrows the same `exception_ptr` (the entry is never erased on read).
- `TtnnQueue::fence_through_sequence(sequence)` overrides the base no-op. Under `fence_mutex_` (one `std::lock_guard` for the entire hook body), it reads `last_finished_seq_`. If `sequence <= last_finished_seq_`, it returns without an SDK call. Otherwise it calls `mesh().mesh_command_queue(0).finish()` under `device_->api_mutex()`; on success, it bumps `last_finished_seq_` to `sequence` and the counter increments by 1; on exception, it leaves `last_finished_seq_` unchanged, the counter increments by 1, and it calls `record_post_completion_failure(sequence, std::current_exception())`. The override is `noexcept` and never propagates a `finish()` exception to the caller of `DeviceOps::wait`.
- The TTNN `fence_complete` callback (post-AR-002 / post-PF-004) is a no-op for TTNN. The per-task `finish()` that AR-002 placed in this callback (per `30-AR-002-share-backend-queue-scaffold/spec.md:338`) is replaced by the wait-boundary fence hook. The wait boundary is the single owner of the device-wide barrier.
- `api_mutex_` is taken only around the exact SDK calls that require it:
  - `ttnn_detail::copy_planes(...)` inside `TtnnQueue::execute` (one lock acquisition per task, released before the function returns to the helper).
  - `TtnnTensor` native construction in `TtnnDevice::create_tensor` (existing behavior unchanged at `src/ttnn/device.cpp:454`).
  - `TtnnTensor::region_from_host` and `region_to_host` (existing behavior unchanged at lines 224, 232).
  - `~TtnnTensor` `planes_.clear()` (existing behavior unchanged at line 212).
  - The new `fence_through_sequence` override's `mesh_command_queue(0).finish()` call.
  - No other call site takes `api_mutex_`.
- Failure capture is preserved end-to-end:
  - A thrown exception from the native copy is captured into the AR-002 helper's path (the lambda propagating out of `submit_copy` → ST-007's `submit` rollback), which still records the failure via `DeviceOps::commit_failure` / surfaces it via Phase 1's rethrow path on `wait()`. No change to `src/iom.cpp:566-569`.
  - A thrown exception from `finish()` is caught inside `fence_through_sequence` and stored via `record_post_completion_failure`. The triggering `wait(token.sequence)` rethrows via Phase 3's reacquired-lock re-check on the same call. Every subsequent `wait(token.sequence)` invocation sees the failure in Phase 1's predicate (or in Phase 3 if the worker hasn't raced) and rethrows the same `exception_ptr` (the entry is never erased on read).
  - If both the worker and the fence recorded a failure for the same sequence, the worker's failure wins: `record_post_completion_failure` is a no-op when `failures_[sequence]` already exists. The triggering `wait()` rethrows via Phase 1's early rethrow (worker-recorded failure path), not Phase 3.
- The serialized fence-failure policy is exactly: each `wait(seq)` whose `seq > last_finished_seq_` issues its own `finish()`. A failed `finish()` leaves `last_finished_seq_` unchanged; the next `wait()` for any strictly higher sequence fences again. There is no automatic "observe-lower-failure" propagation between sequences; each sequence is observed independently via its own `wait()`. The triggering `wait()` that observes a fence failure rethrows before returning; later waits on the same sequence also rethrow the same `exception_ptr`.
- Queue lifetime is unchanged. `~TtnnQueue` continues to call `worker_.shutdown_and_drain()` first (per AR-002), then perform backend-native cleanup. No `finish()` is added to the destructor. The no-destruction-wait contract from AR-002 is preserved verbatim.
- Independent-queue progress is preserved. Multiple `TtnnQueue` instances from one `TtnnDevice` share `api_mutex_` only inside their fence hook (one `finish()` per strictly-advancing wait, brief critical section). Tasks submitted between waits continue to make progress without per-task barriers.
- CPU, CUDA, ROCm behavior is unchanged. The new `fence_through_sequence` default no-op means their `complete()`-then-`wait()` flow already satisfies their fence contract (CPU has no device; CUDA/ROCm use event fences in their own `complete()` paths). Their `record_post_completion_failure` helper is inherited but unused.
- No new virtual method is added beyond `fence_through_sequence`. No change to `submit`, `complete`, `commit_failure`, `encode_token`, or `seek_next_sequence`. AR-002's `iom::detail::StagedWorker<Task>` helper and its `Callbacks` shape (exactly four `std::function` members) are not modified by PF-004.
- PF-004 does NOT modify `TtnnQueue::Task`. The four raw-pointer/raw-vector fields (`source`, `destination`, `source_planes`, `destination_planes`) stay as AR-002 preserved them; the value-owned view replacement is `48-ST-002-own-cpu-ttnn-task-views`'s exclusive scope (out of scope here).
- The verification seam (`include/iom/ttnn/test_counter.hpp`, `ttnn_finish_call_count_load`, `ttnn_finish_call_count_reset`, the file-local `g_finish_call_count` atomic, and the two counter increment sites inside `fence_through_sequence`) exists only during the Step 3 verification window. All four symbols are deleted in Step 5. The counter semantics are: increments ONLY when `mesh().mesh_command_queue(0).finish()` is actually entered — both the success path and the failure path increment by exactly one; the `sequence <= last_finished_seq_` skip path does NOT increment.

## Non-goals

- Host-transfer materialization and per-plane buffer/passes reduction — owned by `45-PF-005-streamline-ttnn-host-transfer`.
- ST-class teardown-without-fence (review `ST-001`) and tensor-destructor fencing (review `ST-003`) — owned by `47-ST-001-fence-backend-queue-teardown` and `49-ST-003-fence-tensor-destruction` respectively. PF-004's hook is independent of the registry/destructor-side fence from `49-ST-003-fence-tensor-destruction`; the wait-boundary fence and the destructor-side fence are complementary and both remain in the shipped source.
- CPU/TTNN queue task view ownership (review `ST-002`) — owned by `48-ST-002-own-cpu-ttnn-task-views`. PF-004 does not change the `Task` view-descriptor shape; that is `48-ST-002`'s exclusive scope. The post-AR-002 `TtnnQueue::Task` retains the pre-AR-002 raw-pointer/raw-vector fields as preserved by AR-002; PF-004 adds no reads or writes to those fields.
- TTNN supported-type knowledge deduplication (review `AR-004`) — owned by `32-AR-004-unify-ttnn-type-mapping`. PF-004's blocker relationship is sequencing-only (same TU); PF-004 does not edit the supported-type / native-dtype tables.
- Per-call CUDA/ROCm default-stream serialization fixes — owned by `ST-005`.
- Replacing `mesh_command_queue(0).finish()` with per-op TTNN events. The TTNN C++ host API in this revision exposes `mesh_command_queue(0).finish()` as the supported device-wide fence; finer-grained events are not part of the current `tt-metalium/host_api.hpp` surface that this change already includes (`src/ttnn/device.cpp:3`). The PF-004 perf improvement is achieved by relocating the existing `finish()` call to the wait boundary, not by introducing new SDK calls.
- Adding a virtual `wait()` override on `TtnnQueue`. `DeviceOps::wait` is non-virtual and shared across all backends; the fence hook is the only new indirection.
- Adding a fifth `Callbacks` slot. AR-002's `StagedWorker<Task>::Callbacks` has exactly four members (`execute`, `fence_complete`, `fence_destroy`, `complete`) per AR-002 line 126; PF-004 modifies the body of one (`fence_complete`) and adds `fence_through_sequence` as a separate `DeviceOps` virtual.
- TTNN profiler / Metal timeline capture. No project-supported timeline tool exists for TTNN in this change; verification uses the `g_finish_call_count` counter (see Verification). The verification seam is removed before the change ships.
- Probing `mesh_command_queue(0).finish()` for a documented exception type. The TTNN host API does not specify a concrete exception hierarchy for `finish()` failures; the catch is `catch (...)` and the resulting `std::exception_ptr` is forwarded to `record_post_completion_failure` without rewrapping.
- Cross-sequence failure propagation between `wait(k)` and `wait(k - 1)`. The contract is per-token: `wait(token)` observes only that token's failure. A caller that wants to observe a sequence's fence failure must call `wait(token)` for that sequence.

## Acceptance criteria

- [ ] `grep -n 'finish()' src/ttnn/device.cpp` returns exactly one call site, inside `TtnnQueue::fence_through_sequence`. There is no `finish()` call inside `TtnnQueue::execute`, no `finish()` call inside `TtnnQueue::fence_complete` (post-PF-004 no-op body), no `finish()` call inside `~TtnnQueue`, and no `finish()` call inside any other TtnnQueue method.
- [ ] `grep -n 'api_mutex_' src/ttnn/device.cpp` shows the lock taken at: (a) `TtnnDevice::create_tensor` line 454, (b) `TtnnQueue::execute` around `ttnn_detail::copy_planes(...)` only (one lock acquisition), (c) `TtnnTensor::region_from_host`, `region_to_host`, and `~TtnnTensor` (existing), and (d) `TtnnQueue::fence_through_sequence` around `mesh().mesh_command_queue(0).finish()`. No lock spans more than the named SDK call.
- [ ] `grep -n 'fence_through_sequence' include/iom/iom.hpp src/iom.cpp src/ttnn/device.cpp` returns the base virtual declaration, the default no-op body, the `TtnnQueue` override, and the single call from `DeviceOps::wait` (Phase 2). No other call site.
- [ ] `grep -n 'record_post_completion_failure' include/iom/iom.hpp src/iom.cpp src/ttnn/device.cpp` returns the base declaration, the body in `src/iom.cpp` that writes into `failures_` only when absent and notifies, and the single call from `TtnnQueue::fence_through_sequence`. No other call site.
- [ ] `grep -n 'fence_mutex_' src/ttnn/device.cpp` returns exactly one declaration of `std::mutex fence_mutex_` as a `TtnnQueue` member and exactly one `std::lock_guard<std::mutex> fence_lock(fence_mutex_)` site inside `fence_through_sequence` (one acquisition for the entire hook body). No second `fence_lock` use exists.
- [ ] `grep -n 'Callbacks' src/ttnn/device.cpp` confirms AR-002's `StagedWorker<Task>::Callbacks` is constructed at the end of `TtnnQueue`'s constructor body with exactly four members (`execute`, `fence_complete`, `fence_destroy`, `complete`). PF-004 does not add a fifth member to `Callbacks`. The `fence_through_sequence` virtual lives on `iom::DeviceOps`, not on `StagedWorker<Task>::Callbacks`.
- [ ] `DeviceOps::wait(token)` invokes `fence_through_sequence(sequence)` exactly once on the Phase 2 path (between Phase 1's lock release and Phase 3's lock reacquire) and zero times on the Phase 1 rethrow path. Verifiable in source at `src/iom.cpp:546-594`.
- [ ] Phase 3's reacquired-lock re-check of `failures_[sequence]` exists in source and contains a `rethrow_exception` of the found entry. The triggering `wait(token)` rethrows the fence-time failure via Phase 3 before returning.
- [ ] A second `wait(token)` call on the same token whose first call recorded a fence failure: Phase 1's predicate `failures_.count(sequence) != 0` is true, Phase 1 rethrows without invoking the hook. Verifiable by reading `src/iom.cpp:563-565`.
- [ ] Two consecutive `wait(token)` calls with the same token on the happy path: the second invocation observes `sequence <= last_finished_seq_` and the hook returns without an SDK call AND without incrementing `g_finish_call_count`. Verifiable via `iom::ttnn_test::ttnn_finish_call_count_load()` (Step 3 only).
- [ ] A `wait(token_k)` whose `sequence_k > last_finished_seq_` increments `g_finish_call_count` by exactly 1 and invokes exactly one `mesh().mesh_command_queue(0).finish()` call. The pre-fix (post-AR-002 / pre-PF-004 baseline) build issues one `finish()` per task inside `TtnnQueue::fence_complete` (the AR-002 callback body per `30-AR-002-share-backend-queue-scaffold/spec.md:338`) — pre-PF-004, the worker fires `finish()` per task; post-PF-004, the worker issues zero `finish()` inside `fence_complete` (the new no-op body) and the wait-boundary hook fires exactly once per strictly-advancing wait.
- [ ] A `wait(token_k)` after a fence-failed `wait(token_j)` with `j < k` issues its own `finish()` (because `last_finished_seq_` was not bumped past `j` on the failure), records `failures_[k]` if that `finish()` also fails, and `failures_[j]` remains as recorded by the prior call. `g_finish_call_count` reflects the two distinct `finish()` invocations. Verifiable by reading the hook's bump-on-success-only path and `record_post_completion_failure`'s insert-only-on-absent path. The triggering `wait(token_k)` rethrows the fence-time failure via Phase 3; the triggering `wait(token_j)` rethrew via its own Phase 3.
- [ ] `~TtnnQueue` continues to call `worker_.shutdown_and_drain()` first and join the worker without invoking `finish()` and without blocking on any device-side work.
- [ ] `iom_conformance::run_async_copy_conformance` (the existing shared harness at `test/backend/backend_conformance_copy_storage.hpp:455-503`) called from `test/ttnn/test_ttnn_conformance.cpp:450-455` (`TTNN conformance: asynchronous copies against the CPU reference`) stays green, including its three-copies-then-`wait(three)`-then-`wait(one)`-then-`wait(three)` pattern and the chained-destination byte-equality assertions at lines 495-502.
- [ ] In the shipped source, `grep -n 'g_finish_call_count\|finish_call_count' src/ttnn/device.cpp` returns no matches and `find include/iom/ttnn/test_counter.hpp` returns no path. The verification-only counter, getter/reset pair, file-local atomic, both increment sites, and temporary header exist only during the Step 3 verification window.
- [ ] PF-004 does NOT modify `TtnnQueue::Task`. The struct retains the post-AR-002 fields (`sequence`, `source`, `destination`, `source_planes`, `destination_planes`, `no_op`, `void* fence = nullptr`) verbatim — no `event` field (CUDA/ROCm only), no fifth pointer, no value-owned view replacement. The value-owned view replacement is `48-ST-002-own-cpu-ttnn-task-views`'s exclusive scope.

## Verification

One concrete verification path. No alternatives, no fallback tools, no timeline.

**Step 1 — Build on TTNN hardware via the project's remote-development procedure.** Follow the build recipe at `docs/changes/0001-tensor-view/13-ttnn-buildable-scaffold/spec.md`. The configuration must set `TTNN_ENABLED=ON` and link against `TT::Metalium` and `TTNN::TTNN` (see `CMakeLists.txt:115-145`). Hardware is required because the device-side barrier is only observable on a real TTNN mesh; a CPU-only build cannot exercise the per-task `finish()` removal. Run the remote build under the project's `ttnn` profile (see `.agents/skills/remote-development/references/hosts.example.conf:17`).

**Step 2 — Run the existing TTNN conformance suite.** From the build tree:

```bash
ctest --test-dir build -R ttnn --output-on-failure
```

This runs every `TEST_CASE` in `test/ttnn/test_ttnn_conformance.cpp` (capability, copy, lifetime, compute-capability, the full shared suite). All cases must remain green; in particular `TTNN conformance: asynchronous copies against the CPU reference` (line 450-455) exercises the three-copies-then-`wait(three)`-then-`wait(one)`-then-`wait(three)` pattern that is the canonical wait-boundary scenario.

**Step 3 — Read the verification-only counter from the wait-boundary scenario.** The verification seam is the temporary `iom::ttnn_test` API declared in `include/iom/ttnn/test_counter.hpp`:

```cpp
// include/iom/ttnn/test_counter.hpp  (verification-only; removed in Step 5)
namespace iom::ttnn_test {
    // Counts actual mesh().mesh_command_queue(0).finish() invocations,
    // including any that threw. NOT incremented on the high-water skip
    // path; NOT incremented when the hook returns without calling finish.
    [[nodiscard]] std::uint64_t ttnn_finish_call_count_load() noexcept;
    void ttnn_finish_call_count_reset() noexcept;
}
```

These free functions are defined in `src/ttnn/device.cpp` and read the file-local `ttnn_test_internal::g_finish_call_count` atomic. The counter is incremented by exactly one each time the hook enters the `try` block (one per `finish()` invocation, success or failure), and is NOT incremented on the early `if (sequence <= now) return;` skip. The smoke program is compiled against `iom_ttnn` plus the temporary test header, executed on the same remote host under the same `ttnn` profile, and is not added to the test suite:

```cpp
#include "iom/device.hpp"
#include "iom/tensor.hpp"
#include "iom/ttnn/device.hpp"
#include "iom/ttnn/test_counter.hpp"

#include <cstdio>
#include <vector>

int main() {
    auto device = iom::make_ttnn_device(0);
    auto queue = device->create_ops();

    // Reset the counter so the measurement is scoped to this run.
    iom::ttnn_test::ttnn_finish_call_count_reset();

    // Submit K = 64 back-to-back copies through one queue.
    constexpr std::size_t K = 64;
    const iom::TensorSpec spec{
        iom::TensorShape(std::vector<std::size_t>{16, 16}),
        iom::DataType::F32,
        iom::QuantizationFormat::NONE};
    auto src = device->create_tensor(spec);
    auto dst = device->create_tensor(spec);
    std::vector<iom::oid> tokens;
    tokens.reserve(K);
    for (std::size_t i = 0; i < K; ++i) {
        tokens.push_back(queue->copy(src->view(), dst->view()));
    }

    // One wait on the last token plus K-1 idempotent waits on the same token.
    const iom::oid last = tokens.back();
    queue->wait(last);
    for (std::size_t i = 0; i + 1 < K; ++i) {
        queue->wait(last);
    }

    const std::uint64_t calls =
            iom::ttnn_test::ttnn_finish_call_count_load();
    std::printf("mesh().mesh_command_queue(0).finish() calls: %llu "
                "(expected 1 for K=64)\n",
                static_cast<unsigned long long>(calls));
    return calls == 1 ? 0 : 1;
}
```

**Expected outcome.** The counter reports `1`. Pre-fix (post-AR-002 / pre-PF-004 baseline) the worker fires `finish()` once per task inside `TtnnQueue::fence_complete` (the AR-002 callback body that PF-004 replaces) — K = 64 calls for K tasks. Post-fix the worker issues zero `finish()` calls inside `fence_complete` (the new no-op body); the single `wait(last_token)` issues exactly one `finish()` call via `fence_through_sequence`; the subsequent 63 `wait()` calls on the same token see `sequence <= last_finished_seq_` under `fence_mutex_`, return early from the hook, and do NOT increment the counter. The smoke program exits 0 on success, nonzero otherwise.

**Step 4 — Confirm failure rethrow.** Two sub-checks:

1. **Worker-recorded failure path.** Under `ctest -R ttnn`, `iom_conformance::run_transfer_error_conformance` (called from `test/ttnn/test_ttnn_conformance.cpp:464-468`) covers rejection paths that never consume a sequence. The TTNN-specific failure surface for the worker is captured by the AR-002 helper's path (the `execute` lambda throwing propagates through `submit_copy` → ST-007's `submit` rollback), which records a failure into `DeviceOps::commit_failure` and surfaces it via Phase 1's rethrow path on `wait()`. This case must remain green and the counter recorded in step 3 must be unchanged because `fence_through_sequence` is not invoked (the rethrow happens in Phase 1 before Phase 2's hook call).

2. **Fence-recorded failure path (triggering wait rethrows).** Verified by code inspection (no TTNN hardware seam exists to inject a `finish()` exception in this change). The implementation in `TtnnQueue::fence_through_sequence` shows the exact `try { ... finish() ... ++g_finish_call_count; store last_finished_seq_; } catch (...) { ++g_finish_call_count; record_post_completion_failure(sequence, std::current_exception()); }` shape with bump-on-success-only. The behavior is: the exception is caught inside the override, the override returns normally (preserving `noexcept`), `last_finished_seq_` is NOT bumped, `failures_[sequence]` gains an entry, the counter was incremented by 1 (the failed `finish()` invocation counts as an attempt), the triggering `wait(token.sequence)` rethrows via Phase 3's reacquired-lock re-check on the SAME call (before returning), and every subsequent `wait(token.sequence)` invocation sees `failures_[sequence]` already present in Phase 1's predicate and rethrows without invoking the hook (and without further counter increments). The smoke program in Step 3 can be extended with one extra code-inspection note (no runtime check) noting that the implementation follows this contract; no test seam exercises the `finish()` exception path because none exists in the current tree.

**Step 5 — Remove the verification-only seam.** `g_finish_call_count`, the `ttnn_finish_call_count_load` and `ttnn_finish_call_count_reset` definitions, the `include/iom/ttnn/test_counter.hpp` header, and the two `g_finish_call_count.fetch_add(1, ...)` increment sites inside `fence_through_sequence` exist solely for Step 3. The implementer deletes all of them before the change ships. The shipped source contains `last_finished_seq_` and `fence_mutex_` only (plus the post-AR-002 baseline: `fence_mutex_` and `last_finished_seq_` are new; everything else in `TtnnQueue` (including the `Task` shape with the four AR-002-preserved raw-pointer/raw-vector fields, the three static callbacks including the now-no-op `fence_complete`, and the `Callbacks` struct with exactly four members) is preserved verbatim from AR-002). `grep -n 'g_finish_call_count\|finish_call_count' src/ttnn/device.cpp` returns no matches in the shipped source. `find include/iom/ttnn/test_counter.hpp` returns no path in the shipped source.
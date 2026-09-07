# Submit rollback reclaims sequence without reclaiming pending_failures_

**Order:** 08
**Priority:** P1
**Blocked by:** 01-CC-003-remove-stale-llama-model-test
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** CC-002
**Review area:** Contract & correctness
**Review severity:** medium
**Review verification:** strongly-supported, confidence 80
**Review scope:** whole-codebase
**Backend scope:** common
**Location:** `include/iom/iom.hpp:322-341` (`DeviceOps::submit`; rollback condition at `:336-339`); `src/iom.cpp:557-585` (`DeviceOps::complete`; pending takeover at `:561-565`); `src/iom.cpp:592-625` (`DeviceOps::commit_failure`)

## Outcome

`DeviceOps::submit`'s rollback restores the queue to its exact pre-submission state: when a sequence is reclaimed after a synchronous `queue_work` throw, its retained `pending_failures_` entry is erased, so a stale post-enqueue failure can never be attributed to reissued work, and no abandoned pending-failure entry survives for the queue's lifetime.

## Current problem

A failure retained for a submission is observable exactly for that submission's token and only while the submission exists; rollback must restore the queue to the exact pre-submission state, so a reclaimed sequence carries no stale failure into its reissue.

`submit` (`include/iom/iom.hpp:322-341`) reserves N, calls `queue_work(N)`, and on a synchronous throw rolls back iff `next_sequence_ == N + 1 && completed_ < N` (`:336-339`). `commit_failure(N, f)` (`src/iom.cpp:592-625`) records `pending_failures_[N]` without touching `next_sequence_` or `completed_`. If a backend's `queue_work` calls `commit_failure(N, f)` and then throws, the rollback condition still holds (`completed_ < N`), so `submit` restores `next_sequence_` to N while `pending_failures_[N]` survives. The next submission reuses N; its later `complete(N)` (`src/iom.cpp:561-565`) takes the stale pending failure over the new result — even a success or a different failure — and `wait()` on the healthy new token permanently rethrows the aborted submission's failure (misattribution). If rollback cannot occur because a later submission already reserved N+1, the pending entry and its `exception_ptr` persist for the queue's lifetime with no token ever able to observe it (invisible-failure leak).

The header comment (`iom.hpp:310-321`) documents the complete-then-throw exemption but says nothing about commit-failure-then-throw. No concrete in-tree backend triggers the path today: CUDA/ROCm `execute` call `commit_failure` as their final statement followed only by a noexcept publish (`src/cuda/copy.cu:316`, `src/rocm/copy.hip:315`), and no other backend calls `commit_failure`. But the protected API permits the combination, and the commit-then-throw path is untested: existing rollback tests (`test/test_iom.cpp:1924-1966`) cover only throw-before-commit and complete-then-throw, and `InlineQueue` (`test/test_iom.cpp:844-907`) has no `commit_failure_then_throw` mode.

## Scope

- Erase the reclaimed sequence's `pending_failures_` entry in `submit`'s catch block when the rollback condition holds, restoring the exact pre-submission queue state.
- Update the `submit` doc comment (`include/iom/iom.hpp:310-321`) to state the commit-failure-then-throw policy: the retained failure is lost when the sequence is reclaimed; it persists when a gap remains.
- Add `InlineQueue::Mode::commit_failure_then_throw` plus a regression test in `test/test_iom.cpp`; leave `complete`'s pending-takeover semantics and all concrete backends unchanged.

## Implementation references

- **Modify:** `include/iom/iom.hpp` — `DeviceOps::submit` (`:322-341`); the single owner of the rollback, which becomes `if (next_sequence_ == sequence + 1 && completed_ < sequence) { pending_failures_.erase(sequence); next_sequence_ = sequence; }`.
- **Read:** `src/iom.cpp` — `complete` (`:557-585`) and `commit_failure` (`:592-625`); the pending-failure lifecycle the rollback must exactly restore.
- **Tests:** `test/test_iom.cpp` — `InlineQueue` modes (`:844-907`) and existing rollback TEST_CASEs (`:1924-1966`); add the new mode and a regression case next to the rollback cases.

## Requirements

- In `submit`'s catch block, when the rollback condition holds, erase the sequence's pending failure before restoring `next_sequence_`; the exact statement is `pending_failures_.erase(sequence);` inside the existing `if` at `iom.hpp:336-339`. A rolled-back submission never returned a token, so no `wait()` can have observed its retained failure — erasing is lossless.
- Do not widen the rollback predicate with a `pending_failures_` check; that would permanently strand the sequence instead of restoring state.
- Do not change `complete`'s pending-takeover semantics (`src/iom.cpp:561-565`); CUDA/ROCm deferred post-enqueue failures depend on it.
- Regression case: first copy throws after committing a failure; second copy on the same thread succeeds and completes sequence N; `wait()` on the second token succeeds repeatedly with no rethrow; `failures_`/`pending_failures_` contain no entry for the reclaimed sequence after rollback (gap-free single-thread shape).

## Non-goals

- No caller-side branches in CUDA/ROCm/SYCL/TTNN/CPU backends; no removal of `commit_failure` from the protected API; no change to StagedWorker publish policies or `record_post_completion_failure`.

## Acceptance criteria

- [ ] With `Mode::commit_failure_then_throw`, the first `submit` throws; `wait()` on the second (successful, same-thread) token never rethrows the aborted submission's failure, including across repeated `wait()` calls.
- [ ] After rollback, no `pending_failures_`/`failures_` entry remains for the reclaimed sequence, and all existing rollback and concurrent-gap tests (`test/test_iom.cpp:1924-2010`) pass unchanged.

## Verification

`actual validation: none (read-only review); proposed gates below`.

Root-run fact (remote bv2 host, CPU-only, at review time): configure OK, but the common `iom_tests` target does not build on the reviewed HEAD because of the missing `iom/llama.hpp` include and `iom::models` test (CC-003 verified). This task is blocked by `01-CC-003-remove-stale-llama-model-test`: the `iom_tests` gate below is executable only after CC-003 lands. The implementation itself touches only `include/iom/iom.hpp` and `test/test_iom.cpp` and can proceed in the meantime.

- After 01-CC-003 lands (or with that change applied): configure with testing enabled, build `iom_tests`, and run `ctest --test-dir <build> -R '^iom_tests$' --output-on-failure`; expected: the new `commit_failure_then_throw` case and all existing DeviceOps submit/commit_failure/rollback cases pass.
- Repro shape if needed: drive `InlineQueue` in `commit_failure_then_throw` mode, then submit the second copy on the same thread and call `wait()` repeatedly.

- `ctest --test-dir <build> -R '^iom_tests$' --output-on-failure`
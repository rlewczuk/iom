# Make CUDA and ROCm asynchronous submission a transactional boundary

**Order:** 19
**Priority:** P0 — high-severity lifetime/stability defect at the asynchronous submission boundary
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-002`
**Review severity:** high
**Review verification:** strongly-supported, confidence 96

## Outcome

Once `CudaQueue::copy` or `RocmQueue::copy` reports a failure, callers either receive a valid `oid` whose repeated `wait(oid)` calls rethrow the same stored failure, or both canaries are drained synchronously before any exception escapes. Source and destination storage are never accessed through a token that the caller does not hold, and every queued backend event remains a lifetime fence until its task completes.

## Current failure

`DeviceOps::submit` (`include/iom/iom.hpp:83-100`) only commits the sequence after the backend callback returns. The CUDA copy callback launches one or more `copy_plane_kernel` planes through `launch_copy_plane` (`src/cuda/copy.cu:288-315`), records an event with `cudaEventRecord` (`src/cuda/copy.cu:438`), then inserts `{sequence, event}` into `staged_` (`src/cuda/copy.cu:440`). Any of those steps can throw after earlier kernels are already queued. The catch at `src/cuda/copy.cu:441-444` destroys only the local `event` and rethrows; `submit` does not advance the sequence, `publish_staged` (`src/cuda/copy.cu:493-500`) is never reached, and `copy` returns no `oid`, yet the stream still references source and destination storage. A second throw site is `tasks_.push_back` inside `publish_staged` (`src/cuda/copy.cu:496`), whose `try`-less call can throw `std::bad_alloc` after the common sequence has been committed, again preventing the local token from reaching the caller. `RocmQueue` repeats the same structure at `src/rocm/copy.hip:364-385` (callback), `src/rocm/copy.hip:452-458` (`publish_staged`).

The fix must treat the boundary as a single transaction. Pick one design and stay with it: **all host task-tracking state must be allocated and linked before the first backend enqueue, and publication must be non-throwing**. On any failure detected after the first possible enqueue, the implementation records the failure on the committed token (the "retained failure" branch). The synchronous-drain branch is rejected in non-goals.

## Scope

- Change the CUDA `CudaQueue::copy` callback to allocate the event and insert the task struct into `staged_` **before** launching any kernel, then mark the in-order publication in `publish_staged` non-throwing.
- Change the ROCm `RocmQueue::copy` callback with the same shape: `Task` already linked before the first `hipLaunchKernelGGL`, and `publish_staged`'s `tasks_.push_back` cannot throw the original `bad_alloc` past the local token.
- Add the **retained-failure** path so that any throw after the first kernel launch commits the local token, stores `std::current_exception()` on the sequence, and lets the caller's `wait(oid)` retrieve it. The original exception does not propagate from `copy`; it is observed only through `wait`.
- Backends must not discard an event that is the only lifetime fence for queued work; the event stays owned by the queue and is destroyed in the worker after `complete` or in the destructor drain.
- Synchronous pre-enqueue validation (`validate_copy`, `identical_window`, `plane_pairs`, `view_planes`) keeps the current behavior: throwing there consumes no sequence and modifies no destination. The handle-table calls in `submit` keep consuming no sequence on failure.
- Test the boundary on real CUDA and ROCm hardware through the repository's remote-development procedure with injectable fault points at the four points named below.

## Implementation references

- **Modify:** `include/iom/iom.hpp` — extend the `DeviceOps` base with a protected `complete(sequence, std::exception_ptr)` test seam for the retained-failure path. The existing `void complete(std::uint64_t sequence, std::exception_ptr failure = {})` at `include/iom/iom.hpp:108` and its definition at `src/iom.cpp:572-588` already accept an exception pointer; the implementer must verify that `complete` may be called from the `submit` callback (it may, because `DeviceOps::complete` is protected, not private, and the queue subclasses already extend `DeviceOps`) and that it can be reached from a CUDA/HIP catch block without holding `completion_mutex_`. If the existing implementation cannot be called from the catch path without deadlock, add a separate `commit_failure(sequence, exception_ptr)` helper on `DeviceOps` that wakes waiters through the same `failures_` map.
- **Modify:** `src/cuda/copy.cu` — `CudaQueue::copy` at `src/cuda/copy.cu:413-448`. Build the local `Task{sequence, event}` and link it into `staged_` before the `for (const PlanePair pair : pairs)` loop. Wrap only the post-link throws (a future launch failure, `cudaEventRecord` error) into the retained-failure branch by calling the protected helper with `sequence` and `std::current_exception()`; the call still returns a valid `oid`. The catch must not destroy the event once it is owned by `staged_`. `publish_staged` at `src/cuda/copy.cu:493-500` must remain non-throwing: pre-reserve a `Task` slot (e.g. add `tasks_.push_back(Task{0, nullptr})` once, then store `std::move` entries), or move entries via a non-allocating `std::deque::splice` between `staged_` and `tasks_`. Because both are members of the same queue and both guards live on the same `mutex_`, the implementer may splice `staged_` into `tasks_` without re-entering the allocator.
- **Modify:** `src/rocm/copy.hip` — `RocmQueue::copy` at `src/rocm/copy.hip:352-385`, and `publish_staged` at `src/rocm/copy.hip:452-458`, mirror the CUDA change. `RocmQueue::run`'s drain block at `src/rocm/copy.hip:488-497` already destroys queued events under `failure`; ensure the retained-failure `Task` enters the same drain.
- **Read:** `include/iom/iom.hpp:83-100` — preserve the contract that pre-enqueue failures consume no sequence and call `wait` only with the returned token. Do not change `encode_token` or `wait`; reuse them unchanged.
- **Read:** `test/backend/backend_conformance_other.hpp:42-140` — `DeferredCopyQueue` records owner addresses without any backend. The retained-failure branch must be reachable through the common `complete(sequence, exception_ptr)` interface so the existing `DeferredCopyQueue` coverage at `test/backend/backend_conformance_other.hpp:255-274` keeps verifying the same observable contract.
- **Tests:** `test/cuda/test_cuda_conformance.cpp` — add a `TEST_CASE("CUDA submission remains transactional across post-enqueue failures")` that drives the retained-failure branch.
- **Tests:** `test/rocm/test_rocm_conformance.cpp` — add a `TEST_CASE("ROCm submission remains transactional across post-enqueue failures")` with the same shape.
- **Tests:** `test/backend/backend_conformance_other.hpp:42-90` — extend `DeferredCopyQueue::copy` (or add a sibling fake that mirrors the real `CudaQueue`/`RocmQueue` boundary) so the common "post-enqueue failure must surface through `wait(oid)`" assertion is reachable from the shared CPU conformance suite and survives tests on every backend.

## Requirements

- The new submit shape in both backends is: `Task local{sequence, event}; staged_.push_back(local);` under `mutex_` **before** the first kernel launch. After `staged_` holds the entry, no allocation or stream touch inside `submit` is allowed to throw past the local token.
- The retained-failure branch is the only branch permitted to throw after the first backend enqueue. Its contract is: catch the exception, call the protected `complete(sequence, std::current_exception())` helper (or the new `commit_failure` alias), and continue as if the work had queued normally. `copy` returns the encoded token; the local exception does not propagate. Any subsequent `wait(oid)` rethrows the stored exception on every call until the queue is destroyed.
- If `cudaEventRecord` or `hipEventRecord` throws after one or more kernels have already been queued, the queue records the failure on the committed token and does not abort the stream. The event is destroyed by the worker on `complete`, exactly as it would be on success.
- The catch at `src/cuda/copy.cu:441-444` and `src/rocm/copy.hip:378-381` no longer destroys the event after the entry is already in `staged_`. If a backend ever reaches a state where the event has been created but not yet linked into `staged_` (e.g. the pre-link `cudaEventCreateWithFlags`/`hipEventCreateWithFlags` call at `src/cuda/copy.cu:430-431` / `src/rocm/copy.hip:367-368` fails), the catch destroys the event and rethrows with no consumption — that path remains valid because no kernel has been queued.
- `publish_staged` becomes non-throwing. The transfer from `staged_` to `tasks_` uses deque `splice` between sibling containers under the same lock, so `tasks_.push_back` cannot fail with `std::bad_alloc`. Insertion into `staged_` itself uses the same protection.
- Synchronous validation in `validate_copy`, `identical_window`, and `plane_pairs` keeps the existing pre-enqueue semantics: throw with no sequence consumed and no destination modification. The `hip/cuda` driver-side `cudaStreamCreateWithFlags`/`hipStreamCreateWithFlags` call in `RocmQueue::RocmQueue`/`CudaQueue::CudaQueue` constructors remains an instance-construction error and is not part of this change.
- The exception categories remain the established repository set: `std::runtime_error` from CUDA/HIP driver wrappers, `std::invalid_argument` from validation, `std::overflow_error` for sequence exhaustion. The retained-failure branch does not invent new categories; it captures `std::current_exception()` as-is.
- Worker destruction of the failed `Task` event must still happen on the worker path under `mutex_`. The destructor drains `staged_` and `tasks_` at `src/cuda/copy.cu:399-409` (and `src/rocm/copy.hip:336-343`) and remains unchanged.
- Operand canaries (host-bounded `std::vector<std::byte>` filled with a known pattern, then `copy_to_host`-ed back to host after the failed token drains) must end with the original pattern when the cleanup order is: copy fails, no further wait is performed, source/destination tensors are destroyed, then their allocator slots are reused by a fresh tensor whose `copy_to_host` reads the same canary bytes. The canary check is part of the test suite, not the runtime contract.

## Non-goals

- The synchronous-drain alternative recommended by the review ("or synchronously drain the stream before throwing"). Drain-then-throw is rejected because it would conflict with the established `oid` wait contract: a token returned under that alternative would have nothing to wait on. Retained failure is the chosen boundary.
- Cancelling queued kernels or aborting the backend stream in response to a launch failure. The stream is left to run; the caller still has a token.
- Changing `DeviceOps::submit`, `DeviceOps::wait`, `DeviceOps::complete`, `DeviceOps::encode_token`, the queue-ID pool, or the sequence bit layout.
- Adding a public "cancel" operation, cross-queue dependencies, or any implicit waiting in `~DeviceOps`.
- Replacing `cudaEvent_t` / `hipEvent_t` lifetime fences with a different fence mechanism. The event must remain the only fence for queued work.
- Promoting the retained-failure path through `add`, `mul`, `silu`, `linear`, `rmsnorm`, or `sdpa`, all of which currently reject with capability `std::runtime_error` before submission.
- New design alternatives, cross-cutting abstraction, and unrelated cleanup. The change is contained to the two `copy` callbacks and `publish_staged`.

## Acceptance criteria

- [ ] `src/cuda/copy.cu:413-448` and `src/rocm/copy.hip:352-385` build the local `Task{sequence, event}`, lock `mutex_`, and push the entry into `staged_` before invoking `launch_copy_plane` / `hipLaunchKernelGGL` for the first plane.
- [ ] Injecting a failure at the third plane launch (CUDA), at `cudaEventRecord`, at `tasks_.push_back` inside `publish_staged`, and at the equivalent HIP points still returns the encoded `oid`; the next `wait(oid)` rethrows the same stored exception; a second `wait(oid)` rethrows it again.
- [ ] A failure injected before any backend enqueue (validation, `cudaEventCreateWithFlags`/`hipEventCreateWithFlags`) consumes no sequence; the next successful `copy` returns sequence `2` (`queue->copy(x, scratch)` after a failed `copy(x, y)`).
- [ ] An asynchronous-failure path that has at least one successful kernel enqueued returns an `oid`; the caller may destroy the source and destination tensors without use-after-free, because the event remains owned by the worker and synchronizes the operands until the worker calls `complete`.
- [ ] Operand canaries (32-byte host pattern replicated to fill `tiled_storage_nbytes()` for the source and destination tensors before the failed copy, then read back through `copy_to_host` after the failed `wait` and a fresh-tensor allocation reusing the same buffers) equal the original pattern.
- [ ] `DeferredCopyQueue::copy` extended in `test/backend/backend_conformance_other.hpp` exercises the same retained-failure contract (build local task → record owner/view addresses → return token → first-store, post-enqueue failure surfaces through `wait`; pre-enqueue failure consumes no sequence). The CPU conformance executable `iom_cpu_tests --test-case="DeviceOps queue*,DeviceOps view signatures*"` still passes; the test list is unchanged in name.
- [ ] `search src/cuda/copy.cu src/rocm/copy.hip` finds no `cudaEventDestroy` / `hipEventDestroy` inside a catch block whose surrounding code already pushed the corresponding event into `staged_`. The only event-destruction paths are the constructor/destructor drain and the worker post-synchronize cleanup at `src/cuda/copy.cu:525` / `src/rocm/copy.hip:483`.

## Verification

Run the verification through the repository's remote-development procedure (`.agents/skills/remote-development`). Pick a CUDA host and a ROCm host from `.remote-hosts.conf`. Use unique task ids such as `st002-cuda` and `st002-rocm`.

For each backend:

1. `cd /home/rlew/iom/src/iom && .agents/skills/remote-development/scripts/remote-exec <host> <task> 'cmake -S . -B build -DBUILD_TESTING=ON -D<BACKEND>_ENABLED=ON -D<OTHER>_ENABLED=OFF -D<BACKEND>_PATH=<path>'` (CUDA: `-DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda`; ROCm: `-DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm`).
2. `remote-exec <host> <task> 'cmake --build build -j --target iom_<backend>_conformance_tests'`.
3. `remote-exec <host> <task> 'ctest --test-dir build --output-on-failure -R iom_<backend>_conformance_tests'`.
4. `remote-exec <host> <task> 'ctest --test-dir build --output-on-failure -R iom_backend_conformance_<backend>_tests'` (or the registered test target for this backend).
5. `remote-exec <host> <task> 'ctest --test-dir build --output-on-failure -R iom_tests'` to confirm the CPU common-base suite still passes after the boundary change.
6. `remote-exec <host> <task> 'flock /tmp/agent-gpu0.lock ./build/test/iom_cuda_conformance_tests -tc="*remains transactional across post-enqueue failures*"'` for CUDA, and the ROCm analogue `flock /tmp/agent-gpu0.lock ./build/test/iom_rocm_conformance_tests -tc="*remains transactional across post-enqueue failures*"` — the new `CUDA submission remains transactional across post-enqueue failures` / `ROCm submission remains transactional across post-enqueue failures` test cases must run under exclusive device access.
7. After verification, `remote-clean <host> <task>` removes the remote mirror.

The fault-injection tests must:

- Wrap the third plane kernel with `cudaLaunchKernel` (or `hipLaunchKernelGGL` for HIP) coverage so a forced failure (returns `cudaErrorInvalidValue` / `hipErrorInvalidValue`) lands on a path that has at least one kernel queued. The test confirms the returned `oid`'s sequence equals the next monotonic value committed and that `wait(oid)` rethrows the captured exception twice.
- Inject a failing `cudaEventRecord` (or `hipEventRecord`) after a successful kernel queue, then confirm the same retained-failure contract through `wait`.
- Cover the publication-time retained-failure path through the shared `DeferredCopyQueue`-style fake in `test/backend/backend_conformance_other.hpp` (the same `DeferredCopyQueue::copy` extension already required by this spec, with its deterministic-throwing hook standing in for `tasks_.push_back` now that publication is a non-throwing deque `splice`); confirm the call still returns the encoded `oid` whose `wait` rethrows. The real-backend `tasks_.push_back` injection itself is uninteresting on CUDA/ROCm because publication is non-throwing per the Requirements; the real-backend retained-failure coverage there focuses on the launch and event-record points below.
- Inject pre-enqueue failures (validation throws from `validate_copy` and a `cudaEventCreateWithFlags` / `hipEventCreateWithFlags` failure) and verify the sequence counter is unchanged across a subsequent successful `copy`.

Each failure case wraps the call in a try/catch, asserts the returned `oid`, then performs `wait` (synchronously or via `std::thread`) and re-asserts the same exception type and `what()` string twice across two separate `wait` invocations on the same `oid`.

The CPU conformance test (`iom_cpu_tests --test-case="DeviceOps queue*,DeviceOps view signatures*"`) must pass, proving the common base's retained-failure contract through `DeferredCopyQueue` is unchanged in shape.

After the remote runs, run `./build/test/iom_cuda_conformance_tests` / `./build/test/iom_rocm_conformance_tests` locally is **not** a substitute — the verification requires hardware. CPU-only evidence is rejected by this finding.

# Restore the no-throw event synchronize in CUDA/ROCm fence_destroy teardown cleanup

**Order:** 56
**Priority:** P0 — restores the accepted 47-ST-001 teardown-safety contract (event synchronized before destroy) that later CUDA/ROCm extraction (AR-004) and event pooling (PF-004) will build on; memory-safety/lifetime work that gates broad progress.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-002`
**Review severity:** medium
**Review verification:** verified, confidence 90

## Outcome

`cuda_fence_destroy` (`src/cuda/copy.cu`) and `hip_fence_destroy` (`src/rocm/copy.hip`) again perform a no-throw event synchronize before the single event destroy, so every CUDA/HIP event is fenced before its exactly-once destruction on every path that reaches the callback: the `StagedWorker` shutdown drain, the staged-list drain, the normal worker path (where this is the sanctioned second synchronize after `fence_complete`), and the in-`execute` post-link rollback that destroys an already-recorded event. The metadata-slot release and resource deletion inside `destroy_cuda_resource_noexcept`/`destroy_hip_resource_noexcept` stay exactly as they are. Destructor ordering, the trailing stream finalization, completion-failure routing, and all fault seams are unchanged.

## Current failure

Accepted contract (spec `docs/changes/0001-tensor-view/47-ST-001-fence-backend-queue-teardown/spec.md`, requirement 1 and its "Event cleanup primitive" section): "Every non-null pending CUDA/HIP event is synchronized before exactly one destroy call, whether it is in AR-002's staged list, task list, normal worker path, or shutdown drain," with the second synchronize in `fence_destroy` explicitly intentional because the same callback is the helper's shutdown-drain cleanup boundary.

Commit `387900b` (PF-002 metadata-slot rework) replaced the original `fence_and_destroy` (`(void)cudaEventSynchronize(event); (void)cudaEventDestroy(event);` — visible at commit `41ce130`) with a sync-less destroy that survives at HEAD:

- `src/cuda/copy.cu:622-632` — `cuda_fence_destroy` only activates the context (swallowing failures) and then calls `destroy_cuda_resource_noexcept` (`:607-614`), which destroys the event (`gpu_policy::destroy_event_noexcept`, `:84-88`), releases the metadata slot, and deletes the resource. No `cudaEventSynchronize`.
- `src/rocm/copy.hip:626-636` — `hip_fence_destroy` mirrors this via `destroy_hip_resource_noexcept` (`:611-618`). No `hipEventSynchronize`.

The drain path that depends on the callback is `iom::detail::StagedWorker<Task>::process(task, wait_for_fence)` (`include/iom/iom.hpp:136-155`): it calls `fence_complete` only when `wait_for_fence` is true, then always calls `fence_destroy` for a non-null fence. Both `drain_list` (`:157-163`) and the shutdown branch of `run()` (`:171-179`) drain with `process(..., false)`, and `shutdown_and_drain()` (`:115-133`) drains both lists that way after joining the worker. `~CudaQueue` (`src/cuda/copy.cu:695-704`) and `~RocmQueue` (`src/rocm/copy.hip:699-708`) run `invalidate_entries_for_queue` → `shutdown_and_drain()` → activate → `synchronize_and_destroy_stream(stream_)`, so during destructor-with-pending-tasks the events are destroyed while their kernels may still be in flight and the queue stream is synchronized only afterward.

The same sync-less destroy is reached from a second in-flight path: when the registry/outcome insertion throws after the event was recorded, `execute`'s catch block calls `cuda_fence_destroy(task.fence)` / `hip_fence_destroy(task.fence)` (`src/cuda/copy.cu:845-848`, `src/rocm/copy.hip:846-849`) on a recorded event whose kernel may be running.

Today this is functionally safe only because current CUDA/HIP drivers defer destruction of busy events and the destructors synchronize the stream afterward — driver behavior that is documented nowhere in the repository. The repository's own accepted P0 contract is violated, and on any runtime that does not honor deferred busy-event destruction the destructor ordering becomes unsafe. No deterministic in-process red/green unit test exists for this defect (deferred destroy masks it on current runtimes), which is why the review's verification burden is a sanitizer run over the teardown scenario plus a source audit; the always-run teardown regressions remain the exercised scenario.

## Scope

- **Modify:** `src/cuda/copy.cu` — add a no-throw event synchronize to `gpu_policy` and call it from `cuda_fence_destroy` before the unchanged `destroy_cuda_resource_noexcept` call. `cuda_fence_complete`, `fence_event`/`make_fence`, `destroy_cuda_resource_noexcept`, `~CudaQueue`, and the pre-link failure cleanup (`:803-809`) are untouched.
- **Modify:** `src/rocm/copy.hip` — the HIP mirror: no-throw synchronize in the policy, called from `hip_fence_destroy` before the unchanged `destroy_hip_resource_noexcept`. `hip_fence_complete`, `destroy_hip_resource_noexcept`, `~RocmQueue`, and the pre-link cleanup (`:807-809`) are untouched.
- **Tests:** strengthen the two existing teardown regressions so the destructor drain reliably encounters published tasks with non-null fences; no new harness, seam, or test file.
- **Affected behavior:** teardown ordering (event synchronized before destroy on all four callback caller classes); no change to error categories, token/wait semantics, completion callbacks, null-fence no-op behavior, or destructor step order.
- **Backends:** cuda, rocm only. CPU, SYCL, and TTNN have no CUDA/HIP event and are out of scope.

## Implementation references

- **Modify:** `src/cuda/copy.cu` — `gpu_policy` (`:84-91`): add `static void synchronize_event_noexcept(event_type event) noexcept` next to `destroy_event_noexcept`, body `(void)cudaEventSynchronize(event);` guarded on `event != nullptr`, mirroring the existing no-throw `record_event_no_fault` convention (`:99-102`). Then `cuda_fence_destroy` (`:622-632`): keep the null-resource early return; keep the existing `try { gpu_policy::activate(resource->context); } catch (...) {}` guard; call `gpu_policy::synchronize_event_noexcept(resource->event)` after the activation attempt (outside the try, so an activation failure cannot skip the synchronize); then call the unchanged `destroy_cuda_resource_noexcept(resource)`.
- **Modify:** `src/rocm/copy.hip` — identical shape with `hipEventSynchronize`: add `gpu_policy::synchronize_event_noexcept` beside `destroy_event_noexcept` and call it from `hip_fence_destroy` (`:626-636`) before `destroy_hip_resource_noexcept`.
- **Read:** `include/iom/iom.hpp:115-186` — `StagedWorker::shutdown_and_drain`, `process`, `drain_list`, `run`: why `fence_destroy` is the only event-cleanup boundary on drain paths and must therefore synchronize by itself.
- **Read:** `docs/changes/0001-tensor-view/47-ST-001-fence-backend-queue-teardown/spec.md` — requirement 1 and the "Event cleanup primitive" section: the accepted contract being restored, including the sanctioned normal-path double synchronize.
- **Read:** `src/cuda/copy.cu:845-857` and `src/rocm/copy.hip:846-858` — the `execute` post-link rollback that routes a recorded event through `fence_destroy`; the restored synchronize fences this path too.
- **Tests:** `test/cuda/test_cuda_conformance.cpp:483-516` — `TEST_CASE("CUDA queue destruction fences pending copies")`; `test/rocm/test_rocm_conformance.cpp:641-673` — `TEST_CASE("ROCm queue destruction fences pending copies")`. Both already submit successful copies, a `third_plane_launch` retained-failure copy, and further copies, then destroy the queue without waiting — exactly the teardown scenario; raise the queued-copy count (e.g., 64 successful copies instead of 2 + 6) so the shutdown drain reliably processes published tasks with non-null fences rather than draining an already-empty list.

## Requirements

1. `cuda_fence_destroy` and `hip_fence_destroy` synchronize every non-null event before its single destroy, no-throw: the synchronize uses the new `gpu_policy::synchronize_event_noexcept` (status discarded, no `check_cuda_kernel`/`check_hip`), never the throwing `synchronize_event` policy helper. This covers all four caller classes — worker success path (after `fence_complete`; the double synchronize is intentional and sanctioned), worker shutdown drain, staged-list drain, and the `execute` post-link rollback.
2. An activation or synchronize failure never skips cleanup: after the null-resource early return, the event destroy, metadata-slot release (`resource->pool->release(resource->metadata_slot)`), and resource delete all run unconditionally, in the existing `destroy_*_resource_noexcept` order. Metadata-slot release stays exactly once per resource.
3. Null handling preserved: a null `opaque` fence returns immediately (no runtime calls); a null event inside a resource skips only the synchronize call (the existing `destroy_event_noexcept` null guard already covers destroy).
4. Exactly-once destruction preserved: no event is destroyed twice and no new direct event-destroy site is added. The pre-link failure cleanup in `execute` (`src/cuda/copy.cu:803-809`, `src/rocm/copy.hip:804-809`) keeps destroying its unrecorded event directly via `gpu_policy::destroy_event_noexcept` without synchronize — an unrecorded event has no pending work.
5. Untouched behavior: `fence_complete` (including its throwing-failure route into the completion path), `fence_event`/`make_fence` registry fences, `StagedWorker` code, `~CudaQueue`/`~RocmQueue` step order (invalidate → `shutdown_and_drain` → activate → `synchronize_and_destroy_stream`), `SubmissionFault` seams, token/wait/retained-failure semantics, and exception categories.
6. No new callback, mutex, thread, seam, or fifth `StagedWorker` callback; no whole-device synchronization is introduced — the fence waits only its own queue's event.

## Non-goals

- Extracting `gpu_policy`, fence callbacks, pools, or metadata machinery into shared CUDA/ROCm code (review finding AR-004 owns that extraction; this fix lands per-backend first so the extraction moves already-correct code).
- Pooling CUDA/ROCm event create/destroy or removing the per-submission metadata H2D memcpy (review finding PF-004).
- Stream finalization changes in the queue destructors; `synchronize_and_destroy_stream` stays the last-resort barrier exactly as placed.
- Tensor-destructor fencing/quarantine (applied 49-ST-003; the uniform quarantine-allocation failure policy is review finding ST-004) and registry structure (review AR-002/AR-003).
- CPU, SYCL (review ST-001 / task `55-ST-001-harden-sycl-queue-lifetimes`), or TTNN queue changes.
- A new runtime failure-injection or observation seam for event synchronize (excluded by 47-ST-001 and not needed for this restore).
- Transactional-submission or bounded-wait changes (applied 19-ST-002 / 27-ST-006 contracts stay as is).

## Acceptance criteria

- [ ] Source audit: `cuda_fence_destroy` and `hip_fence_destroy` each call a no-throw `synchronize_event_noexcept` on the resource's event before the single `destroy_*_resource_noexcept` call; the synchronize is not inside the swallowed-exception try in a way an activation failure could skip; metadata-slot release and resource deletion are byte-for-byte the prior logic.
- [ ] Source audit: exactly one event cleanup boundary per native event — `grep` confirms `destroy_event_noexcept` is reached only from the two `destroy_*_resource_noexcept` bodies and the two pre-link `execute` failure branches, and no new destroy or synchronize site exists elsewhere.
- [ ] Both strengthened teardown regressions (`CUDA queue destruction fences pending copies`, `ROCm queue destruction fences pending copies`) pass with a queued-copy load that reliably leaves published non-null-fence tasks for the shutdown drain, including one retained-failure task whose event was recorded.
- [ ] The focused CUDA teardown case runs clean under Compute Sanitizer memcheck (no busy-event, invalid-handle, or leak diagnostics), and the focused ROCm teardown case runs clean under the ROCm host's memory diagnostics.
- [ ] Normal-path behavior is unchanged: a successful copy + `wait` completes once; the retained-failure token's repeated waits still rethrow; existing fault-injection cases (`event_create`, `event_record`, `third_plane_launch`) stay green on both backends.
- [ ] Common CPU conformance is untouched and unchanged (no common-header edits).

## Verification

Follow `.agents/skills/remote-development` (`remote-sync` → `remote-exec` → `remote-clean`, one task directory per backend). No CPU-only evidence substitutes for this.

- CUDA host: configure/build with `-DCUDA_ENABLED=ON`, then run the focused case and the sanitizer over it:
  - `ctest --test-dir build -R iom_cuda_conformance_tests --output-on-failure`
  - `compute-sanitizer --tool memcheck build/test/iom_cuda_conformance_tests --test-case="CUDA queue destruction fences pending copies"` — expect exit 0 with `ERROR SUMMARY: 0 errors` and no busy-event/invalid-handle records.
  - Full focused suites: `iom_cuda_conformance_tests` and `iom_cuda_smoke_tests` green, including the existing `event_create`/`event_record`/`third_plane_launch` fault-injection and transactional-failure cases.
- ROCm host: configure/build with `-DROCM_ENABLED=ON`, then:
  - `ctest --test-dir build -R iom_rocm_conformance_tests --output-on-failure`
  - Run `build/test/iom_rocm_conformance_tests --test-case="ROCm queue destruction fences pending copies"` under the host's installed ROCm memory diagnostics (e.g., the sanitizer tooling provisioned on that host); expect no busy-event or invalid-handle diagnostics.
  - Full focused suites: `iom_rocm_conformance_tests` and `iom_rocm_smoke_tests` green.
- Source audit (any host): the two acceptance audit greps above; confirm `git diff` touches only `src/cuda/copy.cu`, `src/rocm/copy.hip`, `test/cuda/test_cuda_conformance.cpp`, and `test/rocm/test_rocm_conformance.cpp`.
- Clean each remote mirror with `remote-clean` after verification.

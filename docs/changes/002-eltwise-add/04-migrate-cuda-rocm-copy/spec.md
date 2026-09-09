# Migrate CUDA/ROCm copy to the protected OID hook

**Order:** 04
**Priority:** P0 — required backend migration in the complete OID compatibility cutover; ADD work cannot start until this CUDA/ROCm path returns the common OID results.
**Blocked by:** `02-add-oid-facades`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The CUDA and ROCm copy queues use the protected backend extension point established by the common OID facade instead of overriding public `DeviceOps::copy`. One policy-templated `GpuQueue<Policy>` remains shared by both backends. Synchronous validation, allocation, runtime, and internal failures cross the public boundary only as the exact negative `OidError` categories; accepted work still returns a positive OID and preserves the existing asynchronous EventRing failure, ordering, ownership, and cleanup behavior.

## Scope

- Migrate the shared queue implementation in `src/shared/gpu_queue.hpp:36-360`, including the current public `GpuQueue::copy` body at lines 169-181, to the protected copy hook contract from task 02. The hook receives the facade's already-validated copy arguments and reservation context; it must not re-expose a public copy override or bypass common validation, OID encoding, error mapping, or lifetime registration.
- Keep the existing shared `Task`, `EventLeaseWithFailure`, `StagedWorker<Task>::PublishPolicy::Splice`, `EventRingState`, metadata pool, two-owner registry entries, fence callbacks, `execute`, `complete_task`, queue ordering mutex, and queue-destruction drain protocol. CUDA and ROCm must continue to instantiate this same queue rather than growing separate queue implementations.
- Migrate the CUDA policy/factory and translation-unit seam in `src/cuda/copy.hpp` and `src/cuda/copy.cu`, and the CUDA driver seam in `src/cuda/driver.hpp`. Preserve CUDA context activation, stream/event creation and destruction, driver/runtime allocation and copies, kernel launch checks, fault injection, and `make_queue` construction of `GpuQueue<cuda_detail::gpu_policy>`.
- Migrate the ROCm policy/factory and translation-unit seam in `src/rocm/copy.hpp` and `src/rocm/copy.hip`. Preserve ordinal activation, HIP stream/event creation and destruction, allocation and copies, kernel launch checks, fault injection, and `make_queue` construction of `GpuQueue<rocm_detail::gpu_policy>`.
- Translate failures before positive-token acceptance to the common OID categories: malformed or invalid copy input to `OidError::InvalidArgument` (`-1`), unsupported operation/spec to `OidError::Unsupported` (`-2`), checked arithmetic or sequence exhaustion to `OidError::Overflow` (`-3`), allocation or bounded-resource exhaustion to `OidError::ResourceExhausted` (`-4`), CUDA/HIP driver or runtime failures detected before acceptance to `OidError::DeviceError` (`-5`), and otherwise unclassified failures to `OidError::InternalError` (`-6`). Do not let a synchronous exception escape the public OID-returning facade.
- Keep failures discovered after work has been accepted as retained asynchronous failures on the positive token. `wait(token)` must continue to rethrow the retained failure on every repeated wait; do not convert an accepted launch, event-record, worker, or fence failure into a later negative OID or retry an accepted runtime failure.
- Preserve caller-owned source and destination storage and handles. Keep independent source/destination metadata snapshots, exact device/context or ordinal identity checks, no-op copy behavior, inline versus pooled metadata, staging and transfer pools, ordered worker submission, and cleanup/quarantine when one or both owners are destroyed.

## Implementation references

- **Modify:** `src/shared/gpu_queue.hpp:36-360` — `iom::detail::GpuQueue<Policy>`, especially the `copy` entry point, `execute`, `complete_task`, `EventLeaseWithFailure`, and destructor; move the old override behind the protected hook while retaining the shared worker, EventRing, metadata, fence, and registry machinery.
- **Read:** `src/shared/event_ring.hpp` — `EventRingState`, `Submission`, event acquisition/recording, retained completion failure, and stream-drained cleanup conventions used by the queue.
- **Read:** `src/shared/gpu_queue.hpp:187-347` — existing deferred launch/error and two-owner registration path; preserve its distinction between pre-submission cleanup and post-launch retained failure.
- **Modify:** `src/cuda/copy.hpp` — `cuda_detail::gpu_policy`, `SubmissionFault`, `check_cuda_kernel`, and `make_queue` declarations; preserve `CUcontext`, `cudaStream_t`, `cudaEvent_t`, `CUdeviceptr`, `driver_calls.ctx_set_current`, and CUDA diagnostic labels.
- **Modify:** `src/cuda/copy.cu` — `cuda_detail::consume_submission_fault`, test injection, host-region transfer helpers, and `make_queue`; keep the CUDA expansion of `standard_tiled_copy.inl` before including the shared queue.
- **Modify:** `src/cuda/driver.hpp` — `DriverCalls`, `cuda_error`, and `check_cuda`; retain the injectable primary-context/current-context seam and translate driver status through the common OID boundary rather than changing runtime ownership.
- **Modify:** `src/rocm/copy.hpp` — `rocm_detail::gpu_policy`, `SubmissionFault`, `check_hip`, and `make_queue` declarations; preserve `int` device ordinal, HIP stream/event/device-pointer operations, and ROCm diagnostics.
- **Modify:** `src/rocm/copy.hip` — `rocm_detail::consume_submission_fault`, test injection, host-region transfer helpers, and `make_queue`; keep the HIP expansion of `standard_tiled_copy.inl` before including the shared queue.
- **Read:** `src/cuda/device.cpp:92-165` and `src/rocm/device.cpp:57-117` — existing device context/ordinal activation, `RegistryState`, transfer/staging pools, and queue factory ownership; do not relocate that lifecycle into common code.
- **Read:** `test/cuda/test_cuda_conformance.cpp:293-448,450-625` and `test/rocm/test_rocm_conformance.cpp:399-604,606-786` — existing transactional fault, retained-failure, queue-destruction, pre-wait destruction, shared-operand, and quarantine analogues. CUDA's `test/cuda/test_cuda_smoke.cpp:49-144` is the driver-call probe analogue for context activation and cleanup.
- **Tests:** `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, `test/cuda/test_cuda_smoke.cpp`, and `test/rocm/test_rocm_smoke.cpp`; task 08 owns their public OID expectation cutover, while this task must leave the implementation compatible with those focused conformance cases.

## Requirements

- `GpuQueue<Policy>` must no longer implement public `DeviceOps::copy` as the backend override. Implement the protected hook required by the task-02 facade, with the facade remaining the only public copy entry point and the only place that can reserve/encode an externally visible OID.
- The protected hook must use the common facade's validated arguments and reservation context and return through the common result path. It must not duplicate common OID validation, token bit encoding, queue-ID checks, sequence exhaustion logic, or public capability queries.
- The CUDA and ROCm factories must continue to instantiate the one shared `detail::GpuQueue<gpu_policy>`. Do not copy the queue into backend-specific CUDA or ROCm classes, add a vendor-kind switch to `src/shared/gpu_queue.hpp`, or introduce global backend state.
- Preserve backend policy behavior exactly: CUDA activates with `driver_calls.ctx_set_current`, creates nonblocking CUDA streams/events, uses the driver/runtime memory and copy calls, and reports CUDA operation labels; ROCm activates with `hipSetDevice`, creates nonblocking HIP streams/events, uses HIP memory and copy calls, and reports ROCm operation labels.
- Preserve `SubmissionFault::event_create`, `SubmissionFault::third_plane_launch`, and `SubmissionFault::event_record` seams in both policies. A fault before acceptance is mapped synchronously; a fault after a submission has acquired launch/event state remains attached to that positive token and is observed through repeated `wait` calls.
- Map `std::invalid_argument`-class validation failures, `std::bad_alloc`/pool capacity failures, CUDA/HIP status failures, and unclassified exceptions to the exact categories specified in Scope. A valid copy must not become `Unsupported` merely because the driver path reports a runtime error; `Unsupported` is reserved for an unavailable operation or unsupported specification.
- A negative result from validation or any other pre-acceptance failure must leave source and destination bytes unchanged, create no accepted token, and produce no externally observable registry/event ownership. It must not turn a failed reservation into a waitable value or advance the externally visible sequence in a way that makes a negative result appear submitted.
- A positive result must retain in-order asynchronous completion, visibility, repeat waits, retained failure, and `wait` rejection for negative/zero/foreign/future/skipped/unsubmitted values as provided by the common facade. Do not make CUDA or ROCm wait semantics special.
- Keep the two owner registrations and exact-alias deduplication in `register_copy_entries`/`release_or_invalidate_entries`. Destroying a source before wait, sharing one source across queued destinations, destroying a queue with pending work, and draining a retained failure must not free or recycle caller storage while the runtime can still access it.
- Keep metadata and staging leases alive until the existing EventRing/stream completion protocol proves they are reusable. Preserve context/ordinal activation around worker execution, transfer helpers, pool cleanup, and queue destruction, including best-effort no-throw teardown.
- Do not change tensor allocation, owner handles, `TensorView` metadata, copy semantics, or caller serialization. The queue may use only the existing backend-internal metadata/staging/temporary facilities.

## Non-goals

- Do not redesign `include/iom/oid.hpp`, the common `DeviceOps` facade, token encoding, `wait`, or the protected-hook naming/signature decision owned by task 02; consume that contract.
- Do not modify the CPU, SYCL, or TTNN copy paths.
- Do not edit task-08 backend test expectations or perform the repository-wide test/documentation cutover here; the listed tests are implementation and lifetime analogues, not additional test ownership for this task.
- Do not implement ADD, any other compute operation, storage expansion, a public `add_support`/capability query, or a generic fallback subsystem.
- Do not add separate CUDA and ROCm queue classes, public fallback APIs, caller-visible options, new allocation of operands/results, or relocation/replacement of caller-owned storage.
- Do not update architecture/backend documentation or any unrelated files.

## Acceptance criteria

- [ ] `src/shared/gpu_queue.hpp` has one shared policy-templated queue and no public `copy` override; CUDA and ROCm reach it through their protected hook and their existing `make_queue` factories compile without duplicate queue logic.
- [ ] For both policies, a valid source/destination copy returns a positive OID and preserves copy bytes, in-order submission, repeated successful waits, context/ordinal activation, and caller-owned storage/handles.
- [ ] Invalid input and each applicable pre-acceptance fault produce the exact negative category (`InvalidArgument`, `Unsupported`, `Overflow`, `ResourceExhausted`, `DeviceError`, or `InternalError`), leave output and registry/event ownership unchanged, and do not produce a waitable positive token.
- [ ] CUDA and ROCm `event_create`/pre-launch allocation or runtime failures are synchronous negative results, while `third_plane_launch` and `event_record` failures after acceptance return positive OIDs whose `wait` throws the retained failure repeatedly; no accepted failure is retried or downgraded to a negative OID.
- [ ] Existing CUDA and ROCm lifetime analogues pass after the OID test cutover: pre-wait owner destruction preserves the copy, shared-source copies release only after both completions, queue destruction fences pending work, and retained failed entries remain quarantined until device teardown.
- [ ] Metadata inline/pooled boundaries, staging and transfer pool cleanup, CUDA driver-call probes, ROCm ordinal activation, and both-owner registry cleanup remain covered without introducing vendor logic into the shared queue.

## Verification

- `ctest --test-dir <configured CUDA build> -R '^iom_cuda_(smoke|conformance)_tests$'` — run on a configured CUDA accelerator host; not run for this specification task.
- `ctest --test-dir <configured ROCm build> -R '^iom_rocm_(smoke|conformance)_tests$'` — run on a configured ROCm accelerator host; not run for this specification task.
- Focused cases in those targets must include transactional pre-acceptance faults, retained post-acceptance launch/event failures, repeated waits, metadata boundaries, queue teardown, pre-wait destruction, shared-operand lifetime, and driver/context or ordinal activation. No gates are run while generating this frozen mini-specification.

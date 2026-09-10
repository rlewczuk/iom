# Register CUDA and ROCm copy owners before launch

**Order:** 01
**Priority:** P0 — prevents rejected operations from launching effects or leaving in-flight storage untracked
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** ST-002
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 88
**Review scope:** whole-codebase
**Backend scope:** CUDA and ROCm/HIP through the shared `GpuQueue<Policy>` path
**Location:** `src/shared/gpu_queue.hpp` — `GpuQueue::execute` copy path at the metadata-copy/kernel/event phase and `register_copy_entries` phase; `include/iom/detail/outstanding_work_registry.hpp` — copy owner registration and rollback

## Outcome

CUDA and ROCm copy requests establish both caller-owner registrations and the completion outcome before any native metadata transfer, kernel launch, or event-producing effect. A failure before launch is a negative, effect-free rejection with sequence and registry rollback; a failure after launch is a positive retained failure whose owners and pooled resources remain protected until an event or covering stream drain proves completion.

## Current problem

The invariant is that pre-acceptance failure performs no native work, mutates no output, consumes no accepted sequence, and leaves no owner untracked; after launch, both source and destination owners must remain registered until completion is proven. `GpuQueue::execute` currently acquires metadata/event resources, copies metadata, launches the grid-stride kernel, checks launch status, and records an event before calling `register_copy_entries` and inserting the completion outcome. `OutstandingWorkRegistry::register_entry`/`register_registry_entries` and outcome insertion can throw allocation or ID-exhaustion failures. The catch removes partial entries, calls `on_worker_destroy`, and rethrows through `StagedWorker::submit_copy`, so the public copy can return a negative OID after native work has already changed the destination. If event recording and stream synchronization both fail, removing the entries also leaves in-flight storage without registry protection. Existing event/launch fault tests do not inject registration or outcome allocation failure after launch. The supplied focused CPU probe and remote CUDA/ROCm smoke/conformance/coexistence runs do not exercise this transaction seam.

## Scope

- Move copy owner registration and outcome reservation into one bounded pre-launch transaction shared by CUDA and ROCm, before metadata transfer, kernel launch, or event recording that can produce native effects.
- Preserve distinct source/destination owner registration, exact alias handling, context activation, event-ring/metadata retirement, positive retained post-acceptance failures, and no-op copy behavior.
- Once native work has begun, do not return a synchronous rejection: retain the failure on a positive token and protect owners until completion is proven. The combined event-record/fallback-record/stream-sync no-proof branch must remain quarantined rather than being described as always synchronized.

## Implementation references

- **Modify:** `src/shared/gpu_queue.hpp` — `GpuQueue::copy_impl`, `execute`, and `complete_task`; own the pre-launch transaction and post-launch retained-failure transition.
- **Read:** `include/iom/detail/outstanding_work_registry.hpp` — `register_entry`, `register_registry_entries`, and `register_copy_entries`; reuse the established rollback semantics for both copy owners without changing registry policy.
- **Read:** `include/iom/iom.hpp` — `submit_binary` owner/sequence transaction; this is the existing pre-acceptance registration counterpart, not a new protocol.
- **Read:** `src/cuda/copy.hpp` and `src/rocm/copy.hpp` — event-record and stream-synchronization policy seams; preserve backend-specific context and diagnostics.
- **Tests:** `test/cuda/test_cuda_conformance.cpp` transactional/teardown cases and `test/rocm/test_rocm_conformance.cpp` analogous cases; add deterministic registration/outcome allocation and combined no-proof scenarios.

## Requirements

- Reserve all bounded metadata/event state, register source and destination owners, and reserve the completion outcome before any native copy effect. Every pre-launch exception rolls back entries, outcome, leases, and sequence and performs no launch or output mutation.
- After launch, represent every failure with the accepted positive sequence and retained error. Keep both owners and pooled metadata protected until successful recorded-event synchronization or a covering stream drain proves completion; retain unknown completion when proof is unavailable.
- Use one shared copy transaction for CUDA and ROCm. Do not replace policy-specific event/stream behavior, add a synchronous wait to successful copies, or alter token/error diagnostics.

## Non-goals

- Do not change common validation, token encoding, kernel arithmetic, metadata layout, event-ring sizing, allocator policy, or unrelated binary/SYCL/TTNN paths.
- Do not reopen the separate TTNN completion-proof task or general metadata-retirement task; only the copy acceptance boundary and its direct cleanup path are in scope.
- Do not claim a throughput improvement without measurement.

## Acceptance criteria

- [ ] A deterministic pre-launch registry/outcome allocation fault returns the prescribed negative OID, leaves source and destination unchanged, consumes no accepted sequence, launches no native work, and leaves no registry entries; the next valid request receives sequence one.
- [ ] Any runtime/event failure after native launch remains attached to a positive token whose wait rethrows the same failure repeatedly, while both owners remain protected until an event or covering stream drain proves completion.
- [ ] Combined event-record and stream-synchronization failure does not release or recycle in-flight owner storage; a later successful covering drain or teardown releases it exactly once.
- [ ] CUDA and ROCm exhibit the same transaction boundary while retaining backend-local context activation, no-op behavior, and diagnostics.

## Verification

- `(remote-development: CUDA host)` build the CUDA conformance targets and run the focused registration/outcome fault scenarios plus `compute-sanitizer --tool memcheck --tool racecheck`; assert negative pre-launch rejection and positive post-launch retained failure/owner protection.
- `(remote-development: ROCm host)` build and run the corresponding ROCm conformance/fault scenarios with the configured HIP memory/race instrumentation or ROCm trace; assert no owner reuse before completion and successful later work.
- Add throwaway fault seams that separately fail the pre-launch registry/outcome reservation and the post-launch event/fallback/stream-completion path, then run a healthy copy after each failed request. No candidate-specific fault injection, sanitizer, profiler, or benchmark has run yet; root-supplied local CPU and remote smoke/conformance/coexistence gates passed but are not proof of these seams.

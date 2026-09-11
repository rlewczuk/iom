# Split shared GPU queue implementation

**Order:** 20
**Priority:** P1 — factors the oversized CUDA/ROCm queue template while preserving one declaration
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Factor the shared CUDA/ROCm `iom::detail::GpuQueue<Policy>` template into responsibility-complete included implementation fragments without changing its declaration, visibility, instantiation model, or behavior. The header retains one queue declaration and the complete nested type/state contract; lifecycle members and operation members move to two private `.inl` fragments included by that header. CUDA and ROCm continue to instantiate the same policy-templated queue, with identical asynchronous ordering, ownership, failure, quarantine, and teardown semantics.

## Scope

- Retain one `GpuQueue<Policy>` declaration in `src/shared/gpu_queue.hpp`.
- Retain the nested `MetadataLease`, `Task`, `EventLeaseWithFailure`, and `GpuOutcome` types, including `GpuOutcome::workspace_lease` (current lines 48–127).
- Retain the `IOM_ENABLE_TESTING` accessors (current lines 627–635) and all queue fields (current lines 637–649) in the declaration.
- Add the planned `src/shared/gpu_queue_lifecycle.inl` for out-of-class fence, callback, constructor, and destructor members (current implementation lines 129–300).
- Add the planned `src/shared/gpu_queue_operations.inl` for out-of-class copy/binary submission, rollback, execute, and completion members (current implementation lines 301–626).
- Include both planned `.inl` files at the bottom of `src/shared/gpu_queue.hpp`, after the class declaration and its namespace close as required by the chosen out-of-class definition form.
- Add both planned `.inl` files to the CUDA and ROCm source lists in `CMakeLists.txt` (the `iom_cuda` list around lines 147–160 and the `iom_rocm` list around lines 195–208).
- Keep the change limited to source factoring and build-list integration. No production code, public headers, tests, test source lists, or backend policy declarations are redesigned.

## Implementation references

- **Modify:** `src/shared/gpu_queue.hpp` — `iom::detail::GpuQueue<Policy>` declaration, nested leases/tasks/outcomes, testing accessors, and fields. Preserve the declaration's private visibility and dependent names exactly as required by CUDA and ROCm template instantiation.
- **Add (planned):** `src/shared/gpu_queue_lifecycle.inl` — out-of-class definitions of `GpuQueue<Policy>::fence_invoke`, `build_fence`, `make_worker_callbacks`, `GpuQueue`, and `~GpuQueue`.
- **Add (planned):** `src/shared/gpu_queue_operations.inl` — out-of-class definitions of `copy_impl`, `binary_impl`, `rollback_copy_transaction`, `execute`, and `complete_task`.
- **Modify:** `CMakeLists.txt` — add both included implementation fragments to `iom_cuda` and `iom_rocm`; retain `src/shared/gpu_queue.hpp` and all existing backend/shared entries.
- **Consumers:** `src/cuda/copy.cu` and `src/rocm/copy.hip` include `../shared/gpu_queue.hpp` after backend expansion of `standard_tiled_copy.inl`; these are the CUDA and ROCm template-instantiation consumers that must continue to compile without an additional declaration or compatibility include.
- **Behavior consumers:** `test/cuda/test_cuda_smoke.cpp`, `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_smoke.cpp`, and `test/rocm/test_rocm_conformance.cpp` — existing smoke/conformance coverage for queue construction, copy/binary submission, completion, failure, and backend coexistence; preserve their source lists and observable behavior.
- **Related queue machinery:** `src/shared/event_ring.hpp`, `src/shared/queue_resources.hpp`, `include/iom/detail/outstanding_work_registry.hpp`, and `include/iom/iom.hpp` provide the existing event, metadata, registry, and operation contracts consumed by the queue. Reuse them; do not move or redesign them in this task.

## Requirements

1. Preserve exactly one `GpuQueue<Policy>` class declaration and exactly one definition of every moved member. The `.inl` files are private template implementation sections included by the header, not independently compiled modules or alternative queue implementations.
2. Preserve template visibility and ODR behavior for both CUDA and ROCm. Definitions must remain visible at each backend instantiation site, use the existing `Policy` and nested dependent types, and introduce no explicit-instantiation unit, duplicate declaration, alias, compatibility shim, or second queue template.
3. Move lifecycle responsibilities as a complete group: fence invocation and construction, all four worker callbacks, constructor setup ordering, and destructor drain/quarantine handling. Preserve queue-resource reservation before completion-resource creation, stream/worker startup order, constructor rollback, and destruction ordering.
4. Move operation responsibilities as a complete group: copy and binary submission, copy transaction rollback, native execution, and completion. Preserve submission serialization, FIFO worker order, no-op behavior, metadata-slot acquisition/handoff, event recording and fallback proof, registry entry ownership, retained failures, and repeatable completion outcomes.
5. Preserve `MetadataLease` and `Task` ownership/move behavior and `EventLeaseWithFailure` fence storage constraints. Preserve `GpuOutcome::workspace_lease`; binary workspace leases must continue from accepted request through outcome transfer and completion-proof release, including failure and quarantine paths.
6. Preserve all existing registry semantics. Copy and binary registrations must be rolled back on preacceptance/submission failure, released or invalidated according to the existing success/failure result, and retained when native completion is unproven. No registry or quarantine behavior may be centralized into a new global or cross-backend abstraction.
7. Preserve native and host lifetime rules: event/metadata leases, tensor registrations, workspace leases, stream and worker ownership, queue-resource partitions, and completion resources must have the same destruction and covering-proof ordering as before. Unknown completion must retain the complete unresolved lease at the Device boundary; a different queue's drain is never sufficient.
8. Preserve error categories, validation order, allocation behavior, synchronization behavior, and asynchronous in-order queue behavior. Factoring must not add waits, native allocations/frees, resource resizing, backend switches, locks, retries, or changed failure mapping.
9. Keep CUDA and ROCm policy/device-specific behavior in their existing backend policy and consumer files. Do not introduce a policy/device/common abstraction, public API, backend capability, operation/direction enum, global registry, or compatibility path.
10. Keep private visibility unchanged. Testing accessors remain available only under `IOM_ENABLE_TESTING`, and existing CUDA/ROCm smoke and conformance consumers continue to observe the same metadata pool and event ring.
11. Use the fewest responsibility-complete files and normal formatting. The resulting `src/shared/gpu_queue.hpp` is approximately 270 physical lines, `gpu_queue_lifecycle.inl` approximately 190, and `gpu_queue_operations.inl` approximately 340; every original, touched, or new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/` is at most 499 physical lines.

## Non-goals

- Do not change queue admission, parking, dispatch policy, event-ring/resource policy, workspace validation, kernels, copy/binary arithmetic, numerics, tensor/view behavior, or backend capabilities.
- Do not change public headers, public signatures, factory headers, test source lists, tests, CMake test targets, or the CUDA/ROCm policy/device declarations.
- Do not add a backend-neutral GPU abstraction, shared policy layer, global registry, backend switch, compatibility alias/shim, alternate queue declaration, explicit-instantiation translation unit, or independent `.inl` module.
- Do not redesign synchronization, ownership, failure retention, quarantine, workspace lease propagation, completion proofs, or destructor draining; only relocate definitions while preserving them.
- Do not add a permanent line-count test or modify unrelated oversized-file factoring tasks.

## Acceptance criteria

- `src/shared/gpu_queue.hpp` contains one `GpuQueue<Policy>` declaration with the specified nested types, `workspace_lease`, testing accessors, and fields; lifecycle and operations are defined exactly once in the two included planned `.inl` files.
- CUDA and ROCm compile and link their existing `copy.cu`/`copy.hip` consumers against the included fragments with no missing dependent names, ODR duplicate, visibility regression, or changed public header requirement.
- Construction and destruction preserve resource reservation/rollback, stream and worker startup, queue drain, covering completion proof, unresolved-lease quarantine, and queue-resource reclamation behavior.
- Copy and binary paths preserve FIFO order, no-op behavior, metadata and event leases, registry entry rollback/release/invalidation, retained failure and repeatable wait outcomes, workspace-lease propagation, and completion-proof release.
- No new policy/device/common abstraction, backend switch, global registry, compatibility path, synchronization, allocation behavior, capability, test, or permanent line-count test is introduced.
- The header and both new `.inl` files, plus every touched production source/header, are at most 499 physical lines after normal formatting; the planned sizes remain responsibility-complete rather than arbitrary shards.

## Verification

Do not run gates while writing this mini-spec. After implementation, use the `remote-development` workflow on matching configured CUDA and ROCm hosts. Build the corresponding backend library and the existing smoke, conformance, and coexistence targets, then run:

- CUDA: `cmake --build <cuda-build> --target iom_cuda iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests`; then `ctest --test-dir <cuda-build> --output-on-failure -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$'`.
- ROCm: `cmake --build <rocm-build> --target iom_rocm iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests`; then `ctest --test-dir <rocm-build> --output-on-failure -R '^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$'`.
- Confirm compile/link coverage for both `src/cuda/copy.cu` and `src/rocm/copy.hip`, including their `GpuQueue<Policy>` instantiations and `IOM_ENABLE_TESTING` accessor consumers.
- Run a deterministic one-time scanner over `src/shared/gpu_queue.hpp`, `src/shared/gpu_queue_lifecycle.inl`, `src/shared/gpu_queue_operations.inl`, and any production file touched by this extraction, asserting no file exceeds 499 physical lines and the stated targets are met. Task 21 performs the final repository-wide scan. This scoped scanner is verification only and must not become a permanent test.
- Inspect the final diff for exactly one queue declaration, exactly one definition of each moved member, both `.inl` entries in each backend source list, and no semantic or public-surface changes.

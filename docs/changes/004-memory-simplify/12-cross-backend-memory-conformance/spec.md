# Complete cross-backend memory conformance

**Order:** 12
**Priority:** P2 — this task closes integration, architecture documentation, topology accounting, and final evidence after the behavior in tasks 01–11 is implemented.
**Blocked by:** `09-cuda-rocm-admission`, `10-sycl-admission`, `11-host-transfer-workspace`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

All shared memory, admission, workspace, lifetime, and boundary guarantees are exercised by one backend-neutral conformance layer and by backend-native setup/allocation instrumentation. CUDA, ROCm, SYCL, CPU, and TTNN drivers report the same observable contract without pretending that their storage mechanisms are identical. `test/backend/test_backend_coexistence.cpp` proves that every enabled backend can be constructed and used in one process without a registry, global ordinal cap, or active-backend selector. Architecture and backend-contract documentation describe the final ownership model and distinguish measured evidence from unavailable topology or performance evidence.

## Scope

This task is finishing and evidence work only:

- Consolidate shared normative scenarios in `test/backend`, including setup allocation counts and sizing, address-domain checks, allocator reuse/fragmentation, rank and queue limits, admission/parking/retirement/quarantine, workspace validation and exclusivity, teardown drain, and CPU/TTNN boundary scenarios.
- Keep native allocation/setup instrumentation in each backend driver at the actual runtime allocation/free boundaries. Shared tests must not infer native allocation counts from `iom::Allocator` calls alone.
- Extend `test/backend/test_backend_coexistence.cpp` and its existing CMake target to exercise all enabled factory headers and libraries side by side, while preserving process-wide OID queue IDs and legitimate multiple `Device` objects/backends.
- Update `docs/ARCHITECTURE.md` and `docs/BACKEND_CONTRACT.md`; update README/setup text only where it is stale after the preceding tasks. Record that this repository has no `examples/` directory when auditing factory/example/tool callers.
- Record two-GPU/eight-GPU topology and performance comparisons only when matching hardware and a matching pre-change baseline exist. Unavailable topology or baseline evidence is a reported non-universal risk, never a claimed pass.

Do not implement or defer any missing behavior from tasks 01–11 here. Do not add optional features, kernels, public operation/direction enums, duplicate owner query APIs, global registries, SDK-allocation claims, or unrelated cleanup.

## Implementation references

- `test/backend/backend_conformance_common.hpp` — shared backend-neutral conformance scenarios and device/queue contract checks.
- `test/backend/backend_conformance_copy_storage.hpp` — storage, transfer, view, padding, address, and ownership observations.
- `test/backend/backend_conformance_other.hpp` — non-copy operation, validation, rank, queue, lifetime, and failure scenarios.
- `test/backend/backend_conformance_oracle.hpp` — independent standard-tiled/native storage observation; it must not test an internal copy helper against itself.
- `test/backend/backend_conformance_add.hpp` and `test/backend/backend_conformance_add_gpu.hpp` — existing shared arithmetic and GPU-specific scenario composition; extend rather than duplicate equivalent checks.
- `test/backend/test_backend_coexistence.cpp` — one-process factory/library coexistence and independent-device coverage.
- `test/CMakeLists.txt` — existing exact CTest targets `iom_scalar_add_tests`, `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, `iom_cuda_smoke_tests`, `iom_cuda_conformance_tests`, `iom_rocm_smoke_tests`, `iom_rocm_conformance_tests`, `iom_sycl_smoke_tests`, `iom_sycl_conformance_tests`, `iom_ttnn_smoke_tests`, `iom_ttnn_conformance_tests`, and `iom_backend_coexistence_tests`.
- Backend driver sources under `test/cuda`, `test/rocm`, `test/sycl`, and `test/ttnn`, plus CPU driver setup, for native allocation/setup hooks and backend-specific factory arguments.
- `include/iom/device.hpp`, backend factory headers, `include/iom/alloc.hpp`, and `src/shared/*` for exact `QueueConfig`, `DeviceMemoryConfig`, `RawWorkspace`, `RawWorkspaceView`, `WorkspaceRequirements`, queue, allocator, and completion terminology.
- `docs/ARCHITECTURE.md` and `docs/BACKEND_CONTRACT.md` for the final normative ownership, storage, factory, conformance, and enabled-hardware-test wording.

Any new helper symbol introduced solely for this conformance layer is planned and test-internal; it must not become a public API or a second query path on an owner.

## Requirements

1. **Shared requirement accounting.** The conformance driver must account for every requirement across CPU, CUDA, ROCm, SYCL, and TTNN. It must use the backend's actual factory contract: standard GPU factories receive `DeviceMemoryConfig` and `QueueConfig`, CPU retains `make_cpu_device(Allocator&, QueueConfig = {})`, and TTNN retains `make_ttnn_device(ordinal, QueueConfig = {})`. `Device::create_ops()` remains argument-free. `QueueConfig` is passed by value, immutable, defaults `max_in_flight_per_queue` to 16, and rejects zero. `DeviceMemoryConfig::tensor_arena_bytes` is required for standard GPUs.
2. **Native setup accounting.** For each standard GPU, instrument the real CUDA, ROCm, or SYCL IOM native allocation/free boundary in the backend driver. Successful default and custom `C = 1`, `C = 16`, and `C = 17` setups must show exactly two live backing allocations: one tensor data backing and one metadata backing. Metadata capacity must be exactly checked `4 * C * 512` bytes, with one device-wide `FixedSizeAllocator` spanning exactly those slots; `tensor_arena_bytes` remains separate. Setup failure, zero C, overflow, and unprovisionable resources must fail before publication and release acquired resources.
3. **No post-setup native churn.** After setup, tensor/workspace creation and destruction, operations, waits, view transforms, host transfers, retirement, queue recreation, arena exhaustion, and failure recovery must perform zero subsequent IOM native device allocation/free calls and no resource-array resizing. This is an IOM-native guarantee; SDK/vendor-internal allocations must be reported separately and must not be relabeled as IOM calls.
4. **Address domains and allocator behavior.** Observe that descriptors and fixed metadata leases address only metadata backing, while tensor and `RawWorkspace` data ranges address only tensor backing. Verify data-arena fragmentation can reject a request lacking a contiguous range despite sufficient aggregate free bytes, then verify transactional free/coalescing/reuse and stable addresses for live tensors/workspaces. Verify metadata slot reuse only follows completion proof; unknown completion quarantines the complete unresolved lease and never permits unsafe reuse.
5. **Rank and descriptor limits.** Verify rank 2 through rank 8, including tiled axes, succeeds through owner, view, transform, copy, and binary-result paths. Verify rank 9 and rank-increasing `reshape_leading` reject before allocation, registration, token acceptance, metadata upload, or kernel launch. Include the compiled descriptor-size/alignment checks against the 512-byte metadata slot. Do not flatten rank nine or apply the full-tensor limit to leading-dimension helper spans.
6. **Queues and admission.** With default and custom C, create four queues with disjoint C-slot partitions and reject a fifth before starting native queue resources; failed fifth-queue construction must roll back. Exercise C+2 valid submissions: the first C hold native credits, later requests return positive tokens while parked without native effects, and autonomous completion dispatches parked requests strictly FIFO without a wait or new submission. Verify credit/resource reuse, repeated original token history for successes and failures, no bypass by no-op/inline/no-metadata work, and quarantine when completion is unknown. Verify the limit is local to the exact `Device`: multiple devices/backends remain legal, process-wide OID queue IDs remain intact, and there is no global ordinal cap or registry.
7. **Workspace contract.** Exercise all six execution facades with borrowed `RawWorkspaceView` arguments defaulted empty only for zero-requirement paths. Verify deterministic operation-specific `DeviceOps::{add,mul,sub,div}_workspace_requirements(...)` and `TensorView::{copy_from_host,copy_to_host}_workspace_requirements()` queries do not allocate or submit. Missing, undersized, misaligned, foreign, stale, overlapping, and operand/output-overlapping workspaces reject as invalid; a positively required empty workspace is invalid; an already leased range reports resource exhaustion. Verify disjoint ranges may run concurrently, leases persist while parked and through failure, and callers retain workspace ownership through proven completion.
8. **Drain and failure semantics.** Destroy queues with executing and parked work and verify every accepted request drains FIFO to a terminal outcome, with no silent cancellation or discarded token history. Verify preacceptance allocation/snapshot failures roll back all owner registrations, sequence reservations, and workspace leases. Verify retained failures are repeatable, autonomous retirement returns only proven-safe resources, and unknown native use transfers resources to device-owned quarantine.
9. **Backend boundaries.** CPU must retain borrowed allocator ownership and host/reference semantics without a device metadata arena or fabricated native slots. TTNN must retain native per-plane tensor ownership and host-only scratch behavior, exercise rank/queue/workspace/ownership boundaries, and explicitly report vendor-internal runtime allocation behavior as unproven where it cannot be observed. Standard GPU tests must cover the two raw backing domains; no test may claim a two-allocation guarantee for TTNN.
10. **Coexistence and callers.** Extend `test/backend/test_backend_coexistence.cpp` so every enabled factory header/library participates in one process with independent devices and valid foreign-device rejection. Migrate every factory, tool, and caller to explicit capacities, `QueueConfig`, and workspaces as required by the preceding tasks; audit for stale overloads and stale six-queues-per-Device expectations. Record explicitly that no `examples/` directory exists; do not create one or add an example solely for this task.
11. **Documentation terminology.** `docs/ARCHITECTURE.md` and `docs/BACKEND_CONTRACT.md` must distinguish native backing allocation, arena suballocation, metadata slot lease, completion resource, host allocation, caller workspace, host parking, and TTNN/vendor-internal unproven boundaries. They must describe the `4 * C * 512` metadata geometry, four-live-queue limit, no global ordinal cap/registry, allocator synchronization at the owning `Device`, and CPU/TTNN storage boundaries without claiming hidden locks or universal SDK behavior. Remove obsolete six-queues-per-Device expectations; retain legitimate multiple-device and process-wide OID semantics.
12. **Topology and performance evidence.** When matching hardware exists, account for two-GPU and eight-GPU isolation, per-device reservations, and independent queue/quarantine state. Never claim unavailable topology as tested. When a matching pre-change baseline and hardware exist, record native call counts, peak reservation, first-use and warmed latency/throughput for representative copies, binaries, rank-eight broadcasts, and large transfers; investigate every repeatable warmed compute regression over 5%. Report serialized host-transfer throughput separately from compute throughput. This evidence is not a universal deterministic gate when the baseline or hardware is unavailable.

## Non-goals

- Implementing behavior assigned to tasks 01–11 or deferring a missing implementation to this final task.
- Adding kernels, optional features, public operation/direction enums, owner query APIs, allocator implementations, global registries, global ordinal limits, SDK-allocation guarantees, or a second storage model for TTNN.
- Changing arithmetic, codecs, tile layout, view semantics, supported operations, or unrelated test/documentation cleanup.
- Treating unavailable GPUs, topology, SDK instrumentation, or pre-change performance baselines as successful evidence.

## Acceptance criteria

- Shared normative conformance is centralized under `test/backend`; native setup/allocation instrumentation remains in backend drivers; no duplicate backend-specific normative suite weakens the common contract.
- CPU, CUDA, ROCm, SYCL, and TTNN each have requirement-accounting coverage, with enabled hardware tests configured to fail when hardware is unavailable rather than skip.
- The exact two standard-GPU backing allocations, checked `4 * C * 512` metadata geometry, no-post-setup-IOM-native-call rule, address-domain separation, fragmentation/reuse/stable-address behavior, rank 8/rank 9 behavior, four-queue/fifth rejection, C+2 FIFO parking and autonomous reuse/history/quarantine, workspace validation/exclusivity, queue drain, and CPU/TTNN boundaries are observable assertions.
- `test/backend/test_backend_coexistence.cpp` and `iom_backend_coexistence_tests` preserve multiple independent devices/backends, process-wide OID queue IDs, and foreign-device rejection without a global registry or ordinal cap.
- All factory/tool callers are migrated and stale six-queue expectations are removed; the audit records that no `examples/` directory exists.
- `docs/ARCHITECTURE.md`, `docs/BACKEND_CONTRACT.md`, and only stale README/setup text describe the final ownership and evidence distinctions precisely, including TTNN/vendor-internal limits.
- Two/eight-GPU and performance results are reported only with matching hardware/baseline; unavailable results are explicitly marked non-universal risk, not silently skipped or presented as universal proof.

## Verification

All commands below are proposed gates and are **not run by this task**. Accelerator commands must run through the `remote-development` workflow, and enabled hardware tests must fail rather than skip:

- Local core/CPU: `ctest --test-dir <build> -R '^(iom_scalar_add_tests|iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure`
- CUDA through `remote-development`: `ctest --test-dir <build> -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$' --output-on-failure`
- ROCm through `remote-development`: `ctest --test-dir <build> -R '^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$' --output-on-failure`
- SYCL through `remote-development`: `ctest --test-dir <build> -R '^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$' --output-on-failure`
- TTNN through `remote-development`: `ctest --test-dir <build> -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$' --output-on-failure`
- Caller/documentation audit: search all factory and workspace call sites, confirm no `examples/` directory exists, and inspect the final `test/backend/test_backend_coexistence.cpp`, `docs/ARCHITECTURE.md`, and `docs/BACKEND_CONTRACT.md` diff for stale six-queue, hidden-allocation, global-registry, or universal-topology claims.
- Environment-dependent evidence: run two/eight-GPU accounting and matched-baseline performance only when the required hardware and baseline are present; otherwise report the exact unavailable evidence and its non-universal risk. Record native calls, peak reservation, first/warm latency/throughput, any repeatable warmed compute regression over 5%, and serialized transfer throughput separately.

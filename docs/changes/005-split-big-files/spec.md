## Purpose and status

This change is the implementation-ready factoring contract for `005-split-big-files`. It reduces oversized production translation units and headers by responsibility-complete extraction only. The result is behavior-preserving: it changes source layout and build integration, not the observable library or backend contracts.

Status: approved factoring specification. The implementation must make the complete cutover described here; there are no TBDs, optional shards, compatibility paths, or design questions.

## Scope and baseline

The scope is the following twelve verified oversized production files and every extracted destination or other production file necessarily modified by the factoring:

| File | Baseline physical lines |
| --- | ---: |
| `src/iom.cpp` | 1484 |
| `src/cpu/device.cpp` | 941 |
| `src/cuda/device.cpp` | 710 |
| `src/rocm/device.cpp` | 658 |
| `src/sycl/device.cpp` | 667 |
| `src/sycl/copy.cpp` | 1189 |
| `src/ttnn/device.cpp` | 1157 |
| `src/ttnn/copy.cpp` | 657 |
| `src/shared/gpu_queue.hpp` | 651 |
| `src/shared/standard_tiled_copy.inl` | 739 |
| `include/iom/iom.hpp` | 559 |
| `include/iom/detail/outstanding_work_registry.hpp` | 920 |

No other inspected production `.cpp`/`.cu`/`.hip`/`.hpp`/`.inl` is at least 500 lines. These line references are baseline navigation; named symbols and responsibilities control if lines shift. The line rule applies after normal formatting and counts every physical line, including comments and blanks, in each original oversized file and every touched or new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/`. Tests, documentation/specifications, generated or build trees, and vendor/third-party files are excluded. Every such production file must be at most 499 lines.

Use the fewest files that are complete by responsibility and satisfy the cap. Do not create numbered or arbitrary shards, use filename tricks, or add a permanent line-count unit test. `src/iom.cpp` is removed after its definitions are moved; no compatibility translation unit remains.

## Invariants and non-goals

All moved definitions must be present exactly once, with no lost or duplicated definition. Public headers, public signatures, capabilities, errors, validation order, ownership, lifetimes, queue asynchronous ordering, repeatable waits, and numerics remain unchanged. Existing allocator ownership, registry and quarantine behavior, staging leases, context behavior, and allocation seams remain unchanged where applicable.

This is not a behavior change, API redesign, backend capability change, error-policy change, or synchronization redesign. Do not add a new backend switch, global registry, allocation behavior, cross-backend abstraction, or compatibility alias. Private headers and included `.inl` files are organization mechanisms only; they must preserve visibility and ODR behavior. Public backend headers remain unchanged.

## Exact factoring plan

### Core

The current `src/iom.cpp` blocks are:

* lines 22-221: shape/spec/layout;
* lines 227-543: view/Tensor;
* lines 548-645: workspace and the Device workspace registry;
* lines 649-1482: DeviceOps.

Within the DeviceOps block, binary validation is lines 658-956; binary facades are lines 1076-1170; binary requirement queries are lines 1172-1210 and carry `RawWorkspaceView`, `WorkspaceRequirements`, and `WorkspaceLease`; `WorkspaceValidation` is lines 1212-1275, including `address()`; neural facades are lines 1277-1327; and lifecycle is lines 1328-1482.

Move every definition and remove `src/iom.cpp`:

* `src/tensor.cpp` owns `TensorShape`, `TensorSpec`, `detail::leaf_bits`, `detail::standard_plane_slot`, `detail::standard_layout_slot`, and tensor-local tile/layout arithmetic. Target: about 260 lines.
* `src/tensor_view.cpp` owns leading-dimension/stride helpers, all `TensorView` construction, accessors, transforms, and host-transfer requirement methods, plus the `Tensor` constructor and `view()` overloads. Target: about 295 lines.
* `src/workspace.cpp` owns `RawWorkspaceView`, `RawWorkspace`, and `Device::{owns,register,unregister}_workspace`. Target: about 120 lines.
* `src/device_ops.cpp` owns only shared infrastructure: queue-ID globals, lease/release, constructor/destructor, queue/device/view common validation, shared failure invocation/mapping, `WorkspaceValidation`, token encoding, wait/fence/completion/failure retention and sequence bookkeeping, and the single `UnsupportedOperation` mapping. Target: about 305 lines. Preserve capture/acquire/rollback and completion-proof release of workspace leases, along with validation and error order.
* `src/device_ops_copy.cpp` owns copy validation and identical-window logic, the default `copy_impl`, and the public `copy` facade. Target: about 100 lines.
* `src/device_ops_binary.cpp` owns binary type/spec/view/shape validation and snapshots, the default `binary_impl`, `add`/`mul`/`sub`/`div` facades, and binary workspace requirement queries. It carries only the specified binary code and must not absorb unrelated code. Target: about 455 lines.
* `src/device_ops_neural.cpp` owns default hooks and public facades for `silu`/`linear`/`rmsnorm`/`sdpa`, grouped by operation family even though they are currently unsupported. Target: about 100 lines.

Add `src/iom_internal.hpp` as minimal backend-neutral private linkage for only shared checked arithmetic, rank constants, and `UnsupportedOperation`. It must not become a miscellaneous helper collection. Public headers and signatures are unchanged.

### CPU

The current `src/cpu/device.cpp` spans are:

* lines 23-358: transfer helpers;
* lines 363-390: `CpuDevice`;
* lines 392-401: `CpuWorkspace`;
* lines 403-616: `CpuTensor`;
* lines 618-913: `CpuQueue`;
* lines 915-939: create/factory methods.

Add private `src/cpu/device_internal.hpp`, declaring `CpuDevice` for cross-translation-unit definitions. Add private `src/cpu/transfer_helpers.hpp`, in `iom::cpu_detail`, owning bit access, tile-row copy, and one-view/two-view tile traversal helpers, inline or templated as required.

Keep `src/cpu/device.cpp` for `CpuDevice`, `CpuWorkspace`, workspace creation, and the factory. Move `CpuTensor` and `create_tensor` to `src/cpu/tensor.cpp`. Move `CpuQueue` and `create_ops` to `src/cpu/queue.cpp`.

Preserve allocator ownership, 16x16 tiled/subbyte behavior, padding, queue order, and registry/failure semantics. Preserve `CpuQueue` binary workspace-lease completion before complete. Budgets are: device 130 lines, tensor 260, queue 350, helpers 370.

### CUDA and ROCm

CUDA and ROCm use mirrored organization but not a shared implementation. Add private `src/cuda/device_internal.hpp` and `src/rocm/device_internal.hpp`, each carrying the concrete Device declaration, fields, and friends. Keep each backend's `device.cpp` for runtime/context helpers and guards, concrete Device methods/resources/arena/registry/quarantine state, and `make_*_device`. Add `device_tensor.cpp` in each backend for native-storage validation, `Workspace`/`Tensor` owner classes, and `Device::create_tensor`/`create_workspace`.

CUDA's current spans are helpers/guards 25-111 and 589-614, Device 113-432, Workspace/Tensor/create methods 434-582, and factory 617-708. ROCm's are helpers 27-85 and 554-578, Device 87-392, Workspace/Tensor/create 394-545, and factory 581-656.

Reuse existing `rocm_detail::hip_error`/`check_hip`; do not retain duplicate anonymous helpers. Keep public backend headers unchanged. Each `.cpp` and private header is at most 499 lines; expected device core plus factory is at most 480 lines, tensor is at most 220 lines, and each private header is at most 220 lines. No new CUDA/ROCm common abstraction is permitted.

### SYCL device

The current `src/sycl/device.cpp` spans are globals/enumeration/helpers 23-56, `SyclDevice` 58-386, Workspace/Tensor/create 395-546, test eligible count 550-556, and backing guard/factory 563-665.

Add private `src/sycl/device_internal.hpp`. Retain `device.cpp` for globals, enumeration, `SyclDevice` core, test eligible count, backing guard, and factory. Move Workspace/Tensor/create methods to `device_tensor.cpp`.

If materialization would exceed 499 lines, the responsibility assignment remains fixed: keep the file under the cap by moving declarations, includes, or helper definitions to the private header without changing behavior. An optional factory shard is not permitted. Targets are roughly device at most 480 lines and tensor at most 210 lines.

### SYCL copy and queue

The current `src/sycl/copy.cpp` spans are fence/fault/launch support 42-284, `SyclQueue` 286-1066 (binary helpers/execution 493-861 and regular execution 863-1039), and testing/host-transfer/factory/retained hook 1072-1187.

Add private `src/sycl/queue_support.hpp` for fence declarations, retaining `mark_completion_proven`/`completion_proven`, and private `src/sycl/queue_internal.hpp` for `SyclQueue`/`Task`/outcome declarations. Retain `copy.cpp` for fault/fence/launch support, host transfers, and testing counters/hooks. Add `queue.cpp` for queue lifecycle/submission, regular execute/completion, `make_queue`, and testing snapshot. Add `queue_binary.cpp` for binary staging arithmetic and binary execution.

Preserve context, asynchronous ordering, metadata/registry/quarantine, workspace, and allocation seams. `queue_binary.cpp` preserves accepted workspace offsets/address, host staging only, completion proof, and workspace-lease release. Expected sizes are copy at most 420 lines, queue at most 470, binary at most 455; each private header is at most 499 lines. Leave `src/sycl/add.cpp` untouched: it contains no implementation to receive.

### TTNN

The current `src/ttnn/device.cpp` spans are testing seams 35-114, dtype helpers 117-249, Device/Tensor/Workspace 251-464, fence/queue 465-1046, testing 1049-1108, and public/factory 1110-1155. The current `src/ttnn/copy.cpp` spans are testing seams 18-68, transfer helpers/API 74-418, binary 421-608, and testing API 611-654.

Add private `src/ttnn/device_internal.hpp` for the `TtnnDevice` declaration and dtype/check helpers; keep `TtnnTensor` local. Add `src/ttnn/device_types.cpp` for supported/native dtype mapping, rank/count helpers, and `ttnn_supported_data_types`.

Retain `src/ttnn/device.cpp` for `TtnnDevice`, `TtnnTensor`, `TtnnWorkspace`, create tensor/workspace, and the factory. Move `create_ops` with the queue. Add private `src/ttnn/queue_internal.hpp` for `TtnnQueue`, nested request/outcome declarations, and fence declarations; `Task` and `BinaryOutcome` retain `WorkspaceLease`. Add `src/ttnn/queue.cpp` for queue lifecycle/submission/dispatch/completion, fence helpers and fence-through-sequence, and `TtnnDevice::create_ops`, excluding `execute_copy`. The queue flow preserves submit→`Task`→outcome transfer and completion-proof release. Add `src/ttnn/queue_copy.cpp` for `TtnnQueue::execute_copy`.

Retain `src/ttnn/copy.cpp` for transfer/layout helpers, `region_from_host`/`region_to_host`, and `copy_planes`. Add `src/ttnn/binary.cpp` for the `binary_planes` template and all explicit `add`/`mul`/`sub`/`div` instantiations.

Add private `src/ttnn/testing_internal.hpp` and `src/ttnn/testing.cpp` for all existing device/copy test-seam state, fault consumers, and test API definitions, preserving `IOM_ENABLE_TESTING`. This isolates one existing responsibility and does not alter runtime behavior.

Preserve TTNN native per-plane storage, queue finish ordering, staging leases, registry/quarantine, direct host-transfer finish paths, validation/error order, and public headers. Expected sizes are types 190 lines, device 330, queue at most 440, queue_copy 180, copy 430, binary 225, and testing 270; each private header is at most 499 lines. Do not create `queue_fence.cpp` or a factory-only file.

### Public and shared headers

In `include/iom/iom.hpp`, move the complete `detail::StagedWorker` template (current lines 31-203) to `include/iom/detail/staged_worker.hpp`. Include it transitively from `iom.hpp` so existing callers retain availability. Keep `WorkspaceValidation`, `DeviceOps`, and `Block` together; `BinaryRequest` workspace fields stay with `DeviceOps`. Expected sizes are `iom.hpp` about 387 lines and `staged_worker.hpp` 175.

Preserve `include/iom/detail/outstanding_work_registry.hpp` as a small umbrella include. Move `Fence`/`FenceCaptureOps` to `include/iom/detail/fence.hpp`; `EntryRegistration`, `OutstandingWorkRegistry`, and release/quarantine outcome helpers to `include/iom/detail/outstanding_work_registry_core.hpp`; `CleanupAction`/`AllocatorCleanupAction`/`Quarantine` to `include/iom/detail/outstanding_work_cleanup.hpp` as the polymorphic cleanup/quarantine dependency boundary; and `WorkspaceLease`/`RegistryState`/binary and workspace registration/lease helpers to `include/iom/detail/workspace_registry.hpp`. New names are all under `include/iom/detail/`. Preserve types, synchronization, cleanup order, existing include path, and transitive availability. Each resulting header is estimated at most 400 lines.

Retain `src/shared/standard_tiled_copy.inl` prologue, low-level bit/tile codecs and scatter/gather kernels (current lines 1-312), plus host `launch_view_transfer`, `synchronous_transfer_impl`, and exported `synchronous_transfer` (current lines 594-739). Add `src/shared/standard_tiled_copy_metadata.inl` and move current lines 313-591: `CopyMetadataHeader`, `InlineCopyMetadata` and layout assertions, checked metadata arithmetic/layout/writers, `copy_one_tiled_word`, grid-stride bodies/kernels, and both `launch_grid_stride_copy` overloads. Include the new `.inl` at the old block location inside the existing namespace, after the low-level codecs, so private symbol visibility and include order remain unchanged. Do not add namespace/include wrappers in the new file. Expected retained/new sizes are about 461/279 lines, and both must be at most 499. `standard_tiled_add.inl` remains unchanged at 416 lines.

In `src/shared/gpu_queue.hpp`, the current nested types run through line 127, including `GpuOutcome.workspace_lease`; lifecycle is lines 129-300; operations are lines 301-624, including workspace-lease propagation and completion proof; testing is lines 626-635; and fields are lines 637-650. Retain one `GpuQueue` template declaration, its nested types, accessors, and fields. Define fence/callback/constructor/destructor members out-of-class in included `src/shared/gpu_queue_lifecycle.inl`. Define copy/binary submission, rollback, execute, and completion members in included `src/shared/gpu_queue_operations.inl`. Include both `.inl` files at the bottom of `gpu_queue.hpp`. They remain private template implementation sections, not independent modules; preserve visibility, ODR, and workspace-lease completion proof. Expected sizes are header 270 lines, lifecycle 190, and operations 340.

## Build integration

Replace `src/iom.cpp` in `IOM_SOURCES` with the seven core `.cpp` files: `tensor.cpp`, `tensor_view.cpp`, `workspace.cpp`, `device_ops.cpp`, `device_ops_copy.cpp`, `device_ops_binary.cpp`, and `device_ops_neural.cpp`. Add CPU `tensor.cpp` and `queue.cpp`. Add `staged_worker.hpp` to `IOM_HEADERS`; private core headers need not be public.

CUDA and ROCm target lists add their `device_tensor.cpp`, private device header, both retained `src/shared/standard_tiled_copy.inl` and new `src/shared/standard_tiled_copy_metadata.inl`, and shared `gpu_queue` `.inl` entries, while retaining existing copy, driver, and shared entries. The SYCL target adds `device_tensor.cpp`, `queue.cpp`, `queue_binary.cpp`, both retained `src/shared/standard_tiled_copy.inl` and new `src/shared/standard_tiled_copy_metadata.inl`, and its private headers. The TTNN target adds `device_types.cpp`, `queue.cpp`, `queue_copy.cpp`, `binary.cpp`, `testing.cpp`, and private headers, while retaining `device.cpp`, `copy.cpp`, and existing private headers.

Do not change test target source lists or public factory headers.

## Verification and test strategy

No behavior changes and no new tests are expected. Existing `iom_tests` covers shape/spec/view/workspace/DeviceOps/token/wait/failure; CPU tests and CPU conformance cover moved CPU code; every accelerator smoke, conformance, and coexistence target covers its backend. Hardware tests fail, never skip.

Require a local CPU configure:

```sh
cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
```

Build these targets: `libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests`. Run:

```sh
ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'
```

For CUDA, ROCm, SYCL, and TTNN, use configured accelerator builds and remote-development. Build each backend library plus its smoke, conformance, and coexistence targets, then run:

```sh
ctest --test-dir <backend-build> --output-on-failure -R '^(iom_<backend>_smoke_tests|iom_<backend>_conformance_tests|iom_backend_coexistence_tests)$'
```

Replace `<backend>` with each backend name. Require compile/link proof for preserved `iom.hpp` and `outstanding_work_registry.hpp` umbrellas, CUDA/ROCm template `.inl` consumers, and CUDA/ROCm/SYCL consumers of the included `standard_tiled_copy` template fragment, preserving ODR and private symbol visibility.

Use a deterministic one-time scanner over tracked production suffixes, asserting a maximum of 499 physical lines per original oversized file and every touched/new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl`. This scanner is verification only and must not become a permanent test.

## Acceptance criteria

The change is complete only when:

1. All twelve original files and every touched or new production source, header, or `.inl` are at most 499 physical lines after formatting.
2. Every moved definition exists exactly once; `src/iom.cpp` is obsolete and absent, with no compatibility translation unit.
3. Public headers, signatures, capabilities, errors, validation order, ownership, lifetimes, asynchronous queue behavior, repeatable waits, and numerics are unchanged.
4. CMake includes every specified destination and excludes no retained source; all configured builds compile and link.
5. Existing tests pass for CPU and all configured backends, including smoke, conformance, and coexistence coverage; hardware failures are failures, not skips.
6. No new backend switches, global registry, allocation behavior, cross-backend abstraction, or permanent line-count test is introduced.

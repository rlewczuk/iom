# Split CUDA device components

**Order:** 12
**Priority:** P1 — factors CUDA ownership while preserving runtime/context behavior
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Split the CUDA device implementation by responsibility without changing the CUDA backend's observable behavior. A private `CudaDevice` declaration permits the device core and native tensor/workspace owners to have separate translation units, while the public CUDA factory header, signatures, capabilities, errors, allocation seams, context behavior, and queue/resource boundaries remain unchanged. The resulting CUDA production files stay within their assigned physical-line budgets.

## Scope

Create `src/cuda/device_internal.hpp` as a private header containing the concrete `CudaDevice` declaration, fields, and friends needed by the CUDA translation units. Create `src/cuda/device_tensor.cpp` for CUDA native-storage validation, `CudaWorkspace`, `CudaTensor`, and `CudaDevice::create_tensor`/`CudaDevice::create_workspace`. Refactor `src/cuda/device.cpp` to retain the CUDA runtime/context helpers and guards, the complete `CudaDevice` core and its resources/arena/registry/quarantine state, the backing guard, and `make_cuda_device`.

Update only the CUDA `iom_cuda` target source list in `CMakeLists.txt` to include the planned `src/cuda/device_tensor.cpp` and `src/cuda/device_internal.hpp`. Retain the existing CUDA copy, driver, and shared entries. Shared fragment entries belong to their dedicated factoring tasks and must not be added here. Do not alter test source lists or public factory headers.

Use the fewest responsibility-complete files. The retained device core plus factory must be at most 480 physical lines, `src/cuda/device_tensor.cpp` at most 220 lines, and `src/cuda/device_internal.hpp` at most 220 lines; every touched or new production source/header under `src/` or `include/` must also satisfy the universal 499-line maximum after normal formatting.

## Implementation references

- **Modify:** `src/cuda/device.cpp:25-432,584-708` — retain `invalid_ordinal`, `check_arena_alloc_cuda`, `PrimaryCtxGuard`, the complete `CudaDevice` implementation, queue-resource reservation/release, data-arena allocation/release, registry and quarantine ownership, `CudaBackingGuard`, and `make_cuda_device`.
- **Move to:** `src/cuda/device_tensor.cpp` from `src/cuda/device.cpp:51-89,441-582` — retain `validate_native_storage`, `CudaWorkspace`, `CudaTensor`, and `CudaDevice::create_tensor`/`CudaDevice::create_workspace` with their existing private access and ownership boundaries.
- **Create:** `src/cuda/device_internal.hpp` — the private concrete `CudaDevice` declaration, its fields, method declarations, and required friend declarations; no public type or factory declaration moves here.
- **Modify:** `CMakeLists.txt` — the `iom_cuda` target at lines 147-160; add only `src/cuda/device_tensor.cpp` and `src/cuda/device_internal.hpp` for this task while retaining `src/cuda/device.cpp`, `src/cuda/copy.cu`, `src/cuda/copy.hpp`, `src/cuda/driver.cpp`, `src/cuda/driver.hpp`, and the current shared entries.
- **Preserve boundary:** `src/cuda/copy.hpp` — `cuda_detail::gpu_policy`, `TransferStreamPool`, `StagingSlotPool`, `make_queue`, and CUDA queue-resource/testing declarations remain available without a new cross-backend layer.
- **Preserve boundary:** `src/cuda/driver.hpp` — `DriverCalls`, allocation classification/phase seams, `allocation_attempt`, `free_attempt`, `cuda_error`, and `check_cuda` retain their existing linkage and call-site classification.
- **Public API unchanged:** `include/iom/cuda/device.hpp` — `make_cuda_device(std::uint32_t, DeviceMemoryConfig, QueueConfig)` remains the only public CUDA factory declaration and keeps its existing default argument and contract.
- **Behavior proof:** `test/cuda/test_cuda_smoke.cpp`, `test/cuda/test_cuda_conformance.cpp`, and `test/backend/test_backend_coexistence.cpp` cover CUDA setup, tensor/workspace ownership, runtime/context use, backend behavior, and coexistence.

## Requirements

1. Move only declarations and definitions required to split the stated CUDA responsibilities. Every moved definition must exist exactly once, and the private header must expose no public API or compatibility alias. Keep `CudaDevice` non-copyable and preserve the concrete fields, method signatures, access control, and `CudaTensor` friendship needed by existing definitions.
2. Keep `src/cuda/device.cpp` responsible for runtime/context helpers and guards, `CudaDevice` construction/destruction and `create_ops`, queue-resource geometry and leases, allocator bookkeeping, registry state, quarantine and retained unknown leases, backing cleanup, and factory setup. Do not move these responsibilities into the tensor translation unit or a shared abstraction.
3. Keep `src/cuda/device_tensor.cpp` responsible for `validate_native_storage`, `CudaWorkspace`, `CudaTensor`, and both create methods. Preserve native-storage validation order: the four CUDA pointer attributes (memory type, managed status, context, and device ordinal), then exact-context/ordinal/arena-range checks and the established runtime error. Interior arena addresses remain valid; no base-equality check or fallback is introduced.
4. Preserve tensor construction order and failure cleanup: base `Tensor` validation precedes device interaction; data-arena allocation precedes context activation and native validation; `cudaMemset` and `cudaDeviceSynchronize` retain their order and error mapping; rejected storage returns its arena range exactly once. Tensor destruction retains registry-aware release-or-quarantine behavior, including retained bytes when proof is absent.
5. Preserve workspace behavior: zero bytes create an empty owner without arena or native allocation; positive sizes suballocate the existing data arena, activate the owning context, validate storage, and return `std::bad_alloc` for exhaustion without a fallback. Workspace destruction must retain borrowed allocator/arena ownership and release or quarantine the exact range only under the existing lease rules.
6. Preserve primary-context and factory sequencing. `make_cuda_device` must keep `cuInit`, device-count and ordinal validation, device lookup, primary-context retain, current-context activation, data-backing allocation before metadata-backing allocation, 32-byte alignment checks, allocator construction and metadata block-count validation, and reverse-order guard dismissal/rollback. `PrimaryCtxGuard` must still release a retained primary context on setup failure.
7. Preserve `CudaDevice` teardown order and ownership/lifetime proof: transfer/staging cleanup, registry quarantine drain, retained-lease reclaim/discard, allocator destruction, data/metadata backing frees through the CUDA allocation seam, and primary-context release. Do not release arena backing while owners, workspace leases, registry entries, or unresolved quarantine actions can still reference it.
8. Preserve `activate()` before CUDA runtime/transfer operations and preserve all `src/cuda/copy.hpp` and `src/cuda/driver.hpp` boundaries, allocation classifications, error categories, validation order, asynchronous in-order queue behavior, repeatable waits/failures, and registry/quarantine semantics. This is source factoring only; no synchronization, ownership, allocation, numerics, capability, or context redesign is allowed.
9. Add the two planned CUDA files to `iom_cuda` without changing public include availability, test target source lists, or shared fragment ownership. The CUDA library and existing CUDA smoke, conformance, and coexistence consumers must compile and link with each moved definition resolved once (ODR preserved).
10. Keep all resulting production files within the stated 480/220/220 budgets and the universal 499-line cap after normal formatting. Do not use arbitrary numbered shards, line-count tricks, or a permanent line-count test.

## Non-goals

- Do not change `include/iom/cuda/device.hpp`, any public header, public signature, factory default, capability, error category, validation order, or runtime/context contract.
- Do not factor ROCm, common CUDA/ROCm code, queue templates, shared fragments, or `src/cuda/copy.cu`; those responsibilities remain in their dedicated tasks and existing boundaries.
- Do not redesign allocators, arenas, borrowed workspace leases, registry/quarantine cleanup, primary-context management, synchronization, asynchronous queues, or native allocation instrumentation.
- Do not add backend switches, global registries, cross-backend abstractions, aliases/shims, fallback storage, extra capabilities, or operation behavior.
- Do not add tests, change test source lists, or introduce a permanent line-count test. Existing hardware tests must fail rather than skip when configured hardware is unavailable or behavior is wrong.
- Do not create any file other than `src/cuda/device_internal.hpp` and `src/cuda/device_tensor.cpp` in the production split, their required parent directory, and the specified CUDA CMake list edit.

## Acceptance criteria

- `src/cuda/device.cpp` retains the specified runtime/context, `CudaDevice`, arena/resource/registry/quarantine, backing-guard, and factory responsibilities; `src/cuda/device_tensor.cpp` contains the specified validation, workspace/tensor, and create-method responsibilities; and `src/cuda/device_internal.hpp` supplies the private concrete declaration with no duplicate definitions.
- CUDA tensor and workspace construction/destruction produce the same validation, allocation, alignment, error, context activation, lease, quarantine, and cleanup outcomes as before, including zero-workspace and failed-setup paths.
- `CMakeLists.txt` adds `src/cuda/device_tensor.cpp` and `src/cuda/device_internal.hpp` to `iom_cuda`, retains all existing CUDA copy/driver/shared entries, and changes no test source list or public factory header.
- The retained device file is at most 480 lines, the tensor file at most 220, the private header at most 220, and every touched/new production file is at most 499 physical lines after formatting.
- Configured CUDA smoke, conformance, and backend-coexistence tests compile, link, and pass with existing behavior and ODR/private-visibility boundaries intact; no new test is required.

## Verification

Proposed gates (not run by this task):

- Through the `remote-development` workflow on configured CUDA hardware, build the CUDA library and all affected test executables: `cmake --build <cuda-build> --target libiom iom_cuda iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests`.
- Through the same remote workflow, run exactly: `ctest --test-dir <cuda-build> --output-on-failure -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$'`. Hardware failures are failures, not skips.
- After normal formatting, run one deterministic one-time physical-line scan over `src/cuda/device.cpp`, `src/cuda/device_tensor.cpp`, and `src/cuda/device_internal.hpp`, asserting limits of 480, 220, and 220 respectively (and therefore the universal 499 maximum). The scanner is verification only and must not be committed as a test or tool.

# Split SYCL device ownership

**Order:** 06
**Priority:** P0 — creates the concrete device linkage required by later SYCL queue translation units
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Split the oversized SYCL device implementation by ownership responsibility. A private `SyclDevice` declaration in `src/sycl/device_internal.hpp` provides the concrete device linkage needed by other private SYCL translation units, while `src/sycl/device.cpp` remains the owner of device setup, runtime state, resource bookkeeping, teardown, and factory behavior. `src/sycl/device_tensor.cpp` owns the SYCL tensor and raw-workspace owner classes and their creation methods. The split changes only translation-unit organization: public APIs, runtime behavior, and all ownership and lifetime guarantees remain unchanged.

## Scope

- Add the planned private header `src/sycl/device_internal.hpp` with the concrete `iom::SyclDevice` declaration, its existing private state and friend/access relationships needed by the retained device definitions and the tensor translation unit. This header is not a public API.
- Retain in `src/sycl/device.cpp` the SYCL globals, device enumeration, and ordinal error helper at `:23-56`; the complete `SyclDevice` core at `:58-386`; `sycl_detail::eligible_device_count` at `:550-556`; and the backing guard plus `make_sycl_device` factory at `:563-665`.
- Move the complete `SyclWorkspace` and `SyclTensor` owner classes, `SyclDevice::create_tensor`, and `SyclDevice::create_workspace` from `src/sycl/device.cpp:395-546` to the planned `src/sycl/device_tensor.cpp`. Definitions must occur exactly once after the move.
- Update the SYCL `iom_sycl` source list in `CMakeLists.txt:236-269` to include `src/sycl/device_tensor.cpp` and the private `src/sycl/device_internal.hpp`, while retaining the existing SYCL sources and headers. Do not alter test target source lists or public factory headers.
- Keep the public contract in `include/iom/sycl/device.hpp` unchanged, including `make_sycl_device(std::uint32_t, DeviceMemoryConfig, QueueConfig)` and its default queue configuration.

## Implementation references

- `src/sycl/device.cpp:23-56` — `iom::sycl_detail` globals, `eligible_devices`, and `invalid_ordinal`.
- `src/sycl/device.cpp:58-386` — `SyclDevice` construction/destruction, backend identity, queue-resource provider implementation, context and transfer-queue access, registry state, arena allocation/release, retained-lease handling, backing accessors, fields, and `SyclTensor` friendship.
- `src/sycl/device.cpp:395-419` — `SyclWorkspace`, including raw-workspace registration and release through `SyclDevice`.
- `src/sycl/device.cpp:421-509` — `SyclTensor`, including data-arena allocation, owning-context/native-device validation, eager zeroing, registry quarantine/release cleanup, and host-region transfers through the existing SYCL staging/transfer helpers.
- `src/sycl/device.cpp:511-546` — `SyclDevice::create_tensor` and `SyclDevice::create_workspace`, including zero-byte workspace ownership, arena exhaustion behavior, pointer validation, and rollback.
- `src/sycl/device.cpp:550-556` and `:563-665` — eligible-device test count and backing guard/factory; these remain in `device.cpp`.
- `src/sycl/device.cpp:1-20` — existing public/private includes and dependency boundaries. Rehome declarations, includes, or helper definitions to the private header only when needed to satisfy the line budget; do not alter responsibilities or introduce a general helper header.
- `src/sycl/copy.hpp`, `src/sycl/runtime.hpp`, `src/sycl/staging_pool.hpp`, and `src/shared/standard_tiled_copy.hpp` — preserve the existing private SYCL/runtime and transfer dependencies and their visibility.
- `include/iom/sycl/device.hpp` — public factory declaration that must remain source- and signature-compatible.
- `CMakeLists.txt:236-269` — `iom_sycl` target integration point.
- `test/sycl/test_sycl_smoke.cpp`, `test/sycl/test_sycl_conformance.cpp`, and `test/backend/test_backend_coexistence.cpp` — existing behavior proof for SYCL setup, tensors, workspaces, teardown, and backend coexistence; no new tests are part of this factoring.

## Requirements

1. Keep `SyclDevice` as the one concrete owner of its exact SYCL device and context, eager in-order transfer queue, staging pool, data and metadata allocators/backings, registry state, queue-resource bookkeeping, retained leases, and teardown sequence. The private declaration must preserve non-copyable/non-movable ownership and all existing access/friend relationships.
2. Preserve device enumeration order and filtering (GPU and accelerator devices only), ordinal validation, exception categories, and validation order. `eligible_device_count` must continue to observe the same enumeration helper, and `make_sycl_device` must retain its checked arena sizing, exact-context backing allocations, alignment checks, allocator construction, metadata-capacity check, and reverse-order rollback behavior.
3. Move `SyclWorkspace` without changing raw-workspace registration identity, zero-byte behavior, address/size ownership, destructor release, or the retained workspace-lease and quarantine protocol. It must continue to release through the owning device boundary and never bypass registry checks.
4. Move `SyclTensor` without changing tensor metadata construction, data-arena allocation, owning-context and native-device pointer validation, eager in-order zero initialization, exception rollback, host-region transfer calls, or destructor release-versus-quarantine behavior. Preserve storage addresses, allocation boundaries, and all allocator cleanup failure handling.
5. Keep `create_tensor` and `create_workspace` as the same `SyclDevice` virtual overrides with the same result types, allocation paths, error behavior, and ownership. The tensor translation unit may use private accessors declared by `SyclDevice`, but must not add a second registry, allocator, context, or factory path.
6. If retaining all assigned definitions in `device.cpp` would exceed 499 physical lines, move only declarations, includes, or helper definitions to `device_internal.hpp` until the budget is met. Do not move any assigned responsibility out of `device.cpp`, create a factory-only shard, or introduce an unrelated abstraction. Target approximately at most 480 lines for `device.cpp` and 210 lines for `device_tensor.cpp`; every touched/new production source or private header under `src/` and `include/` must be at most 499 physical lines after normal formatting.
7. Update only the `iom_sycl` target integration needed for this split: add the planned tensor source and private device header, retain all existing SYCL source/header entries, and do not change public include paths, public factory headers, tests, backend switches, or unrelated targets.
8. Preserve all universal factoring invariants: public headers/signatures/capabilities, error categories and validation order, ownership/lifetimes, allocation behavior, asynchronous in-order queues, repeatable waits/failures, registry/quarantine semantics, workspace/staging lease completion proof, context behavior, numerics, private visibility, and ODR. Every moved definition must be present exactly once.

## Non-goals

- Moving or redesigning `SyclQueue`, `Task`, queue submission/completion, binary execution, fence handling, or any implementation from `src/sycl/copy.cpp`; queue implementation remains outside this device factoring.
- Adding a factory shard, compatibility alias/shim, backend switch, global registry, cross-backend abstraction, synchronization redesign, or extra SYCL capability.
- Changing `include/iom/sycl/device.hpp`, any other public header, public signature, allocator/backing policy, arena geometry, registry/quarantine semantics, staging policy, transfer behavior, validation/error policy, or numerical behavior.
- Adding or changing tests, test source lists, CMake test targets, permanent line-count tests, or production behavior.
- Running builds, tests, linters, formatters, or validation gates while writing this specification.

## Acceptance criteria

- Planned `src/sycl/device_internal.hpp` declares the concrete `SyclDevice` needed for private cross-translation-unit linkage, and planned `src/sycl/device_tensor.cpp` contains exactly the `SyclWorkspace`, `SyclTensor`, `create_tensor`, and `create_workspace` responsibilities assigned above.
- `src/sycl/device.cpp` retains globals/enumeration/helpers, the complete device core, eligible count, backing guard, and factory, with no duplicate or orphaned definitions and no factory-only translation unit.
- `CMakeLists.txt:236-269` lists `src/sycl/device_tensor.cpp` and `src/sycl/device_internal.hpp` in `iom_sycl` while retaining the existing SYCL entries; `include/iom/sycl/device.hpp` and all test source lists are unchanged.
- Every touched or newly planned production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/` is at most 499 physical lines after normal formatting, with the device/tensor targets staying within their stated approximate budgets.
- The refactored implementation preserves exact context, arena/allocator and backing cleanup, storage and host transfers, registry/quarantine and retained-lease completion, validation/error order, ownership/lifetimes, ODR, and private visibility. Existing SYCL smoke, conformance, and coexistence behavior remains passing.

## Verification

The following are proposed gates for the future implementer and are not run by this specification writer. Accelerator build, test, and execution commands MUST use the `remote-development` workflow on the configured SYCL host. The host toolchain setup must use nounset disabled while sourcing oneAPI, and `sycl-ls` must run before the tests:

```sh
.agents/skills/remote-development/scripts/remote-sync sycl 005-split-big-files-06
.agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-06 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake --build <configured-build> --target iom_sycl iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests'
.agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-06 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; ctest --test-dir <configured-build> --output-on-failure -R '\''^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$'\'''
```

Run a deterministic one-time line scanner (not a permanent test) over `src/sycl/device.cpp`, `src/sycl/device_tensor.cpp`, `src/sycl/device_internal.hpp`, and any other production file touched by this task, asserting each has no more than 499 physical lines and the stated device/tensor targets are met. Task 21 performs the final repository-wide scan. Also inspect the final diff to confirm that each moved symbol is defined exactly once and that the public SYCL header and test source lists were not changed.

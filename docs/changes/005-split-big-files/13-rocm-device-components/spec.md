# Split ROCm device components

**Order:** 13
**Priority:** P1 — factors ROCm ownership and removes duplicate error helpers.
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Refactor the ROCm device implementation into responsibility-complete private components without changing the public ROCm API or runtime behavior. `RocmDevice` becomes a concrete private declaration in `src/rocm/device_internal.hpp`; `src/rocm/device.cpp` retains ROCm runtime/current-device behavior, the Device core and its resource, arena, registry, and quarantine ownership, setup backing guards, and `make_rocm_device`; and planned `src/rocm/device_tensor.cpp` owns native-storage validation, `RocmWorkspace`, `RocmTensor`, and the Device creation methods. The ROCm target lists both new files. The duplicate anonymous `hip_error` and `check_hip` definitions are removed and every ROCm device-side caller reuses `iom::rocm_detail::hip_error` and `iom::rocm_detail::check_hip` from `src/rocm/copy.hpp:49-58`.

## Scope

- Add the private planned header `src/rocm/device_internal.hpp` and move the concrete `RocmDevice` declaration there, including its public/private methods, fields, nested retained-lease state, and `RocmTensor` friendship. Keep the declaration private to the ROCm implementation; do not expose it through any public header.
- Add planned `src/rocm/device_tensor.cpp`. Move the native-storage validator currently at `src/rocm/device.cpp:66-85`, the complete `RocmWorkspace` owner currently at `:401-425`, the complete `RocmTensor` owner currently at `:427-510`, and `RocmDevice::create_tensor`/`RocmDevice::create_workspace` currently at `:512-545`.
- Retain `src/rocm/device.cpp` for the ROCm runtime/current-device helpers and guards, `RocmDevice` method definitions and resource-provider implementation currently at `:87-392`, and the factory-local `RocmBackingGuard` and `make_rocm_device` currently at `:549-656`. The broad navigation span `:27-392,549-656` is retained only after the explicitly moved tensor block and duplicate helpers are removed.
- Update only the ROCm target's source/header list in `CMakeLists.txt` to include `src/rocm/device_tensor.cpp` and `src/rocm/device_internal.hpp`, while retaining `src/rocm/device.cpp` and existing ROCm sources. Shared fragments and their target-list changes are handled elsewhere.

## Implementation references

- `src/rocm/device.cpp:27-39`: anonymous duplicate `hip_error`/`check_hip`; delete these definitions. Change the remaining device-side error checks, including `check_arena_alloc_hip`, to call `rocm_detail::check_hip` as appropriate.
- `src/rocm/copy.hpp:49-58`: the authoritative inline `iom::rocm_detail::hip_error` and `check_hip` implementations. Include this existing private ROCm header where required; do not create a second helper, alias, wrapper, or common CUDA/ROCm abstraction.
- `src/rocm/device.cpp:41-85`: retain the arena-allocation error mapping and ordinal validation with their established messages and ordering; move only `validate_native_storage` (`:66-85`) to `device_tensor.cpp` and keep its HIP attribute, exact-ordinal, unmanaged-device-memory, and arena-range checks unchanged.
- `src/rocm/device.cpp:87-392`: concrete `RocmDevice` implementation. Preserve `hipSetDevice` activation, supported types, queue-resource reservation/release, retained-lease reclaim/discard, allocator locking boundaries, data/workspace release, registry state, and all fields/lifetimes exactly. Its declaration moves to the private header so tensor definitions can name the type.
- `src/rocm/device.cpp:401-545`: `RocmWorkspace`, `RocmTensor`, and `RocmDevice` creation definitions to move as one tensor/storage responsibility. Preserve zero-byte workspace ownership, arena suballocation, native-storage validation, `hipMemset` plus synchronization, release-on-construction-failure, registry-aware release-or-quarantine, host transfer delegation, and creation validation order.
- `src/rocm/device.cpp:549-578`: retain `RocmBackingGuard` unchanged in responsibility and failure cleanup. `src/rocm/device.cpp:581-656`: retain factory sizing, ordinal checks, `hipSetDevice`, backing allocation/alignment checks, allocator construction, metadata capacity validation, guard dismissal, and error order.
- `src/rocm/copy.hpp` and `src/rocm/copy.hip`: preserve the existing `gpu_policy`, transfer pools, queue factory, instrumentation seams, and private visibility. This task does not change queue implementation or shared fragments.
- `include/iom/rocm/device.hpp`: public `make_rocm_device` declaration remains unchanged.

## Requirements

1. Declare `RocmDevice final : public Device, public detail::QueueResourceProvider` in planned `src/rocm/device_internal.hpp` with the same constructor parameters, deleted copy operations, destructor declaration, overrides, helper methods, private `RetainedLease` type, and `friend class RocmTensor` as the current class. Its concrete state must include `ordinal_`, `detail::RegistryState registry_state_`, ROCm transfer/staging pools, `bookkeeping_mutex_`, data and metadata allocator owners, data/metadata backing pointers and byte count, `queue_slot_count_`, `retained_mutex_`, the fixed retained-lease array, and `retained_count_`, with the same initialization and declaration order.
2. Keep `RocmDevice` private and preserve all friendship and access boundaries needed by `RocmWorkspace`, `RocmTensor`, `detail::QueueResourceProvider`, and existing queue/resource code. Do not move the class into a public header, alter public signatures, add a public factory, or introduce compatibility aliases or shims.
3. In `device.cpp`, remove exactly the anonymous helper definitions at `:27-39`. Reuse `iom::rocm_detail::hip_error`/`check_hip` from `copy.hpp:49-58`; preserve operation strings, HIP status handling, `std::runtime_error` category, and the existing `hipErrorOutOfMemory` to `std::bad_alloc` mapping in `check_arena_alloc_hip`.
4. Preserve current-device behavior: factory setup selects the requested ordinal with `hipSetDevice`, Device activation selects that ordinal before native work, and tensor/workspace creation and host transfers retain their existing activation points. Never replace these calls with global state, a new context abstraction, or cross-backend code.
5. Preserve the native-storage validator exactly in behavior and order. `hipPointerGetAttributes` must remain the first native validation operation; unmanaged device memory must match the Device ordinal and lie within the owning data arena, including valid interior suballocations. Incompatible storage retains the established runtime error category/message.
6. Preserve allocation and alignment behavior. Tensor and positive workspace creation suballocate the existing data arena; zero-byte workspace creation performs no native or arena allocation; factory allocates exactly the data and metadata backings, checks 32-byte alignment and metadata block capacity, and rolls back through `RocmBackingGuard` in reverse acquisition order. No fallback, resizing, native allocation policy, or allocator implementation may be introduced.
7. Preserve `RocmWorkspace` ownership and lease semantics: register through `RawWorkspace` construction before a body can fail, release its exact range at destruction through `RocmDevice::release_workspace`, and defer retained ranges to Device quarantine rather than reusing them. Preserve `RocmTensor` construction cleanup, registry state capture, data release/quarantine behavior, host transfer delegation, and non-copyable ownership.
8. Preserve Device resource and lifetime semantics: allocator bookkeeping remains serialized only at the Device boundary and never across native work, waits, callbacks, or drains; registry/quarantine drains and retained queue-lease reclaim/discard precede allocator/backing destruction; asynchronous in-order queues, completion proof, repeatable waits/failures, workspace/staging leases, and cleanup order remain unchanged.
9. Keep all moved definitions exactly once and keep private symbols/ODR behavior valid across `device.cpp`, `device_tensor.cpp`, `copy.hpp`, and `copy.hip`. Include the new private header at the necessary implementation points without changing public include availability or adding circular or transitive public dependencies.
10. Keep the implementation to the fewest responsibility-complete files and line budgets after normal formatting: `src/rocm/device.cpp` at most 480 physical lines, planned `src/rocm/device_tensor.cpp` at most 220, and planned `src/rocm/device_internal.hpp` at most 220. Every touched or new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/` remains at most 499 lines.
11. Update the ROCm CMake target with both planned files and no unrelated target changes. Do not modify test source lists, public factory headers, CUDA/common abstractions, shared fragments, queue code, backend switches, global registries, or synchronization design.
12. Add no tests or permanent line-count test. Existing ROCm smoke, conformance, and coexistence tests remain the behavior proof; hardware failures must fail rather than skip.

## Non-goals

- Any public ROCm header, public signature, factory contract, capability, error category, validation order, or API availability change.
- CUDA, SYCL, CPU, TTNN, shared queue/resource machinery, `src/rocm/copy.hip`, or shared-fragment factoring beyond the ROCm target entries required here.
- Queue admission, queue scheduling, synchronization redesign, event/stream/resource allocation changes, or cross-backend abstractions.
- Changes to tensor layouts, numerics, native storage semantics, arena sizing, allocator behavior, workspace/lease contracts, registry/quarantine policy, or current-device/context behavior.
- New tests, modified test source lists, compatibility aliases/shims, duplicate error-helper wrappers, global registries, backend switches, or a permanent line-count check.

## Acceptance criteria

- `src/rocm/device_internal.hpp` contains the private concrete `RocmDevice` declaration with all required methods, fields, retained-lease state, and `RocmTensor` friendship; no public header changes expose it.
- `src/rocm/device_tensor.cpp` contains exactly the native-storage validator, `RocmWorkspace`, `RocmTensor`, and both Device creation definitions assigned to it; `src/rocm/device.cpp` contains none of those moved definitions and retains Device core/resource/arena/registry/quarantine, runtime/current-device behavior, backing guard, and factory.
- The anonymous `hip_error` and `check_hip` at `src/rocm/device.cpp:27-39` are gone. All affected calls use the single existing `iom::rocm_detail` helpers from `src/rocm/copy.hpp`, with established messages, categories, and ordering preserved.
- ROCm setup, current-device activation, alignment and arena validation, zero/positive workspace behavior, tensor initialization/cleanup, registry leases, retained quarantine, backing-guard rollback, allocator lifetime, queue-resource ownership, and public factory behavior are unchanged.
- `CMakeLists.txt` includes `src/rocm/device_tensor.cpp` and `src/rocm/device_internal.hpp` in `iom_rocm`, while retaining existing ROCm sources and making no unrelated target or public-header changes.
- After normal formatting, `device.cpp` is at most 480 lines, `device_tensor.cpp` at most 220, `device_internal.hpp` at most 220, and all touched/new production source files satisfy the universal 499-line cap. Every moved definition occurs exactly once.
- Configured ROCm smoke, conformance, and backend-coexistence targets compile, link, and pass on ROCm hardware; hardware failures are reported as failures, not skips.

## Verification

The following are proposed future-implementer gates and are not run by this specification writer. ROCm configure/build/test commands MUST use the `remote-development` workflow and a configured ROCm host; do not use ASAN on RDNA4.

- Build the configured ROCm library and its existing smoke, conformance, and coexistence targets, retaining the existing test source lists. Then run:

  ```sh
  ctest --test-dir <rocm-build> --output-on-failure -R '^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$'
  ```

- Compile/link the existing ROCm public-header and queue/template consumers, including consumers of `include/iom/rocm/device.hpp` and `src/rocm/copy.hpp`, to prove the private declaration does not alter public availability, ODR, or symbol visibility.
- Check exact definition ownership and absence of duplicate helpers with a scoped source search for `RocmDevice::create_tensor`, `RocmDevice::create_workspace`, `RocmWorkspace`, `RocmTensor`, `validate_native_storage`, `hip_error`, and `check_hip`; confirm only the planned owner files define each symbol and `device.cpp` has no anonymous copies.
- Run a one-time physical-line scanner over `src/rocm/device.cpp`, `src/rocm/device_tensor.cpp`, `src/rocm/device_internal.hpp`, and any other production file touched by this task, asserting the 480/220/220 budgets and the universal `<=499` limit. Task 21 performs the final repository-wide scan. This scoped scanner is verification only and must not be committed as a test.

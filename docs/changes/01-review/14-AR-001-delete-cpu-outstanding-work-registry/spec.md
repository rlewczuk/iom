# CPU retains unreachable outstanding-work registry and quarantine state

**Order:** 14
**Priority:** P2
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** AR-001
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** strongly-supported, confidence 98
**Review scope:** whole-codebase
**Backend scope:** cpu
**Location:** `src/cpu/registry_state.hpp:7-10 (CpuRegistryState); src/cpu/device.cpp:424-426, 445-451 (CpuDevice quarantine-drain destructor, registry_state accessor and member); src/cpu/device.cpp:472-495 (CpuTensor::~CpuTensor)`

## Outcome

No CPU production object owns an `OutstandingWorkRegistry` or `Quarantine`. Every CPU tensor destruction releases its allocator block exactly once, directly through `iom::detail::release_aligned_storage`. Because every CPU copy completes inline before `CpuQueue::copy` returns, no deferred registry protection is needed, and the accelerator-style lifetime protocol (registry locking, snapshot scans, quarantine emplacement) is removed from the CPU path entirely.

## Current problem

CPU storage must be released exactly once, and only after any operation that can reference it has finished. The current CPU implementation satisfies this trivially — `CpuQueue::copy` validates, computes the no-op decision, calls `submit`, and invokes `copy_elements(source, destination)` followed by `complete(sequence, nullptr)` inside the same synchronous submission callback (`src/cpu/device.cpp:681-700`; `DeviceOps::submit` runs `queue_work` synchronously before returning its token, `include/iom/iom.hpp:306-340`) — yet `CpuDevice` still carries accelerator-style lifetime machinery:

- `CpuRegistryState` (`src/cpu/registry_state.hpp:7-10`) holds `detail::OutstandingWorkRegistry registry` and `detail::Quarantine quarantine` — its only members.
- `CpuDevice` owns a `registry_state_` member (`src/cpu/device.cpp:450`), exposes a `registry_state()` accessor (`:445-447`), and its destructor drains the quarantine (`:424-426`).
- `CpuTensor::~CpuTensor` routes release through the quarantine path (`:472-495`): on allocator failure it emplaces an `AllocatorCleanupAction` into the quarantine, then calls `release_or_quarantine(device_.registry_state().registry, …)`.

No CPU code path can populate the registry: exhaustive scoped search found no CPU `register_entry`, `register_copy_entries`, fence, deferred worker, or outstanding-work insertion anywhere in `src/cpu` or `include/iom/cpu`. `release_or_quarantine` releases directly whenever its entry snapshot is empty (`include/iom/detail/outstanding_work_registry.hpp:519-554`) — the only case reachable under the current CPU call graph. The CPU cost is registry locking/map traversal on every tensor destruction, an exception-heavy destruction path, and a second, misleading storage-lifetime protocol from which future maintainers can infer that CPU destruction is protected by a mechanism no CPU queue uses. The benchmark contract comment independently records that the CPU `StagedWorker` path was already deleted (`test/cpu/test_cpu_bench.cpp:184-186`). Existing CPU lifetime tests already expect immediate frees and address reuse before waiting: `test/cpu/test_cpu.cpp:949-1045` (allocator free immediately after tensor reset) and `test/cpu/test_cpu.cpp:1096-1155` (both tensors free immediately after queue destruction despite unwaited tokens).

## Scope

- Delete `src/cpu/registry_state.hpp`; remove `CpuDevice::registry_state_`, its `registry_state()` accessor, and the custom `quarantine.drain()` destructor; drop the now-unused registry include/wiring.
- Remove the CPU-only `AllocatorCleanupAction`/`release_or_quarantine` path and make `CpuTensor::~CpuTensor` call `iom::detail::release_aligned_storage(allocator_, address_)` directly, preserving the exact-once, non-throwing release semantics.
- Keep `CpuQueue`'s `DeviceOps` sequence bookkeeping, submission-order mutex, inline copy, no-op behavior, and repeatable token waits unchanged.
- Preserve all accelerator registry/quarantine protocols (CUDA, ROCm, SYCL, TTNN) untouched.
- Update misleading CPU lifetime test names/comments to state that copies complete inline; keep the immediate-free and address-reuse assertions.

## Implementation references

- **Modify:** `src/cpu/device.cpp` — `CpuDevice` destructor (`:424-426`), `registry_state()` accessor (`:445-447`), `registry_state_` member (`:450`), `CpuTensor::~CpuTensor` release path (`:472-495`); replace `release_or_quarantine` with a direct `iom::detail::release_aligned_storage(allocator_, address_)` call and drop the `#include "registry_state.hpp"` at `:21`.
- **Delete:** `src/cpu/registry_state.hpp` — the entire `CpuRegistryState` type.
- **Read:** `include/iom/detail/outstanding_work_registry.hpp:519-554` — `release_or_quarantine`'s empty-snapshot branch already performs exactly the direct release being kept, so replacing the CPU path preserves behavior.
- **Read (counterparts, not in scope):** `src/cuda/copy.cu:274-337`, `src/rocm/copy.hip:273-336`, `src/sycl/copy.cpp:494-645`, `src/ttnn/device.cpp:457-511` — accelerator backends that genuinely register deferred entries and must retain `detail::RegistryState`, fences, and quarantine.
- **Tests:** `test/cpu/test_cpu.cpp:949-1045` and `:1096-1155` — immediate-free/address-recycling lifetime cases that must keep passing, with comments updated to explain inline completion; `test/cpu/test_cpu_bench.cpp:184-186` — comment documenting the already-deleted CPU `StagedWorker` path.

## Requirements

- Delete `src/cpu/registry_state.hpp` and remove every CPU production reference to `OutstandingWorkRegistry`, `Quarantine`, `release_or_quarantine`, and `registry_state`; after the change, no CPU object may own such state and no CPU path may register deferred work (the CPU invariant is: copies complete inline, so no deferred protection exists).
- `CpuTensor::~CpuTensor` MUST release its allocator block exactly once via `iom::detail::release_aligned_storage(allocator_, address_)` and MUST keep its non-throwing destruction behavior.
- Preserve current observable behavior: immediate allocator frees and address reuse in the CPU lifetime tests, queue tokens that wait and re-wait successfully, allocator-failure handling, and no-op copy consumption.
- Do not alter any accelerator registry/quarantine/fence behavior.

## Non-goals

- Do not remove `DeviceOps` queue IDs/completion state or convert CPU copies to a deferred scheduling model.
- Do not share CPU storage with accelerator backends or change tensor layout, host-transfer semantics, the allocator API, or accelerator lifetime code.
- Do not delete or weaken the immediate-free/address-reuse CPU lifetime tests.

## Acceptance criteria

- [ ] No CPU production object contains an `OutstandingWorkRegistry` or `Quarantine`; `grep` over `src/cpu` and `include/iom/cpu` finds no `registry_state`, `release_or_quarantine`, or `AllocatorCleanupAction` references and `src/cpu/registry_state.hpp` no longer exists.
- [ ] Every CPU tensor destruction releases its allocator block exactly once and directly; the existing immediate-free/address-recycling cases (`test/cpu/test_cpu.cpp:949-1045`, `:1096-1155`), queue destruction with unwaited tokens, repeated waits, no-op copies, and allocator-failure cases all pass unchanged.
- [ ] Accelerator builds and their deferred-cleanup conformance/lifetime suites (when enabled) still pass, proving untouched registry paths remain intact.

## Verification

- `actual validation: none (read-only review)`; proposed gates below.
- Remote CPU-capable Linux host (per `remote-development`): configure with `BUILD_TESTING=ON`, then `cmake --build <build> --target iom_cpu_tests iom_backend_conformance_cpu_tests` and `ctest --test-dir <build> -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure` — must retain the immediate-free/address-recycling cases, queue destruction with unwaited tokens, repeated waits, and allocator-failure cases.
- Build and run any enabled accelerator conformance/lifetime suites to confirm their registry/quarantine paths are untouched.
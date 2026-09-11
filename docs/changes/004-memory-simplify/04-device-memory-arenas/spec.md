# Reserve fixed device-memory arenas

**Order:** 04
**Priority:** P0 — the caller-selected capacity, native context, and the two backing allocations are prerequisites for every later resource, queue, tensor, and workspace change.
**Blocked by:** `01-native-allocation-instrumentation`, `02-transactional-allocators`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

Standard tiled GPU devices (CUDA, ROCm, and SYCL) reserve exactly one fixed tensor-data arena and one distinct fixed metadata arena during factory setup. The fully initialized `Device` owns both arenas and their allocators for its entire lifetime; tensor storage is stable, bounded suballocation rather than independently growing native storage, and the data arena is ready for task 05 to add raw workspaces. CPU and TTNN retain their existing storage ownership models while accepting the common queue configuration.

## Scope

This task covers the public factory/configuration cutover and fixed arena setup in `include/iom/device.hpp`, all five backend factory headers and definitions (`include/iom/{cpu,cuda,rocm,sycl,ttnn}/device.hpp` and `src/{cpu,cuda,rocm,sycl,ttnn}/device.cpp`), every factory caller and affected test, and `tools/sycl_add_staging_measure.cpp`. It includes standard-GPU tensor allocation/free bookkeeping, teardown and rollback, and focused setup/lifecycle instrumentation tests. It does not implement workspace APIs, queue partitions/events, parking/admission, host-transfer or SYCL staging removal, or any unrelated task directory.

## Implementation references

- `include/iom/device.hpp`: planned backend-neutral `QueueConfig` and `DeviceMemoryConfig`; `QueueConfig` is passed by value, has immutable `max_in_flight_per_queue` defaulting to 16, and rejects zero. `Device::create_ops()` remains argument-free.
- `include/iom/{cuda,rocm,sycl}/device.hpp` and corresponding `src/*/device.cpp`: planned standard-GPU factory signatures are `make_<gpu>_device(std::uint32_t ordinal, DeviceMemoryConfig, QueueConfig = {})`.
- `include/iom/cpu/device.hpp` / `src/cpu/device.cpp`: planned signature is `make_cpu_device(Allocator&, QueueConfig = {})`; CPU continues to borrow the caller allocator.
- `include/iom/ttnn/device.hpp` / `src/ttnn/device.cpp`: planned signature is `make_ttnn_device(std::uint32_t ordinal, QueueConfig = {})`; TTNN continues native per-plane storage.
- `include/iom/alloc.hpp` and `src/alloc.cpp`: reuse the canonical `ListAllocator` and `FixedSizeAllocator`; do not add a competing allocator.
- `tools/sycl_add_staging_measure.cpp`, backend callers, examples, and tests: migrate completely to the new signatures and explicit standard-GPU capacity; remove every legacy GPU factory overload/call.
- Native allocation instrumentation from `01-native-allocation-instrumentation` is the observation boundary for setup and tensor lifecycle tests.

## Requirements

1. `DeviceMemoryConfig` has required `tensor_arena_bytes`. Standard-GPU factories must validate nonzero capacity, capacity divisible by 32, nonzero `max_in_flight_per_queue`, checked `4 * C * 512` metadata sizing, backend size/resource limits, checked pointer/size arithmetic, and exact native context/device selection before publication. Do not round capacity, infer it from free VRAM, use a percentage default, or provide a live/per-queue override.
2. After establishing the exact native context/device, reserve exactly two native buffers: one `tensor_arena_bytes` data backing and one distinct metadata backing of checked capacity `4 * C * 512` bytes. Both bases must satisfy 32-byte alignment, and the requested data capacity must remain wholly usable; metadata is additional capacity.
3. Construct a device-owned `ListAllocator` over the data backing with alignment 32 and one device-wide `FixedSizeAllocator` over the complete metadata backing with alignment 32, payload 512, and exactly `4 * C` blocks. Publish the device only after both allocations, allocator construction, and all other setup complete under RAII.
4. Map native out-of-memory to `std::bad_alloc`, checked multiplication/size overflow to `std::overflow_error`, and invalid configuration to `std::invalid_argument`; preserve established runtime/context error categories. Roll back every partially acquired backing, allocator, stream, and context resource in reverse order while the native context remains valid, preserving the original failure.
5. CUDA/ROCm base-pointer validation must retain exact device/context identity checks while accepting proven interior addresses in the data or metadata arena; it must not require each suballocation to be the base of an independent native allocation. SYCL must apply the equivalent context/device ownership checks.
6. Tensor creation and destruction must suballocate and free only data-arena ranges under Device-owned bookkeeping locks. Never hold such a lock across native work, waits, callbacks, or worker drains. The arenas never grow, spill, relocate, or fall back; a contiguous-range/fragmentation failure is `std::bad_alloc`, and live tensor addresses remain unchanged.
7. Setup and teardown must be transactional. Device publication occurs only after complete RAII setup. Teardown first prevents new use and proves outstanding native use safe, then destroys allocator bookkeeping and releases both backings; never reset an allocator while live owners, leases, or quarantine remain. Unknown native use retains the referenced backing (or the necessary whole arena) rather than freeing or reusing it.
8. CPU must continue to use its borrowed `Allocator` and must not manufacture arenas. TTNN must continue native per-plane tensor storage and must not manufacture raw arenas. Both accept `QueueConfig`; neither is to claim the standard-GPU two-allocation guarantee.
9. Migrate all factory callers and tests, including backend coexistence/conformance and `tools/sycl_add_staging_measure.cpp`, to the new signatures and explicit configuration. Keep C++20 and repository conventions. Do not add public operation/direction enums, duplicate owner query APIs, hidden allocator locks in canonical standalone allocators, or a legacy GPU overload.
10. Add focused regression coverage using task-01 instrumentation: default and custom C values including 1 and 17; checked metadata sizing; exactly two setup backing calls; no native allocation/free calls for tensor create/destroy; disjoint data/metadata addresses; stable addresses; fragmentation, reuse, coalescing, and concurrent tensor bookkeeping; and rollback at each setup failure point.

## Non-goals

- Workspace API or raw-workspace requirements/leases.
- Fixed queue partitions, completion events/resources, admission credits, parking, or dispatch scheduling.
- Host-transfer staging removal, SYCL binary staging removal, or any host allocation redesign.
- Heuristic/percentage-of-VRAM defaults, automatic growth, spilling, compaction, paging, relocation, unified-memory oversubscription, or peer allocation.
- A global device registry, new public operation/direction enums, or a native TTNN storage redesign.
- Changes to arithmetic, codecs, tiling, view semantics, or unsupported operations.

## Acceptance criteria

- `QueueConfig` and `DeviceMemoryConfig` are present in `include/iom/device.hpp` with the exact value semantics and factory signatures above; all five backend declarations/definitions, callers, tests, and `tools/sycl_add_staging_measure.cpp` compile against the cutover with no legacy GPU overload.
- Standard-GPU setup selects and validates the exact native context/device, accepts default/custom C including 1 and 17 when backend limits permit, rejects zero C and invalid capacities, checks `4 * C * 512` overflow/limits, and reports the required exception categories before publishing a device.
- Successful setup makes exactly two native backing-allocation calls, with disjoint data/metadata addresses, exact requested data capacity, checked metadata capacity, 32-byte alignment, one data `ListAllocator`, and one `FixedSizeAllocator` spanning all `4 * C` metadata blocks.
- Instrumented tensor create/destroy and repeated setup/lifecycle paths perform no native allocation/free calls after setup. Tensor data stays in the data arena, metadata stays separate, addresses do not move, fragmentation reports `std::bad_alloc`, and freed ranges safely reuse/coalesce without invalidating live tensors.
- Concurrent tensor bookkeeping is race-free at the Device boundary and never holds a bookkeeping lock across native work or waits. Every injected setup failure rolls back both backings and context/resources without publishing a partial device; teardown preserves backing memory for unknown native use and never resets live allocator state.
- CPU tests prove borrowed allocator behavior and TTNN tests prove unchanged native per-plane storage; neither allocates a fake arena. Backend conformance and callers use the new QueueConfig without changing unrelated numerical/storage semantics.

## Verification

Proposed gates (not run by this task):

- Local core/CPU: `ctest --test-dir build --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests|iom_backend_coexistence_tests)$'`.
- Accelerator smoke and conformance, through the `remote-development` workflow: `ctest --test-dir build --output-on-failure -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_ttnn_smoke_tests|iom_ttnn_conformance_tests)$'` for enabled backends.
- Through the `remote-development` workflow, run the focused instrumented setup/tensor-arena tests and confirm the allocation journal reports exactly two setup backings, zero post-setup tensor create/destroy native calls, checked metadata capacity for C=1/16/17, disjoint address domains, rollback at each injected setup failure, and fragmentation/reuse/coalescing under concurrent bookkeeping.

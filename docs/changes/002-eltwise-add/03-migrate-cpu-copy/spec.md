# Migrate CPU copy to the OID extension boundary

**Order:** 03
**Priority:** P0 — required backend migration before ADD work can begin
**Blocked by:** `02-add-oid-facades`
**Source:** `docs/changes/002-eltwise-add/spec.md`

## Outcome

The CPU queue's standard-tiled `copy` operation is migrated from a public virtual override to the protected backend extension boundary supplied by the common OID facade. CPU copies retain their current inline completion, validation, token, ordering, view-mapping, allocator ownership, and lifetime behavior while returning the new positive-token or negative-error results through the common boundary.

## Scope

- Migrate `CpuQueue` and its `copy_elements` path in `src/cpu/device.cpp:592-681` to the protected copy extension point introduced by the prerequisite facade migration; do not add a second public copy API or a CPU-specific facade.
- Keep `submission_order_mutex_` around CPU submissions so calls on one queue remain serialized and observable in call order.
- Ensure exact queue-device identity and exact source/destination specification matching (shape, leaf type, and quantization) are enforced before CPU work is accepted. Rejections must return the common negative `OidError::InvalidArgument` result, perform no destination write, and consume no sequence or token.
- Preserve the identical-window test (`native_handle`, plane offset, and plane strides) as a valid waitable no-op: it still receives a positive token and completes inline without moving data.
- Preserve logical element copying for full, offset, stepped, selected, permuted, nested, and overlapping standard-tiled views, including untouched padding and owner planes.
- Preserve CPU inline completion: valid work runs before the copy call returns and calls the common completion path before returning its positive token. There is no requirement for a shared outstanding-work registry for this completed-inline path.
- Keep the public OID boundary non-throwing. Synchronous validation/setup failures are mapped by the common facade to negative error results; accepted work returns a positive token, while `wait` remains repeatable and successful for the inline-completed token.

## Implementation references

- **Modify:** `src/cpu/device.cpp` — `CpuQueue::copy` and the private `copy_elements` implementation at lines 592-681; bind them to the protected extension boundary without changing the standard 16x16 copy mapping.
- **Read:** `include/iom/iom.hpp` — `DeviceOps` protected validation, submission, completion, and token helpers; use the boundary and common error/token machinery delivered by the blocker rather than duplicating it in `CpuQueue`.
- **Read:** `src/cpu/device.cpp` — `CpuDevice`, `CpuTensor`, `CpuTensor::allocator_`, `CpuTensor::address_`, `detail::allocate_aligned_storage`, `detail::release_aligned_storage`, and `CpuTensor::~CpuTensor` around lines 360-407; these are the CPU allocation and lifetime analogues whose behavior must remain unchanged.
- **Tests:** `test/cpu/test_cpu.cpp` — CPU-local asynchronous-copy and unsupported-compute cases at lines 924-1398; update only expectations coupled to the migrated OID/error boundary and inline CPU behavior.
- **Tests:** `test/CMakeLists.txt` — existing `iom_cpu_tests` target and registration; no new test target is required.

## Requirements

- `CpuQueue` must no longer implement the old public virtual OID-returning copy override. Its implementation must be reachable only through the protected backend extension point established by task `02-add-oid-facades`.
- Valid CPU copies must continue to use the common submission/completion machinery, preserve in-order sequencing, and return a positive token with the common 55-bit sequence encoding. A completed-inline token must be safe to wait on repeatedly.
- Exact-device and exact-spec validation must happen before reservation/effects. Device mismatch, shape mismatch, leaf mismatch, quantization mismatch, and other malformed copy inputs must produce the common negative `OidError::InvalidArgument` result, with no write and no sequence/token side effect.
- The CPU path must preserve the existing exact-window no-op and all logical plane-offset/stride behavior in `for_each_tile_lockstep` and `copy_elements`; it must not copy padding or unrelated owner planes.
- Do not change CPU storage behavior: `CpuTensor` continues to allocate exactly `TensorSpec::tiled_storage_nbytes()` through the borrowed allocator with the existing alignment checks, keeps one stable `native_handle`, and releases that allocation exactly once through the existing destructor/lifetime path. Copy operations must not allocate, free, relocate, replace, or re-own caller storage.
- CPU copies complete inline, so destroying source or destination tensor owners after `copy` returns must retain the current immediate allocator-free and address-recycling behavior; queue destruction must not introduce extra frees or a quarantine/registry requirement for this path.
- A pre-acceptance CPU failure must be surfaced as a negative synchronous OID result through the common mapping, with no accepted token and no observable effect. Do not turn a rejected copy into a waitable token.
- In `test/cpu/test_cpu.cpp:924-1137`, retain the inline lifetime and queue-destruction assertions: owner frees occur immediately and exactly once, fresh tensors can recycle addresses, waits are idempotent, and queue reset adds no storage protection or extra allocator traffic.
- In `test/cpu/test_cpu.cpp:1138-1213`, retain same-queue call ordering, monotonic positive tokens, independent queue identities, and foreign-token rejection by `wait`.
- In `test/cpu/test_cpu.cpp:1215-1259`, change migrated copy validation expectations from synchronous exceptions to negative `OidError::InvalidArgument` values while asserting untouched destination storage and that the next valid copy receives the first unused sequence.
- In `test/cpu/test_cpu.cpp:1261-1355`, retain the waitable identical-window no-op, real copies between distinct windows, overlap completion, logical-plane movement, untouched padding, and no allocator traffic during copies.
- In `test/cpu/test_cpu.cpp:1361-1398`, retain negative `OidError::Unsupported` results and no effects for CPU compute hooks that are not implemented. During this preliminary OID phase, do not implement ADD arithmetic; ADD's current unsupported expectation remains until the later ADD task. A valid CPU copy probe must still return the next positive token.

## Non-goals

- Designing or changing the common OID type, 55-bit encoding, public non-throwing facades, protected extension API, error mapping, sequence reservation, or shared completion implementation.
- Migrating shared/core fakes or core OID tests; those belong to their assigned migration task.
- Migrating CUDA, ROCm, SYCL, or TTNN copy paths.
- Implementing ADD arithmetic, validation, broadcasting, codecs, storage expansion, or any other compute operation.
- Adding `add_support`, a capability query, a generic fallback/registry subsystem, or vendor-specific behavior to the CPU path.
- Changing CPU allocation size/alignment, host-transfer behavior, tensor ownership, view representation, allocator lifetime rules, documentation, or unrelated tests and files.

## Acceptance criteria

- [ ] `CpuQueue` reaches CPU copy work through task 02's protected extension boundary, while the public operation returns only common-boundary OID results.
- [ ] A valid standard-tiled CPU copy returns a positive token, completes inline, preserves queue call order, and succeeds on repeated waits.
- [ ] Exact-device/spec-invalid copies return negative `OidError::InvalidArgument`, do not modify destination storage, and do not consume a sequence; the next valid copy receives the next unused token.
- [ ] Identical windows remain waitable no-ops; transformed/overlapping windows preserve logical data movement without touching padding or unrelated planes.
- [ ] CPU tensor allocation and lifetime observations remain unchanged: stable caller handles, exact borrowed-allocator ownership, immediate inline-path frees, address recycling, and no extra queue/registry frees.
- [ ] `iom_cpu_tests` covers the CPU-local cases in `test/cpu/test_cpu.cpp:924-1398`, and unimplemented non-ADD compute hooks still return negative `OidError::Unsupported` without effects.

## Verification

- `cmake --build <configured-build-dir> --target iom_cpu_tests` (focused CPU test build; do not run project-wide validation for this task).
- `ctest --test-dir <configured-build-dir> --output-on-failure -R '^iom_cpu_tests$'`.

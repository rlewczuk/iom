# Split CPU device components

**Order:** 11
**Priority:** P1 — factors the required CPU implementation on the normal backend path
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Complete the CPU vertical cutover by splitting the oversized `src/cpu/device.cpp` into responsibility-complete private components without changing the CPU backend's observable behavior. The normal CPU backend continues to expose the existing `make_cpu_device` factory and public interfaces while device, tensor, transfer, and queue responsibilities are compiled from separate implementation units. Every touched or new production source/header under `src/` remains at most 499 physical lines after normal formatting, with the tighter assignment budgets below.

## Scope

The current CPU implementation is `src/cpu/device.cpp` (941 lines). Use the existing responsibility boundaries:

- `src/cpu/device.cpp:23-358` contains anonymous transfer helpers; `:363-390` contains `CpuDevice`; `:392-401` contains `CpuWorkspace`; and `:923-939` contains workspace creation and the public factory.
- `src/cpu/device.cpp:403-616` contains `CpuTensor` and its host-region transfers.
- `src/cpu/device.cpp:618-921` contains `CpuQueue`, CPU copy/binary execution, and queue submission/completion behavior.

Add these private files:

- `src/cpu/device_internal.hpp` declares `iom::CpuDevice` for definitions shared by the CPU translation units. It is private linkage, not a public factory header. Keep the allocator reference, registry state, public virtual overrides, and the `create_tensor`, `create_workspace`, and `create_ops` declarations consistent with the existing class.
- `src/cpu/transfer_helpers.hpp` defines the private `iom::cpu_detail` transfer machinery needed by tensor and queue code: bit access/value copying, byte-aligned and sub-byte 16-element tile-row copies, and one-view/two-view tiled traversal. The existing anonymous helper behavior from `device.cpp:23-358` may be made inline or templated as needed, but must not be duplicated in multiple translation units.

Retain these responsibilities in `src/cpu/device.cpp`:

- `CpuDevice` construction, backend identity/device ordinal, supported standard data types, registry-state access, and its method definitions that belong to the device core.
- `CpuWorkspace` and `CpuDevice::create_workspace`. Only zero-byte raw workspace is supported; positive workspace requests still fail with the existing invalid-argument category/message behavior and never touch the allocator.
- `make_cpu_device(Allocator&, QueueConfig)` and its existing borrowed-allocator contract.

Create `src/cpu/tensor.cpp` for `CpuTensor` and `CpuDevice::create_tensor`. Keep aligned storage allocation through the existing allocator seam, zero-initialize the complete tiled allocation, release it on initialization failure and destruction, and preserve the tensor's storage handle and host-region transfer methods. Use `cpu_detail` helpers for standard 16x16 tile placement and row movement, including sub-byte bit packing, byte-aligned widths, partial rows/columns, padding, transforms, and multi-plane traversal.

Create `src/cpu/queue.cpp` for `CpuQueue` and `CpuDevice::create_ops`. Keep queue construction/destruction, registry queue-id invalidation, submission-order locking, `copy_impl`, binary operation selection/execution, broadcast plane mapping, scalar arithmetic, and completion/failure handling together. Reuse `cpu_detail` bit/tile helpers rather than retaining duplicate bit access code. The queue remains an independent in-order asynchronous `DeviceOps` queue even though CPU work executes synchronously inside submitted work.

Update the root `IOM_SOURCES` list in `CMakeLists.txt` to retain `src/cpu/device.cpp` and add `src/cpu/tensor.cpp` and `src/cpu/queue.cpp`. Do not add the private CPU headers to public header installation or alter the public CPU factory header. Do not change test source lists.

## Implementation references

- Public CPU factory contract: `include/iom/cpu/device.hpp`, especially `make_cpu_device` and its borrowed allocator, standard 16x16 layout, synchronous transfer, and independent in-order queue documentation.
- Base contracts used by the split declarations/definitions: `include/iom/device.hpp`, `include/iom/tensor.hpp`, and `include/iom/iom.hpp`.
- Source to factor: `src/cpu/device.cpp:23-358,363-401,403-616,618-921,923-939`.
- Build integration: `CMakeLists.txt:IOM_SOURCES` (currently containing `src/cpu/device.cpp`).
- Existing CPU behavior proof: `test/cpu/test_cpu.cpp` and `test/cpu/test_cpu_conformance.cpp`; the latter is the CPU conformance target `iom_backend_conformance_cpu_tests`.

## Requirements

1. Preserve all public headers, signatures, capabilities, error categories, validation order, ownership, lifetimes, and ODR behavior. No public CPU factory change, compatibility alias, shim, backend switch, global registry, cross-backend abstraction, synchronization redesign, or extra capability is permitted.
2. Preserve the borrowed `Allocator&` lifetime contract: the allocator is not owned by `CpuDevice`, and every CPU tensor allocation and release uses the existing allocator and aligned-storage helpers exactly once.
3. Preserve CPU tensor storage semantics: 32-byte aligned standard-layout storage, complete zero initialization, exception-safe release, direct release on destruction, and no deferred registry protection for the synchronous tensor transfers.
4. Preserve logical-to-physical mapping exactly: 16x16 tiled storage, all supported unquantized leaf widths, sub-byte bit order and masks, byte-aligned fast paths, padded tile tails, partial tile rows/columns, transforms, and one-view/two-view plane and stride traversal.
5. Preserve host transfer semantics and failure ordering for `region_from_host` and `region_to_host`, including sub-byte destination tail clearing and the existing handling of overlapping/self row ranges. No numeric conversion may be introduced by factoring.
6. Preserve the zero-only workspace contract. `CpuDevice::create_workspace(0)` returns the existing registered empty workspace; positive sizes fail before any allocator access. Registry identity and failure behavior remain unchanged.
7. Preserve queue ordering and repeatable wait/failure behavior. Submission-order locking, queue IDs, `DeviceOps` sequence bookkeeping, completion tokens, failure retention, and queue destruction invalidation must remain semantically identical.
8. Preserve CPU binary behavior and numerics: operation selection for add/mul/sub/div, data-type handling, broadcast rows/columns, leading-plane mapping, tiled traversal, bit load/store, and existing invalid-operation behavior. Existing validation remains in its current layer and order.
9. In the binary completion path, release or invalidate captured registry entries with the same failure flag and complete the workspace lease before calling `complete(sequence, failure)`. The completion proof and any staging/workspace lease semantics must not be reordered or weakened, even for the CPU's zero-workspace normal path.
10. Keep private visibility private. `device_internal.hpp` and `transfer_helpers.hpp` are implementation mechanisms only; no new public include path or public factory declaration is allowed. Use the fewest files listed here and do not create a factory-only, compatibility, or test-only production file.
11. Meet the assignment budgets after normal formatting: `src/cpu/device.cpp` at most 130 lines, `src/cpu/tensor.cpp` at most 260 lines, `src/cpu/queue.cpp` at most 350 lines, and `src/cpu/transfer_helpers.hpp` at most 370 lines. The new `src/cpu/device_internal.hpp` and every touched/new production file also remain at most 499 physical lines.
12. Add no tests and do not change existing test sources, test target source lists, public headers, or unrelated backends. Existing CPU tests and conformance remain the behavior proof; hardware-test skip behavior is not relevant to this CPU-only cutover.

## Non-goals

- Changing the CPU API, factory arguments/defaults, allocator ownership, supported types, layout, numerics, errors, validation, queue model, registry/quarantine semantics, or workspace/staging lease protocol.
- Introducing a cross-backend helper or abstraction, new synchronization, a new backend switch, a global registry, a new allocation strategy, or a capability not already present.
- Refactoring CUDA, ROCm, SYCL, TTNN, shared code, core code, public headers, tests, or CMake entries beyond adding the two CPU implementation sources to `IOM_SOURCES`.
- Adding compatibility aliases/shims, a permanent line-count test, or new unit/conformance coverage.

## Acceptance criteria

1. `src/cpu/device.cpp`, `src/cpu/tensor.cpp`, `src/cpu/queue.cpp`, `src/cpu/device_internal.hpp`, and `src/cpu/transfer_helpers.hpp` contain the planned responsibilities, with every moved definition present exactly once and no stale duplicate helper/class definition.
2. The CPU build still links the existing `make_cpu_device` public factory from `include/iom/cpu/device.hpp`; private headers remain non-public, and `IOM_SOURCES` includes both new CPU `.cpp` files while retaining `src/cpu/device.cpp`.
3. The CPU tensor path preserves borrowed allocation, alignment, zero initialization/release, standard 16x16 tiled and sub-byte storage, padding, transforms, and host-transfer behavior for all existing tests and conformance cases.
4. The CPU queue path preserves in-order submission, repeatable waits and failures, registry invalidation, binary numerics/broadcasting, and binary workspace-lease completion before `complete`.
5. The exact per-file budgets are met: device <=130, tensor <=260, queue <=350, helpers <=370, and every touched/new production `.cpp`/`.hpp` under `src/` is <=499 physical lines after formatting.
6. No public interface, test source list, or unrelated backend is changed, and no cross-backend abstraction, compatibility path, or permanent line-count test is introduced.

## Verification

These are instructions for the future implementer; do not add them as permanent tests and do not run them as part of authoring this mini-spec.

Configure exactly the CPU-only build:

```sh
cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
```

Build the exact CPU targets:

```sh
cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests
```

Run the exact CPU test set:

```sh
ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'
```

Run a scoped physical-line check over every original/touched/new CPU production file, enforcing both the assignment budgets and the universal 499-line cap:

```sh
for f in src/cpu/device.cpp src/cpu/tensor.cpp src/cpu/queue.cpp src/cpu/device_internal.hpp src/cpu/transfer_helpers.hpp; do
    lines=$(wc -l < "$f")
    test "$lines" -le 499 || { printf '%s: %s lines\n' "$f" "$lines"; exit 1; }
done
test "$(wc -l < src/cpu/device.cpp)" -le 130
test "$(wc -l < src/cpu/tensor.cpp)" -le 260
test "$(wc -l < src/cpu/queue.cpp)" -le 350
test "$(wc -l < src/cpu/transfer_helpers.hpp)" -le 370
```

The existing `iom_tests`, `iom_scalar_add_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` must pass without skips or new test cases. Inspect the resulting source list and symbol definitions to confirm the two new sources are linked and each moved CPU definition is emitted exactly once.

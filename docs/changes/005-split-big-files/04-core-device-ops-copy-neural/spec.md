# Extract copy and neural DeviceOps families

**Order:** 04
**Priority:** P0 — completes operation-family migration needed before iom.cpp removal
**Blocked by:** `01-core-tensor-view`
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Extract the copy and neural `DeviceOps` operation families from `src/iom.cpp` into responsibility-complete translation units while preserving the public API and every existing observable contract. `src/iom.cpp` remains in the build for its definitions that are outside these two families; only the definitions assigned below are removed from it. The resulting core sources remain suitable for the later complete `iom.cpp` removal. Both new translation units must be at most 499 physical lines; the intermediate `src/iom.cpp` is removed by `05-core-device-ops-cutover`, which enforces the final global cap.

## Scope

This task owns exactly two planned production files:

- `src/device_ops_copy.cpp`: copy validation, identical-window detection, the default unsupported `copy_impl`, and the public `DeviceOps::copy` facade.
- `src/device_ops_neural.cpp`: the default unsupported hooks and public facades for `silu`, `linear`, `rmsnorm`, and `sdpa`.

Move the assigned definitions out of `src/iom.cpp`, add both destinations to the core `IOM_SOURCES` list, and retain `src/iom.cpp` for all unassigned definitions. Use the minimal private linkage in planned `src/iom_internal.hpp` only where an already-private shared symbol such as `UnsupportedOperation` is required; it is not a general helper header. Do not change public declarations or signatures in `include/iom/iom.hpp`.

The change is source factoring only. Existing core tests and backend conformance helpers remain the behavior proof; no new tests or test-source-list changes are part of this task.

## Implementation references

- Copy definitions to move: `src/iom.cpp:961-981` (`DeviceOps::validate_copy`, `DeviceOps::identical_window`), `src/iom.cpp:1017-1019` (`DeviceOps::copy_impl`), and `src/iom.cpp:1067-1075` (`DeviceOps::copy`).
- Neural definitions to move: `src/iom.cpp:1021-1035` (default `silu_impl`, `linear_impl`, `rmsnorm_impl`, and `sdpa_impl`) and `src/iom.cpp:1277-1327` (the four public facades).
- Public operation and protected hook declarations that must remain unchanged: `include/iom/iom.hpp:290`, `include/iom/iom.hpp:332-339`, `include/iom/iom.hpp:400-411`, and `include/iom/iom.hpp:442-443`.
- Core source registration: `CMakeLists.txt:74-80`; add `src/device_ops_copy.cpp` and `src/device_ops_neural.cpp` without removing the retained `src/iom.cpp` in this task.
- Core behavior tests, including mixed copy/neural submission and copy sequencing/identical-window cases: `test/test_iom.cpp:2086-2100` and `test/test_iom.cpp:2515-2603`.
- Backend-neutral lifetime, synchronous-failure mapping, unsupported compute, and copy helper scenarios: `test/backend/backend_conformance_other.hpp:238-397` and `test/backend/backend_conformance_other.hpp:535-646`.
- Existing binary operation/reference helpers that must continue to compile unchanged while the family definitions move: `test/backend/backend_conformance_add.hpp:157-322`.

## Requirements

1. **Copy ownership and validation.** Place `DeviceOps::validate_copy` and `DeviceOps::identical_window` in `src/device_ops_copy.cpp` with their existing private linkage and declarations. `validate_copy` must first call `validate_views(device, {&source, &destination})`, then require equal `TensorSpec` values, preserving the current `std::invalid_argument` category, message, and validation order for device ownership and mismatched shape, leaf type, or quantization. `identical_window` must continue to compare native handle, plane offset, and all plane strides exactly as it does today.

2. **Copy default hook and facade.** Define the base `DeviceOps::copy_impl` in `src/device_ops_copy.cpp` with the existing `UnsupportedOperation` behavior. Define `DeviceOps::copy` there as the unchanged `noexcept` facade: obtain `queue_device()`, run `validate_copy`, invoke `copy_impl`, and map every caught failure through `invoke_failure`; successful results still pass through `invoke`. Preserve the no-op/identical-window submission behavior used by concrete queues; this extraction must not short-circuit, allocate, reserve a sequence, or alter backend dispatch.

3. **Neural ownership and unsupported behavior.** Place the default `silu_impl`, `linear_impl`, `rmsnorm_impl`, and `sdpa_impl` definitions in `src/device_ops_neural.cpp`. Each must continue to throw `UnsupportedOperation`, so the existing failure mapper returns `OidError::Unsupported` and no capability is added. Do not move binary hooks or any backend override into either new file.

4. **Neural facade contracts.** Place the four public facades in `src/device_ops_neural.cpp` without changing signatures, `noexcept`, or ordering:
   - `silu` validates `{&x, &y}` and invokes `silu_impl`.
   - `linear` validates `{&x, &w, &y}` and invokes `linear_impl`.
   - `rmsnorm` validates `{&x, &y, &w}`, then rejects `eps < 0` or `dim == 0` with the existing `std::invalid_argument` path before invoking `rmsnorm_impl`.
   - `sdpa` validates `{&q, &k, &v, &attn_out}`, then rejects zero `n_heads`, zero `n_kv_heads`, zero `head_dim`, or a non-divisible `n_heads % n_kv_heads` with the existing `std::invalid_argument` path before invoking `sdpa_impl`.

   Every facade must catch all failures and return `invoke_failure(std::current_exception())`; no synchronous exception may cross the public OID boundary. Preserve validation-before-hook order and the existing unsupported/error mapping.

5. **Private linkage and exact-once definitions.** Include only the headers and private declarations required by the moved definitions. If `src/iom_internal.hpp` is consumed, keep it limited to the already-approved backend-neutral private symbols and preserve visibility/ODR behavior. Remove each moved definition from `src/iom.cpp`; leave no duplicate or compatibility definition and retain every unrelated `iom.cpp` definition exactly once.

6. **Build integration.** Add both planned `.cpp` files to `IOM_SOURCES` alongside retained `src/iom.cpp`. Do not change `include/iom/iom.hpp`, public factory headers, backend target source lists, test target source lists, or any backend implementation. Both new files must satisfy the 499-line production-source limit after normal formatting. The intentionally retained intermediate `src/iom.cpp` may remain oversized until `05-core-device-ops-cutover` removes it.

7. **Behavior proof boundaries.** Existing `test/test_iom.cpp` and the backend `other`/`add` helpers must continue to exercise copy token ordering, identical-window copies, validation failures without unintended submission, repeatable waits/failures, neural facade validation, and unsupported mapping. Do not duplicate those tests or alter their expected behavior. Hardware conformance remains fail-on-error rather than skip, even though this task does not add backend code.

## Non-goals

- Do not move or alter binary validation, binary snapshots, `binary_impl`, binary facades, binary workspace requirement queries, or any `add`/`mul`/`sub`/`div` implementation.
- Do not move queue-ID allocation, constructors/destructors, `wait`, token encoding, sequence bookkeeping, fence/completion processing, failure retention, `WorkspaceValidation`, workspace leases, registry/quarantine logic, or lifecycle code.
- Do not change public headers, signatures, backend capabilities, unsupported/error categories, validation order, ownership/lifetimes, allocations, asynchronous in-order queues, repeatable waits/failures, context behavior, numerics, or private visibility.
- Do not add backend switches, global registries, cross-backend abstractions, synchronization redesign, compatibility aliases/shims, extra capabilities, new tests, changed test source lists, public factory headers, or a permanent line-count test.
- Do not remove `src/iom.cpp` in this task; its later removal is outside this mini-spec. Do not implement production code while authoring this specification.

## Acceptance criteria

- `src/device_ops_copy.cpp` contains exactly the assigned copy validation, identical-window predicate, default hook, and public facade; `src/device_ops_neural.cpp` contains exactly the assigned four default hooks and four public facades.
- The moved definitions are absent from `src/iom.cpp`, all unassigned definitions remain there, and every moved symbol is defined exactly once with no compatibility translation unit or ODR change.
- Copy preserves device/view validation order, identical-window/no-op detection, unsupported mapping, allocation behavior, sequence/token behavior, and backend dispatch; neural facades preserve operand/parameter validation order and map unsupported hooks to `OidError::Unsupported` without adding capability.
- `CMakeLists.txt` lists both new sources and retains `src/iom.cpp`; public headers and all test/backend source lists are unchanged.
- `src/device_ops_copy.cpp` and `src/device_ops_neural.cpp` are each at most 499 physical lines after normal formatting. The intermediate `src/iom.cpp` is removed by `05-core-device-ops-cutover`; no unrelated production file is changed.
- Existing core and CPU/backend-conformance behavior remains covered by the cited tests, including repeatable waits and failures, invalid-input no-effect behavior, and unsupported neural operations.

## Verification

Proposed future gates (not run while writing this specification):

```sh
cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
cmake --build build/split-cpu --target libiom iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests
ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'
```

Run a one-time scoped physical-line check after formatting; it is verification only and must not become a permanent test:

```sh
python3 - <<'PY'
from pathlib import Path
paths = [Path("src/device_ops_copy.cpp"), Path("src/device_ops_neural.cpp")]
counts = {path: sum(1 for _ in path.open()) for path in paths}
print("\n".join(f"{path}: {count}" for path, count in counts.items()))
assert all(count <= 499 for count in counts.values())
PY
```

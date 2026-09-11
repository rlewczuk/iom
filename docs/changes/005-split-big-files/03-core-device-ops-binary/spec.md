# Extract core binary DeviceOps

**Order:** 03
**Priority:** P0 — migrates the largest DeviceOps family before removal of iom.cpp.
**Blocked by:** `01-core-tensor-view`
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

`src/device_ops_binary.cpp` becomes the single implementation home for the core binary `DeviceOps` family currently embedded in `src/iom.cpp`. It contains binary type/spec/rank/view/shape validation and snapshots, the default `binary_impl`, the four binary facades, and the four binary workspace-requirement queries. The move is mechanical: public declarations and all observable behavior remain unchanged, `src/iom.cpp` retains only responsibilities outside this task, and the two translation units have no duplicate definitions.

## Scope

Move only these binary responsibilities from `src/iom.cpp` to planned `src/device_ops_binary.cpp`:

- binary validation helpers and snapshots from `src/iom.cpp:657-956`, including recognized quantization and numeric-type checks, checked binary plane counts, `validate_binary_spec`, `validate_binary_view`, `DeviceOps::snapshot_binary_view`, and `DeviceOps::validate_binary`;
- the default `DeviceOps::binary_impl` definition from `src/iom.cpp:1013-1015`;
- `DeviceOps::add`, `DeviceOps::mul`, `DeviceOps::sub`, and `DeviceOps::div` from `src/iom.cpp:1076-1170`;
- `DeviceOps::binary_workspace_requirements` and the operation-specific `add_`, `mul_`, `sub_`, and `div_workspace_requirements` definitions from `src/iom.cpp:1172-1210`.

Remove each moved definition from `src/iom.cpp`, but retain `src/iom.cpp` in the build for its remaining tensor, workspace, copy, neural, lifecycle, and shared DeviceOps definitions. Add `src/device_ops_binary.cpp` to `IOM_SOURCES` alongside the retained `src/iom.cpp`; the eventual removal of `iom.cpp` is outside this isolated extraction.

The new file consumes `src/iom_internal.hpp` for the private checked-arithmetic, rank-limit, and unsupported-operation facilities. It must not define a second copy of those facilities or turn that header into a general helper bucket. No public header, test source, backend implementation, or test source list is changed by this task.

## Implementation references

- Binary public facades, workspace query declarations, `BinaryOperation`, `BinaryViewSnapshot`, `BinaryRequest`, and protected hooks: `include/iom/iom.hpp:283-435`.
- Binary validation helpers, snapshots, and request validation to relocate: `src/iom.cpp:657-956`.
- Default unsupported binary hook: `src/iom.cpp:1013-1015`.
- Binary facades and workspace queries to relocate: `src/iom.cpp:1076-1210`.
- Private checked arithmetic, rank constants, and `UnsupportedOperation` linkage consumed by the new translation unit: planned `src/iom_internal.hpp`.
- Core source list receiving the new translation unit while retaining `iom.cpp`: `CMakeLists.txt:74-80`.
- Public declarations and binary/workspace behavior coverage: `include/iom/iom.hpp`; `test/test_iom.cpp`.
- Operation-neutral binary request, facade, query, lease, and completion coverage: `test/backend/backend_conformance_add.hpp` and `test/backend/backend_conformance_common.hpp`.

## Requirements

1. **Mechanical ownership split.** Put every definition named in Scope in `src/device_ops_binary.cpp`, exactly once. Delete only those definitions from `src/iom.cpp`; leave copy, neural, lifecycle, `WorkspaceValidation`, and unrelated helpers in their existing owner. Do not add wrappers, aliases, forwarding definitions, compatibility translation units, or alternate binary paths. Keep all moved symbols at their current namespace and private/protected visibility.

2. **Preserve validation and snapshot semantics.** `validate_binary` must retain the exact existing order and categories: validate all three specs; compare data type and quantization; validate each view's owner/device identity, stable native handle, owner spec, plane-stride count and nonzero strides, checked plane bounds and storage coverage; compute checked broadcast-compatible result dimensions; validate the output shape; construct the result shape; snapshot all three views; reject non-exact input/output aliasing; then reject unsupported quantization or data types. Preserve rank two-through-eight and nonzero-dimension rules, checked overflow paths, operation-specific existing diagnostics, and the `UnsupportedOperation` exception for capability rejection. No sequence, registry, metadata, workspace, token, or backend effect may occur before these checks complete.

3. **Preserve complete snapshots.** `snapshot_binary_view` must retain the `TensorSpec`, device and owner identities, stable native handle, plane offset, physical and logical plane strides, row/column broadcast flags, and aggregate broadcast flag in `BinaryViewSnapshot`. Preserve leading-axis alignment, row/column broadcasting, and all vector ownership/copy behavior exactly; do not recompute snapshot state later in a facade or backend.

4. **Preserve facade protocol.** Each of `add`, `mul`, `sub`, and `div` remains a `noexcept` OID facade with the existing signature. Preserve operation selection, `queue_device()` lookup, `validate_binary`, query dispatch, `detail::WorkspaceValidation::validated`, operand collection, request construction, `invoke(binary_impl(...))`, catch-all failure mapping, and their exact ordering. Pass the validated borrowed workspace, returned `WorkspaceRequirements`, and an empty `WorkspaceLease` in the same `BinaryRequest` fields and order. Do not move or redefine `WorkspaceValidation`; call the existing shared implementation.

5. **Preserve hooks and pure queries.** The base `binary_impl` continues to throw the existing `UnsupportedOperation`, allowing the common failure mapper to produce `OidError::Unsupported`. The base `binary_workspace_requirements` continues to return `{0, 1}`. Each operation-specific query must run `validate_binary` with its matching operation and then call the virtual requirement hook, preserving pure, deterministic behavior: no allocation, registration, lease, sequence/token reservation, metadata upload, queue mutation, output mutation, or backend submission. Backend overrides, including SYCL's nonzero calculation, remain effective through the existing virtual hook and are not copied into this file.

6. **Preserve lifetime and error contracts.** Keep `RawWorkspaceView`, `WorkspaceRequirements`, and `detail::WorkspaceLease` as request fields without changing ownership, copying, leasing, completion-proof release, quarantine, or staging behavior. Preserve `std::invalid_argument`, overflow, unsupported-operation, and all OID mappings and their established validation precedence. The extraction must not introduce synchronization, allocation, capability, arithmetic, alias, or queue-order changes.

7. **Build and size integration.** Add `src/device_ops_binary.cpp` to `IOM_SOURCES` without removing the retained `src/iom.cpp`. Include only the dependencies needed by the moved binary code and `src/iom_internal.hpp`. Keep the new production translation unit at no more than 499 physical lines after normal formatting and do not add permanent line-count tests. The complete factoring change must also leave every touched production source under the same cap once the retained `iom.cpp` responsibilities have been extracted.

## Non-goals

- Do not move or implement tensor/view code, raw workspace ownership, `WorkspaceValidation`, copy validation/facades, neural facades, lifecycle/queue/failure machinery, or any backend binary implementation.
- Do not change `include/iom/iom.hpp`, `BinaryRequest`, public/protected signatures, capabilities, error categories, diagnostics, validation order, alias policy, broadcasting, numerics, workspace geometry, leases, registry/quarantine behavior, or asynchronous ordering.
- Do not add tests, alter test sources or CMake test lists, add a backend switch or global registry, introduce cross-backend abstractions, allocate implicit scratch, or create compatibility shims.
- Do not remove `src/iom.cpp` in this task; it remains in `IOM_SOURCES` as the carrier for responsibilities outside this specification.

## Acceptance criteria

- `src/device_ops_binary.cpp` contains exactly one definition of every binary helper, snapshot/validation method, default `binary_impl`, four binary facades, and five workspace-requirement methods in Scope; the corresponding originals are absent from `src/iom.cpp`.
- `src/iom.cpp` remains buildable for its responsibilities outside this scope, and `IOM_SOURCES` contains both `src/iom.cpp` and `src/device_ops_binary.cpp` with no unrelated source-list changes.
- Public signatures and protected `BinaryOperation`, `BinaryViewSnapshot`, and `BinaryRequest` contracts are unchanged. Existing core and conformance tests observe identical validation/error/capability behavior, snapshots, broadcast/alias handling, pure query results, workspace fields, and lease handoff.
- Unsupported base binary dispatch maps through the existing `OidError::Unsupported` path; invalid and overflow cases preserve their existing exception/OID categories and precedence, with no acceptance or native effect on rejected input.
- The new file is at most 499 physical lines after normal formatting, and the eventual integrated source set satisfies the universal 499-line cap without a permanent scanner test.

## Verification

Proposed gates (not run while writing this specification):

- Configure a CPU-only tree: `cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF`.
- Build the requested core/CPU targets: `cmake --build build/split-cpu --target libiom iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests`.
- Run the focused tests: `ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'`.
- Perform a scoped one-time line check after normal formatting, confirming `src/device_ops_binary.cpp` is at most 499 physical lines and that the integrated touched production set (including the final retained/extracted core files) is at most 499; this check must not become a permanent test or source-list target.

# Extract core workspace definitions

**Order:** 02
**Priority:** P0 — isolates workspace ownership and registry behavior needed by the final core cutover
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Move the backend-neutral raw-workspace value/owner definitions and the `Device` live-workspace registry hooks out of the monolithic core implementation and into one responsibility-complete planned translation unit, `src/workspace.cpp`. The extraction changes only source organization and build membership: public workspace/device contracts, identity checks, ownership and lifetime rules, errors, synchronization, and observable behavior remain exactly unchanged. During this task, `src/iom.cpp` remains in the build and retains all definitions outside the specified workspace block; the moved definitions must have exactly one compiled definition each.

## Scope

- Move the definitions currently at `src/iom.cpp:548-645` for `RawWorkspaceView`, `RawWorkspace`, and `Device::{owns_workspace,register_workspace,unregister_workspace}` into planned `src/workspace.cpp`.
- Remove those definitions from `src/iom.cpp`, while retaining `src/iom.cpp` itself and every definition outside the workspace block.
- Add planned `src/workspace.cpp` to `IOM_SOURCES` in `CMakeLists.txt:74-80`, retaining the existing `src/iom.cpp` entry and all existing source entries.
- Keep this as a backend-neutral extraction. `src/workspace.cpp` must use the existing public declarations and must not expose or include backend runtime implementation details.
- Keep the destination responsibility-complete and at most 499 physical lines after normal formatting. `src/iom.cpp` is an intentional intermediate migration carrier and may remain oversized until `05-core-device-ops-cutover` removes it; the final integrated change must satisfy the universal production line cap. Do not evade the final cap with a compatibility translation unit or permanent scanner/test.

## Implementation references

- **Add (planned):** `src/workspace.cpp` — definitions of `iom::RawWorkspaceView::RawWorkspaceView`, `device`, `backend_kind`, `backend_device`, `subrange`, and `range_address`; `iom::RawWorkspace::RawWorkspace`, destructor, `backend_kind`, `backend_device`, `view`, and `workspace_address`; and `iom::Device::owns_workspace`, `register_workspace`, and `unregister_workspace`.
- **Modify:** `src/iom.cpp:548-645` — remove exactly the above definitions and no adjacent anonymous-namespace/device-operations code.
- **Read/retain:** `include/iom/device.hpp:36-116` — `Device::create_workspace`, `owns_workspace`, protected registration hooks, and the mutex/set live-owner state. Do not alter this public header, its signatures, or its private registry fields.
- **Read/retain:** `include/iom/tensor.hpp:150-287` — `WorkspaceRequirements`, `RawWorkspaceView`, and `RawWorkspace` declarations, friends, value traits, and range/address contract. Do not alter these declarations.
- **Read/retain:** `include/iom/iom.hpp:206-239` and workspace-related `DeviceOps` contracts — shared validation consumes exact owner/device identity and `RawWorkspaceView::range_address`; preserve those interfaces and their error behavior.
- **Modify:** `CMakeLists.txt:74-90` — add `src/workspace.cpp` to `IOM_SOURCES` without removing `src/iom.cpp`, changing `IOM_HEADERS`, or changing any test target source list.
- **Read/retain:** `test/test_iom.cpp` workspace/identity cases around `RawWorkspace`, `RawWorkspaceView`, `FakeWorkspace`, and `Device` ownership (including the workspace tests beginning around line 3571). Existing tests are behavior proof; do not add tests in this task.

## Requirements

- Preserve the exact `RawWorkspaceView` constructor validation order and categories. It must store the owner identity and requested range, reject `offset > owner.byte_size()` with the existing `std::out_of_range` behavior, then reject `bytes > owner_bytes - offset` with the existing `std::out_of_range` behavior. Keep the checked subtraction form so no range-end arithmetic is performed before the offset bound is established.
- Preserve `RawWorkspaceView::device()`: an empty view (`owner_ == nullptr`) throws the existing `std::logic_error`, while a non-empty view returns the exact creating `Device` through its owner. `backend_kind()` and `backend_device()` must continue to delegate through `device()` and therefore preserve empty-view failure behavior and ordering.
- Preserve `RawWorkspaceView::subrange(offset, bytes)`: an empty view rejects with the existing `std::invalid_argument`; non-empty views require a 32-byte-aligned subrange offset before constructing the checked owner-absolute range, and range bounds continue to be enforced by the constructor. Do not introduce a pointer/handle constructor, implicit allocation, retargeting, or alternate alignment rule.
- Preserve `RawWorkspaceView::range_address()` as `noexcept`: derive the address from the owner’s `workspace_address()` (or a null base for an empty view) plus the stored offset, without dereferencing an empty owner and without adding allocation or synchronization.
- Preserve `RawWorkspace` ownership/lifetime semantics. Its protected constructor stores the exact creating `Device` and byte size, then registers `this` with that `Device`; its destructor unregisters the same identity. Keep the owner non-copyable/non-movable and do not change the requirement that the creating `Device` outlive the workspace.
- Preserve `RawWorkspace::backend_kind()` and `backend_device()` as `noexcept` delegation to the creating `Device`, `view()` as the full checked range `{*this, 0, bytes_}`, and the default `workspace_address()` result (`nullptr`) for allocation-free empty/base owners. Backend subclasses retain their existing overrides and native allocation behavior in their own files.
- Preserve registry identity and synchronization exactly. `Device::owns_workspace(nullptr)` returns `false`; otherwise it locks `workspace_registry_mutex_` and checks exact pointer membership in `live_workspaces_`. Registration inserts the exact pointer under the same mutex; unregistration erases it under the same mutex and remains `noexcept`. Do not dereference the pointer in the ownership query or create a second registry.
- Preserve all existing validation and lifetime consumers: foreign-device, foreign-owner, dead-owner, address/subrange, overlap, lease, registry/quarantine, and workspace destruction behavior must remain unchanged. The extraction must not reorder validation, registration, unregistration, allocation, completion-proof release, or failure handling.
- Preserve the universal cross-cutting behavior even though this extraction does not define queue operations: asynchronous in-order queues, repeatable waits and retained failures, registry/quarantine transitions, workspace/staging lease completion proof, backend context behavior, and numerical results must remain unchanged because no consumer protocol or synchronization boundary is redesigned.
- Ensure each moved non-inline member has exactly one definition after the edit: no duplicate copies in `src/iom.cpp`, `src/workspace.cpp`, backend files, or another translation unit. Retain `src/iom.cpp` in `IOM_SOURCES` for the remaining definitions during this intermediate cutover; do not add a compatibility alias or duplicate translation unit.
- Keep the public API and capabilities unchanged: no header/signature changes, backend switch, global registry, cross-backend abstraction, synchronization redesign, new allocation, ownership transfer, or new tests. Preserve ODR and private visibility.

## Non-goals

- Splitting or editing any other `src/iom.cpp` responsibility, including tensor/shape/view definitions, `DeviceOps` infrastructure, binary/copy/neural facades, validation, token/wait/failure machinery, or unrelated anonymous-namespace helpers.
- Changing `include/iom/device.hpp`, `include/iom/tensor.hpp`, `include/iom/iom.hpp`, public factory headers, backend implementations, tests, test source lists, or CMake behavior beyond adding the one planned source entry.
- Redesigning workspace allocation, device arenas, native backend workspaces, address computation, range alignment, leases, quarantine, registry semantics, ownership/lifetime, or error mapping/order.
- Adding compatibility shims, aliases, wrappers, backend switches, global state, cross-backend helpers, extra capability, a permanent line-count test, or a new test case.
- Removing `src/iom.cpp` in this task; the final core cutover may remove it only after all of its remaining definitions have been moved by their separately specified tasks.

## Acceptance criteria

- [ ] Planned `src/workspace.cpp` contains exactly one definition of every scoped `RawWorkspaceView`, `RawWorkspace`, and `Device` workspace-registry member, compiles against the unchanged public declarations, and is at most 499 physical lines after normal formatting.
- [ ] `src/iom.cpp` no longer contains the definitions from the workspace block at `548-645`, retains its other definitions, and remains listed in `IOM_SOURCES` alongside `src/workspace.cpp`; no duplicate or compatibility translation unit exists.
- [ ] Empty and non-empty workspace views preserve all existing observable behavior: owner identity, exact `Device` identity, byte size/offset/range, equality/copy traits, backend queries, 32-byte subrange rule, checked bounds, and empty-view error behavior.
- [ ] Owner construction/destruction preserves exact-once registration and unregistration, including zero-byte owners and dead-owner views. `Device::owns_workspace` rejects null/foreign/dead identities and accepts only the live owner registered by that exact `Device`; independent `Device` instances remain isolated.
- [ ] Address behavior remains unchanged for empty, base, and subrange views, including `noexcept` behavior and backend-overridden workspace addresses. No new allocation or native backing is introduced by the extraction.
- [ ] Existing workspace/identity tests in `test/test_iom.cpp` continue to provide the proof for range checks, identity, lifetime, and registry behavior; no tests or test source lists are added or changed.
- [ ] Planned `src/workspace.cpp` is at most 499 physical lines after normal formatting. The intermediate `src/iom.cpp` is removed by `05-core-device-ops-cutover`, which enforces the final tracked-production cap; no permanent line-count test or scanner is added.

## Verification

The following are future implementer/supervisor instructions and are intentionally **not run while writing this specification**:

- `cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF` — CPU-only configure; not run.
- `cmake --build build/split-cpu --target libiom iom_tests` — compile/link the library and focused common workspace/identity test target; not run.
- `ctest --test-dir build/split-cpu --output-on-failure -R '^iom_tests$'` — run the focused existing workspace/identity coverage; not run.
- A one-time scoped physical-line check over planned `src/workspace.cpp`, asserting `<=499`, plus a recorded informational count for the intentionally retained intermediate `src/iom.cpp`. The latter is removed by `05-core-device-ops-cutover`, where the final global line-cap check runs. Verification only, not a permanent test; not run.

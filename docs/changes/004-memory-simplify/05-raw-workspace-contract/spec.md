# Add raw workspace ownership and queries

**Order:** 05
**Priority:** P0 — operation and transfer cutovers require checked explicit scratch ownership.
**Blocked by:** `03-rank-eight-contract`, `04-device-memory-arenas`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

Add a backend-neutral raw device-workspace contract. A `Device` creates stable, explicitly owned, non-copyable/non-movable `RawWorkspace` objects; copy-constructible, non-retargetable `RawWorkspaceView` values describe checked byte ranges; and pure operation/transfer queries report deterministic workspace requirements. Standard GPU workspaces suballocate the existing data arena, while CPU and TTNN support only the empty workspace. Shared validation and lifetime leasing make workspace use safe through queued completion and quarantine.

## Scope

- Add planned public `WorkspaceRequirements{bytes, alignment}`, `RawWorkspace`, and `RawWorkspaceView` types in `include/iom/device.hpp` / `include/iom/tensor.hpp` without exposing CUDA, HIP, SYCL, or TTNN runtime types. Add planned `Device::create_workspace(std::size_t bytes)`; the owner is created by the exact `Device`, has stable storage, and cannot be copied or moved.
- Make `RawWorkspaceView{}` the valid empty default. Views are non-owning, copy-constructible, and non-retargetable; they expose checked byte size/range and exact owning `Device`/`RawWorkspace` identity. Only a live owner can produce a view, and all owner/range arithmetic is checked.
- Add planned `DeviceOps::{add,mul,sub,div}_workspace_requirements(...)` queries with the same three-view inputs as their binary operations, and planned `TensorView::{copy_from_host,copy_to_host}_workspace_requirements()` queries. Do not add public operation or direction enums or duplicate query APIs on owners.
- Record the downstream contract for the four binary facades and both host-transfer facades: each takes a borrowed `RawWorkspaceView` defaulted to `RawWorkspaceView{}` solely for zero-requirement source simplicity; a positive requirement with that empty argument is invalid. Do not implement that signature migration here.
- Keep the six execution facade signatures and existing staging allocations unchanged in this task; tasks 07 and 11 consume this contract by adding borrowed workspace arguments and replacing operation-time staging where specified.

## Implementation references

- **Modify:** `include/iom/device.hpp` — planned `RawWorkspace`, `RawWorkspaceView`, `WorkspaceRequirements`, and `Device::create_workspace`; keep the public header backend-neutral and preserve the exact-`Device` identity model.
- **Modify:** `include/iom/tensor.hpp` — planned `RawWorkspaceView` range/identity declarations and `TensorView` host-transfer requirement queries.
- **Modify:** `include/iom/iom.hpp` — planned four `DeviceOps` binary requirement queries and shared validation/lease hooks; preserve common facade validation and OID error categories.
- **Modify:** `src/iom.cpp:278-288,705-863` — reuse exact view/device validation and binary operation validation/snapshot rules for pure queries and workspace checks.
- **Modify:** concrete Device implementations in `src/cpu/device.cpp`, `src/cuda/device.cpp`, `src/rocm/device.cpp`, `src/sycl/device.cpp`, and `src/ttnn/device.cpp` (and their public headers where declarations require it) — implement backend-specific creation and requirement results.
- **Read:** `include/iom/detail/outstanding_work_registry.hpp:691-749` — reuse transactional owner registration, release/invalidation, and quarantine behavior rather than inventing a separate lifetime registry.
- **Read:** `include/iom/gpu_algorithm.hpp:10-20` — use `gpu_algorithm::compute_staging_size` for GPU host-transfer requirements and its overflow behavior.
- **Read:** `src/sycl/copy.cpp:509-535` — preserve `binary_view_staging_bytes` whole-plane/padding semantics for SYCL binary requirements, with checked arithmetic.
- **Read:** `test/test_alloc.cpp` and `test/backend/backend_conformance_oracle.hpp` — reuse alignment, checked-range, allocator, and independent storage-oracle conventions.
- **Tests:** `test/test_iom.cpp` for public type traits, empty/range/identity behavior, pure-query side-effect checks, and common rejection/lease cases; `test/cpu/test_cpu_conformance.cpp` and enabled backend conformance sources for backend result and lifetime coverage.

## Requirements

- `RawWorkspace` must be non-copyable and non-movable. It is owned by its creating `Device`, retains stable backing/range identity, and must not relocate or retarget while views or accepted work exist. A view must not be constructible from an arbitrary pointer or backend handle.
- `RawWorkspaceView` copies must preserve the exact owner, device, base, offset, and byte extent. Assignment or any operation that retargets an existing view is prohibited. A checked subrange must satisfy `offset <= owner_bytes` and `bytes <= owner_bytes - offset`; overflow, out-of-range, dead-owner, and foreign-owner/device views are invalid.
- `Device::create_workspace(0)` returns a valid empty owner/view and performs no native allocation. Positive creation on CUDA, ROCm, and SYCL allocates a 32-byte-aligned range from the already reserved data arena under the owning `Device`'s allocator lock; it must never reserve a new native backing allocation. Data-arena exhaustion, including fragmentation with no fitting contiguous range, throws `std::bad_alloc` and does not create fallback backing.
- Positive creation on CPU and TTNN is rejected as unsupported device scratch (use the established invalid-argument category) rather than manufacturing dummy native storage. Empty workspaces remain accepted on every backend.
- Every `WorkspaceRequirements` result is deterministic and uses `alignment == 1` when `bytes == 0`. A positive requirement uses the stated alignment and checked byte arithmetic; overflow is reported as `std::overflow_error` (and mapped through the existing OID error mapping where applicable).
- Binary requirement queries must validate the operation, all three views, exact device identity, specifications, rank/view bounds, broadcasting, alias rules, and backend capability exactly as the corresponding binary operation, but must allocate, register, lease, reserve a token/queue resource, upload metadata, or submit nothing. They must not depend on free data-arena capacity, fragmentation, queue occupancy, or completion state.
- Return binary requirements exactly as follows: CPU, TTNN, CUDA, and ROCm return `{0, 1}`. For SYCL, let `lhs_bytes`, `rhs_bytes`, and `out_bytes` be the checked `binary_view_staging_bytes` values; return `{align_up(lhs_bytes, 32) + align_up(rhs_bytes, 32) + out_bytes, 32}` with checked additions. The three slices start at offsets `0`, `align_up(lhs_bytes, 32)`, and `align_up(lhs_bytes, 32) + align_up(rhs_bytes, 32)`; preserve whole-plane padded-storage semantics and check every alignment-up operation.
- Host-transfer requirement queries must validate the `TensorView` and direction-specific transfer preconditions without allocation, registration, leasing, or native effects. CPU and TTNN return `{0, 1}`. CUDA, ROCm, and SYCL return `{gpu_algorithm::compute_staging_size(view.spec().logical_nbytes()), 32}` with checked logical-byte computation and overflow propagation.
- Centralize reusable validation and lease machinery for required capacity, base/range alignment, checked offsets, exact `Device` identity, exact live owner identity, and workspace range overlap with every operand/output storage range. Reject malformed, undersized, misaligned, foreign, dead, or workspace-versus-operand/output-overlapping ranges as invalid input (`std::invalid_argument` / `OidError::InvalidArgument`).
- Provide reusable registry primitives that atomically acquire an exclusive workspace-range lease alongside existing owner-registration state. Overlapping leases on views of the same owner are resource exhaustion (`std::bad_alloc` / `OidError::ResourceExhausted`); disjoint aligned subranges may coexist. A failed lease/registration transaction leaves neither partial state nor allocator changes. Exercise these primitives directly with focused fake registrations; execution-facade integration belongs to tasks 07, 08, and 11.
- A lease primitive must retain owner/range lifetime until its caller supplies a covering completion proof. Runtime failure alone is not proof. Unknown completion quarantines the unresolved range and associated lifetime state until a later proof, without returning it to the data allocator or reusing it.
- Do not hold allocator locks across submission, waits, callbacks, or worker drains. Device-owned bookkeeping must make arena ranges reusable only after proof, preserve stable tensor/workspace addresses, and isolate identities across independent `Device` objects.
- Add focused regression coverage and migrate only callers needed to exercise this contract. Coverage must observe that pure queries cause no allocation, registration, lease, token, or queue traffic; the default view is empty; and positive/zero range validation follows the stated alignment. It must reject undersized, misaligned, foreign, dead, overlapping, and exclusively leased views, allow disjoint concurrent subranges, and prove that the lease primitive's completion/quarantine result controls reuse. Existing binary/transfer execution and staging remain unchanged until tasks 07/11.

## Non-goals

- Changing the four binary or two host-transfer execution facade signatures or removing existing staging allocations; those cutovers belong to tasks 07 and 11.
- Tiled `Tensor` representation, view layout, arithmetic, broadcasting semantics, or rank policy owned by other tasks.
- Implicit workspace allocation, automatic growth, arena compaction, a generic scratch framework, or public operation/direction enums.
- A TTNN native-storage redesign, CPU device scratch allocation, or hidden internal locks in canonical standalone allocators.
- Changes to queue admission, metadata-slot design, or unrelated device factory configuration beyond what is required to create the workspace against the completed data arena.

## Acceptance criteria

- [ ] Compile-time checks prove `RawWorkspace` is non-copyable/non-movable and `RawWorkspaceView` is copy-constructible and non-retargetable; the default view is empty and exposes exact owner/device identity and checked range information without backend runtime types.
- [ ] Standard GPU creation suballocates the existing data arena, preserves stable addresses, produces no new native backing allocation at positive creation or arena exhaustion, and returns a valid allocation-free empty owner/view for zero bytes. CPU and TTNN accept empty workspaces and reject positive scratch without dummy storage.
- [ ] Query tests prove exact `{0,1}` binary results on CPU, TTNN, CUDA, and ROCm; exact `{0,1}` transfer results on CPU and TTNN; exact checked SYCL binary staging sums; exact CUDA/ROCm/SYCL transfer staging sizes with alignment 32; deterministic overflow handling; and no allocation, registration, lease, token, queue, or capacity-dependent side effects.
- [ ] Conformance rejects foreign-device/foreign-owner, dead-owner, malformed, out-of-range, undersized, misaligned, and workspace/operand/output-overlapping views as invalid input; an exclusively leased overlap reports resource exhaustion; disjoint aligned subranges of one owner run concurrently.
- [ ] Fake acceptance transactions prove rollback leaves no lease or owner registration, a retained lease prevents reuse until covering proof, and unknown completion quarantines the range without unsafe reuse. Actual parked-request integration belongs to tasks 08–10.
- [ ] Existing binary and host-transfer numerical behavior, padding preservation, and zero-workspace paths remain unchanged while this task leaves facade migration and staging removal to tasks 07/11.

## Verification

- `ctest --test-dir <build> --output-on-failure -R '^(iom_tests|iom_backend_conformance_cpu_tests)$'` (not run; supervisor gate).
- `ctest --test-dir <remote-build> --output-on-failure -R '^iom_(cuda|rocm|sycl|ttnn)_(smoke|conformance)_tests$'` through the `remote-development` workflow for each enabled accelerator backend (not run; supervisor gate).

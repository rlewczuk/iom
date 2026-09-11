# Cut binary operations over to explicit workspace

**Order:** 07
**Priority:** P1 — this delivers the required binary behavior after workspace ownership exists.
**Blocked by:** `05-raw-workspace-contract`
**Source:** `docs/changes/004-memory-simplify/spec.md`

## Outcome

All four binary operations use an explicit borrowed `RawWorkspaceView` at the public and protected boundaries. Common validation, operation-specific workspace sizing, acceptance leasing, backend execution, and caller migration agree on one contract: a positive workspace requirement is never satisfied by an implicit allocation, while a zero requirement remains source-simple with the empty default view. Binary operations retain existing arithmetic, dtype, broadcast, alias, tile, scalar, and padding behavior.

## Scope

This task covers the atomic cutover of `DeviceOps::add`, `DeviceOps::mul`, `DeviceOps::sub`, and `DeviceOps::div`, their protected binary request/hook path, every concrete backend implementation, direct callers, and focused shared/backend regression coverage. It covers workspace validation and exclusive range leasing from current acceptance through completion, failure, and quarantine. Tasks 08–10 later preserve that lease while requests are parked.

For SYCL, retain the current host scalar algorithm and its three host-USM staging buffers and ordering. Replace only the three per-operation device-USM allocations with non-overlapping slices of the caller workspace. Preserve `binary_view_staging_bytes` addressed-plane padding and gaps, including output padding. The workspace is device arena memory and is never treated as host USM.

## Implementation references

- Public binary facades and protected `BinaryOperation`, `BinaryRequest`, and `binary_impl`: `include/iom/iom.hpp:246-318`.
- Common binary validation, snapshot construction, and OID/error mapping: `src/iom.cpp:705-960`.
- SYCL `binary_view_staging_bytes`, host scalar execution, transfer ordering, and current staging allocations: `src/sycl/copy.cpp:509-535,638-781`.
- Concrete binary implementations to migrate: `src/cpu/device.cpp:804`, `src/shared/gpu_queue.hpp:300`, `src/sycl/copy.cpp:481`, and `src/ttnn/device.cpp:605`.
- Shared operation-neutral binary callers and compile-time signature checks: `test/backend/backend_conformance_common.hpp:297-324,394-622`.
- Backend binary callers and conformance drivers: `test/cpu/test_cpu_conformance.cpp:223-249`, `test/cuda/test_cuda_conformance.cpp:362-384`, `test/rocm/test_rocm_conformance.cpp:516-540`, `test/sycl/test_sycl_conformance.cpp:695-713`, and `test/ttnn/test_ttnn_conformance.cpp`.
- Existing common/core binary validation and lifetime tests: `test/test_iom.cpp:869,1894-2095`.
- Exact CTest names are defined in `test/CMakeLists.txt:17,36,57,96-176,242`.

## Requirements

1. **One consistent API.** Change each planned public signature to accept a final borrowed argument, `RawWorkspaceView workspace = {}`:
   - `DeviceOps::add(lhs, rhs, out, workspace)`;
   - `DeviceOps::mul(lhs, rhs, out, workspace)`;
   - `DeviceOps::sub(lhs, rhs, out, workspace)`; and
   - `DeviceOps::div(lhs, rhs, out, workspace)`.

   The protected binary request and hook path must carry and consume the same borrowed workspace view. Planned symbols include operation-specific `DeviceOps::add_workspace_requirements(lhs, rhs, out)`, `mul_workspace_requirements(lhs, rhs, out)`, `sub_workspace_requirements(lhs, rhs, out)`, and `div_workspace_requirements(lhs, rhs, out)`, each returning `WorkspaceRequirements`. Do not add a public operation enum, direction enum, or duplicate query API on `Device`/workspace owners. Keep `Device::create_ops()` argument-free.

2. **Pure, matching queries.** Each operation query must perform the same request/spec/view/result validation needed by its corresponding facade and return deterministic bytes and alignment without allocating, reserving a metadata slot or sequence, registering a token, leasing workspace, mutating output, or submitting native work. The four query/facade pairs must agree for every supported operation, dtype, broadcast mapping, transformed view, and backend. A positive result must not trigger an automatic workspace allocation when the facade receives `{}`.

3. **Validation precedence and error categories.** The common facade must validate operands, output shape, dtype/quantization, device identity, view geometry, alias rules, and operation support before any sequence/token reservation or native effect, then validate the workspace requirement before acceptance. For a positive requirement, missing, undersized, misaligned, checked-range-invalid, foreign-device, foreign-owner, dead, or operand/output-overlapping workspace is invalid input: throwing paths use `std::invalid_argument` and OID facades return `OidError::InvalidArgument`. Preserve the established precedence for non-workspace errors. An already exclusively leased workspace range is resource exhaustion: throwing paths use `std::bad_alloc` and OID facades return `OidError::ResourceExhausted`. Every invalid or conflicting call consumes no sequence/token capacity, performs no native effect, and does not mutate output.

4. **Atomic acceptance and lifetime.** On acceptance, snapshot the binary request and workspace range plus exact workspace owner/device identity, register the raw workspace owner, and lease the supplied range exclusively. Disjoint ranges of one owner may be accepted concurrently. Workspace leasing, owner registration, sequence/token reservation, queue-node/snapshot preparation, and any other current admission state must roll back transactionally if acceptance fails. A positive request never dispatches without its validated lease. Runtime or asynchronous failure retains the lease and all required lifetime protection until a covering completion proof; only then may the lease be released. Unknown completion quarantines the workspace range rather than returning it to the allocator. Repeated waits observe the original terminal result and cannot observe a later reuse. The parked-state extension is owned by tasks 08–10.

5. **Backend behavior.** Update CPU, shared CUDA/ROCm, SYCL, and TTNN implementations and all direct callers to consume the captured borrowed workspace. CPU, TTNN, CUDA, and ROCm binary requirements remain zero: their four default-empty paths accept `{}` and allocate no scratch or IOM device memory. No backend may silently allocate a replacement workspace, use the metadata arena as payload scratch, or change arithmetic/dtype/broadcast/padding behavior.

6. **SYCL requirement geometry.** For a captured binary request, compute `lhs_bytes`, `rhs_bytes`, and `out_bytes` with the existing `binary_view_staging_bytes` semantics: storage extends through the highest addressed plane, retains untouched gaps, and includes tile padding. Use checked `align_up` with `A = 32` and require workspace capacity of
   `align_up(lhs_bytes, A) + align_up(rhs_bytes, A) + out_bytes`.
   The exact device-workspace slices are:
   - lhs at offset `0`, length `lhs_bytes`;
   - rhs at offset `align_up(lhs_bytes, 32)`, length `rhs_bytes`; and
   - output at offset `align_up(lhs_bytes, 32) + align_up(rhs_bytes, 32)`, length `out_bytes`.

   Require a 32-aligned workspace base and checked offsets/capacity; prove that the slices do not overlap. Use those device-USM slice addresses for the existing device copy kernels and output copy-back. Keep `lhs_stage`, `rhs_stage`, and `out_stage` as separate host-USM allocations, preserve their current transfer/host-scalar/transfer ordering, and wait boundaries. Never call `sycl::malloc_device` or `sycl::free` for these slices, never reinterpret arena memory as host USM or a host span, never switch the arena to shared USM, and never suballocate/free a slice independently. The operation lease, not a native free, governs slice release.

7. **Caller and conformance migration.** Migrate every binary call site, fake/probe queue, backend driver, and compile-time signature assertion to pass the workspace where required and to call the matching operation query. Shared conformance must exercise all four operation/query pairs, exact SYCL slice sizing, default-empty zero-requirement paths, transformed/broadcast views, owner/device identity, overlap and alignment validation, exclusivity conflicts, repeated waits, and failure/quarantine retention. Keep backend-specific native allocation instrumentation at the driver boundary and shared behavioral assertions in `test/backend`.

## Non-goals

- Do not implement raw workspace ownership or alter `RawWorkspace`/`RawWorkspaceView` beyond consuming the task-05 contract.
- Do not implement parking/FIFO admission, host-transfer workspace APIs, allocator synchronization, device arenas, queue/resource setup, or direct SYCL binary kernels. Consume the completed workspace/arena contracts; tasks 08–10 add parked admission without redesigning this binary cutover.
- Do not add new operations, public operation/direction enums, duplicate owner query APIs, or implicit scratch allocation.
- Do not alter arithmetic, supported dtype/quantization policy, broadcasting, alias policy, tiled addressing, host scalar semantics, transfer ordering, or padding preservation.
- Do not reinterpret arena memory as host USM, use shared USM as an escape hatch, or change the three host-USM staging-buffer ownership model.

## Acceptance criteria

- All four public binary APIs, protected binary request/hook path, and four operation-specific queries have matching workspace behavior and compile-time/call-site coverage.
- Positive requirements reject missing, undersized, misaligned, foreign, dead, operand/output-overlapping, and otherwise invalid views with `InvalidArgument` before token/sequence reservation or any native/output effect. A conflicting leased range reports `ResourceExhausted`/`std::bad_alloc` and acceptance rollback leaves no partial lease or token.
- Accepted requests snapshot workspace range and owner identity, retain leases through proven completion, and retain/quarantine them across runtime failure until proof. Repeated waits preserve the original result and workspace ranges are reusable only after proof; tasks 08–10 add parked retention.
- SYCL positive binary cases accept the exact required capacity and exact non-overlapping 32-aligned slices. From acceptance through repeated waits, IOM device allocation/free instrumentation observes no device allocation or free; no slice is independently freed. `binary_view_staging_bytes` gaps, untouched planes, output padding, host scalar values, and existing host-USM ordering match the independent existing oracle.
- CPU, TTNN, CUDA, and ROCm zero-requirement binaries accept the default empty view and remain allocation-free for scratch. All supported binary arithmetic, dtype, broadcast, transformed-view, and padding results remain unchanged.
- Focused regression coverage proves invalid/conflicting precedence and no effects, successful leasing/retirement, runtime-failure retention/quarantine, exact SYCL capacity geometry, and all four operation/query pairs. Shared conformance covers behavior; backend drivers cover native allocation boundaries. Parking-specific coverage belongs to tasks 08–10.

## Verification

Proposed gates (not run in this spec-writing task):

- Local core and CPU: `ctest --test-dir build --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests|iom_scalar_add_tests)$'`.
- Enabled accelerator conformance, executed through the `remote-development` workflow with hardware required (enabled tests must fail rather than skip): `ctest --test-dir build --output-on-failure -R '^(iom_cuda_conformance_tests|iom_rocm_conformance_tests|iom_sycl_conformance_tests|iom_ttnn_conformance_tests)$'`.
- If an enabled backend's target is present, run that exact target from the second command; do not claim disabled-backend results. Inspect doctest cases for binary workspace validation, exact SYCL slice/allocation instrumentation, failure lease retention, and output/padding oracle evidence.

# Single-source the binary submission facade

**Order:** 05
**Priority:** P1 — four duplicated validation/lifetime shells are a divergence surface
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` — whole-codebase review, branch `main`, HEAD `cb88619865ff9bcda0a4e3bac4bae4bd4bf510a8` (`Move cpp-inference-review skill to .omp`), baseline `origin/main`; clean at scope capture; upstream `origin/main` ahead 4/behind 0
**Finding:** AR-002
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** strongly-supported, confidence 96
**Review scope:** whole-codebase
**Backend scope:** common DeviceOps submission boundary and CPU, CUDA, ROCm, SYCL, TTNN queues
**Location:** `src/device_ops_binary.cpp:329-421` — `DeviceOps::add`, `mul`, `sub`, and `div`

## Outcome

The four required public binary methods remain unchanged at the API boundary, while one private common submission facade owns validation, workspace validation, request capture, dispatch, and failure mapping; each wrapper varies only its `BinaryOperation` value.

## Current problem

`DeviceOps::add`, `mul`, `sub`, and `div` contain identical try/catch shells (`src/device_ops_binary.cpp:329-421`): `validate_binary`, `binary_workspace_requirements`, three-view `WorkspaceValidation::validated`, `BinaryRequest` reconstruction, `binary_impl`, `invoke`, and `invoke_failure`. Only the operation enum differs. No backend overrides a public facade; backend queues consume the same `BinaryRequest` and operation enum while retaining legitimate arithmetic and representation differences. This is verified structural duplication at a high-risk validation/ownership boundary: a future correction can land in one copy and silently diverge in the other three. No current numerical or lifetime failure is asserted.

## Scope

- Add one distinctly named private nonvirtual helper, such as `submit_binary_operation`, taking `BinaryOperation`, the three views, and `RawWorkspaceView`, and move the existing shared submission shell into it.
- Replace public `add`, `mul`, `sub`, and `div` bodies with forwarding wrappers that select only Add/Mul/Sub/Div; preserve public signatures, `noexcept`, argument order, OID mapping, and all backend dispatch.
- Keep the existing protected template `submit_binary` as the admission/registration/lease mechanism; this helper must not overload or rename it.

## Implementation references

- **Modify:** `src/device_ops_binary.cpp:329-421` — extract one common private helper and delete only the four repeated bodies.
- **Modify:** `include/iom/iom.hpp` — declare the distinctly named private helper in `DeviceOps`; retain public declarations at `:121-132` and protected templated `submit_binary`/`BinaryOperation`/`BinaryRequest` contracts.
- **Read:** `src/device_ops_binary.cpp:245-321` — `validate_binary` and snapshots; preserve validation precedence and immutable request capture.
- **Read:** `include/iom/iom.hpp:246-270,357-408` — virtual backend hooks and existing protected admission template; do not conflate the new facade with the admission helper.
- **Read:** backend `binary_impl` counterparts in `src/cpu/queue.cpp:235`, `src/shared/gpu_queue_operations.inl:65`, `src/sycl/queue.cpp:263`, and `src/ttnn/queue.cpp:144`; retain backend differences.
- **Tests:** `test/test_iom.cpp:899-940`, common conformance dispatch `test/backend/backend_conformance_common.hpp:344-376`, coexistence, retained-failure, and workspace cases.

## Requirements

- Add one private nonvirtual helper with a distinct name such as `submit_binary_operation`; do not name it `submit_binary`, because the protected template already uses that name.
- Preserve the exact helper flow and ordering: `queue_device`, `validate_binary` with the selected enum, `binary_workspace_requirements`, three-view `WorkspaceValidation::validated`, `BinaryRequest` field/copy construction, `invoke(binary_impl(...))`, and catch-all `invoke_failure`.
- Make each public wrapper one-line enum forwarding while retaining all four explicit public methods, signatures, `noexcept`, workspace defaults, operation enum values, and argument order.
- Delete repeated bodies only. Keep validation/alias/broadcast/capability logic, workspace ranges and leases, request snapshots, OID/exception mapping, queue admission/ordering, repeated waits, retained failures, all backend `binary_impl`s, and the four pure workspace-query wrappers unchanged.

## Non-goals

- Do not remove or merge public add/mul/sub/div methods, alter capabilities/arithmetic/broadcasting/alias policy, change workspace query APIs, queue/admission/fence/registry state, staging, native execution, or conformance matrix.
- Do not refactor the separate pure `*_workspace_requirements` wrappers in this task.
- Do not rename or redesign the existing protected templated `submit_binary` admission helper.

## Acceptance criteria

- [ ] Each public method preserves its ABI/signature and selects only its intended enum; valid and invalid calls across fake and real backends retain validation precedence, diagnostics/categories, no-effect rejection, OID values, workspace lease/overlap behavior, request snapshots, dispatch, numerical outputs, repeated waits, and retained failures.
- [ ] Source has exactly one common binary submission shell in the new distinctly named private helper; the protected admission template remains separate and all backend-specific implementations and operation switches remain unchanged.

## Verification

- `ctest --test-dir <cpu-build> --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` plus enabled backend conformance/coexistence targets; accelerator execution must use remote-development.
- Exercise fake-queue valid, unsupported, overflow, foreign, alias, broadcast, workspace, and retained-failure cases for all four enums, recording no-effect/sequence behavior; perform a source check that only one common submission shell exists. Supplied local CPU targets passed 3/3; accelerator sync attempts timed out before build and no accelerator result is claimed here.

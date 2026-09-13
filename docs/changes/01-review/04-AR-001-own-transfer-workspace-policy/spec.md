# Move host-transfer workspace policy to tensor owners

**Order:** 04
**Priority:** P1 — common backend leakage creates a concrete divergence source
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` — whole-codebase review, branch `main`, HEAD `cb88619865ff9bcda0a4e3bac4bae4bd4bf510a8` (`Move cpp-inference-review skill to .omp`), baseline `origin/main`; clean at scope capture; upstream `origin/main` ahead 4/behind 0
**Finding:** AR-001
**Review area:** Backend architecture & simplicity
**Review severity:** medium
**Review verification:** strongly-supported, confidence 92
**Review scope:** whole-codebase
**Backend scope:** common plus CPU, CUDA, ROCm, SYCL, and TTNN
**Location:** `src/tensor_view.cpp:289-323` — anonymous `host_transfer_workspace_requirements`; `TensorView::copy_from_host_workspace_requirements` and `copy_to_host_workspace_requirements`

## Outcome

Each concrete Tensor owner supplies its host-transfer workspace policy through one const hook receiving checked logical bytes; common TensorView remains backend-neutral while both public direction query names retain their exact values, overflow behavior, and pure-query semantics.

## Current problem

The common TensorView implementation computes checked `spec().logical_nbytes()` and then switches on `view.backend_kind()` to return `{0,1}` for CPU/TTNN or `{gpu_algorithm::compute_staging_size(logical_nbytes),32}` for CUDA/ROCm/SYCL (`src/tensor_view.cpp:295-310`). Both public queries delegate to this helper (`:315-323`). This is a concrete common-layer policy leak and omission/divergence source: a backend can compile while being absent from the switch or receive stale staging rules. `AGENTS.md:15` and `docs/ARCHITECTURE.md:38-40` prohibit backend-kind switches in common code, while each concrete Tensor already owns the transfer boundary. The current query is pure and has no present wrong-result reproduction; the remediation must preserve that behavior.

## Scope

- Add one const backend-owned Tensor hook accepting the already checked logical byte count and move the current CPU/TTNN versus standard-GPU policy to every concrete Tensor and `FakeTensor`.
- Keep both `TensorView::copy_from_host_workspace_requirements` and `copy_to_host_workspace_requirements` public names, checked logical-byte computation, exact values, overflow/error propagation, and no allocation/registration/lease/token/queue/native effect.
- Preserve direct CPU transfer, TTNN native-plane transfer, CUDA/ROCm shared staging, SYCL staging, transfer locks, leases, completion proof, and cleanup; derived classes must rebuild, with no mixed-version ABI promise.

## Implementation references

- **Modify:** `include/iom/tensor.hpp` — `Tensor` protected hooks and `TensorView` query declarations; add one const virtual owner policy hook using checked logical bytes without changing the two public query names.
- **Modify:** `src/tensor_view.cpp:289-323` — delete anonymous `host_transfer_workspace_requirements` and its `BackendKind` switch; compute `spec_.logical_nbytes()` once in each public query and delegate to the owner hook.
- **Modify:** `src/cpu/tensor.cpp`, `src/cuda/device_tensor.cpp`, `src/rocm/device_tensor.cpp`, `src/sycl/device_tensor.cpp`, and `src/ttnn/device.cpp` — implement current owner policies: CPU/TTNN `{0,1}`, CUDA/ROCm/SYCL `{gpu_algorithm::compute_staging_size(checked_logical_nbytes),32}`.
- **Modify:** `test/test_iom.cpp:734-776` — implement the hook in `FakeTensor` and use a distinctive owner value in the focused pure-query case.
- **Read:** `include/iom/tensor.hpp:391-399` and `src/shared/standard_tiled_copy.inl:389-440` — retain the existing owner transfer boundary and checked staging utility.
- **Tests:** backend query/conformance and coexistence tests; query callers in `test/backend/backend_conformance_common.hpp`, backend tests, and `tools/sycl_add_staging_measure.cpp`.

## Requirements

- Add exactly one const backend-owned Tensor hook that receives checked `logical_nbytes`; it must be pure and preserve checked arithmetic/error behavior. Do not add a registry, backend selector, direction enum, cache, allocation, synchronization, or second policy mechanism.
- Have each current concrete Tensor and `FakeTensor` implement the hook. CPU and TTNN return `{0,1}`; CUDA, ROCm, and SYCL call the existing `gpu_algorithm::compute_staging_size` and return alignment 32.
- Make both public direction query methods compute `spec_.logical_nbytes()` with existing validation and delegate to the owner hook. Preserve their names, public signatures, direction-independent current values, BOOL transfer-time validation boundary, and pure no-effect contract.
- Delete the common anonymous helper and its `BackendKind` policy switch. `BackendKind` and `backend_device` identity accessors remain.
- Rebuild all derived Tensor classes and test doubles because the pure virtual protected hook changes source/vtable requirements; do not promise mixed-version ABI compatibility, consistent with the repository contract.

## Non-goals

- Do not remove identity accessors, unify CPU/TTNN/standard-GPU representations, alter transfer direction APIs, host validation, staging allocation/lease policy, queue synchronization, capability tables, or native cleanup.
- Do not add a common transfer framework, allocator, registry, cache, or device-global policy selector.

## Acceptance criteria

- [ ] Both public query names return exactly the prior `{0,1}` or checked standard-GPU staging/alignment values for every current backend and transformed view, and checked overflow/errors remain identical; a `FakeTensor` reporting `BackendKind::CPU` can return a distinctive owner-hook value without common TensorView policy changes.
- [ ] Queries perform no allocation, registration, lease, token, queue, native, staging, or mutex effect; host transfers and BOOL checks remain unchanged, and no production common `BackendKind` policy switch remains while all concrete owners compile and implement the hook.

## Verification

- `ctest --test-dir <cpu-build> --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` plus enabled backend query/conformance/coexistence targets; use remote-development for accelerator hosts.
- Run the FakeTensor owner-policy/overflow/purity scenario and a source check for no production `BackendKind` switch in common `src`; supplied local CPU validation already passed `iom_tests`, `iom_scalar_add_tests`, and `iom_backend_conformance_cpu_tests` (3/3), while accelerator sync attempts timed out before build and are not claimed as run for this task.

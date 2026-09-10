# Select binary operation once per request

**Order:** 09
**Priority:** P1 — satisfies explicit once-per-request contract and removes per-element dispatch
**Blocked by:** `06-NT-003-cover-real-queue-mul-sub-div-values`, `07-AR-002-unify-binary-scalar-codec`
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** AR-003
**Review area:** Backend architecture & simplicity
**Review severity:** medium
**Review verification:** strongly-supported, confidence 93
**Review scope:** whole-codebase
**Backend scope:** CUDA, ROCm/HIP, and TTNN
**Location:** `src/shared/standard_tiled_add.inl` — `BinaryMetadata`, `add_integer`, `add_value`, `binary_body`, and launch path; `src/ttnn/copy.cpp` — `binary_planes` per-element lambda/switch; once-per-request counterparts `src/cpu/device.cpp` and `src/sycl/copy.cpp`

## Outcome

For each immutable binary request, the host selects one operation-specialized GPU entry/kernel and TTNN selects one operation-specialized `binary_planes` traversal before entering logical elements. GPU metadata no longer carries a runtime operation, TTNN has no per-element operation switch or indirect function pointer, and all four operation results, ownership, staging, and completion semantics remain unchanged.

## Current problem

The 003 contract requires arithmetic selection once per request outside logical loops. GPU `BinaryMetadata` stores `operation`; `add_integer` and `add_value` branch on it for each packed field, and `binary_body` passes `m.operation` for every field. TTNN's `binary_planes` creates a scalar lambda and switches on `request.operation` inside every logical element. `BinaryRequest` is immutable and validated before submission, so no accepted request can legitimately mix operations. CPU and SYCL already select operation-specialized implementations once before traversal. This source-level contract breach and repeated work are established; exact wall-time benefit and compiler elimination are not. Root CPU and remote CUDA/ROCm/TTNN smoke/conformance/coexistence passed, but no generated-assembly/profile gate ran.

## Scope

- Make the GPU entry/kernel and scalar arithmetic operation-specialized, with one host-side selection per request for ADD/MUL/SUB/DIV.
- Remove `BinaryMetadata`'s operation field and runtime operation parameters/branches from `add_integer`/`add_value`; retain one packed-word kernel/traversal body and existing policy launch.
- Select `binary_planes` once in `TtnnQueue::execute`, delete the per-element lambda/switch and dead TTNN enum/translation if no longer used, while retaining the common immutable `BinaryRequest::operation`, TTNN native staging, and API mutex.

## Implementation references

- **Modify:** `src/shared/standard_tiled_add.inl` — `BinaryMetadata`, `add_integer`, `add_value`, `binary_body`, `grid_stride_binary_kernel`, and `launch_grid_stride_binary`; operation specialization and metadata deletion belong here.
- **Modify:** `src/shared/gpu_queue.hpp`, `src/cuda/copy.cu`, and `src/rocm/copy.hip` — host-side operation selection and policy-specific entry instantiation; preserve CUDA/ROCm shared source.
- **Modify:** `src/ttnn/copy.hpp`, `src/ttnn/copy.cpp`, and `src/ttnn/device.cpp` — select operation-specialized `binary_planes` once in `execute`; remove dead `BinaryOperation` adapter only if no caller remains.
- **Read:** `src/cpu/device.cpp` and `src/sycl/copy.cpp` — positive once-per-request dispatch counterparts.
- **Tests:** CUDA/ROCm/TTNN operation conformance, mapping, alias, ownership, repeated-wait, and failure tests; NT-003 supplies real-queue MUL/SUB/DIV coverage.

## Requirements

- Select operation-specialized GPU entry/kernel on the host before traversal. Remove `BinaryMetadata::operation` and all per-element runtime operation branches/arguments; do not replace them with a per-element function pointer.
- Select `binary_planes` once in `TtnnQueue::execute` before its element loop. Delete the dead TTNN internal enum/translation when no longer used, but retain common `BinaryRequest::operation` and the native API/storage boundary.
- Preserve numerical semantics from AR-002/NT-001, mapping from AR-001, metadata sizing apart from the removed operation field, launch geometry, owner registration, staging, API mutex, async completion, repeated waits, and failure handling. Add no allocation, transfer, or synchronization.

## Non-goals

- Do not merge CUDA/ROCm with TTNN storage or queue semantics, remove common validation, alter capability/error policy, or change scalar numeric rules.
- Do not change kernel mapping, staging, ownership, completion, or operation domains beyond moving dispatch outside loops.
- Do not add a runtime framework, virtual dispatch, or an indirect per-element function call.

## Acceptance criteria

- [ ] Every accepted GPU request selects one operation-specialized entry/kernel before logical traversal; `BinaryMetadata` has no runtime operation and no `add_integer`/`add_value` operation switch remains.
- [ ] TTNN has one execute-level operation selection and no per-element lambda/switch or per-element function pointer; dead internal enum/translation is deleted when unused.
- [ ] CUDA, ROCm, and TTNN all-operation outputs, broadcasts, aliases, views, errors, owner registration, async completion, and repeated waits remain correct with unchanged metadata/resource behavior.
- [ ] Real-queue NT-003 tests and AR-002 shared codec are in place before accepting this dispatch cutover; no allocation, wait, transfer, or synchronization is added.

## Verification

- `cmake --build <cuda-build> --target iom_cuda_conformance_tests && ctest --test-dir <cuda-build> --output-on-failure -R '^iom_cuda_conformance_tests$'` — run all CUDA operations, mappings, tails, broadcasts, views, aliases, and storage-oracle cases.
- `cmake --build <rocm-build> --target iom_rocm_conformance_tests && ctest --test-dir <rocm-build> --output-on-failure -R '^iom_rocm_conformance_tests$'` — run the equivalent ROCm cases.
- `cmake --build <ttnn-build> --target iom_ttnn_conformance_tests && ctest --test-dir <ttnn-build> --output-on-failure -R '^iom_ttnn_conformance_tests$'` — run TTNN ADD/MUL/SUB/DIV plus repeated-wait and failure cases.
- Inspect source and generated device code or profile a representative request to prove request-level specialization and absence of per-element operation branches; compare metadata size, launch count, allocations, waits, and outputs before/after.
- Apply a temporary wrong-operation selection and require NT-003/operation-value tests to fail, then remove it. Root-supplied CPU and remote CUDA/ROCm/TTNN smoke/conformance/coexistence passed; no candidate-specific assembly, profile, mutation, sanitizer, or benchmark has run.

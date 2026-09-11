# Verify split-file integration

**Order:** 21
**Priority:** P2 — required cross-cutting build, ODR, hardware, and line-cap proof after every factoring slice
**Blocked by:** `01-core-tensor-view`, `02-core-workspace`, `03-core-device-ops-binary`, `04-core-device-ops-copy-neural`, `05-core-device-ops-cutover`, `06-sycl-device`, `07-ttnn-device-types`, `08-ttnn-testing`, `09-staged-worker-header`, `10-outstanding-work-registry-headers`, `11-cpu-device-components`, `12-cuda-device-components`, `13-rocm-device-components`, `14-sycl-queue-core`, `15-sycl-queue-binary`, `16-ttnn-queue-core`, `17-ttnn-queue-copy`, `18-ttnn-binary`, `19-standard-tiled-copy-metadata`, `20-gpu-queue-implementation`
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

The complete split-file change is integrated and proven as one build graph. Every planned destination is listed exactly where its owning library requires it, `src/iom.cpp` is gone, no test source list or public factory header was changed, and every moved definition has one and only one linked definition. Existing umbrella include paths remain usable, private template fragments compile in every intended CUDA/ROCm/SYCL consumer without visibility or ODR regressions, and the CPU plus all configured accelerator libraries and existing smoke/conformance/coexistence tests pass. The one-time production line scanner proves that every original, touched, and new production source/header fragment is below the 500-line threshold. If required hardware is unavailable, the corresponding verification fails; it is never reported as a skip or pass.

## Scope

This is a final integration verification and narrowly scoped repair task only:

- Inspect the final root `CMakeLists.txt` source lists and confirm `libiom` contains `tensor.cpp`, `tensor_view.cpp`, `workspace.cpp`, `device_ops.cpp`, `device_ops_copy.cpp`, `device_ops_binary.cpp`, and `device_ops_neural.cpp`, and does not contain obsolete `src/iom.cpp`. Confirm each enabled backend target lists every planned destination and retained source from the factoring plan.
- Confirm no test target source list and no public backend factory header changed. Repairs are limited to split-caused CMake entries, include/dependency edges, visibility, or link/ODR defects; do not alter behavior or add tests.
- Prove exact-once definitions and link resolution for moved symbols, including core `Tensor*`, `TensorView`, `RawWorkspace`, `WorkspaceValidation`, `DeviceOps`, CPU device/tensor/queue symbols, each CUDA/ROCm/SYCL/TTNN moved symbol, registry components, staged-worker definitions, tiled metadata symbols, and GPU queue template members. `src/iom.cpp` must be absent rather than retained as a compatibility translation unit.
- Compile and link umbrella-only consumers of `include/iom/iom.hpp` and `include/iom/detail/outstanding_work_registry.hpp`; preserve their existing transitive availability without adding duplicate include ownership or public compatibility aliases.
- Compile and link CUDA and ROCm consumers of `src/shared/gpu_queue.hpp`, `src/shared/standard_tiled_copy.inl`, and `src/shared/standard_tiled_copy_metadata.inl`, and SYCL consumers of the included tiled-copy fragments. Verify private symbols remain private, included `.inl` definitions are emitted exactly once per intended backend, and no cross-backend or multi-translation-unit ODR violation is introduced.
- Run the exact local CPU configure/build/test gates and the remote-development CUDA, ROCm, SYCL, and TTNN configure/build/test gates. Each backend library and its smoke, conformance, and coexistence targets must build and link before its exact CTest expression runs.
- Run a deterministic one-time scanner over tracked production files under `src/` and `include/` whose suffix is `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl`. It must assert at most 499 physical lines for every original oversized file and every touched or new production file. Remove any temporary scanner after use; do not add a permanent test or test source.

No production factoring is designed here, no new tests or documentation are added, and no unrelated cleanup is allowed.

## Implementation references

- `CMakeLists.txt` — planned final `IOM_SOURCES`, `IOM_HEADERS`, and `iom_cuda`, `iom_rocm`, `iom_sycl`, and `iom_ttnn` source lists.
- `test/CMakeLists.txt` — existing test target source lists and registrations; this file is verification-only and must remain unchanged.
- `src/tensor.cpp` (planned), `src/tensor_view.cpp` (planned), `src/workspace.cpp` (planned), `src/device_ops.cpp` (planned), `src/device_ops_copy.cpp` (planned), `src/device_ops_binary.cpp` (planned), `src/device_ops_neural.cpp` (planned), and `src/iom_internal.hpp` (planned) — core split definitions and private linkage.
- `src/cpu/device.cpp`, `src/cpu/tensor.cpp` (planned), `src/cpu/queue.cpp` (planned), `src/cpu/device_internal.hpp` (planned), and `src/cpu/transfer_helpers.hpp` (planned) — CPU split and private cross-translation-unit declarations.
- `src/cuda/device.cpp`, `src/cuda/device_tensor.cpp` (planned), `src/cuda/device_internal.hpp` (planned), `src/rocm/device.cpp`, `src/rocm/device_tensor.cpp` (planned), and `src/rocm/device_internal.hpp` (planned) — mirrored but independent CUDA/ROCm splits.
- `src/sycl/device.cpp`, `src/sycl/device_tensor.cpp` (planned), `src/sycl/device_internal.hpp` (planned), `src/sycl/copy.cpp`, `src/sycl/queue.cpp` (planned), `src/sycl/queue_binary.cpp` (planned), `src/sycl/queue_support.hpp` (planned), and `src/sycl/queue_internal.hpp` (planned) — SYCL device, copy, and queue split.
- `src/ttnn/device.cpp`, `src/ttnn/device_types.cpp` (planned), `src/ttnn/device_internal.hpp` (planned), `src/ttnn/copy.cpp`, `src/ttnn/queue.cpp` (planned), `src/ttnn/queue_copy.cpp` (planned), `src/ttnn/binary.cpp` (planned), `src/ttnn/testing.cpp` (planned), `src/ttnn/queue_internal.hpp` (planned), and `src/ttnn/testing_internal.hpp` (planned) — TTNN split.
- `include/iom/iom.hpp` and `include/iom/detail/staged_worker.hpp` (planned) — public umbrella and extracted staged-worker implementation.
- `include/iom/detail/outstanding_work_registry.hpp`, `include/iom/detail/fence.hpp` (planned), `include/iom/detail/outstanding_work_registry_core.hpp` (planned), `include/iom/detail/outstanding_work_cleanup.hpp` (planned), and `include/iom/detail/workspace_registry.hpp` (planned) — registry umbrella and private component headers.
- `src/shared/gpu_queue.hpp`, `src/shared/gpu_queue_lifecycle.inl` (planned), `src/shared/gpu_queue_operations.inl` (planned), `src/shared/standard_tiled_copy.inl`, and `src/shared/standard_tiled_copy_metadata.inl` (planned) — included template implementation fragments and their CUDA/ROCm/SYCL consumers.
- `include/iom/{cpu,cuda,rocm,sycl,ttnn}/device.hpp` — public factory headers; they are explicitly unchanged.
- Existing consumers include `src/cpu/device.cpp`, `src/cuda/copy.cu`, `src/rocm/copy.hip`, `src/sycl/copy.cpp`, `src/shared/gpu_queue.hpp`, `src/shared/event_ring.hpp`, backend `copy.hpp` files, and the existing `iom_tests`, backend smoke/conformance, and coexistence targets.

## Requirements

1. **Build-graph completeness.** Replace `src/iom.cpp` in `IOM_SOURCES` with all seven core destinations. Add the planned CPU `tensor.cpp` and `queue.cpp`; CUDA/ROCm `device_tensor.cpp`, private device headers, both tiled-copy `.inl` entries, and GPU queue `.inl` entries; SYCL `device_tensor.cpp`, `queue.cpp`, `queue_binary.cpp`, tiled-copy entries, and private headers; and TTNN `device_types.cpp`, `queue.cpp`, `queue_copy.cpp`, `binary.cpp`, `testing.cpp`, and private headers. Retain every source still owning code. Add `staged_worker.hpp` to `IOM_HEADERS`; private headers need not be public.
2. **No test/public-surface edits.** The final diff must contain no test target source-list change and no public factory header change. Preserve all public headers, signatures, capabilities, error categories, validation order, ownership/lifetimes, allocation behavior, asynchronous in-order queues, repeatable waits/failures, registry/quarantine semantics, workspace/staging lease completion proof, context behavior, and numerics.
3. **Exact-once and ODR proof.** For each moved non-template definition, inspect symbol definitions and link output (or equivalent object/archive evidence) to establish one definition and one owning destination. For included templates, establish intentional per-backend instantiation and no duplicate definition. Confirm no stale object from `src/iom.cpp`, no duplicate registration/factory, no duplicate test seam, and no compatibility translation unit.
4. **Umbrella and visibility proof.** Existing consumers must compile through `iom.hpp` and `outstanding_work_registry.hpp` alone where they previously did. The extracted staged-worker and registry component headers remain transitively available as specified. Private headers/fragments must not become public API; no aliases or shims may hide an integration failure.
5. **Template-fragment proof.** CUDA `src/cuda/copy.cu` and ROCm `src/rocm/copy.hip` must compile/link with `standard_tiled_copy.inl`, `standard_tiled_copy_metadata.inl`, and `gpu_queue.hpp` under their backend macro expansions. SYCL `src/sycl/copy.cpp` must compile/link its tiled-copy inclusion with the same private visibility and ODR guarantees. `gpu_queue_lifecycle.inl` and `gpu_queue_operations.inl` are included implementation sections, not independent modules.
6. **Local CPU gate.** In a clean local CPU build, run exactly:

   ```sh
   cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF
   cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests
   ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'
   ```

   The configure, all five targets, and all four selected tests must succeed.
7. **Accelerator gates through remote-development.** Use the `remote-development` workflow for configured builds on the selected remote Linux host; do not substitute local accelerator execution. For each enabled backend, build its library plus smoke, conformance, and coexistence targets exactly as follows, then run the exact matching expression:

   ```sh
   cmake --build <cuda-build> --target iom_cuda iom_cuda_smoke_tests iom_cuda_conformance_tests iom_backend_coexistence_tests
   ctest --test-dir <cuda-build> --output-on-failure -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$'

   cmake --build <rocm-build> --target iom_rocm iom_rocm_smoke_tests iom_rocm_conformance_tests iom_backend_coexistence_tests
   ctest --test-dir <rocm-build> --output-on-failure -R '^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$'

   cmake --build <sycl-build> --target iom_sycl iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests
   ctest --test-dir <sycl-build> --output-on-failure -R '^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$'

   cmake --build <ttnn-build> --target iom_ttnn iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests
   ctest --test-dir <ttnn-build> --output-on-failure -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$'
   ```

   Each accelerator configure must enable only the intended backend as appropriate for the host and must preserve coexistence linkage for all enabled backends. Smoke, conformance, and coexistence must execute on matching hardware. Missing CUDA, ROCm, SYCL, or TTNN hardware/runtime is a failure, never a skip or successful configuration substitute.
8. **Deterministic line-cap proof.** After normal formatting, scan the tracked file set obtained from `git ls-files -- src include` for exactly the suffixes `.cpp`, `.cu`, `.hip`, `.hpp`, and `.inl`, count physical lines (including blank and comment lines), and fail if any count exceeds 499. The scan must cover all twelve original oversized files and every touched/new production file, report path and count for failures, run once, and be removed if materialized as a temporary script. It must not become a permanent test, CMake target, or source list.
9. **Repair boundary.** If integration proof finds a split-caused defect, fix only the relevant CMake entry, include, private visibility, or exact-once/link defect. Do not redesign factoring, change runtime behavior, add a backend switch/global registry/cross-backend abstraction, alter synchronization, add capability, change test sources/public factory headers, add tests/docs, or perform unrelated cleanup.

## Non-goals

- Implementing or redesigning any factoring slice; this task writes no production implementation beyond a narrowly necessary integration repair.
- Changing public headers, signatures, capabilities, error behavior/order, allocation policy, ownership/lifetimes, queue ordering, wait/failure repeatability, registry/quarantine behavior, workspace/staging lease semantics, context behavior, numerics, or private visibility contracts.
- Adding compatibility aliases/shims, backend switches, global registries, cross-backend abstractions, synchronization redesign, extra capability, new tests, changed test source lists, changed public factory headers, or a permanent line-count test.
- Treating unavailable accelerator hardware or runtime as a skip, pass, or universal evidence; running accelerator gates locally instead of through `remote-development`.
- Adding documentation, examples, generated files, or unrelated cleanup.

## Acceptance criteria

1. All twelve original files and every touched or new production source, header, or `.inl` are at most 499 physical lines after formatting.
2. Every moved definition exists exactly once; `src/iom.cpp` is obsolete and absent, with no compatibility translation unit.
3. Public headers, signatures, capabilities, errors, validation order, ownership, lifetimes, asynchronous queue behavior, repeatable waits, and numerics are unchanged.
4. CMake includes every specified destination and excludes no retained source; all configured builds compile and link, including umbrella and CUDA/ROCm/SYCL tiled-fragment/private-visibility proof.
5. Existing tests pass for CPU and all configured backends, including smoke, conformance, and coexistence coverage; hardware failures are failures, not skips.
6. No new backend switches, global registry, allocation behavior, cross-backend abstraction, or permanent line-count test is introduced; no test source list or public factory header changed.

## Verification

All commands in this section are proposed gates for the future implementer and are **not run by this task**. Run accelerator commands only through `remote-development`, and record unavailable required hardware as a failure. Before and after any narrowly scoped repair, inspect only the integration diff and rerun the affected build/link gate; after all slices land, run the complete local and remote gates above, the exact-once/umbrella/template consumer proof, and the one-time scanner. Remove any temporary scanner before completion.

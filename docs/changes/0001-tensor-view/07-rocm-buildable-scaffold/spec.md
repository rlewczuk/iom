# Add the buildable ROCm backend scaffold

**Order:** 07
**Priority:** P0 — backend dependency discovery and context ownership must be proven before ROCm storage work starts.
**Blocked by:** `06-shared-backend-conformance`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

With `ROCM_ENABLED=ON`, ROCm is an independent optional library with a public factory that validates an ordinal, owns the corresponding runtime context, reports its backend identity, and passes a hardware-backed construction/destruction smoke test.

## Scope

- Scaffold ROCm first because the current workstation has an AMD GPU; this task is the dependency-installation and build checkpoint before implementing ROCm tensors.
- Add only dependency discovery, target layering, public factory, context lifetime, and smoke coverage.

## Implementation references

- **Create (planned):** `include/iom/rocm/device.hpp` — declare `make_rocm_device(std::uint32_t, Allocator&)` without exposing HIP types.
- **Create (planned):** `src/rocm/device.cpp` — implement ordinal validation, owned HIP context/device state, identity accessors, and deterministic teardown.
- **Create (planned):** `test/rocm/test_rocm_smoke.cpp` — hardware-backed valid/invalid ordinal and context lifetime cases.
- **Modify:** `CMakeLists.txt` — make `ROCM_ENABLED` an independent branch and create optional library target `iom_rocm` linked to `libiom`.
- **Modify:** `test/CMakeLists.txt` — add `iom_rocm_smoke_tests` only when ROCm is enabled.
- **Read:** existing `CMakeLists.txt` — retain the `ROCM_PATH` cache convention and ROCm 7.2 minimum check while removing CUDA/ROCm mutual exclusion.

## Requirements

- The core `libiom` target and common headers remain free of HIP headers, compiler settings, and ROCm link dependencies when `ROCM_ENABLED=OFF`.
- Enabling ROCm discovers the configured SDK, enables only the required HIP language/runtime for `iom_rocm`, and fails configuration clearly when dependencies are unavailable; it must not silently disable the backend.
- `make_rocm_device(ordinal, allocator)` checks the backend-local ordinal before creating tensor or queue resources and throws for an unavailable or invalid ordinal.
- A successful factory call owns the runtime context for that ordinal, reports `BackendKind::ROCM` and the exact ordinal through both `Device` and later views, and retains only a non-owning reference to the caller allocator.
- The context is destroyed exactly once after the device's dependent tensors and queues have already been destroyed. Borrowed contexts, global active-device state, singleton factories, and a `BackendKind` registry are prohibited.
- During this explicit scaffold milestone, `create_tensor` and `create_ops` may report the not-yet-delivered ROCm capability with `std::runtime_error`, but must allocate no fake tensor/queue resources and consume no queue sequence. Task `08-rocm-storage-copy` removes those temporary capability failures for required storage and copy behavior.
- The smoke test uses real ROCm hardware/runtime, does not skip when the option is enabled, covers at least one valid device and the first invalid ordinal, and proves the CPU target still builds independently.

## Non-goals

- ROCm tensor allocation, host transfer, kernels, asynchronous copy, or conformance cases.
- Numerical compute operations, borrowed HIP contexts, queue tuning, or runtime backend selection.
- Scaffolding CUDA, SYCL, or TTNN in parallel with this milestone.

## Acceptance criteria

- [ ] Core-plus-CPU configures with ROCm disabled and has no ROCm dependency.
- [ ] ROCm-only optional configuration builds `iom_rocm` and its smoke executable with ROCm-specific compiler/link settings isolated to those targets.
- [ ] Valid construction reports ROCm and the requested ordinal; invalid ordinal rejection happens before tensor/queue creation.
- [ ] Instrumented runtime setup/teardown proves one owned context and one matching destruction.
- [ ] The enabled smoke test executes on hardware and contains no skip path.

## Verification

- `cmake -S . -B build/rocm-smoke -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm`
- `cmake --build build/rocm-smoke --target iom_rocm_smoke_tests`
- `ctest --test-dir build/rocm-smoke --output-on-failure -R '^iom_rocm_smoke_tests$'`

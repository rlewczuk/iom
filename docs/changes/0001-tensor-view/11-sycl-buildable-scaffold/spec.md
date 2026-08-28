# Add the buildable SYCL backend scaffold

**Order:** 11
**Priority:** P0 — SYCL dependency discovery and context ownership must pass before SYCL storage work.
**Blocked by:** `10-cuda-storage-copy`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

With `SYCL_ENABLED=ON`, SYCL is an independent optional library with a public factory that validates a backend-local ordinal, owns the selected SYCL context, reports backend identity, and passes a hardware-backed construction/destruction smoke test.

## Scope

- Begin SYCL only after CUDA storage/copy conformance is complete.
- Add the new option, dependency discovery, isolated target settings, public factory, owned context lifecycle, and smoke coverage.

## Implementation references

- **Create (planned):** `include/iom/sycl/device.hpp` — declare `make_sycl_device(std::uint32_t, Allocator&)` without SYCL header types.
- **Create (planned):** `src/sycl/device.cpp` — enumerate supported devices in a deterministic backend-local order, validate ordinals, own the SYCL device/context, and report identity.
- **Create (planned):** `test/sycl/test_sycl_smoke.cpp` — real-device ordinal validation and context lifetime cases.
- **Modify:** `CMakeLists.txt` — add independent `SYCL_ENABLED` and optional target `iom_sycl` with target-local compiler/link settings.
- **Modify:** `test/CMakeLists.txt` — add `iom_sycl_smoke_tests` only when SYCL is enabled.
- **Read:** `include/iom/cuda/device.hpp` and `include/iom/rocm/device.hpp` — reuse backend-neutral public factory/header layering.

## Requirements

- With SYCL disabled, common headers, `libiom`, CUDA, and ROCm contain no SYCL headers, flags, runtime links, or public definitions.
- Enabling SYCL discovers the required compiler/runtime package and fails configuration clearly if unavailable. Do not silently disable or silently select a host fallback when the backend was requested.
- Define and test a deterministic ordinal enumeration for eligible accelerator devices; CPU fallback devices are not accepted as this backend's accelerator ordinal.
- `make_sycl_device(ordinal, allocator)` rejects an unavailable/invalid ordinal before tensor or queue resources, then owns the selected device context and reports `BackendKind::SYCL` plus the exact ordinal.
- The caller allocator is non-owning from the device's perspective. Context teardown happens exactly once after all dependent tensors and queues.
- No borrowed context, global selector mutation, singleton, registry, or process-global active backend may be introduced.
- During this explicit scaffold milestone, `create_tensor` and `create_ops` may throw `std::runtime_error` for not-yet-delivered SYCL capabilities without creating fake resources or consuming a sequence. Task `12-sycl-storage-copy` removes those temporary failures.
- The smoke test runs on actual eligible SYCL hardware, checks the first invalid ordinal, and never skips when `SYCL_ENABLED=ON`.
- All previously implemented backend options remain independent and may still configure individually.

## Non-goals

- USM tensor allocation, host transfer, copy kernels, or conformance.
- Numerical compute, borrowed contexts, queue tuning, or host-device fallback behavior.
- TTNN scaffolding before SYCL conformance passes.

## Acceptance criteria

- [ ] Core-plus-CPU and existing backend targets build with SYCL disabled and no SYCL dependency.
- [ ] SYCL-only configuration builds `iom_sycl` and `iom_sycl_smoke_tests` with settings isolated to those targets.
- [ ] Valid construction reports SYCL and the configured ordinal; invalid ordinal rejection precedes tensor/queue resources.
- [ ] Runtime instrumentation proves one owned context and one matching teardown.
- [ ] Enabled smoke coverage uses real accelerator hardware with no skip or CPU fallback path.

## Verification

- `cmake -S . -B build/sycl-smoke -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF`
- `cmake --build build/sycl-smoke --target iom_sycl_smoke_tests`
- `ctest --test-dir build/sycl-smoke --output-on-failure -R '^iom_sycl_smoke_tests$'`

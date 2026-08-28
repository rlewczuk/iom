# Add the buildable CUDA backend scaffold

**Order:** 09
**Priority:** P0 — CUDA dependency discovery and owned-context construction must pass before CUDA storage work.
**Blocked by:** `08-rocm-storage-copy`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

With `CUDA_ENABLED=ON`, CUDA is an independent optional library with a public factory that validates an ordinal, owns the corresponding CUDA context, reports backend identity, and passes a hardware-backed construction/destruction smoke test.

## Scope

- Begin CUDA only after the ROCm implementation and hardware conformance milestone is complete.
- Add CUDA dependency discovery, isolated target/compiler settings, public factory, owned context lifecycle, and construction smoke coverage.

## Implementation references

- **Create (planned):** `include/iom/cuda/device.hpp` — declare `make_cuda_device(std::uint32_t, Allocator&)` without exposing CUDA types.
- **Create (planned):** `src/cuda/device.cpp` — implement ordinal validation, owned CUDA context/device state, identity accessors, and teardown.
- **Create (planned):** `test/cuda/test_cuda_smoke.cpp` — real-device valid/invalid ordinal and context lifetime cases.
- **Modify:** `CMakeLists.txt` — make `CUDA_ENABLED` an independent branch and create optional target `iom_cuda` linked to `libiom`, never folding CUDA sources into core.
- **Modify:** `test/CMakeLists.txt` — add `iom_cuda_smoke_tests` only when CUDA is enabled.
- **Read:** existing `CMakeLists.txt` — retain the `CUDA_PATH` cache convention while removing the former shared kernel-language and CUDA/ROCm `elseif` model.
- **Read:** `include/iom/rocm/device.hpp` — reuse public factory layering and header hygiene, not ROCm runtime types.

## Requirements

- Core and ROCm targets remain buildable without CUDA headers, compiler settings, libraries, or definitions when `CUDA_ENABLED=OFF`.
- Enabling CUDA discovers the configured toolkit, enables CUDA only for `iom_cuda`, and fails configuration clearly rather than silently disabling a requested backend.
- `make_cuda_device(ordinal, allocator)` rejects an unavailable or invalid ordinal before creating tensor or queue resources.
- A successful factory owns a CUDA runtime context for the ordinal, reports `BackendKind::CUDA` and the exact backend-local ordinal, and stores only a non-owning allocator reference.
- Context teardown occurs exactly once after all dependent tensors and queues. Do not use borrowed contexts, a process-global active device, a singleton, a registry, or runtime dispatch by `BackendKind`.
- During this explicit scaffold milestone, `create_tensor` and `create_ops` may throw `std::runtime_error` for the not-yet-delivered CUDA capability, but must create no fake storage/queue and consume no sequence. Task `10-cuda-storage-copy` removes those temporary failures for required behavior.
- The smoke test uses actual CUDA hardware/runtime, rejects the first unavailable ordinal, and never skips when the backend option is enabled.
- Enabling CUDA must not disable or mutate the already independent ROCm option; combined linkage is verified only after every backend is complete.

## Non-goals

- CUDA allocation, host transfer, copy kernels, or conformance.
- Numerical compute operations, borrowed contexts, queue tuning, or backend registries.
- SYCL or TTNN scaffolding before CUDA conformance is complete.

## Acceptance criteria

- [ ] Core-plus-CPU configures with CUDA disabled and has no CUDA dependency.
- [ ] CUDA-only optional configuration builds `iom_cuda` and `iom_cuda_smoke_tests` with target-local CUDA settings.
- [ ] Valid construction reports CUDA and the requested ordinal; invalid ordinal rejection precedes tensor/queue resources.
- [ ] Runtime instrumentation proves one owned context and one matching teardown.
- [ ] The enabled hardware smoke test has no skip path and existing ROCm configuration remains independently selectable.

## Verification

- `cmake -S . -B build/cuda-smoke -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda`
- `cmake --build build/cuda-smoke --target iom_cuda_smoke_tests`
- `ctest --test-dir build/cuda-smoke --output-on-failure -R '^iom_cuda_smoke_tests$'`

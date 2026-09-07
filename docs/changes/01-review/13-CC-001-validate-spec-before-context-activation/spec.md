# CUDA/ROCm create_tensor activates the runtime context before TensorSpec validation, inverting the validation-first creation contract

**Order:** 13
**Priority:** P2
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** CC-001
**Review area:** Contract & correctness
**Review severity:** low
**Review verification:** verified, confidence 82
**Review scope:** whole-codebase
**Backend scope:** cuda, rocm
**Location:** `src/cuda/device.cpp:212-216 (CudaDevice::create_tensor); src/rocm/device.cpp:190-194 (RocmDevice::create_tensor)`

## Outcome

A rejected `TensorSpec` makes `create_tensor` throw the deterministic spec-validation error with zero `cuCtxSetCurrent`/`hipSetDevice` driver calls and zero allocator allocations. A valid spec still activates the runtime exactly once, at allocation time, through the tensor constructor's `pre_allocate` callback — the canonical activation owner. On ROCm the calling thread's current device is no longer mutated on the rejection path.

## Current problem

Both backend entry points activate the runtime context before the base `Tensor` constructor can run spec validation, inverting the validation-first contract that CPU and TTNN already follow.

- `CudaDevice::create_tensor` (`src/cuda/device.cpp:212-216`) calls `activate()` (which issues `cuCtxSetCurrent`, `src/cuda/device.cpp:112-116`) before `std::make_unique<CudaTensor>` runs the base `Tensor` ctor, whose `validated_dense_plane_strides` (`src/iom.cpp:242-247`, called from `Tensor::Tensor` at `src/iom.cpp:465-467`) begins with `TensorSpec::validate()` (`src/iom.cpp:75-110`).
- `RocmDevice::create_tensor` (`src/rocm/device.cpp:190-194`) does the same with `hipSetDevice` (`src/rocm/device.cpp:97-101`).
- The mandatory current ordering is therefore: (1) `activate()`, (2) `spec.validate()` inside the base ctor, (3) `allocate_aligned_storage`, whose `pre_allocate` lambda (`src/cuda/device.cpp:147`, `src/rocm/device.cpp:126`) activates the context again — `pre_allocate()` runs before `allocator.alloc(nbytes)` at `include/iom/detail/aligned_storage.hpp:21-27`.

Consequences: on an unhealthy/lost context a driver error from the entry-point `activate()` masks the deterministic invalid-spec error the contract dictates; every `create_tensor` (success and reject paths) issues a redundant context-set driver call because the allocation callback already activates; on ROCm a rejected creation still mutates the calling thread's current device before throwing. Healthy-path allocation, alignment, and validation behavior are unchanged; validation does still run and no allocation happens for rejected specs.

## Scope

- Delete the standalone `activate()` call from `CudaDevice::create_tensor` (`src/cuda/device.cpp:214`) and from `RocmDevice::create_tensor` (`src/rocm/device.cpp:192`); allocation-time activation remains owned by the ctors' `pre_allocate` callbacks.
- Add behavioral coverage proving a rejected spec causes zero context-set driver calls and preserves the spec-error type, and that ROCm's thread current device is untouched after rejection.
- Leave unchanged: `CudaTensor`/`RocmTensor` ctors and their `pre_allocate` lambdas, base `Tensor::Tensor` + `validated_dense_plane_strides`, `TensorSpec::validate`, allocator alignment semantics, `create_ops`/factory activation, and the TTNN/CPU validation-first orderings.

## Implementation references

- **Modify:** `src/cuda/device.cpp` — `CudaDevice::create_tensor` (`:212-216`); delete the `activate()` at `:214`. The success path is already activated by the `pre_allocate` lambda at `:147` right before allocation; nothing between base-ctor validation and allocation touches the device.
- **Modify:** `src/rocm/device.cpp` — `RocmDevice::create_tensor` (`:190-194`); delete the `activate()` at `:192`. The success path is already activated by the `pre_allocate` lambda at `:126`.
- **Read:** `include/iom/detail/aligned_storage.hpp:21-27` — `allocate_aligned_storage` invokes `pre_allocate()` before `allocator.alloc(nbytes)`; this is the canonical activation owner.
- **Read:** `src/ttnn/device.cpp:574-586` (`TtnnDevice::create_tensor`) and `src/cpu/device.cpp` (`CpuDevice::create_tensor`) — validation-first counterpart orderings to preserve.
- **Tests:** `test/cuda/test_cuda_smoke.cpp` — driver-call seam `DriverCallProbe::ctx_set_current_count` (`:60-88`) with `pass_through_ctx_set_current` counting wrapper (`:83-88`); misaligned-allocation case at `:245-255` (valid spec, expects `runtime_error`) must stay green. `test/rocm/test_rocm_smoke.cpp:49-83` — checks the factory's device selection via `hipGetDevice`, not `create_tensor` ordering, so it does not pin the current sequence.

## Requirements

- A `create_tensor` call with an invalid spec (rank &lt; 2, zero dimension, or grouped quantization) MUST throw the same spec-validation error the base ctor produces today (`std::invalid_argument` for rank/dimension violations, `std::runtime_error` for grouped quantization) while making zero `cuCtxSetCurrent` (CUDA) / `hipSetDevice` (ROCm) calls and zero allocator allocations.
- A valid-spec `create_tensor` MUST activate the runtime exactly once, at allocation time, via the ctor's `pre_allocate` callback; no device call may occur between base-ctor validation and allocation.
- On ROCm, after a rejected `create_tensor`, the calling thread's current device MUST be unchanged.
- Preserve error types, error messages, and all healthy-path allocation/alignment behavior; existing CUDA misalignment and ROCm factory-selection tests must pass unchanged.

## Non-goals

- Do not duplicate `spec.validate()` into the CUDA/ROCm `create_tensor` bodies (the base ctor owns validation).
- Do not touch `make_cuda_device`/`make_rocm_device` factory activation or `create_ops`.
- Do not change the TTNN/CPU validation-first orderings or allocator alignment semantics.

## Acceptance criteria

- [ ] `create_tensor` with a rejected spec (e.g., grouped quantization or a zero dimension) throws the spec-validation error while the CUDA driver-call probe records `ctx_set_current_count == 0` and no allocator allocation occurs; `create_tensor` with a valid spec activates exactly once at allocation.
- [ ] On ROCm, `hipGetDevice` returns the same current device after a rejected `create_tensor`.
- [ ] Existing CUDA misaligned-allocation case and ROCm factory-selection case pass unchanged; repeated valid creates still activate once per allocation and allocate 32-byte-aligned storage.

## Verification

- `actual validation: none (read-only review)`; proposed gates below.
- Remote CUDA and ROCm Linux hosts (accelerators required; run per `remote-development`): configure each backend with `BUILD_TESTING=ON`, then `cmake --build <build> --target iom_cuda_smoke_tests iom_rocm_smoke_tests` and `ctest --test-dir <build> -R '^iom_(cuda|rocm)_smoke_tests$' --output-on-failure`.
- Add a CUDA smoke case using the existing `DriverCallProbe`/`pass_through_ctx_set_current` seam (`test/cuda/test_cuda_smoke.cpp:60-88`): a rejected spec (grouped quantization or zero dimension) throws the spec error with `ctx_set_current_count == 0`. Add a ROCm smoke assertion that `hipGetDevice` is unchanged after a rejected `create_tensor`. The existing CUDA misalignment test (valid spec, expects `runtime_error`) and ROCm factory-selection test (`test/rocm/test_rocm_smoke.cpp:49-83`) must remain green.
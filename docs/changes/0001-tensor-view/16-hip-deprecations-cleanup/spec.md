# Remove deprecated HIP context APIs from the ROCm backend

**Order:** 16
**Priority:** P2 — cleanup that keeps ROCm 7.2+ builds warning-free.
**Blocked by:** `15-backend-coexistence`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

The ROCm backend builds against current ROCm headers with no `-Wdeprecated-declarations` warnings. Every `hipCtx*` driver context call is replaced by the HIP-preferred model: device ordinal + per-thread `hipSetDevice` for device-scoped setup calls and explicit streams for ongoing work. Observable device, tensor, transfer, and queue behavior is unchanged; the public factory signature is unchanged.

## Current behavior (verified)

HIP has deprecated the entire `hipCtx*` and `hipDevicePrimaryCtx*` family since ROCm 1.9.0; ROCm 7.2 headers emit the quoted `-Wdeprecated-declarations` warning. Complete inventory of the deprecated context usage in this repository:

- `src/rocm/device.cpp`
  - `make_rocm_device`: `hipCtxCreate(&context, 0, device)` after ordinal validation; `hipCtxDestroy` on the exception path. `hipCtxCreate` also makes the new context current on the calling thread.
  - `RocmDevice`: stores `hipCtx_t context_`; constructor takes it; destructor calls `hipCtxDestroy`; `activate()` calls `hipCtxSetCurrent(context_)`; `context()` getter exposes it. `activate()` is called from `create_tensor`, `create_ops`, the `RocmTensor` constructor/destructor, and both `region_*_host` transfer wrappers.
- `src/rocm/copy.hpp`
  - `region_from_host`, `region_to_host`, and `make_queue` declarations take `hipCtx_t context`.
- `src/rocm/copy.hip`
  - `ContextGuard` wraps `hipCtxSetCurrent`; it is used in `synchronous_transfer`, the `RocmQueue` constructor (stream creation), the `copy` submission lambda, the worker thread `run()`, and the `RocmQueue` destructor (event/stream teardown).
  - `synchronous_transfer` takes `hipCtx_t context`; `RocmQueue` stores `hipCtx_t context_`; `region_from_host`, `region_to_host`, and `make_queue` definitions take `hipCtx_t`.
- `test/rocm/test_rocm_smoke.cpp`
  - Lines 56–58 assert a non-null current context via `hipCtxGetCurrent` immediately after factory creation (also deprecated).
- `include/iom/rocm/device.hpp` — clean; declares only `make_rocm_device` and references no HIP context types.

No other HIP API used by the backend or its tests (`hipMalloc`, `hipFree`, stream/event/memcpy/memset calls, `hipLaunchKernelGGL`, `hipGetDeviceCount`, error-string helpers) is deprecated.

## Scope

- Replace context creation, ownership, and activation with ordinal-based per-thread device selection across `src/rocm` and `test/rocm`.
- Keep observable behavior and all existing tests' intent intact.

## Implementation references

- **Modify:** `src/rocm/device.cpp` — drop `hipCtx_t` from `RocmDevice` (constructor, `context_` member, `context()` getter); `activate()` becomes `hipSetDevice(static_cast<int>(ordinal_))`; remove the destructor's `hipCtxDestroy` and the factory's `hipCtxCreate`/`hipCtxDestroy`; the factory calls `hipSetDevice(device_ordinal)` after the existing ordinal validation.
- **Modify:** `src/rocm/copy.hpp` and `src/rocm/copy.hip` — `ContextGuard` becomes a device guard calling `hipSetDevice(ordinal)`; `region_from_host`, `region_to_host`, `make_queue`, `synchronous_transfer`, and `RocmQueue` carry the device ordinal (`int`, matching the `hipSetDevice` signature) instead of `hipCtx_t`.
- **Modify:** `test/rocm/test_rocm_smoke.cpp` — replace the `hipCtxGetCurrent` assertion with `hipGetDevice` returning the factory ordinal; update the case title ("owns its context") and the stale comment about deterministic context teardown.
- **Read:** `src/cuda/device.cpp` — left unchanged; CUDA's driver primary-context API (`cuDevicePrimaryCtxRetain`/`cuCtxSetCurrent`) is not deprecated.

## Requirements

- Factory validation is unchanged: `hipGetDeviceCount` bounds-check rejects invalid ordinals with `std::invalid_argument` before any device call. On success the factory calls `hipSetDevice(ordinal)` so the calling thread observes the device as current — matching the thread currency the old `hipCtxCreate` established — with failures surfacing as `std::runtime_error` via `check_hip`.
- `RocmDevice` owns no HIP context and its destructor performs no HIP calls. The runtime-managed primary context is process state and is never explicitly retained, released, or reset.
- Every thread that issues HIP calls first selects the device with `hipSetDevice(ordinal)`: tensor storage allocation/free (`activate()` in the `RocmTensor` constructor/destructor), synchronous host transfers (device guard in `synchronous_transfer`), queue stream creation and destruction, the `copy` submission path, and the queue worker thread. Guard sites mirror the existing `ContextGuard` sites one-for-one.
- `RocmQueue` keeps its `hipStream_t`, worker thread, staging, and all submission/completion/wait semantics; only the context member becomes a device ordinal.
- The `rocm_detail` transfer/queue entry points take the device ordinal instead of `hipCtx_t`; parameters and behavior are otherwise unchanged.
- No `hipCtx*` or `hipDevicePrimaryCtx*` symbol remains in `src/rocm`, `include/iom/rocm`, or `test/rocm`.
- `include/iom/rocm/device.hpp` and the public `Device`/`Tensor`/`DeviceOps` contracts are unchanged.
- Ownership invariants are unchanged: device and allocator outlive tensors and queues; the queue still destroys its own stream and events; tensor storage is still freed exactly once by the caller allocator.
- Warning-suppression flags are not an acceptable fix; the deprecated calls are removed.

## Non-goals

- CUDA backend changes; its driver-API primary context usage is not deprecated.
- Changes to copy/transfer kernels, plane-pair logic, queue threading, or validation rules.
- `hipDeviceReset` or other process-wide runtime teardown.
- Any new public API, context injection, or queue tuning options.

## Acceptance criteria

- [ ] A ROCm-enabled build compiles every iom ROCm library and test target with zero `-Wdeprecated-declarations` warnings.
- [ ] No `hipCtx*`/`hipDevicePrimaryCtx*` reference remains in `src/rocm`, `include/iom/rocm`, or `test/rocm`.
- [ ] After `make_rocm_device(0, …)`, the smoke test observes `hipGetDevice` == 0, and device destruction plus recreation remains deterministic.
- [ ] `iom_rocm_smoke_tests` and `iom_rocm_conformance_tests` pass on ROCm hardware with no skips.
- [ ] The core-plus-CPU build with `ROCM_ENABLED=OFF` is unaffected, and the CUDA backend is byte-for-byte untouched.

## Verification

- Accelerator build, test, and execution follow the `remote-development` skill on the ROCm host.
- `cmake -S . -B build/rocm -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm`
- `cmake --build build/rocm --target iom_rocm_smoke_tests iom_rocm_conformance_tests` — build log shows no deprecation warnings.
- `ctest --test-dir build/rocm --output-on-failure -R '^iom_rocm_(smoke|conformance)_tests$'`

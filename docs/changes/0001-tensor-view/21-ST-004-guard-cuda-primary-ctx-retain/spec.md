# Guard the CUDA primary-context retain against factory activation failure

**Order:** 21
**Priority:** P1 — bounded failure-path resource-leak remediation in the CUDA factory that does not broadly gate other work.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-004`
**Review severity:** medium
**Review verification:** verified, confidence 97

## Outcome

Every failure exit of `make_cuda_device` after a successful `cuDevicePrimaryCtxRetain` — activation failure and `CudaDevice` allocation failure — releases the retained primary context exactly once before the original exception reaches the caller, proven on CUDA hardware by fault-injecting regression tests that count matching `cuDevicePrimaryCtxRelease` calls through a new injectable CUDA driver seam. Successful construction still performs exactly one release in total, at `CudaDevice` destruction.

## Current failure

Invariant: every successfully acquired backend resource must be released on every later construction failure.

In `src/cuda/device.cpp` (`make_cuda_device`, lines 165-200), the factory validates the ordinal, then at lines 188-191 calls `cuDevicePrimaryCtxRetain` — acquiring a reference on the process-wide primary context — and at line 192 calls `check_cuda("cuCtxSetCurrent", cuCtxSetCurrent(context))` *before* entering the `try` block. The only failure-path `cuDevicePrimaryCtxRelease` is in the `catch (...)` at lines 196-199, which covers only the `std::make_unique<CudaDevice>` call. If `cuCtxSetCurrent` fails, `check_cuda` throws `std::runtime_error` immediately and the retain is never balanced. A failed factory call therefore leaks a primary-context retain; repeated failed constructions accumulate unmatched retains and make later process teardown or CUDA configuration nondeterministic. The `CudaDevice` allocation-failure path (lines 193-199) is already balanced but relies on the manual `try`/`catch`; the remediation must preserve it under the new guard.

## Scope

- `make_cuda_device` in `src/cuda/device.cpp`: replace the manual retain/try/catch/release structure with a local RAII guard over the retained `(CUdevice, CUcontext)` that begins immediately after a successful `cuDevicePrimaryCtxRetain` and is dismissed only once `CudaDevice` has taken ownership.
- Keep activation (`cuCtxSetCurrent`) and `CudaDevice` construction inside the guarded scope; error types, messages, and ordinal-validation behavior are unchanged.
- Introduce a minimal injectable seam for the three primary-context lifecycle driver calls so tests can inject an activation failure and observe releases; default seam behavior is identical to the current direct calls.
- Regression tests live in the existing CUDA smoke suite and run on real CUDA hardware (fail, never skip).

## Implementation references

- **Modify:** `src/cuda/device.cpp` — `make_cuda_device` (lines 165-200): wrap the retain at lines 188-191 in the guard; move/keep the activation at line 192 and `std::make_unique<CudaDevice>` at lines 194-195 inside the guarded scope; delete the `catch (...)` release at lines 196-199 (the guard replaces it). Route the factory's retain/set-current/release, `CudaDevice::activate` (lines 77-79), and `~CudaDevice`'s release (lines 54-59) through the seam below so release counting observes the whole lifecycle.
- **Create (planned):** `src/cuda/driver.hpp` — private header declaring, in namespace `iom::cuda_detail` (mirroring how `src/cuda/copy.hpp` declares `iom::cuda_detail::make_queue` and is included from `device.cpp` as `#include "copy.hpp"`), a `DriverCalls` struct of plain function pointers `CUresult (*primary_ctx_retain)(CUcontext*, CUdevice)`, `CUresult (*ctx_set_current)(CUcontext)`, and `CUresult (*primary_ctx_release)(CUdevice)`, each default-initialized to the real driver entry (`&cuDevicePrimaryCtxRetain`, `&cuCtxSetCurrent`, `&cuDevicePrimaryCtxRelease`), plus `extern DriverCalls driver_calls;`. Define the object once in `src/cuda/device.cpp`.
- **Modify:** `src/cuda/device.cpp` — add a file-local RAII guard class in the existing anonymous namespace (suggested name `PrimaryCtxGuard`): constructed with the retained `CUdevice` immediately after the successful retain, non-copyable per the repository owner convention, destructor calls `(void)driver_calls.primary_ctx_release(device_)` only while armed, and a `dismiss() noexcept` member that disarms.
- **Modify:** `CMakeLists.txt` — `iom_cuda` target (lines 94-126): add `src/cuda/driver.hpp` to the source list alongside `src/cuda/copy.hpp`.
- **Modify:** `test/CMakeLists.txt` — `iom_cuda_smoke_tests` block (lines 88-104): add `target_include_directories(iom_cuda_smoke_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda)` so the test can `#include "driver.hpp"`.
- **Modify:** `test/cuda/test_cuda_smoke.cpp` — append the three regression cases below next to the existing factory cases; the file already includes `<cuda.h>`, `<new>`, and `<stdexcept>` and links `CUDA::cuda_driver`.
- **Read:** `include/iom/cuda/device.hpp` — the public factory contract stays byte-identical; no CUDA types may leak into public headers, which is why the seam is a private header.
- **Read:** `src/rocm/device.cpp` (`make_rocm_device`, lines 152-171) — the ROCm factory uses `hipSetDevice` with no retained primary context; it has no analogous leak and must not be touched.
- **Tests:** `test/cuda/test_cuda_smoke.cpp` — existing cases "CUDA factory reports a live hardware device and owns its context" and "CUDA factory rejects the first unavailable ordinal" define the hardware-guard and allocator-fixture conventions (`REQUIRE(cuInit(0) == CUDA_SUCCESS)`, `REQUIRE(device_count > 0)`).

## Requirements

- Immediately after `check_cuda("cuDevicePrimaryCtxRetain", ...)` succeeds in `make_cuda_device`, construct the armed `PrimaryCtxGuard` for the retained device. Every subsequent statement in the factory — the activation `check_cuda("cuCtxSetCurrent", ...)` and the `std::make_unique<CudaDevice>(device_ordinal, device, context, allocator)` — executes inside the guard's scope.
- Guard destruction while armed performs exactly one `driver_calls.primary_ctx_release(device)` and nothing else; it must not swallow or replace the in-flight exception. Disarm via `dismiss()` only after `std::make_unique<CudaDevice>` has returned, storing the result in a local `unique_ptr` that is returned after dismissal, so `~CudaDevice` remains the sole release on the success path and no double release is possible.
- All failure exits after a successful retain release exactly once before the exception is observable to the caller, preserving the original exception: activation failure still throws `std::runtime_error` from `check_cuda`/`cuda_error` with a `what()` beginning `"cuCtxSetCurrent failed with"`, and `CudaDevice` allocation failure still propagates `std::bad_alloc`. Remove the now-redundant `try`/`catch` release in the factory; no second release path may remain.
- The seam covers exactly the three primary-context lifecycle calls (`cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxRelease`) at their `src/cuda/device.cpp` call sites (factory, guard, `CudaDevice::activate`, `~CudaDevice`). With the seam untouched, every entry equals the direct driver call and observable behavior, error text, and public ABI are unchanged. The seam is internal (`iom::cuda_detail` in a private header), never exported through `include/iom/cuda/device.hpp`.
- Driver calls in `src/cuda/copy.cu` (queue/transfer paths) are not routed through the seam.
- Regression tests in `test/cuda/test_cuda_smoke.cpp`, each guarded by the existing `REQUIRE(cuInit(0) == CUDA_SUCCESS)` / `REQUIRE(device_count > 0)` hardware prelude, each saving and restoring `iom::cuda_detail::driver_calls` with an RAII restore helper, and each using a counting pass-through allocator-agnostic `UnusedAllocator` fixture as today:
  1. **Activation failure:** wrap `primary_ctx_retain` to record the retained `(CUdevice, CUcontext)` and call through to the real driver; wrap `ctx_set_current` to return `CUDA_ERROR_OUT_OF_MEMORY` without calling the real function; wrap `primary_ctx_release` to record the `CUdevice`, increment a counter, and call through to the real driver (so the test itself stays balanced). Assert `make_cuda_device(0, allocator)` throws `std::runtime_error` whose `what()` begins `"cuCtxSetCurrent failed with"`, and that exactly one `cuDevicePrimaryCtxRelease` with the same recorded `CUdevice` executed by the time the throw is observed.
  2. **`CudaDevice` allocation failure after successful activation:** pass retain and set-current through to the real driver via the counting wrappers, and define a plain `void* operator new(std::size_t)` in the test TU (outside any anonymous namespace; forward other allocation forms unchanged) that throws `std::bad_alloc` exactly once when armed via a one-shot flag and otherwise forwards to `std::malloc` (returning `std::bad_alloc` on null). Arm it immediately before the factory call: the factory performs no host allocation between activation and `std::make_unique<CudaDevice>`, so the one-shot failure deterministically fails the `CudaDevice` allocation. Assert `std::bad_alloc` propagates and exactly one matching release executed; no destructor release follows because no `CudaDevice` was constructed.
  3. **Dismissal on success:** with pure pass-through wrappers, assert a successful `make_cuda_device(0, allocator)` performs zero seam releases while the returned device is alive, and exactly one (same `CUdevice`) after the `unique_ptr` is destroyed — proving no guard/destructor double release.
- The new tests fail for the reviewed bug: with the seam present but the guard absent (current structure), case 1 records zero releases and case 2 keeps the old catch-only behavior detectable only through case 3's lifecycle counting; case 1 is the direct discriminator.

## Non-goals

- The transactional submission boundary, orphaned-kernel `oid`, and queue/event/task fault injection of `ST-002` (separate task with its own wrappers); the seam here is only the primary-context lifecycle in `src/cuda/device.cpp`.
- No changes to `src/rocm`, `src/ttnn`, `src/sycl`, CPU code, or the ROCm `hipSetDevice` factory, which retains nothing.
- No general CUDA driver wrapper framework, no virtual-device or mock-CUDA abstraction, and no routing of `src/cuda/copy.cu` calls through the seam.
- No public API, error-message, or ordinal-validation changes; `include/iom/cuda/device.hpp` is untouched.

## Acceptance criteria

- [ ] Injected `cuCtxSetCurrent` failure after a successful retain propagates the unchanged `std::runtime_error` (`"cuCtxSetCurrent failed with"`) after exactly one `cuDevicePrimaryCtxRelease` with the same `CUdevice` recorded at retain.
- [ ] One-shot `CudaDevice` allocation failure after successful real activation propagates `std::bad_alloc` after exactly one matching release, with no destructor release.
- [ ] Successful construction performs zero releases until device destruction, which performs exactly one — no double release from guard plus destructor.
- [ ] With the seam untouched, the existing CUDA smoke cases and the CUDA conformance suite pass unchanged on hardware.
- [ ] No `try`/`catch` release path remains in `make_cuda_device`; the guard is the only failure-path release.

## Verification

Follow `.agents/skills/remote-development` (local workspace authoritative; unique remote task directory). The fault injection itself executes host-side, but `cuDevicePrimaryCtxRetain` must genuinely succeed, so the regression cases require real CUDA hardware and must not be replaced by CPU-only evidence.

```bash
.agents/skills/remote-development/scripts/remote-sync <cuda-host-alias> st004-ctx-guard
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st004-ctx-guard \
  'cmake -S . -B build/cuda-st004 -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build/cuda-st004 --target iom_cuda_smoke_tests iom_cuda_conformance_tests'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st004-ctx-guard \
  'ctest --test-dir build/cuda-st004 --output-on-failure -R "^iom_cuda_(smoke|conformance)_tests$"'
.agents/skills/remote-development/scripts/remote-clean <cuda-host-alias> st004-ctx-guard
```

Expected observations: all three new fault-injection cases pass with exactly one matching release in each failure case; the pre-existing smoke cases (`CUDA factory reports a live hardware device and owns its context`, `CUDA tensor rejects misaligned allocator storage exactly once`, `CUDA factory rejects the first unavailable ordinal`) and `iom_cuda_conformance_tests` pass unchanged. As a bug-discriminator check, reverting only the guard (seam retained) must make the activation-failure case report zero releases and fail.

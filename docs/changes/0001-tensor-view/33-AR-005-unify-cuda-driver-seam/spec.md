# Complete the CUDA driver-call injection seam and centralize the duplicated driver-API error helpers in `src/cuda/driver.hpp`

**Order:** 33
**Priority:** P1 — bounded test-injection architecture fix. The seam is already three-fifths implemented and is exercised by `test_cuda_smoke.cpp` (`DriverCallsRestore`, `DriverCallProbe`); completing it removes a misleading partial seam.
**Blocked by:** 30-AR-002-share-backend-queue-scaffold, 31-AR-003-expose-device-capabilities. AR-005's driver-helper and seam edits apply to the post-AR-002 / post-AR-003 file state: `src/cuda/copy.cu` after AR-002 collapses the queue scaffolding behind `iom::detail::StagedWorker<Task>` (`src/cuda/copy.cu`'s `CudaQueue` becomes a thin composition of the helper plus three static `Callbacks` members and a `PublishPolicy::Splice`; the `Task` struct, `PendingEvent` RAII, worker `run` loop, `publish_staged`, destructor drain, `validate_copy` / `identical_window` / `unsupported` static helpers, six compute-op stubs, and `record_event` test-seam wrapper stay verbatim per AR-002's CUDA residual contract) and `src/cuda/device.cpp` after AR-003 adds `CudaDevice::supported_data_types()` returning a span over `kCudaSupportedDataTypes`. AR-005 does not land in parallel with AR-002 or AR-003 in those two TUs; it lands strictly after both. AR-005's contract: the `iom::cuda_detail::DriverCalls` six-pointer struct (`init` / `device_get_count` / `device_get` / `primary_ctx_retain` / `ctx_set_current` / `primary_ctx_release`), the inline `cuda_error` / `check_cuda` helpers, and the `driver_calls` extern/definition in `src/cuda/driver.{hpp,cpp}` are exactly the symbols and shapes named in this spec — no concurrent edits by AR-002 or AR-003 add or shadow them.
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-005`
**Review severity:** low
**Review verification:** verified, confidence 90

## Outcome

The six claimed lifecycle/factory/context driver calls the CUDA backend issues (`cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxRelease`) pass through `iom::cuda_detail::driver_calls` (`src/cuda/driver.hpp:7-14`); runtime allocation calls (`cuMemAlloc`, `cuMemFree`) remain direct in `synchronous_transfer` and are out of this seam's coverage. The duplicated driver-API error helpers (`cuda_error`, `check_cuda`) move out of the two `src/cuda/*.cpp` translation units and live exactly once in a private header at `src/cuda/driver.hpp`. The tests at `test/cuda/test_cuda_smoke.cpp:77-296` extend the existing `DriverCallsRestore` / `DriverCallProbe` / `counting_*` fixture families with three new counters and three new counting fixtures (`counting_init`, `counting_device_get_count`, `counting_device_get`), add the three corresponding `calls.init = ...` / `calls.device_get_count = ...` / `calls.device_get = ...` assignments to the three existing failure-injection test cases (`:204-236`, `:238-270`, `:272-296`), and add one new `TEST_CASE("CUDA driver-call seam intercepts every claimed driver call")` that asserts every seam entry fires on the simple `make_cuda_device(0, allocator)` smoke path. The factory path's behavior on the existing failure-injection scenarios (`factory releases retained context when activation fails`, `factory releases retained context when device allocation fails`, `factory dismisses the primary-context guard on success`) is preserved bit-for-bit. The CUDA smoke suite remains green.

The fix is *complete the seam*: the table covers the six claimed lifecycle/factory/context driver calls (`cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxRelease`); runtime allocation calls (`cuMemAlloc`, `cuMemFree`) in `synchronous_transfer` stay direct and are not part of this seam. The alternative (`delete the table`) is rejected because three test cases at `test_cuda_smoke.cpp:204-296` already encode the seam as the production hook; deleting it would force a substantial test rewrite and discard useful coverage for the failure-injection path.

## Current failure

Invariant: the injection seam is the single dispatch path for the calls it claims to cover; error attribution keeps operation names for every failure (`docs/changes/0001-tensor-view/review.md` "AR-005" → Invariant).

`src/cuda/driver.hpp:7-14` declares:

```cpp
struct DriverCalls {
    CUresult (*primary_ctx_retain)(CUcontext*, CUdevice) = &cuDevicePrimaryCtxRetain;
    CUresult (*ctx_set_current)(CUcontext) = &cuCtxSetCurrent;
    CUresult (*primary_ctx_release)(CUdevice) = &cuDevicePrimaryCtxRelease;
};
extern DriverCalls driver_calls;
```

That table covers three of the seven driver calls the backend issues:

| Call | Source | Routed today? |
|---|---|---|
| `cuInit` | `src/cuda/device.cpp:192` | direct |
| `cuDeviceGetCount` | `src/cuda/device.cpp:195` | direct (special-cased for `CUDA_ERROR_NO_DEVICE`) |
| `cuDeviceGet` | `src/cuda/device.cpp:212` | direct |
| `cuDevicePrimaryCtxRetain` | `src/cuda/device.cpp:216` | via `driver_calls.primary_ctx_retain` |
| `cuCtxSetCurrent` (factory) | `src/cuda/device.cpp:220` | via `driver_calls.ctx_set_current` |
| `cuCtxSetCurrent` (`ContextGuard` inside `synchronous_transfer`) | `src/cuda/copy.cu:64` | **direct** |
| `cuDevicePrimaryCtxRelease` (`PrimaryCtxGuard`) | `src/cuda/device.cpp:56` | via `driver_calls.primary_ctx_release` |
| `cuDevicePrimaryCtxRelease` (`~CudaDevice`) | `src/cuda/device.cpp:79` | via `driver_calls.primary_ctx_release` |
| `cuMemAlloc` / `cuMemFree` | `src/cuda/copy.cu:369,400,404` | direct (`synchronous_transfer` driver-API allocation) |

So at `src/cuda/copy.cu:62-66` the `ContextGuard` constructor invokes `cuCtxSetCurrent` directly while the factory and `CudaDevice::activate` route the same call through `driver_calls.ctx_set_current`. A test that stubs `ctx_set_current` to fail (the failure-injection case `factory releases retained context when activation fails` at `test/cuda/test_cuda_smoke.cpp:204-236`) exercises the factory path but never touches the `synchronous_transfer` path that `ContextGuard` guards. The injection seam is therefore *partial*, not a single dispatch path.

The error helpers are *also* duplicated:

- `src/cuda/device.cpp:21-37` defines the private `cuda_error(operation, status)` constructor and the `void check_cuda(operation, status)` function in an anonymous namespace. The body calls `cuGetErrorName` and `cuGetErrorString`, builds `<operation> failed with <name>: <description>`, and throws `std::runtime_error` (`CUDA_SUCCESS` short-circuits). Used by `device.cpp:101-103, 192, 199, 210-212, 214-216, 218-220`.
- `src/cuda/copy.cu:35-51` defines identical bodies in another anonymous namespace. Used by `copy.cu:64, 369, 404`.

The two TUs cannot share the helpers through the project's existing `driver.hpp` because that header only declares `DriverCalls`; it does not include the helper definitions. The shared header carries the helpers as inline function definitions.

## Scope

- **`src/cuda/driver.hpp`** becomes the single private declaration site for the CUDA driver's dispatch seam and the driver-API error helpers. The header keeps the `iom::cuda_detail::DriverCalls` struct verbatim (operator names unchanged) and adds:
  - `[[nodiscard]] std::runtime_error cuda_error(const char* operation, CUresult status);`
  - `void check_cuda(const char* operation, CUresult status);`
  - the `driver_calls` extern stays where it is.

  The two function bodies move verbatim from `src/cuda/device.cpp:21-37` into the new functions (they are short and declared `inline`; one definition per build, no ODR violation even if the header is included by both `src/cuda/device.cpp` and `src/cuda/copy.cu`). Because they are now in a `cuda.h`-included header, the `<cuda.h>` include at the top of `driver.hpp` (`src/cuda/driver.hpp:3`) is sufficient — `cuGetErrorName` / `cuGetErrorString` are reachable through `<cuda.h>`. `<stdexcept>`, `<string>`, `<cstddef>`, and `<cstdint>` are already in the `<cuda.h>`-included header transitively or are added in the header as additional includes. No new include requirement is imposed on translation units that already include `<cuda.h>` and `<stdexcept>` (`device.cpp:3-10` and `copy.cu:1-17` already have both).

- **`src/cuda/driver.cpp`** is created as a tiny (~10-line) translation unit that holds the single definition of `iom::cuda_detail::driver_calls` (`DriverCalls driver_calls{};`). This is the existing definition at `src/cuda/device.cpp:16` relocated to its own TU because `device.cpp` is a non-`inline` definition and a header-only definition would force every TU that includes `driver.hpp` to allocate its own copy or to switch to `inline`. The `driver.cpp` is a private TU added to the `iom_cuda` target only.

- **`DriverCalls` grows three new function pointers appended after the existing three** (each defaults to the real driver symbol) so every `cu*` call the backend issues is reachable from the test seam. The struct's field order is the existing three first (preserving the existing struct verbatim), then the three new ones appended at the tail: `primary_ctx_retain` → `ctx_set_current` → `primary_ctx_release` → `init` → `device_get_count` → `device_get`.

  ```cpp
  struct DriverCalls {
      CUresult (*primary_ctx_retain)(CUcontext*, CUdevice) = &cuDevicePrimaryCtxRetain;
      CUresult (*ctx_set_current)(CUcontext) = &cuCtxSetCurrent;
      CUresult (*primary_ctx_release)(CUdevice) = &cuDevicePrimaryCtxRelease;
      CUresult (*init)(unsigned int) = &cuInit;
      CUresult (*device_get_count)(int*) = &cuDeviceGetCount;
      CUresult (*device_get)(CUdevice*, int) = &cuDeviceGet;
  };
  ```

  The struct's layout extends at the tail; aggregate-initialization by name is unchanged, so the existing three function pointers and the existing three test assignments at `test/cuda/test_cuda_smoke.cpp:215-218, 249-252, 283-286` continue to compile without change. The CUDA factory continues to work the same way it does today; the only visible difference is that the three calls now go through a function pointer instead of statically, which is the entire point of the seam.


- **`src/cuda/device.cpp`** is updated:

  - `cuda_detail::DriverCalls cuda_detail::driver_calls{};` (`src/cuda/device.cpp:16`) is removed (now in `driver.cpp`).
  - The `cuda_error` / `check_cuda` helpers at `src/cuda/device.cpp:21-37` are deleted; the two anonymous-namespace definitions are replaced with `#include "driver.hpp"` (already present) and the symbol resolves to the inline functions in the header.
  - `cuInit(0)` at `src/cuda/device.cpp:192` becomes `check_cuda("cuInit", cuda_detail::driver_calls.init(0));`.
  - `cuDeviceGetCount(&device_count)` at `src/cuda/device.cpp:195-199` becomes `CUresult count_status = cuda_detail::driver_calls.device_get_count(&device_count);`. The `CUDA_ERROR_NO_DEVICE` special-case at `src/cuda/device.cpp:196-198` is preserved verbatim because the special-case exists in the production code today and the seam must not change semantics.
  - `cuDeviceGet(&device, ordinal)` at `src/cuda/device.cpp:210-212` becomes `cuda_detail::driver_calls.device_get(&device, static_cast<int>(device_ordinal));`.
  - The `CudaDevice::activate` body at `src/cuda/device.cpp:100-104` keeps `driver_calls.ctx_set_current(context_)`; the direct `cuCtxSetCurrent` it currently invokes is the one being delegated (it already is).
  - The `CudaDevice::~CudaDevice` body at `src/cuda/device.cpp:77-82` and `PrimaryCtxGuard::~PrimaryCtxGuard` at `src/cuda/device.cpp:54-58` keep `driver_calls.primary_ctx_release(device_)`; unchanged.
  - The factory `make_cuda_device` body at `src/cuda/device.cpp:190-227` (lines 218-220 of `cuCtxSetCurrent` via `driver_calls.ctx_set_current`, lines 214-216 of `cuDevicePrimaryCtxRetain` via `driver_calls.primary_ctx_retain`) is otherwise unchanged.

- **`src/cuda/copy.cu`** is updated:

  - The `cuda_error` / `check_cuda` anonymous-namespace definitions at `src/cuda/copy.cu:35-51` are deleted (they become header-inline through `driver.hpp`, which `copy.cu` already `#include`s indirectly via `copy.hpp`).
  - `ContextGuard::ContextGuard` at `src/cuda/copy.cu:62-66` changes from `check_cuda("cuCtxSetCurrent", cuCtxSetCurrent(context));` to `check_cuda("cuCtxSetCurrent", cuda_detail::driver_calls.ctx_set_current(context));`. The `ContextGuard` struct body and constructor signature are otherwise byte-identical.

- **`test/cuda/test_cuda_smoke.cpp`** is extended (not rewritten) to prove the three new seam entries intercept at the production sites:

  - The `DriverCallProbe` struct at `test/cuda/test_cuda_smoke.cpp:41-47` gains three more counters (`init_count`, `device_get_count_count`, `device_get_count_value`, `device_get_count_device`, `device_get_count_ordinal`) to match the three new entries.
  - The three test fixtures `counting_primary_ctx_retain` / `failing_ctx_set_current` / `pass_through_ctx_set_current` / `counting_primary_ctx_release` at `test/cuda/test_cuda_smoke.cpp:51-75` gain three corresponding `counting_init` / `counting_device_get_count` / `counting_device_get` implementations. Each records the call's arguments and forwards to the real driver.
  - The three existing failing-probe tests at `test/cuda/test_cuda_smoke.cpp:204-236, 238-270, 272-296` keep their `DriverCallsRestore` + `DriverCallProbe` setup and add an `iom::cuda_detail::driver_calls.init = &counting_init;` (etc.) assignment line each. The test seams continue to use the same `DriverCallsRestore` RAII at `test/cuda/test_cuda_smoke.cpp:77-94`.

- **`CMakeLists.txt:146-178`** (the `iom_cuda` target) adds `src/cuda/driver.cpp` to the source list. No other target touches this file. The `iom_cuda` target's `target_include_directories(iom_cuda PRIVATE ${PROJECT_SOURCE_DIR}/src/cuda)` (implicit through the source list) makes `driver.hpp` available to `driver.cpp`, `device.cpp`, and `copy.cu`.

- **Public headers are not touched.** `include/iom/cuda/device.hpp:1-19` keeps its declaration-free, public-API contract. The `src/cuda/` private headers (`copy.hpp`, `driver.hpp`) continue to be `#include`d via the `iom_cuda` target's own include path, not through the public include directory. The shared conformance harness under `test/backend/` is not touched.

## Implementation references

- **Create:** `src/cuda/driver.cpp` — a single TU holding `iom::cuda_detail::DriverCalls iom::cuda_detail::driver_calls{};` (the existing definition at `src/cuda/device.cpp:16` relocated). The TU `#include`s `driver.hpp` and nothing else; no `main()`, no helpers, no anonymous namespace. ~10 lines including the include guard preamble.

- **Modify:** `src/cuda/driver.hpp:1-16` — extend the header. After the existing `DriverCalls` struct, add:
  ```cpp
  [[nodiscard]] std::runtime_error cuda_error(
          const char* operation, CUresult status);
  void check_cuda(const char* operation, CUresult status);
  ```
  Inside the `DriverCalls` struct, append three more fields after the closing brace of the existing struct body:
  ```cpp
  CUresult (*init)(unsigned int) = &cuInit;
  CUresult (*device_get_count)(int*) = &cuDeviceGetCount;
  CUresult (*device_get)(CUdevice*, int) = &cuDeviceGet;
  ```
  Add `#include <stdexcept>` and `#include <string>` after `#include <cuda.h>` (the helpers throw and build a `std::runtime_error` whose `std::string` carries the operation name + name + description; both must be included at the declaration site). At the bottom of the header, after `extern DriverCalls driver_calls;`, append the inline bodies of `cuda_error` and `check_cuda` verbatim from `src/cuda/device.cpp:21-37`. Both functions become `inline` (the function definitions) — the header is private and included from two TUs at most.

- **Modify:** `src/cuda/device.cpp`:
  - Delete `cuda_detail::DriverCalls cuda_detail::driver_calls{};` (line 16).
  - Delete the anonymous-namespace `cuda_error` (lines 21-31) and `check_cuda` (lines 33-37). The functions resolve from `driver.hpp`.
  - Replace `check_cuda("cuInit", cuInit(0));` (line 192) with `check_cuda("cuInit", cuda_detail::driver_calls.init(0));`.
  - Replace `CUresult count_status = cuDeviceGetCount(&device_count);` (line 195) with `CUresult count_status = cuda_detail::driver_calls.device_get_count(&device_count);`. The `CUDA_ERROR_NO_DEVICE` branch (lines 196-198) is preserved.
  - Replace `cuDeviceGet(&device, static_cast<int>(device_ordinal))` (line 212) with `cuda_detail::driver_calls.device_get(&device, static_cast<int>(device_ordinal))`.
  - The `primary_ctx_retain`, `ctx_set_current`, `primary_ctx_release` paths at lines 56, 79, 103, 216, 220 stay unchanged (they already route through the table).
  - `PrimaryCtxGuard` (lines 47-65), `invalid_ordinal` (lines 39-45), `CudaDevice` (lines 67-180), `CudaTensor` (lines 119-180), and `make_cuda_device` body's remaining shape (lines 190-227) are otherwise unchanged.

- **Modify:** `src/cuda/copy.cu`:
  - Delete the anonymous-namespace `cuda_error` (lines 35-45) and `check_cuda` (lines 47-51). Both functions resolve from `driver.hpp` (already transitively included through `copy.hpp`).
  - Replace `check_cuda("cuCtxSetCurrent", cuCtxSetCurrent(context));` (line 64) inside `ContextGuard::ContextGuard` with `check_cuda("cuCtxSetCurrent", cuda_detail::driver_calls.ctx_set_current(context));`.
  - `synchronous_transfer` at `src/cuda/copy.cu:354-405` keeps the direct `cuMemAlloc` and `cuMemFree` calls (lines 369, 400, 404). The `check_cuda("cuMemAlloc", ...)` and `check_cuda("cuMemFree", ...)` wrappers at lines 369 and 404 continue to be the helpers from `driver.hpp` (now header-inline). The `ContextGuard guard(context_)` at line 358 now goes through `driver_calls.ctx_set_current` (the seam is complete).
  - `cudaMemcpy` / `cudaMemset` (lines 371-396) and the `cudaStreamSynchronize(nullptr)` calls (lines 389-390) stay as-is — they are runtime API calls, not driver API calls, and are not in the seam's coverage.

- **Modify:** `CMakeLists.txt:150-156` — add `src/cuda/driver.cpp` to the `iom_cuda` source list:
  ```cmake
  add_library(iom_cuda STATIC
          src/cuda/device.cpp
          src/cuda/copy.cu
          src/cuda/copy.hpp
          src/cuda/driver.cpp
          src/cuda/driver.hpp
          include/iom/cuda/device.hpp
  )
  ```
  The `target_compile_definitions(iom_cuda PRIVATE IOM_CUDA_ENABLED=1)` at `CMakeLists.txt:175-178` is unchanged. The `iom_cuda` target's existing `-fprivate-...` include resolution through `add_library` is sufficient because the source paths are relative to `PROJECT_SOURCE_DIR`; `driver.hpp` is found by `driver.cpp`, `device.cpp`, and `copy.cu` through `target_include_directories(iom_cuda ...)` plus the source path resolution. The `iom_rocm` / `iom_sycl` / `iom_ttnn` targets are NOT modified.

- **Modify:** `test/cuda/test_cuda_smoke.cpp`:
  - Extend `DriverCallProbe` (lines 41-47) with three more counters tracking `init_count`, the most recent `device_get_count_value` returned, and the most recent `device_get` ordinal.
  - Add three new fixture functions after `counting_primary_ctx_release` (line 75): `counting_init`, `counting_device_get_count`, `counting_device_get`. Each increments its counter (and stores its argument if relevant) and forwards to the real driver.
  - Inside the existing three test cases (`factory releases retained context when activation fails` lines 204-236; `factory releases retained context when device allocation fails` lines 238-270; `factory dismisses the primary-context guard on success` lines 272-296), after the existing three `calls.primary_ctx_retain = ...` / `calls.ctx_set_current = ...` / `calls.primary_ctx_release = ...` assignments, add `calls.init = &counting_init; calls.device_get_count = &counting_device_get_count; calls.device_get = &counting_device_get;`. The `DriverCallsRestore` RAII (lines 77-94) restores every field of the table (including the three new ones) on destruction — no changes to the RAII body.
  - Add a new `TEST_CASE("CUDA driver-call seam intercepts every claimed driver call")` after line 296 that exercises all six entries and asserts each counter increments on the simple `make_cuda_device(0, allocator)` smoke path. The test follows the existing "factory dismisses the primary-context guard on success" pattern at lines 272-296 and asserts `init_count == 1`, `device_get_count_count == 1`, `device_get_count_value > 0`, `device_get_count_value == device_get_ordinal + 1`, `primary_ctx_retain` returned a context, and `release_count == 0` while the `CudaDevice` is alive (and `release_count == 1` after `device.reset()`).

- **Read:** `docs/changes/0001-tensor-view/29-AR-001-share-cuda-rocm-copy-queue/spec.md` (read for boundary only) — AR-001's shared algorithm lives in `src/shared/standard_tiled_copy.inl`, and its backend-neutral staging helper is `iom::gpu_algorithm::compute_staging_size` in `include/iom/gpu_algorithm.hpp`. AR-001 does not own `cuda_error`, `check_cuda`, `ContextGuard`, or `DriverCalls`; those remain the AR-005 CUDA-private seam. AR-005 must remain link-compatible with the post-AR-001 thin CUDA translation unit and must not recreate shared algorithm helpers.
- **Read:** `docs/changes/0001-tensor-view/26-ST-005-dedicate-cuda-transfer-stream` (predecessor task) — the CUDA transfer-stream change touches the `synchronous_transfer` function body, including the `cudaStreamCreateWithFlags` call. Adding the AR-005 seam does not change the meaning of `synchronous_transfer`. The CUDA concurrency fix at ST-005 lands independently; this task does not block it.
- **Read:** `docs/changes/0001-tensor-view/21-ST-001-pad-rocm-subbyte-staging-word/spec.md` — only relevant here to confirm `synchronous_transfer`'s overall shape is unchanged.
- **Tests:** `test/cuda/test_cuda_smoke.cpp:77-94` (the `DriverCallsRestore` RAII), `:204-296` (the three existing test cases exercising the seam) — these exercise every entry of the seam-by-name and provide the conformance pattern this task extends. `test/backend/backend_conformance_other.hpp` — unchanged.
- **No driver-symbol direct call outside the seam.** The source-call regex `grep -nP 'cuInit\s*\(|cuDeviceGetCount\s*\(|cuDeviceGet\s*\(|cuDevicePrimaryCtxRetain\s*\(|cuCtxSetCurrent\s*\(|cuDevicePrimaryCtxRelease\s*\('` against `src/cuda/device.cpp` and `src/cuda/copy.cu` returns no matches; operation-name string literals such as `"cuInit"` passed as the first argument to `check_cuda(...)` are allowed and are NOT evidence of a direct call. The default-initializer regex `grep -nE '&(cuInit|cuDeviceGetCount|cuDeviceGet|cuDevicePrimaryCtxRetain|cuCtxSetCurrent|cuDevicePrimaryCtxRelease)'` against `src/cuda/driver.hpp` returns exactly six matches — the default-initialized function pointers `&cuInit` / `&cuDeviceGetCount` / `&cuDeviceGet` / `&cuDevicePrimaryCtxRetain` / `&cuCtxSetCurrent` / `&cuDevicePrimaryCtxRelease`. The default-initializer regex against `src/cuda/driver.cpp` returns zero matches because the definition `DriverCalls driver_calls{};` is aggregate-default-initialized with no per-field explicit `&…` initializers. `cuMemAlloc` / `cuMemFree` direct calls in `synchronous_transfer` (`src/cuda/copy.cu:369, 400, 404`) are not part of this seam and stay direct.
## Requirements

- **`src/cuda/driver.hpp`** contains the `iom::cuda_detail::DriverCalls` struct (now with six function pointers in this exact order: `primary_ctx_retain`, `ctx_set_current`, `primary_ctx_release`, `init`, `device_get_count`, `device_get`), the `cuda_error` and `check_cuda` declarations, and the inline bodies of both. The header's `#include` block contains `<cuda.h>`, `<stdexcept>`, `<string>`, and any other standard headers required for the inline bodies. The `extern DriverCalls driver_calls;` declaration stays where it is.
- **`src/cuda/driver.cpp`** is the single TU defining `iom::cuda_detail::DriverCalls iom::cuda_detail::driver_calls{};`. The definition moves from `src/cuda/device.cpp:16`. No other TU contains the definition.
- **`src/cuda/device.cpp`** does not contain a definition of `cuda_error` or `check_cuda` (both resolve from `driver.hpp`); the `driver_calls` definition is gone (it is in `driver.cpp`); the `cuInit(0)` call at the previous line 192 is `check_cuda("cuInit", cuda_detail::driver_calls.init(0));`; the `cuDeviceGetCount(&device_count)` call at the previous line 195 is `cuda_detail::driver_calls.device_get_count(&device_count)`; the `cuDeviceGet(&device, ...)` call at the previous line 212 is `cuda_detail::driver_calls.device_get(&device, ...)`.
- **`src/cuda/copy.cu`** does not contain a definition of `cuda_error` or `check_cuda` (both resolve from `driver.hpp`); the `ContextGuard::ContextGuard` body is `check_cuda("cuCtxSetCurrent", cuda_detail::driver_calls.ctx_set_current(context));` — no direct `cuCtxSetCurrent` call.
- **`CMakeLists.txt:150-156`** lists `src/cuda/driver.cpp` and `src/cuda/driver.hpp` in the `iom_cuda` source list. The `iom_rocm` / `iom_sycl` / `iom_ttnn` / `iom_cpu` targets are NOT modified.
- **No public header changes.** `include/iom/cuda/device.hpp` retains its existing declaration-free contract. `include/iom/iom.hpp`, `include/iom/device.hpp`, `include/iom/tensor.hpp`, `include/iom/alloc.hpp` are not touched.
- **No driver-symbol direct call outside the seam.** Source-call regex `grep -nP 'cuInit\s*\(|cuDeviceGetCount\s*\(|cuDeviceGet\s*\(|cuDevicePrimaryCtxRetain\s*\(|cuCtxSetCurrent\s*\(|cuDevicePrimaryCtxRelease\s*\('` against `src/cuda/device.cpp` and `src/cuda/copy.cu` returns no matches; operation-name string literals such as `"cuInit"` passed as the first argument to `check_cuda(...)` are allowed and are NOT evidence of a direct call. Default-initializer regex `grep -nE '&(cuInit|cuDeviceGetCount|cuDeviceGet|cuDevicePrimaryCtxRetain|cuCtxSetCurrent|cuDevicePrimaryCtxRelease)'` against `src/cuda/driver.hpp` returns exactly six matches — the default-initialized function pointers `&cuInit` / `&cuDeviceGetCount` / `&cuDeviceGet` / `&cuDevicePrimaryCtxRetain` / `&cuCtxSetCurrent` / `&cuDevicePrimaryCtxRelease`. The default-initializer regex against `src/cuda/driver.cpp` returns zero matches because the definition `DriverCalls driver_calls{};` is aggregate-default-initialized with no per-field explicit `&…` initializers. `cuMemAlloc` / `cuMemFree` direct calls in `synchronous_transfer` (`src/cuda/copy.cu:369, 400, 404`) are not part of this seam and stay direct.
- **No duplicated error helpers.** `grep -nE '(cuda_error|check_cuda)\b' src/cuda/device.cpp src/cuda/copy.cu src/cuda/driver.cpp src/cuda/driver.hpp` returns one definition site (in `driver.hpp`, inline) and zero per-TU duplicates.
- **Operation names preserved.** Every `check_cuda` call retains its operation-name argument (`"cuInit"`, `"cuDeviceGetCount"`, `"cuDeviceGet"`, `"cuDevicePrimaryCtxRetain"`, `"cuCtxSetCurrent"`, `"cuDevicePrimaryCtxRelease"`, `"cuMemAlloc"`, `"cuMemFree"`). The error message string `"<operation> failed with <name>: <description>"` produced by `cuda_error` is byte-identical to today's text. The exception category stays `std::runtime_error`.
- **Test-probe reachability preserved.** The existing test seam at `test/cuda/test_cuda_smoke.cpp:204-296` continues to pass without changes other than adding the three new `calls.init = &counting_init; calls.device_get_count = &counting_device_get_count; calls.device_get = &counting_device_get;` assignments. The `DriverCallsRestore` RAII at lines 77-94 saves and restores the entire `DriverCalls` aggregate, including the three new fields, automatically.
- **Boundary with AR-001 preserved.** This task owns the seam (six function pointers in `DriverCalls`) and the inline private `cuda_error` / `check_cuda` helpers in `src/cuda/driver.hpp`; AR-005 is the single source for those symbols. AR-001 (`29-AR-001-share-cuda-rocm-copy-queue`) owns the shared algorithm in `src/shared/standard_tiled_copy.inl` and the backend-neutral `compute_staging_size` helper in `include/iom/gpu_algorithm.hpp`; AR-001 does not own, duplicate, or shadow the CUDA driver seam. The post-AR-001 CUDA translation unit consumes `cuda_error` / `check_cuda` through `driver.hpp` rather than defining them locally.

## Non-goals

- Removing `DriverCalls` (the alternative the review names as "`remove the partial seam`"). Three test cases (`:204-236`, `:238-270`, `:272-296`) depend on the seam as the production hook; the seam-completion path is strictly less invasive.
- Routing `cuMemAlloc` and `cuMemFree` through the seam. They are part of the synchronous-transfer code at `src/cuda/copy.cu:369, 400, 404`; the review's seam coverage is the six driver-API calls (`cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxRelease`). `cuMemAlloc` / `cuMemFree` stay as direct calls in `synchronous_transfer`.
- Promoting `cuda_error` / `check_cuda` to a public header. They stay in the `src/cuda/driver.hpp` private header (the same `-private`-include style that `copy.hpp` already uses). They do not appear in `include/iom/`.
- Splitting `cuda_error` / `check_cuda` into `iom::cuda_detail::cuda_error` (a free function) versus inlining them as anonymous-namespace helpers in each TU. The header-inline free-function shape pins the seam's namespace and deduplicates the helpers.
- Touching the AR-001 workstream. `docs/changes/0001-tensor-view/29-AR-001-share-cuda-rocm-copy-queue/spec.md` is read for boundary only and is not modified. AR-005 owns `src/cuda/driver.hpp` (the seam and the inline `cuda_error` / `check_cuda` helpers); AR-001 reuses that header unmodified.
- [ ] Source-call regex `grep -nP 'cuInit\s*\(|cuDeviceGetCount\s*\(|cuDeviceGet\s*\(|cuDevicePrimaryCtxRetain\s*\(|cuCtxSetCurrent\s*\(|cuDevicePrimaryCtxRelease\s*\(' src/cuda/device.cpp src/cuda/copy.cu` returns no matches in those two TUs; operation-name string literals such as `"cuInit"` passed as the first argument to `check_cuda(...)` are allowed and are NOT evidence of a direct call. Default-initializer regex `grep -nE '&(cuInit|cuDeviceGetCount|cuDeviceGet|cuDevicePrimaryCtxRetain|cuCtxSetCurrent|cuDevicePrimaryCtxRelease)' src/cuda/driver.hpp` returns exactly six matches — the default-initialized function pointers. Default-initializer regex against `src/cuda/driver.cpp` returns zero matches because `DriverCalls driver_calls{};` is aggregate-default-initialized with no explicit per-field initializers.
- Touching any public tensor-view, allocator, or `DeviceOps` contract in `include/iom/*.hpp`.
- Adding a fault-injection seam for `cuMemAlloc` / `cuMemFree` / `cudaEventCreateWithFlags` / `cudaEventRecord`. The submission-fault seam at `test/cuda/test_cuda_conformance.cpp` is preserved verbatim.
- Modifying `src/cuda/driver.hpp`'s `driver_calls` struct into a class with non-public state, or introducing any RAII handle around it. The struct stays a plain aggregate of `CUresult`-returning function pointers exactly as it is today.
- Modifying the CMakeLayout. The new TU is added to the existing `iom_cuda` target's source list only; no new library targets are created, no test target is added, no `target_compile_definitions` change.
- Modifying the ROCm mirror at `src/rocm/driver.hpp` (it does not exist; ROCm uses the runtime API throughout `src/rocm/copy.hip`).

## Acceptance criteria

- [ ] Source-call regex `grep -nP 'cuInit\s*\(|cuDeviceGetCount\s*\(|cuDeviceGet\s*\(|cuDevicePrimaryCtxRetain\s*\(|cuCtxSetCurrent\s*\(|cuDevicePrimaryCtxRelease\s*\(' src/cuda/device.cpp src/cuda/copy.cu` returns no matches in those two TUs; operation-name string literals such as `"cuInit"` passed as the first argument to `check_cuda(...)` are allowed and are NOT evidence of a direct call. Default-initializer regex `grep -nE '&(cuInit|cuDeviceGetCount|cuDeviceGet|cuDevicePrimaryCtxRetain|cuCtxSetCurrent|cuDevicePrimaryCtxRelease)' src/cuda/driver.hpp` returns exactly six matches — the default-initialized function pointers. Default-initializer regex against `src/cuda/driver.cpp` returns zero matches because `DriverCalls driver_calls{};` is aggregate-default-initialized with no explicit per-field initializers.
- [ ] `grep -nE '(cuda_error|check_cuda)\b' src/cuda/device.cpp src/cuda/copy.cu` returns no matches for definitions; only the `#include "driver.hpp"` line in each TU is allowed to bring the symbols in (the helpers are inline definitions in the header).
- [ ] `grep -nE 'DriverCalls\s+driver_calls' src/cuda/device.cpp src/cuda/copy.cu src/cuda/driver.cpp` returns exactly one definition, in `src/cuda/driver.cpp`.
- [ ] `grep -nE '\bcuda_error\b|\bcheck_cuda\b' src/cuda/driver.hpp` returns the two function declarations followed by their inline bodies (one definition site per function).
- [ ] `iom::cuda_detail::DriverCalls` has exactly six function-pointer fields in this order: `primary_ctx_retain`, `ctx_set_current`, `primary_ctx_release` (the existing three, preserving the current struct body verbatim), then `init`, `device_get_count`, `device_get` (the three new pointers appended at the tail). The first three retain their existing defaults (`&cuDevicePrimaryCtxRetain`, `&cuCtxSetCurrent`, `&cuDevicePrimaryCtxRelease`). The last three default to `&cuInit`, `&cuDeviceGetCount`, `&cuDeviceGet`.
- [ ] Every `check_cuda(...)` invocation in `src/cuda/device.cpp` and `src/cuda/copy.cu` uses the helper from `driver.hpp` (the symbol resolves through the include). The helper signatures are unchanged.
- [ ] `src/cuda/copy.cu:62-66` `ContextGuard::ContextGuard` body uses `cuda_detail::driver_calls.ctx_set_current(context)`. A test that stubs `driver_calls.ctx_set_current` to return `CUDA_ERROR_OUT_OF_MEMORY` (matching the existing `failing_ctx_set_current` fixture at `test_cuda_smoke.cpp:61-63`) sees the `ContextGuard` constructor throw with the message `"cuCtxSetCurrent failed with CUDA_ERROR_OUT_OF_MEMORY: <description>"` (pre/post-fix message identical because the operation name is unchanged; only the dispatch path differs).
- [ ] `src/cuda/device.cpp:192` `make_cuda_device` calls the `cuInit` path through `driver_calls.init(0)`. The `factory rejects the first unavailable ordinal` test at `test_cuda_smoke.cpp:188-202` continues to pass when `device_count == 0` (the `cuDeviceGetCount` direct call is replaced by a function-pointer call returning the same status; `CUDA_ERROR_NO_DEVICE` is still routed to `invalid_ordinal(ordinal, 0)`).
- [ ] `src/cuda/device.cpp:212` `make_cuda_device` calls `cuDeviceGet` through `driver_calls.device_get`. The factory's `device != nullptr` post-condition is preserved.
- [ ] The three existing failure-injection tests at `test_cuda_smoke.cpp:204-236, 238-270, 272-296` continue to pass after the three new `calls.init = …` / `calls.device_get_count = …` / `calls.device_get = …` assignments are added.
- [ ] The new `TEST_CASE("CUDA driver-call seam intercepts every claimed driver call")` test at `test_cuda_smoke.cpp` passes with all six counters (`init_count`, `device_get_count_count`, `device_get_ordinal`, `primary_ctx_retain_count`, `ctx_set_current_count`, `primary_ctx_release_count` at `device.reset()`) showing the expected production-path call pattern: `init_count == 1`, `device_get_count_count == 1` with `device_get_count_value > 0`, `device_get_count_value == device_get_ordinal + 1`, `primary_ctx_retain` recorded a context, `ctx_set_current_count >= 1` (the factory calls it at least once during `make_cuda_device` and the device's `activate()` may call it again on `create_tensor`), and `primary_ctx_release_count == 1` only after `device.reset()`.
- [ ] The full CUDA smoke suite (`iom_cuda_smoke_tests` target at `test/CMakeLists.txt:89-110`) passes on a CUDA host. The full CUDA conformance suite (`iom_cuda_conformance_tests` target at `test/CMakeLists.txt:111-134`) passes on a CUDA host with the same or higher assertion count as before the change. The `factory dismisses the primary-context guard on success` case (`test_cuda_smoke.cpp:272-296`) is the regression-detection point for the seam-completion invariant.
- [ ] `grep -nE 'DriverCalls\s+driver_calls|extern\s+DriverCalls' src include` returns exactly one definition site (in `src/cuda/driver.cpp`) and exactly one declaration site (in `src/cuda/driver.hpp`).

## Verification

Run the verification through the repository's remote-development procedure (`.agents/skills/remote-development`). Pick a CUDA host from `.remote-hosts.conf` and use a unique task id such as `ar005-cuda`. Wrap hardware steps with `flock /tmp/agent-gpu0.lock` per the procedure.

1. Sync and rebuild:
   ```bash
   .agents/skills/remote-development/scripts/remote-sync <cuda-host> ar005-cuda
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda \
        && cmake --build build -j --target iom_cuda_smoke_tests iom_cuda_conformance_tests'
   ```

2. Run the new seam-completion probe test:
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'flock /tmp/agent-gpu0.lock ./build/test/iom_cuda_smoke_tests -tc="*driver-call seam intercepts every claimed driver call*"'
   ```
   Expected: pass with all six counters showing the production-path call pattern.

3. Run the three existing failure-injection tests:
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'flock /tmp/agent-gpu0.lock ./build/test/iom_cuda_smoke_tests -tc="*factory releases retained context when activation fails*" -tc="*factory releases retained context when device allocation fails*" -tc="*factory dismisses the primary-context guard on success*"'
   ```
   Expected: each test passes; the `release_count == 1` and `released_device == retained_device` assertions hold; the new `init_count` / `device_get_count_count` / `device_get_count_value` assertions (added in the `factory dismisses` case) hold.

4. Run the rest of the CUDA smoke suite unchanged-green:
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'flock /tmp/agent-gpu0.lock ctest --test-dir build --output-on-failure -R "^iom_cuda_smoke_tests$"'
   ```
   Expected: all assertions pass and no skips.

5. Run the CUDA conformance suite unchanged-green:
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'flock /tmp/agent-gpu0.lock ctest --test-dir build --output-on-failure -R "^iom_cuda_conformance_tests$"'
   ```
   Expected: assertion count equal to or greater than the pre-change count; no per-test case fails.

6. Diff evidence — prove every claimed driver call is now seam-routed, the duplicated helpers collapsed, and the `DriverCalls` default initializers are present. The verification uses two distinct greps:

   a. **Source-call grep** against `device.cpp` and `copy.cu` (no matches expected):
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'grep -nP "cuInit\s*\(|cuDeviceGetCount\s*\(|cuDeviceGet\s*\(|cuDevicePrimaryCtxRetain\s*\(|cuCtxSetCurrent\s*\(|cuDevicePrimaryCtxRelease\s*\(" src/cuda/device.cpp src/cuda/copy.cu'
   ```
   Expected: no matches. Operation-name string literals (e.g., `"cuInit"` passed to `check_cuda(...)`) are NOT evidence of a direct call.

   b. **Default-initializer grep** against `driver.hpp` (six matches expected, three in `primary_ctx_retain`/`ctx_set_current`/`primary_ctx_release` defaults and three in the appended `init`/`device_get_count`/`device_get` defaults):
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'grep -nE "&(cuInit|cuDeviceGetCount|cuDeviceGet|cuDevicePrimaryCtxRetain|cuCtxSetCurrent|cuDevicePrimaryCtxRelease)" src/cuda/driver.hpp'
   ```
   Expected: exactly six matches — the six default-initialized function pointers (`&cuInit` / `&cuDeviceGetCount` / `&cuDeviceGet` / `&cuDevicePrimaryCtxRetain` / `&cuCtxSetCurrent` / `&cuDevicePrimaryCtxRelease`).

   c. **Helper grep** against all four files (exactly one definition site, in `driver.hpp`):
   ```bash
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar005-cuda \
     'grep -nE "\b(cuda_error|check_cuda)\b" src/cuda/device.cpp src/cuda/copy.cu src/cuda/driver.cpp src/cuda/driver.hpp'
   ```
   Expected: exactly one inline definition site (in `driver.hpp`'s body); zero per-TU duplicates; every reference in `device.cpp` / `copy.cu` is a `check_cuda(...)` call.

7. After verification, `remote-clean <cuda-host> ar005-cuda` removes the remote mirror.


The expected observation: `iom_cuda_smoke_tests` reports the same or higher assertion count as before this change (the three existing failure-injection cases unchanged plus the new seam-reachability case adds 6 assertions; the rest of the suite is byte-identical); `iom_cuda_conformance_tests` reports the same or higher assertion count; `grep` confirms the seam-completion invariant holds (no direct `cu*` calls in `device.cpp` / `copy.cu` apart from the `cuMemAlloc` / `cuMemFree` direct calls that are out of this seam's coverage; one private header holds both the seam and the helpers; the test probe still intercepts every entry).

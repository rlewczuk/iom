# Provide a per-device non-blocking CUDA stream pool for synchronous host transfers

**Order:** 26
**Priority:** P1 — required bounded CUDA host-transfer concurrency defect whose smallest complete fix is contained in one CUDA-side file plus a small lifetime-safe ownership protocol on `CudaDevice`; does not gate unrelated work.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-005`
**Review severity:** low
**Review verification:** verified (code); impact restated after adversarial check

## Outcome

`iom::cuda_detail::synchronous_transfer` in `src/cuda/copy.cu` executes every `cudaMemcpyAsync`, `cudaMemsetAsync`, scatter/gather kernel launch, and `cudaStreamSynchronize` on a non-blocking stream reserved from a per-device pool. The pool lives on the owning `CudaDevice`; each active transfer acquires one exclusive non-blocking stream, the pool grows by lazy creation when the current pool is exhausted, and idle streams are reused on subsequent transfers. Two threads issuing concurrent `copy_from_host`/`copy_to_host` against tensors on the same device therefore run on distinct non-blocking streams and overlap in wall time on the CUDA timeline; sharing a single stream would still serialize them and would not satisfy the finding. The whole-device-stall premise from the original finding is rejected: the queue's `CudaQueue::stream_` is created with `cudaStreamNonBlocking` (`src/cuda/copy.cu:412-414`) and therefore does not interlock with the legacy default stream. The residual defect — every CUDA host transfer funnelling through the single process-wide legacy default stream and so serializing concurrent host transfers against each other — is fixed by reserving one exclusive non-blocking stream per active transfer from a pool owned by `CudaDevice`. The synchronous-at-return contract from `docs/changes/0001-tensor-view/spec.md` §5.4 (every byte observed on the device before `copy_from_host` returns, every byte read back to the host before `copy_to_host` returns) is preserved by ordering `cudaStreamSynchronize(stream)` on the reserved stream before the synchronous `cudaMemcpy(..., cudaMemcpyDeviceToHost)` on the download path and by issuing the final `cudaStreamSynchronize` before `synchronous_transfer` returns.

The `TransferStreamPool::Scope` destructor distinguishes a healthy transfer from a poisoned one. A healthy scope performs a best-effort `cudaStreamSynchronize(stream_)` and then `pool_->release(stream_)` (which returns the stream to `idle_`). A poisoned scope (marked via `Scope::poison() noexcept`) performs a best-effort `cudaStreamSynchronize(stream_)` and then destroys the stream itself with `cudaStreamDestroy(stream_)` without ever touching `pool_->release(stream_)`; the stream is dropped from the pool entirely. Poisoning is the seam `PF-003` consumes to keep a failed stream out of `idle_` without needing private-member access into `TransferStreamPool`. The synchronize on both paths is best-effort: the error code is observed but not thrown, the destructor is `noexcept`, and any in-flight exception from `synchronous_transfer`'s body propagates unchanged. The cleanup-order invariant on every exit path is therefore: the `Scope` destructor runs while the caller's `ContextGuard guard(context)` is still active (so the synchronize/destroy uses the correct context) and runs *before* the catch handler runs `cuMemFree(staging)` (so the kernel is observed to have finished reading staging before staging is freed).

The pool's lifetime is owned by `CudaDevice`, not by any individual `CudaQueue`, because `Device::create_ops()` (`include/iom/device.hpp:36`, `src/cuda/device.cpp:95-98`) returns a fresh `unique_ptr<DeviceOps>` per call and host-region transfers flow through `Device::region_from_host`/`region_to_host` (`src/cuda/device.cpp:161-175`) without touching any queue. Multiple `CudaQueue` instances on the same `Device` and any concurrent host-region transfer must therefore draw from the same per-device pool; the first `~CudaQueue` to run must not destroy a pool still in use by another `CudaQueue` or by an in-flight `region_from_host`/`region_to_host` call. The lifetime contract from `docs/changes/0001-tensor-view/spec.md` §6 ("The `Device` and every caller-supplied allocator must outlive the tensors and queues that use them") and §5.4 ("Before a host read, the caller waits for outstanding `DeviceOps` writes to participating storage…") is the basis of the protocol: by the time `~CudaDevice` runs, every `Tensor` and every `DeviceOps` on the device is gone, so no transfer can still be in flight and no new transfer can start. The pool's `acquire`/`destroy` operations are serialized under a single mutex with a `closing` state; a transfer that arrives after `closing` is set is rejected, but this is unreachable under the documented contract.

The pool class lives in a private shared header `src/cuda/transfer_pool.hpp` (mirroring the established `src/cuda/driver.hpp` convention used by change 21-ST-004 for the `DriverCalls` struct). `TransferStreamPool` is fully defined in that header so it can be a value member of `CudaDevice` (which lives in `src/cuda/device.cpp`); member function bodies are defined in `src/cuda/copy.cu`. `src/cuda/copy.hpp` includes `src/cuda/transfer_pool.hpp` so that the four-argument declarations of `region_from_host` and `region_to_host` (which take `TransferStreamPool&` as the first parameter) name the complete type and so that `src/cuda/device.cpp` (which already includes `copy.hpp`) sees the type through the transitive include. The header is listed in the `iom_cuda` static-library target alongside `copy.hpp` and `driver.hpp` (mirroring how `driver.hpp` is listed at `CMakeLists.txt:154`).

## Current failure

Invariant: host-side synchronization scope is no broader than necessary, and concurrent host transfers on the same CUDA device each observe an exclusive non-blocking stream so they do not serialize against each other. The pool's lifetime is owned by the device that owns the `CUcontext`, and the `acquire`/`destroy` operations on the pool are serialized so that no transfer observes a partially destroyed pool. On every exit path from `synchronous_transfer` — success, error inside the try block, or error inside the post-try `cuMemFree` — the reserved stream is synchronized before either being released to idle (healthy scope) or being destroyed outright (poisoned scope), and the staging buffer is freed only after the synchronize has returned.

In `src/cuda/copy.cu`, `synchronous_transfer` (lines 354-405) is the single host-transfer path:

- `ContextGuard guard(context)` activates the caller's `CUcontext` (line 358).
- Staging is allocated with `cuMemAlloc(&staging, staging_nbytes)` (line 369), which (driver API) consumes the legacy default stream.
- Upload issues synchronous `cudaMemcpy(..., cudaMemcpyHostToDevice)` (lines 371-375) on the default stream; download first issues `cudaMemset` (lines 380-384) on the default stream.
- Kernel launches go through `launch_view_transfer(nullptr, view, ...)` at lines 376-378 and 385-387; the inner `<<<...>>>` triple-chevron launches `scatter_plane_kernel`/`gather_plane_kernel` on `stream == nullptr`, the legacy default stream handle.
- `cudaStreamSynchronize(nullptr)` at lines 389-390 fences the default stream.
- Download `cudaMemcpy(..., cudaMemcpyDeviceToHost)` at lines 392-396 is synchronous and also targets the default stream.

`nullptr` is `cudaStreamLegacy` (synchronizes with every other legacy-default-stream activity on the device). `CudaQueue::stream_` is `cudaStreamNonBlocking` (`copy.cu:412-414`) and the legacy default stream does not interlock with non-blocking streams, so the whole-device-stall candidate is rejected: queued async work is not stalled by an in-flight transfer. The residual defect: two `std::thread`s each issuing `tensor->view().copy_from_host(...)` against tensors on the same device serialize against each other because every operation funnels through the single shared legacy default stream. A single cached non-blocking stream per device would remove the default-stream interlock but would still serialize every concurrent transfer against every other concurrent transfer on the same stream; the fix therefore provides one exclusive non-blocking stream per active transfer drawn from a per-device pool that grows on demand and reuses idle streams. ROCm demonstrates the same per-call stream pattern in `src/rocm/copy.hip:294-343` (`hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)` per transfer at line 311-312, with `hipStreamSynchronize(stream)` at line 325 and `hipStreamDestroy(stream)` at line 341); CUDA's host transfers have no equivalent mechanism.

Impact is bounded because transfers are synchronous at return: a single caller still completes in the same time on the fixed build as on the pre-fix build. The concurrency defect only manifests when more than one caller issues host transfers concurrently. The original proposal to remove the pool from `~CudaQueue` is rejected as lifetime-unsafe: a queue destructor cannot bound the pool's lifetime because the device, not the queue, owns the `CUcontext`, and concurrent host-region transfers flow through the device, not through any queue. The fix instead moves ownership to `CudaDevice` and serializes `acquire`/`destroy` so a transfer never observes a partially destroyed pool. The earliest draft of this spec put `TransferStreamPool`'s class definition in `src/cuda/copy.cu`'s anonymous namespace; that fails to compile because `CudaDevice` in `src/cuda/device.cpp` needs the complete type for its value member, so the class is moved to a private shared header (`src/cuda/transfer_pool.hpp`) following the established `src/cuda/driver.hpp` convention. A subsequent draft declared `region_from_host`/`region_to_host` in `copy.hpp` without including the new header; that fails because the four-argument signatures must name the complete `TransferStreamPool` type, so `copy.hpp` `#include`s `transfer_pool.hpp` to expose the complete type to both `device.cpp` (which calls the functions) and `copy.cu` (which defines them). A further draft had the `Scope` destructor call `pool_->release(stream_)` unconditionally; that returns a possibly in-flight stream to idle before the catch handler frees staging, leaving a kernel that may still be reading freed staging. The fix performs a best-effort `cudaStreamSynchronize(stream_)` inside `Scope::~Scope()` before any release, so the stream is quiescent before the staging is freed by the catch block. A still later draft had only the healthy-vs-poisoned distinction; `PF-003` needs to drop a stream out of the pool entirely when a transfer fails, and the only seam exposed was the healthy-vs-poisoned destructor. The fix adds an explicit `Scope::poison() noexcept` member that marks the scope so its destructor synchronizes and destroys the stream without returning it to `idle_`. The pool's public `destroy()`, `acquire()`, `release()`, `Scope::stream()` and the data members remain unchanged.

## Scope

- `src/cuda/transfer_pool.hpp` (new private shared header) — declares and defines `iom::cuda_detail::TransferStreamPool` with a `std::vector<cudaStream_t> idle_{}`, a `std::unordered_set<cudaStream_t> in_use_{}`, a `std::mutex mutex_`, a `std::condition_variable cv_`, a `bool closing_ = false`, and the public member functions `Scope acquire()`, `void release(cudaStream_t)`, and `void destroy()`. The nested `Scope` class is declared with public `stream()` accessor, public `poison() noexcept` mutator, and destructor declared in the header. The `Scope` destructor calls `cudaStreamSynchronize(stream_)` (best-effort, error code captured but not thrown), then either `pool_->release(stream_)` if `poisoned_` is `false` or `cudaStreamDestroy(stream_)` (followed by no further pool interaction) if `poisoned_` is `true`. The `Scope` destructor never calls `pool_->release(stream_)` after `poison()` has been called. The header includes `<cuda.h>`, `<cuda_runtime_api.h>`, `<condition_variable>`, `<mutex>`, `<stdexcept>`, `<unordered_set>`, and `<vector>`.
- `src/cuda/copy.hpp` — adds `#include "transfer_pool.hpp"` (relative include, same directory) so the four-argument declarations of `region_from_host`/`region_to_host` name the complete `iom::cuda_detail::TransferStreamPool` type. The declarations become:
  ```cpp
  void region_from_host(
          TransferStreamPool& pool, CUcontext context,
          const TensorView& destination,
          std::span<const std::byte> source);
  void region_to_host(
          TransferStreamPool& pool, CUcontext context,
          const TensorView& source,
          std::span<std::byte> destination);
  ```
  Both declarations take the pool as the first parameter so the device's `TransferStreamPool&` can be passed through without naming the device type.
- `src/cuda/copy.cu` — provides member-function bodies for `TransferStreamPool::acquire()`, `TransferStreamPool::release(cudaStream_t)`, `TransferStreamPool::destroy()`, `TransferStreamPool::Scope::~Scope()`, and `TransferStreamPool::Scope::poison() noexcept`. (The declarations are in `transfer_pool.hpp`; the bodies live in `copy.cu` so they can use the project's existing `check_cuda`/`cuda_error` helpers at `copy.cu:35-51`.) Replaces `synchronous_transfer`'s dependence on the legacy default stream with `auto scope = pool.acquire(); cudaStream_t stream = scope.stream();` plus `stream`-based `cudaMemcpyAsync`/`cudaMemsetAsync`/`launch_view_transfer`/`cudaStreamSynchronize`. The `Scope` RAII ensures the guarded `cudaStreamSynchronize(stream_)` runs at function-scope exit (which is after the explicit `cudaStreamSynchronize(stream)` at lines 389-390 on the success path and after the throw on the error path), before either `pool_->release(stream_)` (healthy) or `cudaStreamDestroy(stream_)` (poisoned) returns the stream to idle or drops it entirely.
- `src/cuda/device.cpp` — `CudaDevice` (lines 67-117) gains a `TransferStreamPool transfer_pool_;` value member (default-constructed; the pool starts empty and grows lazily). The complete type is visible because `copy.hpp` (which `device.cpp:12` already `#include`s) transitively pulls in `transfer_pool.hpp`. `CudaTensor::region_from_host` (lines 161-167) and `CudaTensor::region_to_host` (lines 169-175) forward `transfer_pool_` and `device_.context()` into `cuda_detail::region_from_host`/`region_to_host`. `~CudaDevice` (lines 77-82) is reordered so it runs:
  1. Activate the device's context (`device_.activate()`).
  2. `transfer_pool_.destroy();` — sets `closing_ = true`, waits on `cv_` until `in_use_.empty()`, then `cudaStreamDestroy(s)` for every stream in `idle_`.
  3. Then the existing `driver_calls.primary_ctx_release(device_)` (line 79).
  The destructor therefore marks the pool closing, drains all in-flight transfers, destroys every cached stream while the context is still current, and only then releases the primary context. A `try { ... } catch (...) {}` envelope matches the `~CudaQueue` convention so that driver errors during drain do not prevent the primary-context release. By the time `~CudaDevice` runs, every `Tensor` and every `DeviceOps` on the device is gone under the documented contract; `in_use_` is therefore empty promptly.
- `src/cuda/copy.cu` — `~CudaQueue` (lines 424-448). **No change** to this destructor beyond what already exists; do not erase from any pool, do not call `destroy` on the pool, do not add `cudaStreamSynchronize` on any pool stream. The pool is not the queue's resource.
- `src/cuda/copy.cu` — `class CudaQueue` ctor (lines 409-422). No change; the pool is owned by `CudaDevice`, not by the queue.
- `src/cuda/copy.cu` — `make_queue` (lines 620-622) does not change signature; the queue does not see the pool.
- `CMakeLists.txt` — `iom_cuda` static-library target (lines 150-156): add `src/cuda/transfer_pool.hpp` to the source list alongside `src/cuda/copy.hpp` and `src/cuda/driver.hpp`, mirroring the listing of `driver.hpp` at line 154. No new translation unit is introduced.
- No public API, ABI, exception category, or kernel change. `include/iom/iom.hpp`, `include/iom/device.hpp`, and `include/iom/cuda/device.hpp` remain byte-identical except for comments. `cuda_detail::region_from_host`/`region_to_host` are internal to `iom_cuda` and not part of any public ABI.
- CUDA backend only. `src/rocm/copy.hip` (per-call stream), `src/cpu`, `src/ttnn`, `src/sycl` are untouched.

## Implementation references

- **Create:** `src/cuda/transfer_pool.hpp` — private shared header (mirrors `src/cuda/driver.hpp` style):
  ```cpp
  #pragma once

  #include <cuda.h>
  #include <cuda_runtime_api.h>

  #include <condition_variable>
  #include <cstdio>
  #include <mutex>
  #include <stdexcept>
  #include <unordered_set>
  #include <vector>

  namespace iom::cuda_detail {

  struct TransferStreamPool {
      class Scope {
      public:
          Scope(TransferStreamPool& pool, cudaStream_t stream)
                  : pool_(&pool), stream_(stream), poisoned_(false) {}
          ~Scope();
          Scope(const Scope&) = delete;
          Scope& operator=(const Scope&) = delete;
          Scope(Scope&&) = delete;
          Scope& operator=(Scope&&) = delete;
          [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
          void poison() noexcept;
      private:
          TransferStreamPool* pool_;
          cudaStream_t stream_;
          bool poisoned_;
      };

      [[nodiscard]] Scope acquire();
      void release(cudaStream_t s);
      void destroy();
      [[nodiscard]] std::size_t idle_count_for_testing() const noexcept;


  private:
      std::vector<cudaStream_t> idle_{};
      std::unordered_set<cudaStream_t> in_use_{};
      std::mutex mutex_;
      std::condition_variable cv_;
      bool closing_ = false;
  };

  }  // namespace iom::cuda_detail
  ```
  The header carries only declarations. Member-function bodies live in `src/cuda/copy.cu` so they can use `cudaStreamCreateWithFlags`/`cudaStreamSynchronize`/`cudaStreamDestroy` calls with the project's existing `check_cuda`/`cuda_error` helpers (`copy.cu:35-51`). The `Scope` carries a `poisoned_` flag initialized to `false`; `poison()` flips it to `true`; the destructor reads it to choose between the healthy release path and the poisoned destroy path.
- **Modify:** `src/cuda/copy.hpp` — add `#include "transfer_pool.hpp"` near the existing `#include`s (line 1-9). Update the four-argument declarations:
  ```cpp
  void region_from_host(
          TransferStreamPool& pool, CUcontext context,
          const TensorView& destination,
          std::span<const std::byte> source);

  void region_to_host(
          TransferStreamPool& pool, CUcontext context,
          const TensorView& source,
          std::span<std::byte> destination);
  ```
  Both declarations take the pool as the first parameter so the device's `TransferStreamPool&` is passed through. The complete type of `TransferStreamPool` is visible because `transfer_pool.hpp` is included at the top of `copy.hpp`.
- **Modify:** `src/cuda/copy.cu` — add `#include "transfer_pool.hpp"` near the existing `#include "copy.hpp"` at line 1 (the include is redundant given `copy.hpp` now pulls it in transitively, but the explicit include makes the dependency visible at the TU level and matches the `copy.hpp`/`driver.hpp` precedent). Add member-function bodies:
  ```cpp
  void TransferStreamPool::Scope::poison() noexcept {
      poisoned_ = true;
  }

  std::size_t TransferStreamPool::idle_count_for_testing() const noexcept {
      std::lock_guard<std::mutex> lock(mutex_);
      return idle_.size();
  }

  void TransferStreamPool::Scope::~Scope() noexcept {
      // Best-effort synchronize so any in-flight work on the reserved stream
      // has been observed before the destructor dispatches between the
      // healthy release path and the poisoned drop path. The caller's
      // ContextGuard (constructed before this Scope) is still active here
      // because Scope destruction is part of stack unwinding that happens
      // before ContextGuard destruction. The synchronize status is captured
      // but not thrown: this destructor is implicitly noexcept and must not
      // replace any in-flight exception from synchronous_transfer.
      //
      // If the synchronize itself reports failure (returned status !=
      // cudaSuccess), the stream's state is unknown and the stream must
      // not be returned to idle. We treat this case the same as an
      // explicit poison(): the stream is removed from in_use_ and destroyed,
      // and the synchronize error is observed but not surfaced. The caller
      // already has an exception (or success) from synchronous_transfer; the
      // destructor must not add a second one.
      const cudaError_t sync_status = cudaStreamSynchronize(stream_);
      const bool sync_failed = (sync_status != cudaSuccess);
      const bool drop_stream = poisoned_ || sync_failed;
      if (drop_stream) {
          // Drop the stream entirely; do NOT touch pool_->release(). The
          // stream is removed from in_use_ under pool_->mutex_ (mirroring
          // release()'s erase and cv_.notify_all() without pushing onto
          // idle_) and then destroyed.
          std::lock_guard<std::mutex> lock(pool_->mutex_);
          auto it = pool_->in_use_.find(stream_);
          if (it != pool_->in_use_.end()) {
              pool_->in_use_.erase(it);
              pool_->cv_.notify_all();
          }
          (void)cudaStreamDestroy(stream_);
      } else {
          pool_->release(stream_);
      }
  }

  TransferStreamPool::Scope TransferStreamPool::acquire() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (closing_) {
          throw std::runtime_error(
                  "cudaStreamCreateWithFlags failed with "
                  "cudaErrorStreamDestroyed: TransferStreamPool is closing");
      }
      if (idle_.empty()) {
          cudaStream_t s = nullptr;
          check_cuda(
                  "cudaStreamCreateWithFlags",
                  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
          idle_.push_back(s);
      }
      cudaStream_t s = idle_.back();
      idle_.pop_back();
      try {
          in_use_.insert(s);
      } catch (...) {
          // The reserved stream was never tracked as in-use; return it to the
          // idle pool so it can be reused or destroyed by ~TransferStreamPool.
          idle_.push_back(s);
          throw;
      }
      return Scope{*this, s};
  }

  void TransferStreamPool::release(cudaStream_t s) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = in_use_.find(s);
      if (it != in_use_.end()) {
          in_use_.erase(it);
          idle_.push_back(s);
          cv_.notify_all();
      }
      // If `s` is not in `in_use_` (e.g. acquire() already returned it to
      // idle_ on a throw before reaching `in_use_.insert`, or the destructor
      // already erased it before calling release), the call is a no-op so a
      // duplicated release is harmless.
  }

  void TransferStreamPool::destroy() {
      std::unique_lock<std::mutex> lock(mutex_);
      closing_ = true;
      cv_.wait(lock, [this] { return in_use_.empty(); });
- **Modify:** `src/cuda/copy.cu` — `synchronous_transfer` (lines 354-405). The body uses a nested scope structure that guarantees `Scope::~Scope()` runs *before* `cuMemFree(staging)` on every exit path, so the stream is synchronized or destroyed while the kernel has finished reading staging. The first statement is `ContextGuard guard(context);` (line 358), then `CUdeviceptr staging = 0;` and `std::exception_ptr failure;` are declared, then the function shape is:
  ```cpp
  try {
      check_cuda("cuMemAlloc", cuMemAlloc(&staging, staging_nbytes));
      {
          auto scope = pool.acquire();
          cudaStream_t stream = scope.stream();
          try {
              // transfer body using stream (replacing nullptr with stream
              // and cudaMemcpy with cudaMemcpyAsync, etc.)
          } catch (...) {
              scope.poison();
              throw;
          }
          // Scope destroyed here: poisoned_==true on error -> drop path;
          // poisoned_==false on success -> release path.
      }
  } catch (...) {
      failure = std::current_exception();
  }
  if (failure) {
      if (staging != 0) { (void)cuMemFree(staging); staging = 0; }
      std::rethrow_exception(failure);
  }
  check_cuda("cuMemFree", cuMemFree(staging));
  ```
  The transfer body inside the innermost `try` replaces `cudaMemcpy(..., cudaMemcpyHostToDevice)` (lines 371-375) with `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)`, replaces `cudaMemset(...)` (lines 380-384) with `cudaMemsetAsync(..., staging_nbytes, stream)`, replaces the `nullptr` arguments to the two `launch_view_transfer` calls (lines 376-378 and 385-387) with `stream` (which flows through to the `<<<...>>>` launch stream in `launch_view_transfer` and `launch_copy_plane`), replaces `cudaStreamSynchronize(nullptr)` (lines 389-390) with `cudaStreamSynchronize(stream)`, and keeps the synchronous `cudaMemcpy(..., cudaMemcpyDeviceToHost)` (lines 392-396) which still correctly observes the device buffer because the explicit `cudaStreamSynchronize(stream)` preceding it is on the same stream. The cleanup precedence on every exit path is: inner scope closes first (forcing `Scope::~Scope()` to run while `ContextGuard` is still active) → inner `catch` runs `scope.poison()` and rethrows on the error path → `Scope::~Scope()` runs `cudaStreamSynchronize(stream_)` (best-effort, status observed but not thrown) and dispatches the drop or release path → outer `catch` captures the original exception → outer post-catch `cuMemFree(staging)` runs and `std::rethrow_exception(failure)` re-propagates the original exception. **The stream is synchronized or destroyed before staging is freed on every exit path**, satisfying the failure-path quiescence invariant (Requirements §5) and eliminating the kernel-reads-freed-staging hazard. `synchronous_transfer` itself does not call `scope.poison()` outside the inner `catch`; the nested-structure invariant places the poison call before rethrow so that any thrown exception (from `cudaMemcpyAsync`, `cudaMemsetAsync`, the kernel-launch `check_kernel`, the explicit `cudaStreamSynchronize`, or the synchronous `cudaMemcpy(..., DeviceToHost)`) is always followed by `scope.poison()` before the inner scope closes and the `Scope::~Scope()` destructor runs.
- **Modify:** `src/cuda/device.cpp` — `CudaDevice` (lines 67-117) gains a `TransferStreamPool transfer_pool_;` value member (default-constructed; the pool starts empty and grows lazily). The complete type is visible because `copy.hpp` (which `device.cpp:12` already `#include`s) transitively pulls in `transfer_pool.hpp`. `CudaTensor::region_from_host` (lines 161-167) and `CudaTensor::region_to_host` (lines 169-175) forward `transfer_pool_` and `device_.context()` into `cuda_detail::region_from_host`/`region_to_host`. `~CudaDevice` (lines 77-82) is reordered as described in Scope.
- **Modify:** `CMakeLists.txt` — `iom_cuda` static-library target (lines 150-156): add `src/cuda/transfer_pool.hpp` to the source list alongside `src/cuda/copy.hpp` and `src/cuda/driver.hpp`, mirroring the listing of `driver.hpp` at line 154.
- **Modify:** `src/cuda/copy.cu` — the anonymous namespace no longer contains `TransferStreamPool`; only the member-function bodies live in the TU.
- **Read:** `src/cuda/copy.cu:255-317` `launch_view_transfer` — the existing function already takes a `cudaStream_t stream` and passes it to the `<<<...>>>` triples; passing the reserved pool stream instead of `nullptr` is the only required behaviour change here.
- **Read:** `src/cuda/copy.cu:35-51` `cuda_error`/`check_cuda` — the existing exception category (`std::runtime_error`) and message format ("<operation> failed with <name>: <description>") are reused for any `cudaStreamCreateWithFlags` failure; the `closing_` rejection uses the same exception category and the same message prefix.
- **Read:** `src/cuda/device.cpp:95-98` `CudaDevice::create_ops` — confirms `make_queue` returns a fresh `unique_ptr<DeviceOps>` per call and therefore multiple queues per device are possible; the pool must be device-owned.
- **Read:** `src/cuda/device.cpp:161-175` `CudaTensor::region_from_host`/`region_to_host` — these are the only host-transfer call sites; they now forward the device's pool.
- **Read:** `src/cuda/driver.hpp` — established convention for a private shared header used by both `copy.cu` and `device.cpp`; the new `transfer_pool.hpp` follows the same pattern.
- **Read:** `src/rocm/copy.hip:294-343` — reference for stream creation/destruction discipline and error-message shape (`hipStreamCreateWithFlags` at line 311-312, `hipStreamSynchronize` at line 325, `hipStreamDestroy` at line 341); the device-owned CUDA pool is the chosen minimal delta against this ROCm per-call pattern.
- **Read:** `include/iom/iom.hpp` and `include/iom/cuda/device.hpp` — public surface; must remain untouched.
- **Read:** `docs/changes/0001-tensor-view/spec.md:374-377` (`Tensor`/`Device` lifetime contract) and `:402-440` (§6 device construction) — the `CudaDevice`-owns-pool protocol relies on these: the device outlives its tensors and queues, so by the time `~CudaDevice` runs there can be no in-flight or pending transfer.
- **Tests:** `test/cuda/test_cuda_smoke.cpp` — append three new test cases (see Requirements §10, §11, and §12) next to the existing factory cases (lines 141, 176, 188, 204, 238, 272). `test/CMakeLists.txt:88-104` already wires `iom_cuda_smoke_tests` with `${PROJECT_SOURCE_DIR}/src/cuda` in `target_include_directories`, so the file's existing `#include <cuda.h>` plus `#include "copy.hpp"`/the file's own helpers are sufficient; the new cases use only public `iom::Device`/`TensorView`/`DeviceOps` APIs.
- **Tests:** `test/cuda/test_cuda_conformance.cpp:262-268` (`run_storage_and_transfer_conformance`) and the rest of the CUDA conformance cases at lines 297, 305, 313, 337, 347 — unchanged; the pool is bit-equivalent to the default stream for serial call patterns (a single-threaded caller reserves and releases the same stream repeatedly), so the existing green-keep invariant holds.
- **Tests:** `test/cpu/test_cpu.cpp` and `test/cpu/test_cpu_conformance.cpp` — irrelevant to CUDA; unchanged.
- **PF-003 predecessor contract:** change 26 is a strict predecessor to `PF-003` (`Synchronous host transfers allocate/free full-size device staging per call (plus per-call stream create/destroy on ROCm, plus a full-size memset per download)`, `docs/changes/0001-tensor-view/review.md` §5). On CUDA, change 26 introduces a per-device pool exposed through `iom::cuda_detail::TransferStreamPool` (declared and defined in `src/cuda/transfer_pool.hpp`, member-function bodies in `src/cuda/copy.cu`) that grows on demand and reuses idle streams, removing the per-call `cudaStreamCreateWithFlags`/`cudaStreamDestroy` churn on the hot path. The public seam `PF-003` consumes is `Scope::poison() noexcept`, which marks the reserved scope so its destructor synchronizes and destroys the stream without releasing it to idle. `PF-003`'s staging-reuse layer keeps a cached staging buffer alive across calls; when the staging allocator reaches a state where a stream cannot be reused safely (e.g. the staging pointer changed and an in-flight kernel still references the old pointer), `PF-003` calls `scope.poison()` before the scope destructs, ensuring the poisoned stream is destroyed rather than returned to `idle_` where a future transfer would reuse it. Because `synchronous_transfer` itself now calls `scope.poison()` in its catch block for any transfer error (Requirements §4), `PF-003`'s staging-reuse layer inherits this drop-on-error behavior automatically: a failed transfer causes `synchronous_transfer` to throw and the stream to be dropped from the pool, so `PF-003` cannot accidentally reuse an errored stream. `Scope::stream()`, `TransferStreamPool::destroy()`, `TransferStreamPool::idle_count_for_testing()`, and the data members remain unchanged. `PF-003` consumes the seam through the public `transfer_pool.hpp` header; no private-member access into `TransferStreamPool` is required. The same seam is mirrored on ROCm (`src/rocm/copy.hip:294-343`) and TTNN through their own pool types when those migrations land. Change 26 must land first so the CUDA host-transfer path uses a pool rather than churning one stream per call, leaving `PF-003` a single-axis problem.
## Requirements

1. **Device-owned `TransferStreamPool`, one per `CudaDevice`.** Implementation is grounded in repository convention: `CudaDevice` (the sole owner of a `CUcontext` per `src/cuda/device.cpp:67-117` and `include/iom/device.hpp:36`) holds a `TransferStreamPool transfer_pool_;` value member. The pool's `idle_` starts empty and grows lazily via `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` in `acquire()` when no idle stream is available, so it does not interlock with the legacy default stream and does not cause any whole-device stall. Each `acquire` reserves one idle stream and inserts it into `in_use_`; each `release` removes the stream from `in_use_` and pushes it onto `idle_`. The mutex is held across the `closing_` check, the lazy creation, the pop from `idle_`, the insert into `in_use_`, the erase from `in_use_`, and the push onto `idle_`. The pool is keyed only by the device (no separate `CUcontext` key needed: `CudaDevice::context_` is the single `CUcontext` for that device). *Repository evidence:* `CudaDevice` already owns the primary context (`device.cpp:69-82`); the `Device` virtual interface in `include/iom/device.hpp:36` returns a fresh `unique_ptr<DeviceOps>` per `create_ops()` call, which proves a single device can host multiple `CudaQueue` instances and that `CudaDevice` is the correct lifetime owner.

2. **Class definition in `src/cuda/transfer_pool.hpp`; member-function bodies in `src/cuda/copy.cu`; `src/cuda/copy.hpp` `#include`s the new header.** `iom::cuda_detail::TransferStreamPool` is declared and defined in `src/cuda/transfer_pool.hpp`. The header is included by `src/cuda/copy.hpp` (so the four-argument declarations of `region_from_host`/`region_to_host` name the complete type) and is included directly by `src/cuda/copy.cu` (so the member-function bodies see the class definition). `src/cuda/device.cpp` already includes `src/cuda/copy.hpp` (line 12), so the complete type is visible there transitively; no additional include is required in `device.cpp`. The header is listed in the `iom_cuda` static-library target alongside `copy.hpp` and `driver.hpp` (`CMakeLists.txt:154`). Member-function bodies in `copy.cu` are defined out-of-line using the project's existing `check_cuda`/`cuda_error` helpers. The earlier drafts that placed `TransferStreamPool`'s definition in `copy.cu`'s anonymous namespace or that declared `region_from_host`/`region_to_host` in `copy.hpp` without the new header were both unbuildable: an anonymous-namespace class cannot be a value member of a class in another TU, and a `copy.hpp` declaration that names a type not visible to its includer is ill-formed. The shared header with a transitive include through `copy.hpp` is the chosen fix.

3. **Serialized `acquire`/`destroy` with a `closing` state and exclusive stream per active transfer.** `acquire()` holds `mutex_` while checking `closing_`, while creating a new stream if `idle_` is empty, while popping from `idle_`, and while inserting into `in_use_`. The reserved stream is exclusive to the calling transfer for its entire lifetime on the pool; no other transfer observes the same stream until `release(stream)` runs in the `Scope` destructor. `destroy()` holds `mutex_` while setting `closing_ = true` and while waiting on `cv_` until `in_use_.empty()`; then, while still holding `mutex_`, it calls `cudaStreamDestroy(s)` for every stream in `idle_`. Because `acquire()` and `destroy()` both hold `mutex_`, the `closing_ == true` check cannot race with the destroy body: a thread that observes `closing_ == false` has entered the critical section before `destroy()` set `closing_ = true`, and `destroy()` cannot proceed past the `cv_.wait` until that thread's `Scope` destructor calls `release(stream)` (which also runs under `mutex_`). The lifetime contract from `docs/changes/0001-tensor-view/spec.md` §6 guarantees no transfer enters after `~CudaDevice` is reached, so the `closing_` rejection is unreachable in correct usage; the throw exists only to make a contract violation observable.

4. **`Scope::poison() noexcept`, sync-failure handling, and the drop-on-error destructor path.** The `Scope` class carries a `bool poisoned_` flag initialized to `false` and exposes `void poison() noexcept` that flips the flag to `true`. The destructor is declared `noexcept` and dispatches on `const bool drop_stream = poisoned_ || (cudaStreamSynchronize(stream_) != cudaSuccess)`:
   - Drop path (`drop_stream == true`): runs a best-effort `cudaStreamSynchronize(stream_)` (status captured but not thrown); if `poisoned_` was already `true` (catch path) or the synchronize returned non-`cudaSuccess` (stream state unknown), the destructor under `pool_->mutex_` removes the stream from `in_use_`, notifies `cv_`, and calls `(void)cudaStreamDestroy(stream_)` directly without touching `pool_->release()`. The stream is dropped from the pool entirely; no future `acquire()` will return it.
   - Release path (`drop_stream == false`): runs a best-effort `cudaStreamSynchronize(stream_)` (error observed but not thrown), then calls `pool_->release(stream_)` which removes the stream from `in_use_` and pushes it onto `idle_`. The stream is reused by the next `acquire()`. This path is reached only when no exception propagated from `synchronous_transfer` AND the destructor's `cudaStreamSynchronize(stream_)` returned `cudaSuccess`.
   `synchronous_transfer`'s inner `catch (...)` (Implementation references §"Modify copy.cu synchronous_transfer") runs `scope.poison(); throw;` — every exception inside the inner transfer block (from `cudaMemcpyAsync`, `cudaMemsetAsync`, `launch_view_transfer`, the explicit `cudaStreamSynchronize`, or the synchronous `cudaMemcpy(..., DeviceToHost)`) calls `scope.poison()` before the inner block's closing brace triggers `Scope::~Scope()`. Combined with the destructor's sync-failure handling, the only path that returns the stream to `idle_` is the success path with a successful destructor synchronize. The `Scope::poison()` seam is also available to `PF-003` (and any future staging-reuse layer) that needs to keep a stream out of `idle_` without needing private-member access into `TransferStreamPool`. The public `Scope::poison()`, `Scope::stream()`, `TransferStreamPool::destroy()`, `acquire()`, `release()`, `idle_count_for_testing()` and the data members are the complete seam; `PF-003` does not add a `StreamLease` or `transfer_pool.cpp` translation unit.
5. **Failure-path quiescence: nested scopes force `Scope::~Scope()` to run *before* `cuMemFree(staging)` on every exit path; the error path always drops the stream, never releases it to idle.** `synchronous_transfer` (Implementation references §"Modify copy.cu synchronous_transfer") wraps the per-stream work in an inner block `{ auto scope = pool.acquire(); ... try { ... } catch (...) { scope.poison(); throw; } }` followed by an outer `try { ... } catch (...) { failure = std::current_exception(); }`, then a post-catch `if (failure) { if (staging != 0) { (void)cuMemFree(staging); staging = 0; } std::rethrow_exception(failure); } else { check_cuda("cuMemFree", cuMemFree(staging)); }`. The inner block's closing brace forces `Scope::~Scope()` to run during stack unwinding before the outer `catch` runs and before `cuMemFree(staging)` fires. Concretely:
   ```cpp
   try {                                                       // outer
       check_cuda("cuMemAlloc", cuMemAlloc(&staging, staging_nbytes));
       {                                                       // inner (forces Scope destructor)
           auto scope = pool.acquire();
           cudaStream_t stream = scope.stream();
           try {                                               // innermost
               // transfer body using stream
           } catch (...) {
               scope.poison();
               throw;
           }
           // Scope destroyed here on success (poisoned_==false -> release).
       }
   } catch (...) {
       failure = std::current_exception();
   }
   if (failure) {
       if (staging != 0) { (void)cuMemFree(staging); staging = 0; }
       std::rethrow_exception(failure);                        // original exception
   }
   check_cuda("cuMemFree", cuMemFree(staging));
   ```
   On the error path, the inner `catch (...)` calls `scope.poison()` and rethrows; the inner block's closing brace runs `Scope::~Scope()` next, which dispatches the drop path (because `poisoned_ == true`) — synchronize (best-effort, status observed but not thrown), erase `stream_` from `in_use_` under `pool_->mutex_`, notify `cv_`, `cudaStreamDestroy(stream_)`. Only after `Scope::~Scope()` returns does the outer `catch` capture `failure`; only after the outer `catch` runs does `cuMemFree(staging)` fire. The original exception from the inner block propagates to the caller via `std::rethrow_exception(failure)`. On the success path, the innermost `try` body completes; the inner block's closing brace runs `Scope::~Scope()` with `poisoned_ == false`, which dispatches the release path — synchronize (returns `cudaSuccess` because the explicit `cudaStreamSynchronize(stream)` on the line above already returned), `pool_->release(stream_)` → `idle_`. The outer `catch` is not entered; the post-catch `check_cuda("cuMemFree", cuMemFree(staging))` runs. **The stream is synchronized or destroyed before staging is freed on every exit path.** The `Scope` destructor synchronizes the reserved stream on every exit path; the synchronize is best-effort (an error from the synchronize call is observed but does not replace the in-flight exception (if any) from `synchronous_transfer`'s body, and the destructor never throws); the synchronize runs while the caller's `ContextGuard guard(context)` is still active because `Scope` is constructed inside the inner block, so `Scope`'s destruction is part of the stack-unwinding sequence inside the inner block — `ContextGuard`'s destruction only runs at function-scope exit. The staging buffer is freed only after the stream has been synchronized or destroyed; no kernel reads from freed staging. The stream is returned to `idle_` only when no exception escaped the inner block AND the destructor's synchronize returned `cudaSuccess`; under all other conditions (any exception in `synchronous_transfer`, or any synchronize failure in `~Scope`) the stream is dropped from the pool. No concurrent transfer ever reuses a stream that observed an errored state, and no caller of `cudaStreamDestroy` races an in-flight kernel.

6. **Cleanup-on-throw in `acquire`.** If `cudaStreamCreateWithFlags` succeeds and `in_use_.insert(s)` throws (e.g. `std::bad_alloc` from the unordered_set), `acquire()` pushes `s` back onto `idle_` under the same `mutex_` lock before propagating the exception, so the just-created stream is not leaked. `release()` is idempotent against a duplicated call: if a caller has already pushed `s` back onto `idle_`, the `in_use_.find(s)` check in `release()` returns `end()` and the call is a no-op. The `Scope`'s destructor (Requirements §4 and §5) synchronizes the reserved stream, then dispatches the drop path (erasing from `in_use_` and calling `cudaStreamDestroy`) when `scope.poisoned_` is `true` or the destructor synchronize returns non-`cudaSuccess`, and dispatches the release path (`pool_->release(stream_)`) only when `scope.poisoned_` is `false` AND the synchronize returned `cudaSuccess`. If `acquire()` itself throws before constructing the `Scope` (the `std::bad_alloc` path above), no `Scope` is constructed and no destructor runs.

7. **Failure path of stream creation.** A driver error from `cudaStreamCreateWithFlags` propagates as `std::runtime_error` from the existing `cuda_error` helper with the unchanged message format `"cudaStreamCreateWithFlags failed with <name>: <description>"`. No stream is inserted into `in_use_` on failure, and `idle_` is not modified. The inner `catch (...)` block at the bottom of the inner block in `synchronous_transfer` (Implementation references §"Modify copy.cu synchronous_transfer") runs `scope.poison(); throw;` for any post-creation failure (from `cudaMemcpyAsync`, `cudaMemsetAsync`, `launch_view_transfer` kernel launch, `cudaStreamSynchronize`, or the synchronous `cudaMemcpy(..., DeviceToHost)`); the inner block closes, `Scope::~Scope()` dispatches the drop path (because `poisoned_ == true`) and erases `stream_` from `in_use_` and `cudaStreamDestroy`s it; then the outer `catch` captures the exception pointer and the post-catch path frees staging via `(void)cuMemFree(staging)` and rethrows the original exception via `std::rethrow_exception(failure)`. The reserved stream is never released to `idle_` on the error path; the original exception from `synchronous_transfer` is the one that propagates to the caller.

8. **Synchronous-at-return contract preserved.** Every byte written into the device buffer is observed before `synchronous_transfer` returns because the explicit `cudaStreamSynchronize(stream)` on the reserved stream precedes the synchronous download `cudaMemcpy`. Every byte read back to the host buffer completes before `synchronous_transfer` returns because `cudaMemcpy(..., cudaMemcpyDeviceToHost)` is synchronous and ordered after the same `cudaStreamSynchronize`. The host-callable `TensorView::copy_from_host`/`copy_to_host` API at `include/iom/iom.hpp:296-297` is unchanged. On the success path, the reserved stream is synchronized, observed to be quiescent, and released to `idle_` so the next `acquire()` may reuse it; on every error path, the reserved stream is synchronized, observed to be in some state, and destroyed outright without ever reaching `idle_` (Requirements §4 and §5).

9. **Bit-exact transfer semantics.** No kernel, no host-encoding, and no staging-size change. `logical_nbytes` and the staging padding `+ sizeof(unsigned int)` at lines 359-366 are unchanged. `cuMemAlloc`/`cuMemFree` remain the staging allocator (driver-API; the staging allocation is independent of which stream executes the transfer work because every kernel and every async memcpy targets the reserved pool stream and so observes the staging through it).

10. **Exception categories preserved.** All `check_cuda`/`check_kernel` call sites inside `synchronous_transfer` continue to throw `std::runtime_error` from `cuda_error`/`kernel_error` with the existing message format. `std::overflow_error` for staging overflow at line 363 is unchanged. `std::invalid_argument` for wrong-size host transfers is unchanged (raised upstream in `TensorView::copy_from_host`/`copy_to_host`). The `Scope::~Scope()` synchronize error is observed but does not throw and does not replace any in-flight exception; a failed destructor synchronize triggers the drop path (Requirements §4) but never adds an exception. `Scope::poison()` is `noexcept`.

11. **Regression test `CUDA host transfers on independent threads do not share a stream`.** Add to `test/cuda/test_cuda_smoke.cpp` next to the existing factory cases. The case must:
    - precondition with `REQUIRE(cuInit(0) == CUDA_SUCCESS)` and at least one CUDA device (mirror the existing factory cases at lines 141-175).
    - create two distinct tensors on the same CUDA device with `logical_nbytes` large enough that each `copy_from_host` call dominates kernel-launch latency on the target machine (≥ a few MiB of F32).
    - launch two `std::thread`s, each performing `N ≥ 4` alternating `copy_from_host`/`copy_to_host` on its tensor, with `std::barrier`-style rendezvous before the timed section so both threads start within microseconds of each other.
    - record wall-clock start and end timestamps around the timed section, compare to the sum of per-transfer wall times of the same operations issued serially in the same process.
    - assert `(end - start) < 2 * serial_total_time`. A single cached stream per device (the rejected single-stream variant) measures close to `2 * serial_total_time` because of intra-pool serialization; the pre-fix legacy-default-stream control also measures close to `2 * serial_total_time`. The post-fix pool reserves an exclusive stream per active transfer so the two threads overlap in wall time and the bound is satisfied.
    - be hardware-only; the case is guarded by the same `cuInit(0) == CUDA_SUCCESS && device_count > 0` precondition already used in this file (lines 142-148, 177-185, 188-198, 204-208, 238-243, 272-277).

12. **Multi-queue regression test `CUDA host transfers from multiple queues on one device share the transfer-stream pool`.** Add to `test/cuda/test_cuda_smoke.cpp` next to the case in Requirements §11. The case must:
    - precondition as in Requirements §11.
    - create one CUDA `Device` and two `unique_ptr<DeviceOps>` queues from `device->create_ops()`. Create two tensors on the same device.
    - launch two `std::thread`s. Thread A calls `tensor_a->view().copy_from_host(...)` repeatedly on its tensor and never touches a queue. Thread B submits `oid` tokens on the second queue via `queue_b->copy(...)` and waits on them with `queue_b->wait(oid)`. The case exercises concurrent host-region transfer and queue-issued copy on the same device; both the host-region transfer and the queue's copy observe the device's `TransferStreamPool` (for host transfers) and the queue's `stream_` (for queue work) without lifetime conflict because both threads respect the device-lifetime contract (no `~CudaDevice` runs while either thread is active).
    - assert no `cudaErrorInvalidHandle` is observed across either thread's work, and that the host-region thread's `(end - start) < 2 * serial_total_time` bound still holds. A control that erases the pool from `~CudaQueue` (the rejected ownership protocol) destroys pool streams while the host-region thread is still in flight, and the host-region thread trips `cudaErrorInvalidHandle` on its next transfer.
    - be hardware-only and use the same `cuInit(0) == CUDA_SUCCESS && device_count > 0` precondition; the test name is fixed to `CUDA host transfers from multiple queues on one device share the transfer-stream pool`.

12a. **Errored-transfer-drop regression test `CUDA errored host transfer drops its stream from the pool`.** Add to `test/cuda/test_cuda_smoke.cpp` next to the case in Requirements §13. The case must:
    - precondition as in Requirements §11.
    - create one CUDA `Device`, one tensor, one transfer pool implicitly via the device. Set `g_submission_fault.store(SubmissionFault::third_plane_launch)` via `cuda_detail::inject_submission_fault_for_testing` (mirrors the existing `test_cuda_conformance.cpp:347` pattern) so that `launch_copy_plane` throws at `copy.cu:348-350` and the catch path runs. Issue one `tensor->view().copy_from_host(...)` wrapped in `REQUIRE_THROWS_AS(...)`. Reset the fault afterward.
    - assert that the next transfer (with no fault injection) starts cleanly: `idle_count_for_testing() == 0` immediately after the faulted transfer (no stream was released to idle, because the inner catch at the bottom of the inner block ran `scope.poison()` and the destructor's drop path destroyed it before the outer catch freed staging). On the rejected control (inner catch does not call `scope.poison()` and the Scope destructor takes the release path), the failed stream is released to `idle_` and `idle_count_for_testing() == 1`, and the next transfer would reuse that stream with unknown state. Additionally, the test asserts that the thrown exception category matches `std::runtime_error` (the `check_kernel` throws) and that staging is freed before `REQUIRE_THROWS_AS` returns (verifiable: querying `cuMemAlloc`'s address range after the throw shows the address is unmapped, demonstrating that the post-catch `cuMemFree` ran *after* the inner scope closed and *after* the `Scope::~Scope()` destructor synchronized/dropped the stream).
    - be hardware-only and use the same `cuInit(0) == CUDA_SUCCESS && device_count > 0` precondition; the test name is fixed to `CUDA errored host transfer drops its stream from the pool`.

13. **Poison-seam regression test `CUDA poisoned Scope drops its stream from the pool`.** Add to `test/cuda/test_cuda_smoke.cpp` next to the case in Requirements §12. The case must:
    - precondition as in Requirements §11.
    - include a CUDA smoke-only test TU that exercises the `Scope::poison()` path through a small harness that calls `cuda_detail::make_queue` (or constructs a `TransferStreamPool` directly via a test-only helper that lives in `src/cuda/transfer_pool.hpp`'s public interface; the test must not reach into private members). The harness acquires a `Scope`, calls `scope.poison()`, lets the `Scope` go out of scope, and asserts that `TransferStreamPool::idle_.empty()` is `true` afterward (the stream was destroyed, not returned to idle). Because `idle_` is private, the assertion is exposed via a `transfer_pool.hpp` public function `[[nodiscard]] std::size_t TransferStreamPool::idle_count_for_testing() const noexcept` (declared in the header, defined in `copy.cu`) that locks `mutex_` and returns `idle_.size()`. The test asserts `idle_count_for_testing() == 0` after a poisoned scope destructs and `idle_count_for_testing() == 1` after a healthy scope destructs (the test reserves one stream, lets it go out of scope via the healthy path, and observes that idle is non-empty).
    - assert no `cudaError_t cudaStreamSynchronize` reports an invalid-handle error from `nsys profile --cuda-um-cpu-page-faults=true` on the poisoned path.
    - be hardware-only and use the same `cuInit(0) == CUDA_SUCCESS && device_count > 0` precondition; the test name is fixed to `CUDA poisoned Scope drops its stream from the pool`.

14. **Destructor parity.** `~CudaDevice` (lines 77-82) is reordered to: activate the device's context; `transfer_pool_.destroy();`; then the existing `driver_calls.primary_ctx_release(device_)` (line 79). `~CudaQueue` (lines 424-448) gains no new code; it does not erase the pool, does not destroy any pool stream, does not add any new `cudaEventSynchronize`/`cudaStreamSynchronize` (those additions remain ST-001/ST-003 scope).

15. **No public API or ABI change.** Public headers (`include/iom/iom.hpp`, `include/iom/device.hpp`, `include/iom/cuda/device.hpp`) are byte-identical pre- and post-fix. `src/cuda/transfer_pool.hpp` is private (mirrors `src/cuda/driver.hpp`) and never exported through any public header. The `iom_cuda` exported symbol set is unchanged (verifiable with `nm -D --defined-only libiom_cuda.so`).

16. **Performance discriminator on hardware.** A focused `nsys profile` run of the case in Requirements §11 must show the two threads' `cudaStreamSynchronize` intervals on their respective reserved pool streams **overlapping** in wall time post-fix, and **falling strictly end-to-end** on the pre-fix control (which uses the legacy default stream) and on the rejected single-cached-stream variant (which uses one shared non-blocking stream).

17. **PF-003 predecessor contract.** This change is the explicit predecessor to `PF-003` for the CUDA host-transfer stream-churn axis: per-call `cudaStreamCreateWithFlags`/`cudaStreamDestroy` is replaced by a per-device pool exposed through `iom::cuda_detail::TransferStreamPool` (header `src/cuda/transfer_pool.hpp`, bodies in `src/cuda/copy.cu`) that grows on demand and reuses idle streams; `PF-003` then owns the `cuMemAlloc`/`cuMemFree` staging churn and the per-call `cudaMemset` churn on the download path. Change 26 does not address the allocator churn; `PF-003` does not address the stream churn. The two changes are non-overlapping and must land in this order on the CUDA host-transfer path. The public seam `PF-003` consumes is `Scope::poison() noexcept`, plus the existing `Scope::stream()`, `Scope::~Scope()`, `TransferStreamPool::destroy()`, `acquire()`, `release()`. `PF-003` does not add a `StreamLease`, a `transfer_pool.cpp` translation unit, or any private-member access into `TransferStreamPool`. The `Scope::poison()` seam is mirrored on ROCm and TTNN through their own pool types when those migrations land.

## Non-goals

- The whole-device-stall premise in the original finding is rejected and explicitly excluded; no fix here re-asserts that premise.
- The allocator churn flagged by `PF-003` (`cuMemAlloc`/`cuMemFree` per transfer, per-call stream create/destroy on ROCm, full-size `cudaMemset` per download) is out of scope; only the per-call stream create/destroy on the CUDA host-transfer path is replaced here. The remaining allocator/`cudaMemset` churn is `PF-003` scope.
- Queue teardown fences (`ST-001`) and tensor-destruction fences (`ST-003`) are unrelated tasks; this change does not add `cudaEventSynchronize`/`cudaStreamSynchronize` inside `~CudaQueue`'s event/stream drain, and the `cudaStreamDestroy(s)` calls inside `TransferStreamPool::destroy()` are on the cached pool streams only, not on the queue's `stream_`.
- Shared kernel hoisting (`AR-001`) and staged-worker-queue scaffolding unification (`AR-002`); the per-backend copy paths stay separate.
- ROCm, CPU, TTNN, SYCL code paths; the fix is CUDA-only.
- Replacing the driver-API `cuMemAlloc`/`cuMemFree` with `cudaMalloc`/`cudaFree`; that is `PF-003` scope.
- Any change to `TensorView::copy_from_host`/`copy_to_host` or to `docs/changes/0001-tensor-view/spec.md` §5.4.
- Lifetime-safe ownership that erases the pool from any individual `CudaQueue` destructor; that protocol is rejected because it cannot bound the pool's lifetime across multiple queues and concurrent host-region transfers sharing one `CUcontext`. The only accepted protocol is device-owned with serialized `acquire`/`destroy` under `mutex_`.
- A single shared cached stream per device; that protocol does not satisfy the finding because concurrent transfers on the same device would still serialize against each other. The only accepted concurrency protocol is one exclusive non-blocking stream per active transfer drawn from a per-device pool.
- Concurrent `~CudaDevice` with in-flight host transfers. The repository contract (`docs/changes/0001-tensor-view/spec.md` §6: the `Device` must outlive its tensors and queues; §5.4: host transfers are synchronous and the caller waits for outstanding `DeviceOps` writes before a host read) excludes this scenario. The fix relies on that contract; no regression test exercises device destruction mid-transfer.
- A pimpl unique_ptr for the pool; the chosen seam is the value-member-with-complete-type pattern via the private shared header `transfer_pool.hpp` (included by `copy.hpp` so its declarations name the complete type and transitively visible to `device.cpp` through `copy.hpp`). Pimpl would require a custom deleter and a forward declaration in `device.cpp`, both unnecessary given the small size of the class and the existing `driver.hpp` precedent.
- Replacing the original exception with a synchronize error from `Scope::~Scope`. The `Scope` destructor is `noexcept`; the synchronize error is captured but never thrown, so the original exception from `synchronous_transfer`'s body (if any) is the one that propagates.
- A `StreamLease` or `transfer_pool.cpp` translation unit. The seam for `PF-003` is the public `transfer_pool.hpp` header (declaration of `TransferStreamPool` + `Scope` + `Scope::poison()`) and the bodies in `copy.cu`; no new translation unit is introduced and no alternative class name is exposed.

## Acceptance criteria

- [ ] `src/cuda/transfer_pool.hpp` exists, declares `iom::cuda_detail::TransferStreamPool` with the data members `std::vector<cudaStream_t> idle_`, `std::unordered_set<cudaStream_t> in_use_`, `std::mutex mutex_`, `std::condition_variable cv_`, `bool closing_ = false`, and the public methods `Scope acquire()`, `void release(cudaStream_t)`, `void destroy()`, plus the nested `Scope` class declaration. `Scope` declares `cudaStream_t stream() const noexcept`, `void poison() noexcept`, and a destructor.
- [ ] `src/cuda/copy.hpp` `#include`s `src/cuda/transfer_pool.hpp` and carries the four-argument declarations `void region_from_host(TransferStreamPool&, CUcontext, const TensorView&, std::span<const std::byte>)` and `void region_to_host(TransferStreamPool&, CUcontext, const TensorView&, std::span<std::byte>)` naming the complete `TransferStreamPool` type.
- [ ] `src/cuda/copy.cu` provides the out-of-line member-function bodies for `TransferStreamPool::acquire`, `TransferStreamPool::release`, `TransferStreamPool::destroy`, `TransferStreamPool::Scope::~Scope()`, and `TransferStreamPool::Scope::poison() noexcept`. `Scope::~Scope()` is declared `noexcept`; its body captures the return value of `cudaStreamSynchronize(stream_)` into `const cudaError_t sync_status` (the value is observed but not thrown), computes `const bool drop_stream = poisoned_ || (sync_status != cudaSuccess)`, and dispatches: if `drop_stream` is `true`, erase `stream_` from `in_use_` under `pool_->mutex_`, notify `cv_`, and call `(void)cudaStreamDestroy(stream_)`; if `drop_stream` is `false`, call `(void)` (the destructor's cudaStreamSynchronize returned `cudaSuccess` so the return value is intentionally discarded) then `pool_->release(stream_)`. The destroy body, the acquire body, the release body, and the poison body in `transfer_pool.hpp` are unchanged in their high-level shape from the previous draft: clean-up ordering with respect to `mutex_` and `cv_`, lazy stream creation with rollback on `std::bad_alloc`, idempotent release with linear lookup, and a flag-only `poison` that returns immediately.
- [ ] `transfer_pool.hpp` declares `[[nodiscard]] std::size_t TransferStreamPool::idle_count_for_testing() const noexcept` (a public function whose body in `copy.cu` locks `mutex_` and returns `idle_.size()`). No private-member access from tests.
- [ ] `src/cuda/device.cpp` declares `CudaDevice::transfer_pool_` as a value member of type `iom::cuda_detail::TransferStreamPool`. The complete type is visible through the transitive include of `transfer_pool.hpp` via `copy.hpp` (`device.cpp:12`). The compile succeeds (no incomplete-type error at the `CudaDevice` definition).
- [ ] `CMakeLists.txt` `iom_cuda` static-library target (lines 150-156) lists `src/cuda/transfer_pool.hpp` alongside `src/cuda/copy.hpp` and `src/cuda/driver.hpp`.
- [ ] Inside `src/cuda/copy.cu::synchronous_transfer` (lines 354-405), `grep -n 'nullptr' src/cuda/copy.cu` returns zero matches where `nullptr` is passed as the stream argument to `cudaMemcpyAsync` / `cudaMemsetAsync` / `launch_view_transfer` / `cudaStreamSynchronize`. (Pre-fix returns matches at lines 372-375, 376-378, 380-384, 385-387, 389-390.)
- [ ] `synchronous_transfer` builds the nested scope structure: `ContextGuard guard(context)` at line 358; `CUdeviceptr staging = 0; std::exception_ptr failure;` declared; an outer `try { check_cuda("cuMemAlloc", ...) ... {` that opens an inner block `{ auto scope = pool.acquire(); cudaStream_t stream = scope.stream(); try { ... transfer body ...; } catch (...) { scope.poison(); throw; } }`; the outer `catch (...) { failure = std::current_exception(); }`; and the post-catch `if (failure) { if (staging != 0) { (void)cuMemFree(staging); staging = 0; } std::rethrow_exception(failure); } else { check_cuda("cuMemFree", cuMemFree(staging)); }`. The inner block's closing brace forces `Scope::~Scope()` to run *before* the outer `catch` and before `cuMemFree(staging)` on every exit path. The inner catch calls `scope.poison()` and rethrows so any exception in the transfer body (from `cudaMemcpyAsync`, `cudaMemsetAsync`, `launch_view_transfer`, the explicit `cudaStreamSynchronize`, or the synchronous `cudaMemcpy(..., DeviceToHost)`) marks the scope poisoned before the `Scope::~Scope()` destructor runs.
- [ ] No stream with an in-flight or unknown state is ever returned to idle, and staging is never freed before the stream has been synchronized or destroyed: on every error path, the inner catch calls `scope.poison()` and rethrows → the inner scope closes, `Scope::~Scope()`'s `cudaStreamSynchronize(stream_)` runs (best-effort, status observed but not thrown) → because `poisoned_ == true` the destructor dispatches the drop path (erase `stream_` from `in_use_` under `pool_->mutex_`, notify `cv_`, then `cudaStreamDestroy(stream_)`) → the outer catch captures `failure` → the post-catch `(void)cuMemFree(staging)` runs → `std::rethrow_exception(failure)` rethrows the original exception. On the success path, no exception propagates, `scope.poison()` is not called, the destructor's `cudaStreamSynchronize(stream_)` returns `cudaSuccess`, and the release path returns the stream to `idle_`; then the outer try completes and `check_cuda("cuMemFree", cuMemFree(staging))` runs. Verified by `nsys profile --cuda-um-cpu-page-faults=true` of the CUDA conformance case `run_transfer_error_conformance` (`test/cuda/test_cuda_conformance.cpp:313`) and the submission-failure case at line 347: no `cudaErrorInvalidHandle` is observed, no use-after-free diagnostic on staging, and `idle_count_for_testing() == 0` after each error case (because the failed transfers' streams are dropped, not released).
- [ ] `Scope::~Scope()` and `Scope::poison()` are `noexcept`; if `cudaStreamSynchronize` returns an error, that error is observed but neither function throws, neither replaces any in-flight exception from `synchronous_transfer`'s body, and the destructor dispatches the drop path because `drop_stream == true`.
- [ ] The new test case `CUDA host transfers on independent threads do not share a stream` in `test/cuda/test_cuda_smoke.cpp` passes on the fixed build with `(end - start) < 2 * serial_total_time`, and is observed to fail against the pre-fix control (legacy-default-stream serialization) on CUDA hardware.
- [ ] The new test case `CUDA host transfers from multiple queues on one device share the transfer-stream pool` in `test/cuda/test_cuda_smoke.cpp` passes on the fixed build with no `cudaErrorInvalidHandle` observed in `nsys profile`. The control that erases the pool from `~CudaQueue` (the rejected ownership protocol) trips `cudaErrorInvalidHandle` on the second queue's destructor or on the next host-region transfer after that destructor.
- [ ] The new test case `CUDA errored host transfer drops its stream from the pool` in `test/cuda/test_cuda_smoke.cpp` passes on the fixed build: when a host transfer throws from any of `cudaMemcpyAsync`, `cudaMemsetAsync`, the kernel-launch `check_kernel`, the explicit `cudaStreamSynchronize` (`SubmissionFault::event_create` or `SubmissionFault::third_plane_launch` injection at `src/cuda/copy.cu:25-32` triggers `check_kernel("CUDA copy kernel launch", cudaErrorInvalidValue)` at line 348-350, which throws), the catch block calls `scope.poison()`, and the destructor's drop path removes the stream from `in_use_` and destroys it. The test asserts `idle_count_for_testing() == 0` after the failed transfer; on the rejected control (catch block does not call `scope.poison()`), the stream is released to `idle_` and `idle_count_for_testing() == 1`, demonstrating that the failed stream would have been reused by a future transfer. Verified by `nsys profile --cuda-um-cpu-page-faults=true` of `run_transfer_error_conformance` (`test/cuda/test_cuda_conformance.cpp:313`) and the submission-failure case at line 347: no `cudaErrorInvalidHandle` and no use-after-free diagnostic.
- [ ] The new test case `CUDA poisoned Scope drops its stream from the pool` in `test/cuda/test_cuda_smoke.cpp` passes on the fixed build: after a `Scope` is acquired and `scope.poison()` is called and the `Scope` destructs, `idle_count_for_testing() == 0`; after a healthy `Scope` acquires and destructs without poisoning, `idle_count_for_testing() == 1`. No `cudaError_t cudaStreamSynchronize` reports an invalid-handle error from `nsys profile --cuda-um-cpu-page-faults=true`.
- [ ] `nsys profile` of the case in Requirements §11 records the two threads' `cudaStreamSynchronize` intervals on distinct pool streams overlapping in wall time post-fix; on the unfixed control those intervals fall strictly end-to-end on the legacy default stream. The single-cached-stream variant also fails the timing bound and is therefore not an acceptable simplification.
- [ ] All existing CUDA smoke cases (`CUDA factory reports a live hardware device and owns its context`, `CUDA tensor rejects misaligned allocator storage exactly once`, `CUDA factory rejects the first unavailable ordinal`, `CUDA factory releases retained context when activation fails`, `CUDA factory releases retained context when device allocation fails`, `CUDA factory dismisses the primary-context guard on success`) and CUDA conformance cases (`run_storage_and_transfer_conformance`, `run_async_copy_conformance`, `run_copy_error_conformance`, `run_transfer_error_conformance`, `submission remains transactional across post-enqueue failures`, `full shared suite`) pass unchanged on CUDA hardware.
- [ ] Public API and ABI parity: `include/iom/iom.hpp`, `include/iom/device.hpp`, and `include/iom/cuda/device.hpp` are byte-identical pre- and post-fix except for comments; `iom_cuda`'s exported symbol set is unchanged (verifiable with `nm -D --defined-only libiom_cuda.so`).
- [ ] `~CudaDevice` runs `device_.activate(); transfer_pool_.destroy(); driver_calls.primary_ctx_release(device_);` in that fixed order. `~CudaQueue` adds no new code.
- [ ] `TransferStreamPool::destroy()` holds `mutex_`, sets `closing_ = true`, waits on `cv_` until `in_use_.empty()`, and only then calls `cudaStreamDestroy(s)` for every stream in `idle_` while still holding `mutex_`. `TransferStreamPool::acquire()` holds `mutex_` while reading `closing_`, while performing the lazy `cudaStreamCreateWithFlags`, while popping from `idle_`, and while inserting into `in_use_`, so a concurrent `destroy()` cannot observe `closing_ == false` and proceed to destroy the pool after a new transfer reserves a stream. `acquire()` pushes `s` back onto `idle_` under the same lock if `in_use_.insert(s)` throws, so the just-created stream is not leaked on a `std::bad_alloc` from the unordered_set.
- [ ] No regression test exercises `~CudaDevice` while a host transfer is in flight; the lifetime contract from `docs/changes/0001-tensor-view/spec.md` §6 forbids that scenario, and the spec records this exclusion.

## Verification

Follow `.agents/skills/remote-development` (local workspace authoritative; unique remote task directory). The fix's mechanism and the regression cases both require real CUDA hardware; CPU-only builds do not exercise the stream path.

```bash
.agents/skills/remote-development/scripts/remote-sync <cuda-host-alias> st005-stream-pool
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'cmake -S . -B build/cuda-st005 -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda && cmake --build build/cuda-st005 --target iom_cuda_smoke_tests iom_cuda_conformance_tests'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'ctest --test-dir build/cuda-st005 --output-on-failure -R "^iom_cuda_(smoke|conformance)_tests$"'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'grep -n "nullptr" src/cuda/copy.cu'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'nsys profile --stats=true --output=st005-overlap ./build/cuda-st005/iom_cuda_smoke_tests --test-case="CUDA host transfers on independent threads do not share a stream"'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'nsys profile --stats=true --output=st005-multi-queue ./build/cuda-st005/iom_cuda_smoke_tests --test-case="CUDA host transfers from multiple queues on one device share the transfer-stream pool"'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'nsys profile --stats=true --output=st005-errored ./build/cuda-st005/iom_cuda_smoke_tests --test-case="CUDA errored host transfer drops its stream from the pool"'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'nsys profile --stats=true --output=st005-poison ./build/cuda-st005/iom_cuda_smoke_tests --test-case="CUDA poisoned Scope drops its stream from the pool"'
.agents/skills/remote-development/scripts/remote-exec <cuda-host-alias> st005-stream-pool \
  'nsys profile --cuda-um-cpu-page-faults=true --stats=true --output=st005-error-quiescence ./build/cuda-st005/iom_cuda_conformance_tests --test-case="CUDA conformance: transfer failures keep metadata and ownership"'
.agents/skills/remote-clean <cuda-host-alias> st005-stream-pool
```

Expected observations:

- The case `CUDA host transfers on independent threads do not share a stream` passes on the fixed build; on the unfixed control it fails the `(end - start) < 2 * serial_total_time` bound and Nsight Systems reports the two threads' intervals strictly end-to-end on the legacy default stream.
- The case `CUDA host transfers from multiple queues on one device share the transfer-stream pool` passes on the fixed build with no `cudaErrorInvalidHandle` from `nsys profile`; the rejected ownership protocol (erase on `~CudaQueue`) trips `cudaErrorInvalidHandle` either on the second queue's destructor or on the next host-region transfer after that destructor.
- The case `CUDA errored host transfer drops its stream from the pool` passes on the fixed build: after a host transfer that throws (triggered via `SubmissionFault::third_plane_launch` injection), the inner block's catch ran `scope.poison()` and the `Scope::~Scope()` destructor dispatched the drop path, so the stream was destroyed and `idle_count_for_testing() == 0`. On the rejected control (inner catch does not call `scope.poison()` and the destructor takes the release path), the failed stream is released to `idle_` and `idle_count_for_testing() == 1`. The thrown exception category is `std::runtime_error`; the staging buffer is freed before `REQUIRE_THROWS_AS` returns.
- The case `CUDA poisoned Scope drops its stream from the pool` passes on the fixed build: `idle_count_for_testing() == 0` after a poisoned scope destructs, `idle_count_for_testing() == 1` after a healthy scope destructs, no `cudaError_t cudaStreamSynchronize` reports an invalid-handle error from `nsys profile --cuda-um-cpu-page-faults=true`.
- The CUDA conformance case `CUDA conformance: transfer failures keep metadata and ownership` passes on the fixed build; `nsys profile --cuda-um-cpu-page-faults=true` shows no use-after-free diagnostic on staging and no `cudaErrorInvalidHandle` from `cudaStreamSynchronize` after a transfer failure. The pre-fix control (no destructor synchronize) shows a use-after-free diagnostic on staging under the same harness.
- All existing CUDA smoke and conformance cases remain green; `iom_cuda`'s exported symbol set is unchanged.
- `~CudaDevice` runs `device_.activate(); transfer_pool_.destroy(); driver_calls.primary_ctx_release(device_);` in that fixed order; `~CudaQueue` adds no new code; the event/stream drain at `src/cuda/copy.cu:436-444` remains ST-001/ST-003 scope.
- The lifetime contract from `docs/changes/0001-tensor-view/spec.md` §6 (Device outlives tensors/queues) and §5.4 (host transfers are synchronous at return and callers wait for outstanding `DeviceOps` writes) is the precondition under which the protocol is correct; no verification step assumes concurrent destruction.
- ROCm (per-call `hipStreamNonBlocking`), CPU, TTNN, and SYCL code paths are unchanged. The CUDA host-transfer path consumes at most one new `cudaStreamCreateWithFlags` per *distinct* concurrent transfer (a per-device pool grows on demand and reuses idle streams); the per-call stream churn on CUDA is removed and `PF-003` may now proceed against the remaining allocator/`cudaMemset` churn using the same `transfer_pool.hpp` seam (`Scope::poison()` plus the existing public surface).

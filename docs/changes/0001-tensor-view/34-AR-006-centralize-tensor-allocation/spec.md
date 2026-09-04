# Centralize the allocator-backed tensor construction and the 32-byte alignment contract behind one backend-neutral helper

**Order:** 34
**Priority:** P1 — bounded backend-architecture defect (review severity low) that concentrates the allocate/validate/single-free pattern in four structurally divergent copies; preserves the established 32-byte alignment invariant on the `Allocator::alloc` interface and each backend's existing exception message verbatim. Required finishing work to keep the standard-layout tensor construction contract uniform across CPU, CUDA, ROCm, and SYCL. This task applies after its two blockers land: AR-003 (`31-AR-003-expose-device-capabilities`) edits the per-backend `Device` API and capability surface; AR-005 (`33-AR-005-unify-cuda-driver-seam`) edits the CUDA driver-call seam. AR-006 modifies the same per-backend `Device::create_tensor` and `Tensor` ctor/dtor TUs as AR-003 (its `CudaDevice`/`RocmDevice`/`SyclDevice` member edits and capability predicate additions live in `src/cuda/device.cpp`, `src/rocm/device.cpp`, `src/sycl/device.cpp`) and the same CUDA TU as AR-005 (`src/cuda/device.cpp` driver-context plumbing). The three edits must be ordered: AR-003 first, AR-005 second, AR-006 last; concurrent edits to the same backend-device TU would collide on `kStorageAlignment` removal and the per-backend `allocate_aligned_storage` substitution.
**Blocked by:** `31-AR-003-expose-device-capabilities`, `33-AR-005-unify-cuda-driver-seam`
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-006`
**Review severity:** low
**Review verification:** verified, confidence 85

## Outcome

A single backend-neutral construction helper pair in `include/iom/detail/aligned_storage.hpp` (header-only) owns the `allocate → null-check → alignment-check → free-on-misalign → throw` and `activate → free → null` contracts for every standard-layout backend's tensor constructor and destructor. CPU, CUDA, ROCm, and SYCL each call into those helpers with one pre-allocation / pre-release callback (the backend's device-context activation) and one exact `std::runtime_error` message preserving each backend's current text verbatim. The `kStorageAlignment = 32` constant collapses to one declaration in the shared header. The `Allocator::alloc` interface gains a documented alignment obligation that names the 32-byte requirement. TTNN's native-storage path is untouched.

## Current failure

Invariant: caller-allocator storage is rejected (`std::bad_alloc` for null, `std::runtime_error` for misalignment) before the tensor is live; the rejected allocation is freed exactly once; the constructor never lets a misaligned allocation escape.

`kStorageAlignment = 32` is triplicated today at `src/cpu/device.cpp:23`, `src/cuda/device.cpp:19`, and `src/rocm/device.cpp:18` (SYCL was added since the review and carries the same constant at `src/sycl/device.cpp:27`). The allocate/check/free-and-throw dance exists in four structurally divergent variants:

- **CPU** (`src/cpu/device.cpp:162-181`): calls `allocator_.alloc(nbytes)`, throws `std::bad_alloc` on null, checks the alignment, calls `allocator_.free(address)` (without nullifying the local first), throws `std::runtime_error("CPU tensor storage is not 32-byte aligned")`. The destructor (`src/cpu/device.cpp:183-185`) calls `allocator_.free(address_)` unconditionally — no null check, no activate. The "32-byte alignment" obligation appears nowhere on the `Allocator` interface (`include/iom/alloc.hpp:8-14`).
- **CUDA** (`src/cuda/device.cpp:119-154`): calls `device_.activate()` first (driver-API `cuCtxSetCurrent`), then `allocator_.alloc(nbytes)`, throws `std::bad_alloc` on null, sets `address_ = nullptr` BEFORE `allocator_.free(misaligned)`, throws `std::runtime_error("CUDA tensor storage is not 32-byte aligned")`. Destructor activates the device under a try/catch, then frees under a try/catch, nulls `address_`. The free-through-member-then-null pattern is correct; the file is one refactor away from a double-free if `free` ever throws — the maintenance hazard the review names.
- **ROCm** (`src/rocm/device.cpp:80-114`): mirrors CUDA byte-for-byte modulo `hipSetDevice` for `activate`, `std::runtime_error("ROCm tensor storage is not 32-byte aligned")` text, and the absence of an explicit HIP `DeviceGuard`.
- **SYCL** (`src/sycl/device.cpp:107-156`): adds a context-compatibility check via `sycl::get_pointer_type(address_, context)` after the alignment check; the post-alignment USM rejection wraps the free in `std::exchange(address_, nullptr)` and a nested try/catch around `allocator_.free(rejected)` (since SYCL's runtime can throw inside the call). Destructor (`src/sycl/device.cpp:146-156`) does `if (address_ == nullptr) return;` then `try { allocator_.free(address_); } catch (...) {}`. The misalignment throw uses `std::runtime_error("SYCL tensor storage is not 32-byte aligned")`.

`Allocator::alloc(size_t)` in `include/iom/alloc.hpp:11` declares no alignment obligation. The 32-byte requirement is established only by the parent spec at `docs/changes/0001-tensor-view/spec.md:248-252,447` and by the four duplicated constants in the per-backend TUs. A caller-supplied allocator configured with a smaller alignment (e.g. `LinearAllocator(..., alignment = 16)`) silently violates the contract because no per-backend test exercises a non-32-byte allocator for every standard-layout backend.

TTNN (`src/ttnn/device.cpp:177-239`) is intentionally outside this contract: its native storage is created through `ttnn::create_device_tensor`, never through `iom::Allocator`, and its storage byte count differs from `TensorSpec::tiled_storage_nbytes()`. The TTNN constructor must not call the shared helper.

## Scope

The four standard-layout tensor constructors and destructors (CPU, CUDA, ROCm, SYCL) each delegate their allocate/validate/single-free contract to one backend-neutral helper pair in `include/iom/detail/aligned_storage.hpp`. The helper enforces the 32-byte base-address alignment once. Each backend supplies exactly one pre-allocation / pre-release callback (its device-context activation) and exactly one `std::runtime_error` message preserving its current `what()` text byte-for-byte. The `kStorageAlignment` constant collapses to one declaration in the shared header. The `Allocator::alloc` interface gains a documented alignment obligation. TTNN's native-storage path is untouched.

## Implementation references

- **Create:** `include/iom/detail/aligned_storage.hpp` — a header-only backend-neutral helper pair exposing `iom::detail::allocate_aligned_storage` and `iom::detail::release_aligned_storage`. The header includes the alignment constant `iom::detail::kStorageAlignment = 32` exactly once. The helper takes a caller-supplied exact message (`std::string_view misalignment_message`) so each backend preserves its current `what()` text byte-for-byte. No new translation unit is added.
- **Modify:** `src/cpu/device.cpp:23` — remove the local `kStorageAlignment` constant; replace the inline `alloc/null-check/align-check/free-and-throw` block in `CpuTensor` (`src/cpu/device.cpp:162-181`) with one call to `iom::detail::allocate_aligned_storage(allocator_, view().spec().tiled_storage_nbytes(), "CPU tensor storage is not 32-byte aligned")` (CPU has no `activate()` — the helper's pre-allocation callback parameter defaults to a no-op). Replace the unconditional `allocator_.free(address_)` in the destructor (`src/cpu/device.cpp:183-185`) with `iom::detail::release_aligned_storage(allocator_, address_)` (CPU has no `activate()`).
- **Modify:** `src/cuda/device.cpp:19,119-154` — remove the local `kStorageAlignment` constant; replace the inline allocate/check/free-and-throw block in `CudaTensor::CudaTensor` with `address_ = iom::detail::allocate_aligned_storage(allocator_, view().spec().tiled_storage_nbytes(), [this] { device_.activate(); }, "CUDA tensor storage is not 32-byte aligned");`. Replace the destroyer's `try { device_.activate(); } catch (...) {}; try { allocator_.free(address_); } catch (...) {}; address_ = nullptr;` shape with `iom::detail::release_aligned_storage(allocator_, address_, [this] { device_.activate(); });`.
- **Modify:** `src/rocm/device.cpp:18,80-114` — same shape as CUDA, with `device_.activate()` calling `hipSetDevice` rather than `cuCtxSetCurrent`, and the message `"ROCm tensor storage is not 32-byte aligned"`.
- **Modify:** `src/sycl/device.cpp:27,107-156` — remove the local `kStorageAlignment` constant; replace the inline allocate/check/free-and-throw block in `SyclTensor::SyclTensor` with `address_ = iom::detail::allocate_aligned_storage(allocator_, view().spec().tiled_storage_nbytes(), [] {}, "SYCL tensor storage is not 32-byte aligned");` (SYCL pre-allocate does not need a device-activation hook; the USM compatibility check below is the per-backend concern). The misalignment free path is now inside `allocate_aligned_storage`; the SYCL TU no longer carries its own misalignment-rejection literal. After the helper returns, the existing `sycl::get_pointer_type(address_, device_.context())` check stays in the SYCL TU as a backend-specific post-allocation contract verification; on rejection the SYCL constructor must capture the member into a local before nulling — `void* rejected = std::exchange(address_, nullptr);` — and pass that local by reference to the helper as `iom::detail::release_aligned_storage(allocator_, rejected)`. Passing `address_` directly after `std::exchange` would feed the helper a `nullptr` reference (its step-1 early-return) and leak the USM-incompatible allocation. After `release_aligned_storage` returns, `rejected` is `nullptr` and `address_` is already `nullptr` from the earlier `std::exchange`, so the constructor rethrows with no observable state change. The destructor (`src/sycl/device.cpp:146-156`) collapses to one `iom::detail::release_aligned_storage(allocator_, address_)` call (no local needed: the destructor does not have a prior `std::exchange` to undo).
- **Modify:** `include/iom/alloc.hpp:11` — add a single-line comment above `virtual void* alloc(std::size_t sz) = 0;` documenting the alignment obligation: "Every backend's `create_tensor` requires the returned address to satisfy the engine's 32-byte base-address alignment. Backends must inject an allocator configured to honor this requirement (`LinearAllocator(..., alignment = 32)` by default). The contract is enforced by `iom::detail::allocate_aligned_storage`."
- **Read:** `include/iom/alloc.hpp:36-110` — `LinearAllocator`, `ListAllocator`, `FixedSizeAllocator` already default to `alignment = 32` (`include/iom/alloc.hpp:38`); the alignment parameter remains a caller choice. The fix documents, but does not change, that choice.
- **Read:** `test/cpu/test_cpu.cpp:439-462` — `"CPU tensors reject null and misaligned allocations exactly once"`. The test already exercises the CPU invariant (`std::bad_alloc` on null; `std::runtime_error` on misaligned; `frees == allocations` exactly once). No test file is modified.
- **Read:** `test/cuda/test_cuda_smoke.cpp:176-186` — `"CUDA tensor rejects misaligned allocator storage exactly once"`. The test already exercises the CUDA invariant (`std::runtime_error`; `allocator.frees == 1`). No test file is modified.
- **Tests:** `test/test_alloc.cpp:90,202,314` (allocator-exhaustion paths), `test/cpu/test_cpu_conformance.cpp`, `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, `test/sycl/test_sycl_conformance.cpp` — every existing per-backend allocation/copy/transfer scenario continues to drive `Device::create_tensor` and observe the documented exception categories. No test file is modified.

## Requirements

- One header `include/iom/detail/aligned_storage.hpp` declares `inline constexpr std::size_t kStorageAlignment = 32;` exactly once. The header is included by `src/cpu/device.cpp`, `src/cuda/device.cpp`, `src/rocm/device.cpp`, `src/sycl/device.cpp`. No other TU declares `kStorageAlignment`.
- The helper signature for allocation is `void* iom::detail::allocate_aligned_storage(Allocator& allocator, std::size_t nbytes, std::invocable<> auto pre_allocate, std::string_view misalignment_message);`. The helper:
  1. Invokes `pre_allocate()` exactly once before calling `allocator.alloc(nbytes)`.
  2. Calls `allocator.alloc(nbytes)` exactly once. If the return value is `nullptr`, throws `std::bad_alloc` without invoking `allocator.free`.
  3. Checks `reinterpret_cast<std::uintptr_t>(address) % kStorageAlignment == 0`. If not, calls `allocator.free(address)` exactly once and throws `std::runtime_error(std::string(misalignment_message))`. The conversion through `std::string` is required: `std::runtime_error` has no portable constructor from `std::string_view` in C++20 (only from `const std::string&` and `const char*`); an explicit `std::string` materialization is the only spelling that compiles on every supported toolchain. The resulting `what()` carries the caller-supplied message verbatim.
  4. On success, returns the address.
- The helper signature for release is `void iom::detail::release_aligned_storage(Allocator& allocator, void*& address, std::invocable<> auto pre_release = [] {});`. The helper:
  1. If `address == nullptr`, returns immediately (no-op).
  2. Invokes `pre_release()` inside a `try { ... } catch (...) {}` block (mirrors the existing CUDA/ROCm/SYCL destructor try/catch around `activate`).
  3. Invokes `allocator.free(address)` inside a `try { ... } catch (...) {}` block (mirrors the existing CUDA/ROCm/SYCL destructor try/catch around `free`).
  4. Sets the caller's reference to `nullptr`.
- The helper functions are `inline` in the header. No new translation unit is introduced. The header is header-only.
- Each backend's tensor constructor passes its exact current `what()` text to the helper:
  - CPU: `"CPU tensor storage is not 32-byte aligned"`.
  - CUDA: `"CUDA tensor storage is not 32-byte aligned"`.
  - ROCm: `"ROCm tensor storage is not 32-byte aligned"`.
  - SYCL: `"SYCL tensor storage is not 32-byte aligned"`.
  These strings are identical to the four strings currently in `src/cpu/device.cpp:177-179`, `src/cuda/device.cpp:136-138`, `src/rocm/device.cpp:96-98`, and `src/sycl/device.cpp:125-127`. No textual regression is introduced.
- The CUDA and ROCm constructors pass `[this] { device_.activate(); }` as the pre-allocation callback (CUDA invokes `cuCtxSetCurrent`; ROCm invokes `hipSetDevice`). The CPU and SYCL constructors pass `[] {}` as the pre-allocation callback (CPU has no device activation; SYCL's USM check is post-allocation).
- SYCL's existing post-allocation USM compatibility check at `src/sycl/device.cpp:129-143` is preserved verbatim and continues to throw `std::runtime_error("SYCL tensor storage is incompatible with the owned context")`. On rejection, the SYCL constructor must capture the rejected member into a local (`void* rejected = std::exchange(address_, nullptr)`) and pass that local by reference to `iom::detail::release_aligned_storage(allocator_, rejected)` so the helper sees the actual allocation address; passing `address_` directly after `std::exchange` would feed the helper `nullptr` and skip the free. The `rejected` local is `nullptr` after the call; `address_` was already nulled by the `std::exchange`; the constructor rethrows the USM exception.
- Each backend's tensor destructor passes `[this] { device_.activate(); }` as the pre-release callback for CUDA and ROCm; CPU and SYCL use the default no-op. The exception-suppressing `try { ... } catch (...) {}` shape lives once inside the helper.
- The exception categories produced through the public surface are unchanged: `std::bad_alloc` on exhausted allocator; `std::runtime_error` on misaligned allocator; `std::runtime_error` on SYCL USM incompatibility. The `what()` text is byte-identical to the pre-change text on every backend.
- The alignment obligation is documented on `iom::Allocator::alloc` in `include/iom/alloc.hpp:11` with one comment block. The comment is the single source of truth that backends rely on 32-byte alignment; no caller-visible interface change is made.
- The single-free invariant is preserved: the helper calls `allocator.free` exactly once on the misaligned address, and `release_aligned_storage` calls `allocator.free` exactly once on a non-null address. The CUDA / ROCm "free-through-member-then-null" hazard the review names (a refactor away from a double-free if `free` ever throws) is now structurally impossible because `release_aligned_storage` nulls the reference inside the helper, after the free's try/catch swallows any throw.
- TTNN (`src/ttnn/device.cpp:177-239`) is unmodified. The TTNN constructor does not call `iom::detail::allocate_aligned_storage` and does not call `iom::detail::release_aligned_storage`. `TtnnTensor` does not gain an `Allocator&` member, does not include the helper header, and does not reference `kStorageAlignment`.
- `include/iom/alloc.hpp` is otherwise unchanged. The `LinearAllocator`, `ListAllocator`, `FixedSizeAllocator` declarations and their default `alignment = 32` parameter are preserved verbatim. The `SingleBufferAllocatorBase::owns()` predicate the review's AR-007 sibling removes is not touched by this task.
- The header `include/iom/detail/aligned_storage.hpp` lives under `include/iom/detail/`. The header includes `<concepts>` (for `std::invocable`), `<cstddef>`, `<cstdint>`, `<new>`, `<stdexcept>`, `<string_view>`, and the project's `include/iom/alloc.hpp`. No CUDA, HIP, SYCL, TTNN, or other backend header is included.
- The CMake target layout is unchanged: `iom_commons` already lists `include/iom/alloc.hpp` at `CMakeLists.txt:75`; adding `include/iom/detail/aligned_storage.hpp` is a header-only change. No new translation unit is compiled.

## Non-goals

- TTNN native storage (`src/ttnn/device.cpp:177-239`). TTNN retains its `ttnn::create_device_tensor` path; the helper does not participate in TTNN tensor construction or destruction. `TtnnTensor` does not gain an `Allocator&` member, does not include the helper header, and does not call `kStorageAlignment`.
- Tensor destructor lifetime / queue synchronization. AR-006 does not touch `~Tensor`, the per-backend queue drain, the `ST-003` "wait all tokens before destroying operands" requirement, the `ST-001` event/stream destroy fence, or any synchronization around tensor destruction. The helper's try/catch around `allocator.free` mirrors the existing per-backend destructor try/catch; it is not a synchronization point.
- Changing the per-backend `region_from_host` / `region_to_host` implementations, the per-backend queue classes, or the `DeviceOps` staged worker. AR-006 owns construction and destruction of the allocation only.
- The CUDA `DriverCalls` table race (`AR-005`) and the driver-call seam unification. AR-006 does not touch `src/cuda/driver.hpp`, `cuda_detail::driver_calls`, or the `PrimaryCtxGuard` RAII.
- `SingleBufferAllocatorBase::owns()` removal (`AR-007`). The review rejects this method as dead; AR-006 documents the alignment obligation on `Allocator::alloc` and leaves `owns()` alone for `AR-007` to delete.
- New public API surface. The helper is in `iom::detail::`; it is not added to `iom::` or to any user-facing header. The only user-facing change is the comment on `Allocator::alloc`.
- An alignment-aware allocator (allocator that returns `std::align_val_t`). The 32-byte requirement is documented on `Allocator::alloc` and enforced by the helper; an allocator-side alignment abstraction is a separate workstream.
- Performance work on the allocate/free path. The helper performs the same number of function calls as today's inline bodies. No microbenchmark, no GPU timing, no allocator-side caching.
- Adding new test cases for ROCm or SYCL misalignment rejection. The review's invariant is observable through the existing CPU test at `test/cpu/test_cpu.cpp:439-462` and the existing CUDA test at `test/cuda/test_cuda_smoke.cpp:176-186`; the helper's behavior is exercised through the same allocation paths on every backend. No test file is modified by this task.

## Acceptance criteria

- [ ] `grep -rn "kStorageAlignment" include/iom src/` matches the declaration `inline constexpr std::size_t kStorageAlignment = 32;` exactly once in `include/iom/detail/aligned_storage.hpp`. No per-backend TU declares or defines `kStorageAlignment`.
- [ ] `grep -rn "tensor storage is not 32-byte aligned" src/cuda/device.cpp src/cuda/copy.cu src/rocm/device.cpp src/rocm/copy.hip src/cpu/device.cpp src/sycl/device.cpp include/` returns four matches: the per-backend string literal in each backend's tensor constructor call site (`"CPU …"`, `"CUDA …"`, `"ROCm …"`, `"SYCL …"`), all byte-identical to the strings currently at `src/cpu/device.cpp:177-179`, `src/cuda/device.cpp:136-138`, `src/rocm/device.cpp:96-98`, `src/sycl/device.cpp:125-127`. No per-backend TU retains an inline `throw std::runtime_error("…tensor storage is not 32-byte aligned")` literal; the throw is owned by `iom::detail::allocate_aligned_storage`.
- [ ] `grep -rn "allocate_aligned_storage\|release_aligned_storage" src/` returns one definition site for each helper in the header and one call site per backend's tensor constructor (CPU, CUDA, ROCm, SYCL) and one call site per backend's tensor destructor. The four constructors and four destructors each call exactly once into the helper. The TTNN constructor (`src/ttnn/device.cpp:184-209`) and destructor (`src/ttnn/device.cpp:211-214`) carry zero matches.
- [ ] `wc -l src/cpu/device.cpp src/cuda/device.cpp src/rocm/device.cpp src/sycl/device.cpp` shows each file shrinking by approximately the size of the inline allocate/validate/free-and-throw block (about 18 lines for CPU, 22 lines for CUDA, 22 lines for ROCm, 35 lines for SYCL). The CUDA and ROCm files additionally lose the duplicated `try { activate(); } catch (...) {}; try { free(); } catch (...) {}; address_ = nullptr;` destructor shape.
- [ ] `include/iom/alloc.hpp:11` carries a one-line comment block documenting the 32-byte alignment obligation; the comment cites `iom::detail::allocate_aligned_storage` as the enforcer. No other file re-documents the obligation.
- [ ] The `RecordingAllocator` test at `test/cpu/test_cpu.cpp:439-462` compiles and passes unchanged: `device->create_tensor(spec)` with `null_next == true` throws `std::bad_alloc`; with `misalign_next == true` throws `std::runtime_error`; the second path records exactly three events (`alloc` null, `alloc` misaligned, `free` of the misaligned address); `live_empty()` is true after both paths.
- [ ] The `MisalignedAllocator` test at `test/cuda/test_cuda_smoke.cpp:176-186` compiles and passes unchanged: `device->create_tensor(spec)` throws `std::runtime_error` with `what()` exactly equal to `"CUDA tensor storage is not 32-byte aligned"`; `allocator.allocations == 1`; `allocator.frees == 1`. The exception's `what()` is byte-identical to the pre-change text.
- [ ] The CPU allocation-failure test at `test/test_alloc.cpp:90,202,314` continues to assert `std::bad_alloc` on exhausted `LinearAllocator`/`ListAllocator`/`FixedSizeAllocator` (untouched by this task — the helper preserves the `bad_alloc` throw on a null `alloc` return). The CC-004-aligned test (`docs/changes/0001-tensor-view/25-CC-004-align-allocation-failures/spec.md`) asserting `std::bad_alloc` from `create_tensor` through an exhausted `LinearAllocator` continues to pass.
- [ ] SYCL's post-allocation USM compatibility check at `src/sycl/device.cpp:129-143` continues to throw `std::runtime_error("SYCL tensor storage is incompatible with the owned context")`. The rejection path captures `address_` into a local before nulling (`void* rejected = std::exchange(address_, nullptr)`), passes `rejected` by reference to `iom::detail::release_aligned_storage(allocator_, rejected)`, and rethrows. The misalignment-rejection throw is now inside `allocate_aligned_storage` (header), not in `src/sycl/device.cpp`. `grep -nE "std::exchange\(address_, nullptr\)" src/sycl/device.cpp` returns exactly one match (the USM path), followed by exactly one `release_aligned_storage(allocator_, rejected)` call site — never `release_aligned_storage(allocator_, address_)`. The address is freed exactly once through the helper's `allocator.free` and the constructor leaves `address_ == nullptr` after rethrow.
- [ ] The full local CPU conformance suite (`iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, `iom_tests`) passes on the local host (`Ryzen AI 9 HX 370`, `g++ 15.2.0 -O2`) with no test file modifications. The CPU misallocation/exhaustion/alignment scenarios at `test/test_alloc.cpp` and the CPU conformance driver at `test/cpu/test_cpu_conformance.cpp` continue to drive `CpuDevice::create_tensor` through the centralized helper and observe identical exception categories.
- [ ] The CUDA, ROCm, and SYCL per-backend conformance suites (`iom_<backend>_conformance_tests`) pass on hardware (`flock /tmp/agent-gpu0.lock` per the repository's remote-development procedure) without test file modifications. The `MisalignedAllocator` smoke test (CUDA), the per-backend `create_tensor` exhaustion test, and the per-backend alignment scenarios observe identical exception categories.
- [ ] `include/iom/detail/aligned_storage.hpp` compiles under `iom_commons` (header-only). `git grep -n "aligned_storage" src/cpu src/cuda src/rocm src/sycl src/ttnn` returns matches only in the four standard-layout backend TUs and the header itself. TTNN (`src/ttnn/`) carries zero matches.

## Verification

Run through the repository's remote-development procedure (`.agents/skills/remote-development`). Use unique task ids such as `ar006-cpu`, `ar006-cuda`, `ar006-rocm`, `ar006-sycl`.

Local CPU-only baseline (always runnable):

1. Sync and rebuild:
   ```bash
   cd /home/rlew/iom/src/iom && \
   cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF \
     && cmake --build build -j --target iom_cpu_tests iom_cpu_conformance_tests iom_tests
   ```
2. Run the full local suite:
   ```bash
   ctest --test-dir build --output-on-failure
   ```
3. Run the existing CPU misalignment and exhaustion scenarios (the only CPU-runnable invariant check this task owns):
   ```bash
   ./build/test/iom_cpu_tests -tc="*rejects null and misaligned allocations exactly once*"
   ```
   Expected: the test passes; `null_next == true` produces `std::bad_alloc`; `misalign_next == true` produces `std::runtime_error` with `what()` exactly equal to `"CPU tensor storage is not 32-byte aligned"`; the recorded events confirm one `alloc` + one `free` for the misaligned path and `live_empty()` is true after both paths.
4. Run the existing CPU allocation-exhaustion scenarios:
   ```bash
   ./build/test/iom_cpu_tests -tc="*allocator failures*" -tc="*exhausted*"
   ```
   Expected: `std::bad_alloc` flows through the helper; the helper preserves the pre-change exception category.
5. Run the full CPU conformance matrix:
   ```bash
   ./build/test/iom_cpu_conformance_tests
   ```
   Expected: same assertion count (or higher) as before this change; no skipped cases; the conformance matrix at `test/backend/backend_conformance_copy_storage.hpp` and `test/backend/backend_conformance_other.hpp` continues to drive `CpuDevice::create_tensor` and observe identical exception categories.

Per-backend hardware verification (one host per backend; `flock` per the procedure):

1. CUDA host:
   ```bash
   .agents/skills/remote-development/scripts/remote-sync <cuda-host> ar006-cuda
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar006-cuda \
     'cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DCUDA_PATH=/usr/local/cuda \
        && cmake --build build -j --target iom_cuda_conformance_tests iom_cuda_smoke_tests'
   .agents/skills/remote-development/scripts/remote-exec <cuda-host> ar006-cuda \
     'flock /tmp/agent-gpu0.lock ctest --test-dir build --output-on-failure -R "iom_cuda_(conformance|smoke)_tests"'
   ```
   Expected: all assertions pass; `MisalignedAllocator` test at `test/cuda/test_cuda_smoke.cpp:176-186` throws `std::runtime_error` with `what()` exactly equal to `"CUDA tensor storage is not 32-byte aligned"`; `allocator.allocations == 1`; `allocator.frees == 1`.

2. ROCm host:
   ```bash
   .agents/skills/remote-development/scripts/remote-sync <rocm-host> ar006-rocm
   .agents/skills/remote-development/scripts/remote-exec <rocm-host> ar006-rocm \
     'cmake -S . -B build -DBUILD_TESTING=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_PATH=/opt/rocm \
        && cmake --build build -j --target iom_rocm_conformance_tests iom_rocm_smoke_tests'
   .agents/skills/remote-development/scripts/remote-exec <rocm-host> ar006-rocm \
     'flock /tmp/agent-gpu1.lock ctest --test-dir build --output-on-failure -R "iom_rocm_(conformance|smoke)_tests"'
   ```
   Expected: all assertions pass; the per-backend conformance suite continues to drive `RocmDevice::create_tensor` and observe identical exception categories.

3. SYCL host:
   ```bash
   .agents/skills/remote-development/scripts/remote-sync <sycl-host> ar006-sycl
   .agents/skills/remote-development/scripts/remote-exec <sycl-host> ar006-sycl \
     'cmake -S . -B build -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -D<OTHER>_ENABLED=OFF \
        && cmake --build build -j --target iom_sycl_conformance_tests iom_sycl_smoke_tests'
   .agents/skills/remote-development/scripts/remote-exec <sycl-host> ar006-sycl \
     'flock /tmp/agent-gpu2.lock ctest --test-dir build --output-on-failure -R "iom_sycl_(conformance|smoke)_tests"'
   ```
   Expected: SYCL's USM compatibility rejection at `src/sycl/device.cpp:129-143` continues to throw `std::runtime_error("SYCL tensor storage is incompatible with the owned context")` and free the address exactly once through the centralized helper; misalignment rejection through the helper preserves the `std::runtime_error` with `what()` exactly equal to `"SYCL tensor storage is not 32-byte aligned"`.

4. After verification, `remote-clean <host> ar006-<backend>` removes the remote mirror.

Static grep verification (run on the local repository):

```bash
grep -rn "kStorageAlignment" include/iom src/        # exactly one match in include/iom/detail/aligned_storage.hpp
grep -rn "tensor storage is not 32-byte aligned" src/ include/iom   # four matches, one per backend's ctor call site; zero in src/ttnn/
grep -rn "allocate_aligned_storage\|release_aligned_storage" src/   # one definition per helper in header; one call per backend's ctor and dtor; zero in src/ttnn/
grep -rn "aligned_storage" src/ttnn/                # zero matches
```

The expected observation: the four standard-layout backend TUs each delegate the allocate/null-check/alignment-check/free-and-throw contract to one helper call; the helper enforces the 32-byte alignment invariant once and emits the per-backend message verbatim; the `Allocator::alloc` interface documents the obligation; TTNN remains untouched; the existing CPU and CUDA misalignment tests (`test/cpu/test_cpu.cpp:439-462`, `test/cuda/test_cuda_smoke.cpp:176-186`) continue to pass byte-identical on the pre-change exception text. The local CPU suite plus each backend's hardware conformance suite pass with no test file modifications.

CPU-only evidence is sufficient for the always-built local baseline; CUDA / ROCm / SYCL hardware confirmation is required to verify the per-backend `activate()` callbacks integrate with the existing driver-API and runtime context management. No project-wide build / lint / test suite runs in this task; sibling tasks edit concurrently and would block on a mid-flight validation. Project-wide validation is the main agent's responsibility after all writers land.
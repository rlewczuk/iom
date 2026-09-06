# Generate the three mechanical `Fence` capture-storage ops from one shared template instead of three byte-identical per-backend copies

**Order:** 80
**Priority:** P2 — required test-free structural finishing work; deletes the duplicated launder/placement-new/destroy boilerplate (SYCL's copy among them) into one shared template beside `detail::Fence`. Low severity, gates no other work.
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` (Area 3 of `cpp-inference-code-review`) — whole-codebase working tree at `a9d8d0ccc3fb0d03e4746082481669dc969e7111`
**Finding:** `AR-011`
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** verified, confidence 85
**Review scope:** whole-codebase
**Backend scope:** multi-backend
**Location:** `src/sycl/copy.cpp:385-415` (`sycl_fence_copy_construct`, `sycl_fence_move_construct`, `sycl_fence_storage_destroy`); identical copies at `src/cuda/copy.cu:61-93` (`cuda_fence_*`) and `src/rocm/copy.hip:60-92` (`hip_fence_*`)

## Outcome

The three purely mechanical `detail::Fence` storage operations — copy-construct a capture into `Fence::storage`, move-construct it and `destroy_at` the source, and `destroy_at` the storage — are produced by one shared template `iom::detail::FenceCaptureOps<Capture>` defined beside `Fence` in `include/iom/detail/outstanding_work_registry.hpp`. Each non-trivially-copyable backend capture (`cuda_detail::EventLeaseWithFailure`, `rocm_detail::EventLeaseWithFailure`, `sycl_detail::SyclFenceCapture`) instantiates the template and wires its three function pointers from it. The three byte-identical `*_fence_copy_construct`/`*_fence_move_construct`/`*_fence_storage_destroy` blocks (SYCL's included) are deleted. Each backend keeps its own `invoke` and `build_*_fence`, which encode the real per-backend semantics (`EventRingState::invoke_result(slot)` on CUDA/ROCm vs `SyclFenceState::result()` on SYCL). Fence behavior — copy/move/destroy noexcept guarantees, the `static_assert`s on capture size/alignment, the invoke result — is unchanged.

## Current problem

- **Invariant:** one mechanical decision has one source of truth. "How to copy-, move-, and destroy-construct a capture object inside `detail::Fence::storage`" is a backend-neutral fact determined entirely by the capture type; `Fence` (`include/iom/detail/outstanding_work_registry.hpp:46-133`) already owns the type-erased storage buffer and the four function-pointer slots it is filled with.
- **Failing path:** each of CUDA, ROCm, and SYCL hand-writes the same three launder/`reinterpret_cast`/placement-new/`destroy_at` functions, differing only in the capture type name. After renaming `cuda`→`hip`/`X`, the CUDA and ROCm blocks (`cuda_fence_copy_construct`..`build_cuda_fence`) are byte-identical (MD5 `43854b5b…` for both). SYCL's three (`src/sycl/copy.cpp:385-412`) are the same operations over `SyclFenceCapture`: `copy_construct` placement-news `{*launder<const Capture>(src.storage)}`, `move_construct` placement-news `{std::move(*launder<Capture>(src->storage))}` then `destroy_at(launder<Capture>(src->storage))`, `storage_destroy` calls `destroy_at(launder<Capture>(fence->storage))`. The only intra-block variation is CUDA/ROCm writing the copy as an explicit field list `{capture.state, capture.slot_index, capture.retained_failure}` where SYCL writes the whole-object `{*launder(...)}` — semantically identical for these aggregate captures.
- **Evidence:** the MD5 identity above; reading of all three blocks; `grep -rn "FenceCaptureOps\|make_fence_capture" include src` returns zero matches (no shared owner exists yet). TTNN already proves the mechanical ops are capture-determined: its `TtnnFenceCapture` is `static_assert(is_trivially_copyable)` and `build_ttnn_fence` (`src/ttnn/device.cpp:352-357`) sets only `invoke`, relying on `Fence`'s `memcpy` fallback for copy/move and leaving `destroy` null — exactly the path the template's trivially-copyable specialization would also select.
- **Impact:** the storage-buffer ABI discipline (launder before reinterpret_cast, destroy the moved-from source, keep every op `noexcept`) is restated three times. A correction to that discipline — e.g. a stricter `static_assert`, a fix to the move-then-destroy order — must be applied in three files; a backend that misses it silently keeps the weaker or wrong form, and the error surfaces only as fence corruption under copy/move of a live `Fence`.

## Scope

- Add `template <typename Capture> struct FenceCaptureOps` to `include/iom/detail/outstanding_work_registry.hpp` next to `Fence`, exposing three `static` member functions with the exact `Fence` pointer signatures (`void (*)(Fence*, const Fence&) noexcept`, `void (*)(Fence*, Fence*) noexcept`, `void (*)(Fence*) noexcept`) and a `static_assert` that `Capture` is nothrow copy/move constructible and nothrow destructible.
- Delete `cuda_fence_copy_construct`/`cuda_fence_move_construct`/`cuda_fence_storage_destroy` (`src/cuda/copy.cu:61-93`), the `hip_*` trio (`src/rocm/copy.hip:60-92`), and the `sycl_*` trio (`src/sycl/copy.cpp:385-412`).
- In each backend's `build_*_fence`, assign `fence.copy_construct = &detail::FenceCaptureOps<Capture>::copy_construct;` etc., keeping the existing `invoke` and the per-backend capture construction.
- Affected backends: cuda, rocm, sycl (production code). TTNN is unchanged — it sets no copy/move/destroy ops and continues to rely on the `memcpy` fallback. CPU has no fence capture.

## Implementation references

- **Modify:** `include/iom/detail/outstanding_work_registry.hpp` — add, immediately after the `Fence` definition and its `static_assert`s (`:135-140`), the template:
  ```cpp
  template <typename Capture>
  struct FenceCaptureOps {
      static_assert(std::is_nothrow_copy_constructible_v<Capture>
              && std::is_nothrow_move_constructible_v<Capture>
              && std::is_nothrow_destructible_v<Capture>);
      static_assert(sizeof(Capture) <= kFenceStorageBytes);
      static_assert(alignof(Capture) <= kFenceStorageAlign);

      static void copy_construct(Fence* dst, const Fence& src) noexcept {
          ::new (dst->storage) Capture{
                  *std::launder(reinterpret_cast<const Capture*>(src.storage))};
      }
      static void move_construct(Fence* dst, Fence* src) noexcept {
          ::new (dst->storage) Capture{
                  std::move(*std::launder(reinterpret_cast<Capture*>(src->storage)))};
          std::destroy_at(std::launder(reinterpret_cast<Capture*>(src->storage)));
      }
      static void destroy(Fence* f) noexcept {
          std::destroy_at(std::launder(reinterpret_cast<Capture*>(f->storage)));
      }
  };
  ```
  (`<new>`, `<memory>`, `<type_traits>` are already included by the header.)
- **Modify:** `src/cuda/copy.cu` — delete `:61-93`; in `build_cuda_fence` (`:111-122`) replace the three pointer assignments with `&detail::FenceCaptureOps<EventLeaseWithFailure>::{copy_construct,move_construct,destroy}`. Keep `cuda_fence_invoke` and the `EventLeaseWithFailure` capture. The redundant per-backend `static_assert(sizeof/alignof(EventLeaseWithFailure) …)` at `:109-110` and `src/cuda/copy.hpp:230-231` may stay or move into the template instantiation; the template's own asserts cover them.
- **Modify:** `src/rocm/copy.hip` — the exact ROCm mirror: delete `:60-92`, rewire `build_hip_fence` (`:110-121`).
- **Modify:** `src/sycl/copy.cpp` — delete `:385-412` (`sycl_fence_copy_construct`/`move_construct`/`storage_destroy` and their `static_assert(noexcept(...))` guards); in `build_sycl_fence` (`:417-426`) assign the three pointers from `detail::FenceCaptureOps<SyclFenceCapture>`. Keep `sycl_fence_invoke` (`:371-380`) and the `SyclFenceCapture`/`SyclFenceState` design (55-ST-001/60-AR-001 territory).
- **Read:** `include/iom/detail/outstanding_work_registry.hpp:46-140` — the `Fence` storage buffer, the four function-pointer slots, the copy/move/assign paths that call `copy_construct`/`move_construct`/`destroy`, and the `memcpy` fallback used when those pointers are null (TTNN's path).
- **Read:** `src/ttnn/device.cpp:323-357` — the trivially-copyable capture that sets only `invoke`; confirms the template must not be forced on captures that rely on the `memcpy` fallback.
- **Tests:** the fence copy/move/destroy paths are exercised indirectly by every queued-copy conformance and lifetime case (`run_async_copy_conformance`, `run_lifetime_conformance`, the tensor/queue-destruction-fences cases) on CUDA, ROCm, and SYCL; no new test code.

## Requirements

- `include/iom/detail/outstanding_work_registry.hpp` contains exactly one definition of the three mechanical capture-storage operations, as `FenceCaptureOps<Capture>`, with signatures matching `Fence::copy_construct`/`move_construct`/`destroy` and `noexcept` guaranteed by the template's `static_assert` on the capture type.
- CUDA, ROCm, and SYCL each retain exactly one `*_fence_invoke` and one `build_*_fence`; the three mechanical functions are gone from all three backends and replaced by `FenceCaptureOps<Capture>` instantiations.
- Fence copy, move, move-assign, copy-assign, and destruction produce identical observable behavior: a copied `Fence` shares the capture's `shared_ptr`/`exception_ptr` state (refcount incremented), a moved `Fence` transfers it and destroys the source capture exactly once, and a destroyed `Fence` runs the capture destructor exactly once. The move-then-`destroy_at` ordering and the launder-before-reinterpret_cast discipline are preserved.
- The per-backend capture-size/alignment `static_assert`s remain enforced (by the template's asserts at each instantiation, or by the retained backend-local asserts — at least one site asserts each capture fits `kFenceStorageBytes`/`kFenceStorageAlign`).
- TTNN is untouched: `build_ttnn_fence` still sets only `invoke` and relies on the `memcpy` fallback for its trivially-copyable `TtnnFenceCapture`. The template is not instantiated for TTNN.
- No change to `Fence`'s public shape, the registry, the `SubmissionFault` seams, or any queue/tensor protocol.

## Non-goals

- Unifying `*_fence_invoke` or `build_*_fence` across backends — these encode the real semantic difference (`EventRingState::invoke_result(slot_index)` on CUDA/ROCm, `SyclFenceState::result()` on SYCL, `finish_native(device)` on TTNN) and stay per-backend.
- The `EventLeaseWithFailure` / `SyclFenceCapture` / `TtnnFenceCapture` capture types themselves and the `SyclFenceState`/`EventRingState` designs — owned by 55-ST-001/56-ST-002/60-AR-001; untouched.
- The CUDA/ROCm `EventRingState` event-fencing machinery and the SYCL `SyclFenceState` wait/cache protocol.
- Forcing the template onto TTNN's trivially-copyable capture (which would add a no-op `destroy` where the `memcpy` fallback currently suffices).
- Any numerical, performance, or queue-completion behavior change.

## Acceptance criteria

- [ ] `grep -rn "_fence_copy_construct\|_fence_move_construct\|_fence_storage_destroy" src/` returns zero matches; `grep -rn "FenceCaptureOps" include src` shows one definition in `outstanding_work_registry.hpp` and three instantiations (cuda, rocm, sycl).
- [ ] `grep -rn "_fence_invoke\|build_.*_fence" src/` shows exactly one `invoke` and one `build` per backend (cuda, rocm, sycl, ttnn) — four of each, unchanged in count.
- [ ] `src/ttnn/device.cpp` is unmodified (`git diff --stat src/ttnn` empty); `build_ttnn_fence` still sets only `invoke`.
- [ ] On CUDA, ROCm, and SYCL hardware, the queued-copy conformance and lifetime suites pass unchanged: `run_async_copy_conformance`, `run_lifetime_conformance`, the tensor-destruction-fences and queue-destruction-fences cases, and the repeated-`wait` deferred-failure cases (which copy/move live `Fence` objects through the registry).
- [ ] The `iom_tests` common suite passes unchanged in a CPU-only build (it exercises the registry/`Fence` copy/move/destroy contract directly via `test/test_iom.cpp`'s registry cases).

## Verification

- `cd /home/rlew/iom/src/iom && cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_tests && ctest --test-dir build --output-on-failure -R '^iom_tests$'` — local CPU baseline exercising `Fence` copy/move/destroy through the registry tests in `test/test_iom.cpp`.
- SYCL via `.agents/skills/remote-development`, `sycl` profile from `.remote-hosts.conf`: `remote-exec sycl <task-id> 'cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/sycl -j --target iom_sycl_conformance_tests iom_sycl_smoke_tests iom_tests && ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$|^iom_tests$"'`.
- CUDA and ROCm, each on its own host per `.remote-hosts.conf`: build and run `iom_cuda_conformance_tests` / `iom_rocm_conformance_tests` (and the matching smoke suites); expect identical results to the pre-change baseline, with the deferred-failure and destruction-fence cases specifically green (they copy/move live fences).
- Static audits: the `grep` criteria under Acceptance criteria, plus `git diff --stat src/ttnn src/cpu` empty.

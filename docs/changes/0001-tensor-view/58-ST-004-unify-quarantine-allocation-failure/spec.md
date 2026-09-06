# Align TTNN's quarantine-allocation failure policy with the CPU/CUDA/ROCm/SYCL leak convention, and amend §7 to record the uniform exception

**Order:** 58
**Priority:** P1 — required uniform failure-path behavior across the registry backends. The defect does not gate other remediation, but leaving two policies for one fault class forces every future destructor change (including AR-002's shared helper) and every future parent-spec re-reader to re-litigate the choice. Task54's forthcoming "exactly once" wording would otherwise introduce a second-order inconsistency that immediately contradicts this task's established leak policy.
**Blocked by:** `54-CC-004-align-allocator-lifetime-contract` — task54 rewrites parent-spec §7's free-exactly-once bullet (`docs/changes/0001-tensor-view/spec.md:449`) into the deferred-free wording this task appends its exception sentence to. If task58 ran first, its anchor bullet ("destruction calls `allocator.free(address)` exactly once: at tensor destruction, or deferred to device destruction …") would not exist — §7 would still read the pre-task54 `destruction calls allocator.free(address) exactly once;` — and the exception sentence would attach to the wrong text. The TTNN runtime change (delete `std::terminate`, `unique_ptr<std::vector<ttnn::Tensor>>` member, fault seam, CMake `IOM_ENABLE_TESTING`) has no dependency on task54 and could land independently, but the §7 amendment does; the single blocker keeps the parent-spec edit ordered.
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-004`
**Review severity:** low
**Review verification:** verified, confidence 95

## Outcome

Every registry backend's tensor destructor (CPU, CUDA, ROCm, SYCL, TTNN) applies one identical failure policy when the quarantine-add step fails: the destructor returns normally (no `std::terminate()`, no exception escape), the failed storage becomes intentionally unreachable (leaked) for the remainder of the process rather than being returned to its allocator or native runtime, and no quarantine entry is recorded for that tensor. TTNN joins the convention the other four backends already follow. `TtnnTensor::planes_` is restructured as a `std::unique_ptr<std::vector<ttnn::Tensor>>` so `planes_.release()` provides a nothrow leak primitive matching the sibling backends' raw-pointer leak; `TtnnNativeCleanupAction` needs no change because both `std::vector` and `std::function` move constructors are `noexcept`, so the existing constructor is already nothrow once the parameters are bound.

A focused TTNN fault-injection seam — public-to-tests `iom::ttnn_test::fail_next_quarantine_action_for_testing()` and `iom::ttnn_test::quarantine_action_fault_consumed_for_testing()`, both declared in the internal TTNN header `src/ttnn/registry_state.hpp` (always visible to the TTNN test TU; no `#ifdef IOM_ENABLE_TESTING` guard at the declaration), with the definitions and a single TU-private atomic flag living in `src/ttnn/device.cpp` only when `IOM_ENABLE_TESTING` is compiled into `iom_ttnn` — replaces today's non-discriminating global-`operator new` override pattern. The declarations mirror the established per-backend `_for_testing` style already in `src/cuda/staging_pool.hpp:60-62` (declared, ungated) / `src/cuda/staging_pool.cpp:25-29` (consumed, gated by `#ifdef IOM_ENABLE_TESTING`); they are internal to TTNN and absent from any public `include/iom/` header, use C++ linkage (no `extern "C"`), and are not part of any public API. `IOM_ENABLE_TESTING` is added to `iom_ttnn` *only* inside the existing `if(BUILD_TESTING)` block beside `CMakeLists.txt:271-275`, never in non-test builds.

The umbrella specification at `docs/changes/0001-tensor-view/spec.md` §7 is amended once, after task54's free-exactly-once bullet, to record the uniform exception in implementation-generic terms (no backend-specific mechanism, no `Allocator&` reference): "If constructing or recording the quarantine cleanup action itself fails inside a `noexcept` tensor destructor, the storage is deliberately leaked for the remainder of the process." TTNN's `unique_ptr`-wrapped member mechanism is recorded inside this task's spec, not in the umbrella contract, so task61's later vector `Quarantine` recording step (a different failure class preserved by the same durable contract) consumes the same wording without rewriting §7.

## Current failure

The invariant under review is one uniform failure policy for quarantine-add failure in `noexcept` tensor destructors — the 49-ST-003 contract "Destructors and quarantine drain are no-throw effective" (`docs/changes/0001-tensor-view/49-ST-003-fence-tensor-destruction/spec.md` requirement 11) applied uniformly to the same fault class on every registry backend.

`~TtnnTensor`'s `quarantine_native` lambda at `src/ttnn/device.cpp:244-272` allocates a heap `std::vector<ttnn::Tensor>` node with `new (std::nothrow)` at `:245-247`, then calls `std::terminate()` at `:249` when that allocation returns null. The sibling backends swallow the equivalent failure and intentionally leak: `src/cpu/device.cpp:500-510`, `src/cuda/device.cpp:161-174`, `src/rocm/device.cpp:141-154`, and (after 55-ST-001) `src/sycl/device.cpp` all wrap the quarantine `emplace` in `try { ... } catch (...) { /* intentional leak */ }` and follow with the operation that prevents direct reuse (a raw-pointer assignment to nullptr, or the SYCL `destroy-while-quarantined` equivalent). Identical fault conditions (host OOM at the quarantine-cleanup step) therefore produce `std::terminate()` on TTNN and a silent leak on the other four backends. Both policies are individually defensible, but their coexistence is contract drift: a maintainer porting AR-002's shared destructor helper must re-litigate which behavior to preserve.

The defect is currently unreachable in practice — it requires a genuine host OOM at a very narrow allocation point — so severity is low, not because the behavior is acceptable.

The sibling backends achieve "leak rather than free" because `address_` is a raw pointer whose value becomes unreachable when set to null. TTNN's `planes_` is a `std::vector<ttnn::Tensor>` member (`src/ttnn/device.cpp:343`) whose destructor destroys `ttnn::Tensor` elements and frees native TTNN buffers when the member destructor runs at `~TtnnTensor` completion. Moving the buffer out to a leaked heap node requires an allocation — the very resource that just failed — so the plain `std::vector` member cannot reproduce the sibling leak semantics. The fix must change the member representation to support nothrow release-without-delete.

A secondary defect, called out by integration: the obvious fault-injection mechanism (arming a global `operator new` override before `~TtnnTensor` runs) is non-discriminating. `~TtnnTensor` allocates a `std::vector<EntrySnapshot>` inside `snapshot_for` (`src/ttnn/device.cpp:278`), and any std library or TTNN runtime call along the destructor path also allocates. A global-new override can be consumed by any of those allocations and never reach the action-node allocation the test is trying to fault. The seam used by `src/cuda/staging_pool.cpp:25-29` is per-class and consumed at exactly one allocation point; this task adopts the same pattern.

A third defect, also called out by integration: a single function that both arms and observes the fault is undiagnosable when the test fails. A test that arms the seam, destroys a tensor, and observes an unexpected post-condition has no way to distinguish "arm was consumed elsewhere" from "arm was not consumed at all". The seam must expose an explicit observe predicate so the test can prove the intended site fired.

A fourth reconciliation: task54 (CC-004) lands the parent-spec §7 wording `"destruction calls `allocator.free(address)` exactly once: at tensor destruction, or deferred to device destruction for storage quarantined because a queued operation referencing it failed or was invalidated"`. ST-004 must amend §7 once more so both remaining conditions — quarantining the storage itself fails to *construct* the cleanup action (ST-004's reviewed terminate branch) or fails to *record* the constructed action into the quarantine (a later failure class task61 will encounter after reshaping the quarantine intrusive list to a vector whose growth can fail after construction) — are documented as deliberate leaks, not undocumented deviations from "exactly once". The durable wording must be implementation-generic (the umbrella contract is not the place for backend-specific mechanism) and release-generic (TTNN owns native storage through its runtime and has no `Allocator&`, so the wording must avoid `allocator.free(address)` even by paraphrase). ST-004 owns the durable wording; task61 consumes it without rewriting §7.

## Scope

- **Modify:** `src/ttnn/device.cpp` — `TtnnTensor::planes_` member declaration and every use site. The member becomes `std::unique_ptr<std::vector<ttnn::Tensor>>` so that `planes_.release()` (noexcept, no allocation) abandons the vector buffer without native destruction.
- **Modify:** `src/ttnn/device.cpp` — `~TtnnTensor`'s `quarantine_native` lambda. Delete the `std::terminate()` call; restructure so both throw sources on the quarantine path (the `std::function` finish temporary's construction and the `TtnnNativeCleanupAction` node's `operator new`) land inside one `try` block whose catch clause intentionally leaks the released planes pointer. The lambda also consults the TU-private fault consume helper immediately before constructing the action node.
- **Modify (test seam, declarations):** `src/ttnn/registry_state.hpp` — add to the existing `iom::ttnn_detail` namespace file the public-to-tests functions

  ```cpp
  namespace iom::ttnn_test {
      void fail_next_quarantine_action_for_testing() noexcept;
      bool quarantine_action_fault_consumed_for_testing() noexcept;
  }
  ```

  The declarations are **unguarded** (always visible to the TTNN test TU). The header is internal to TTNN (no public `include/iom/` symbol); the namespace uses C++ linkage (no `extern "C"`); the declarations are not part of any public API.
- **Modify (test seam, definitions and flag):** `src/ttnn/device.cpp` — define the two declarations behind `#ifdef IOM_ENABLE_TESTING`, plus a TU-private `std::atomic<bool>` arm-and-consume flag, plus a TU-private `bool` consumed-predicate flag, all in an anonymous namespace inside `iom`. The definitions are whole-body guarded by `#ifdef IOM_ENABLE_TESTING`; a release or non-test build contains no symbols, no flag, no consumes, no probe.

  ```cpp
  #ifdef IOM_ENABLE_TESTING
  namespace {
      std::atomic<bool> g_fail_next_quarantine_action{false};
      bool g_quarantine_action_fault_consumed = false;
  }
  namespace iom::ttnn_test {
      void fail_next_quarantine_action_for_testing() noexcept {
          g_fail_next_quarantine_action.store(true,
                  std::memory_order_release);
      }
      bool quarantine_action_fault_consumed_for_testing() noexcept {
          return g_quarantine_action_fault_consumed;
      }
  }
  #endif
  ```

  Additionally, define a TU-private helper `consume_quarantine_action_fault_locked() noexcept(false)` (whole-body guarded by `#ifdef IOM_ENABLE_TESTING`) that performs `exchange(false, std::memory_order_acquire)` on the atomic and, when the observed value was `true`, sets `g_quarantine_action_fault_consumed = true` and throws `std::bad_alloc`. When the observed value was `false`, the helper returns without throwing and without setting the consumed flag. The helper is the *only* consult site and lives in the same anonymous-namespace block.

- **Modify (consumption in quarantine_native):** `src/ttnn/device.cpp` — inside `~TtnnTensor`'s `quarantine_native`, immediately before the `new ttnn_detail::TtnnNativeCleanupAction(...)` call, call the TU-private helper. The destructor's existing `try`/`catch` routes the helper's `std::bad_alloc` through the same `(void)retained.release();` clause as the node-allocation throw. There is no path-aware branching in the destructor. The helper's `#ifdef IOM_ENABLE_TESTING` guard is inside the lambda body so the helper itself vanishes in non-test builds without affecting the destructor's control flow.

- **Modify (build wiring):** `CMakeLists.txt:270-277` — inside the existing `if(BUILD_TESTING)` block, add `if(TTNN_ENABLED) target_compile_definitions(iom_ttnn PRIVATE IOM_ENABLE_TESTING) endif()` beside the CUDA/ROCm equivalents. `iom_ttnn_conformance_tests` is *not* given `IOM_ENABLE_TESTING`; it compiles the test TU without the macro and consumes the seam by linking `iom_ttnn` (which carries the macro). This mirrors how CUDA/ROCm tests consume the staging-pool seam.

- **Modify:** `docs/changes/0001-tensor-view/spec.md` §7 — amend the task54 deferred-free bullet with the implementation-generic, release-generic deliberate-leak exception listed under Implementation references. Do not introduce any implementation-specific paragraph (no `std::vector<ttnn::Tensor>`, no `unique_ptr`, no backend-specific phrasing). The TTNN-specific mechanism is documented inside this task's spec, not in the umbrella contract.

- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — one new `TEST_CASE` that arms the seam, observes the consume, verifies the destructor returned normally, and runs a follow-up healthy-path destruction to prove quarantine still drains at device teardown.

## Implementation references

- **Modify:** `src/ttnn/device.cpp:343` — `std::vector<ttnn::Tensor> planes_;` becomes `std::unique_ptr<std::vector<ttnn::Tensor>> planes_;`.
- **Modify:** `src/ttnn/device.cpp:214-240` — `TtnnTensor` constructor. Initialize `planes_ = std::make_unique<std::vector<ttnn::Tensor>>();` before the `reserve`/`push_back` loop; translate `planes_.reserve` → `planes_->reserve`, `planes_.push_back` → `planes_->push_back`.
- **Modify:** `src/ttnn/device.cpp:242-318` — `~TtnnTensor`. `original_address` reads `planes_->data()` (`:243`); the safe-release path calls `planes_->clear()` under the API mutex (`:308`); `quarantine_native` is restructured as specified under Requirements.
- **Modify:** `src/ttnn/device.cpp:321-323,325-331,333-339` — `storage_handle()`, `region_from_host`, and `region_to_host` read `planes_->data()` instead of `planes_.data()`.
- **Modify:** `src/ttnn/device.cpp` inside `~TtnnTensor`'s `quarantine_native` — immediately before `new ttnn_detail::TtnnNativeCleanupAction(...)`, call the TU-private consume helper (whole-body guarded by `#ifdef IOM_ENABLE_TESTING`). The helper does `exchange(false, std::memory_order_acquire)` on the atomic; when armed, it sets `g_quarantine_action_fault_consumed = true` and throws `std::bad_alloc`; otherwise it returns without effect. The destructor's `try`/`catch` then handles the throw exactly as for a real node-allocation `std::bad_alloc`, routing the released planes pointer through `(void)retained.release();`.
- **Modify:** `src/ttnn/registry_state.hpp` — at the top of the file (after `#include "iom/outstanding_work_registry.hpp"` and before `namespace iom::ttnn_detail`), add an `iom::ttnn_test` namespace declaring the two public-to-tests functions (no `#ifdef` at the declaration; the declarations are always visible to the TTNN test TU). Use C++ linkage (no `extern "C"`). The declarations are not part of any public API.
- **Modify:** `src/ttnn/device.cpp` — at file scope (inside the existing `namespace iom { ... }` and outside any anonymous namespace), add the matching definitions guarded by `#ifdef IOM_ENABLE_TESTING`: a TU-private `std::atomic<bool> g_fail_next_quarantine_action`, a TU-private `bool g_quarantine_action_fault_consumed`, the two `iom::ttnn_test::fail_next_quarantine_action_for_testing() noexcept` and `iom::ttnn_test::quarantine_action_fault_consumed_for_testing() noexcept` definitions, and the TU-private `consume_quarantine_action_fault_locked()` helper that does `exchange(false)` and throws on arm. All definitions and the helper are whole-body guarded by `#ifdef IOM_ENABLE_TESTING`.
- **Modify:** `CMakeLists.txt:270-277` — inside the existing `if(BUILD_TESTING)` block, add `if(TTNN_ENABLED) target_compile_definitions(iom_ttnn PRIVATE IOM_ENABLE_TESTING) endif()` beside the CUDA/ROCm equivalents. `iom_ttnn_conformance_tests` itself does not get `IOM_ENABLE_TESTING` (the test TU links `iom_ttnn`, which carries the macro and provides the seam symbols).
- **Modify:** `docs/changes/0001-tensor-view/spec.md` §7 — append the exception sentence inside task54's deferred-free bullet (the bullet already keeps one statement per line; the new sentence folds into the same declaration). The exact insertion text:

  `If constructing or recording the quarantine cleanup action itself fails inside a `noexcept` tensor destructor, the storage is deliberately leaked for the remainder of the process.`

  The insertion names neither backend nor mechanism. It says "the storage" (not "the allocation"), says "leaked" (not "freed"), and says "for the remainder of the process" (not "until device destruction"), so it is release-generic and matches both CPU/CUDA/ROCm/SYCL (which leak a raw `address`) and TTNN (which leaks a native-plane vector).

  Anchor by text, not line number: task54 rewrites the §7 free-exactly-once bullet into a longer deferred-free bullet and inserts a new sentence before the TTNN paragraph, so every §7 line number below task54's edit shifts. Locate the bullet whose text begins `- destruction calls \`allocator.free(address)\` exactly once: at tensor destruction, or deferred to device destruction` and append the exception sentence to that same bullet. Do not add any extra paragraph at the TTNN paragraph ("TTNN owns native materialized storage through its runtime …") or anywhere else in §7. The TTNN-specific mechanism is recorded inside this task's spec, not in the umbrella contract.
- **Read:** `src/cpu/device.cpp:500-510`, `src/cuda/device.cpp:161-174`, `src/rocm/device.cpp:141-154`, and the SYCL posture (after 55-ST-001) — the established four-backend swallow-and-leak pattern: `try { quarantine.emplace<...>(...); } catch (...) { /* intentional leak */ } address_ = nullptr;` (CUDA/ROCm/SYCL variant) or the equivalent raw-address quarantine leak. The TTNN restructured lambda follows the same shape with a `unique_ptr` member instead of a raw `address_` member.
- **Read:** `src/cuda/staging_pool.hpp:60-62` (declaration, ungated) and `src/cuda/staging_pool.cpp:25-29` (definition + consumption, gated by `#ifdef IOM_ENABLE_TESTING`) — the established per-class, single-point fault-injection seam the TTNN seam mirrors exactly: header declaration ungated so test targets see it without `IOM_ENABLE_TESTING`; definition and consumption gated so non-test builds compile no dead code. ROCm's twin lives at `src/rocm/staging_pool.cpp:39-43` with the same shape.
- **Read:** `CMakeLists.txt:270-277` — the existing `IOM_ENABLE_TESTING` definition block for CUDA/ROCm inside `if(BUILD_TESTING)`; mirror the same gate for TTNN.
- **Read:** `include/iom/outstanding_work_registry.hpp:115-167` — `Quarantine::add(std::unique_ptr<CleanupAction>) noexcept` never throws; only the action node's own allocation/construction and (in task61's later vector `Quarantine`) the recording path can fail. `Quarantine::drain() noexcept` invokes each action's `run()` exactly once.
- **Read:** `docs/changes/0001-tensor-view/54-CC-004-align-allocator-lifetime-contract/spec.md` — task54's parent-spec §7 amendment (the deferred-free bullet it writes in place of the current `spec.md:449` text) that this task's exception sentence must follow, folded into the same bullet with no separate paragraph. Because this task is blocked by task54, locate the bullet by its post-task54 text, not by a pre-task54 line number.
- **Read:** `docs/changes/0001-tensor-view/49-ST-003-fence-tensor-destruction/spec.md` requirement 11 — "Destructors and quarantine drain are no-throw effective. Cleanup errors remain recorded for diagnostics and do not permit unsafe direct reuse."
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — the conformance driver already constructs real `TtnnDevice` instances via `iom::make_ttnn_device(0)` and runs doctest `TEST_CASE`s under the `iom_ttnn_conformance_tests` target (`test/CMakeLists.txt:269-292`). Add the fault-injection case in this file. The test consumes `iom::ttnn_test::*` directly; the test TU compiles with `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN` and without `IOM_ENABLE_TESTING`, and calls into `iom_ttnn` (which is built with `IOM_ENABLE_TESTING` whenever `BUILD_TESTING && TTNN_ENABLED`).

## Requirements

1. Delete the `std::terminate()` call at `src/ttnn/device.cpp:249`. It appears nowhere else in `src/ttnn/`.
2. `TtnnTensor::planes_` is a `std::unique_ptr<std::vector<ttnn::Tensor>>`, allocated once in the constructor with `std::make_unique` before any `ttnn::create_device_tensor` call. The pointer is non-null for the lifetime of a successfully-constructed tensor.
3. The restructured `quarantine_native` lambda:
   - takes ownership of the planes vector with `std::unique_ptr<std::vector<ttnn::Tensor>> retained(planes_.release());` — noexcept, no allocation, leaves `planes_` null so the member destructor cannot destroy native planes;
   - if `retained` is null (nothing to quarantine), returns immediately;
   - inside one `try` block, consults the TU-private consume helper and then attempts the `TtnnNativeCleanupAction` node allocation; on any throw from either step, the `catch` clause calls `(void)retained.release();` and returns — the vector buffer and its native planes become intentionally unreachable, matching the sibling backends' `address_ = nullptr` leak;
   - on success, calls `state_->quarantine.add(std::move(action));` (noexcept) and lets `retained` go out of scope with the vector already moved into the action.
4. The uniform observable policy, identical on CPU, CUDA, ROCm, SYCL, and TTNN: on quarantine-cleanup-step failure (construction or, per task61, recording), the destructor returns normally, the failed storage is not freed and not returned to its allocator/native runtime, no quarantine entry is created for that tensor, and the process continues.
5. `Quarantine::add` remains `noexcept`. No throwing operation is introduced on the quarantine path outside the `try` block of requirement 3.
6. The safe-release path (`planes_->clear()` under the API mutex), `storage_handle()`, `region_from_host`, and `region_to_host` behavior is unchanged. The member type change is implementation-only; no public interface (`Device`, `Tensor`, `TensorView`, `Allocator`, `DeviceOps`) is affected.
7. `TtnnNativeCleanupAction`'s definition and `run()` teardown sequence (finish barrier, then `planes_.clear()`) are unchanged.
8. This policy is the established four-backend convention. AR-002's future shared destructor helper must preserve it verbatim for every backend it absorbs.
9. Existing 49-ST-003 acceptance regressions (CPU poisoning/recycling, CUDA/ROCm `ReusingCudaAllocator`, SYCL teardown fencing after 55-ST-001, TTNN quarantine ownership) remain observable and green.
10. The TTNN fault-injection seam is declared in `src/ttnn/registry_state.hpp` (internal header, not part of any public API, no `extern "C"`, no public `include/iom/` symbol). The declarations are ungated so the TTNN test TU always sees them. The definitions and a single TU-private atomic arm flag plus a `bool` consumed flag live in `src/ttnn/device.cpp` whole-body guarded by `#ifdef IOM_ENABLE_TESTING`. A non-test build contains no symbol, no flag, no consume site, no probe.
11. Two exposed-to-tests functions plus a TU-private helper implement the seam exactly:

    ```cpp
    namespace iom::ttnn_test {
        void fail_next_quarantine_action_for_testing() noexcept;
        bool quarantine_action_fault_consumed_for_testing() noexcept;
    }
    ```

    `fail_next_quarantine_action_for_testing()` arms the atomic (`store(true, release)`); the consume helper, called from exactly one site inside `quarantine_native` immediately before the action-node allocation, does `exchange(false, acquire)` on the atomic and throws `std::bad_alloc` when the observed value was `true` and the consumed flag has not yet been set. The destroy-side `try`/`catch` treats the helper's throw identically to a real action-node `std::bad_alloc`. `quarantine_action_fault_consumed_for_testing()` returns the `bool` consumed flag, allowing the test to prove that the intended consume site fired (not some other allocation along the destructor path).

12. CMake defines `IOM_ENABLE_TESTING` for `iom_ttnn` whenever both `BUILD_TESTING` and `TTNN_ENABLED` are on, mirroring the CUDA/ROCm pattern at `CMakeLists.txt:271-275` and living inside the existing `if(BUILD_TESTING)` block. `iom_ttnn_conformance_tests` itself does not get `IOM_ENABLE_TESTING`; it consumes the seam via the `iom_ttnn` library, which carries the macro and provides the seam symbols.
13. `docs/changes/0001-tensor-view/spec.md` §7 is amended by appending exactly one sentence (the exception in requirement 14's clause a) to the task54 deferred-free bullet. No other §7 sentence is touched. The amendment:
    - folds the deliberate-leak exception into the free-exactly-once bullet itself (not a separate paragraph), so a reader of §7 sees the canonical rule and its sole deliberate exception in one sentence;
    - names neither backend nor mechanism;
    - says "constructing or recording", so task61 can preserve the durable wording verbatim when it swaps the quarantine intrusive list for a vector whose post-construction growth may fail (a recording failure, not a construction failure, but the same exception class under the same contract);
    - does not introduce any extra §7 paragraph referencing `std::vector<ttnn::Tensor>`, `unique_ptr`, `ttnn::Tensor`, or any other implementation detail.

14. The exact insertion text appended to task54's bullet is:

    `If constructing or recording the quarantine cleanup action itself fails inside a `noexcept` tensor destructor, the storage is deliberately leaked for the remainder of the process.`

## Non-goals

- Sharing the destructor protocol or registry glue across backends (AR-002 owns that convergence; this task establishes the failure policy AR-002 must preserve).
- Porting the 49-ST-003 registry/quarantine pattern to SYCL (55-ST-001 / AR-001 own that work).
- Reducing `Quarantine`'s intrusive-list structure to a `std::vector` (AR-003 / task61).
- Modifying the CPU/CUDA/ROCm/SYCL quarantine paths — their behavior is the established policy this task aligns TTNN to.
- Changing TTNN native plane representation, layout, supported types, or `TtnnNativeCleanupAction`'s definition or `run()` finish-then-clear sequence.
- Adding fault injection on the other backends — ST-004's verification burden is the TTNN side of the uniform policy; CPU/CUDA/ROCm/SYCL already follow the established policy.
- Adding a global `operator new` override or any other project-wide runtime seam for fault injection. The TTNN seam stays internal to `iom_ttnn`, is consumed only inside `quarantine_native`, and is observable through `iom::ttnn_test::*` only.
- Amending parent-spec §6 — task54 owns the §6 lifetime-rules changes; this task amends §7 only.
- Adding any extra §7 paragraph describing TTNN's specific mechanism (`std::vector<ttnn::Tensor>` planes, `unique_ptr` wrap, etc.). The mechanism belongs in this task's spec; the umbrella contract stays implementation-generic.
- Building, running, or verifying anything outside the spec text and the `IOM_ENABLE_TESTING` build wiring. Project-wide validation is the main agent's responsibility, run once after all subagents land.

## Acceptance criteria

- [ ] `std::terminate` appears nowhere in `src/ttnn/device.cpp` (verified by source inspection).
- [ ] `~TtnnTensor`'s quarantine path contains no throwing operation outside the `try` block wrapping the action-node allocation; the consume helper (when compiled with `IOM_ENABLE_TESTING`) throws inside the same `try` block, so the helper-throw and the node-allocation-throw land in one `catch` clause and both leak rather than terminate or destroy native planes.
- [ ] `iom::ttnn_test::fail_next_quarantine_action_for_testing()` and `iom::ttnn_test::quarantine_action_fault_consumed_for_testing()` are declared in `src/ttnn/registry_state.hpp` (no public `include/iom/` symbol, no `extern "C"`, no public API), defined in `src/ttnn/device.cpp` whole-body under `#ifdef IOM_ENABLE_TESTING`, and consulted through a TU-private consume helper called from exactly one site inside `quarantine_native`.
- [ ] `CMakeLists.txt` defines `IOM_ENABLE_TESTING` for `iom_ttnn` only inside `if(BUILD_TESTING)` when `TTNN_ENABLED` is on. `iom_ttnn_conformance_tests` does not get `IOM_ENABLE_TESTING` itself; it consumes the seam via `iom_ttnn`.
- [ ] A TTNN hardware `TEST_CASE` runs three sequential sub-assertions, each scoped:
  1. arm-and-destroy-invalidated: call `iom::ttnn_test::fail_next_quarantine_action_for_testing()`, arrange an invalidated fence (queue teardown before wait — the existing 49-ST-003 invalidation path), destroy the destination tensor, then assert
     - `iom::ttnn_test::quarantine_action_fault_consumed_for_testing()` returns `true` (consume helper fired at the intended site);
     - the destructor returned normally (no `SIGABRT`, no `std::terminate` diagnostic, no exception escape);
     - source inspection of the destructor shows the `(void)retained.release();` statement on the catch path (verified by static read of `src/ttnn/device.cpp` around the `quarantine_native` lambda).
  2. healthy follow-up: with the seam not re-armed (the consumed flag stays `true`; the helper only throws on arm→consume transitions, so subsequent destructions take the normal path), destroy one more tensor with a working quarantine, assert the device's quarantine received exactly one entry and that the follow-up destruction produced no `consumed_for_testing()` state change.
  3. drop the `iom_ttnn_conformance_tests` `add_test` registration: `ctest -N` shows the new fault-injection case only inside `iom_ttnn_conformance_tests`, with no orphan tests registered under a different executable and no global-new override anywhere in `test/`.
- [ ] No global `operator new` override, no `extern "C"`, no public `include/iom/` symbol is added by this task. The file diff touches `src/ttnn/device.cpp`, `src/ttnn/registry_state.hpp`, `CMakeLists.txt`, `docs/changes/0001-tensor-view/spec.md`, and `test/ttnn/test_ttnn_conformance.cpp`.
- [ ] The existing CPU quarantine unit case `Quarantine runs allocator cleanup at most once` (`test/test_iom.cpp:2137`) still passes unchanged, confirming CPU quarantine behavior is untouched.
- [ ] `docs/changes/0001-tensor-view/spec.md` §7 contains task54's deferred-free bullet text followed by the deliberate-leak exception sentence specified under Implementation references; the bullet carries one declaration with both halves visible; no other §7 sentence changed. `grep -n "deliberately leaked" docs/changes/0001-tensor-view/spec.md` returns exactly one match, inside §7's free-exactly-once bullet.
- [ ] TTNN conformance suite (`iom_ttnn_conformance_tests`) passes on TT hardware via the remote-development workflow with no new failures relative to the pre-fix baseline.

## Verification

- Build and run the TTNN suite on TT hardware via `.agents/skills/remote-development`: from a configured TTNN build directory, `ctest -R iom_ttnn` must pass, including the new fault-injection case in `iom_ttnn_conformance_tests`. Use exclusive accelerator access; no sanitizer is required for this fault class.
- Run the CPU quarantine unit test locally: `ctest -R iom_tests` (or the doctest filter `-tc="Quarantine runs allocator cleanup at most once"`) must pass unchanged.
- Source audit: `grep -n "std::terminate" src/ttnn/device.cpp` returns no matches; `grep -n "planes_" src/ttnn/device.cpp` shows only `planes_->` accesses plus the single `planes_.release()` leak path; `grep -n "fail_next_quarantine_action_for_testing\|quarantine_action_fault_consumed_for_testing" src/ttnn/` returns the expected declaration (one match in `registry_state.hpp`, definitions and consume site in `device.cpp`); no `operator new` override appears anywhere in `test/`; `grep -n "IOM_ENABLE_TESTING" CMakeLists.txt` now lists `iom_ttnn` alongside `iom_cuda` and `iom_rocm`, all guarded by `if(BUILD_TESTING)`.
- Spec audit: `grep -n "deliberately leaked" docs/changes/0001-tensor-view/spec.md` returns exactly one match, the §7 free-exactly-once bullet's deliberate-leak exception; `grep -n "std::vector<ttnn::Tensor>\|unique_ptr-wrapped" docs/changes/0001-tensor-view/spec.md` returns no matches (mechanism stays in the task spec, not the umbrella contract).
- Build wiring check: configure a TTNN-enabled build with `BUILD_TESTING=ON`, confirm `iom_ttnn` is compiled with `-DIOM_ENABLE_TESTING`, and confirm `iom_ttnn_conformance_tests` links the seam symbols by name (`fail_next_quarantine_action_for_testing`, `quarantine_action_fault_consumed_for_testing`) without any global-new override or undefined reference.
- Fault-injection scenario, TT hardware: submit a queued copy, force the fence to fail (queue teardown before wait), call `iom::ttnn_test::fail_next_quarantine_action_for_testing()`, destroy the destination tensor, then assert `iom::ttnn_test::quarantine_action_fault_consumed_for_testing() == true`, the test continues to the next assertion, and the process exits 0 at suite completion.
- C++ semantics the fix depends on (verified with a standalone probe): for `new T(args...)`, a throwing `operator new` fires before `T`'s constructor is invoked, so by-value parameters are never bound and the caller's `std::move(*retained)` source is untouched; the catch clause's `retained.release()` therefore always leaks the full planes buffer on any throw inside the `try` block.

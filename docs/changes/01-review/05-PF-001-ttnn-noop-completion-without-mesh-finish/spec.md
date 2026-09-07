# Complete TTNN no-op copies without a mesh finish

**Order:** 05
**Priority:** P1 — remove unnecessary device-wide blocking from identical-window no-op completion
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `PF-001`
**Review area:** Performance
**Review severity:** high
**Review verification:** strongly-supported, confidence 96
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/device.cpp:581-587` (`TtnnQueue::execute`), `src/ttnn/device.cpp:668-724` (`complete_task`), `src/ttnn/device.cpp:435-452` (`finish_locked`/`finish_native`)

## Outcome

An identical-window TTNN copy completes as a waitable no-op without calling mesh `finish()`. Mixed completion batches perform one finish only when at least one native operation was submitted, while sequence advancement and registry release remain correct.

## Current problem

The invariant is that an identical source/destination window submits no native work and must not impose device-wide ordering beyond its own token. `TtnnQueue::copy` detects `identical_window` at `src/ttnn/device.cpp:510-525`; `execute` records outcome state and returns through the no-op branch at `:581-587` without submitting a native plane. `complete_task` at `:668-724` nevertheless calls `finish_native` for every outcome batch. `finish_native` takes the API mutex and `finish_locked` calls `mesh_command_queue(0).finish()` at `:435-452`.

Consequently, a no-op token blocks for unrelated pending mesh work and serializes other TTNN API users. CUDA/shared GPU and SYCL no-op paths return without outcome/fence registration, and CPU completes inline, demonstrating that this device-wide finish is not required by the no-op contract. Existing TTNN finish accounting covers native copies but does not assert zero finish for a no-op.

## Scope

- Carry an explicit `native_work_submitted` bit or equivalent no-op completion marker through TTNN completion state.
- Skip `finish_native` for no-op-only batches, advance `last_finished_seq_`, and release no-op registry entries without a device finish.
- For mixed batches, perform exactly one finish when at least one member submitted native work, preserving token ordering and ownership release.
- Keep the common identical-window predicate and all native copy behavior unchanged.

## Implementation references

- **Modify:** `src/ttnn/device.cpp:510-525,581-587,668-724` — no-op detection, `TtnnQueue::execute`, and `complete_task`; carry native-submission state and advance completion markers.
- **Modify:** `src/ttnn/device.cpp:435-452` — `finish_locked`/`finish_native` call ownership; ensure no-op paths do not reach it.
- **Read:** `src/shared/gpu_queue.hpp:190-198` and `src/sycl/copy.cpp:485-490` — existing no-op completion convention without runtime/device wait.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:1107-1157` — add no-op finish-count and mixed-batch completion coverage.

## Requirements

- Identical-window copies must produce a successful waitable token without native plane submission or `mesh_command_queue(0).finish()`.
- No-op completion must advance `last_finished_seq_` sufficiently that `DeviceOps::wait` does not fence and finish again.
- A mixed batch must finish once if and only if at least one member submitted native work; no-op entries still complete in sequence order.
- Release no-op registry entries without device finish, and preserve API mutex ownership and native copy failure behavior.

## Non-goals

- Do not alter the common no-op predicate, TTNN native per-plane copy semantics, host-transfer synchronization, failure retention, or API-mutex ownership.
- Do not add a benchmark-only fast path, bypass token ordering, or change queue API shape.
- Do not remove finishes required by batches containing native work.

## Acceptance criteria

- [ ] An identical-window TTNN copy submits no native plane, returns a waitable token, and increments the finish counter zero times through both completion and repeated wait.
- [ ] A no-op token completes while unrelated mesh work is pending without waiting for that work or serializing another queue.
- [ ] A batch mixing no-op and native copies performs exactly one finish for native members and releases all registry entries in order.
- [ ] Existing TTNN native copy logical results and ownership behavior remain unchanged.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync ttnn 05-PF-001-ttnn-noop-completion-without-mesh-finish`
- `.agents/skills/remote-development/scripts/remote-exec ttnn 05-PF-001-ttnn-noop-completion-without-mesh-finish 'cmake -S . -B build-ttnn -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=ON -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-ttnn --target iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-ttnn --output-on-failure -R "iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — TTNN smoke, conformance, and coexistence pass.
- On TTNN hardware, instrument `copy_finish_count_for_testing`, run no-op-only and mixed batches with pending native work, and profile the mesh/API timeline; expect zero finishes and no pending-work drain for no-op-only tokens, exactly one finish for mixed batches, and lower no-op latency without changed token ordering.

# Retain quarantined TTNN native planes until completion is proven

**Order:** 02
**Priority:** P0 — prevent quarantined native planes from being destroyed after failed mesh finish
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `ST-003`
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 98
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/registry_state.hpp:66-100` (`TtnnNativeCleanupAction`), `src/ttnn/device.cpp:326-361` (`quarantine_native`), `include/iom/detail/outstanding_work_registry.hpp:264-306`

## Outcome

A TTNN native cleanup action retains its planes whenever mesh completion is not proven. Native planes are cleared only after a successful finish; `Quarantine::drain` holds `mutex_` across the bounded single-pass run and in-place compaction so concurrent `add` cannot create a merge requiring allocation, preserves reverse execution order, and swaps incomplete actions back without allocation or same-drain retry. The `Quarantine` destructor deliberately releases/leaks still-incomplete action owners after its final bounded drain instead of destroying planes while device work may still reference them.

## Current problem

The invariant is that native TTNN planes associated with outstanding mesh work remain alive until a successful finish establishes quiescence. `TtnnNativeCleanupAction::run` at `src/ttnn/registry_state.hpp:66-100` sets `attempted_`, catches finish failure, and then unconditionally executes `planes_.clear()` at `:84`. `device.cpp:326-361` moves native planes into quarantine, while `include/iom/detail/outstanding_work_registry.hpp:264-306` destroys cleanup actions after one drain run. A failed finish therefore can be followed by destruction of planes still referenced by queued mesh work.

The vendor `MeshDevice::close` path performs no finish, and `ScopedDevices` closes with `skip_synchronize=true`; neither is a quiescence guarantee. Marking an action attempted before success also prevents a later drain from retrying the failed finish. The result is an asynchronous native-plane use-after-free risk during failed quarantine and teardown.

## Scope

- Make `TtnnNativeCleanupAction` clear planes only after successful finish and expose completion through `CleanupAction::completed() const noexcept`.
- Make `Quarantine::drain` hold `mutex_` across the bounded single-pass run and in-place compaction, preserve reverse execution order, and swap incomplete actions back without allocation or retrying them during the same drain.
- At terminal teardown, after the final bounded drain, deliberately release/leak still-incomplete action owners rather than destroy their native planes. Do not treat vendor close or `skip_synchronize=true` as completion.
- Keep quarantine ownership and native plane ordering unchanged.

## Implementation references

- **Modify:** `src/ttnn/registry_state.hpp:66-100` — `TtnnNativeCleanupAction`; implement `completed() const noexcept`, clear planes only after successful finish/plane clear, and report incomplete after failure.
- **Modify:** `src/ttnn/device.cpp:326-361` — `quarantine_native`; preserve failed actions and their planes for the bounded quarantine drain and terminal retention disposition.
- **Modify:** `include/iom/detail/outstanding_work_registry.hpp:264-306` — generic `CleanupAction::completed() const noexcept` and `Quarantine::drain`; allocator cleanup reports completed after its one attempt, while TTNN cleanup reports completed only after successful finish/plane clear; run each initial action once, compact incomplete actions in place, and swap them back without allocation.
- **Tests:** TTNN quarantine/conformance coverage adjacent to native cleanup and failure seams; add persistent mesh-finish-failure and later-success drain cases.

## Requirements

- `TtnnNativeCleanupAction::completed() const noexcept` must return true only after a finish succeeds and all native planes are cleared.
- Every non-TTNN cleanup action reports `completed() == true` after its one attempted cleanup; `TtnnNativeCleanupAction` reports completed only after successful finish and plane clear.
- `Quarantine::drain` must hold `mutex_` across the bounded single-pass run and in-place compaction, preserve reverse execution order, and swap incomplete actions back without allocation or same-drain retry; concurrent `add` must not trigger a merging allocation.
- Cleanup and drain paths must be `noexcept` and must not replace or escape the original failure; `planes_` may be cleared only after a successful finish for the represented work.
- The `Quarantine` destructor must perform its final bounded drain, then deliberately release/leak still-incomplete action owners rather than destroy their planes without a proven barrier.
- Preserve native plane ordering, dtype mapping, queue API, and successful cleanup behavior.

## Non-goals

- Do not change TTNN native dtype mapping, plane ordering, copy semantics, or mesh queue API.
- Do not add a vendor-specific finish assumption, global synchronization unrelated to quarantine, or a new allocator policy.
- Do not change ordinary successful quarantine behavior or unrelated outstanding-work actions.

## Acceptance criteria

- [ ] A successful mesh finish clears quarantined planes and marks the cleanup action complete exactly once.
- [ ] A forced finish failure leaves the action and every native plane retained; a later successful drain retries and then clears them without use-after-free.
- [ ] Repeated finish failures remain non-throwing; each mutex-held drain attempts each initial action once in reverse execution order, leaves incomplete actions for a later drain, and terminal teardown releases/leaks them rather than clearing planes without a barrier.
- [ ] TTNN quarantine and subsequent native operations preserve logical data and plane ownership, with no native-plane access after release under supported TTNN/vendor runtime memory diagnostics.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync ttnn 02-ST-003-ttnn-quarantine-native-plane-retention`
- `.agents/skills/remote-development/scripts/remote-exec ttnn 02-ST-003-ttnn-quarantine-native-plane-retention 'cmake -S . -B build-ttnn -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=ON -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-ttnn --target iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-ttnn --output-on-failure -R "iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — TTNN smoke, conformance, and coexistence pass.
- On TTNN hardware, inject persistent mesh-finish failure, drain repeatedly, then allow one successful finish; expect one attempt per initial action per drain, retained planes throughout failures, one successful cleanup, and terminal release/leak rather than premature plane destruction. Run TTNN/vendor runtime memory diagnostics if available and expect no access after release.

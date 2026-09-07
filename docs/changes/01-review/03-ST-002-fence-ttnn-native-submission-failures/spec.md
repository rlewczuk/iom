# TTNN native copies can outlive failed submission untracked

**Order:** 03
**Priority:** P0 — lifetime
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** ST-002
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 89
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/copy.hpp:39-44`; `src/ttnn/copy.cpp:284-292`; `src/ttnn/device.cpp:438-482` (`TtnnQueue::execute`); `include/iom/iom.hpp:66-104,310-339`

## Outcome

Every TTNN native command submitted for an IOM operation remains associated with an owner/completion record until the `MeshCommandQueue` has reached a terminal result. A failed `copy()` never leaves untracked device work: after any failure point the caller receives either a waitable token whose repeated wait reports the retained failure after queue completion, or a thrown call that has synchronously established completion of all submitted native work before owners can be destroyed or reused.

## Current problem

Native TTNN plane copies are enqueued before any ownership/completion record exists, so a submission failure can strand pending device work. Timeline: (1) `TtnnQueue::execute` acquires the device API mutex and calls `ttnn_detail::copy_planes`; (2) `copy_planes` loops over view planes calling `ttnn::copy`, whose contract explicitly states it "returns after submission" while the caller must finish the mesh command queue (`src/ttnn/copy.hpp:39-42`); (3) one or more plane copies can be pending when `copy_planes` returns, or an exception can occur after an earlier plane was submitted; (4) only after that call does `execute` build a fence and call `register_copy_entries`/`outcomes_.emplace` (`src/ttnn/device.cpp:438-470`); (5) if native submission throws, control exits before registration; if registry allocation or outcome insertion throws, the catch removes any entries and rethrows (`src/ttnn/device.cpp:459-482`) without `mesh_command_queue(0).finish()`, without retaining a task fence, and without publishing the task; (6) `StagedWorker::submit_copy` erases the staged task on execute exception (`include/iom/iom.hpp:66-104`) and `DeviceOps::submit` rolls back the sequence (`include/iom/iom.hpp:310-339`), so there is no token the caller can wait. Subsequent native copies or `TtnnTensor` destruction can race the still-pending planes; TTNN owner destruction clears native planes immediately when no registry snapshot remains (`src/ttnn/device.cpp:259-295`). Impact: a synchronously reported failure can leave native writes running after `copy()` has thrown with no IOM token, registry entry, or queue callback protecting storage — later copies may observe partial state, owner destruction may release native tensor storage while commands still reference it, and an exception from a later operation can be misattributed.

## Scope

- Rework TTNN submission transaction boundaries so native work cannot precede its ownership record without a complete rollback protocol; on the success path keep the current single mesh finish per operation.
- Cover every failure point — after zero, one, or many native plane submissions — and the failure of native finish itself.

## Implementation references

- **Modify:** `src/ttnn/device.cpp:438-482` — `TtnnQueue::execute` transaction ordering: prefer constructing the per-operation completion/fence and registering source/destination ownership **before** any native plane submission; on failure after registration, retain a failed fenced task/sequence until `finish_native` establishes the terminal result.
- **Modify:** `src/ttnn/device.cpp:489-530` — `TtnnQueue::{complete_task,~TtnnQueue}` plus `finish_native` (`src/ttnn/device.cpp:336-345`) must handle a retained failed task and finish under `api_mutex_`.
- **Modify:** `src/ttnn/copy.cpp:284-292` / `src/ttnn/copy.hpp:39-44` — `ttnn_detail::copy_planes` must report whether any plane was actually submitted, so the caller can distinguish pre-submission throws from post-submission throws.
- **Modify:** `include/iom/iom.hpp:66-104` — `StagedWorker::submit_copy` must publish/complete the failed operation when native work was submitted rather than silently erasing it.
- **Read:** `src/ttnn/device.cpp:259-295` (`TtnnTensor::{release_native,quarantine_native}`) and the registry helpers `register_copy_entries` — the cleanup/quarantine conventions to reuse.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:384-420,458-518,520-550` — existing conformance/async cases; add partial-plane and registration-failure cases.

## Requirements

- No registry entry, sequence rollback, queue destruction, or native tensor clear may leave an in-flight TTNN command without a completion owner.
- If a pre-registration design is retained, `copy_planes` must report whether any plane was submitted, and every throw after the first submission must synchronously finish under `api_mutex_` before ownership is removed and the exception rethrown; if the finish itself fails, preserve an invalidated/quarantined ownership record instead of releasing native planes.
- When work was submitted, the caller must be able to observe the outcome: either `copy()` returns a waitable token whose repeated wait reports the retained failure after queue completion, or the thrown call has already synchronously completed all submitted native work before owners can be destroyed or reused.
- Successful copies must retain current logical-byte/oracle behavior and exactly one mesh finish per operation.

## Non-goals

- Altering TTNN supported dtype policy, native tile mapping, view semantics, public token format, or unrelated quarantine-action allocation-failure behavior.
- Adding a second global TTNN scheduler or broad device synchronization for unrelated operations.

## Acceptance criteria

- [ ] At every failure point after zero, one, or many native plane submissions (test seams: throw from `copy_planes` after a chosen plane; inject `bad_alloc` during registry/outcome insertion), either `copy()` returns a waitable failed token whose repeated wait reports the retained failure after queue completion, or the thrown call synchronously finished all submitted native work before owners could be destroyed/reused.
- [ ] Destroying and reusing source and destination after each failure injection leaves no untracked native command; a later operation never observes partial writes, and no exception from a later operation is misattributed to the failed one.
- [ ] Successful multi-plane copies produce identical logical bytes/oracle results with exactly one mesh finish per operation; queue destruction and repeated failed waits drain without leaks or double release.

## Verification

Actual validation: none (read-only review); proposed gates below.

Proposed gates (TTNN hardware on a remote host per remote-development; sync the workspace, then run via `remote-exec`):

- Add a TTNN test seam that throws from `copy_planes` after a chosen plane and another that injects `bad_alloc` during registry/outcome insertion; submit a multi-plane copy under each seam, require either a returned failed token or a throw only after mesh completion, then destroy/reuse source and destination and assert no native command remains untracked and no later operation sees partial writes.
- Run the TTNN conformance, queue-destruction, repeated failure-wait, and native storage oracle tests: `ctest --test-dir <build> -R '^iom_ttnn_(smoke|conformance)_tests$' --output-on-failure`.

- `ctest --test-dir <build> -R '^iom_ttnn_(smoke|conformance)_tests$' --output-on-failure`
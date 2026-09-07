# Retire TTNN upload staging only after proven completion

**Order:** 01
**Priority:** P0 — prevent asynchronous TTNN host-staging use-after-free and reuse
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `ST-001`
**Review area:** C++/GPU stability
**Review severity:** high
**Review verification:** strongly-supported, confidence 98
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/copy.cpp:304-344` (`ttnn_detail::region_from_host`), `src/ttnn/staging.hpp:50-92,240-258` (`UploadLease`, `release_upload`)

## Outcome

Every TTNN upload staging buffer referenced by a submitted mesh read remains owned and non-reusable until a successful queue completion is proven. A partial submission or failed drain preserves the original synchronous exception and retains submitted bytes without freeing them from an exception or destructor path.

## Current problem

The invariant is that host staging referenced by an asynchronous TTNN mesh read must remain alive and unchanged until a successful queue finish, event, or equivalent completion proof. `region_from_host` acquires all per-plane `UploadLease` objects at `src/ttnn/copy.cpp:320-334`, submits each plane before the single finish at `:335-341`, and unwinds without a catch/drain when a later plane submission throws. The two-argument `ttnn::copy_to_device` path enqueues writes; the non-pinned path is non-blocking and generic `HostBuffer`/`MemoryPin` does not retain the backing bytes. The installed TT-Metalium queue contract at `/usr/include/tt-metalium/mesh_command_queue.hpp:97-110` requires host bytes to remain unchanged until blocking completion/finish/event.

On unwind, `UploadLease` destruction invokes `release_upload(..., poisoned=true)` (`staging.hpp:50-59,240-258`), which swaps/free-releases the retained vector. Thus a plane-0 write can still read staging after plane-1 submission fails, while a later transfer can reuse the bytes. The existing plane-1 fault seam in `test/ttnn/test_ttnn_conformance.cpp:706-726` checks follow-up data but does not prove plane-0 completion before release. A throwing submission must be treated as possibly in flight; vendor close performs no finish and `skip_synchronize=true` teardown is not a quiescence proof.
Each upload slot therefore needs an allocation-free per-slot `retired` state. A retired slot must be skipped by `acquire_upload`; every later successful finish that covers the slot must invoke staging reclamation, while terminal failure intentionally abandons/leaks protected slot storage rather than freeing it without a barrier.

## Scope

- Replace upload failure unwinding with an explicit completion/disposition protocol in `region_from_host`, `upload_plane`, `UploadLease`, and `release_upload`.
- After any plane has been submitted, attempt exactly one bounded mesh/device finish before disposing of leases. Discard poisoned upload storage only when that finish succeeds.
- When the finish fails, mark each submitted upload slot retired in place without allocation. Every later successful covering finish invokes a staging reclamation method; if completion can never be proven at terminal teardown, intentionally abandon/leak the protected slot storage.
- Preserve the original submission exception; recovery failure must not replace it or escape a destructor.

## Implementation references

- **Modify:** `src/ttnn/copy.cpp` — `ttnn_detail::region_from_host` and `upload_plane`; establish the submission count, one bounded finish attempt, and original-exception preservation.
- **Modify:** `src/ttnn/staging.hpp` — `TtnnHostStaging::UploadLease` and `release_upload`; add allocation-free per-slot `retired` state, make retirement `noexcept`, skip retired slots in `acquire_upload`, and reclaim only after a successful covering finish.
- **Read:** `/usr/include/tt-metalium/mesh_command_queue.hpp:97-110` — queue finish/host-byte lifetime contract; use its completion semantics rather than assuming a throwing submission drained prior work.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:706-726` — extend the partial plane-1 submission fault with failed-finish and repeated-retirement assertions.

## Requirements

- Treat every submission that may have reached the mesh as in flight, including a submission that throws after earlier planes were enqueued.
- Attempt one bounded finish after the first submission failure; release submitted staging only after a successful finish proves completion.
- Make `UploadLease::retire` and all failed-drain/destructor paths non-throwing and allocation-free; preserve submitted byte vectors in per-slot retired storage when completion is unknown.
- `acquire_upload` must skip every slot marked `retired`; each later successful finish covering that slot must call the staging reclamation method. At terminal teardown without a proven barrier, intentionally abandon/leak protected slot storage rather than free it.
- Preserve the original synchronous exception from the submission path, plane ordering, host-byte contents, native dtype mapping, and queue API.

## Non-goals

- Do not change TTNN native dtype mapping, plane ordering, copy semantics, mesh command queue API, or general allocator policy.
- Do not redesign TTNN host pinning or assume vendor runtime behavior not established by a finish/event.
- Do not alter successful uploads beyond the ownership and failure-disposition protocol.

## Acceptance criteria

- [ ] Injecting a plane-1 host submission fault after plane 0 has been submitted attempts one finish; when it succeeds, the original exception is propagated and staging is safely discardable.
- [ ] When the finish also fails, all submitted staging bytes remain retained in per-slot retired state, a subsequent transfer skips those slots, and each later successful covering finish reclaims them; terminal unproven completion abandons/leaks them.
- [ ] Recovery and lease destruction never throw or allocate, and the original submission exception remains observable even when recovery fails repeatedly.
- [ ] A clean follow-up upload produces the expected logical bytes and no staging access occurs after release under supported TTNN/vendor runtime memory diagnostics.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync ttnn 01-ST-001-ttnn-upload-staging-retirement`
- `.agents/skills/remote-development/scripts/remote-exec ttnn 01-ST-001-ttnn-upload-staging-retirement 'cmake -S . -B build-ttnn -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=ON -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-ttnn --target iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-ttnn --output-on-failure -R "iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — TTNN smoke, conformance, and coexistence pass.
- On TTNN hardware, run the plane-1 submission-fault and failed-finish fault seam, then repeat a clean upload; expect the original exception, per-slot retired state, no slot reuse while completion is unknown, reclamation after a successful covering finish, and intentional abandonment/leak rather than unsafe terminal free. Run TTNN/vendor runtime memory diagnostics if available; expect no staging access after release.

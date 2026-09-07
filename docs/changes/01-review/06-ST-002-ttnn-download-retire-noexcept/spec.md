# Make TTNN download retirement allocation-free

**Order:** 06
**Priority:** P1 — prevent noexcept termination while retaining failed TTNN downloads
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase review of main at fb88959fb884b9bb495e055b9fdd0756e430dbde (clean working tree)`
**Finding:** `ST-002`
**Review area:** C++/GPU stability
**Review severity:** medium
**Review verification:** strongly-supported, confidence 99
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `src/ttnn/staging.hpp:146-153` (`DownloadLease::retire`), `src/ttnn/staging.hpp:265-280` (`release_download`), `src/ttnn/copy.cpp:409-430` (failed-drain path)

## Outcome

A failed TTNN download drain never allocates or throws from `noexcept` retirement. The existing sole download slot is `in_use` during normal work; on failed drain, `release_download(Retire)` sets an in-place `retired` bit, clears `in_use`, and leaves `slot.data` in place. Before acquisition, `region_to_host` attempts a covering finish when the slot is retired; a successful finish calls `reclaim_download()`, while an unsuccessful finish preserves the prior failure and acquisition rejects. Terminal unproven completion intentionally abandons/leaks the protected slot storage.

## Current problem

The invariant is that a download staging buffer referenced by submitted mesh work remains alive until completion is proven, and failure handling must not terminate the process from a `noexcept` path. `DownloadLease::retire` is declared `noexcept` at `src/ttnn/staging.hpp:146-153`, but `release_download` at `:265-280` currently performs `retired_.push_back`, which may allocate and throw. `src/ttnn/copy.cpp:409-430` invokes the failed-drain path while unwinding a transfer failure. An allocation failure during `retire` therefore calls `std::terminate` and can lose the ownership needed to protect bytes still referenced by the mesh.

The correct disposition is an in-place state transition on the sole slot: failed-drain retirement leaves `slot.data` untouched, clears `in_use`, and sets `retired`. `acquire_download` must reject that slot until the caller has established a covering finish and `reclaim_download()` clears the bit.

## Scope

- Keep the existing sole download slot `in_use` during normal work; on `release_download(Retire)`, set its in-place `retired` bit, clear `in_use`, and leave `slot.data` in place. Delete the `retired_` retirement vector.
- Make `DownloadLease::retire` a non-allocating, non-throwing state transition that preserves bytes and excludes the retired slot from acquisition.
- In `region_to_host`, before acquisition, attempt a covering finish when the slot is retired; on success call `reclaim_download()`, otherwise preserve the prior failure and reject acquisition.
- Preserve the original transfer exception; terminal unproven completion intentionally abandons/leaks the protected slot storage rather than freeing it.

## Implementation references

- **Modify:** `src/ttnn/staging.hpp:146-153` — `DownloadLease::retire`; remove all allocation and throwing operations from the `noexcept` path.
- **Modify:** `src/ttnn/staging.hpp:265-280` — `release_download`; set the sole slot's in-place `retired` bit, clear `in_use`, leave `slot.data`, implement `reclaim_download()`, and reject retired acquisition.
- **Modify:** `src/ttnn/copy.cpp:409-430` — `region_to_host` failed-drain and pre-acquisition covering-finish path; preserve the prior failure and invoke `reclaim_download()` only after successful coverage.
- **Read:** TTNN upload lease retirement in the same staging owner and existing slot acquisition/release state; reuse its no-throw lifetime conventions without merging the distinct defects.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` download failure/retirement cases; inject allocation failure during failed drain and verify no termination or reuse.

## Requirements

- No code reachable from `DownloadLease::retire`, its destructor, or failed-drain release may allocate, throw, or call a potentially throwing container operation.
- The sole download slot must transition in place from `in_use` to `retired` on `release_download(Retire)`, with `slot.data` left intact; the `retired_` vector must be deleted.
- `region_to_host` must attempt a covering finish before acquiring a retired slot, call `reclaim_download()` only after success, and reject while the slot remains retired while preserving the prior failure.
- Preserve the original download/enqueue exception; terminal unproven completion intentionally abandons/leaks protected slot storage.

## Non-goals

- Do not change TTNN download representation, native dtype mapping, plane ordering, queue API, or general allocator policy.
- Do not make failed downloads silently succeed or discard bytes whose completion is unknown.
- Do not merge upload retirement with this download-specific `noexcept` contract.

## Acceptance criteria

- [ ] With allocation failure injected during a failed-drain path, `retire` does not terminate, allocate, or replace the original exception.
- [ ] A failed drain changes only the sole slot state (`in_use` cleared, `retired` set, `slot.data` retained); acquisition rejects until a covering finish succeeds and `reclaim_download()` clears the bit.
- [ ] Repeated failed drains remain non-throwing and preserve ownership; terminal unproven completion abandons/leaks the slot rather than freeing it.
- [ ] Successful TTNN downloads and existing conformance behavior remain unchanged.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — local common/CPU gates remain green.
- `.agents/skills/remote-development/scripts/remote-sync ttnn 06-ST-002-ttnn-download-retire-noexcept`
- `.agents/skills/remote-development/scripts/remote-exec ttnn 06-ST-002-ttnn-download-retire-noexcept 'cmake -S . -B build-ttnn -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=ON -DSYCL_ENABLED=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-ttnn --target iom_ttnn_smoke_tests iom_ttnn_conformance_tests iom_backend_coexistence_tests && ctest --test-dir build-ttnn --output-on-failure -R "iom_ttnn_(smoke|conformance)_tests|iom_backend_coexistence_tests"'` — TTNN smoke, conformance, and coexistence pass.

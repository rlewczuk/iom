# Copy SYCL synchronous host-transfer spans with host memcpy through the shared-USM staging buffer

**Order:** 59
**Priority:** P1 — close the strict-SYCL host-pointer contract hazard on the supported path without blocking other work
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `ST-005`
**Review severity:** low
**Review verification:** verified (mechanism; impact conditional on runtime), confidence 75

## Outcome

SYCL synchronous host transfers (`Tensor::copy_from_host`/`copy_to_host` on a SYCL tensor) keep their exact bytes, synchronous timing, and error behavior while no SYCL runtime data-movement call ever receives an ordinary host pointer. The host leg of `synchronous_transfer` moves bytes with `std::memcpy` between the caller's span and the existing shared-USM staging buffer on the calling thread, and every pointer handed to the SYCL runtime — the staging buffer and the tensor storage — is a USM allocation valid in the device's owned context. Word-rounded staging bounds, tail-bit zeroing, pad-byte exclusion, and all error categories are unchanged. The design is deliberately the smallest one compatible with the later AR-001 pooling work: the per-call `sycl::malloc_shared` staging and the per-call in-order queue remain, so AR-001 can pool them without touching this host-leg contract again.

## Current failure

`synchronous_transfer` (`src/sycl/copy.cpp:197-245`) allocates a word-rounded shared-USM staging buffer and then moves the host leg with `queue::memcpy`, whose SYCL 2020 contract requires USM allocations for both operands:

- line 220: `queue.memcpy(staging, source.data(), logical_nbytes);` passes `source.data()` — the caller's ordinary host buffer — as the source operand;
- lines 229-230: `queue.memcpy(destination.data(), staging, logical_nbytes).wait_and_throw();` passes `destination.data()` — the caller's ordinary host buffer — as the destination operand.

A raw host pointer is out of contract on strict SYCL implementations. DPC++ — the only compiler this build permits (`CMakeLists.txt:39-52` rejects any configured compiler that is not `icpx`/`dpcpp`) — stages such copies internally, so values are correct today and the defect is a latent portability hazard: a different SYCL runtime could fault or silently mishandle the call. The surrounding staging logic is sound and must not be disturbed: `staging_nbytes` is `logical_nbytes` rounded up to a 4-byte word with an explicit `std::overflow_error` guard (lines 207-212), the to-host path zeroes staging before the store kernel (`std::memset(staging, 0, logical_nbytes)`, line 225), and only `logical_nbytes` bytes move in each direction so pad bytes are never copied. The device leg is already clean: `launch_view_transfer` (lines 131-195) reads and writes only the shared staging and the tensor storage, and `SyclTensor` rejects a storage address the context cannot classify as USM at construction (`src/sycl/device.cpp:140-151`, `sycl::get_pointer_type(...) == sycl::usm::alloc::unknown` throws). These two host-pointer operands are the only `queue::memcpy` calls in `src/sycl`.

## Scope

- **Modify:** `src/sycl/copy.cpp` — `synchronous_transfer` only. Replace the two host-leg `queue::memcpy` statements with host-side `std::memcpy` on the calling thread:
  - from-host branch: `std::memcpy(staging, source.data(), logical_nbytes);` before `launch_view_transfer`. The host write completes before the first kernel submission; shared-USM coherence at the kernel boundary makes it visible to the device — the same consistency model the adjacent `std::memset(staging, ...)` on line 225 already relies on for the to-host direction.
  - to-host branch: after the staging kernel's `queue.wait_and_throw()` (line 228), `std::memcpy(destination.data(), staging, logical_nbytes);` replaces `queue.memcpy(destination.data(), staging, logical_nbytes).wait_and_throw();`. The kernel wait already guarantees the device writes are visible to host access of the shared allocation, so the synchronous-at-return contract is preserved.
  - No other statement in the function changes: the per-call in-order `sycl::queue` construction, the `std::overflow_error("SYCL transfer staging size overflows")` guard, the word-rounded `staging_nbytes` computation, `sycl::malloc_shared` with the null → `std::bad_alloc` check, the `std::memset` zeroing, both `launch_view_transfer` calls, the `queue.wait_and_throw()` calls, the catch-path cleanup (`queue.wait_and_throw()` and `sycl::free(staging, context)` inside nested try/catch, rethrow), and the success-path `sycl::free(staging, context)` stay exactly as they are. `<cstring>` is already included (line 9).

Callers are unaffected and must not be edited: `region_from_host`/`region_to_host` (`src/sycl/copy.cpp:438-452`) and the `SyclTensor::region_from_host`/`region_to_host` overrides (`src/sycl/device.cpp:163-177`) keep their signatures, synchronous behavior, and exception propagation.

## Implementation references

- **Modify:** `src/sycl/copy.cpp` — `synchronous_transfer`, the two `queue::memcpy` statements at lines 220 and 229-230.
- **Read:** `src/shared/standard_tiled_copy.inl:381-438` — `detail::synchronous_transfer_impl`, the established CUDA/ROCm analogue: kernels move data only between the staging buffer and device storage while the host leg is a host-side copy through staging (`Policy::copy_from_host`/`Policy::copy_to_host`). The SYCL fix is the same shape with the runtime-agnostic host copy; it does not adopt the pooled pools (that is AR-001).
- **Read:** `src/sycl/device.cpp:140-151` — the constructor's `sycl::get_pointer_type` USM-validity rejection. This already guarantees the `storage` operand of `launch_view_transfer` is USM-valid; do not duplicate any pointer-type check inside `synchronous_transfer`.
- **Tests:** `test/sycl/test_sycl_conformance.cpp:306-313` — "SYCL conformance: storage oracle covers every leaf width and padded shape" observes complete native storage through host transfers in both directions via `SyclStorageOracle`; `:331-337` ("transfer failures keep metadata and ownership") pins the wrong-byte-count rejections; `:355-361` runs the full shared suite. The existing cases are the regression coverage; no new test is added because DPC++ cannot expose this hazard at runtime.

## Requirements

1. `synchronous_transfer` contains no `queue::memcpy` (or other SYCL runtime data-movement call) whose operand derives from a caller host span. `source.data()` and `destination.data()` appear only as `std::memcpy` operands on the calling thread; host-span bytes cross into and out of USM memory only through host `std::memcpy`.
2. The from-host `std::memcpy(staging, source.data(), logical_nbytes)` executes before the first kernel submission of the transfer, and the to-host `std::memcpy(destination.data(), staging, logical_nbytes)` executes after the staging kernel's `queue.wait_and_throw()` completes. `copy_from_host`/`copy_to_host` remain synchronous at return per parent spec §5.4.
3. Exactly `logical_nbytes` bytes move in each direction; the word-rounded `staging_nbytes` allocation, the `std::overflow_error` staging-size guard, the `std::memset(staging, 0, logical_nbytes)` zeroing before the store kernel, and pad-byte exclusion (host-buffer padding is neither read nor written) are preserved unchanged.
4. Error categories and cleanup are unchanged: `std::overflow_error` for staging-size overflow, `std::bad_alloc` for a null `sycl::malloc_shared`, SYCL exceptions from kernel submission or wait propagate through the existing catch path, and `sycl::free(staging, context)` runs exactly once on success and at most once on the failure path. The host copies add no new failure mode.
5. Only `src/sycl/copy.cpp` changes: CPU, CUDA, ROCm, and TTNN transfer paths, all public headers, and the shared harness under `test/backend/` are untouched.

## Non-goals

- Pooling the per-call `sycl::malloc_shared` staging, replacing it with device staging plus a device-owned transfer queue, or any other queue/staging architecture change: the AR-001 SYCL convergence change (Order 60) owns those and must preserve this host-`std::memcpy` staging contract.
- Adopting the shared word-oriented copy kernels or deleting the per-plane launches (`launch_view_transfer` stays as is): AR-001/PF-002/46-PF-006 territory.
- `SyclQueue`/tensor lifetime hardening (`55-ST-001-harden-sycl-queue-lifetimes`), SYCL coexistence wiring (`53-CC-003-add-sycl-backend-coexistence`), the async-copy storage-oracle wiring (review NT-002), or any new conformance harness code.
- A `sycl::malloc_host` staging buffer or a second staging allocation: the review offered `malloc_host` staging or host `std::memcpy` as alternatives; host `std::memcpy` through the existing shared staging is the selected smallest design, and adding `malloc_host` would add an allocation and cleanup for no contract gain.
- Supporting non-DPC++ SYCL runtimes or changing the compiler gate (`CMakeLists.txt:39-52`): the fix removes the contract hazard; it does not add a new supported runtime.

## Acceptance criteria

- [ ] Source audit: `grep -n "memcpy" src/sycl/copy.cpp` shows the two host-leg moves as `std::memcpy`/`std::memset` only; no `queue.memcpy` call remains anywhere under `src/sycl`, and `source.data()`/`destination.data()` never appear as operands of a SYCL runtime call.
- [ ] On SYCL hardware, the storage-oracle host-transfer case passes unchanged: complete native storage observed through `SyclStorageOracle` matches the independent standard-tiled encoding for every leaf width and padded shape in both transfer directions, proving bytes, layout, tail-bit canonicality, and pad-byte exclusion are intact.
- [ ] Transfer-error conformance passes unchanged: wrong-size `copy_from_host`/`copy_to_host` spans still throw `std::invalid_argument` before any write, and sentinel bytes around the rejected writes are preserved.
- [ ] Host-transfer round trips on sub-byte leaf widths (including I4) return bytes identical to the seeded pattern through the oracle comparison, confirming the word-rounded staging and tail-bit zeroing behavior survived the cutover.
- [ ] The full SYCL suite (`iom_sycl_smoke_tests`, `iom_sycl_conformance_tests`) passes on SYCL hardware with unchanged case coverage, and the common `iom_tests` suite passes unchanged.

## Verification

All SYCL execution goes through `.agents/skills/remote-development` with the `sycl` profile from `.remote-hosts.conf` (DPC++ via `source /opt/intel/oneapi/setvars.sh`); CPU-only evidence is not a substitute for this backend change. Use a unique task id and exclusive device access (`flock /tmp/agent-gpu0.lock`) for the focused case:

1. `.agents/skills/remote-development/scripts/remote-sync sycl <task-id>`
2. `remote-exec sycl <task-id> 'cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF'` — the root `CMakeLists.txt` DPC++ gate must accept the configure.
3. `remote-exec sycl <task-id> 'cmake --build build/sycl --target iom_sycl_conformance_tests iom_sycl_smoke_tests -j'`
4. `remote-exec sycl <task-id> 'ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"'` — all cases pass, including the storage-oracle, transfer-error, and full-suite cases listed under Implementation references.
5. Focused host-transfer observation under exclusive access: `remote-exec sycl <task-id> 'flock /tmp/agent-gpu0.lock ./build/sycl/test/iom_sycl_conformance_tests -tc="*storage oracle*"'` — both oracle cases pass, exercising both host-transfer directions against full native storage.
6. `remote-exec sycl <task-id> 'ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"'` — the common/CPU suites are untouched.
7. Static audit of the source invariant in the first acceptance criterion.
8. `.agents/skills/remote-development/scripts/remote-clean sycl <task-id>`

Per the review's verification method: DPC++ conformance unchanged is the required and sufficient hardware evidence on the currently supported runtime; if a second SYCL runtime is ever supported, the same conformance suite must be run there before relying on the portability fix.

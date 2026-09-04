# Observe Full Accelerator Storage After Queued Copies

**Order:** 38
**Priority:** P2 — required cross-backend conformance coverage for the physical write-scope contract
**Blocked by:** 22-NT-001-add-accelerator-storage-oracles
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-001`
**Review severity:** medium
**Review verification:** verified, confidence 85

## Outcome

Prerequisite task `22-NT-001-add-accelerator-storage-oracles` already supplies the shared `AcceleratorStorageOracle` plus the CUDA, ROCm, and TTNN native oracle drivers that this task reuses. With those in place, accelerator copy conformance must prove both logical results and physical write scope: after every queued `DeviceOps::copy` case completes, the complete destination owner allocation equals its seeded storage with only the destination view window replaced by the source view's logical bytes, while the complete source owner allocation remains byte-for-byte at its seed.

## Current failure

`run_async_copy_conformance` in `test/backend/backend_conformance_copy_storage.hpp` checks only `require_logical_bytes` on the destination view. The existing `AcceleratorStorageOracle` is exercised by `run_storage_oracle_conformance` only for synchronous `copy_from_host` transfers and never observes storage after `DeviceOps::copy`. Consequently an accelerator kernel that writes padded tiles, overshoots a launch-chunk tail, or modifies planes outside the destination window can pass all current shared copy cases. CPU storage behavior is the only full-allocation oracle coverage.

## Scope

- This task depends on prerequisite task `22-NT-001-add-accelerator-storage-oracles` having landed the shared `AcceleratorStorageOracle` and the CUDA, ROCm, and TTNN native oracle drivers; this task does not add or modify any oracle implementation, only consumes them.
- Extend the shared asynchronous copy scenario to use the prerequisite accelerator storage oracle around each queued `DeviceOps::copy` for every existing `CopyCase`, owner shape, and supported leaf type.
- Cover CUDA, ROCm, and TTNN through the native oracle drivers delivered by the prerequisite task; preserve the current CPU reference and candidate logical-value checks and the existing queue/wait behavior.

## Implementation references

- **Modify:** `test/backend/backend_conformance_copy_storage.hpp` — `run_async_copy_conformance`; add the oracle-backed queued-copy path alongside the existing reference/candidate copy loop without removing its logical assertions. Reuse only the oracle surfaces that prerequisite task `22-NT-001-add-accelerator-storage-oracles` provides (`set_owner_spec`, `seed`, `observe`); do not extend them here.
- **Read:** `test/backend/backend_conformance_oracle.hpp` — `AcceleratorStorageOracle::set_owner_spec`, `seed`, and `observe`, plus `encode_standard_tiled_storage` and `apply_standard_tiled_view`; these are the prerequisite-task surfaces this spec consumes, not modifies.
- **Read:** `test/backend/backend_conformance_copy_storage.hpp` — `run_storage_oracle_conformance` and `copy_cases_for`; follow its deterministic seeding, expected-storage construction, and mismatch reporting conventions.
- **Tests:** `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, and `test/ttnn/test_ttnn_conformance.cpp` — existing storage-oracle fixtures and asynchronous-copy conformance registrations established by prerequisite task `22-NT-001-add-accelerator-storage-oracles`; wire the extended shared scenario to each backend's native oracle instance using the established gate/observer pattern.

## Requirements

- For each supported type, owner shape, and `CopyCase`, create separate source and destination candidate owners and construct the same source/destination views used by the existing case. Set the oracle owner specification to the owner tensor specification before seeding or observing each owner.
- Generate and retain independent initial standard-tiled byte vectors for source and destination. Call `oracle.seed(candidate_source->view(), initial_source)` and `oracle.seed(candidate_destination->view(), initial_destination)` before queuing, which writes the full owner allocation through the view's native handle. Verify each seed by reading it back with `oracle.observe(candidate_source->view())` / `oracle.observe(candidate_destination->view())` and confirming equality with the corresponding initial vector.
- Submit exactly one queued `candidate_queue->copy(source_view, destination_view)` for the case and call `candidate_queue->wait(candidate_token)` before either observation. Do not observe device storage while the operation is pending.
- Build the destination expectation from its retained initial bytes by applying the source logical pattern in destination view coordinate order with the shared storage-model helper. Compare the oracle's full destination-owner observation against that expectation, including padding, untouched planes, tails, and packed sub-byte bits. Compare the full source-owner observation against its unchanged initial bytes.
- Keep the existing CPU reference queue submission, waits, and `require_logical_bytes` assertions. A physical-oracle mismatch must fail the case with the case label and identify the first differing byte using the existing oracle comparison diagnostics.
- Use the deliberately widened copy-kernel discriminator during accelerator verification: a temporary kernel variant that writes full padded tiles (including bytes outside the logical destination window) must be detected by the new full-storage assertion, while the unmodified kernel passes.
- Retain queue cleanup and observer completion semantics, and ensure every successful token is waited before queues or tensor owners are destroyed.

## Non-goals

- Do not change CUDA, ROCm, or TTNN copy kernels, indexing, staging, allocation, synchronization, or production implementation code.
- Do not replace or weaken the existing logical copy/reference checks, alter `CopyCase` definitions, or change the specified behavior for overlapping non-identical windows.
- Do not redesign `AcceleratorStorageOracle`, introduce a new storage layout model, or extend coverage to compute operations or host-transfer scenarios already handled by `run_storage_oracle_conformance`.

## Acceptance criteria

- [ ] Every accelerator `CopyCase` runs after independently oracle-seeding both owners, and after `wait` the full destination observation matches initial destination storage with exactly the modeled destination-window update while full source storage matches its initial bytes.
- [ ] Existing logical destination assertions and CPU reference comparisons remain active and passing for all supported types and copy shapes, including padded, strided, permuted, nested, and identical-window cases.
- [ ] On accelerator hardware, a deliberately widened copy kernel that writes full padded tiles causes the new conformance scenario to report an out-of-window byte mismatch; the production kernel passes.

## Verification

After prerequisite task `22-NT-001-add-accelerator-storage-oracles` has landed, run the queued-copy extension on every supported accelerator:

- `.agents/skills/remote-development/scripts/remote-sync cuda nt-001-queued-copy-oracle`
- `.agents/skills/remote-development/scripts/remote-exec cuda nt-001-queued-copy-oracle 'cmake -S . -B build -DCUDA_ENABLED=ON && cmake --build build --target iom_cuda_conformance_tests'`
- `.agents/skills/remote-development/scripts/remote-exec cuda nt-001-queued-copy-oracle 'ctest --test-dir build -R iom_cuda_conformance_tests --output-on-failure'`
- `.agents/skills/remote-development/scripts/remote-sync rocm nt-001-queued-copy-oracle`
- `.agents/skills/remote-development/scripts/remote-exec rocm nt-001-queued-copy-oracle 'cmake -S . -B build -DROCM_ENABLED=ON && cmake --build build --target iom_rocm_conformance_tests'`
- `.agents/skills/remote-development/scripts/remote-exec rocm nt-001-queued-copy-oracle 'ctest --test-dir build -R iom_rocm_conformance_tests --output-on-failure'`
- `.agents/skills/remote-development/scripts/remote-sync ttnn nt-001-queued-copy-oracle`
- `.agents/skills/remote-development/scripts/remote-exec ttnn nt-001-queued-copy-oracle 'cmake -S . -B build -DTTNN_ENABLED=ON && cmake --build build --target iom_ttnn_conformance_tests'`
- `.agents/skills/remote-development/scripts/remote-exec ttnn nt-001-queued-copy-oracle 'ctest --test-dir build -R iom_ttnn_conformance_tests --output-on-failure'`
- Run the focused CUDA, ROCm, and TTNN conformance targets on their configured accelerator hosts. Repeat with a deliberately widened copy kernel that writes complete padded tiles; the baseline must pass and the widened variant must fail at the first out-of-window destination byte while source storage remains checked unchanged.

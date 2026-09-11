# Split outstanding-work registry headers

**Order:** 10
**Priority:** P0 — preserves the registry umbrella/public transitive contract while establishing responsibility boundaries
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

`include/iom/detail/outstanding_work_registry.hpp` remains the stable, small umbrella include for the complete outstanding-work registry detail contract. Its definitions are factored into four responsibility-complete headers without changing any type, symbol, signature, capability, error category, validation order, synchronization, ownership, lifetime, cleanup order, or observable registry behavior. Existing users that include the umbrella continue to see every existing `iom::detail` name transitively; users are not migrated to new include paths by this task.

The planned headers are:

- `include/iom/detail/fence.hpp`
- `include/iom/detail/outstanding_work_registry_core.hpp`
- `include/iom/detail/outstanding_work_cleanup.hpp`
- `include/iom/detail/workspace_registry.hpp`

Every resulting production header, including the retained umbrella, is at most 499 physical lines after normal formatting.

## Scope

Retain the umbrella at `include/iom/detail/outstanding_work_registry.hpp` as the compatibility and transitive-availability boundary. It should contain only the required dependency-safe includes (and no duplicate definitions), with `#pragma once` and the existing namespace contract preserved.

`include/iom/detail/fence.hpp` owns the foundational fence contract from the current header lines 26–193:

- fence storage constants and the existing `EntryId`, `QueueId`, and `EntryState` foundational names;
- `FenceResult` and its success/pending/failed factories;
- `Fence`, including its inline storage, copy/move/destroy operations, invocation, boolean conversion, and static assertions;
- `FenceCaptureOps` and its capture lifetime operations;
- invalidated-fence invocation and construction helpers.

`include/iom/detail/outstanding_work_registry_core.hpp` owns the registry and non-workspace outcomes from current lines 196–199 and 327–599, plus the binary outcome portion of current lines 724–793:

- `EntryRegistration`;
- `OutstandingWorkRegistry`, its entry/index types, map operations, validation, invalidation, exact-address removal, and private mutex-protected helpers;
- `release_or_quarantine`, `SequenceOutcome`, and `release_or_invalidate_entries`;
- `BinaryEntryRegistration`, `BinaryOwnerRegistration`, and binary outcome-release helpers.

The `RegistryState`-dependent binary registration routine is kept with `workspace_registry.hpp` (the dependency-safe binary/workspace carve-out), while the binary result types and outcome semantics remain in the core header.

`include/iom/detail/outstanding_work_cleanup.hpp` owns current lines 201–325:

- `CleanupAction`;
- `AllocatorCleanupAction`;
- `Quarantine`, including reverse-order cleanup, failure retention, repeated draining, mutex coverage, and destruction behavior.

`include/iom/detail/workspace_registry.hpp` owns the workspace and per-device state from current lines 611–920, with the source-contract carve-out for binary helpers:

- `WorkspaceLease`, `WorkspaceLeaseState`, and `RegistryState`;
- registry ID/queue-ID allocation and all `RegistryState`-dependent copy/binary registration helpers;
- workspace lease range overlap, acquisition, completion, and retention helpers;
- the workspace registration transaction and rollback behavior.

Use the fewest files above; do not introduce numbered shards, compatibility aliases, forwarding wrappers, a replacement registry, or any public include migration. Do not add these detail headers to a public API list unless an existing CMake convention requires listing them. The current `outstanding_work_registry.hpp` is not listed in the public API/header list.

## Implementation references

- `include/iom/detail/outstanding_work_registry.hpp:26-193` — fence constants, fence value type, capture operations, and invalidated-fence helpers.
- `include/iom/detail/outstanding_work_registry.hpp:196-599` — entry registration, registry maps/indices, release/quarantine decision, sequence outcome, and entry release/invalidation.
- `include/iom/detail/outstanding_work_registry.hpp:611-920` — workspace leases, `RegistryState`, ID allocation, registration, lease acquisition/completion, and retention.
- `include/iom/detail/outstanding_work_registry.hpp:724-793` — binary registration/outcome helpers; split result/outcome definitions from `RegistryState`-dependent registration so includes remain acyclic.
- `include/iom/iom.hpp:20` — public umbrella consumer and transitive `detail` names used by `DeviceOps`/`BinaryRequest`.
- `src/cpu/device.cpp:17` — CPU registry consumer.
- `src/cuda/copy.hpp:19`, `src/rocm/copy.hpp:20` — CUDA/ROCm registry consumers.
- `src/shared/event_ring.hpp:21`, `src/shared/gpu_queue.hpp:41` — shared GPU event/queue consumers.
- `src/sycl/copy.hpp:10` — SYCL registry consumer.
- `src/ttnn/registry_state.hpp:10` — TTNN cleanup-action consumer.
- `test/test_iom.cpp:2874-3989` — fence, registry, cleanup/quarantine, binary registration, and workspace lease behavior proof.
- `docs/changes/005-split-big-files/spec.md:30-34,116-120,126-158` — universal factoring invariants, public-header contract, build integration, and verification strategy.

## Requirements

1. Preserve the existing namespace `iom::detail`, all existing names, signatures, inline behavior, exception types/messages, validation order, and static assertions. A caller including `outstanding_work_registry.hpp` must compile without adding any new include.
2. Use a one-way include graph with no cycles: `fence.hpp` is foundational; `outstanding_work_cleanup.hpp` depends only on the allocator contract and its standard-library requirements; `outstanding_work_registry_core.hpp` depends on `fence.hpp`; `workspace_registry.hpp` depends on `fence.hpp`, `outstanding_work_cleanup.hpp`, and `outstanding_work_registry_core.hpp`; the umbrella includes the four shards. No shard may include the umbrella, and no definition may be duplicated across shards.
3. Keep `Fence`'s inline storage and capture copy/move/destroy operations allocation-free and `noexcept` as before. Preserve invalidated fences as repeatable failed outcomes with the existing error category and message.
4. Keep all `OutstandingWorkRegistry` map/index operations under the existing registry mutex. Preserve exact-address matching, duplicate/missing-ID validation, queue-specific invalidation, transactional registration rollback, and the distinction between live and invalidated entries.
5. Preserve cleanup/quarantine semantics exactly: cleanup runs in reverse insertion order, failed or incomplete actions remain retained, repeated drains are safe, cleanup failures are retained on the action, and mutex coverage and destructor release behavior do not change.
6. Preserve the completion proof boundary. `release_or_quarantine` may release storage only when every relevant fence proves success; otherwise it must create cleanup/quarantine state and remove registry entries in the existing order. Sequence and binary outcomes must retain failure history and never convert unknown, pending, invalidated, or failed completion into proof.
7. Preserve `RegistryState` ownership and mutex coverage. ID counters, queue IDs, workspace lease records, registry entries, and quarantine state must remain per-device state; `allocation_mutex` must continue to cover allocation/registration/lease transactions without holding an allocator lock across submission, waits, callbacks, or drains.
8. Preserve workspace lease invariants: owner identity is exact and opaque; malformed, empty, misaligned, overflowed, zero-sequence, zero-queue, and empty-fence inputs fail with the existing categories; overlapping ranges for one owner exhaust resources; disjoint ranges and different owners remain independent; acquisition is all-or-nothing; completion proof releases the lease and entry; non-proof invalidates/quarantines and retains the range; retention queries prevent unsafe reuse.
9. Keep binary outcome types/helpers in the core shard while placing helpers that access `RegistryState` (including binary registration) in the workspace shard. This is an implementation boundary only: binary owner deduplication, maximum-owner validation, entry allocation, rollback, and release/invalidation behavior must remain byte-for-byte equivalent in observable effect.
10. Preserve public/transitive include behavior for `iom.hpp`, CPU/GPU/SYCL/TTNN consumers, and the registry tests. Do not change public factory headers, test source lists, capabilities, backend switches, global registries, allocation behavior, synchronization design, or ODR/linkage behavior.
11. Keep the umbrella and each new detail header at or below 499 physical lines. Do not add a permanent line-count test. If the build system has a convention for tracking private/detail headers, follow that existing convention only; do not add the detail shards to a public API list merely because they are new files.

## Non-goals

- Changing registry, fence, cleanup, quarantine, sequence, binary, or workspace semantics.
- Renaming, removing, deprecating, aliasing, or shimming any existing detail symbol or include path.
- Migrating consumers from the umbrella to shard-specific includes.
- Adding a backend switch, global registry, cross-backend abstraction, new capability, allocator behavior, synchronization redesign, or test/API surface.
- Adding new tests or changing test source lists; existing tests remain the behavior proof.
- Modifying public factory headers, unrelated production files, CMake target composition except where an established header-list convention strictly requires it, or documentation outside this mini-spec.

## Acceptance criteria

- `outstanding_work_registry.hpp` is a small umbrella that transitively exposes every pre-existing registry/fence/cleanup/workspace name and contains no duplicate implementation.
- The four planned shards have the exact responsibility boundaries above, with an acyclic dependency graph and no include of the umbrella from a shard.
- Existing public and private consumers compile unchanged, including `iom.hpp`, CPU/GPU/SYCL/TTNN headers, and `test/test_iom.cpp`.
- Registry mutex coverage, reverse cleanup, failure retention, invalidated-fence behavior, sequence/binary outcome retention, workspace lease completion-proof release, quarantine, and ODR behavior are unchanged.
- No public API/header list is changed unless an existing repository convention requires it; no public include migration is made.
- The umbrella and every touched/new production header are at most 499 physical lines after normal formatting, and every moved definition exists exactly once.

## Verification

All commands below are proposed instructions for the future implementer and are **not run by this task**:

- Configure CPU/core coverage: `cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF`.
- Build the relevant CPU/core targets: `cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests`.
- Run registry and CPU behavior coverage: `ctest --test-dir build/split-cpu --output-on-failure -R '^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'`.
- Compile and link a temporary translation unit using project compiler settings that includes both `iom/iom.hpp` and only `iom/detail/outstanding_work_registry.hpp`, instantiates representative `Fence`, `OutstandingWorkRegistry`, `Quarantine`, `RegistryState`, and `WorkspaceLease` names, and links against `libiom`; do not commit the temporary source.
- Perform a one-time scoped line-count check over `include/iom/detail/outstanding_work_registry.hpp`, `fence.hpp`, `outstanding_work_registry_core.hpp`, `outstanding_work_cleanup.hpp`, and `workspace_registry.hpp`, asserting each is `<=499` physical lines. The scanner is verification only and must not become a permanent test.

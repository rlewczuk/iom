# Publish one TTNN binary completion proof

**Order:** 04
**Priority:** P0 — restores one completion owner and removes two redundant first-wait drains
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** ST-001
**Review area:** C++/GPU stability
**Review severity:** medium
**Review verification:** strongly-supported, confidence 93
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `src/ttnn/copy.cpp` — `ttnn_detail::binary_planes`; `src/ttnn/device.cpp` — `TtnnQueue::complete_task`, `finish_locked`, and `fence_through_sequence`; common `DeviceOps::wait`

## Outcome

Each successful TTNN binary request has one completion owner: the required mesh drain is published to the common completion marker, completion does not drain it again, and the first or repeated successful waits add no drain. Failed or partially submitted work retains its failure and owner/staging protection until a real completion proof.

## Current problem

A successful `binary_planes` request uploads output planes and directly calls `mesh_command_queue(0).finish()` before returning. `TtnnQueue::complete_task` then sees `native_work_submitted` and calls `finish_native` again, but does not advance `last_finished_seq_`. The first common `DeviceOps::wait` consequently enters `fence_through_sequence` and calls `finish_native` a third time because its marker is still below the binary token. Repeated successful waits stop after that marker is set; this is not a claim that all repeated waits finish. The first successful wait therefore currently adds the third finish after the binary path's direct drain. Existing root TTNN smoke/conformance/coexistence passed, but no binary finish-count trace or candidate-specific fault injection ran.

## Scope

- Make exactly one state owner publish successful binary drain completion to the common wait fence while preserving the required output-plane drain before staging release.
- Suppress duplicate completion/fence drains after a successful proof, but retain a retry/quarantine path when the final drain throws or completion is unproven.
- Keep TTNN API mutex, native per-plane staging/storage, positive retained failures, owner registry, and copy batching distinctions unchanged.

## Implementation references

- **Modify:** `src/ttnn/copy.cpp` — `binary_planes`; return explicit native-drained state or use the existing completion publication mechanism after its required direct drain.
- **Modify:** `src/ttnn/device.cpp` — binary `complete_task`, `finish_locked`, and `fence_through_sequence`; make one completion owner update the marker and avoid repeated successful drains.
- **Read:** `src/ttnn/device.cpp` — copy completion path that updates `last_finished_seq_`; reuse its proven marker/retirement convention without merging copy and binary storage.
- **Read:** `src/iom.cpp` — `DeviceOps::wait`; preserve repeat-wait and retained-failure semantics.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` finish-count tests around copy completion and binary numerical/failure cases; add binary ADD/MUL/SUB/DIV count coverage.

## Requirements

- A successful binary native drain must publish completion exactly once to the sequence marker used by `fence_through_sequence`; `complete_task` and the first wait must not issue post-drain duplicate finishes.
- First and repeated waits after successful completion must not issue another finish. A completion/final-drain exception before proof retains the failure and quarantine/owner protection; a finish fault armed only after the published marker must not be consumed by later waits or retroactively fail the token.
- Keep one required direct mesh drain before releasing upload leases, preserve TTNN's API mutex and native staging, and retain positive-token repeated failure behavior.

## Non-goals

- Do not remove TTNN native 32x32 storage, host scalar arithmetic, required staging/downloads, API mutex, owner registry, native copy batching, or the output drain needed before lease release.
- Do not change numerical or validation semantics, add a scheduler/cache, or generalize backend synchronization across CUDA/ROCm/SYCL.
- Do not create a separate PF task; PF-001 is merged into this stability root.

## Acceptance criteria

- [ ] For successful ADD, MUL, SUB, and floating DIV, a finish trace shows one required binary completion drain, no second drain in `complete_task`, and no third drain in the first wait; repeated waits add none after the marker.
- [ ] Logical outputs, owner lifetime, staging release, and repeated successful waits remain correct for single- and multi-plane requests.
- [ ] A finish failure before completion proof remains a positive repeatable retained failure and keeps owners/staging protected; successful completion later releases exactly once.
- [ ] A finish fault armed only after the successful completion marker remains unconsumed by repeated waits and cannot retroactively fail the completed token.

## Verification

- `(remote-development: TTNN host)` instrument `mesh_command_queue(0).finish()` and run one multi-plane ADD, MUL, SUB, and floating DIV, wait once, then wait repeatedly; expect one required drain and zero later drains with correct independent-oracle output.
- Exercise the existing finish-failure and partial-submit seams before and after the first-wait marker; assert repeatable retained failure and owner/staging quarantine until proof, without treating a post-marker injection as required to fail.
- Run TTNN smoke/conformance/coexistence targets. Root-supplied TTNN smoke/conformance/coexistence passed; no candidate-specific finish trace, fault injection, sanitizer, profiler, or benchmark has run.

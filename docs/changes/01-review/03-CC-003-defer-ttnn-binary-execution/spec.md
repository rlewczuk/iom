# Defer TTNN binary execution until after token publication

**Order:** 03
**Priority:** P0 — public async queue contract is violated
**Blocked by:** None
**Review source:** `cpp-inference-contract-correctness` — whole-codebase review, branch `main`, HEAD `cb88619865ff9bcda0a4e3bac4bae4bd4bf510a8` (`Move cpp-inference-review skill to .omp`), baseline `origin/main`; clean at scope capture; upstream `origin/main` ahead 4/behind 0
**Finding:** CC-003
**Review area:** Contract & correctness
**Review severity:** high
**Review verification:** strongly-supported, confidence 98
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `src/ttnn/queue.cpp:144-250` — `TtnnQueue::binary_impl`/`execute`; `include/iom/detail/staged_worker.hpp:59-75` — synchronous callback; `src/ttnn/binary.cpp:39-50,91-156,169-233` — host fallback

## Outcome

TTNN binary submission publishes an accepted task and returns its token before blocking plane downloads, host arithmetic, uploads, and mesh finish; worker-side completion remains FIFO, API-mutex serialized, lease-safe, and repeatable through `wait(token)`.

## Current problem

The public contract states binary work is in-order asynchronous and permits inline completion only for CPU (`docs/BACKEND_CONTRACT.md:415-417`). On the normal TTNN path, `TtnnQueue::binary_impl` calls `worker_.submit_copy` from its dispatch callback (`src/ttnn/queue.cpp:144-172`). `StagedWorker::submit_copy` invokes `callbacks_.execute(*staged)` synchronously before publishing the task or returning (`include/iom/detail/staged_worker.hpp:59-75`). TTNN binary `execute` then locks the API mutex and `binary_planes` performs blocking source-plane downloads, the complete host scalar loop, output uploads, and `finish` (`src/ttnn/queue.cpp:177-220`; `src/ttnn/binary.cpp:39-50,91-156,169-233`). Thus `queue->add/mul/sub/div` can absorb the complete fallback before returning its positive token; failure and completion are observed at the wrong boundary. Existing tests wait immediately and do not assert publication-before-execution. The source/contract defect is deterministic; TTNN timing and hardware behavior were not run.

## Scope

- Change only the TTNN binary ownership boundary so `binary_planes` runs after the accepted immutable task is published to the worker, or use the smallest equivalent TTNN-specific deferred handoff.
- Preserve FIFO admission and sequence/token semantics, API mutex serialization, owner registration, workspace and staging leases, one covering `finish`, retained/repeatable failures, output arithmetic/representation, and existing TTNN copy/other backend behavior.

## Implementation references

- **Modify:** `src/ttnn/queue.cpp` — `TtnnQueue::binary_impl` and binary `execute`; defer the binary callback's full work without changing public operation signatures.
- **Read/modify only if required:** `include/iom/detail/staged_worker.hpp` — `StagedWorker::submit_copy`; any change must be narrowly TTNN-specific and must not globally alter CPU inline semantics or other StagedWorker users.
- **Read:** `src/ttnn/binary.cpp` — `binary_planes`; retain blocking plane conversion, host scalar codec, output uploads, padding, and finish ordering inside worker-owned execution.
- **Read:** `src/ttnn/queue_copy.cpp` — asynchronous native-copy counterpart and completion ownership.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp:1897-2210` — binary values, ordering, retained failure, and finish cases; shared asynchronous queue conformance; add a deterministic barrier/fault seam around worker-side binary execution.

## Requirements

- Publish/enqueue an immutable TTNN binary task before executing `binary_planes`; the public admitted call must return its positive token without running the complete host fallback on the caller thread. Do not globally change `StagedWorker` or CPU inline behavior.
- Keep TTNN API-mutex protection around native plane access and preserve FIFO order. Keep source/output owner registrations, workspace/staging leases, and native plane lifetimes until completion proof; do not expose reusable buffers early.
- Keep the existing blocking downloads, scalar binary arithmetic, output plane writes, one mesh `finish`, cleanup, and exception-to-retained-token mapping on the worker-side completion path. A failure must remain waitable and repeatable with the existing category/OID semantics.
- Use a deterministic barrier/fault seam rather than timing-only assertions to prove publication before execution. The seam must hold worker-side binary execution, permit a second independent submission to return, then release and validate through `wait`.
- Do not globally change `StagedWorker` publication ordering for copy, CPU, SYCL, or other TTNN operations; the correction is TTNN-binary-specific.

## Non-goals

- Do not redesign TTNN native per-plane storage, host scalar codec, API mutex policy, binary capabilities, public signatures, common validation, broadcasting/alias rules, copy semantics, or queue capacity.
- Do not optimize coordinate allocation, claim a separate performance root, or require a timing percentage; any secondary submission-latency observation is owned by this contract task.
- Do not claim prefill/decode/KV-cache or model-level overlap improvement; no production model executor exists in this tree.

## Acceptance criteria

- [ ] With the deterministic worker barrier held, TTNN `add`, `mul`, `sub`, and `div` return accepted tokens before `binary_planes` passes the barrier; an independent second submission can be issued without the first caller running its fallback, and releasing the barrier followed by `wait(token)` yields oracle-correct output.
- [ ] FIFO ordering, API mutex and one-finish behavior, owner/workspace/staging lease retention, output padding/representation, repeated waits, and injected failure retention remain unchanged; no public API or CPU inline semantics change and no timing magnitude is used as acceptance.

## Verification

- `ctest --test-dir <build> --output-on-failure -R '^(iom_ttnn_conformance_tests|iom_backend_conformance_cpu_tests)$'` with the focused deterministic barrier/fault cases; run TTNN on the configured remote host.
- Hold execution at the worker-side seam, submit two operations, assert first submission returns before release, then release and wait both in FIFO order; repeat with an injected binary failure and repeated waits. The supplied root context passed only local CPU targets (3/3); TTNN remote sync attempts timed out before build, so no TTNN runtime result is claimed.

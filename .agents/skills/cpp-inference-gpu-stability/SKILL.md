---
name: cpp-inference-gpu-stability
description: Independently review C++ inference-engine ownership, asynchronous GPU lifetime, synchronization, visibility, concurrency, cleanup, errors, integer safety, and multi-device state, then emit direct remediation subtasks. Also serves as Area 2 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [backend focus optional]"
---

# C++ / GPU Stability Review

Treat asynchronous accelerator execution as a first-class correctness model. C++ lexical scope does not prove device work completed.

Read `.agents/cpp-review/references/review-process.md`, `.agents/cpp-review/references/finding-rubric.md`, `.agents/cpp-review/references/cpp-gpu-stability.md`, `.agents/cpp-review/checklists/common.md`, and every affected backend checklist.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope and return only `ST-###` candidate packets. Do not write tasks before cross-area synthesis.
- **Standalone:** resolve scope and optional task destination, perform this stability pass, then invoke `cpp-inference-review-synthesis` to adversarially verify and directly materialize tasks.

In selected-commit mode, accept only root causes introduced or materially exposed/worsened by the target.

## Review order

### 1. Resource ownership

For each changed or central buffer, allocation, stream/queue, event/fence, module/kernel, graph, descriptor, command object, mapped region, cache entry, or context/device handle, establish:

- owner versus borrower;
- copy/move behavior;
- acquisition and release protocol;
- partial-construction cleanup;
- destruction thread/context/device assumptions;
- whether destruction synchronizes or can fail.

RAII must match host ownership, but it does not by itself prove safe release after asynchronous device use.

### 2. Asynchronous lifetime

For every enqueue, copy, kernel, graph, or callback:

- Which host/device resources remain referenced after the API returns?
- Which completion establishes safe reuse/destruction?
- Can allocator reuse race with in-flight work?
- Can host staging memory disappear early?
- Can callbacks capture destroyed state?
- Do command/descriptor objects meet backend submission-lifetime rules?

### 3. Ordering, synchronization, and visibility

Trace producer → dependency → consumer. Check:

- missing, misplaced, or wrong-device event/fence dependencies;
- implicit default/global stream assumptions;
- whole-device synchronization where queue/event ordering suffices;
- synchronization hidden in destructors or hot accessors;
- incomplete visibility/barrier semantics;
- graph capture incompatibility;
- invalid cross-device handle/event use.

Missing ordering is correctness risk. Excess ordering is a performance consequence of the same root when one mechanism causes both; do not duplicate it.

### 4. Concurrency and multi-device state

Inspect allocator pools, module/graph/capability caches, initialization flags, registries, queue pools, scratch arenas, and mutable globals. Verify:

- keys include every required backend/device/context/architecture fact;
- supported concurrent calls cannot race;
- current-device/context state is explicit and restored;
- peer access is capability checked with correct fallback;
- synchronization does not serialize unrelated devices.

### 5. Error propagation and cleanup

Verify:

- immediate enqueue errors are observed;
- deferred execution errors remain observable and correctly attributed;
- completed failures remain repeatable when the contract requires it;
- exceptions/status paths cannot bypass cleanup;
- partial state is not used after failure;
- cleanup failure does not mask the primary failure improperly.

### 6. Size and arithmetic safety

Check multiplication/addition/round-up before casts and narrowing for dimensions, strides, bytes, launch geometry, kernel indices, and backend parameter widths.

## Mandatory simplicity pass

Stability code becomes unsafe when ownership and state are represented repeatedly. Look for:

- multiple lifetime trackers for the same in-flight work;
- duplicated event/quarantine/release protocols across backends when semantics match;
- flags whose combinations represent invalid resource states;
- wrapper owners that do not add a release invariant;
- caches or registries duplicating backend/runtime state;
- cleanup paths copied with inconsistent failure behavior;
- synchronization added to compensate for unclear ownership instead of fixing ownership.

Prefer one explicit owner and one completion mechanism. Preserve backend-specific lifetime primitives where semantics differ.

## Candidate acceptance

Use IDs `ST-###` and the full common packet. State the exact asynchronous timeline, owner/reuse point, device/context, error path, current symbols/tests, remediation, and falsifier.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
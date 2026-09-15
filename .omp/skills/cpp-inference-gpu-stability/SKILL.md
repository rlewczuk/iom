---
name: cpp-inference-gpu-stability
description: Independently review C++ inference-engine ownership, asynchronous GPU lifetime, synchronization, visibility, concurrency, cleanup, errors, integer safety, and multi-device state, then emit direct remediation subtasks. Also serves as Area 2 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [backend focus optional]"
---

# C++ / GPU Stability Review

Treat asynchronous accelerator execution as a first-class correctness model. C++ lexical scope does not prove device work completed.

Before any repository work, read `skill://boss` first. Then read `.omp/cpp-review/references/review-process.md`, `.omp/cpp-review/references/finding-rubric.md`, `.omp/cpp-review/references/cpp-gpu-stability.md`, `.omp/cpp-review/checklists/common.md`, and every affected backend checklist.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope and return only `ST-###` candidate packets. This is candidate-only `boss-reviewer` execution at `@task`: do not resolve a new frontier, delegate recursively, invoke synthesis, write task files, or run validation gates.
- **Standalone:** run as the root orchestration for this area after applying Boss's root-model approval and warning-and-consent gate. Resolve scope and the optional task destination using the shared process, own the complete stability pass and acceptance, then invoke `cpp-inference-review-synthesis` to adversarially verify and directly materialize tasks. The skill cannot switch an already-running model; `@slow` remains preferred, `@csw-yoda` is automatically approved, and any other mismatch requires explicit user consent.

In selected-commit mode, accept only root causes introduced or materially exposed/worsened by the target.

## Task metadata

When a destination is supplied, task lifecycle controls belong exclusively to task_ctl-managed `task.yml`: generated remediation records use `type: impl`, `status: new`, assigned `order`, P0–P2 `priority`, canonical `blocked-by` IDs, and the parent `spec.md` as `source`. Use `.omp/csw/bin/task_ctl` CLI/API (`task_dir`, `get_task`, `set_task`, `list_tasks`) for paths, ordering, metadata, and dependencies; never parse or hand-write YAML. Keep all review evidence in `spec.md`.

## Boss routing for this area

Follow the canonical review process rather than restating it. The standalone invocation is owned by the running root under Boss's root-model consent policy, which resolves scope, accepts candidates, freezes the assignment table, and invokes synthesis; the orchestrated invocation is a candidate-only `boss-reviewer` leaf at `@task`. Neither mode may claim an already-running model was switched, and the leaf may not delegate, recurse, write tasks, or broaden discovery.

- Project agent `scout` at `@smol` handles broad ownership/lifetime inventory, enqueue/callback timeline collection, stream/event/fence dependency tracing, cache/device-state search, cleanup/error-path discovery, and size-arithmetic search. Use `boss-errand` at `@smol` only for atomic factual followups. Return exact `path:line`/symbol evidence, source facts versus inference, negative evidence, search coverage, and uninspected areas.
- `boss-reviewer` may inspect the bounded assigned source ranges and callpaths directly, alongside the scout evidence, to perform asynchronous reasoning and independent falsification; it returns candidate packets only.
- For a genuinely hard or disputed ownership/lifetime, synchronization/visibility, concurrency/device-context, deferred-error, or cleanup decision, the root may send compact packet **content** to `boss-advisor` at `@advisor`. The advisor is tool-free, packet-only, never delegates, and may answer `NEED EVIDENCE` with one exact question; agreement never verifies a claim.
- The root reads only cheap exact-known ranges when cheaper than another dispatch, not broad scans or whole-diff ingestion. Batch independent work and make no gratuitous calls.
- Workers skip builds, tests, benchmarks, formatters, and other validation; they propose exact gates. The root executes gates and mandatory synthesis. If delegation is unavailable, disclose routing/coverage limits and request permission before any materially costlier fallback.

### Cheap evidence assignments

Ask the `@smol` scout to map each relevant allocation, buffer, queue/stream, event/fence, module/kernel, graph, descriptor, mapped region, cache entry, and context/device handle to owner, borrower, acquisition, release, asynchronous uses, completion, error cleanup, and device identity; collect producer/dependency/consumer paths, callback captures, mutable cache/registry keys, and arithmetic-width sites with exact locations. Ask the `@task` reviewer to inspect only the assigned bounded ranges/callpaths and packet, reconstructing asynchronous timelines, reuse/destruction and visibility, concurrent-device state, deferred errors, and missing versus excess ordering. Escalate to the advisor only for a hard lifetime/synchronization interpretation, disputed device/context guard, or high-risk acceptance/remediation choice after evidence is complete; otherwise the root decides.

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
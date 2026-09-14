---
name: cpp-inference-contract-correctness
description: Independently review a C++ inference-engine scope for specification compliance, semantic/operator/tensor contracts, capability and fallback correctness, error behavior, compatibility, and duplicated contract enforcement; finish with direct remediation subtasks. Also serves as Area 1 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [operator/backend focus optional]"
---

# Contract & Correctness Review

Review **what the system promises and whether the implementation preserves it**. Ignore style unless it obscures a semantic defect or creates duplicated sources of truth.

Before any repository work, read `skill://boss` first. Then read `.omp/cpp-review/references/review-process.md`, `.omp/cpp-review/references/finding-rubric.md`, and `.omp/cpp-review/checklists/common.md`.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope/specification map; return only `CC-###` candidate packets to the orchestrator. This is candidate-only `boss-reviewer` execution at `@task`: do not resolve a new frontier, delegate recursively, invoke synthesis, write task files, or run validation gates.
- **Standalone:** run as the root orchestration for this area after applying Boss's root-model warning-and-consent gate. Resolve scope and the optional `docs/changes/...` destination using the shared process, own the complete area review and acceptance, then invoke `cpp-inference-review-synthesis`. The final deliverable is direct remediation subtasks, not a review report. The skill cannot switch an already-running model; `@slow` remains preferred, while an explicitly accepted mismatch may continue.

In selected-commit mode, inspect necessary surrounding code and backend counterparts but accept only defects introduced or materially exposed/worsened by the target commit.

## Task metadata

When a destination is supplied, task lifecycle controls belong exclusively to task_ctl-managed `task.yml`: generated remediation records use `type: impl`, `status: new`, assigned `order`, P0–P2 `priority`, canonical `blocked-by` IDs, and the parent `spec.md` as `source`. Use `.omp/csw/bin/task_ctl` CLI/API (`task_dir`, `get_task`, `set_task`, `list_tasks`) for paths, ordering, metadata, and dependencies; never parse or hand-write YAML. Keep all review evidence in `spec.md`.

## Boss routing for this area

Follow the canonical review process rather than restating it. The standalone invocation is owned by the running root under Boss's root-model consent policy, which resolves scope, accepts candidates, freezes the assignment table, and invokes synthesis; the orchestrated invocation is a candidate-only `boss-reviewer` leaf at `@task`. Neither mode may claim an already-running model was switched, and the leaf may not delegate, recurse, write tasks, or broaden discovery.

- Project agent `scout` at `@smol` handles broad specification inventory, changed-interface/callsite/counterpart search, capability/dispatch tracing, and test/error-path discovery. Use `boss-errand` at `@smol` only for atomic factual followups. Return exact `path:line`/symbol evidence, source facts versus inference, negative evidence, search coverage, and uninspected areas.
- `boss-reviewer` may inspect the bounded assigned source ranges and callpaths directly, alongside the scout evidence, to perform semantic reasoning and independent falsification; it returns candidate packets only.
- For a genuinely hard or disputed semantic, capability/fallback, compatibility, or operator/tensor decision, the root may send compact packet **content** to `boss-advisor` at `@advisor`. The advisor is tool-free, packet-only, never delegates, and may answer `NEED EVIDENCE` with one exact question; agreement never verifies a claim.
- The root reads only cheap exact-known ranges when cheaper than another dispatch, not broad scans or whole-diff ingestion. Batch independent work and make no gratuitous calls.
- Workers skip builds, tests, benchmarks, formatters, and other validation; they propose exact gates. The root executes gates and mandatory synthesis. If delegation is unavailable, disclose routing/coverage limits and request permission before any materially costlier fallback.

### Cheap evidence assignments

Ask the `@smol` scout to enumerate authoritative requirements, exact interfaces/callers, operator/tensor representation and ownership assumptions, capability predicates, dispatch/fallback/error paths, compatibility counterparts, and supported/unsupported tests. Ask the `@task` reviewer to inspect only the assigned bounded ranges/callpaths and packet, checking observable semantics, validation-before-mutation, shape/rank/dtype/layout/aliasing and overflow boundaries, and canonical contract ownership. Escalate to the advisor only for a hard semantic interpretation, disputed guard/counterpart, or high-risk acceptance/remediation choice after evidence is complete; otherwise the root decides.

## Establish the contract map

Identify:

- authoritative specification/design requirements;
- public and internal interfaces plus every affected caller;
- operator/graph transformation semantics;
- tensor representation and ownership rules;
- backend capability, dispatch, partitioning, and fallback;
- established error categories and compatibility guarantees;
- relevant behavioral tests and unsupported-case tests.

Separate required observable behavior from suggested implementation structure.

## Review questions

### Specification and observable behavior

- Are all mandatory requirements implemented?
- Do defaults, errors, feature gates, and compatibility behavior match the contract?
- Did an implementation detail accidentally become observable?
- Does validation happen before work or partial state mutation?
- Is unsupported input rejected or routed intentionally?

### Operator and tensor semantics

Check relevant assumptions about:

- shapes, ranks, zero/empty and dynamic dimensions;
- dtype and quantization metadata combinations;
- strides, non-contiguity, layout, tiling, and alignment;
- broadcasting and alias/in-place behavior;
- device placement and identity;
- output ownership and mutation;
- arithmetic overflow before allocation, indexing, or narrowing.

### Capability, dispatch, and fallback

Treat this as one contract:

```text
semantic operation
→ capability/support predicate
→ dispatch/partitioning
→ backend implementation constraints
→ fallback/error behavior
→ tests and benchmarks
```

Flag support predicates that overclaim implementation constraints or reject supported paths. Trace hidden fallback, representation conversions, and changed graph partitioning.

### Compatibility

Inspect effects on:

- other accelerators and CPU/reference paths;
- builds with each optional backend enabled or disabled;
- serialized models/caches/configurations when relevant;
- promised public ABI/API behavior.

## Mandatory simplicity pass

Contract logic is especially vulnerable to duplication. Look for:

- the same shape/dtype/layout condition enforced differently at multiple layers;
- capability facts duplicated between common dispatch and backend implementation;
- repeated fallback/error policy with divergent categories or messages;
- stored derived flags that can disagree with canonical tensor/device state;
- new caller-side special cases that belong in the lowest stable owner;
- parallel helpers that duplicate an established validation or capability mechanism.

Prefer one canonical contract owner. Do not centralize backend-specific constraints that materially differ.

## Candidate acceptance

Use IDs `CC-###` and the complete candidate packet contract. Include exact scope relation, current symbols/callers/tests, smallest structural remediation, acceptance seed, non-goals, and falsifier.

Do not emit a generic “missing test” candidate without an unprotected observable behavior. Do not turn multiple symptoms of one contract source-of-truth defect into separate candidates.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
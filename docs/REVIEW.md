# Running C++ Inference Reviews

Run reviews from the repository root with one `@slow` supervisor. Boss routes broad discovery to `scout @smol`, atomic lookups to `boss-errand @smol`, bounded specialist analysis to `boss-reviewer @task`, and only hard decisions to the tool-free `boss-advisor @advisor`.

## Configure and start OMP

Confirm the four model roles and any per-agent overrides before starting a review:

```bash
omp config get modelRoles --json
omp config get task.agentModelOverrides --json
omp config get task.agentAdvisor --json
```

The effective configuration should satisfy:

- `slow` is the capable orchestration model;
- `task` is the cheaper substantive review model;
- `smol` is the cheapest useful discovery/lookup model;
- `advisor` is the deep-reasoning model;
- `task.agentModelOverrides` does not unintentionally route `scout`, `boss-errand`, or `boss-reviewer` to a frontier model;
- `task.agentAdvisor` does not attach a passive advisor to cheap workers.

Start the root session on the configured slow role:

```bash
omp --model @slow
```

Then submit one of the `/boss` prompts below. Naming the review skill explicitly avoids ambiguous skill selection.

A review is read-only with respect to implementation and existing specifications. When a destination under `docs/changes/` is supplied, accepted findings are written as new remediation task specifications. Without a destination, the same self-contained tasks are returned in chat. The workflow never creates `review.md`.

## 1. Whole-codebase review

Use all five specialists through `cpp-inference-code-review`:

```text
/boss Use the cpp-inference-code-review skill to review the whole checked-out
codebase. Cover contract/correctness, GPU stability, backend simplicity,
numerical correctness/testing, and performance. Record HEAD, working-tree
changes, sampled versus exhaustive coverage, validation actually run, and
uninspected areas. Run mandatory cpp-inference-review-synthesis and return
implementation-ready remediation tasks. Do not modify implementation files.
```

To materialize accepted tasks beneath an existing change specification, add:

```text
Use docs/changes/<change>[/<subchange>] as the specification and task
destination. Read its requirements and existing numbered tasks before review.
```

Expected execution:

1. The `@slow` root records the reviewed state and resolves the optional destination.
2. Batched `scout @smol` workers build one shared, risk-ranked source/specification map.
3. Five `boss-reviewer @task` leaves run concurrently, one per specialist area.
4. The root runs focused verification proposed by the leaves. Accelerator checks use the `remote-development` workflow.
5. `cpp-inference-review-synthesis` rejects weak findings, deduplicates by root cause, freezes ordering and blockers, and emits the final tasks.

A clean result is valid and is reported as `No material findings; no remediation tasks generated.` together with coverage and validation limits.

## 2. Incremental review

### One commit

Prefer an exact full commit hash:

```text
/boss Use the cpp-inference-code-review skill in selected-commit mode. Review
exact commit <full-commit-hash> against its parent without changing checkout
state. Inspect the complete diff plus the necessary callers, tests, backend
counterparts, capability/fallback paths, and surrounding contracts. Accept only
root causes introduced or materially exposed or worsened by this commit. Run
all five specialists and mandatory synthesis. Do not report unrelated existing
problems.
```

An exact complete commit message may be used instead, but a hash is less ambiguous:

```text
Review the commit whose complete message is: <exact message>
```

The resolver rejects ambiguous or fuzzy matches. A root commit is compared with the empty tree.

### Series of commits

Selected-commit mode has one commit and one parent baseline per review. Do not pass a range as if it were one selected commit. List the series in oldest-to-newest order, for example:

```bash
git rev-list --reverse <base-exclusive>..<tip>
```

Run one selected-commit review for every returned hash. This preserves causality: each finding must be introduced or materially exposed/worsened by that commit. Reuse the same `docs/changes/...` destination when all commits implement the same specification; synthesis checks for equivalent existing tasks and advances numbering without overwriting earlier output.

After the per-commit reviews, an optional current-state subset review can check interactions among the commits:

```text
/boss Use the cpp-inference-code-review skill to review the current integrated
state, restricted to <paths/symbols/subsystem changed by the series>. Run all
five specialists and mandatory synthesis. Treat this as a current-state subset
review, not as selected-commit causality, and state the exact coverage boundary.
```

Keep the distinction explicit: per-commit findings have commit causality; the final integrated review finds current interaction defects within the named subset.

## 3. Review a subset of code

Use the full orchestrator when all five review areas should examine only a subsystem, backend, operator, or exact path set:

```text
/boss Use the cpp-inference-code-review skill to review the current checked-out
state, with primary scope restricted to:
- <path-or-directory>
- <path:Symbol>
- <backend/operator/API boundary>

Run all five specialists and mandatory synthesis. Inspect code outside the
primary scope only when needed to establish callers, contracts, ownership,
backend counterparts, dispatch/fallback behavior, tests, or a falsifier. Do not
accept unrelated findings and do not claim whole-codebase coverage. Report the
exact inspected and uninspected boundaries.
```

For a subset introduced by one commit, combine path and commit restrictions:

```text
/boss Use the cpp-inference-code-review skill in selected-commit mode for exact
commit <hash>, with primary review scope limited to <paths/symbols>. Accept only
findings caused or materially exposed/worsened by that commit in the named
scope. Inspect external callers and counterparts only as supporting evidence.
Run all five specialists and mandatory synthesis.
```

Make the boundary concrete. Prefer exact paths, symbols, operators, backends, dtypes, or execution phases over requests such as “review the GPU code.”

## 4. Use only selected specialists

Available specialists:

| Skill | Area |
|---|---|
| `cpp-inference-contract-correctness` | contracts, operator/tensor semantics, capability, fallback, errors, compatibility |
| `cpp-inference-gpu-stability` | ownership, asynchronous lifetime, ordering, visibility, concurrency, cleanup, integer safety |
| `cpp-inference-backend-simplicity` | backend boundaries, duplicated policy/state, special cases, dead or weightless machinery |
| `cpp-inference-numerical-testing` | reference parity, tolerances, precision, quantization, reductions, edge cases, behavioral tests |
| `cpp-inference-performance` | synchronization, transfers, fallback, allocation, scheduling, overlap, memory, compilation, kernels |

### One specialist

Invoke that specialist directly. Standalone mode includes its own reconnaissance, focused verification, and mandatory synthesis:

```text
/boss Use only the cpp-inference-gpu-stability skill in standalone mode to
review <whole codebase | exact commit HASH | paths/symbols>. Do not claim
coverage of the other four review areas. Run mandatory
cpp-inference-review-synthesis and return or materialize direct remediation
tasks. Report inspected coverage, validation, and gaps.
```

Replace the skill name and scope as needed. For a selected commit, retain the exact-commit causality rule. For a specification-linked review, name the existing `docs/changes/...` destination.

### Two to four specialists

Do not invoke `cpp-inference-code-review`, because that orchestrator requires all five areas. Ask Boss to coordinate only the named specialist passes and synthesize them once:

```text
/boss Run a bounded orchestrated review using only these specialist skills:
- cpp-inference-contract-correctness
- cpp-inference-gpu-stability
- cpp-inference-performance

Scope: <whole current tree | exact commit HASH | exact paths/symbols>.
Resolve one shared scope and reconnaissance map. Dispatch the named specialist
passes concurrently as boss-reviewer @task candidate-only leaves. Explicitly
state that backend-simplicity and numerical-testing were not reviewed. After
collecting all named-area packets, run cpp-inference-review-synthesis exactly
once for cross-area falsification, deduplication, priority, blockers, and direct
task output. Do not run each specialist as a separate standalone finalizer.
```

This form preserves cross-specialist deduplication. A single standalone specialist is simpler when only one area is required.

## Review output and operational limits

Every accepted finding becomes one self-contained remediation task with evidence, scope relation, severity, verification state, confidence, affected symbols, acceptance criteria, non-goals, falsifier, and focused verification. Findings linked to a destination are written as:

```text
docs/changes/<change>[/<subchange>]/<NN>-<AREA-ID>-<remediation-slug>/spec.md
```

The finalizer never fills earlier numbering gaps, overwrites an existing task, or duplicates an equivalent root cause. Without a destination, tasks remain inline and no directory is invented.

Workers propose validation commands; the `@slow` root runs and records them after candidate collection. Advisor agreement is not verification. If a cheap lane, configured role, accelerator host, or required tool is unavailable, the review must report the resulting coverage limit and ask before using a materially more expensive fallback.
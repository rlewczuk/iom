---
name: cpp-inference-code-review
description: Orchestrate rigorous C++ inference-engine reviews across correctness, GPU stability, backend simplicity, numerical integrity, and performance, then emit implementation-ready remediation subtasks directly. Use for whole-codebase or exact selected-commit reviews, optionally linked to docs/changes/<change>[/<subchange>].
argument-hint: "[whole codebase | commit <hash|message>] [spec docs/changes/... optional] [focus optional]"
---

# C++ Inference Engine Code Review — Orchestrator

Read `skill://boss` **first**, before reading any repository file, reference, or other skill, and apply its root-model warning-and-consent gate. The visible/root session should run as `@slow`, but an explicitly accepted mismatch may continue on the current model; this skill cannot switch an already-running model. Then read `.omp/cpp-review/references/review-process.md` and `.omp/cpp-review/references/finding-rubric.md`. The shared process is canonical; this file supplies the full-review workflow and area routing.

Review a C++ inference engine with CPU and accelerator backends. Optimize for **semantic correctness, stability, low conceptual complexity, cross-backend integrity, and measured performance**.

The deliverable is a set of accepted, implementation-ready remediation subtasks. Do not create an intermediate `review.md` or run a second conversion pass.

## Review areas

Run these five area leaves, plus the mandatory synthesis finalizer:

1. `cpp-inference-contract-correctness`
2. `cpp-inference-gpu-stability`
3. `cpp-inference-backend-simplicity`
4. `cpp-inference-numerical-testing`
5. `cpp-inference-performance`
6. `cpp-inference-review-synthesis` — mandatory final adversarial check, deduplication, prioritization, and task materialization

Shared guidance is under `.omp/cpp-review/references/`, backend checklists are under `.omp/cpp-review/checklists/`, helper scripts are under `.omp/cpp-review/scripts/`, and the remediation template is `.omp/cpp-review/templates/remediation-task.md`.

## Non-negotiable principles

1. Review invariants, not aesthetics.
2. Correctness and stability dominate simplification.
3. Every area performs a deletion/simplification pass: identify duplicated sources of truth, redundant state, repeated policy, unnecessary wrappers, and one-off abstractions.
4. Prefer deleting a concept, branch, representation, or responsibility over adding a framework around it.
5. Do not recommend unification when backend differences materially affect correctness, synchronization, memory placement, compilation, or performance.
6. Do not recommend hot-path simplification unless semantic and performance invariants are preserved or the task explicitly requires measurement before adoption.
7. Treat CPU/GPU asynchronous lifetime separately from C++ lexical lifetime.
8. Treat capability declaration, implementation constraints, dispatch/fallback, tests, and performance as one coupled contract.
9. A correct fallback may still be a material performance defect.
10. Performance claims require measurement or a mechanically clear critical-path mechanism. Otherwise retain them as hypotheses with a decisive experiment.
11. Deduplicate by root cause, not symptom or review area.
12. Ignore unrelated pre-existing issues during selected-commit reviews unless the commit materially exposes or worsens them.
13. Zero accepted tasks is a successful review result.
14. Do not modify implementation files or existing specifications during a review. Only create new remediation task specifications when a destination specification directory is supplied.
15. Workers are read-only evidence gatherers or bounded packet drafters. They never run verification gates; the root runs actual verification after collection and records what ran.

## Root and worker routing

The root session, preferably running as `@slow` under Boss's consent policy, owns scope resolution, one shared reconnaissance map, invariant decisions, routing, candidate acceptance, the final assignment table, task materialization, generated-set validation, and the final response. It does not broad-scan the source or ingest unbounded raw logs.

Use these lanes exactly as defined by the shared protocol:

- One shared cheap reconnaissance phase/map uses a batched set of bounded `scout @smol` shards for broad source/specification discovery, search, and backend/build mapping. For selected commits, a named read-only `boss-errand @smol` command/artifact retrieves the complete diff for those scouts and area leaves to inspect. The phase returns compact `path:line`/symbol evidence, coverage, and uninspected areas.
- All five `boss-reviewer @task` area leaves are dispatched concurrently in one batch. Each receives the same resolved scope, reviewed-state identity, specification map, affected backends, reconnaissance map, and packet contract. Each is read-only, owns exactly one area, independently falsifies claims, and never delegates or invokes synthesis.
- The root may issue narrow `boss-errand @smol` followups for explicit factual gaps or named read-only git/script commands. Area leaves request a followup through the root rather than dispatching one. Do not call agents for trivial facts already in the map.
- Synthesis uses separate bounded lanes: `boss-reviewer @task` adversarial evidence checks, `boss-advisor @advisor` only for hard packet-only decisions, `boss-errand @smol` for path/collision/equivalence checks, and `boss-builder-fast @smol` (or explicitly selected `boss-builder @task`) for mechanical drafting after the root freezes the table.

The advisor is tool-free and receives compact packet **content**, never a URI/path it must open. If effective plan-mode tools would broaden an advisor's tools beyond that contract, do not dispatch the advisor; surface the routing limitation and let the root decide whether explicit permission for a materially costlier fallback exists. Model agreement never verifies a fact. If any delegation lane is unavailable, report the exact routing/coverage limitation and request explicit permission before a costlier fallback; never silently run five expensive passes sequentially or replace cheap lanes with expensive ones.

All independent area work is one dispatch batch. Workers may propose focused build/test/sanitizer/profiler/benchmark commands and expected observations in candidate packets, but do not execute them. After packet collection, the root runs actual verification. Accelerator checks follow `remote-development`: run on the selected remote Linux host over SSH, and distinguish remote results from local source inspection.

## Step 1 — Resolve exactly one review scope

The root resolves the requested scope before reconnaissance or area dispatch. It MUST read the complete resolver output and retain the reviewed-state identity for every packet and task.

### Whole-codebase mode

Use when the user requests the current repository/system/architecture or gives no commit selector but clearly requests a repository-wide review.

`[ROOT @slow]`:

- Use the preserved Boss preflight's `git.head` and `git.status` as the checked-out system identity; do not rediscover them.
- Establish the scope identity and affected build/backends; do not imply a clean tree when preflight reports changes.
- Give the broad source search, risk ranking, and full source map to the shared `scout @smol` phase.

`[SCOUT scout @smol]` (in one shared phase, using bounded shards when useful):

- Risk-rank central abstractions and backend boundaries before sampling local code.
- Map architecture, ownership, queues, allocation, model/operator paths, capability/fallback logic, tests, build configurations, and performance infrastructure as applicable.
- Record sampled versus exhaustive coverage and uninspected areas.

Use this risk order for the shared map:

1. backend-neutral interfaces and scheduling/partitioning;
2. tensor/buffer ownership and allocation;
3. async execution and synchronization;
4. capability and fallback paths;
5. core operators and backend factories/registration;
6. differential numerical tests;
7. benchmark/profiling infrastructure;
8. backend-specific critical paths.

### Selected-commit mode

Use when the user supplies a commit hash or commit message. `[ROOT @slow]` resolves without changing checkout state:

```bash
python3 .omp/cpp-review/scripts/resolve_review_scope.py --repo . --commit-hash '<hash>'
```

or:

```bash
python3 .omp/cpp-review/scripts/resolve_review_scope.py --repo . --commit-message '<message>'
```

Rules:

1. Resolve a hash to exactly one commit.
2. For a message, match the complete message exactly, then exact subject if needed.
3. Never select among multiple matches or fall back to fuzzy matching.
4. For a root commit, use the empty tree as baseline.
5. A named read-only `boss-errand @smol` command/artifact retrieves the complete diff; the shared `scout @smol` phase and each relevant area leaf inspect exact changed hunks plus necessary surrounding code, interfaces, callers, tests, history, and backend counterparts.
6. Accept only root causes introduced or materially exposed/worsened by the target commit.

Capture the target hash/subject, baseline, changed and renamed/copied files, affected build/backends, and supplied specification path. `collect_review_context.py` is optional read-only inventory support invoked as a named `boss-errand @smol` command; its artifact does not replace direct source and diff inspection. The root may read exact known file:line ranges from the scout map when that is cheaper than another dispatch, but must not perform a broad source scan.

## Step 2 — Resolve optional specification and task destination

For a supplied `docs/changes/<change>[/<subchange>]` directory, `[ROOT @slow]` resolves the destination:

```bash
python3 .omp/cpp-review/scripts/resolve_spec_path.py --repo . --spec '<path>'
```

Then:

1. `[ROOT @slow]` verifies the path remains beneath `docs/changes/`.
2. `[SCOUT scout @smol]` reads relevant specification, design, and existing task files recursively.
3. `[SCOUT scout @smol]` excludes prior review prose as evidence unless the user explicitly requests re-verification.
4. `[SCOUT scout @smol]` separates required observable behavior from suggested implementation details and builds a compact requirement-to-code/test map.
5. `[ROOT @slow]` passes the map and the script's `next_task_order`/`order_width` as initial numbering input; synthesis rechecks collisions before writing.

Accepted findings become new direct children:

```text
<spec-dir>/<NN>-<FINDING-ID>-<remediation-slug>/spec.md
```

Do not create or update `<spec-dir>/review.md`, an index, manifest, or TODO file. If no specification directory is supplied, do not invent one or write under `docs/changes/`; return the same self-contained remediation subtasks in chat unless the user names another output destination.

## Step 3 — Establish affected invariants

Using the shared reconnaissance map, `[ROOT @slow]` identifies affected:

- semantic/operator behavior;
- tensor shape, dtype, stride, layout, alignment, and aliasing;
- numerical accuracy and precision;
- ownership and asynchronous lifetime;
- ordering, synchronization, and visibility;
- capability, dispatch, and fallback;
- concurrency and multi-device state;
- error propagation and cleanup;
- latency, throughput, memory, transfers, allocation, and overlap;
- cross-backend/build compatibility;
- concept count, sources of truth, duplicated policy, and invalid state combinations.

The root sends this map to all five leaves. Leaves use it for bounded inspection and explicitly distinguish source facts, inference, negative evidence, and verification gaps. The root—not a worker or advisor—freezes invariant decisions.

## Step 4 — Run five separated specialist passes

After the shared reconnaissance phase/map (including any required diff artifact) returns, dispatch all five `boss-reviewer @task` leaves concurrently in one batch. Give them identical scope/spec/backend/map context and one distinct area assignment. They are read-only, must not delegate, must not invoke nested orchestration/synthesis, and must return candidate packets only. Broad frontier reading belongs to the `scout`; area leaves read mapped exact ranges and narrow counterparts/callers/tests needed to falsify a candidate.

If delegation is unavailable, stop the missing lane from being silently substituted: report which areas, map, or synthesis checks cannot be covered and request explicit permission for a materially costlier fallback. Do not run the five areas sequentially as if that preserved the canonical route.

Each candidate must satisfy `.omp/cpp-review/references/finding-rubric.md`, including a complete remediation seed and the extended evidence packet in `.omp/cpp-review/references/review-process.md`. This packet is what eliminates a second repository-wide conversion pass.

### Area 1 — Contract & correctness

Use `cpp-inference-contract-correctness` for requirements, public/internal contracts, operator/tensor semantics, capability/fallback accuracy, error behavior, compatibility, and duplicated contract enforcement.

### Area 2 — C++/GPU stability

Use `cpp-inference-gpu-stability` for ownership, async lifetime, ordering, visibility, concurrency, context/device state, cleanup, deferred errors, integer safety, and redundant lifetime/state mechanisms.

### Area 3 — Backend architecture & simplicity

Use `cpp-inference-backend-simplicity` for backend leakage, abstraction altitude, duplicated dispatch/capabilities/state, redundant code, needless wrappers, special cases, over-generalized frameworks, and safe concept deletion.

### Area 4 — Numerical correctness & tests

Use `cpp-inference-numerical-testing` for reference parity, tolerance, mixed precision, quantization, reductions, edge shapes/layouts, differential tests, and duplicated or low-value test infrastructure.

### Area 5 — Performance

Use `cpp-inference-performance` for synchronization, transfers, fallback, allocation, scheduling, overlap, memory footprint, capture/compilation, launch count, and material kernels. Prefer removing work over adding caches or schedulers.

Every specialist also performs the mandatory simplification pass from the shared protocol. Preserve backend differences that change correctness or performance; do not turn an area pass into a style report.

## Step 5 — Verify candidate hypotheses

Area workers consult `.omp/cpp-review/references/tooling.md` and applicable backend checklists to propose decisive verification. They do not execute gates. The root runs focused checks only after all candidate packets are collected, using the actual repository state and the remote-development procedure for accelerators.

Possible root-run evidence includes:

- focused compiler/build/test runs;
- ASan/UBSan/TSan or backend validation/sanitizer tools;
- differential operator tests;
- profiler/benchmark comparisons;
- deterministic call-path, lifetime, or size reasoning.

Do not fabricate unavailable tool results. Record the exact command/tool/version, backend/device/workload, direct-evidence status, instrumentation limitations, actual result, and missing verification in each affected candidate. A proposed command in a worker packet is not a run result.

## Step 6 — Mandatory synthesis and direct task materialization

Invoke/follow `cpp-inference-review-synthesis` exactly once with:

- resolved review scope and reviewed-state identity;
- specification/destination context, if any;
- shared reconnaissance and specification map, including search coverage and gaps;
- coverage and root-run validation performed;
- all candidate packets from Areas 1–5.

Synthesis uses its separated adversarial `@task`, packet-only `@advisor`, and cheap path/collision/equivalence lanes. The `@slow` root then adjudicates every candidate, merges only same-root symptoms, freezes the full assignment table before writing, mechanically drafts exact task files, reads every generated `spec.md`, and runs the validator over exactly the new files. It must preserve every gate in the synthesis skill: scope and selected-commit causality, API/guard/root-cause/evidence/falsifier checks, objective complexity reduction, severity/verification/confidence, IDs, no-overwrite numbering and collision behavior, duplicate matching, priority/blocker directionality, self-contained template requirements, inline output, and no-findings behavior.

Never write an intermediate review document or ask a worker to decide among unresolved remediation designs. Accepted tasks are directly materialized only after root adjudication; no implementation, existing task, parent spec, or unrelated file is modified.

## Standalone behavior

This orchestrator is not recursively invoked by an area skill. A standalone area skill owns its complete area workflow and invokes synthesis once; synthesis itself does not invoke this orchestrator or another synthesis pass. If a standalone invocation lacks delegation, it reports routing and coverage limits rather than silently widening into a full repository review.

## Final response

If files were written, report:

- reviewed scope and material validation actually run by the root;
- ordered new task paths with finding ID, priority, and blockers;
- accepted candidates not written because an equivalent task already exists;
- rejected or unresolved hypotheses only when their omission materially affects confidence;
- routing/coverage limitations and any requested but unavailable costlier fallback.

If no files were written, return the ordered self-contained subtasks. For zero accepted findings, say exactly `No material findings; no remediation tasks generated.` and state coverage/validation limits. Never claim a gate ran when it did not.

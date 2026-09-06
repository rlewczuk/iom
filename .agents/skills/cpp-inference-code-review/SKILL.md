---
name: cpp-inference-code-review
description: Orchestrate rigorous C++ inference-engine reviews across correctness, GPU stability, backend simplicity, numerical integrity, and performance, then emit implementation-ready remediation subtasks directly. Use for whole-codebase or exact selected-commit reviews, optionally linked to docs/changes/<change>[/<subchange>].
argument-hint: "[whole codebase | commit <hash|message>] [spec docs/changes/... optional] [focus optional]"
---

# C++ Inference Engine Code Review — Orchestrator

Review a C++ inference engine with CPU and accelerator backends. Optimize for **semantic correctness, stability, low conceptual complexity, cross-backend integrity, and measured performance**.

The deliverable is a set of accepted, implementation-ready remediation subtasks. Do not create an intermediate `review.md` or run a second conversion pass.

## Review areas

1. `cpp-inference-contract-correctness`
2. `cpp-inference-gpu-stability`
3. `cpp-inference-backend-simplicity`
4. `cpp-inference-numerical-testing`
5. `cpp-inference-performance`
6. `cpp-inference-review-synthesis` — mandatory final adversarial check, deduplication, prioritization, and task materialization

Shared guidance is under `.agents/cpp-review/references/`, backend checklists under `.agents/cpp-review/checklists/`, helper scripts under `.agents/cpp-review/scripts/`, and the remediation template is `.agents/cpp-review/templates/remediation-task.md`.

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

## Step 1 — Resolve exactly one review scope

Read `.agents/cpp-review/references/review-process.md` and `.agents/cpp-review/references/finding-rubric.md` first.

### Whole-codebase mode

Use when the user requests the current repository/system/architecture or gives no commit selector but clearly requests a repository-wide review.

- Review the checked-out working tree as a system.
- Risk-rank central abstractions and backend boundaries before sampling local code.
- Cover architecture, ownership, queues, allocation, model/operator paths, capability/fallback logic, tests, build configurations, and performance infrastructure as applicable.
- Record the HEAD hash and whether reviewed files had working-tree changes so generated tasks identify the reviewed state.
- State sampled versus exhaustive coverage. Never imply every file was inspected when it was not.

### Selected-commit mode

Use when the user supplies a commit hash or commit message. Resolve without changing checkout state:

```bash
python3 .agents/cpp-review/scripts/resolve_review_scope.py --repo . --commit-hash '<hash>'
```

or:

```bash
python3 .agents/cpp-review/scripts/resolve_review_scope.py --repo . --commit-message '<message>'
```

Rules:

1. Resolve a hash to exactly one commit.
2. For a message, match the complete message exactly, then exact subject if needed.
3. Never select among multiple matches or fall back to fuzzy matching.
4. For a root commit, use the empty tree as baseline.
5. Inspect the exact diff plus necessary surrounding code, interfaces, callers, tests, history, and backend counterparts.
6. Accept only root causes introduced or materially exposed/worsened by the target commit.

Capture the target hash/subject, baseline, changed and renamed/copied files, affected build/backends, and supplied specification path. `collect_review_context.py` is a convenience; inspect source and diff directly.

## Step 2 — Resolve optional specification and task destination

For a supplied `docs/changes/<change>[/<subchange>]` directory, run:

```bash
python3 .agents/cpp-review/scripts/resolve_spec_path.py --repo . --spec '<path>'
```

Then:

1. Verify the path remains beneath `docs/changes/`.
2. Read relevant specification, design, and existing task files recursively.
3. Exclude prior review prose as evidence unless the user explicitly requests re-verification of it.
4. Separate required observable behavior from suggested implementation details.
5. Build a compact requirement-to-code/test map.
6. Use the returned `next_task_order` and `order_width` only as initial numbering input; the finalizer rechecks collisions before writing.

Accepted findings become new direct children:

```text
<spec-dir>/<NN>-<FINDING-ID>-<remediation-slug>/spec.md
```

Do not create or update `<spec-dir>/review.md`, an index, manifest, or TODO file.

If no specification directory is supplied, do not invent one or write under `docs/changes/`. Return the same self-contained remediation subtasks in chat unless the user names another output destination.

## Step 3 — Establish affected invariants

Before delegating, identify affected:

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

## Step 4 — Run five separated specialist passes

When delegation is available, give all five specialists the same resolved scope, specification map, affected backend list, and candidate packet contract. Run independent passes concurrently. Each specialist owns only its area and returns candidate packets; it must not write task files in orchestrated mode. If delegation is unavailable, run the same passes sequentially and keep candidates partitioned until synthesis.

Each candidate must satisfy `.agents/cpp-review/references/finding-rubric.md`, including a complete remediation seed. This is what eliminates the second repository-wide conversion pass.

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

## Step 5 — Verify candidate hypotheses

Use `.agents/cpp-review/references/tooling.md` and applicable backend checklists. Tools verify hypotheses; raw diagnostics are not findings.

Possible evidence:

- focused compiler/build/test runs;
- ASan/UBSan/TSan or backend validation/sanitizer tools;
- differential operator tests;
- profiler/benchmark comparisons;
- deterministic call-path, lifetime, or size reasoning.

Do not fabricate unavailable tool results. Record exact validation and missing verification in each affected candidate.

## Step 6 — Mandatory synthesis and direct task materialization

Invoke/follow `cpp-inference-review-synthesis` once with:

- resolved review scope and reviewed-state identity;
- specification/destination context, if any;
- coverage and validation performed;
- all candidate packets from Areas 1–5.

The finalizer must:

1. try to disprove every candidate;
2. reject weak, out-of-scope, duplicate, or purely aesthetic claims;
3. merge symptoms with one root cause;
4. verify that simplification tasks delete objective complexity without hiding required semantics;
5. choose final IDs, severity, confidence, priority, order, blockers, and precise slugs;
6. ensure each accepted root cause has exactly one self-contained task;
7. write and validate task specs directly when a spec directory exists, or return them directly otherwise.

Never write an intermediate review document.

## Final response

If files were written, report:

- reviewed scope and material validation;
- ordered new task paths with finding ID, priority, and blockers;
- accepted candidates not written because an equivalent task already exists;
- rejected or unresolved hypotheses only when their omission materially affects confidence.

If no files were written, return the ordered self-contained subtasks. For zero accepted findings, say `No material findings; no remediation tasks generated.` and state coverage/validation limits.
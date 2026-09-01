---
name: cpp-inference-code-review
description: Orchestrate rigorous code-quality reviews of C++ inference engines with multiple GPU backends. Use for whole-codebase reviews or exact single-commit reviews selected by hash/message, especially when simplicity, stability, numerical parity, backend architecture, and performance matter. When a docs/changes/<change>[/<subchange>] specification is supplied, write the final review to that directory's review.md.
argument-hint: "[whole codebase | commit <hash|message>] [spec docs/changes/... optional]"
---

# C++ Inference Engine Code Review — Orchestrator

Perform a high-signal review of a C++ inference engine that may use CUDA, HIP/ROCm, Vulkan, SYCL/oneAPI, Metal, OpenCL, CPU, or other execution backends. Optimize for **semantic correctness, stability, structural simplicity, cross-backend integrity, and measured performance**.

This is an orchestrator. Delegate or conceptually separate work using the six sibling skills listed below. If subagents/skill delegation are unavailable, perform the same passes sequentially and keep their candidate findings separate until synthesis.

## Sibling specialist skills

1. `cpp-inference-contract-correctness`
2. `cpp-inference-gpu-stability`
3. `cpp-inference-backend-simplicity`
4. `cpp-inference-numerical-testing`
5. `cpp-inference-performance`
6. `cpp-inference-review-synthesis`

Shared guidance is under `.agents/cpp-review/references/`, backend checklists under `.agents/cpp-review/checklists/`, helper scripts under `.agents/cpp-review/scripts/`, and the report template under `.agents/cpp-review/templates/` relative to this file.

## Non-negotiable review principles

1. Review **invariants**, not aesthetics.
2. Correctness/stability constraints dominate simplification.
3. Do not recommend a simplification of performance-critical code unless semantic and performance invariants are preserved or the change is explicitly proposed for measurement.
4. Treat CPU/GPU asynchronous lifetime separately from C++ lexical lifetime.
5. Treat backend capability declaration, implementation constraints, fallback, tests, and performance as one coupled contract.
6. A correct fallback may still be a material performance regression.
7. Do not claim a performance regression without either measured evidence or a mechanically clear critical-path mechanism. Otherwise classify it as a hypothesis and prescribe verification.
8. Deduplicate by **root cause**, not symptom.
9. Ignore unrelated pre-existing issues during selected-commit reviews unless the commit materially exposes/worsens them.
10. Zero material findings is a successful review outcome.
11. Do not modify implementation files during a review. If a specification path is supplied, only `review.md` should be written unless the user explicitly requests fixes.

## Step 1 — Resolve review scope exactly

Determine one and only one scope from the user's prompt.

### A. Whole-codebase mode

Use when the user explicitly requests the whole repository/current codebase/system/architecture, or provides no commit selector but clearly asks for a repository-wide review.

In this mode:

- Review the current checked-out repository state as a system.
- Include architecture, backend layering, common abstractions, critical lifetime/synchronization paths, tests, build configuration, and performance infrastructure.
- Do not pretend every file was deeply inspected if the repository is too large. Use risk-based sampling and state coverage limits in the report.
- Prioritize central abstractions and backend boundaries over local style.

### B. Selected-commit mode

Use when the user supplies a commit hash or commit message.

Resolve without changing checkout state. Prefer:

```bash
python .agents/cpp-review/scripts/resolve_review_scope.py --repo . --commit-hash '<hash>'
```

or:

```bash
python .agents/cpp-review/scripts/resolve_review_scope.py --repo . --commit-message '<message>'
```

Resolution rules:

1. A hash must resolve to exactly one commit.
2. For a message, first match the complete commit message exactly; then exact subject if needed.
3. Never silently choose among multiple matches.
4. Never silently fall back to a fuzzy/substring match.
5. If the selector is ambiguous or missing, report that review scope cannot be established rather than reviewing an arbitrary commit.
6. For a root commit, use the empty tree as the baseline.
7. Review the exact commit diff plus necessary surrounding code, interfaces, tests, call sites, history/blame, and backend counterparts.
8. Findings must be introduced by the target commit or materially exposed/worsened by it.

Capture and report:

- commit hash and subject;
- parent/baseline hash;
- changed files;
- rename/copy status when present;
- build/backend areas touched;
- specification path, if any.

Use `collect_review_context.py` as a convenience, but inspect the actual source and diff directly.

## Step 2 — Resolve specification context

If the user identifies a specification directory of the form:

```text
docs/changes/<change-name>[/<subchange>]
```

then:

1. Normalize it to a repository-relative directory. Prefer `python .agents/cpp-review/scripts/resolve_spec_path.py --repo . --spec '<path>'`.
2. Verify that it exists and remains beneath `docs/changes/`.
3. Read all relevant specification/design/task files recursively, excluding an existing `review.md` as review evidence unless the user explicitly asks to revisit it.
4. Distinguish **required external behavior** from incidental implementation details.
5. Build a compact requirement-to-code map before judging correctness.
6. Set the final report target to exactly:

```text
<spec-dir>/review.md
```

If no specification directory is supplied, do not invent one and do not write into `docs/changes/`.

## Step 3 — Establish review invariants

Before producing findings, identify which invariants are affected:

- semantic/operator behavior;
- tensor shape/dtype/stride/layout/alignment/aliasing;
- numerical accuracy and precision;
- ownership and lifetime;
- ordering/synchronization;
- memory visibility;
- backend capability and fallback;
- concurrency/thread safety;
- error propagation and cleanup;
- critical-path latency/throughput/memory;
- cross-backend/build compatibility.

Use `.agents/cpp-review/references/review-process.md` and `.agents/cpp-review/references/finding-rubric.md`.

## Step 4 — Run six separated review areas

Keep candidate findings partitioned by area until synthesis.

### Area 1 — Contract & correctness

Invoke/follow `cpp-inference-contract-correctness`.

Focus on implemented requirements, public/internal contracts, graph/operator semantics, capability/fallback accuracy, compatibility, and missing cases.

### Area 2 — C++/GPU stability

Invoke/follow `cpp-inference-gpu-stability`.

Focus on ownership, resource lifetime, async execution, stream/queue/event ordering, device/context state, race conditions, cleanup, deferred errors, and size/overflow hazards.

### Area 3 — Backend architecture & simplicity

Invoke/follow `cpp-inference-backend-simplicity`.

Focus on correct abstraction altitude, backend leakage into common code, duplicated dispatch/capabilities/state, needless wrappers, special cases, and opportunities to reduce concepts without weakening semantics or performance.

### Area 4 — Numerical correctness & tests

Invoke/follow `cpp-inference-numerical-testing`.

Focus on reference parity, dtype/operator-specific tolerance, mixed precision, reductions, quantization, edge shapes/layouts, cross-backend differential tests, and regression-test adequacy.

### Area 5 — Performance

Invoke/follow `cpp-inference-performance`.

First inspect system-level effects: synchronization, copies, fallback, allocations, scheduling, overlap, compilation/capture, kernel launch count, and memory footprint. Only then inspect kernel-level behavior where material.

### Area 6 — Synthesis & adversarial verification

Invoke/follow `cpp-inference-review-synthesis` only after Areas 1–5 produce candidate findings.

For each candidate, try to disprove it before publishing it.

## Step 5 — Verification and tooling

Use mechanical tools to **verify hypotheses**, not to flood the review with diagnostics.

Read `.agents/cpp-review/references/tooling.md` and the relevant backend checklist(s).

Examples include:

- compiler/build/tests;
- clang-tidy / static analysis;
- ASan/UBSan/TSan where applicable;
- CUDA Compute Sanitizer;
- Vulkan validation/synchronization validation;
- ROCm tracing/profiling;
- cross-backend differential operator tests;
- benchmark/profile comparisons.

If a tool cannot be run, do not fabricate results. State the missing verification explicitly.

## Step 6 — Finding acceptance gate

A final finding should normally have all of:

- a concrete affected invariant;
- the smallest useful code location;
- a reproducible or clearly reasoned failure scenario;
- evidence tied to the reviewed scope;
- observable impact;
- a smallest reasonable structural fix;
- a verification method;
- severity, confidence, and verification state.

Do not publish:

- formatting/style trivia;
- ordinary linter/compiler diagnostics without deeper impact;
- unsupported personal preferences;
- unrelated pre-existing issues in commit mode;
- speculative micro-optimizations stated as facts;
- abstractions that merely move complexity around;
- duplicate symptoms of one root cause.

## Step 7 — Produce the report

Use `.agents/cpp-review/templates/review.md` as the structural contract.

The final report must contain these top-level review areas in this order:

1. Contract & correctness
2. C++/GPU stability
3. Backend architecture & simplicity
4. Numerical correctness & tests
5. Performance
6. Synthesis / overall assessment

Within each area, order material findings by severity. If there are no material findings in an area, write `No material findings.` and optionally note verification performed.

Every final finding uses the schema in `.agents/cpp-review/references/finding-rubric.md`.

For selected-commit reviews, explicitly state whether each finding is introduced by the selected commit or materially exposed by it.

For whole-codebase reviews, state the reviewed coverage and sampling limitations.

### Specification-linked output

If a specification directory was supplied, write the completed report to:

```text
<spec-dir>/review.md
```

Then validate:

```bash
python .agents/cpp-review/scripts/validate_review.py \
  --review-file '<spec-dir>/review.md' \
  --spec-dir '<spec-dir>'
```

If no specification directory was supplied, return the report to the user; only write a file if the user asked for one.

## Final response behavior

When a file was written, state the path and summarize only the highest-impact conclusions. Do not duplicate the entire report in chat.

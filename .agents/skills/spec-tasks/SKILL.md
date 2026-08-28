---
name: spec-tasks
description: Split a change specification into dependency-ordered, prioritized, self-contained mini-specifications for small implementer models. Use only through /spec-tasks <spec-name>[/subdirectory] or when explicitly requested.
hide: true
---

# Spec Tasks

Turn one change specification into the smallest set of implementation tasks needed to deliver it. Write one focused mini-specification per task; do not implement the change.

The command supplies a path relative to `docs/changes/`:

```text
<spec-name>[/subdirectory...]
```

For target `<target>`, use exactly:

- parent specification: `docs/changes/<target>/spec.md`;
- user remarks, optional: `docs/changes/<target>/spec-fixme.md`;
- generated task specifications: `docs/changes/<target>/<NN>-<task-slug>/spec.md`.

The numeric prefix is part of the task directory name and records dependency order. The ordered directories plus the completion response are the task list; do not create a separate TODO or index file unless the user requests one.

## Guardrails

- Require one non-empty target argument. Reject absolute paths, `.` or `..` components, empty path components, backslashes, and any target that escapes `docs/changes/`.
- Allow safe nested targets such as `allocator/gpu-layout`; the argument names the exact directory containing the parent `spec.md`.
- If the parent `spec.md` does not exist, report its expected path and stop. A missing `spec-fixme.md` is normal.
- Read the complete parent specification and complete fixme file when present.
- Treat direct conversation instructions as highest priority, then `spec-fixme.md` as user corrections or additions, then `spec.md`. For claims about current repository behavior, the repository is authoritative. Ask only when a material intent or policy conflict remains.
- Do not modify the parent `spec.md`, `spec-fixme.md`, implementation files, or unrelated task directories.
- Keep the design minimal. Exclude speculative abstractions, future-proofing, generic frameworks, optional extras, unrelated cleanup, and complementary features not required by the source specification.
- Do not invent functionality to make a task feel complete. Every requirement in every mini-spec must trace to the parent spec, the fixme remarks, a direct user instruction, or a repository constraint necessary to implement them correctly.
- Do not split work merely to produce more tasks. If the specification is already one cohesive, context-sized unit, create one task.
- Do not create tasks for work that is already complete and conforms to the specification.
- Do not implement the change. The deliverables are the task mini-specifications and the concise ordered list in the completion response.

## Workflow

### 1. Read and normalize the requested change

Read `spec.md`, then `spec-fixme.md` if present. Extract:

- required outcomes and observable behavior;
- explicit scope and non-goals;
- interfaces, data shapes, state transitions, errors, compatibility requirements, and constraints;
- named files, symbols, commands, tests, examples, and external or local references;
- every actionable fixme remark;
- unresolved decisions and assumptions.

Build one coherent requirement set using the precedence rules above. A fixme correction replaces the conflicting parent requirement; do not preserve both alternatives. Do not propagate brainstorming, rejected alternatives, or editorial commentary as implementation work.

### 2. Ground the work in the repository

Explore only the project areas needed to decompose and anchor the change. Read referenced files and enough surrounding implementation, call sites, tests, configuration, schemas, and project design documentation to establish:

- what is already implemented and correct;
- what is partial, missing, or inconsistent with the specification;
- the actual files and symbols each remaining change touches;
- local implementation and test conventions to reuse;
- existing facilities or extension points that avoid new machinery;
- dependency edges between pieces of work.

Classify each source requirement as **done**, **partial**, **missing**, or **discrepant**. Generate tasks only for the remaining work. Do not trust a path, symbol, signature, or current-behavior claim until checked when it affects a task.

Repository exploration is for reducing implementer search, not for proposing adjacent improvements. Ignore unrelated defects unless one directly blocks the specified behavior; if it does, include only the smallest necessary correction.

### 3. Decide whether and how to decompose

A task must be:

- one narrow concern or tracer-bullet slice with a clear outcome;
- independently verifiable after its declared blockers are complete;
- small enough for one fresh context window used by a flash-class implementer;
- large enough to avoid handoff-only tasks and meaningless scaffolding;
- complete across the layers needed for that outcome, rather than a horizontal list such as “all types,” “all logic,” then “all tests.”

Prefer vertical slices that leave the repository in a coherent state. Include focused tests with the behavior they cover. Create a separate test task only for genuinely cross-cutting integration, end-to-end, migration, performance, or compatibility verification spanning multiple slices.

Create a prerequisite refactor task only when the current structure makes the required change unsafe or impractical. Keep it mechanical and no broader than necessary. For an unavoidable wide refactor, use the smallest dependency-safe expand–migrate–contract sequence; do not introduce compatibility layers otherwise.

Split a candidate task when it has independent outcomes, unrelated file groups, different blockers, or cannot be understood and implemented in one context. Merge candidates when one only scaffolds another, neither is useful alone, or separating them duplicates substantial context.

If no implementation work remains, create no task directories and report that the specification is already satisfied, with the repository evidence that supports that conclusion.

### 4. Build and prioritize the task graph

For every candidate task, identify only genuine blocking edges. A blocker is work whose output is required before the task can be implemented or verified; conceptual similarity is not a dependency.

Assign a priority:

- **P0** — required prerequisite, public contract, migration, or correctness/safety work that gates broad progress;
- **P1** — required feature behavior on the normal implementation path;
- **P2** — required finishing work such as cross-cutting verification or documentation that does not gate implementation.

All generated tasks are required; P2 never means optional. Order tasks with a stable topological sort: blockers first, then higher priority, then the order requirements appear in the parent specification. When tasks are independent, say so instead of adding a false edge.

Number the sorted tasks from `01`. Use short lowercase kebab-case slugs that describe the delivered behavior, for example `01-load-manifest`. Avoid generic slugs such as `setup`, `misc`, `changes`, or `cleanup`.

Before writing, check every destination. Never silently overwrite an unrelated file or user-authored task specification. If a destination already contains a task for the same source and outcome, revise it carefully; otherwise choose a distinct precise slug and report the collision.

### 5. Write one self-contained mini-spec per task

Each mini-spec is an implementation contract for a smaller, cheaper model. It must be understandable without reading the parent specification, fixme file, conversation, or sibling task specs. Repeat the few shared decisions needed by the task instead of saying “follow the parent spec” or “same as the previous task.” References to blockers provide sequencing, not missing requirements.

Use this structure, omitting only sections that truly do not apply:

```markdown
# <Task title>

**Order:** <NN>
**Priority:** <P0|P1|P2> — <brief reason>
**Blocked by:** <task directory names, or “None”>
**Source:** `docs/changes/<target>/spec.md`

## Outcome

<One short paragraph describing the complete, observable result.>

## Scope

- <Precise behavior this task must implement.>
- <Inputs, outputs, state transitions, errors, or compatibility rules needed here.>

## Implementation references

- **Modify:** `path/to/file` — `<symbol or section>`; <why it is the touchpoint>.
- **Read:** `path/to/analogue` — `<symbol or section>`; <specific convention to reuse>.
- **Tests:** `path/to/test` — <existing suite, fixture, or nearest pattern>.

## Requirements

- <Normative, unambiguous requirement.>
- <Relevant boundary or failure behavior.>

## Non-goals

- <Nearby work explicitly excluded from this task.>

## Acceptance criteria

- [ ] <Observable criterion that distinguishes success from failure.>
- [ ] <Relevant negative or boundary criterion.>

## Verification

- `<specific existing test/build/format command>`
- <Any focused manual or runtime scenario required to observe the result.>
```

Mini-spec writing rules:

- Keep only information needed to implement this task. Do not copy the entire parent specification.
- State concrete behavior, not vague directions such as “handle appropriately,” “support as needed,” or “update relevant files.”
- Name exact verified repository paths and important symbols. Mark a path as planned when the file does not exist yet. Include the nearest useful implementation and test analogues when they materially reduce searching.
- Explain current behavior only where the implementer needs it to make the change safely.
- Preserve precise interface, format, error, ordering, lifecycle, and compatibility decisions from the source requirement.
- Include non-goals that prevent likely scope drift, especially tempting complementary features or generalization.
- Keep code snippets out unless a compact type, schema, state machine, or algorithm fragment captures a decision more precisely than prose.
- Make acceptance criteria observable and automatable where possible. Include meaningful happy-path, negative, and boundary behavior proportionally; do not add a generic edge-case checklist.
- Name exact focused verification commands when the repository provides them. Do not prescribe a project-wide suite when a narrower command proves the task.
- Keep tests in the same task as non-trivial behavior. Do not create test-only busywork or test implementation details.
- Do not require the implementer to rediscover decisions, search broadly, or choose among unresolved alternatives.

### 6. Resolve only blocking ambiguity

Use repository evidence and established project conventions for factual and low-risk implementation details. If a source ambiguity materially changes behavior, task boundaries, or dependency order and cannot be resolved from the repository, ask one focused question at a time with a recommended minimal answer. Wait for the answer before writing affected mini-specs.

Do not ask the user to approve an otherwise clear breakdown. Do not manufacture choices, expand the product design, or turn decomposition into a design interview.

### 7. Final consistency pass

Before completing:

- account for every required source behavior exactly once, except intentional repetition needed to keep mini-specs self-contained;
- confirm completed behavior has no task and every remaining requirement has one;
- verify each blocker points to an earlier generated task and no dependency cycle exists;
- verify numeric directory order matches the dependency and priority order;
- verify each mini-spec contains sufficient local code and test references;
- remove duplicated work, scaffolding-only tasks, speculative improvements, and optional extras;
- confirm acceptance criteria collectively deliver the parent specification and applicable fixme remarks.

## Completion response

Report only:

- the ordered task list with order, priority, directory path, and blockers;
- one sentence explaining the chosen granularity, especially when the spec remained one task;
- any material ambiguity, collision, or residual risk.

Keep the response concise. The generated mini-specifications are the primary deliverable.

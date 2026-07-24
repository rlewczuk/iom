---
name: specflow-execution
description: Use only after specification.md and tasks.md are approved; resumes from persistent task state, dispatches fresh implementation subagents, then independent review and testing subagents, repairs failures, and performs final whole-change verification.
---

# Subagent-driven execution

## Preconditions

Stop unless both files say `status: approved`:

```text
.specs/<change-id>/specification.md
.specs/<change-id>/tasks.md
```

If approved scope changes materially, stop execution and return to
`specflow-refinement`.

## Core principle

Fresh implementer per task plus independent review and testing.

The coordinator does not implement tasks itself. It reads durable state,
constructs precise subagent prompts, dispatches subagents, evaluates evidence,
and updates `.specs`.

## Startup and resume

1. Read `progress.md`.
2. Read `tasks.md`.
3. Identify tasks already `complete`.
4. Verify their implementation, review, and test evidence files exist.
5. Select the first dependency-ready incomplete task or parallel-safe wave.
6. Read each selected detailed task in full.
7. Include durable discoveries from earlier tasks.
8. Set selected tasks to `in_progress`.

Never restart completed work merely because the chat session is new.

## Model routing

Choose the lowest-cost adequate model on each `task` call:

- detailed task `model_class: smol` → `@smol`;
- `model_class: task` → `@task`;
- `model_class: slow` → `@slow`;
- test execution with straightforward commands → `@smol`;
- failed-test diagnosis or normal review → `@task`;
- final review or high-risk review → `@slow`.

Escalate only after a concrete failure, blocker, or quality problem. Do not
retry the same failed prompt and model unchanged.

## Isolation and concurrency

Default to non-isolated sequential execution because review and tests must see
the current project state.

Parallel implementation is permitted only for one execution wave whose task
files prove non-overlapping write scopes and state. When using isolated
worktrees, merge and retest each result before review.

Never parallelize review ahead of implementation or testing ahead of review.

## Per-task loop

### 1. Dispatch implementation

Spawn `specflow-implementer` with:

- change ID;
- task ID;
- exact specification path and relevant excerpts;
- full detailed task content;
- approved plan constraints;
- accumulated discoveries;
- explicit write scope;
- instruction to load `specflow-tdd`;
- instruction not to edit `.specs`;
- requested status format.

The implementer must inspect actual code and execute tests. It must not trust
the plan blindly when repository facts differ. Genuine ambiguity returns
`NEEDS_CONTEXT`; a hard blocker returns `BLOCKED`.

Write the result to:

```text
.specs/<change-id>/reports/<task-id>-implementation.md
```

Set task status `implemented` only after confirming that expected files changed
and claimed commands are credible.

### 2. Dispatch independent review

Spawn `specflow-reviewer` after implementation. Give it:

- approved specification;
- approved detailed task;
- implementation report;
- actual diff or commit range;
- relevant durable discoveries.

The reviewer is read-only and must inspect actual code. It returns one of:

- `APPROVED`
- `CHANGES_REQUIRED`
- `BLOCKED`

Write the full review to:

```text
.specs/<change-id>/reviews/<task-id>-review.md
```

If changes are required:

1. set task status `review_failed`;
2. dispatch a fresh `specflow-implementer` with the original task and exact
   review findings;
3. overwrite or append implementation evidence;
4. rerun review from scratch.

### 3. Dispatch independent testing

Only after review approval, spawn `specflow-tester`. Give it:

- task acceptance criteria;
- exact required commands;
- implementation report;
- review report;
- actual repository state.

The tester may run commands and inspect files but must not modify production
code. It returns:

- `PASS`
- `FAIL`
- `BLOCKED`

Write the full result to:

```text
.specs/<change-id>/reviews/<task-id>-test.md
```

If tests fail:

1. set task status `test_failed`;
2. dispatch a fresh implementer with exact failure evidence;
3. rerun independent review;
4. rerun independent testing.

### 4. Complete task

Only after `APPROVED` and `PASS`:

1. set detailed task status `complete`;
2. update the corresponding row in `tasks.md`;
3. add evidence links;
4. update `progress.md`;
5. append durable discoveries needed by later tasks;
6. continue without asking the developer between tasks.

Stop only for:

- genuine missing developer decision;
- approved plan invalidated by repository reality;
- unresolved blocker after appropriate model escalation;
- unsafe working-tree or merge state.

## Final whole-change gates

After every detailed task is complete:

### Final review

Spawn `specflow-final-reviewer` on `@slow` unless the change is demonstrably
low risk. It checks the entire approved specification, task plan, diff, tests,
architecture, security, compatibility, migrations, and production readiness.

Write:

```text
.specs/<change-id>/reviews/final-review.md
```

Repair every blocking finding through a fresh implementer, then rerun relevant
task review/testing and final review.

### Final testing

Spawn `specflow-tester` with whole-change scope. It runs the complete relevant
suite plus build, lint, typecheck, migration, or integration commands required
by the project and specification.

Write:

```text
.specs/<change-id>/reviews/final-test.md
```

Repair and repeat until `PASS` or blocked.

## Completion

When final review is `APPROVED` and final testing is `PASS`:

1. set `tasks.md` `status: complete`;
2. set `progress.md` `stage: complete`;
3. clear current task and blocker;
4. update timestamps;
5. do not create a final summary report;
6. tell the developer only that the change reached complete state and point to
   the state directory.

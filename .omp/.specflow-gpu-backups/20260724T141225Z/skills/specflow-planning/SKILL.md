---
name: specflow-planning
description: Use after .specs/<change-id>/specification.md is approved; creates tasks.md and self-contained detailed tasks/*.md, validates coverage and dependencies, waits for explicit plan approval, and never edits production code.
---

# Implementation planning

## Preconditions

Stop unless:

```yaml
# .specs/<change-id>/specification.md
status: approved
```

Read:

- the entire approved specification;
- repository guidance;
- relevant source, tests, build files, and interfaces;
- `.specs/<change-id>/progress.md`.

Planning may inspect but must not edit production code.

## Outputs

Create or update:

```text
.specs/<change-id>/tasks.md
.specs/<change-id>/tasks/001-<slug>.md
.specs/<change-id>/tasks/002-<slug>.md
...
```

## Planning standard

Plan for an implementer with no conversation history and limited project
context. Each detailed task must be independently executable in a fresh
subagent session.

### Task granularity

A task should produce one coherent, independently reviewable outcome. Split a
task when:

- it mixes unrelated responsibilities;
- it exceeds one focused agent session;
- it changes too many architectural layers without an intermediate test;
- it has separable dependencies;
- it requires different risk or model classes.

Do not split so finely that each file edit becomes a task. Steps inside a task
should be bite-sized RED-GREEN-REFACTOR actions.

### Detailed task requirements

Every `tasks/NNN-<slug>.md` must include:

- stable task ID and status frontmatter;
- objective;
- exact specification references;
- complete context;
- in-scope and out-of-scope behavior;
- exact create/modify/test paths;
- interfaces, signatures, schemas, and invariants;
- concrete implementation steps;
- exact test commands;
- expected evidence;
- acceptance criteria;
- review focus;
- dependencies;
- write scope;
- risk and model class.

Prohibited placeholders:

- "add validation";
- "handle errors";
- "write tests";
- "update as needed";
- "similar to the previous task";
- an undefined type, function, method, schema, or command.

### Task index requirements

`tasks.md` must contain:

- goal and approach;
- global constraints;
- coverage map from specification to task files;
- dependency-aware execution waves;
- Markdown links to every detailed task;
- risk and model class;
- task status;
- validation checklist;
- review history.

Every task listed in `tasks.md` must resolve to an existing file. No orphaned
task files are allowed.

### Parallelism

Put tasks in the same execution wave only when all are true:

- no dependency path exists between them;
- their `write_scope` values do not overlap;
- they do not mutate shared generated state, schemas, migrations, lockfiles, or
  integration fixtures;
- merge ordering cannot change behavior.

When uncertain, schedule sequentially.

### Model class

Classify each task:

- `smol`: mechanical or narrow, low-risk work.
- `task`: normal implementation requiring repository reasoning.
- `slow`: architecture-heavy, security-sensitive, migration-critical, or
  difficult debugging work.

## Self-review before developer review

1. Map every specification requirement and acceptance criterion to tasks.
2. Check all task-file links.
3. Check dependency IDs and cycles.
4. Check parallel waves for overlapping files and mutable state.
5. Search for placeholders and vague instructions.
6. Check cross-task names, types, and interfaces for consistency.
7. Confirm every task has executable verification.
8. Confirm the plan fits the approved scope and contains no speculative extras.

Fix every issue inline.

Set:

```yaml
# tasks.md
status: review
```

Update `progress.md`:

```yaml
stage: planning
plan_status: review
```

## Plan-review gate

Present the plan and ask:

> Please review `.specs/<change-id>/tasks.md` and its linked task files. What
> should change, or do you explicitly approve this plan and want me to proceed
> with implementation?

Do not implement while waiting.

## Corrections

When the developer requests corrections:

1. update `tasks.md`;
2. update, add, split, renumber, or remove detailed files as needed;
3. repair all links and dependencies;
4. preserve review history;
5. increment plan revision;
6. rerun the full self-review;
7. return to the plan-review gate.

## Explicit approval

Only when the developer explicitly approves and asks to proceed:

1. set `tasks.md` `status: approved`;
2. set `approved_at`;
3. record concise approval evidence;
4. update `progress.md`:
   - `stage: implementation`
   - `plan_status: approved`;
5. immediately load and follow `specflow-execution`.

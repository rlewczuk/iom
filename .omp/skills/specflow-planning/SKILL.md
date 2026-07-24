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

Planning may inspect but must not edit production code. Load `specflow-exploration` and use one or more routed `scout` instances when repository breadth makes direct inspection inefficient. Internet research remains optional and disabled by default.

## Exploration during planning

When exploration is enabled, use a single batch of distinct routed `scout` tasks rather than repeatedly searching the same paths. Record decisive file references and wiki freshness in the plan context. Native `librarian` is allowed only after developer-approved internet research and `/explorer-web on`; external information must be cited and cannot override repository reality.

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
- risk and model class;
- `prewalk_policy: auto|on|off`;
- `prewalk_situation: mechanical|repository-heavy-small-edit|normal|cross-cutting|high-risk|debugging`;
- `exploration_policy: auto|on|off`;
- `internet_research: false|true`;
- `docs_update: auto|on|off`.

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
- prewalk policy and situation;
- exploration, internet-research, and docs-update policy;
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

### Model class and prewalk policy

Load `specflow-cost-routing` and classify each task before plan review.

Model adequacy floor:

- `smol`: mechanical or narrow, low-risk work.
- `task`: normal implementation requiring repository reasoning.
- `slow`: architecture-heavy, security-sensitive, migration-critical,
  numerically delicate GPU work, or difficult debugging work.

Prewalk metadata:

- `prewalk_policy: on` only when the developer wants repository exploration on
  a stronger model before a localized, stable edit proceeds on the configured
  target model;
- `prewalk_policy: off` for read-only work, debugging, broad architectural
  changes, high-risk/security work, migrations, or tasks whose first edit is
  likely to invalidate the plan;
- `prewalk_policy: auto` delegates to `.omp/gpu-lab/cost-policy.json`;
- `prewalk_situation` must describe the actual task, not merely the desired
  cost: `mechanical`, `repository-heavy-small-edit`, `normal`, `cross-cutting`,
  `high-risk`, or `debugging`.

Expose these choices in `tasks.md`. They are approval-scope decisions: the
developer may change them before approving the plan. Never assume prewalk is a
quality improvement; it is a cost-routing mechanism for an editing subagent.

## Self-review before developer review

1. Map every specification requirement and acceptance criterion to tasks.
2. Check all task-file links.
3. Check dependency IDs and cycles.
4. Check parallel waves for overlapping files and mutable state.
5. Search for placeholders and vague instructions.
6. Check cross-task names, types, and interfaces for consistency.
7. Confirm every task has executable verification.
8. Confirm every task has an adequate model floor and a justified prewalk policy/situation.
9. Confirm prewalk is not assigned to read-only, debugging, cross-cutting, or high-risk work by default.
10. Confirm the plan fits the approved scope and contains no speculative extras.

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
> should change—including any model class or prewalk choice—or do you explicitly
> approve this plan and want me to proceed with implementation?

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

## GPU-aware planning

For GPU-relevant tasks, populate `gpu_targets`, `gpu_profile`, `benchmark_required`, and `debug_policy`. Include exact local checks and `.omp/gpu-lab/bin/gpu-labctl` validation phases. Keep source writing in one implementer; plan CUDA/ROCm validators and portability review as a parallel read-only wave after ordinary review. Require a full matrix rerun after repairs.

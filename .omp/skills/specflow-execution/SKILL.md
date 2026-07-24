---
name: specflow-execution
description: Use only after specification.md and tasks.md are approved; resumes from persistent task state, routes fresh subagents through the project cost/prewalk policy, then performs independent review, testing, GPU validation, repair, and final verification.
---

# Cost-aware subagent-driven execution

## Preconditions

Stop unless both files say `status: approved`:

```text
.specs/<change-id>/specification.md
.specs/<change-id>/tasks.md
```

If approved scope changes materially, stop execution and return to
`specflow-refinement`.

Load `specflow-cost-routing` before every `task` spawn. Load `specflow-exploration` before any scout/librarian spawn and `specflow-doc-librarian` before project-wiki maintenance. Validate:

```bash
.omp/gpu-lab/bin/gpu-lab-cost validate
```

## Core principle

Fresh implementer per task plus independent review and testing. The coordinator
is the only owner of durable state. It does not implement tasks itself.

The parent reads the active cost policy and passes an explicit model selector
or fallback chain on every task item. Agent frontmatter and
`.omp/gpu-lab/omp-config.yml` are fallback defaults only.

## Startup and resume

1. Read `progress.md`, `tasks.md`, and the next detailed task.
2. Validate `.omp/gpu-lab/cost-policy.json` and record the active profile.
3. Verify evidence for tasks already marked `complete`.
4. Select the first dependency-ready incomplete task or safe wave.
5. Include durable discoveries from earlier tasks.
6. Set selected tasks to `in_progress`.

Never restart completed work merely because the chat session is new.

## Model and prewalk routing

For each spawn, resolve the role through `specflow-cost-routing` and persist a
`COST_ROUTING` block in the resulting evidence. Enforce the detailed task's
`model_class` as an adequacy floor.

For an implementation task:

1. Read `prewalk_policy` and `prewalk_situation` from the approved detailed task.
2. Resolve the effective policy, including developer overrides.
3. Use `specflow-implementer-prewalk` only when effective prewalk is `on`.
4. Otherwise use `specflow-implementer`.
5. Pass the resolved model fallback chain explicitly in the task item.

Prewalk is bounded to that one editing subagent. Never turn on parent-session
`/prewalk` for the workflow. Never prewalk validators, reviewers, testers,
performance agents, or debugger agents.

Escalate only after a concrete failure, malformed tool use, confirmed quality
miss, or an approved risk floor. Do not retry an unchanged prompt on the same
failed model.

## Isolation and concurrency

Default to non-isolated sequential implementation because review and tests must
see the current project state. Parallel implementation is permitted only when
approved task files prove non-overlapping write scopes and state, and isolated
worktrees are used.

Never parallelize multiple source writers into the same checkout. Review and
GPU validators may run concurrently only after implementation is stable.

## Per-task loop

### 1. Dispatch implementation

Resolve the implementation route. Spawn either `specflow-implementer` or
`specflow-implementer-prewalk` with:

- explicit resolved `model`;
- change ID and task ID;
- approved specification and full detailed task;
- accumulated discoveries;
- exact write scope;
- instruction to load `specflow-tdd`;
- instruction not to edit `.specs`;
- approved `exploration_policy`, `internet_research`, current session explorer/web gates, and maximum fan-out;
- required `COST_ROUTING` and `EXPLORATION_USED` output blocks.

The prewalk variant must inspect and plan before its first edit/write; its first
edit must be part of the approved TDD sequence, not a dummy handoff trigger.

Write the result to:

```text
.specs/<change-id>/reports/<task-id>-implementation.md
```

Set status `implemented` only after confirming expected changes and credible
commands.

### 2. Optional project-wiki synchronization

Read the task's `docs_update` policy. If it is `on`, or `auto` and the implementation/report contains durable architectural, interface, toolchain, CUDA/ROCm, debugging, operational, or failure-mode knowledge, resolve role `doc_librarian` and spawn `specflow-doc-librarian` serially. It may write only `docs/agent-wiki/**`.

Write its report to:

```text
.specs/<change-id>/reports/<task-id>-docs.md
```

The docs update must finish before independent review so review and testing see the complete task diff. If the librarian edits outside its boundary, stop and mark the task blocked.

### 3. Dispatch independent review

Resolve role `reviewer`; apply the task's adequacy floor. Spawn
`specflow-reviewer` with an explicit model, approved scope, implementation
report, actual diff, durable discoveries, and the approved exploration/internet policy.

Write:

```text
.specs/<change-id>/reviews/<task-id>-review.md
```

On `CHANGES_REQUIRED`, set `review_failed`, dispatch a fresh routed implementer
with exact findings, and rerun review from scratch. Repair prewalk is a new
policy decision: keep the task override unless the failure proves the handoff
was inappropriate, in which case disable it for that repair and record why.

### 4. Dispatch independent testing

Only after review approval, resolve role `tester` and spawn
`specflow-tester` with an explicit model and the approved exploration/internet policy. Deterministic command execution may
use the lowest configured tier; semantic diagnosis must be escalated to the
reviewer/debugger rather than invented by a cheap model.

Write:

```text
.specs/<change-id>/reviews/<task-id>-test.md
```

On `FAIL`, set `test_failed`, dispatch a fresh routed implementer with exact
failure evidence, then rerun independent review and testing.

### 5. Multi-host CUDA/ROCm gate

When `gpu_targets` is non-empty, load `specflow-gpu-orchestrator`. Resolve and
pass explicit models for each CUDA/ROCm validator and portability reviewer.
Use one source writer, one shared run ID, parallel read-only validators, and
durable evidence under `.specs/<change-id>/gpu-lab/`.

A required target failure prevents completion. Repair only through a fresh
routed implementer followed by ordinary review and a complete matrix rerun.
Debugging is serial, uses the `debugger` role, and never uses prewalk.

### 6. Complete task

Only after ordinary review/testing and every required GPU gate pass:

1. set the detailed task status `complete`;
2. update the task index and progress file;
3. link implementation, docs, review, test, GPU, exploration, and `COST_ROUTING` evidence;
4. append durable discoveries;
5. continue without asking between approved tasks.

Stop only for a missing developer decision, invalidated approved plan,
unresolved blocker after justified escalation, or unsafe working-tree state.

## Final whole-change gates

### Final docs synchronization

Before final review, run one routed `specflow-doc-librarian` pass when any task used `docs_update: on|auto` and produced durable knowledge. Review the docs-only diff and record `.specs/<change-id>/reports/final-docs.md`. Do not change documentation after final approval without rerunning affected gates.

### Final review

Resolve role `final_reviewer` and spawn `specflow-final-reviewer` with an
explicit model. It checks the full specification, plan, diff, tests,
architecture, security, compatibility, migrations, GPU portability, and
production readiness.

Write `.specs/<change-id>/reviews/final-review.md`. Repair blocking findings
through a fresh routed implementer and rerun all affected gates.

### Final testing

Resolve role `tester` for whole-change execution. Spawn with an explicit model
and run the complete relevant suite. Write
`.specs/<change-id>/reviews/final-test.md`.

## Completion

When final review is `APPROVED`, final testing is `PASS`, and the required GPU
matrix passes:

1. set `tasks.md` `status: complete`;
2. set `progress.md` `stage: complete`;
3. clear current task and blocker;
4. update timestamps;
5. do not create a redundant summary report;
6. point the developer to the durable state directory.

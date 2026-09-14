---
name: spec-run-all
description: Execute all eligible direct implementation children under docs/changes continuously in dependency-aware parallel worktrees.
hide: true
---

# Spec Run All

Run all unfinished **direct child** tasks below `docs/changes/<task-name>`. Read `skill://spec-run-task` first. This is an orchestration wrapper around its worktree and Git control plane, not a separate implementation workflow. The target directory need not have its own control or spec. Nested task containers are reported as unsupported direct children and are not expanded.

## Mechanical control plane

Use only the bundled scripts and shared preflight:

```text
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty scan <task-name>
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty queue <task-name> [--state <temporary-json-path>]
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty prepare <task-name> [--state <temporary-json-path>]
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty control
.omp/csw/bin/csw_preflight --repo <repo> --workflow spec-run-all --pretty
```

The script loads `task_ctl` through spec-run-task's public API. `task_ctl` alone discovers, validates, and orders direct canonical `task.yml` controls. The script does not parse task metadata from `spec.md` or `task.md`. It preserves task_ctl's numeric-order/canonical-ID order and never guesses basenames. PyYAML 6.0.3 is the supported runtime dependency used by `task_ctl`.

Missing or malformed direct controls are retained as per-task blockers through `task_ctl.scan_tasks`; independent valid tasks continue. `task.md` contains only execution Outcome, Summary, Verification, and Errors evidence.

Run `csw_preflight` exactly once per invocation before the first `prepare`. Preserve its JSON verbatim. Exit `2` or `ok: false` is a concrete blocked result; never repeat or bypass its Git, OMP, profile, model, tool, spawn, or recursion validation.

`scan` returns immediate task children in task_ctl order. Valid records contain `name`, canonical `task_id`, relative `task_path`, exact spec/control/evidence paths, type, lifecycle status, order/priority, and canonical blocker records. Invalid controls retain the exact task identity and a concrete error instead of stopping sibling discovery. A missing spec is a blocker. A direct HLD/nested container is unsupported for this command.

`queue` returns:

- `ready`: every dependency-satisfied direct implementation child in `new`, `critic`, `planned`, or resumable `ready` state, with no skill concurrency cap;
- `waiting`: lifecycle/dependency waits and retained running/ready owner outcomes;
- `blocked`: invalid task inputs, unsupported containers, and failed/blocked execution outcomes;
- `already_done`: direct tasks whose canonical integrated lifecycle is done;
- `finished`: true only when all discovered direct tasks are done, including an empty target.

`prepare` provisions or reuses all currently ready worktrees and preserves independent preparation failures. `control` refreshes the validated integration branch/head without target discovery or preparation.

## Local invocation state

Create one parent-owned temporary JSON file outside the checkout, initialized to `{}`:

```json
{
  "dependencies": {
    "child-name": ["docs/changes/target/other-child"]
  },
  "outcomes": {
    "child-name": {
      "status": "running|blocked|failed|ready",
      "reason": "nonempty explanation"
    }
  }
}
```

`dependencies` contains only supplemental semantic prerequisites and every value is an exact canonical `.cswd/tasks/...` or `docs/changes/...` ID. They augment, never erase, the canonical `blocked-by` records returned by task_ctl. Preserve actual prerequisites; missing canonical dependencies wait. Numeric order and priority are scheduling order, not dependencies.

`outcomes` prevents duplicate dispatch/retry within this invocation. Record prepared leaves as running before dispatch. Change to ready only after the owner ends with a provisional commit; use failed for a concrete implementation/runtime failure and blocked for an external prerequisite. Running/ready suppress redispatch but never satisfy dependencies. Canonical integrated done overrides a stale local outcome.

Only canonical `done` in the integration checkout unlocks an edge. Ready, verified, a private branch, child report, or existing commit does not. Failed/blocked/running are local execution outcomes, never lifecycle statuses.

## Continuous orchestration

1. Run `scan`, create the outside-checkout state, and run `queue`. Do not parse, rewrite, or sort controls. Do not prepare an empty selection.
2. Before any preparation or child dispatch, run the shared preflight exactly once. Stop on failure.
3. Run `prepare`. Record preparation failures as blocked. Mark every successful prepared leaf running, then dispatch one task per record in returned order with `agent: "spec-run-all-implementer"`. Submit every eligible leaf, including a single leaf, without a skill-level cap. Harness admission limits may queue work; queued owners stay running and are never redispatched.
4. Give each child exact `repo_root`, `task_id`, `task_path`, worktree/spec/control/evidence paths, feature branch/base, integration branch/head, exclusive scope, known interface contracts, and required verification. The child uses assigned-worktree mode, skips validation, and never integrates/rebases or edits task files directly.
5. Consume individual owner completions continuously. Retain the owner/job identity and prepared record until settled. Record crashes/dispatch failures as failed, external prerequisites as blocked, and provisional success as ready. An owner may report implementation failure only after the required one-time `spec-run-debug` rescue attempt.
6. As soon as one leaf is ready, serialize parent takeover for that leaf. Run `control`, rebase its one ready commit onto the current canonical head, and run focused plus repository-required combined verification in that exact worktree. Do not wait for unrelated owners. Route a recoverable failure back to the same owner after marking it running; preserve its worktree and one commit.
7. After observed success, use spec-run-task `annotate --outcome verified` with exact evidence, `commit --status verified`, and `check --status verified`. Then call `integrate`. Do not write done yourself. `integrate` validates the historical verified control using `task_ctl.parse_config`, rebuilds the same task commit with task_ctl's verified-to-done update, rechecks the canonical head, and fast-forwards. Failure restores the private verified commit, so dependencies remain locked and no completion commit exists.
8. Immediately after successful integration, rerun `queue` on the canonical tree, then prepare and dispatch **all** newly eligible children before processing another completion or waiting. If A integrates while B runs, start every direct child now unlocked by A; B is not a barrier. Rescan after every settled outcome. Stop only at `finished` or a true stall with no eligible, running/queued, or completed owner.

If the integration head advances before integration, refresh through `control`, rebase, rerun affected verification, update evidence, and retry. Never use the head captured at dispatch. Never reprepare active work merely to refresh control state.

## Child ownership and rescue

The assigned child reads `skill://spec-run-task`, calls `show` first, works only in its exact worktree, reads complete requirements, implements the leaf, and uses:

```text
spec_run_task.py ... annotate '<task_path>' --outcome ready --summary '<summary>'
spec_run_task.py ... commit '<task_path>' --status ready --outcome '<behavior>'
```

It skips builds/tests/linters/formatters during the parallel pass and reports exact parent verification still required. It never directly edits `task.yml` or `task.md`, mutates Git, integrates, rebases, expands nested tasks, or delegates except to one `spec-run-debug` when genuinely stuck.

The debugger receives exact worktree/spec paths, constraints, current changes, error/dead end, observations, and attempted approaches. It is read-only and returns root cause evidence plus a concrete proposal to the same owner. The owner retains all implementation decisions and edits.

## Reporting

Report separately: already done; done and integrated with final rewritten commit and observed verification; failed with retained local outcome; nondependency blocked; and dependency waiting with exact canonical cause. Include target, integration branch/head, prepared worktrees/commits, and residual risk. `finished: false` is not completion.

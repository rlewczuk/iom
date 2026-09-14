---
name: spec-run-task
description: Implement one docs/changes task or its unfinished implementation descendants in isolated reusable worktrees, using task_ctl-backed lifecycle controls and deterministic Git integration.
hide: true
---

# Spec Run Task

Implement the requested specification completely. Each executable implementation task owns one deterministic feature branch, one registered worktree under `.work/`, sibling `spec.md`, `task.yml`, and `task.md`, and exactly one final task commit. A target with implementation descendants is a container: execute those leaves, never the container as implementation.

## Mandatory control plane

Use the bundled helper for every repeatable task, evidence, worktree, and Git-history operation:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo <integration-checkout> --pretty <command> ...
```

The helper loads `.omp/csw/bin/task_ctl` through its public `runpy.run_path` API. `task_ctl` alone discovers and validates canonical task metadata in `task.yml`: type, lifecycle status, numeric order, priority, canonical `blocked-by` records, and source. The helper owns execution evidence in `task.md`, local worktree state, and Git mechanics. PyYAML is therefore a runtime dependency of `task_ctl`; the supported environment provides PyYAML 6.0.3.

Never read metadata from Markdown. Never parse, sort, create, or edit `task.yml` yourself, and never ask a child agent to do so. Use helper JSON in its returned order. `task.md` is implementation evidence only: `**Outcome:**`, Summary, Verification, and Errors are not lifecycle metadata.

Missing or malformed `task.yml` is a hard error.

Do not replace a failed helper operation with hand-written Git plumbing or reconstructed paths. Never directly run `git worktree add`, `git add`, `git commit`, `git reset`, `git rebase`, `git merge`, or `git update-ref`. The helper is authoritative for prepare/reuse, checkpoint, one-commit consolidation, rebase conflict continuation, historical control validation, verified-to-done finalization, and fast-forward integration.

Exit status `2` is a validation or safety failure. Exit status `3` is a helper-managed rebase conflict; resolve only the named files in the owning worktree and call `continue-rebase`.

## Invariants

- The integration checkout is the canonical control plane. Do not edit, build, test, format, or run task code there.
- Only canonical `done` in the integrated tree satisfies a dependency. Missing dependencies wait. `ready`, `verified`, a child claim, or a private branch never unlocks an edge.
- Lifecycle values are `new`, `critic`, `planned`, `ready`, `verified`, and `done`. `running`, `failed`, and `blocked` are execution outcomes only.
- Execution can start from `new`, `critic`, or `planned`, or resume from `ready`, once blockers are done. Successful implementation advances to `ready`; observed verification advances to `verified`; only `integrate` may advance it to `done`.
- Failed and blocked attempts retain evidence and their one task commit without changing the current lifecycle; they are never integrated and never satisfy dependencies.
- Each attempted leaf retains exactly one non-merge commit with subject `spec-run-task(<full-task-path>): <outcome>`.
- Rebase task commits; never merge task branches. Keep worktrees and feature branches as resumable state.
- Reuse registered state. Never stash, clean, remove, or recreate an existing task worktree.

## 1. Inspect

Require one target relative to `docs/changes/`, then run before reading task contents or preparing worktrees:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo . --pretty inspect '<target>'
```

`inspect` requires canonical controls, uses task types to distinguish an implementation leaf from a container, recursively discovers through the `task_ctl` API, preserves its numeric-order/canonical-ID ordering, resolves exact canonical blockers, detects selected dependency cycles, and validates the integration checkout.

Use `repo_root`, `integration_branch`, `integration_head`, `requested_target`, `target_kind`, target paths, and every `leaves[]` path and identifier verbatim. Each leaf includes `task_id`, `task_path`, `spec_path`, `control_path`, evidence path, exact worktree paths, branch, lifecycle status, canonical blocker records, and `explicitly_ready`.

If the target is a leaf, the main agent implements it. For a container, schedule every unfinished implementation leaf whose `explicitly_ready` is true. Skip only lifecycle `done`. A missing dependency remains waiting; only its canonical integrated `done` status satisfies it. Numeric order and priority are scheduling order, never inferred dependencies. Do not add an LLM-side sort or parse controls.

## 2. Prepare

For each eligible leaf, copy values from the same inspection result:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty prepare '<task_path>' \
  --run-target '<requested_target>' \
  --integration-branch '<integration_branch>' \
  --integration-base '<integration_head>'
```

`prepare` rechecks type `impl`, canonical dependencies, branch/head, repository identity, and deterministic worktree ownership. Scheduled implementation starts at `ready`. The helper also permits an explicit worktree for `new`, `critic`, or `planned` solely so an early failed/blocked attempt can be retained without falsifying lifecycle. It refuses `verified` and `done`.

Treat `worktree`, `spec_path`, `control_path`, and `annotation_path` as opaque. On reuse, inspect `status_entries` and `task_commits`, or call:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty show '<task_path>'
```

If coherent dirty state must be preserved, use `checkpoint`; never checkpoint merely to dismiss an ownership error:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty checkpoint '<task_path>'
```

## 3. Implement in the assigned worktree

Prefix every Read/Edit/Write path with the returned worktree and set every Bash `cwd` to it. Read the complete spec, repository guidance, relevant code/tests, and applicable skills. Implement only that leaf.

For a container wave, provision each ready leaf before dispatch and give one owner its exact repo root, task IDs/paths, worktree paths, branch/base, exclusive scope, canonical dependency contracts, and verification still required. Children skip all validation during the parallel pass. They never edit controls or evidence directly and never mutate Git except through this helper.

A stuck implementation owner must invoke exactly one `spec-run-debug` rescue agent before reporting an implementation failure. Pass exact worktree/spec paths, constraints, current changes, concrete error, observations, and attempted approaches. The debugger is read-only; the owner resumes and applies or rejects its proposed solution with evidence. External prerequisites may be blocked without debugger escalation.

For `spec-run-all`, reuse the parent's successful preflight record. Otherwise run `.omp/csw/bin/csw_preflight --repo <repo> --workflow spec-run-task --pretty` once before container dispatch. A failed preflight is retained failure evidence; never substitute another profile/model.

## 4. Record evidence and consolidate

Never edit `task.md` or `task.yml` directly. `annotate` preserves unrelated task prose and replaces generated Outcome/Summary/Verification/Errors. For `ready` and `verified`, it delegates the lifecycle write to `task_ctl.set_task`; failed/blocked affect evidence only.

Provisional success:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome ready --summary '<factual provisional summary>'
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty commit '<task_path>' \
  --status ready --outcome '<concise behavioral outcome>'
```

Observed verification:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome verified --summary '<delivered behavior>' \
  --verification '<exact command or scenario> — <observed result>'
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty commit '<task_path>' --status verified
```

Repeat `--verification` for distinct observations. Once the final task subject exists, omit the commit `--outcome` to preserve it.

Failure or external blocker uses the unchanged `lifecycle_status` returned by `show`/`prepare`:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome failed --summary '<reachable retained work>' \
  --error '<operation — concrete failure>'
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty commit '<task_path>' \
  --status '<unchanged-lifecycle-status>' --outcome '<concise attempted outcome>'
```

Use `--outcome blocked` only for an external prerequisite. The unchanged status may be `new`, `critic`, `planned`, or `ready`. The one failed/blocked task commit is retained but cannot integrate, and the orchestrator records its local execution outcome.

## 5. Rebase and verify

Refresh the integration head with `inspect` or the orchestration helper, then rebase through the helper:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty rebase '<task_path>' --onto '<commit>'
```

On exit `3`, resolve only returned conflicts and call `continue-rebase`; use `abort-rebase` only to abandon that replay. If conflict resolution changes behavior, rerun affected verification and reconsolidate.

Children stop at lifecycle `ready`. The parent performs focused behavioral verification and repository-required combined verification from the exact task worktree. On failure, return concrete evidence to the same owner, amend its one commit, and rerun. On success, annotate `verified`, commit `--status verified`, and mechanically check:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty check '<task_path>' --status verified
```

For a container train, rebase verified commits in helper order. Combined verification runs from the final train worktree. If an earlier task changes, rebuild every later replay. Every final train commit must contain its own verified `task.yml` transition and task evidence.

## 6. Container roll-up

When every implementation leaf beneath a requested container is either done in the canonical tree or verified in the final train, the final leaf owner records each ancestor roll-up as verified before its final commit:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<final-leaf-task>' \
  --for-task '<container-task-path>' --outcome verified \
  --summary '<all implementation leaves verified>' \
  --verification '<combined command or scenario> — <observed result>'
```

The helper allows only true ancestor containers inside the run target. Their verified controls and evidence are folded into the final leaf's single commit. Integration finalizes those controls to done in that same commit. Never create a separate roll-up/completion commit.

If all implementation leaves are already done but a container roll-up is missing or stale, choose the stable last completed leaf as owner and invoke normal `prepare` with the container as `--run-target`. The helper permits only this completion repair, requires all descendants done, and returns `rollup_only: true`; it preserves dirty or unintegrated state rather than overwriting it. Do not implement anything. Record verified evidence for the owner and each stale ancestor with `annotate`, then `commit --status verified --outcome '<container completion outcome>'`, check, and integrate the single annotation-only task commit. No new implementation task or container worktree is needed.

## 7. Integrate

For a leaf or the final branch of a verified train:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty integrate '<final-task-path>'
```

`integrate` requires a clean fast-forward verified train. It validates each historical `task.yml` blob with `task_ctl.parse_config`, requires every task commit to contain its own verified control and successful verified evidence, validates ancestor roll-up ownership, and rejects merges, duplicate/non-task commits, malformed controls, failed evidence, or non-verified status.

Only inside this operation, the helper rebuilds the train in the same order, changes every verified control owned by each commit to done through `task_ctl.set_task`, and recreates that same one commit—no completion commit. It then rechecks the integration head and fast-forwards. If finalization or fast-forward fails, it restores the private branch to the verified train; canonical status remains unchanged. Thus dependencies can observe done only after successful integration.

If the integration branch advanced, refresh/rebase, rerun affected verification, refresh evidence, and retry. After an integrated wave, inspect again and schedule newly eligible work from the canonical tree.

## Completion response

Report the target and integration branch; every leaf as already done, done/integrated, failed, blocked, or dependency-waiting; exact prepared worktree/branch/final commit for attempted leaves; observed verification; roll-up and integrated head; and retained errors or blockers. Never claim unobserved validation or partial completion.

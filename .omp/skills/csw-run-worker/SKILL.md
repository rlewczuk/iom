---
name: csw-run-worker
description: Implement one .cswd/tasks task or its unfinished implementation descendants in isolated reusable worktrees, using task_ctl-backed lifecycle controls and deterministic Git integration.
hide: true
---

# CSW Run Worker

Implement the requested specification completely. Each executable implementation task owns one deterministic feature branch, one registered worktree under `.work/`, sibling `spec.md`, `task.yml`, and `task.md` under `.cswd/tasks/`, and exactly one final code commit. A target with implementation descendants is a container: execute those leaves, never the container as implementation.

When invoked by `csw-run`, that skill's role boundaries take precedence: the root only orchestrates, its control-mode `csw-verifier` owns discovery/preparation, and one dedicated leaf `csw-verifier` owns rebase, verification, review evidence, and integration. The original implementer still owns code edits and the unchanged ready-result contract. Standalone `csw-run-worker` execution retains the workflow below.

## Mandatory control plane

Use the shared helper for every repeatable task, evidence, worktree, and Git-history operation:

```text
.omp/csw/bin/csw_run_worker --repo <integration-checkout> --pretty <command> ...
```

The helper loads `.omp/csw/bin/task_ctl` through its public `runpy.run_path` API. `task_ctl` alone discovers and validates canonical task metadata in `task.yml`: type, lifecycle status, numeric order, priority, canonical `blocked-by` records, and source. The helper owns execution evidence in `task.md`, local worktree state, and Git mechanics. PyYAML is therefore a runtime dependency of `task_ctl`; the supported environment provides PyYAML 6.0.3.

Never read metadata from Markdown. Never parse, sort, create, or edit `task.yml` yourself, and never ask a child agent to do so. Use helper JSON in its returned order. `task.md` is implementation evidence only: `**Outcome:**`, Summary, Verification, and Errors are not lifecycle metadata.

Missing or malformed `task.yml` is a hard error.

Do not replace a failed helper operation with hand-written Git plumbing or reconstructed paths. Never directly run `git worktree add`, `git add`, `git commit`, `git reset`, `git rebase`, `git merge`, or `git update-ref`. The helper is authoritative for prepare/reuse, checkpoint, one-commit consolidation, rebase conflict continuation, commit-bound local evidence validation, verified-to-done finalization, and fast-forward integration.

Exit status `2` is a validation or safety failure. Exit status `3` is a helper-managed rebase conflict; resolve only the named files in the owning worktree and call `continue-rebase`.

## Invariants

- The integration checkout is the canonical control plane. Do not edit, build, test, format, or run task code there.
- Only canonical `done` in the shared local task store satisfies a dependency. Missing dependencies wait. `ready`, `verified`, a child claim, or a private branch never unlocks an edge; the helper writes done only after successful integration.
- Lifecycle values are `new`, `critic`, `planned`, `ready`, `verified`, and `done`. `running`, `failed`, and `blocked` are execution outcomes only.
- Execution can start from `new`, `critic`, or `planned`, or resume from `ready`, once blockers are done. Successful implementation advances to `ready`; observed verification advances to `verified`; only `integrate` may advance it to `done`.
- Failed and blocked attempts retain evidence and their one task commit without changing the current lifecycle; they are never integrated and never satisfy dependencies.
- Each attempted leaf retains exactly one non-merge commit with subject `csw-run-worker(<full-task-path>): <outcome>`.
- Rebase task commits; never merge task branches. Keep worktrees and feature branches as resumable state.
- Reuse registered state. Never stash, clean, remove, or recreate an existing task worktree.
- `.cswd` is local, shared, and unversioned. Every prepared worktree links to the integration checkout's `.cswd`; task status and evidence changes are immediately visible across worktrees. Never stage the link or its contents, copy metadata into code commits, or use it on a remote host.
- The helper keeps commit-bound task controls and evidence in local worktree Git state. Retain that state together with the worktrees; code history alone does not contain task specifications or verification records.

## 1. Inspect

Require one target relative to `.cswd/tasks/`, then run before reading task contents or preparing worktrees:

```text
.omp/csw/bin/csw_run_worker --repo . --pretty inspect '<target>'
```

`inspect` requires canonical controls, uses task types to distinguish an implementation leaf from a container, recursively discovers through the `task_ctl` API, preserves its numeric-order/canonical-ID ordering, resolves exact canonical blockers, detects selected dependency cycles, and validates the integration checkout.

Use `repo_root`, `integration_branch`, `integration_head`, `requested_target`, `target_kind`, target paths, and every `leaves[]` path and identifier verbatim. Each leaf includes `task_id`, `task_path`, `spec_path`, `control_path`, evidence path, exact worktree paths, branch, lifecycle status, canonical blocker records, and `explicitly_ready`.

If the target is a leaf, the main agent implements it. For a container, schedule every unfinished implementation leaf whose `explicitly_ready` is true. Skip only lifecycle `done`. A missing dependency remains waiting; only its canonical integrated `done` status satisfies it. Numeric order and priority are scheduling order, never inferred dependencies. Do not add an LLM-side sort or parse controls.

## 2. Prepare

For each eligible leaf, copy values from the same inspection result:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty prepare '<task_path>' \
  --run-target '<requested_target>' \
  --integration-branch '<integration_branch>' \
  --integration-base '<integration_head>'
```

`prepare` rechecks type `impl`, canonical dependencies, branch/head, repository identity, and deterministic worktree ownership. It links the worktree's `.cswd` to the integration checkout's local metadata directory and preserves the current lifecycle. Reuse validates the existing link; conflicting directories or links are blockers, never overwritten. Task metadata must be untracked. Scheduled execution starts from `new`, `critic`, `planned`, or resumable `ready`; verified work awaits the integration owner (`csw-verifier` under `csw-run`), and completed work is eligible only for the container repair below.

Treat `worktree`, `spec_path`, `control_path`, and `annotation_path` as opaque. On reuse, inspect `status_entries` and `task_commits`, or call:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty show '<task_path>'
```

If coherent dirty state must be preserved, use `checkpoint`; never checkpoint merely to dismiss an ownership error:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty checkpoint '<task_path>'
```

## 3. Implement in the assigned worktree

Prefix every Read/Edit/Write path with the returned worktree and set every Bash `cwd` to it. Read the complete spec, repository guidance, relevant code/tests, and applicable skills. Implement only that leaf.

For container execution, provision each ready leaf before dispatch and give one owner its exact repo root, task IDs/paths, worktree paths, branch/base, exclusive scope, canonical dependency contracts, and verification still required. Children skip all validation during the parallel pass. They never edit controls or evidence directly and never mutate Git except through this helper. `csw-run` dispatches newly eligible leaves continuously, never in waves.

A stuck implementation owner must invoke exactly one `spec-run-debug` rescue agent before reporting an implementation failure. Pass exact worktree/spec paths, constraints, current changes, concrete error, observations, and attempted approaches. The debugger is read-only; the owner resumes and applies or rejects its proposed solution with evidence. External prerequisites may be blocked without debugger escalation.

For `csw-run`, reuse the control peer's successful preflight record forwarded by the root. Otherwise run `.omp/csw/bin/csw_preflight --repo <repo> --workflow csw-run-worker --pretty` once before container dispatch. A failed preflight is retained failure evidence; never substitute another profile/model.

## 4. Record evidence and consolidate

Never edit `task.md` or `task.yml` directly. `annotate` preserves unrelated task prose and replaces generated Outcome/Summary/Verification/Errors in the shared `.cswd/tasks/` store. For `ready` and `verified`, it delegates the lifecycle write to `task_ctl.set_task`; failed/blocked affect evidence only. `commit` records the code and binds local metadata to that commit without staging `.cswd`. Metadata-only attempts retain an empty code commit.

Provisional success:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome ready --summary '<factual provisional summary>'
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty commit '<task_path>' \
  --status ready --outcome '<concise behavioral outcome>'
```

Observed verification:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome verified --summary '<delivered behavior>' \
  --verification '<exact command or scenario> — <observed result>'
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty commit '<task_path>' --status verified
```

Repeat `--verification` for distinct observations. Once the final task subject exists, omit the commit `--outcome` to preserve it.

Failure or external blocker uses the unchanged `lifecycle_status` returned by `show`/`prepare`:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome failed --summary '<reachable retained work>' \
  --error '<operation — concrete failure>'
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty commit '<task_path>' \
  --status '<unchanged-lifecycle-status>' --outcome '<concise attempted outcome>'
```

Use `--outcome blocked` only for an external prerequisite. The unchanged status may be `new`, `critic`, `planned`, `ready`, or `verified`. A later failure can therefore be retained without falsifying an existing lifecycle, but failed/blocked evidence always prevents integration. The orchestrator records the local execution outcome.

## 5. Rebase and verify

Refresh the integration head with `inspect` or the orchestration helper, then rebase through the helper:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty rebase '<task_path>' --onto '<commit>'
```

On exit `3`, resolve only returned conflicts and call `continue-rebase`; use `abort-rebase` only to abandon that replay. If conflict resolution changes behavior, rerun affected verification and reconsolidate. Under `csw-run`, the root grants the original implementer an edit-only conflict lease; only the verifier continues replay after those edits stop. The owner reports continuation pending without annotating/committing mid-rebase, then resumes normal ready consolidation after replay under a new grant.

Children stop at lifecycle `ready`. The integration owner performs focused behavioral verification and repository-required combined verification from the exact task worktree: the parent for standalone `csw-run-worker`, the dedicated leaf `csw-verifier` for `csw-run`. The latter uses `csw_verify` for bounded gates, commit-bound review authorization, and locked integration; the root runs none of these operations. On failure, return concrete evidence to the same owner, amend its one commit, and rerun. On success, annotate `verified`, commit `--status verified`, and mechanically check:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty check '<task_path>' --status verified
```

For a container train, rebase task commits in helper order and refresh verification after replay. Combined verification runs from the final train worktree. If an earlier task changes, rebuild every later replay. Every final train commit must have matching verified controls and evidence in its owner's local helper state; task metadata is not stored in Git commits.

## 6. Container roll-up

When every implementation leaf beneath a requested container is either done in the shared task store or verified in the final train, the final leaf owner records each ancestor roll-up as verified before its final commit:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty annotate '<final-leaf-task>' \
  --for-task '<container-task-path>' --outcome verified \
  --summary '<all implementation leaves verified>' \
  --verification '<combined command or scenario> — <observed result>'
```

The helper allows only true ancestor containers inside the run target. Their verified controls and evidence are bound to the final leaf's single commit in local helper state. Integration advances those shared controls to done after the code fast-forward. Never create a separate roll-up/completion commit or stage task metadata.

If all implementation leaves are already done but a container roll-up is missing or stale, choose the stable last completed leaf as owner and invoke normal `prepare` with the container as `--run-target`. The helper permits only this completion repair, requires all descendants done, and returns `rollup_only: true`; it preserves dirty or unintegrated state rather than overwriting it. Do not implement anything. Record verified evidence for the owner and each stale ancestor with `annotate`, then `commit --status verified --outcome '<container completion outcome>'`, check, and integrate the single metadata-only task commit. The completed owner's lifecycle remains done throughout repair; `verified` describes its new evidence, not a regression of its shared status. No new implementation task or container worktree is needed.

## 7. Integrate

For a leaf or the final branch of a verified train:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty integrate '<task_path>'
```

`integrate` requires a clean fast-forward verified train. It validates each task commit's local metadata record against the shared controls and successful verified evidence, validates ancestor roll-up ownership, and rejects merges, duplicate/non-task commits, stale metadata, malformed controls, failed evidence, or non-verified status.

The helper rechecks the integration head and fast-forwards the existing code commits before advancing the shared controls to done through `task_ctl.set_task`. It neither rewrites code commits to embed lifecycle changes nor creates a completion commit. A failed fast-forward leaves lifecycle unchanged, so dependencies remain locked. Keep the local helper state and report any metadata finalization failure; do not repair lifecycle by hand.

If the integration branch advanced, refresh/rebase, rerun affected verification, refresh evidence, and retry. After each successful integration, inspect again and schedule newly eligible work from the canonical tree immediately. `csw-run` additionally requires review authorization for the exact final commit; do not bypass its verifier receipt by calling standalone `integrate`.

## Completion response

Report the target and integration branch; every leaf as already done, done/integrated, failed, blocked, or dependency-waiting; exact prepared worktree/branch/final commit for attempted leaves; observed verification; roll-up and integrated head; and retained errors or blockers. Never claim unobserved validation or partial completion.

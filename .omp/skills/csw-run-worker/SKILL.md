---
name: csw-run-worker
description: Implement one .cswd/tasks task or its unfinished implementation descendants in isolated reusable worktrees, using task_ctl-backed lifecycle controls and deterministic Git integration.
hide: true
---

# CSW Run Worker

Implement the requested specification completely. Each executable implementation task owns one deterministic feature branch, one registered worktree under `.work/`, sibling `spec.md`, `task.yml`, and `task.md` under `.cswd/tasks/`, and exactly one final code commit. A target with implementation descendants is a container: execute those leaves, never the container as implementation.

When invoked by `csw-run`, that skill's role boundaries take precedence: the root only orchestrates, its control-mode `csw-verifier` owns discovery/preparation, and one dedicated leaf `csw-verifier` owns rebase, independent final verification, review evidence, and integration. The original implementer owns code edits and mandatory focused testing before the ready handoff. Standalone `csw-run-worker` execution retains the workflow below.

## Mandatory control plane

Use the shared helper for every repeatable task, evidence, worktree, and Git-history operation:

```text
.omp/csw/bin/csw_run_worker --repo <integration-checkout> --pretty <command> ...
```

The helper loads `.omp/csw/bin/task_ctl` through its public `runpy.run_path` API. `task_ctl` alone discovers and validates canonical task metadata in `task.yml`: type, lifecycle status, numeric order, priority, optional implementer agent, canonical `blocked-by` records, and source. The helper owns execution evidence in `task.md`, local worktree state, and Git mechanics. PyYAML is therefore a runtime dependency of `task_ctl`; the supported environment provides PyYAML 6.0.3.

Never read metadata from Markdown. Never parse, sort, create, or edit `task.yml` yourself, and never ask a child agent to do so. Use helper JSON in its returned order. `task.md` is implementation evidence only: `**Outcome:**`, Summary, Verification, and Errors are not lifecycle metadata.

Missing or malformed `task.yml` is a hard error.

Do not replace a failed helper operation with hand-written Git plumbing or reconstructed paths. Never directly run `git worktree add`, `git add`, `git commit`, `git reset`, `git rebase`, `git merge`, or `git update-ref`. The helper is authoritative for prepare/reuse, checkpoint, one-commit consolidation, rebase conflict continuation, commit-bound local evidence validation, verified-to-done finalization, and fast-forward integration.

Exit status `2` is a validation or safety failure. Exit status `3` is a helper-managed rebase conflict; resolve only the named files in the owning worktree and call `continue-rebase`.

## Invariants

- The integration checkout is the canonical control plane. Do not edit, build, test, format, or run task code there.
- Only canonical `done` in the shared local task store satisfies a dependency. Missing dependencies wait. `ready`, `verified`, a child claim, or a private branch never unlocks an edge; the helper writes done only after successful integration.
- Lifecycle values are `new`, `critic`, `planned`, `ready`, `verified`, and `done`. `running`, `failed`, and `blocked` are execution outcomes only.
- Execution can start from `new`, `critic`, or `planned`, or resume from `ready`, once blockers are done. Successful implementation advances to `ready`; observed verification advances to `verified`; only `integrate` may advance a leaf to `done`.
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

Use `repo_root`, `integration_branch`, `integration_head`, `requested_target`, `target_kind`, target paths, and every `leaves[]` path and identifier verbatim. Each leaf includes `task_id`, `task_path`, `spec_path`, `control_path`, evidence path, exact worktree paths, branch, lifecycle status, canonical blocker records, `explicitly_ready`, and effective `implementer`. The latter is the task control's optional agent name, or `csw-implementer` when absent; `prepare` and `show` return it too. There is no runtime ancestor lookup: planning copies explicit parent selections into child controls.

For standalone execution, prepare and dispatch each eligible leaf using the task tool's `agent` set to its returned `implementer`, even for a single-leaf target; the invoker retains standalone verification/integration duties. An already-dispatched owner implements its assigned leaf directly and never redispatches itself. For a container, schedule every unfinished implementation leaf whose `explicitly_ready` is true. Skip only lifecycle `done`. A missing dependency remains waiting; only its canonical integrated `done` status satisfies it. Numeric order and priority are scheduling order, never inferred dependencies. Do not add an LLM-side sort or parse controls.

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

When verification uses remote hosts, first read `skill://csw-remote`. Prefer the integration checkout's `.omp/csw/bin/test_<backend> <unique-leaf-owner-attempt-mirror-prefix>` for a standard backend suite; it handles sync, configure/build/test, environment setup, deadlines, and automatic accelerator locking. Set Bash `cwd` to the exact assigned worktree, `CSW_REMOTE_TASK_DIR` to `dirname(spec_path)`, and `CSW_REMOTE_WORKSPACE` to the returned `worktree`. CPU and general unit tests MUST run through the `cpu` remote profile, never locally. The runners append `-<backend>` to the mirror prefix and write `remote.log` beside `spec.md` and `task.yml`; do not use a remote path or the worktree root as the evidence directory. For focused scenarios not covered by a standard runner, invoke the integration checkout's current `csw-remote-*` helpers with the same environment, a fresh sync before every exec, and confirm the exact assigned worktree. Accelerator scenarios must use the same bounded remote `/tmp/agent-gpu0.lock`; CPU scenarios take no device lock.

Remote sync excludes ordinary untracked files. Before syncing newly added source/tests, use helper `checkpoint` to make them tracked without marking the task ready; never stage directly or test a mirror missing those files. Checkpoints are temporary helper-owned history, consolidated by the final `commit`, not extra final task commits. When checkpoints exist, supply commit `--outcome` explicitly (reuse the existing final outcome on repairs); omission only works with exactly one existing final task commit. Temporary executable probes may use the remote skill's `_local/` exception; remove them before final consolidation.

Provision each ready leaf before dispatch and give one owner `MODE: owner`, the selected `implementer` and successful preflight record, its exact repo root, task IDs/paths, worktree paths, branch/base, exclusive scope, canonical dependency contracts, focused checks it must execute, and final verification still required. Use the same owner/result contract for all selected profiles: `TASK`, `STATUS`, `COMMIT`, `FILES`, `GATES`, `DEVIATIONS`, `OPEN`, and `RESCUE` as specified in `csw-run`. Keep the original profile and peer through repairs. Missing or incompatible profiles are blockers, not permission to fall back to the default or invoker model. Children MUST build and run focused tests during implementation, even when other children run concurrently; they do not own combined integration gates. They never edit controls or evidence directly and never mutate Git except through this helper. `csw-run` dispatches newly eligible leaves continuously, never in waves.

### Mandatory escalation when progress stalls

The protocol below applies to the selected profile's permitted rescue stages from preflight's `agents[implementer].spawns`. Default `csw-implementer` requires both stages unchanged. A nondefault profile may permit only `csw-debug` or no delegates; never expand its permissions. Last-resort Yoda requires the debugger stage and follow-through first. Stages excluded by the profile are **not applicable**, not attempted or completed; record that restriction in `RESCUE`. A selected `csw-yodacoder` is the original owner in `MODE: owner`, retains `spawns: []`, and must not dispatch itself or a debugger. A nondelegating owner performs its own investigation and may report a concrete unresolved implementation failure after evidence-driven attempts without fabricating rescue evidence. These selection rules qualify every rescue requirement below, including repair handoffs.

The implementation owner MUST dispatch its one `csw-debug` rescue as soon as it is stuck: after two distinct evidence-driven attempts fail to advance the same issue, or immediately when it cannot identify a safe next experiment. Do not wait until giving up. This applies to source implementation, build errors, unit/conformance failures, runtime defects, and verifier-requested repairs. Repeating the same failing command is not a new approach. Difficulty, uncertainty, and unexplained code/test failures are not external prerequisites and MUST NOT be relabeled `blocked` to bypass rescue.

First use the task tool with `agent: "csw-debug"`. The only permitted implementer delegates are this first-stage analyst and the subsequent terminal `csw-yodacoder` rescue below. The debugger profile is `.omp/agents/csw-debug.md`, routed through `@slow` with high reasoning, not the implementer's model or a default agent. Supply exact task/worktree/spec paths, scope and contracts, current changes, expected versus observed behavior, failing commands and diagnostics (including backend/profile and remote logs where relevant), attempted hypotheses/fixes and their results, and the precise unresolved question. Request read-only root-cause analysis and a concrete repair/verification plan. Do not request an isolated worktree, code edits, builds or tests from the debugger; the owner retains those duties.

Wait for the completed debugger result before deciding the task outcome. Dispatch intent, an unawaited job, or self-debugging is not escalation evidence. Keep the assigned source snapshot stable during analysis; then resume implementation, apply the proposal or reject it with concrete counterevidence, and rerun the affected checks. Preserve the actual debugger agent/job ID, report artifact, proposal disposition, and post-rescue commands/results through helper Summary/Verification/Errors and the result's `RESCUE` field, even when the rescue succeeds.

If the debugger cannot produce a usable diagnosis, its proposal is disproved or unsafe with concrete counterevidence, or the owner still cannot implement a working repair after follow-through, the owner MUST invoke `agent: "csw-yodacoder"` as the last resort before an implementation-failure handoff. This is `.omp/agents/csw-yodacoder.md` on `@csw-yoda` with high reasoning, never a substitute/default model. Do not invoke it before debugger follow-through or while the debugger is active. Supply the original failure packet plus the debugger identity/report, proposal disposition, subsequent attempts/results, exact remaining objective, and one explicit mode:

- `MODE: debug`: read-only root-cause analysis and concrete repair/check plan. Keep the source snapshot stable; the owner later implements and tests.
- `MODE: implement`: bounded in-place diagnosis, repair and focused testing in the same prepared worktree. The parent's implementation brief MUST explicitly authorize the original owner to sublease its current scope to Yoda after the first rescue fails. Under `csw-run`, root grants this authorization; standalone execution uses its existing implementation ownership. Pass that grant and any conflict-only restriction verbatim. The owner pauses all edits, tests and mutating helpers until Yoda returns; no verifier/reviewer may be active on the leaf. Yoda may use only helper `checkpoint` where required to sync new source/tests, never annotate lifecycle/evidence, consolidate the final commit, rebase or integrate. Conflict-only subleases permit only named file edits and defer checks/checkpoints until replay completes.

Wait for the completed Yoda result and confirmation that all owned local/remote processes have stopped before resuming or transferring ownership. Apply or reject a debug proposal with evidence, or inspect implementation-mode changes; the original owner MUST rerun affected implementer checks itself, record evidence and consolidate the one final task commit. A child result alone is not ready/verified authorization. On cancellation or a crashed child, retain its identity/artifacts and block transfer until cleanup is confirmed.

Each applicable stage is once per leaf across all retries, including later verifier repairs, not once per handoff or failing symptom. Retain reports and subsequent attempts; do not reset the stages, start another rescue identity or recurse from Yoda. A failed implementation report requires all applicable stages and owner follow-through to have left a real unresolved failure. `RESCUE` records `csw-debug` and `csw-yodacoder` separately: not needed with reason, not applicable with the profile restriction, or actual agent/job ID, report artifact, selected mode, proposal/change disposition, post-rescue commands/results and cleanup state. Preserve those facts through helper Summary/Verification/Errors even when rescue succeeds.

Demonstrated external prerequisites (unavailable hardware, SSH, SDK, or required external information) may be reported blocked without rescue, with evidence. If either rescue's dispatch or model/tool access fails, report that exact attempted operation/error as a blocker; never silently substitute a model, claim rescue completed, or continue to an unrescued implementation-failure handoff. This infrastructure failure is not an exhausted diagnosis permitting the next stage.

For `csw-run`, reuse the control peer's successful preflight record forwarded by the root. Otherwise run `.omp/csw/bin/csw_preflight --repo <repo> --workflow csw-run-worker --implementer <agent-name> ... --pretty` once before implementation dispatch, repeating `--implementer` for every distinct effective selection among all unfinished inspected leaves, including those waiting on dependencies. It validates selected implementation profiles plus both rescue profiles and aliases, including `@csw-yoda`. Forward the successful record to each owner; owners do not rerun it. A failed preflight is retained failure evidence; never substitute another profile/model. A new selection absent from that record requires a new invocation, not an unvalidated dispatch.

### Required implementer checks before ready

Testing is part of implementation, not work deferred entirely to a verifier. The owner MUST execute real checks, diagnose failures, fix its code, and rerun affected checks after the final edit and each repair before handing off:

- Backend-specific code: run `.omp/csw/bin/test_<backend>` from the integration checkout for **each touched backend**, selecting the exact worktree/evidence directory as above; add any focused regression scenario the suite does not exercise. A CPU pass does not validate CUDA, ROCm, or SYCL code.
- Backend-neutral code: run relevant unit/regression tests and conformance on **at least one supported remote backend** that exercises the change. Use `test_cpu` for general/CPU unit tests; no local CPU substitute. State the selection and coverage limits. The integration owner still runs all four standard backend scripts before integration.
- Exercise the changed behavior, not just configuration, compilation, test discovery, or unrelated smoke tests. Use existing tests and keep a regression test where a plausible bug warrants it; if existing tests miss the behavior, run a focused executable scenario. Report actual executed tests/results; an empty selection, disabled tests, or skipped hardware checks is not a pass.
- Documentation/workflow-only changes with no backend behavior use the relevant executable helper checks instead; explain the backend-test non-applicability. Do not invent a backend test merely to fill the report.

The standard `test_<backend>` scripts are directly callable for implementer checks; `csw_verify` and its final receipts remain the verifier's responsibility under `csw-run`. Do not reconstruct standard build/test recipes or manually create their SYCL overrides. Additional focused scenarios use `csw-remote` sync/exec: keep build artifacts ignored or external, bound builds to 1800 seconds and tests to 900 seconds with remote-side `timeout --kill-after=30s ...`, and use CTest `--timeout 300` plus `--no-tests=error`. Acquire `/tmp/agent-gpu0.lock` with bounded `flock -w` for accelerator tests, never CPU tests; independent mirrors do not eliminate shared-device contention. No detached tests or lock held across editing/handoffs. Supplemental SYCL executions still need the outside-checkout no-setup profile override and nounset-safe initialization on **every** remote command.

Do not return `ready` with failing or unexecuted required implementer checks. Code/test failures stay with the owner for repair, first-stage `csw-debug` rescue and then last-resort `csw-yodacoder` when debugger follow-through fails; missing hardware, SSH access, SDKs, or other external prerequisites produce `blocked` with concrete evidence. Preserve failures even after recovery: backend/profile, mirror, sync/exec operation and exact command, exit/timeout, diagnostic excerpt, log paths including `remote.log`, and cleanup state. Stop owned processes, including any rescue child's processes, and confirm remote execution has stopped before relinquishing the worktree; unconfirmed cleanup blocks transfer. The edit-only rebase-conflict lease remains an exception: defer checks until replay completes and the root grants normal implementation ownership again.

## 4. Record evidence and consolidate

Never edit `task.md` or `task.yml` directly. `annotate` preserves unrelated task prose and replaces generated Outcome/Summary/Verification/Errors in the shared `.cswd/tasks/` store. For `ready` and `verified`, it delegates the lifecycle write to `task_ctl.set_task`; failed/blocked affect evidence only. `commit` records the code and binds local metadata to that commit without staging `.cswd`. Metadata-only attempts retain an empty code commit.

Provisional success after the required implementer checks pass (still `ready`, not final verification):

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty annotate '<task_path>' \
  --outcome ready --summary '<factual provisional summary>' \
  --verification '<implementer: backend/profile, exact command — observed result, log path>'
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty commit '<task_path>' \
  --status ready --outcome '<concise behavioral outcome>'
```

Observed final verification by the integration owner:

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

On exit `3`, resolve only returned conflicts and call `continue-rebase`; use `abort-rebase` only to abandon that replay. The helper retains old/new commit, base and tree plus conflict paths in `rebase_history`, including conflicts resolved through continuation; aborted replay never counts as clean provenance. Under `csw-run`, the root grants the original implementer an edit-only conflict lease; only the verifier continues replay after those edits stop. The owner reports continuation pending and explains the resolution and any algorithmic/structural impact in `OPEN`, without annotating/committing mid-rebase, then resumes normal ready consolidation and required checks after replay under a new grant. The verifier independently reruns unit/regression and conformance tests. Previously approved conflict-free rebases retain dual-review approval; conflicted rebases receive a bounded verifier assessment and require another full pair only for very complex, substantial algorithmic or structural changes. The owner's assessment is context, never authorization.

Children stop at lifecycle `ready` **after** required implementer checks pass and their evidence is recorded. The integration owner independently performs focused and combined verification from the exact task worktree: the parent for standalone `csw-run-worker`, the dedicated leaf `csw-verifier` for `csw-run`. Implementer passes never replace final commit-bound gates or all-backend coverage. Under `csw-run`, invoke the integration checkout's `csw_verify verify <task_path> --backend-tests <unique-leaf-attempt-mirror-prefix> --summary <behavior>` (with `--repo <integration-checkout>` before `verify`); it runs the four standard backend scripts concurrently and owns receipts, review authorization, and locked integration. An optional `--plan` adds only specification gates missing from the scripts, after all four pass. Standalone combined verification also runs those four scripts concurrently with exact worktree/task environment. All CPU/general unit tests use the CPU remote host. Preserve `remote.log` and every failing backend's profile, operation/command, exit/timeout, diagnostic, gate log, and cleanup state; an inaccessible child session or bare artifact path is not an error report. On failure return concrete evidence to the same owner, amend its one commit, and rerun. On success, annotate `verified`, commit `--status verified`, and mechanically check:

```text
.omp/csw/bin/csw_run_worker --repo '<repo_root>' --pretty check '<task_path>' --status verified
```

For a container train, rebase task commits in helper order and refresh verification after replay. Combined verification runs from the final train worktree. If an earlier task changes, rebuild every later replay. Every final train commit must have matching verified controls and evidence in its owner's local helper state; task metadata is not stored in Git commits.

## 6. Container roll-up

Under `csw-run`, skip the standalone protocol below: `csw_run queue` completes the requested HLD through `task_ctl` once all direct children are canonically done. It also repairs an already-completed selection without roll-up annotations, worktree preparation, or another commit.

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

If the integration branch advanced, refresh/rebase, rerun required unit/regression and conformance tests, refresh evidence, and retry. After each successful integration, inspect again and schedule newly eligible work from the canonical tree immediately. `csw-run` retains completed dual-review approval across helper-recorded clean rebases and verifier-assessed simple conflict resolutions, but still requires fresh test evidence and review authorization rebound to the exact final commit. Follow its review-reuse policy; do not dispatch a new pair merely because the hash changed or bypass the verifier receipt by calling standalone `integrate`.

## Completion response

Report the target and integration branch; every leaf as already done, done/integrated, failed, blocked, or dependency-waiting; exact prepared worktree/branch/final commit for attempted leaves; observed verification; roll-up and integrated head; and retained errors or blockers. Never claim unobserved validation or partial completion.

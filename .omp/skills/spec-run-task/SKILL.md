---
name: spec-run-task
description: Implement one change task or its unfinished leaf subtasks from docs/changes in isolated reusable worktrees, using the bundled deterministic helper for discovery, paths, annotations, Git history, rebases, and fast-forward integration. Use only through /spec-run-task <change-name>[/subpath] or when explicitly requested.
hide: true
---

# Spec Run Task

Implement the requested specification completely. Each executable leaf owns one deterministic feature branch, one registered worktree under `.work/`, one sibling `task.md`, and exactly one final task commit. A target with descendant specifications is a container: execute its unfinished leaf specifications, never the container's own implementation.

## Mandatory mechanical control plane

Use the bundled helper for every repeatable path, annotation, worktree, and Git-history operation:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo <integration-checkout> --pretty <command> ...
```

The script is authoritative for:

- validating the invocation, repository, named integration branch, clean checkout, and ignored `.work/`;
- resolving `spec.md`, sibling `task.md`, executable leaves, statuses, explicit blockers, dependency cycles, feature branches, and worktree paths;
- creating or conservatively reusing worktrees and persisting each task's exact replay base in its Git administrative directory;
- reporting reusable worktree state;
- writing annotations at computed sibling paths while preserving unrelated sections;
- checkpointing dirty reusable state, consolidating all owned work into one task commit, rebasing that commit, validating it, and fast-forwarding the integration branch.

Do not replace a failed helper operation with hand-written Git plumbing or a manually reconstructed path. Correct the reported input or owned file state, then rerun the helper. In particular, never directly run `git worktree add`, `git add`, `git commit`, `git reset`, `git rebase`, `git merge`, or `git update-ref` for this workflow. Never manually create or edit a `task.md`; use `annotate`.

The model remains responsible only for work that requires judgment: reading specifications and source, adding genuine semantic dependency edges, implementing behavior, running verification, attributing failures, and resolving the contents of rebase-conflicted files. The helper stages conflict resolutions and continues the rebase.

Exit status `2` means a validation or safety invariant failed. Exit status `3` means a rebase stopped for conflicts; resolve only the named files in the owning worktree and invoke `continue-rebase`.

## Invariants

- The integration checkout is a read-only control plane. Use it only to invoke the helper and for read-only discovery. Never edit, build, test, format, or run task code there.
- Capture the integration branch and head from `inspect`. Every later path, branch, worktree, and commit identifier comes from helper JSON; do not derive substitutes.
- All task Read, Edit, Write, and LSP paths use the exact worktree-prefixed paths returned by `prepare`. Every task Bash command sets `cwd` to that worktree.
- The exact feature branch is `run-task/<path-components-joined-by-->`; the exact worktree is `.work/<task-path>`. The helper validates rather than slugifies invalid components.
- Reuse registered state. Never stash, clean, remove, or recreate an existing task worktree or branch.
- Each attempted leaf retains exactly one final non-merge commit with subject `spec-run-task(<full-task-path>): <outcome>`. Failed and blocked attempts also retain their state in that one commit, but are not integrated.
- Rebase task commits; never merge branches together. The integration checkout advances only through the helper's verified fast-forward.
- Only `done` is complete. `ready` means implementation and focused evidence may exist but combined verification or integration is pending.
- Do not integrate failed, blocked, partial, dirty, conflicted, or unverified work.
- Keep worktrees and feature branches after completion as resumable task state.
- Treat repository guidance and applicable change specifications as acceptance criteria. Read any additionally applicable skill before implementation or verification, including `remote-development` for accelerator work.

## 1. Inspect the requested target

Require exactly one target argument relative to `docs/changes/`, then run:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo . --pretty inspect '<target>'
```

Do this before reading task contents or provisioning worktrees. The helper rejects absolute paths, file names, traversal, malformed annotations or blockers, detached or dirty integration checkouts, unsafe branch components, unignored `.work/`, missing specs, and explicit dependency cycles.

Use these JSON fields verbatim:

- `repo_root`, `integration_branch`, and `integration_head` identify the control plane;
- `requested_target`, `target_kind`, `target_spec_path`, and `target_annotation_path` identify the request;
- each `leaves[]` record supplies `task_path`, source paths, exact worktree paths, feature branch, status, resolved blockers, and `explicitly_ready`.

If `target_kind` is `leaf`, the main agent owns and implements it on the current session model; do not replace it with the `spec-run-all` implementer profile or another model-routed implementation agent. If it is `container`, the main agent orchestrates general-purpose subagents, one owner per ready leaf. Skip a leaf only when its sibling annotation status is `done`.

## 2. Complete the dependency graph

The helper resolves the mechanical graph from `**Blocked by:**` fields. Read every unfinished executable leaf spec from the returned `spec_path` before provisioning. Add a semantic edge only when a leaf consumes an interface, artifact, migration, or repository state produced by another leaf. Numeric prefixes, order, and priority are stable scheduling tie-breakers, not dependencies.

Reject unresolved semantic ambiguity before edits. Topologically schedule unfinished leaves in stable dependency/priority/path order. A wave contains leaves whose explicit and semantic blockers are satisfied by the integration branch and which have no dependency edges between one another. Continue independent work when another leaf fails; do not run its descendants.

## 3. Prepare an executable leaf

For every ready leaf, invoke `prepare` with values copied from the same `inspect` result:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty prepare '<task_path>' \
  --run-target '<requested_target>' \
  --integration-branch '<integration_branch>' \
  --integration-base '<integration_head>'
```

`prepare` creates or reuses only the exact deterministic branch/worktree, verifies repository identity and branch attachment, and records the task base. It stops on path/ref collisions or an unregistered non-empty destination.

Treat returned `worktree`, `spec_path`, and `annotation_path` as opaque exact paths. Before any implementation action, read the complete returned worktree `spec_path`, repository guidance, relevant implementation/tests, and applicable skills.

For a reused worktree, inspect `status_entries` and `task_commits` returned by `prepare`, or refresh them mechanically:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty show '<task_path>'
```

Determine whether every retained change belongs to this exact task. If ownership is uncertain, stop rather than rewriting it. If coherent reusable state is dirty and must be protected, invoke:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty checkpoint '<task_path>'
```

Do not checkpoint a path merely to make an error disappear. The helper never stashes or discards work. A checkpoint is temporary and must later be consolidated by `commit` before rebase, terminal status, or integration.

## 4. Implement only in the assigned worktree

Use the prepared worktree as the sole project root:

- prefix every Read, Edit, Write, and LSP path with the returned `worktree`;
- set every Bash `cwd` to the returned `worktree`;
- reproduce bugs there when required and reachable;
- implement every requirement and acceptance criterion without unrelated cleanup;
- keep implementation, tests, generated repository artifacts, and annotation in the same task commit.

For a container wave, provision every leaf before spawning. Launch one general-purpose subagent per leaf in one parallel Task batch. Do not use automatic isolated-agent worktrees or a model-routed `spec-run-all` implementer profile. Give each owner the exact `repo_root`, `task_path`, `worktree`, `spec_path`, `annotation_path`, feature branch, integration branch/head, exclusive files/interfaces, sibling contracts, focused verification still required, and the stuck-implementation escalation below.

A child must:

1. verify state with `show`;
2. work only in the assigned worktree;
3. implement the complete leaf;
4. skip formatters, builds, tests, and project-wide validation during the parallel implementation pass;
5. use `annotate --status ready` and `commit --status ready --outcome '<behavioral outcome>'`;
6. return the helper's commit, changed paths, retained risks, and verification still required.

No child may issue direct Git mutation commands or edit an annotation by file operation.

### Stuck implementation escalation

An implementation owner that is stuck and would otherwise give up or report an implementation failure must first invoke exactly one `spec-run-debug` subagent for that leaf. This is the only permitted nested delegation. The profile requests the `@slow` model and is a read-only rescue analyst; the implementation owner remains responsible for all edits and decisions.

Pass the debugger the exact worktree and specification paths, task scope, relevant repository constraints, current changes, concrete error or dead end, observations, and approaches already attempted. Use `agent: "spec-run-debug"`, do not request another isolated worktree, wait for its result, then resume implementation using the supported proposed solution or explain with evidence why it cannot work. Do not invoke the debugger for normal planning, review, verification failures owned by the parent, or an external prerequisite that makes the task `blocked`. Do not repeat the escalation for the same leaf.

Before dispatch, verify that `.omp/agents/spec-run-debug.md` declares `model: "@slow"` and that effective model-role or per-agent overrides do not route it incompatibly. If the profile or model is unavailable, retain the concrete dispatch failure; never substitute another profile or silently continue to an implementation-failed result. The debugger must return root cause evidence and a proposed solution directly to the requesting implementer and must not edit, annotate, commit, rebase, integrate, or delegate.

## 5. Write state and consolidate the task commit

The helper computes the annotation as the sibling of the owned `spec.md`. Never pass a `task.md` path.

Provisional success:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<task_path>' \
  --status ready --summary '<factual provisional summary>'
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty commit '<task_path>' \
  --status ready --outcome '<concise behavioral outcome>'
```

Observed completion:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<task_path>' \
  --status done --summary '<delivered behavior>' \
  --verification '<exact command or scenario> — <observed result>'
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty commit '<task_path>' --status done
```

Repeat `--verification` for distinct checks. Once a final task commit exists, omit `--outcome` to preserve its subject automatically.

Failure or blocker:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<task_path>' \
  --status failed --summary '<reachable retained work>' \
  --error '<operation or prerequisite> — <concrete failure>'
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty commit '<task_path>' \
  --status failed --outcome '<concise attempted outcome>'
```

An implementation failure status is valid only after the required `spec-run-debug` rescue attempt was completed and its proposed solution was attempted or rejected with concrete evidence. An unavailable debugger/profile is itself a concrete retained failure. External prerequisites may be marked `blocked` without invoking the debugger.

Use `blocked` instead of `failed` only for an external prerequisite. Repeat `--error` when needed. The helper replaces stale generated Summary, Verification, and Errors sections, preserves unrelated sections, rejects misplaced annotations, stages the complete worktree, refuses unknown commits, and safely consolidates recognized checkpoints/task commits onto the recorded base.

## 6. Rebase with the helper

Before verification/integration, refresh the integration head with `inspect`. Rebase one finalized task commit onto that head or the current wave-train commit:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty rebase '<task_path>' --onto '<commit>'
```

The helper replays only the recorded task commit and updates its persisted base. It refuses multiple commits, dirty worktrees, unknown history, and replay cycles. A clean branch with no task commit may only move forward to a descendant base.

On exit `3`, edit conflict contents only in the returned owning worktree, then invoke:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty continue-rebase '<task_path>'
```

Repeat if another conflict is reported. Use `abort-rebase` only when the attempted replay must be abandoned. Never stage or continue manually. After any conflict resolution, rerun affected verification and reconsolidate with `commit` if annotation evidence or implementation changed.

Validate the result mechanically:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty check '<task_path>' --status ready
```

`check` proves the worktree is clean, the recorded base is valid, exactly one non-merge task commit exists, its subject names the exact full task path, its diff contains the sibling annotation, annotation changes are owned, and the requested status matches.

## 7. Verify and assemble a container wave

After the implementation barrier:

1. Run each leaf's focused behavioral verification from its assigned worktree. Use the actual runtime surface, hardware, sanitizer, or remote procedure required by the spec. A compile is not runtime proof.
2. Route a concrete failure to the same owner. The owner fixes only its worktree, updates its annotation, and invokes `commit` to amend the one task commit. Rerun the failed check.
3. Keep successful leaves `ready` until repository-required combined verification passes. Mark irrecoverable leaves `failed` or `blocked`; retain but do not integrate them.
4. Assemble ready commits in stable order. Rebase the first onto the current integration head. For every next owner, rebase its task onto the preceding helper-returned `commit`. The last branch is the provisional wave train.
5. Run repository-required combined verification once from the final train worktree, including applicable backend conformance suites. On failure, send evidence to the responsible owner, amend that owner's one commit, rebuild every later replay with `rebase`, and rerun affected checks.
6. After combined success, rebuild the finalized train in the same order. For each owner: rebase onto the latest finalized commit, `annotate --status done` with grounded focused and combined evidence, then `commit --status done`. Use each returned `commit` as the next base.
7. If rebase conflict resolution changes implementation, rerun affected focused and combined verification before integration.

This rebuild ensures every final annotation is inside its own one-commit leaf change without separate completion or fix commits.

## 8. Roll up container annotations

When every executable leaf beneath the requested container is `done` in the finalized train, the final leaf owner writes roll-ups before its final `commit` invocation. Work bottom-up through every container spec at or below the requested target:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty annotate '<final-leaf-task>' \
  --for-task '<container-task-path>' --status done \
  --summary '<all leaf subtasks completed>' \
  --verification '<combined command or scenario> — <observed result>'
```

The helper permits only true ancestor containers within the requested run target and requires their sibling specs. It folds these roll-ups into the final leaf's task commit; no separate roll-up commit.

If all leaves were already `done` but a container roll-up is stale or missing, choose the stable last completed leaf as owner, prepare/rebase its normal branch, use `annotate --for-task`, and create one annotation-only task commit through `commit`. Do not create a container worktree that collides with descendant worktrees.

Do not mark a container `done` while any leaf is failed, blocked, or unfinished.

## 9. Integrate mechanically

For a single leaf, finalize it as `done`, run `check --status done`, then integrate that branch. For a container, use the final task branch in the fully verified train:

```text
python3 .omp/skills/spec-run-task/scripts/spec_run_task.py --repo '<repo_root>' --pretty integrate '<final-task-path>'
```

`integrate` rechecks the integration branch and cleanliness, requires the train to fast-forward the current integration tip, rejects merges/non-task/duplicate task commits, requires every commit to contain its own `done` sibling annotation, validates container annotation ownership, then performs the fast-forward.

If the integration branch advanced, do not work around the refusal. Rebuild the train from the new integration head with `rebase`, rerun affected focused and combined verification, refresh annotation evidence when it changed, and retry `integrate`.

After each integrated wave, rerun `inspect` and schedule the next newly ready wave from the new integration head.

## Completion response

Report concisely:

- requested target and integration branch;
- each leaf as `already done`, `done and integrated`, `failed`, `blocked`, or `not run because <blocker>`;
- for attempted leaves, exact helper-returned worktree, feature branch, single final task commit, and observed verification;
- container roll-up and final integration commit;
- retained error state or residual blockers.

Do not call partial work complete or claim unobserved verification.

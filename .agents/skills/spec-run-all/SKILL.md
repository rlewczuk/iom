---
name: spec-run-all
description: Execute the direct executable subtasks of docs/changes/<task-name> in dependency-aware barrier waves, using spec-run-task's worktree and Git control plane.
hide: true
---

# Spec Run All

Run all unfinished **direct child** tasks below `docs/changes/<task-name>`. Read `skill://spec-run-task` first: this is an orchestration wrapper around that workflow, not a second implementation workflow. The target directory itself does not need a `spec.md`, and nested specifications are not enumerated as additional direct tasks. Run only the user's requested target; `01-review` is an example, not an implicit invocation.

## Scope and mechanical control plane

The bundled script is the only source of truth for discovery, metadata, queueing, and preparation:

```text
python3 .agents/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty scan <task-name>
python3 .agents/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty queue <task-name> [--state <temporary-json-path>]
python3 .agents/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty prepare <task-name> [--state <temporary-json-path>]
```

`--state` is accepted only by `queue` and `prepare`. Use the integration checkout as `<repo>` and preserve the script's JSON values verbatim. Do not duplicate its directory, annotation, dependency, or status logic in instructions or ad-hoc shell/Python.

`scan` reads metadata only. It does not require a clean Git checkout and returns `repo_root`, `target`, and alphabetically sorted `tasks[]`. Each task contains its child basename `name`, `task_path` relative to `docs/changes`, `spec_path`, `annotation_path`, `status`, `annotation_exists`, and raw `blocked_by`. Only immediate child directories containing `spec.md` are tasks. A parent `spec.md` is neither required nor executed.

`queue` reads the same direct children and returns:

- `ready`: canonical names in the next barrier wave, alphabetically ordered, with at most five;
- `waiting`: names and concrete dependency causes;
- `blocked`: names and concrete nondependency or invalid-metadata reasons;
- `already_done`: names whose canonical sibling `task.md` is done;
- `finished`: true only when all discovered tasks are done (including an empty target); stalled or blocked work is not success.

`prepare` selects exactly that queue wave, validates the integration control plane, and sequentially calls the existing `spec-run-task` `prepare_task` helper with `run_target=task_path` for each leaf. It creates or reuses each task's deterministic worktree/branch and returns `integration_branch`, `integration_head`, and `prepared[]`. Each prepared record includes the exact `prepare_task` output; use its opaque `worktree`, `spec_path`, `annotation_path`, branch, and task identifiers without deriving alternatives. A preparation error is retained as that leaf's blocked diagnostic while independent leaves remain reachable. If an immediate child has descendant `spec.md` files, treat it as an explicitly unsupported nested-container blocker for this command; do not silently run its container implementation or skip its descendants.

## State file and dependency policy

Create one temporary state JSON outside the checkout and make the parent the only writer. Initialize it with `{}` (not an empty file), or use this structure:

```json
{
  "dependencies": {"child-name": ["canonical-sibling-name"]},
  "outcomes": {"child-name": {"status": "blocked|failed|ready", "reason": "nonempty explanation"}}
}
```

Never store orchestration state in `docs/changes` or a task worktree. Preserve the state for the whole invocation and remove it only as ordinary temporary-file cleanup after the final report; never alter user files to do so.

Dependency interpretation is conservative. The script accepts `None`, comma-separated names, backtick-quoted names, and explicit task paths, while preserving the raw field. External task paths are checked for completion but never added to the execution scope. Unsupported or ambiguous syntax fails closed until the parent interprets it: write an explicit `dependencies` entry with canonical sibling names, then rerun `queue`. Preserve every actual prerequisite; never replace an unresolved dependency with an empty list merely to unblock work. Missing dependencies remain waiting with the missing name. Only the dependency's canonical sibling `task.md` status `done` satisfies an edge; `ready`, a child claim, an existing commit, or an unintegrated branch does not.

`outcomes` prevents automatic retry after a runtime failure or explicit block. Record a nonempty, concrete reason. An outcome suppresses reattempt in this invocation unless the canonical task annotation is now `done` (for example, after the parent completes recovery); never loop on a failed leaf. Dependency cycles and stalled prerequisites are diagnosed by `queue`; continue independent work and report the cycle/stall separately from nondependency blockers.

## Barrier-wave orchestration

1. Run `scan` once for target metadata, then initialize the outside-checkout state file. Run `queue`. Interpret unresolved dependency fields into state where possible and rerun `queue`; never hand-sort the wave. If `ready` is empty, finish only when `finished` is true, otherwise report the remaining blockers. Do not prepare an empty wave.
2. Before any child starts, verify that `.omp/agents/spec-run-all-implementer.md` is available, declares the exact `@implementer` model alias, and permits only `spec-run-debug` as its nested spawn. Verify that `.omp/agents/spec-run-debug.md` is available and declares the exact `@slow` model alias. Check effective model roles and per-agent overrides for both profiles; if a profile or alias is unavailable or overridden incompatibly, surface that concrete dispatch failure in state and final blockers, never silently fall back to another model/profile. Then call `prepare`.
3. Call `task` with one item per prepared leaf in a single `tasks[]` batch, setting **`agent: "spec-run-all-implementer"`** on every item. This profile requests **`model: "@implementer"`**; do not substitute a generic agent or a prompt asking another model to act as an implementer. Preserve alphabetical ready-name order. Even one ready leaf runs in a child. Never exceed five children, including repair attempts. Disable automatic isolated-agent worktrees: the helper already prepared them. There is no running-list/refill behavior: the wave is a fixed barrier.
4. Pass each child the exact `repo_root`, prepared paths and task identity, integration branch/head, exclusive task scope, known cross-task interface contracts, and the stuck-implementation rescue contract below. The child must use assigned-worktree mode below. It must not delegate except to one `spec-run-debug` rescue agent when stuck, implement nested descendants, integrate, or perform validation.
5. Record every preparation failure in `outcomes` as `blocked`, including failures where no worktree could be prepared. Wait until every dispatched child and any nested debugger it invokes have ended. Record child runtime failures as `failed`, external prerequisites as `blocked`, and provisional successes as `ready` only when no same-wave preparation refresh is needed. An owner about to report an implementation failure must first complete the required debugger rescue; agent crashes and dispatch failures remain concrete runtime failures. Retain reasons, debugger findings when used, and returned worktree/commit evidence. Independent tasks remain eligible in later queue calls.
6. Only after the implementation barrier, the parent performs focused verification for each `ready` leaf in its exact prepared worktree and attributes failures to that leaf. Do not claim success from an unrun check. Failed or blocked leaves are retained and not integrated; the combined repository verification is run by the parent on the finalized train in the next step.
7. For the surviving ready leaves, the parent alone follows spec-run-task's rebase/train/integration protocol using the validated `integration_branch` and wave `integration_head` returned by `prepare`: rebase each finalized task commit onto that head and then the preceding train commit, resolve conflicts only in the named worktree through `continue-rebase`, run combined verification on the final train, mark final annotations `done`, mechanically `check`, and `integrate` with the helper. Do not use `spec-run-task inspect` for this orchestration target (it requires a root spec and can misresolve a selected leaf); if the same wave needs a refreshed control-plane head, call `prepare` again before outcomes suppress it. Do not run direct Git mutation commands. No container rollup is created unless the actual specifications and existing helper ownership require one.
8. After a successful integrated wave, rerun `queue` against the updated canonical tree. Continue fixed alphabetical dependency-aware waves until `finished` or no task can progress. A single ready leaf still uses the child agent and the same parent verification/integration barrier.

The parent must observe canonical `task.md` status `done` after integration before scheduling dependents. Do not schedule a task merely because another child says it is ready or because its branch exists. If a verification, rebase, train, integration, or dispatch failure cannot be recovered, retain it in state and let independent waves continue; descendants remain dependency-waiting.

## Child brief and existing worktree behavior

The child receives the exact `repo_root`, `worktree`, `task_path`, `spec_path`, `annotation_path`, feature branch, integration branch/head, exclusive task scope, and known interface contracts. It reads `skill://spec-run-task` explicitly and follows section 4's assigned-worktree child mode, including its stuck-implementation escalation. Invoke the existing helper with `--repo '<repo_root>'` from the assigned worktree; never reinterpret that worktree as the integration checkout:

- use `show` first and work only under the exact prepared worktree;
- read the complete spec, applicable repository guidance, relevant source, and skills;
- implement the complete leaf and include owned artifacts/annotation in the same task commit;
- skip builds, tests, linters, formatters, and all validation during the parallel pass;
- use the helper's `annotate --status ready` and `commit --status ready --outcome ...` operations;
- never use direct Git mutation, manually edit `task.md`, integrate, or rebase; never delegate or run a nested task except for the one permitted `spec-run-debug` rescue when stuck;
- return the helper commit, changed paths, retained risks, and verification still required.

An existing registered worktree is resumable state. The child must inspect `show` and preserve coherent owned changes; it must not stash, clean, recreate, or overwrite uncertain state. Helper checkpoint/annotate/commit operations are allowed and required where applicable. The child agent profile intentionally permits these helper invocations; the generic builder prohibition on commits is not inherited here.

### Stuck implementation rescue

When an `@implementer` owner is stuck and would otherwise give up or return `failed`, it must invoke exactly one task with `agent: "spec-run-debug"` and wait for the result. It supplies the exact assigned worktree/specification, task scope and constraints, current changes, concrete error or dead end, observations, and attempted approaches. It does not request an isolated worktree. The `@slow` debugger performs read-only root-cause analysis and returns evidence plus a proposed solution directly to that owner; it does not edit, annotate, commit, validate, rebase, integrate, or delegate.

The same `@implementer` owner then resumes implementation and owns every edit and judgment. It may give up only after attempting the proposed solution or rejecting it with concrete evidence. Do not spawn the debugger for ordinary planning, review, parent-owned verification, or an external prerequisite, and do not repeat the rescue for a leaf. A missing or incompatible debugger profile/model is a concrete retained dispatch failure, not permission to substitute another agent.

## Reporting

The final report must separately list, with concrete reasons:

- already done;
- done and integrated (including integrated commit and observed verification);
- failed (runtime or verification failure, with retained outcome);
- nondependency blocked (invalid metadata, unsupported nested container, unavailable profile/model, preparation/rebase/train/integration failure, or other concrete prerequisite);
- dependency waiting (the exact unresolved sibling, cycle, or stalled cause).

Report only observed results. Do not call work complete when `finished` is false or any task remains blocked/waiting. Include the target, integration branch/head as returned by helpers, prepared worktrees/commits for attempted leaves, and residual risks. Validation/build/lint/test/formatter commands are intentionally skipped during the parallel child pass; the parent may run only the focused and combined verification demanded by spec-run-task after the barrier.

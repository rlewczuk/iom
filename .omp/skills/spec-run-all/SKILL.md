---
name: spec-run-all
description: Execute all eligible direct executable subtasks of docs/changes/<task-name> in continuous dependency-aware parallel execution, using spec-run-task's worktree and Git control plane.
hide: true
---

# Spec Run All

Run all unfinished **direct child** tasks below `docs/changes/<task-name>`. Read `skill://spec-run-task` first: this is an orchestration wrapper around that workflow, not a second implementation workflow. The target directory itself does not need a `spec.md`, and nested specifications are not enumerated as additional direct tasks. Run only the user's requested target; `01-review` is an example, not an implicit invocation.

## Scope and mechanical control plane

The bundled script is the only source of truth for discovery, metadata, queueing, and preparation:

```text
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty scan <task-name>
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty queue <task-name> [--state <temporary-json-path>]
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty prepare <task-name> [--state <temporary-json-path>]
python3 .omp/skills/spec-run-all/scripts/spec_run_all.py --repo <repo> --pretty control
python3 .omp/skills/spec-run-all/scripts/check_omp_config.py --repo <repo> --pretty
```

Run the preflight exactly once per invocation, before the first `prepare`. It is the sole OMP configuration check: it invokes `omp config list --json` and `omp models --json` from `<repo>`, reads only the two project-owned agent profile Markdown files, and returns deterministic JSON. Preserve that JSON verbatim. Exit status `0` means `ok: true`; exit status `2` is a concrete blocked outcome/final blocker and must not be bypassed. Do not inspect OMP configuration files, SQLite databases, or separately invoke `omp config`/`omp models` for this purpose.

`--state` is accepted only by `queue` and `prepare`. Use the integration checkout as `<repo>` and preserve the script's JSON values verbatim. Do not duplicate its directory, annotation, dependency, or status logic in instructions or ad-hoc shell/Python.

`scan` reads metadata only. It does not require a clean Git checkout and returns `repo_root`, `target`, and alphabetically sorted `tasks[]`. Each task contains its child basename `name`, `task_path` relative to `docs/changes`, `spec_path`, `annotation_path`, `status`, `annotation_exists`, and raw `blocked_by`. Only immediate child directories containing `spec.md` are tasks. A parent `spec.md` is neither required nor executed.

`queue` reads the same direct children and returns:

- `ready`: all eligible canonical names, alphabetically ordered, with no skill-imposed concurrency cap;
- `waiting`: names and concrete dependency causes or retained `running`/`ready` outcomes awaiting owner completion or parent integration;
- `blocked`: names and concrete nondependency or invalid-metadata reasons;
- `already_done`: names whose canonical sibling `task.md` is done;
- `finished`: true only when all discovered tasks are done (including an empty target); stalled or blocked work is not success.

`prepare` selects all currently eligible tasks, validates the integration control plane, and sequentially calls the existing `spec-run-task` `prepare_task` helper with `run_target=task_path` for each leaf. It creates or reuses each task's deterministic worktree/branch and returns `integration_branch`, `integration_head`, and `prepared[]`. Each prepared record includes the exact `prepare_task` output; use its opaque `worktree`, `spec_path`, `annotation_path`, branch, and task identifiers without deriving alternatives. A preparation error is retained as that leaf's blocked diagnostic while independent leaves remain reachable. If an immediate child has descendant `spec.md` files, treat it as an explicitly unsupported nested-container blocker for this command; do not silently run its container implementation or skip its descendants.

`control` validates the integration checkout and returns its current `repo_root`, `integration_branch`, and `integration_head` without discovering tasks or preparing worktrees. It takes neither a target nor `--state`. Use it to refresh the head before each completed leaf's rebase/integration; never re-prepare a running or completed leaf merely to refresh the control plane.

## State file and dependency policy

Create one temporary state JSON outside the checkout and make the parent the only writer. Initialize it with `{}` (not an empty file), or use this structure:

```json
{
  "dependencies": {"child-name": ["canonical-sibling-name"]},
  "outcomes": {"child-name": {"status": "running|blocked|failed|ready", "reason": "nonempty explanation"}}
}
```

Never store orchestration state in `docs/changes` or a task worktree. Preserve the state for the whole invocation and remove it only as ordinary temporary-file cleanup after the final report; never alter user files to do so.

Dependency interpretation is conservative. The script accepts `None`, comma-separated names, backtick-quoted names, and explicit task paths, while preserving the raw field. External task paths are checked for completion but never added to the execution scope. Unsupported or ambiguous syntax fails closed until the parent interprets it: write an explicit `dependencies` entry with canonical sibling names, then rerun `queue`. Preserve every actual prerequisite; never replace an unresolved dependency with an empty list merely to unblock work. Missing dependencies remain waiting with the missing name. Only the dependency's canonical sibling `task.md` status `done` satisfies an edge; `ready`, a child claim, an existing commit, or an unintegrated branch does not.

`outcomes` prevents duplicate dispatch and automatic retry. After preparation and before dispatch, record every selected leaf as `running`; retain that status through any debugger rescue or owner repair. Record a nonempty, concrete reason for every transition. Replace `running` with `ready` only when that owner and its debugger have ended and the task is awaiting parent verification/integration; use `failed` or `blocked` for concrete failures. Both `running` and `ready` suppress redispatch without satisfying dependencies. An outcome suppresses reattempt in this invocation unless the canonical task annotation is now `done` (for example, after the parent completes recovery); never loop on a failed leaf. Dependency cycles and stalled prerequisites are diagnosed by `queue`; continue independent work and report the cycle/stall separately from nondependency blockers.

## Continuous dependency-aware orchestration

This section replaces spec-run-task's container-wave scheduling and global implementation barrier for this command only. Keep its assigned-worktree, single-task-commit, verification, rebase, and integration safety rules. Independent owners may keep implementing in their isolated worktrees while the parent verifies and integrates a completed leaf. Only the parent mutates the integration checkout, one leaf at a time.

1. Run `scan` once for target metadata, then initialize the outside-checkout state file and a parent-owned map of dispatched leaf names to agent/job identities and exact prepared records. Run `queue`. Interpret unresolved dependency fields into state where possible and rerun `queue`; never hand-sort or truncate `ready`. If `ready` is empty, finish only when `finished` is true; otherwise wait for active owners or process completed leaves before declaring a stall. Do not prepare an empty selection.
2. Before any child starts or any leaf is prepared, invoke `python3 .omp/skills/spec-run-all/scripts/check_omp_config.py --repo <repo> --pretty` exactly once and preserve its JSON result verbatim. The helper validates both project profiles, exact `@implementer`/`@slow` requests and spawn contracts, effective role-alias chains and model availability, overrides, `task.agentAdvisor`, `task.agentPrewalk`, `task.disabledAgents`, and recursion depth for root → implementer → debugger. If it exits `2` or returns `ok: false`, record the helper's concrete `errors` as one blocked outcome/final blocker and stop before `prepare`; never fall back, inspect config files or SQLite, or separately call `omp config`/`omp models`. Only a successful `ok: true` result permits `prepare`.
3. Run `prepare` with the current state. Record every preparation failure in `outcomes` as `blocked`, including failures without a worktree. Record all successfully prepared leaves as `running` before calling `task`, then submit one item per prepared leaf in alphabetical order, setting **`agent: "spec-run-all-implementer"`** on every item. This profile requests **`model: "@implementer"`**; do not substitute a generic agent or a prompt asking another model to act as an implementer. Dispatch all eligible leaves, including a single leaf, without a five-agent cap or fixed cohort. Respect actual harness admission limits without adding a skill-level limit: submit all eligible items, retain queued submissions as `running`, and never redispatch them. Disable automatic isolated-agent worktrees: the helper already prepared them.
4. Pass each child the exact `repo_root`, prepared paths and task identity, integration branch/head, exclusive task scope, known cross-task interface contracts, and the stuck-implementation rescue contract below. The child must use assigned-worktree mode below. It must not delegate except to one `spec-run-debug` rescue agent when stuck, implement nested descendants, integrate, or perform validation.
5. Drive a continuous event loop, consuming individual owner completions rather than waiting for all dispatched children. Keep each leaf's agent/job identity and prepared record until its outcome is settled. Record child runtime or dispatch failures as `failed`, external prerequisites as `blocked`, and provisional successes as `ready`. Wait only for that leaf's owner and any debugger it invokes to end before taking over its worktree. An owner about to report an implementation failure must first complete the required debugger rescue; crashes and dispatch failures remain concrete runtime failures. Retain reasons, debugger findings when used, and returned worktree/commit evidence. An empty `ready` list is not a stall while any owner is queued/running or any completed leaf awaits parent processing.
6. As soon as a leaf becomes `ready`, the parent runs `control` to refresh the validated integration branch/head, rebases that leaf's finalized task commit onto the current head with spec-run-task's helper, then performs focused and repository-required combined verification in that exact prepared worktree. Do not wait for unrelated owners. Serialize parent verification/integration and use isolated build/runtime resources so it cannot touch live owners' worktrees. Resolve conflicts only in the named worktree through `continue-rebase`, rerunning affected checks. Attribute failures to that leaf; retain failed or blocked work without integrating it. If returning a recoverable failure to the same owner, set `running` before the handoff and do not inspect, verify, or mutate its worktree until it returns. Never automatically redispatch failed work through `queue`.
7. After observed focused and combined success, the parent uses the helper to `annotate --status done` with grounded evidence, `commit --status done`, `check --status done`, and `integrate` that single leaf. This is a one-leaf integration unit, not a train that waits for unfinished siblings. If the integration head advanced, refresh it through `control`, rebase and rerun affected focused/combined verification, update evidence, and retry through the helper. Never use the head captured at dispatch as the current integration head. Do not use `spec-run-task inspect` for this orchestration target (it requires a root spec and can misresolve a selected leaf), re-prepare active leaves, or run direct Git mutation commands. No container rollup is created unless the actual specifications and existing helper ownership require one.
8. Immediately after each successful integration, rerun `queue` against the canonical tree and current state, then `prepare` and dispatch **all** newly eligible leaves before processing another completed leaf or waiting for another owner. For example, if A is integrated while unrelated B is still running, start both X and Y that depend on A immediately; B is neither a barrier nor eligible for redispatch. Also rescan eligibility after any other settled outcome. Continue consuming individual completions and refilling until `finished`, or until there are no eligible tasks, queued/running owners, or completed leaves left to process; only then report the remaining blockers.

The parent must observe canonical `task.md` status `done` after integration before scheduling dependents. A child claim, a `done` annotation in an unintegrated worktree, or an existing branch does not unlock an edge. If verification, rebase, integration, or dispatch failure cannot be recovered, retain it in state and continue independent work; descendants remain dependency-waiting. Never wait for unrelated owners merely to form a verification, integration, or dispatch batch.

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

Report only observed results. Do not call work complete when `finished` is false or any task remains blocked/waiting. Include the target, integration branch/head as returned by helpers, prepared worktrees/commits for attempted leaves, and residual risks. Validation/build/lint/test/formatter commands are intentionally skipped by child owners; the parent runs the focused and combined verification demanded by spec-run-task in each completed leaf's worktree before integrating it, without a global implementation barrier.

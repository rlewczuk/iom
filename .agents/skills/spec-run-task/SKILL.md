---
name: spec-run-task
description: Implement one change task or its unfinished subtasks from docs/changes in isolated, reusable Git worktrees on readable task-derived branches, then commit, verify, and merge the completed work. Use only through /spec-run-task <change-name>[/subpath] or when explicitly requested.
hide: true
---

# Spec Run Task

Implement the requested specification completely. Every executable task gets its own deterministic, task-named feature branch and registered worktree under `.work/`. The integration checkout is never an implementation or verification workspace: no project file may be changed, built, tested, formatted, or run there on the task's behalf. A target with descendant task specifications is an orchestration target: run its unfinished leaf subtasks instead of implementing the target's own `spec.md`.

The command supplies one path relative to `docs/changes/`:

```text
<change-name>[/subpath...]
```

For target `<target>`, the requested specification is exactly `docs/changes/<target>/spec.md` and its optional annotation is `docs/changes/<target>/task.md`.

## Mandatory worktree execution boundary

Treat the integration checkout as a read-only control plane, not as the task's working directory. Use it only for read-only discovery and orchestration, repository validation, status/ref/worktree inspection, worktree provisioning, and the final verified fast-forward merge.

Before any project-file edit, annotation edit, LSP mutation, build, test, formatter, or runtime command:

1. Provision or reuse the executable task's exact `.work/<task-path>` worktree.
2. Verify inside that worktree that its checked-out branch is the expected task-derived feature branch and that its common Git directory belongs to this repository.
3. Bind all subsequent task implementation and verification paths and command working directories to that worktree. Prefix every Read, Edit, Write, and LSP path used for task files with `.work/<task-path>/` (or its exact absolute path), and set every task-related Bash `cwd` to the worktree.

Never edit in the integration checkout with the intent to copy or commit the change later. Never rely on the shell's current directory or on an unprefixed relative path for task work after provisioning. If a task tool path or command `cwd` would resolve to the integration checkout, stop before the action and retarget it to the assigned worktree.

## Non-negotiable invariants

- Capture the project repository's current branch as the **integration branch** before provisioning any worktree. New feature branches start at that branch's then-current tip.
- Require a named integration branch and a clean integration checkout. Do not stash, reset, clean, switch, commit, or otherwise disturb pre-existing user changes to make it clean.
- Do not edit, build, test, format, or run task code in the integration checkout. All implementation, annotation, code-intelligence, build, test, and runtime actions happen in the executable task's assigned worktree.
- Use one feature branch and one registered Git worktree per executable task. Reuse the deterministic branch and exact worktree when they already exist; never discard unfinished contents.
- The exact worktree for task path `<task-path>` is `<repo-root>/.work/<task-path>`. For example, `0001-foo/02-bar` uses `.work/0001-foo/02-bar`.
- Keep `.work/` ignored. Never stage files through the integration checkout merely because a worktree lives below it.
- A completed task's implementation, tests, and `task.md` must be committed on its feature branch before integration.
- Do not expose merge conflicts in the integration checkout. The main agent resolves conflicts for a single-task run. For an orchestrated run, the subagent that owns the feature being added resolves the conflict in that feature worktree. The integration checkout accepts only a verified fast-forward.
- Do not delete feature branches or worktrees after completion. They are the resumable task state.
- Never mark a task `done` before its required verification succeeds. Never merge a failed, blocked, partial, or unverified task branch.
- Treat the specification and repository guidance as acceptance criteria. Read any applicable skills before implementation or verification, including `remote-development` for accelerator work.

## Target validation

Before reading or changing task contents:

1. Require exactly one non-empty target argument after trimming surrounding whitespace.
2. Reject an absolute path, a path ending in `spec.md` or `task.md`, a backslash, an empty component, and any `.` or `..` component. The argument names a directory, not a file.
3. Join the target only beneath `docs/changes/` and verify that lexical normalization cannot escape that directory.
4. Require `docs/changes/<target>/spec.md` to be a regular file. Report the exact expected path and stop if it is absent.
5. Locate the repository root with Git. Require that the command was invoked from this repository, the current checkout has a named local branch, and `git status --porcelain` is empty. Record the integration branch and its starting commit.
6. Require `.work` to be inside the repository root and ignored by Git. If it is not ignored, stop before creating a worktree rather than risk staging nested worktree contents.

Quote every path and ref passed to Git or a shell. Do not evaluate the target as shell text.

## Task annotations

`task.md` is persistent execution state. A missing file means a fresh, unfinished task. The first case-insensitive `**Status:** <value>` field is authoritative. Multiple status fields are malformed and must be repaired by the task owner before completion.

Use these lowercase values:

- `ready` — implementation is committed on the feature branch, but required verification or integration is still pending;
- `done` — implementation and required verification succeeded and the feature commit is eligible for integration;
- `failed` — the attempted implementation or verification failed and the feature branch must not be merged;
- `blocked` — an external prerequisite prevents execution;
- `in progress` — optional transient state while an agent is working.

Only `done` is finished. A missing field or any other value is unfinished.

On success, create or update the annotation in this form while preserving unrelated user-authored sections:

```markdown
**Status:** done

## Summary

<Concise description of the delivered behavior and important files or interfaces changed.>

## Verification

- `<exact command or scenario>` — <observed successful result>
```

Replace stale generated Summary and Verification contents rather than appending duplicate sections. Remove a stale generated Errors section after a successful rerun. Do not claim a command passed unless its output was observed.

On failure or a blocker, set `failed` or `blocked`, summarize the reachable work, and add or replace:

```markdown
## Errors

- `<command, operation, or prerequisite>` — <concise concrete failure and retained worktree state>
```

Keep useful partial work committed on the feature branch for a later invocation, but do not merge it. The error annotation may therefore exist only on the retained feature branch until that task succeeds; report this explicitly.

## Discover executable tasks

Use the Glob and Read tools, not shell `find`, to enumerate `spec.md` files recursively below the requested directory.

A specification directory is an **executable leaf** only when it has no descendant directory containing another `spec.md`:

- If the requested directory has no descendant `spec.md`, the requested target itself is the sole executable task. Run it with the main agent.
- If descendant `spec.md` files exist, the requested directory is a container. Do not implement its own `spec.md`. Recursively select leaf specification directories beneath it and orchestrate them through subagents.
- An intermediate directory with both its own `spec.md` and deeper task specifications is also a container; run its leaf descendants, not its own spec.
- Read each leaf's `task.md` when present. Skip leaves whose status is `done`; run every other leaf whose dependencies can be satisfied.
- Child status is authoritative over stale container status. If a container says `done` but any leaf is unfinished, run the leaf and later correct the container roll-up.

Do not infer that a task is complete merely from existing implementation or an old feature branch. The annotation contract controls scheduling; the task's verification controls the next transition to `done`.

## Build the dependency graph

Before provisioning or launching any subagent, read the complete executable specs and build one assignment table containing:

- task path and spec path;
- current status;
- explicit blockers;
- additional genuine semantic blockers found by comparing the specs' produced and consumed contracts;
- priority and order metadata, when present;
- feature branch and exact worktree path;
- owned files or interfaces and cross-task contracts needed by parallel siblings;
- required focused verification and applicable repository-wide verification.

Interpret a `**Blocked by:**` field when present. Resolve a blocker by exact relative task path first, then by an unambiguous task-directory basename within the requested tree. A blocker outside the selected leaf set is satisfied only when its `task.md` says `done` or repository evidence proves it is an external completed prerequisite. Do not use numeric prefixes, `Order`, or priority as fake dependencies; they are stable scheduling tie-breakers only.

Add a semantic edge only when one task needs an interface, artifact, migration, or repository state produced by another. Reject unknown or ambiguous blocker names and dependency cycles before editing project files. Report the exact specs and fields that need correction.

Topologically schedule unfinished leaves in waves. Tasks in one wave have all blockers satisfied by the integration branch and no dependency edge between them, so they may run in parallel. After a successful wave is integrated, recompute readiness against the updated integration branch. Continue independent work when another branch fails; do not run descendants of a failed or blocked task.

## Provision a task worktree

Provision every task before its agent changes any project file or runs any task command. Derive a deterministic, readable, Git-safe feature branch from the exact task path:

```text
components = the exact slash-separated task-path components
require every component to match [A-Za-z0-9]+(?:-[A-Za-z0-9]+)*
branch suffix = components joined with --
feature branch = run-task/<branch-suffix>
worktree = <repo-root>/.work/<task-path>
```

For example, task path `task-name/subtask-name` uses `run-task/task-name--subtask-name`; `0001-tensor-view/02-core-metadata-layout` uses `run-task/0001-tensor-view--02-core-metadata-layout`.

Joining components with `--` is collision-free because a valid component contains only single hyphen separators. Preserve each component exactly, including case and numeric prefixes. Do not silently slugify or hash an invalid name: report the exact invalid component and stop before creating a ref or worktree. Validate the final ref with `git check-ref-format --branch`.

Provision or reuse conservatively:

1. Inspect `git worktree list --porcelain` and local refs.
2. If the exact worktree is already registered on the deterministic feature branch, reuse it, including its committed or uncommitted unfinished state. Do not reset, clean, or recreate it.
3. If that path is registered on another branch, or the deterministic branch is checked out at another path, stop and report both paths and refs. Do not move or remove either worktree automatically.
4. If the path exists but is not a registered worktree, require it to be empty. Never delete or overwrite an unregistered non-empty path.
5. If the deterministic branch exists but has no worktree, add the exact worktree for that branch.
6. Otherwise create the feature branch and worktree atomically from the integration branch's current tip, equivalent to:

   ```text
   git worktree add -b <feature-branch> <repo-root>/.work/<task-path> <integration-branch>
   ```

7. Verify from the new worktree that its current branch is the deterministic feature branch and that the worktree's common Git directory belongs to this repository.

For a reused clean feature branch, have its owning agent merge the latest integration branch into the feature branch before editing. For a reused dirty worktree, the owner must inspect and preserve the existing task work, commit a coherent checkpoint when necessary, and then merge the integration branch. Never auto-stash user or prior-agent work. Any conflict is resolved and verified by that task's owner in this worktree.

## Execute one leaf task

When the requested target has no descendant task specs, the main agent is the task owner:

1. Before any implementation action, provision or reuse the target worktree and feature branch, then verify both from inside that worktree.
2. Treat the assigned worktree as the sole project root for the task. Perform every Read, Edit, Write, LSP, build, test, and runtime action there. File tools must use paths rooted at `.work/<target>/`; Bash commands must set that worktree as `cwd`. An unprefixed project path or integration-checkout `cwd` is an error: stop and retarget rather than modifying the integration checkout.
3. Read the complete `spec.md`, repository guidance, referenced implementation and tests, and applicable skills. Resume coherent prior work when reusing a worktree.
4. Reproduce a bug before editing when the spec requires a reproduction and it remains reachable.
5. Implement the complete specified behavior and acceptance criteria. Run the focused verification in the spec and the repository-required conformance coverage.
6. Update `task.md` to `done` with grounded Summary and Verification sections. Stage only task-owned changes, review the staged set, and commit them on the feature branch. If nothing changed but verification proves the task already satisfied, still record the verified completion in `task.md` and commit that annotation.
7. Merge the latest integration branch into the feature branch. Resolve conflicts in this task worktree, rerun affected verification, and commit the resolution and any refreshed annotation.
8. From the clean integration checkout, integrate only with `git merge --ff-only <feature-branch>`. If the fast-forward fails because the integration branch advanced, return to step 7; never resolve or create a non-fast-forward merge in the integration checkout.

On failure, update and commit the task's `failed` or `blocked` annotation with concrete errors and retained state, leave the feature branch/worktree intact, and do not merge it.

If the leaf annotation already says `done`, make no project change and report it as already complete.

## Execute a container through subagents

Subtasks must be implemented by general-purpose subagents, not by the orchestrating main agent. Do not use automatic isolated-agent worktrees: the orchestrator must provision and verify each required `.work/<task-path>` worktree explicitly before spawning, then assign each child its exact existing worktree and task-derived feature branch. The child works only in that worktree; the integration checkout remains a read-only control plane.

### Prepare a wave

Provision every unfinished task in the ready wave. Build the complete assignment batch before spawning. Give each child exclusive ownership of one task's implementation and annotation, plus explicit shared contracts decided by the orchestrator. No two agents own the same task or feature branch.

Launch one child per ready task in one parallel Task batch, up to the harness concurrency cap. Store each returned agent ID; all revisions, completion annotation edits, and merge-conflict resolution for that feature must go back to the same owner.

Use this assignment shape:

```text
# Target
- Task: <task-path>
- Spec: <worktree>/docs/changes/<task-path>/spec.md
- Annotation: <worktree>/docs/changes/<task-path>/task.md
- Worktree: <exact absolute or repository-relative worktree path>
- Feature branch: <deterministic branch>
- Integration branch and base: <branch> at <commit>
- Exclusive ownership: this task's implementation, tests, and annotation only

# Change
- **WORKTREE ONLY:** use the assigned worktree as the sole project root. Prefix every Read, Edit, Write, and LSP path with the exact worktree path and set every Bash `cwd` to it. Before the first implementation action, verify the current branch from inside that worktree. Never edit, build, test, format, or run task code in the integration checkout, even temporarily.
- Read the complete spec, repository guidance, relevant source/tests, and applicable skills.
- Preserve and resume coherent existing work when the worktree was reused.
- If needed, merge the integration branch into the feature branch before implementation and resolve any conflict here.
- Implement every requirement and acceptance criterion without unrelated changes.
- Do not run formatters, builds, linters, tests, or project-wide validation in this parallel implementation pass; the orchestrator runs verification once after the wave barrier.
- Update task.md to ready with a factual provisional summary, or to failed/blocked with an Errors section.
- Commit all coherent task-owned changes on the feature branch. Do not merge into the integration checkout.

# Acceptance
- The assigned feature branch contains a complete implementation commit and task.md is ready, or it contains a precise failed/blocked annotation and resumable state.
- No integration-checkout or sibling-worktree files changed.
- The response names commits, changed paths, unresolved risks, and verification still required; it does not claim unrun checks passed.
```

### Verify each task

After the implementation children return, the orchestrator performs focused validation from the assigned worktrees; do not duplicate project-wide commands in every child:

1. Run each spec's focused behavioral verification against its feature worktree. Use the actual runtime surface, hardware, sanitizer, or remote procedure required by the spec. A compile alone is not proof when the spec requires runtime behavior.
2. If a check fails, send the exact failure to the same owning subagent. The owner fixes and commits in its task worktree without running project-wide validation; rerun the focused check from the orchestrator. Repeat until it passes or a concrete blocker remains.
3. After observed focused success, retain `Status: ready` and record the evidence with the assignment. Repository-required combined verification has not passed yet, so the task is not `done` and its branch is not independently eligible for integration.
4. If implementation or focused verification cannot finish, have the owner commit a `failed` or `blocked` annotation on that feature branch. Retain but do not integrate it. Continue with independent ready tasks and mark its dependency descendants unschedulable for this invocation.

### Build and integrate a wave train

Never merge divergent feature branches directly in the integration checkout. Combine the ready branches into a verified **wave train**, making the incoming feature's owner responsible for conflicts:

1. Choose the first ready feature branch in stable dependency/priority/path order as the initial train.
2. For each remaining ready feature branch in that order, send its owner the current train branch and commit. In the owner's own feature worktree, merge the train branch into that feature branch, resolve all conflicts, and commit. The owner's branch becomes the new train. If resolution changes behavior, the orchestrator reruns the affected focused checks.
3. Run the repository-required combined verification, including applicable backend conformance suites, once in the final train worktree. On failure, route the concrete failure to the subagent that owns the responsible feature. That owner merges the current train into its own feature branch, resolves there, commits the fix, and becomes the new train owner. Re-run focused and combined checks until the train passes. Do not advance the integration branch on failure.
4. Only after the combined checks pass, finalize every ready leaf annotation through its same owner. In stable order, send the grounded focused and combined evidence to the owner; the owner merges the current train into its feature branch, changes only its own `task.md` from `ready` to `done`, writes the final Summary and Verification sections, removes stale generated errors, and commits. That branch becomes the new train. Re-read each resulting annotation. If this fold changes implementation rather than annotations or conflict metadata, rerun affected focused and combined verification.
5. If this wave completes every leaf beneath the requested container, perform the container roll-up below on the final train before integration.
6. Confirm the integration checkout is still on the recorded integration branch and clean. Integrate only with `git merge --ff-only <final-train-branch>`.
7. If fast-forward fails because the integration branch advanced, send the latest integration branch to the final train owner. That owner merges it in the train worktree and resolves conflicts. The orchestrator reruns affected focused and combined verification, refreshes completion evidence when necessary, and retries the fast-forward.

A fast-forward at step 6 makes every feature commit in the train reachable from the current integration branch while ensuring all conflict resolution happened in an owning task worktree.

After integrating the wave, recompute the dependency graph and provision the next ready wave from the integration branch's new tip. New downstream feature branches must therefore include their completed blockers at creation time.

## Roll up container annotations

When all executable leaves beneath the requested container are `done` in the final train, the final train owner updates container annotations before the last fast-forward:

- Work bottom-up through every specification directory at or below the requested target that has descendant executable leaves.
- Set a container to `done` only when every executable leaf below it is `done` in that train.
- Create `task.md` when missing; otherwise preserve unrelated annotations. Replace generated Summary/Verification/Errors sections consistently.
- The Summary states that all leaf subtasks completed and gives a concise count or list. The Verification section records the combined checks that covered the integrated train. Remove stale generated errors.
- Commit the roll-up on the final train branch, rerun any metadata validation, then fast-forward the integration branch.

If every leaf was already `done` when the command began but a container annotation is missing or stale, choose a deterministic completed leaf as the annotation owner, provision or reuse its normal task worktree and feature branch, merge the current integration branch into it, and launch a subagent to make only the roll-up annotation change. Verify and fast-forward that branch by the same rules. This preserves the no-direct-edits invariant without creating a parent worktree that would collide with nested task worktrees.

If any leaf fails or remains blocked, do not mark its containers `done`. Report the incomplete leaf paths and the feature branches/worktrees retaining their errors.

## Final checks

Before completing the command, establish all of the following from observed state:

- every non-finished leaf whose blockers became satisfied was attempted;
- no container `spec.md` was implemented while it had descendant specs;
- each attempted task used its exact `.work/<task-path>` worktree and deterministic `run-task/<task-name>[--<subtask-name>...]` feature branch;
- each integrated task has `Status: done`, grounded verification evidence, and committed implementation and annotation changes;
- every integrated feature commit is reachable from the recorded integration branch;
- no failed, blocked, partial, dirty, or unverified feature branch was merged;
- all conflicts were resolved and checked in the responsible main-agent or subagent worktree, never in the integration checkout;
- all eligible container annotations were rolled up only after their leaves completed;
- the integration checkout ends clean on the same named branch from which the run began.

## Completion response

Report concisely:

- requested target and integration branch;
- each leaf as `already done`, `done and merged`, `failed`, `blocked`, or `not run because <blocker>`;
- for attempted leaves, exact worktree, feature branch, final commit, and verification evidence;
- container annotation roll-up and final integration commit;
- retained error state or residual blockers.

Do not call partial work complete and do not claim unobserved verification.
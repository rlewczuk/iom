---
name: spec-run-task
description: Implement one change task or its unfinished subtasks from docs/changes in isolated, reusable Git worktrees on readable task-derived branches, then produce one named task commit, verify it, rebase it onto the integration branch, and fast-forward the completed work. Use only through /spec-run-task <change-name>[/subpath] or when explicitly requested.
hide: true
---

# Spec Run Task

Implement the requested specification completely. Every executable task gets its own deterministic, task-named feature branch and registered worktree under `.work/`, and every successful executable task contributes exactly one clearly named task-local commit containing its code, tests, and final `task.md` update. The integration checkout is never an implementation or verification workspace: no project file may be changed, built, tested, formatted, or run there on the task's behalf. A target with descendant task specifications is an orchestration target: run its unfinished leaf subtasks instead of implementing the target's own `spec.md`.

The command supplies one path relative to `docs/changes/`:

```text
<change-name>[/subpath...]
```

For target `<target>`, the requested specification is exactly `docs/changes/<target>/spec.md` and its optional annotation is exactly the sibling file `docs/changes/<target>/task.md`.

**Annotation location is an invariant, not a convention.** For every leaf or container specification, derive its annotation path from that specification path as `<directory-containing-spec.md>/task.md`. Preserve the complete nested task path. Never create or update a bare `task.md` relative to the current working directory, `<repo-root>/task.md`, `<worktree>/task.md`, `docs/changes/task.md`, or any other non-sibling location. The corresponding `spec.md` and `task.md` must always have the same parent directory.

## Mandatory worktree execution boundary

Treat the integration checkout as a read-only control plane, not as the task's working directory. Use it only for read-only discovery and orchestration, repository validation, status/ref/worktree inspection, worktree provisioning, and the final verified fast-forward update.

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
- For any specification at `<repo-root>/.work/<owner-task-path>/docs/changes/<annotated-task-path>/spec.md`, its only valid annotation path is `<repo-root>/.work/<owner-task-path>/docs/changes/<annotated-task-path>/task.md`. Always pass this full worktree-prefixed path to Read, Edit, Write, and staging commands; a bare or repository-root-relative `task.md` path is prohibited.
- A task's implementation, tests, and `task.md` update must be grouped into exactly one final task-local commit on its feature branch. Its subject must be `spec-run-task(<task-path>): <concise outcome>`, using the exact task or subtask path so the commit is immediately identifiable.
- Rebase task-local commits onto the latest integration branch or current wave train; never merge the integration branch or another feature branch into a task branch. Resolve rebase conflicts in the owning task worktree. The integration checkout accepts only a verified fast-forward.
- Temporary checkpoints needed to preserve reused work must be squashed into that one task commit before the task is retained as `ready`, `done`, `failed`, or `blocked`. After consolidation, fold every fix, conflict resolution, and annotation refresh into the same commit with `git commit --amend`; never stack follow-up commits.
- Do not delete feature branches or worktrees after completion. They are the resumable task state.
- Never mark a task `done` before its required verification succeeds. Never integrate a failed, blocked, partial, or unverified task branch.
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

Before every annotation creation or update, apply this mandatory location gate:

1. Start from the exact full path of the associated `spec.md` inside the assigned worktree.
2. Compute the annotation path by replacing only the final filename `spec.md` with `task.md`; do not reconstruct it from the shell working directory or shorten it to a basename.
3. Verify that the computed `task.md` parent is identical to the `spec.md` parent and that the path remains under that worktree's `docs/changes/`.
4. Use the computed full worktree-prefixed path for the file operation. Never use bare `task.md`, even when a command's `cwd` appears correct.
5. After writing, re-read that exact path and inspect the task commit diff. The intended sibling annotation must be present, and the invocation must not add or modify any `task.md` outside the explicitly owned leaf or container specification directories. If this invocation created a misplaced annotation, remove it before committing and create the sibling annotation at the computed path.

Use these lowercase values:

- `ready` — the single task commit exists on the feature branch, but required verification or integration is still pending;
- `done` — implementation and required verification succeeded and the single task commit is eligible for integration;
- `failed` — the attempted implementation or verification failed and the feature branch must not be integrated;
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

Keep useful partial work and the error annotation grouped in one amendable task commit on the feature branch for a later invocation, but do not integrate it. The error annotation may therefore exist only on the retained feature branch until that task succeeds; report this explicitly.

## Single task commit

Each attempted executable leaf has one task-local commit, including retained failed or blocked work. A single-leaf invocation therefore lands exactly one commit; a container invocation lands one independently reviewable commit per executed leaf. That commit contains all task-owned implementation, tests, generated artifacts that belong in the repository, and the leaf's `task.md` update. Do not split annotation, conflict resolution, formatting, or verification-driven corrections into separate commits. Do not include unrelated cleanup or another task's changes.

Use this exact subject shape:

```text
spec-run-task(<task-path>): <concise outcome>
```

Use the full path relative to `docs/changes/`, including every task and subtask component. Keep the outcome short and behavioral. During implementation, an owner may create a temporary checkpoint only to make a dirty reused worktree safe for rebase. Before reporting any terminal task status or offering a branch for integration, use rebase squash/fixup to consolidate all commits attributable to that task, then use `git commit --amend` for every later change.

Before integration, verify from history and the final diff that the task contributes exactly one non-merge commit, its subject identifies the exact task path, and that commit contains both the implementation changes and the final leaf annotation. A task that only needs an annotation change still contributes exactly one annotation-only commit. If a reused feature branch contains commits or changes that cannot be proven to belong to the assigned task, stop rather than squash or rewrite them.

## Discover executable tasks

Use the Glob and Read tools, not shell `find`, to enumerate `spec.md` files recursively below the requested directory.

A specification directory is an **executable leaf** only when it has no descendant directory containing another `spec.md`:

- If the requested directory has no descendant `spec.md`, the requested target itself is the sole executable task. Run it with the main agent.
- If descendant `spec.md` files exist, the requested directory is a container. Do not implement its own `spec.md`. Recursively select leaf specification directories beneath it and orchestrate them through subagents.
- An intermediate directory with both its own `spec.md` and deeper task specifications is also a container; run its leaf descendants, not its own spec.
- For each leaf spec at `docs/changes/<leaf-task-path>/spec.md`, read only its sibling annotation at `docs/changes/<leaf-task-path>/task.md` when present. Do not treat a repository-root, `docs/changes/`-root, ancestor, or other misplaced `task.md` as that leaf's state. Skip leaves whose sibling annotation status is `done`; run every other leaf whose dependencies can be satisfied.
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

For a reused feature branch, the owner must inspect its task-local commits and dirty state before changing history. If the worktree is dirty, preserve all coherent task-owned state in a temporary checkpoint commit; never auto-stash. Rebase the task-local work onto the latest integration branch, resolve conflicts in this worktree, and then squash every task-local checkpoint into the single required task commit. Never merge the integration branch into the feature branch, discard unfinished contents, or rewrite changes whose ownership is uncertain.

## Execute one leaf task

When the requested target has no descendant task specs, the main agent is the task owner:

1. Before any implementation action, provision or reuse the target worktree and feature branch, then verify both from inside that worktree.
2. Treat the assigned worktree as the sole project root for the task. Perform every Read, Edit, Write, LSP, build, test, and runtime action there. File tools must use paths rooted at `.work/<target>/`; Bash commands must set that worktree as `cwd`. An unprefixed project path or integration-checkout `cwd` is an error: stop and retarget rather than modifying the integration checkout.
3. Read the complete `spec.md`, repository guidance, referenced implementation and tests, and applicable skills. Resume coherent prior work when reusing a worktree.
4. Reproduce a bug before editing when the spec requires a reproduction and it remains reachable.
5. Implement the complete specified behavior and acceptance criteria. Run the focused verification in the spec and the repository-required conformance coverage.
6. Update the exact sibling annotation `.work/<target>/docs/changes/<target>/task.md` to `done` with grounded Summary and Verification sections. Derive this path from `.work/<target>/docs/changes/<target>/spec.md`; do not write a bare `task.md` or any repository-root annotation. Re-read the exact sibling path after writing. Stage only task-owned implementation, tests, and annotation changes. Create the single task commit with subject `spec-run-task(<target>): <concise outcome>`, or amend and squash existing task-local commits into it. If only verified completion state changed, the one commit may be annotation-only.
7. Rebase the single task commit onto the latest integration branch. Resolve conflicts in this task worktree, keep the resolution in that same rebased commit, rerun affected verification, and amend the commit and annotation if the observed evidence changed. Never merge the integration branch into the feature branch.
8. Confirm that the feature branch is exactly one non-merge task commit ahead of the current integration tip, that its subject contains the exact target path, and that its diff contains the final annotation at `docs/changes/<target>/task.md` plus every task-owned change, with no annotation added or modified outside the associated specification directory. From the clean integration checkout, advance only with `git merge --ff-only <feature-branch>`. If the fast-forward fails because the integration branch advanced, return to step 7.

On failure, update the task's `failed` or `blocked` annotation with concrete errors and retained state, consolidate all task-owned partial work and that annotation into the one named task commit, leave the feature branch/worktree intact, and do not integrate it.

If the leaf annotation already says `done`, make no project change and report it as already complete.

## Execute a container through subagents

Subtasks must be implemented by general-purpose subagents, not by the orchestrating main agent. Do not use automatic isolated-agent worktrees: the orchestrator must provision and verify each required `.work/<task-path>` worktree explicitly before spawning, then assign each child its exact existing worktree and task-derived feature branch. The child works only in that worktree; the integration checkout remains a read-only control plane.

### Prepare a wave

Provision every unfinished task in the ready wave. Build the complete assignment batch before spawning. Give each child exclusive ownership of one task's implementation and annotation, plus explicit shared contracts decided by the orchestrator. No two agents own the same task or feature branch.

Launch one child per ready task in one parallel Task batch, up to the harness concurrency cap. Store each returned agent ID; all revisions, completion annotation edits, rebase-conflict resolution, and task-commit amendments for that feature must go back to the same owner.

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
- If needed, preserve dirty task-owned state in a temporary checkpoint, rebase the task-local work onto the integration branch, resolve conflicts here, and squash all task-local commits into one. Never merge.
- Implement every requirement and acceptance criterion without unrelated changes.
- Do not run formatters, builds, linters, tests, or project-wide validation in this parallel implementation pass; the orchestrator runs verification once after the wave barrier.
- **ANNOTATION PATH:** update exactly `<worktree>/docs/changes/<task-path>/task.md`, the sibling of the assigned `spec.md`. Derive it by replacing only `spec.md` with `task.md`, use that full worktree-prefixed path for every file operation, and re-read it after writing. Never create or update bare `task.md`, `<worktree>/task.md`, repository-root `task.md`, `docs/changes/task.md`, an ancestor task's annotation, or any other path. Set this exact file to `ready` with a factual provisional summary, or to `failed`/`blocked` with an Errors section.
- Group every task-owned code, test, and annotation change into exactly one commit with subject `spec-run-task(<task-path>): <concise outcome>`. Amend that commit for later fixes; do not add follow-up commits or merge into the integration checkout.

# Acceptance
- The assigned feature branch contributes exactly one non-merge task commit whose diff includes the complete implementation and exactly its sibling annotation at `docs/changes/<task-path>/task.md`; it adds or modifies no `task.md` at the repository/worktree root or outside the explicitly owned specification directory. The annotation is `ready`, or precisely `failed`/`blocked`.
- The commit subject contains the exact full task/subtask path.
- No integration-checkout or sibling-worktree files changed.
- The response names the single task commit, changed paths, unresolved risks, and verification still required; it does not claim unrun checks passed.
```

### Verify each task

After the implementation children return, the orchestrator performs focused validation from the assigned worktrees; do not duplicate project-wide commands in every child:

1. Run each spec's focused behavioral verification against its feature worktree. Use the actual runtime surface, hardware, sanitizer, or remote procedure required by the spec. A compile alone is not proof when the spec requires runtime behavior.
2. If a check fails, send the exact failure to the same owning subagent. The owner fixes the task in its worktree and amends the existing task commit without running project-wide validation; rerun the focused check from the orchestrator. Repeat until it passes or a concrete blocker remains.
3. After observed focused success, retain `Status: ready` in that same task commit and record the evidence with the assignment. Repository-required combined verification has not passed yet, so the task is not `done` and its branch is not independently eligible for integration.
4. If implementation or focused verification cannot finish, have the owner amend the same commit with a `failed` or `blocked` annotation. Retain but do not integrate it. Continue with independent ready tasks and mark its dependency descendants unschedulable for this invocation.

### Rebase and integrate a wave train

Never merge divergent feature branches. Combine ready branches into a verified **wave train** by rebasing one task-local commit at a time. Before assembly, each ready branch must contain exactly one task commit on the recorded wave integration base; record that commit and its parent so only the owning task's commit is ever replayed.

1. Choose the first ready feature branch in stable dependency/priority/path order. Rebase its single task commit onto the current integration tip if necessary; this branch is the initial train.
2. For each remaining ready feature branch in that order, send its owner the current train commit. In the owner's feature worktree, rebase only its recorded task commit onto the train, equivalent to `git rebase --onto <train-commit> <recorded-task-parent> <feature-branch>`. The owner resolves conflicts during that rebase and keeps every resolution in the rebased task commit. The owner's branch becomes the new train. Never replay another task's commits from the old branch prefix and never create a merge or conflict-resolution commit.
3. Run the repository-required combined verification, including applicable backend conformance suites, once in the final train worktree. On failure, route the concrete failure to the owner of the responsible task. That owner amends its single task commit; then rebuild the train from that task onward by rebasing each later task's single recorded commit in stable order. Re-run affected focused checks and combined checks until the rebuilt train passes. Do not add fix commits or advance the integration branch on failure.
4. Only after combined checks pass, rebuild the final train from the integration tip in stable order. For each ready leaf, send its owner the grounded focused and combined evidence. The owner rebases only that leaf's single task commit onto the current finalized train, derives the exact sibling annotation path from `<worktree>/docs/changes/<task-path>/spec.md`, changes that exact `<worktree>/docs/changes/<task-path>/task.md` from `ready` to `done`, writes the final Summary and Verification sections, removes stale generated errors, and amends the task commit. A bare `task.md` path is prohibited. Re-read the exact sibling annotation and verify the commit subject and diff before using that branch as the next finalized train. If this fold changes implementation rather than annotations or rebase metadata, rerun affected focused and combined verification.
5. If this wave completes every leaf beneath the requested container, perform the container roll-up below in the final train owner's worktree and amend it into that owner's still-HEAD task commit; do not create a separate roll-up commit.
6. Confirm the integration checkout is still on the recorded integration branch and clean. Advance it only with `git merge --ff-only <final-train-branch>`.
7. If the fast-forward fails because the integration branch advanced, rebuild the finalized train onto the latest integration tip by rebasing each task's one commit in stable order through its owner. Resolve every conflict in the owning task worktree, amend the same task commit when resolution or evidence changes, rerun affected focused and combined verification, and retry the fast-forward. Never merge the new integration tip into the train.

The fast-forward at step 6 makes every named task commit reachable from the integration branch without introducing merge, fixup, annotation-only follow-up, or conflict-resolution commits.

After integrating the wave, recompute the dependency graph and provision the next ready wave from the integration branch's new tip. New downstream feature branches must therefore include their completed blockers at creation time.

## Roll up container annotations

When all executable leaves beneath the requested container are `done` in the final train, the final train owner updates container annotations before the last fast-forward:

- Work bottom-up through every specification directory at or below the requested target that has descendant executable leaves.
- Set a container to `done` only when every executable leaf below it is `done` in that train.
- For each container specification, derive its annotation by replacing that container's final `spec.md` filename with `task.md`; create or update exactly that sibling path inside the final train owner's worktree. Never use a bare filename or place the roll-up at the repository root, worktree root, `docs/changes/` root, leaf directory, or a different ancestor. Preserve unrelated annotations and replace generated Summary/Verification/Errors sections consistently.
- The Summary states that all leaf subtasks completed and gives a concise count or list. The Verification section records the combined checks that covered the integrated train. Remove stale generated errors.
- Stage the roll-up with the final leaf's own final annotation and amend both into that leaf's single task commit, then rerun metadata validation and fast-forward the integration branch. Do not create a separate roll-up commit.

If every leaf was already `done` when the command began but a container annotation is missing or stale, choose a deterministic completed leaf as the annotation owner, provision or reuse its normal task worktree and feature branch, rebase it onto the current integration branch, and launch a subagent to make only the roll-up annotation change. Create exactly one annotation-only commit named `spec-run-task(<task-path>): roll up <target>`, verify it, and fast-forward the integration branch by the same rules. This preserves the no-direct-edits invariant without creating a parent worktree that would collide with nested task worktrees.

If any leaf fails or remains blocked, do not mark its containers `done`. Report the incomplete leaf paths and the feature branches/worktrees retaining their errors.

## Final checks

Before completing the command, establish all of the following from observed state:

- every non-finished leaf whose blockers became satisfied was attempted;
- no container `spec.md` was implemented while it had descendant specs;
- each attempted task used its exact `.work/<task-path>` worktree and deterministic `run-task/<task-name>[--<subtask-name>...]` feature branch;
- each integrated task has `Status: done`, grounded verification evidence, and exactly one non-merge task commit containing its implementation and annotation changes;
- every leaf annotation exists exactly beside its associated `spec.md` at `docs/changes/<task-path>/task.md`, every container roll-up exists exactly beside that container's `spec.md`, and no attempted commit adds or modifies a misplaced `task.md` elsewhere;
- every task commit subject contains the exact task/subtask path in the required `spec-run-task(<task-path>): ...` form;
- every integrated task-local commit was rebased onto the applicable integration tip or preceding train commit and is reachable from the recorded integration branch;
- no separate fix, conflict-resolution, completion-annotation, or roll-up commit was left in the integrated history;
- no failed, blocked, partial, dirty, or unverified feature branch was integrated;
- all conflicts were resolved and checked in the responsible main-agent or subagent worktree, never in the integration checkout;
- all eligible container annotations were rolled up only after their leaves completed;
- the integration checkout ends clean on the same named branch from which the run began.

## Completion response

Report concisely:

- requested target and integration branch;
- each leaf as `already done`, `done and integrated`, `failed`, `blocked`, or `not run because <blocker>`;
- for attempted leaves, exact worktree, feature branch, single final task commit, and verification evidence;
- container annotation roll-up and final integration commit;
- retained error state or residual blockers.

Do not call partial work complete and do not claim unobserved verification.
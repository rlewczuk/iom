---
name: csw-run
description: Execute all eligible direct implementation children under .cswd/tasks continuously in dependency-aware parallel worktrees, reviewing each candidate commit before integration.
hide: true
---

# CSW Run

Run all unfinished **direct child** tasks below `.cswd/tasks/<task-name>`. Read `skill://csw-run-worker` first. This is an orchestration wrapper around its worktree and Git control plane, not a separate implementation workflow. The target directory need not have its own control or spec. Nested task containers are reported as unsupported direct children and are not expanded.

## Mechanical control plane

Use only the shared helpers and preflight:

```text
.omp/csw/bin/csw_run --repo <repo> --pretty scan <task-name>
.omp/csw/bin/csw_run --repo <repo> --pretty queue <task-name> [--state <temporary-json-path>]
.omp/csw/bin/csw_run --repo <repo> --pretty prepare <task-name> [--state <temporary-json-path>]
.omp/csw/bin/csw_run --repo <repo> --pretty control
.omp/csw/bin/csw_preflight --repo <repo> --workflow csw-run --pretty
```

The script loads `task_ctl` through csw-run-worker's public API. `task_ctl` alone discovers, validates, and orders direct canonical `task.yml` controls. The script does not parse task metadata from `spec.md` or `task.md`. It preserves task_ctl's numeric-order/canonical-ID order and never guesses basenames. PyYAML 6.0.3 is the supported runtime dependency used by `task_ctl`.

Missing or malformed direct controls are retained as per-task blockers through `task_ctl.scan_tasks`; independent valid tasks continue. `task.md` contains only execution Outcome, Summary, Verification, and Errors evidence.

Run `csw_preflight` exactly once per invocation before the first `prepare`. Preserve its JSON verbatim. Exit `2` or `ok: false` is a concrete blocked result; never repeat or bypass its Git, OMP, profile, model, tool, spawn, or recursion validation.

`scan` returns immediate task children in task_ctl order. Valid records contain `name`, canonical `task_id`, relative `task_path`, exact spec/control/evidence paths, type, lifecycle status, order/priority, and canonical blocker records. Invalid controls retain the exact task identity and a concrete error instead of stopping sibling discovery. A missing spec is a blocker. A direct HLD/nested container is unsupported for this command.

`queue` returns:

- `ready`: every dependency-satisfied direct implementation child in `new`, `critic`, `planned`, or resumable `ready` state, with no skill concurrency cap;
- `waiting`: lifecycle/dependency waits and retained running/ready owner outcomes;
- `blocked`: invalid task inputs, unsupported containers, and failed/blocked execution outcomes;
- `already_done`: direct tasks whose canonical integrated lifecycle is done;
- `finished`: true only when all discovered direct tasks are done, including an empty target.

`prepare` provisions or reuses all currently ready worktrees and preserves independent preparation failures. Each worktree's `.cswd` is a symbolic link to the integration checkout's shared local metadata directory. Specifications, controls, and evidence remain unversioned; updates are immediately visible to every owner. Never stage or remotely synchronize `.cswd`. `control` refreshes the validated integration branch/head without target discovery or preparation.

## Local invocation state

Create one parent-owned temporary JSON file outside the checkout, initialized to `{}`:

```json
{
  "dependencies": {
    "child-name": [".cswd/tasks/target/other-child"]
  },
  "outcomes": {
    "child-name": {
      "status": "running|blocked|failed|ready",
      "reason": "nonempty explanation"
    }
  }
}
```

`dependencies` contains only supplemental semantic prerequisites and every value is an exact canonical `.cswd/tasks/...` ID. They augment, never erase, the canonical `blocked-by` records returned by task_ctl. Preserve actual prerequisites; missing canonical dependencies wait. Numeric order and priority are scheduling order, not dependencies.

`outcomes` prevents duplicate dispatch/retry within this invocation. Record prepared leaves as running before dispatch. Change to ready only after the owner ends with a provisional commit; use failed for a concrete implementation/runtime failure and blocked for an external prerequisite. Running/ready suppress redispatch but never satisfy dependencies. Canonical integrated done overrides a stale local outcome.

Only canonical `done` in the shared local task store unlocks an edge. The helper writes it only after successful integration. Ready, verified, a private branch, child report, or existing commit does not unlock an edge. Failed/blocked/running are local execution outcomes, never lifecycle statuses.

## Continuous orchestration

1. Run `scan`, create the outside-checkout state, and run `queue`. Do not parse, rewrite, or sort controls. Do not prepare an empty selection.
2. Before any preparation or child dispatch, run the shared preflight exactly once. Stop on failure. Discover the optional commit-review aliases as described below; they are not required implementer/rescue preflight dependencies.
3. Run `prepare`. Record preparation failures as blocked. Mark every successful prepared leaf running, then dispatch one task per record in returned order with `agent: "spec-run-all-implementer"`. Submit every eligible leaf, including a single leaf, without a skill-level cap. Harness admission limits may queue work; queued owners stay running and are never redispatched.
4. Give each child exact `repo_root`, `task_id`, `task_path`, worktree/spec/control/evidence paths, feature branch/base, integration branch/head, exclusive scope, known interface contracts, and required verification. The child uses assigned-worktree mode, skips validation, and never integrates/rebases or edits task files directly.
5. Consume individual owner completions continuously. Retain the owner/job identity and prepared record through review, any remediation rounds, and integration. Record crashes/dispatch failures as failed, external prerequisites as blocked, and provisional success as ready. An owner may report implementation failure only after the required one-time `spec-run-debug` rescue attempt.
6. As soon as one leaf is ready, serialize parent takeover for that leaf. Run `control`, rebase its one ready commit onto the current canonical head, and run focused plus repository-required combined verification in that exact worktree. Do not wait for unrelated owners. Route a recoverable failure back to the same owner after marking it running; preserve its worktree and one commit.
7. After observed verification success, use csw-run-worker `annotate --outcome verified` with exact evidence, `commit --status verified`, and `check --status verified`. Capture the helper-returned full commit ID, then perform the **Commit review gate** below in that exact worktree. Do not integrate while a review is pending, critical/high findings remain open, or an all-reviewer availability failure lacks developer consent. Remediation returns to step 6 with the same owner and worktree.
8. Only after the review gate permits this exact commit, call `integrate`. Do not write done yourself. The helper validates commit-bound local controls and evidence against the shared task store, rechecks the canonical head, and fast-forwards before advancing lifecycle to done through task_ctl. A rejected integration leaves dependencies locked. No task metadata or completion commit is added to Git history.
9. Immediately after successful integration, rerun `queue` on the canonical tree, then prepare and dispatch **all** newly eligible children before processing another completion or waiting. If A integrates while B runs, start every direct child now unlocked by A; B is not a barrier. Rescan after every settled outcome. Stop only at `finished` or a true stall with no eligible, running/queued, or completed owner/reviewer.

If the integration head advances before integration, refresh through `control`, rebase, rerun affected verification, update evidence, and repeat the review gate for the resulting commit before retrying. Never use the head captured at dispatch or reuse approval for an earlier commit ID. Never reprepare active work merely to refresh control state.

## Commit review gate

### Optional reviewer routing

After successful preflight, run `omp config list --json` from `repo_root` once and preserve the result as separate review-routing evidence. Read the effective `modelRoles` value; aliases may have keys with or without a leading `@`. The preflight JSON contains only implementer/rescue-relevant roles and models, so absence there does not establish that a reviewer alias is missing. Configuration discovery failure is a concrete blocker, not proof that no reviewers are configured.

Dispatch exactly one read-only subagent per configured alias on **every review round**:

| Configured alias | Agent profile | Required model |
| --- | --- | --- |
| `@csw-review` | `csw-review` | `@csw-review` |
| `@csw-review-2` | `csw-review-2` | `@csw-review-2` |

Use these `.omp/agents` profiles, not the default task/scout model or the root model. Honor alias chains, but never substitute another alias/model on failure. Inspect effective `task.agentModelOverrides` and disabled-agent settings from the same configuration: an override resolving to a different model or a disabled/missing profile is a routing blocker for that reviewer, not permission to silently change models. Do not edit user configuration. Keep optional reviewer failures separate from the mandatory preflight; do not rerun or weaken that preflight.

If neither alias is configured, record `review not configured` and continue after the existing verification gate; absent aliases are not failed reviewer attempts. If just one is configured, dispatch it. If both are configured, dispatch both concurrently in one task batch, even if they resolve to the same model, and collect both outcomes before deciding. Do not cancel a pending reviewer merely because the other succeeded.

### Review brief and result

The root passes each reviewer the exact full candidate commit ID, absolute prepared worktree path, canonical task ID, and exact specification path, with this explicit instruction:

> Read `skill://csw-review-commit` and follow it to review commit `<full-commit-id>` in worktree `<absolute-worktree-path>`. Read the supplied task specification for intent. Return its complete Markdown commit-review summary, including the Issues list with severity, locations, evidence, impact, and remedy for every finding, or an explicit empty list. Identify the reviewed commit and worktree, coverage, and limitations. Stay read-only; do not edit files, task metadata, review.md, or Git state. Skip builds, tests, linters, formatters, and project-wide validation; the parent owns verification. Do not delegate.

Reviewers use the existing owner worktree, not a new isolated worktree. Freeze mutations to that leaf while its reviewers run; unrelated implementers may continue. A successful review means a complete usable `csw-review-commit` report for the exact supplied commit/worktree, including a findings list or explicit no-findings result. Process exit alone, partial output, an incomplete review, or a report for another commit is not success. Retain supported findings from partial reports, but never treat partial coverage as approval.

### Findings, remediation, and evidence

The root compiles all returned issue lists and deduplicates by underlying defect and affected behavior/locations, not reviewer-local IDs or wording. Assign stable task-local finding IDs across rounds; preserve each source alias, source finding ID, severity, and commit. For severity disagreements retain the highest supported severity and record the rationale; do not lose a critical/high finding because another reviewer omitted it. Medium/low findings are recorded but do not block integration.

The root alone appends review evidence to `review.md` beside the exact supplied `spec_path` for this leaf, never the container's directory or a guessed basename. Create it if absent; preserve existing content. It is shared, local, unversioned `.cswd` evidence: never stage or remotely synchronize it, and do not change `task.yml` or `task.md` directly. Append a new round containing:

- Candidate full commit ID, baseline, worktree, attempted aliases, outcomes, and original reviewer summaries (including all reported findings, limitations, and failure evidence).
- **Open issues:** the deduplicated current list at every severity, with stable IDs, provenance, evidence, impact, and remedies. Include unresolved findings from earlier rounds; omission in a later report does not close them.
- **Solved issues:** a separate section identifying resolved finding IDs, their original report/commit, the fixing commit, and evidence from parent verification and review confirming the fix. An implementer claim alone does not mark an issue solved. Reopen regressions under the same ID.
- The gate decision, residual risks, and any explicit developer waiver. Record rejected findings separately with counterevidence; do not label an unfixed or waived finding solved.

Append before returning findings or integrating. Never replace earlier rounds; later round sections describe the current state while preserving history.

If any critical/high issue remains, send the compiled deduplicated list at **all severities**, with blocking IDs and both original summaries, back to the same implementer subagent that produced the commit and owns the worktree. Mark its local outcome running and resume that owner; do not replace it with a reviewer or let the root edit its implementation. Require it to fix every critical/high issue within the task scope and report dispositions for the list. It must use csw-run-worker `show`, `annotate --outcome ready`, and `commit --status ready` to amend/consolidate its existing one task commit, preserving its subject by omitting `--outcome` on `commit`; no extra fixup commit or direct Git mutation.

On the owner's next ready result, return to orchestration step 6: refresh/rebase, rerun required verification, bind verified evidence, and dispatch every configured reviewer again against the new full commit ID. Give reviewers the prior finding IDs and claimed fixes as additional context, but require a complete commit review, not only a fix check. Repeat until no critical/high findings remain. Unrecoverable owner failures retain the worktree and block integration; never bypass unresolved findings to finish the queue.

### Reviewer failures

- **At least one successful review:** proceed using the compiled findings without asking the developer about failed peer reviewers. Record those failures and reduced coverage. This permits the normal findings/remediation path, not integration with open critical/high issues.
- **No successful review, and every configured reviewer failed because its model was inaccessible or quota was exceeded:** display a warning naming the commit, aliases, and concrete errors, then ask the developer whether to ignore this review failure and proceed or retain the task as blocked. Only explicit consent permits proceeding without a successful review; silence, refusal, or unavailable interactive input leaves it blocked. Scope consent to this exact commit and round, record it in `review.md`, and retain any already-known critical/high blockers.
- **No successful review for other reasons:** retain a concrete failed/blocked review outcome and do not integrate. Invalid routing, malformed/incomplete reports, wrong commits, and tool/worktree failures are not quota failures and must not be silently waived by this availability exception.

Warnings/waivers never replace repository-required verification or satisfy dependency edges. Keep a review-blocked leaf in local outcomes with the precise cause and its owner/worktree intact; independent eligible leaves may continue.

## Child ownership and rescue

The assigned child reads `skill://csw-run-worker`, calls `show` first, works only in its exact worktree, reads complete requirements, implements the leaf, and uses:

```text
.omp/csw/bin/csw_run_worker ... annotate '<task_path>' --outcome ready --summary '<summary>'
.omp/csw/bin/csw_run_worker ... commit '<task_path>' --status ready --outcome '<behavior>'
```

It skips builds/tests/linters/formatters during the parallel pass and reports exact parent verification still required. It never directly edits `task.yml` or `task.md`, mutates Git, integrates, rebases, expands nested tasks, or delegates except to one `spec-run-debug` when genuinely stuck.

The debugger receives exact worktree/spec paths, constraints, current changes, error/dead end, observations, and attempted approaches. It is read-only and returns root cause evidence plus a concrete proposal to the same owner. The owner retains all implementation decisions and edits.

## Reporting

Report separately: already done; done and integrated with final commit and observed verification; failed with retained local outcome; nondependency blocked; and dependency waiting with exact canonical cause. Include target, integration branch/head, prepared worktrees/commits, review aliases/outcomes, deduplicated open/solved counts by severity, each leaf's `review.md` path, review waivers or unconfigured review, and residual risk. `finished: false` is not completion.

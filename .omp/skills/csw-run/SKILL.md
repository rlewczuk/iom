---
name: csw-run
description: Continuously orchestrate dependency-ready implementation worktrees, parallel dual reviews, and dedicated @csw-verifier verification and integration agents.
hide: true
---

# CSW Run

Run every unfinished **direct child** implementation task below `.cswd/tasks/<task-name>`. Read `skill://csw-run-worker` for its shared control-plane contracts, not permission for the root to implement or integrate. The target need not have its own control/spec. Report nested containers as unsupported; do not expand them.

## Ownership: root orchestrates, children execute

| Role | Profile/model | Responsibility |
| --- | --- | --- |
| Root | Invocation model | Determine scope, supplemental dependencies and interfaces; dispatch/resume agents; grant ownership; process structured outcomes; ask about explicit waivers; report. |
| Control | `csw-verifier` / `@csw-verifier`, control mode | Preflight once, scan/queue/prepare, atomic invocation-state updates, canonical completion confirmation. No implementation, review, or tests. |
| Implementation owner | `spec-run-all-implementer` / `@implementer` | Existing assigned-worktree implementation and ready-result contract, including its one-time `spec-run-debug` rescue. |
| Leaf verifier | One dedicated `csw-verifier` / `@csw-verifier` per implementation owner | All worktree/Git checks, rebase, required verification, review synthesis/evidence, and integration. Retain this peer across retries. Never implement fixes. |
| Two independent reviewers | `csw-review` / `@csw-review` and `csw-review-2` / `@csw-review-2` | Concurrent read-only exact-commit reviews using `skill://csw-review-commit`. No tests, fixes, or integration. |

The root MUST NOT inspect source/diffs to review, edit implementation or evidence, run preflight/Git/worktree helpers, build, test, verify, rebase, merge, or mark lifecycle done. Reading skill/spec requirements and agent/control JSON to decide what to schedule is orchestration. Delegating a command is not permission to run it locally as a fallback. Review deduplication, severity adjudication, and confirming fixes belong to the leaf verifier, not the root. A failed/unavailable specialist remains a blocker; never substitute the root or a default-model agent.

Use the named project profiles. No isolated/new harness worktrees: use the exact helper-prepared worktree. The control peer uses the supplied integration checkout. Do not alter user model configuration.

## Deterministic control plane

The **control peer**, never the root, runs:

```text
.omp/csw/bin/csw_run --repo <repo> --pretty session
.omp/csw/bin/csw_run --repo <repo> --pretty scan <target>
.omp/csw/bin/csw_run --repo <repo> --pretty queue <target> --state <state_path>
.omp/csw/bin/csw_preflight --repo <repo> --workflow csw-run --pretty
.omp/csw/bin/csw_run --repo <repo> --pretty prepare <target> --state <state_path>
.omp/csw/bin/csw_run --repo <repo> --pretty outcome <target> --state <state_path> --name <child> --status <status> --reason <reason>
.omp/csw/bin/csw_run --repo <repo> --pretty control
.omp/csw/bin/csw_run --repo <repo> --pretty depend <target> --state <state_path> --name <child> --on <canonical-prerequisite>
```

`task_ctl` alone discovers, validates, and orders canonical `task.yml` controls. Preserve returned numeric-order/canonical-ID ordering; never parse metadata from Markdown or guess basenames. PyYAML 6.0.3 is the supported dependency. Invalid/missing direct controls, missing specs, and unsupported nested containers are per-task blockers; valid independent siblings continue.

`session` creates `{}` in an outside-checkout temporary JSON file. Control is its **sole writer**. Preserve the existing state format:

```json
{
  "dependencies": {"child-name": [".cswd/tasks/target/prerequisite"]},
  "outcomes": {"child-name": {"status": "running", "reason": "owner identity and current phase"}}
}
```

Supplemental semantic dependencies only augment canonical `blocked-by`; order/priority are not dependencies. Never erase real prerequisites. Missing dependencies wait. Outcomes are `running`, `ready`, `failed`, or `blocked`, not lifecycle statuses. Mark successful preparations running **before dispatch**, ready only on the owner's provisional commit; keep ready throughout verification/review. Return to running only when the owner receives a repair lease. Retain failed/blocked records to suppress accidental redispatch. Canonical integrated `done` overrides stale local outcomes.

`queue` returns `ready`, `waiting`, `blocked`, `already_done`, and `finished`. Only canonical `done` unlocks dependencies: a ready/verified commit, reviewer approval, or agent success never does. `finished` requires every direct child done (including the empty selection). `prepare` provisions/reuses all currently ready leaves and preserves independent preparation failures. `.cswd` is a shared local symlink, never staged or remotely synchronized. Never recreate active worktrees merely to refresh the head.

Before returning `finished` for a nonempty selection, `queue` calls `task_ctl.complete_hld` under the existing integration lock to mark the requested HLD `done`. It rechecks that all direct children are canonically `done`; rerunning `queue` also repairs a stale parent when children were already done. Empty, uncontrolled, non-HLD, and cancelled targets stay unchanged; no ancestor walk or new control is created. Invalid controls or write failures are errors, not successful completion. No LLM lifecycle edit, roll-up annotation, worktree, or completion commit is needed.

Run preflight **exactly once per invocation**, before the first preparation or implementation dispatch. Preserve its JSON verbatim. It validates implementer/rescue, verifier, **both** review profiles, alias chains, model availability, tools/spawns, overrides, disabled profiles, and recursion. Both review aliases must resolve to different provider/model identities; different thinking settings on the same model do not qualify. Missing/misrouted aliases are blockers, not an optional no-review path. Do not rediscover reviewer routing, rerun failed preflight, or weaken it. The control agent itself uses the fixed verifier alias; if it cannot be dispatched, report that routing failure without starting implementation.

## Continuous event loop — never waves

Create a root-owned orchestration ledger with the invocation ID, control peer/state path, each prepared record, original owner/job ID, leaf verifier/job ID, monotonically increasing attempt number, candidate commit, active review pair, phase, grant holder, timer IDs, and artifact references. This is scheduling state, not a second task lifecycle. Preserve original implementer results without rewriting them. Use exact peer IDs returned by tools, never invented names.

1. Dispatch one control-mode verifier. It creates state, scans/queues, performs preflight once, and returns the records. Root resolves only genuinely necessary supplemental semantic dependencies from specs/contracts; control applies them. Do not prepare an empty selection.
2. Ask control to prepare **all** currently eligible leaves in helper order, record failures blocked, and mark each prepared leaf running. Dispatch one `spec-run-all-implementer` for each returned record, including a single leaf. Preserve current dispatch and result collection: same profile, exact paths/contracts, no validation, one ready commit, rescue before implementation failure. No skill concurrency cap or dependency waves; submit all ready work to harness admission, chunking only if its per-call limit requires it. Queued jobs retain ownership and are never redispatched.
3. Consume **individual** owner completions immediately. Keep owner identity/worktree through all remediation. Crashes/dispatch/implementation failures are failed; external prerequisites are blocked; provisional commits are ready. Immediately create that owner's dedicated leaf verifier and pass the untouched result and prepared record. Do not wait for unrelated owners.
4. The verifier stabilizes/rebases and verifies first. On failure it returns repair/blocked evidence; do not spend a review pair on a known failing candidate. On successful helper-bound verification it returns `review_required` with exact full commit/worktree, gate receipt and prior findings context.
5. Root dispatches **both reviewers concurrently in one task batch** for that candidate. Freeze mutations only to this leaf until **both** review attempts settle. Other owners, review pairs and verifiers continue. Forward each result/artifact to the verifier without interpreting source or findings; give it both outcomes before it decides the gate. Never cancel the second reviewer because the first succeeded.
6. The verifier synthesizes findings, appends evidence through `csw_verify`, and returns `repair`, `blocked`, `waiver_required`, or continues to integration. Root routes decisions; it does not redo review. Critical/high findings go to the original owner with the complete all-severity ledger and original reports. After owner repair, resume the same verifier with the new ready result. Keep worktree and one commit; never reprepare or replace the owner just for a retry.
7. The verifier integrates only with exact tested-and-reviewed authorization. All verifiers may run gates simultaneously. Serialize **only** short control/Git transactions through the helper's bounded repository lock, never tests, review, model work or IPC. Busy locks are retryable within the attempt deadline. A moved canonical head is stale verification, not an implementation defect: verifier refreshes/rebases and reruns affected gates; do not wake the owner unless a conflict or real failure requires edits. Approval for an old commit never authorizes a new one.
8. On successful integration, the verifier immediately notifies control **and** root. Control confirms canonical done with `queue`, prepares/claims all newly ready leaves, and returns them for immediate root dispatch **before** processing another unrelated completion or waiting. If A unlocks C while B runs, C starts now. Every settled failure/blocker also triggers a queue refresh so independent work is not stranded.
   While that control request is in flight, dispatch independently ready reviewer/verifier actions rather than waiting for the refill response. Prioritize newly prepared children as soon as the response arrives; control preparation is not a global scheduling barrier.
9. Wait only when no ready action is pending. A pending owner, reviewer, verifier, controller request, repair handoff, queued job, or undelivered completion is not a stall. Stop only on `finished`, or a concrete no-runnable-work stall with all remaining dependencies/errors reported. `finished: false` is never complete.

### Handoffs and direct IRC

Use OMP IRC to avoid copying logs/reports through root context. Root gives each peer the actual controller, original-owner, leaf-verifier and root IDs. Messages carry **invocation ID, canonical task ID, attempt, full candidate commit (or explicitly unavailable), worktree, event, and artifact reference**. Verifiers may send detailed repair packets directly to the original owner; reviewers may send reports directly to the verifier when their tools permit, but must still return their normal complete report. Root receives a compact notification and retains the authoritative completion/artifact. No source/log blobs in root messages.

Only root grants/resumes a worktree writer. A repair packet is **not** an edit grant. Before `implementer → verifier → frozen review → implementer` ownership transfer, settle/cancel and acknowledge the prior active lease and all review jobs. Reject stale attempt/commit messages, ignore duplicate events, and never accept an IRC claim alone as canonical done or review approval. Only control writes invocation state; only the owning verifier writes review evidence; only the original implementer edits code. If a peer is gone, retain artifacts and block that leaf rather than silently losing findings.

On a helper-managed rebase conflict, the verifier supplies the exact conflicts and grants no edits itself. Root temporarily returns the leaf to its original owner for **only the named conflict-file edits**. The owner still uses its normal result shape, reporting continuation pending in `OPEN`; it must not invoke Git, annotate, or commit while replay is stopped. After the owner relinquishes the files, the verifier calls helper `continue-rebase`. Repeat only for further named conflicts. Once replay finishes, return normal ready annotation/one-commit consolidation to the original owner, then verify afresh. No simultaneous editing and rebase continuation; root never resolves a conflict.

### Real deadlines, not invented task parameters

A verification attempt has a **7200-second command budget**; root gives the active verifier phase **7800 seconds wall time**, including model/tool overhead. Control requests get 600 seconds (preflight discovery commands have their own bound); review attempts get 1800 seconds each. Do not charge a parked verifier for time spent awaiting owner repairs, developer consent, or its reviewers. Each resumed active phase gets a new attempt/deadline; preserve retry evidence. After three consecutive no-progress failures/timeouts, block with evidence instead of an infinite resume loop. New code, a completed gate, or a canonical head advance is progress; unchanged repeated errors are not.

The task tool has no documented per-agent timeout field: **do not invent one**. Root starts a cancellable finite asynchronous `sleep <seconds>` job for each active phase (`bash`, `async: true`, `timeout: 0`), records its job ID next to the agent job, and cancels it when that phase settles. These clock-only commands are the sole root shell exception. Timer completion is an orchestration event: cancel the corresponding overdue agent, notify control, retain its artifacts, and require child/process cleanup acknowledgement before transferring the worktree. Ignore stale timers for settled attempts. If timer dispatch fails, stop dispatching unsupervised verifier work and report the concrete error.

The verifier also uses `csw_verify`'s real command deadlines: test command 900 seconds, build 1800 seconds, total attempt 7200 seconds; smaller project limits take precedence. CTest gets `--timeout 300` per test (or a justified finite project value). Remote commands need a **remote-side** `timeout --kill-after=30s ...` as well as the local bound; killing SSH alone does not kill remote tests. Bound GPU `flock` acquisition and execution separately. Follow `skill://csw-remote`, use unique mirrors for concurrent leaves, preserve SYCL setup, and enforce TTNN timeouts. No infinite waits, unbounded retries, detached tests, or lock held over remote execution.
For each remote helper call, the verifier sets `CSW_REMOTE_TASK_DIR` to the prepared record's local `spec_path` parent so `remote.log` is retained beside that task's control and specification.

## Review policy and retries

Review the green, helper-returned final commit, not the owner's earlier hash. Both reviewers receive:

> Read `skill://csw-review-commit` and review `<full-commit>` in `<absolute-worktree>`, using `<exact-spec-path>` for intent. Return its complete Markdown report, all findings or explicit no findings, exact commit/worktree, coverage and limitations. Include prior finding IDs and claimed fixes as context; perform a complete independent review. Stay read-only; no task/evidence/Git mutation or delegation. Skip builds, tests, linters and formatters; the dedicated verifier owns verification.

A complete usable report for the supplied commit/worktree is success; agent exit alone, partial coverage, malformed output or another commit is not. Retain supported findings even from partial/failed reviews. Medium/low findings are recorded but nonblocking; unresolved critical/high findings block regardless of peer omissions. Verifier assigns stable task-local IDs, deduplicates by defect, retains provenance/highest supported severity, and confirms fixes with actual evidence, not implementer claims.

The verifier appends each round to `review.md` **beside the exact spec**, preserving original reports, attempted aliases/outcomes, all open findings, solved findings with fixing commit/evidence, rejected findings with counterevidence, waivers, limitations and gate decision. Never label waived/unfixed findings solved, drop earlier unresolved findings, overwrite history, stage `.cswd`, or hand-edit lifecycle/evidence. `csw_verify` validates packet/receipt identity and enforces the gate; semantic adjudication remains the verifier's work.

- At least one complete successful review after both settle: retain reduced coverage/failures and proceed only if no critical/high blockers.
- Both attempts fail solely due to inaccessible model/quota: verifier returns `waiver_required`; root asks the developer whether to proceed without review for **this exact commit/round**. Explicit consent only; record it through verifier. Silence/refusal/unavailable input blocks. Existing critical/high findings remain blocking.
- No success for other reasons: blocked, no availability waiver. Missing profiles, same-model routing, wrong commits, malformed/incomplete reports and worktree failures are not quota exceptions.

Cache review authorization for the **exact final commit** in verifier state. Lock contention, an unchanged retry, or an infrastructure failure does not automatically launch another expensive pair. Inspect/reuse the script's valid receipt. Any changed commit, rebase, changed verification plan, or invalidated evidence requires the helper to reestablish gates; a different final commit requires a fresh pair **after** those gates pass. Never reuse approval by subject, branch, patch ID, implementer assertion or apparent similarity. Conservatively paying for a new commit's review is preferable to unreviewed integration. There is no global review or merge wave.

## Implementer brief and evidence

Keep the existing `spec-run-all-implementer` brief/result contract: exact `repo_root`, `task_id`, `task_path`, worktree/spec/control/evidence paths, feature branch/base, integration branch/head, exclusive scope, interfaces, and required verification. It reads `csw-run-worker`, calls `show` first, implements only its leaf, skips all validation, and uses helper `annotate --outcome ready` plus `commit --status ready`. Preserve the subject on amendments by omitting commit `--outcome`. No extra fixup commits, direct task-file edits, Git mutation, nested task expansion, or delegation except one `spec-run-debug` rescue before implementation failure. Report proposed gates as **not run**. Replace only the old statement that the parent verifies: the dedicated leaf verifier now owns that work.

## Reporting

Root reports from authoritative control/verifier artifacts, not a local audit: target and integration branch/head; already done; done/integrated with final commit and observed checks; failed/blocked with retained worktrees; dependency waits with exact canonical causes. Include prepared paths, owner/verifier identities, both reviewer aliases/outcomes, open/solved counts by severity, each review.md path, explicit waivers, deadlines/timeouts, retained blockers and residual risks. Do not claim a verifier's process exit proves integration; require its helper result **and** control's canonical done confirmation.

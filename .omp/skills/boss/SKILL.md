---
name: boss
description: Multi-model orchestration for coding and C++ inference reviews. Use only through /boss or when the user explicitly requests Boss orchestration by name; never activate it based on task complexity, parallelism, or review scope alone.
---

# Boss — multi-model orchestration for Oh My Pi

Activation requirement: Apply this skill only when the user explicitly invokes `/boss` or requests Boss orchestration by name. Task complexity, parallelism, delegation opportunities, or review scope alone must not activate it.

You are the supervisor. Own routing, briefs, scope, acceptance, synthesis, verification, and the final answer. Do not outsource accountability.

The intended deployment is asymmetric: use the visible/root OMP session on the strongest model appropriate for the job; use cheaper role-routed agents for bounded exploration and routine implementation; escalate only when the expected total cost of rework exceeds the stronger worker's premium.

### Deterministic preflight

Immediately after loading this skill and before inspecting repository state, Git metadata, OMP configuration, models, or agent profiles, run exactly once:

```text
.omp/csw/bin/csw_preflight --repo . --workflow boss --pretty
```

Preserve the complete JSON result as the invocation's environment evidence. This executable is the sole source for the Git baseline, effective OMP settings and role resolution, model availability, and bundled Boss-agent profile contracts. Exit status `0` means `ok: true`; exit status `2` or `ok: false` is a concrete blocker to Boss dispatch. Report its `errors` and stop before dispatch; do not bypass it, inspect OMP configuration files or databases, invoke `omp config`/`omp models`, read agent frontmatter for discovery, or reconstruct Git state with ad-hoc commands. A skill cannot switch the already-running root model; compare the running session identity supplied by OMP with the preflight's resolved `@slow` model without performing additional environment discovery.

After a successful preflight, a root-model mismatch is a consent gate, not a preflight error or an unconditional blocker. Before any further repository inspection or dispatch, display a warning that names both the running root model and the model resolved for `@slow`, explains that continuing may reduce review quality or invalidate the workflow's root-model assumptions, and ask the user: `Proceed with Boss using <running-model>?` Offer explicit `Proceed` and `Stop` choices and wait for the answer. If the user proceeds, record the accepted mismatch as an invocation deviation and continue with the current root model; if the user stops, end the workflow. Ask only once per invocation. When the models match, continue without prompting. Throughout Boss and its dependent workflows, references to a root `@slow` session describe the preferred deployment and include a current-model deviation explicitly accepted through this gate.


## 1. Triage

Work inline for coding only when ALL are true:
- at most ~3 files and ~100 changed lines are expected;
- the repository already contains a clear pattern to follow;
- there is one obvious implementation path;
- verification is at most two fast deterministic checks.

Always delegate or seek independent review for auth/security boundaries, permissions, migrations/persistent data, concurrency, destructive/external effects, breaking API changes, unresolved design choices, and C++ inference-engine reviews.

Keep progress updates terse. Do not narrate routine tool calls.

## 2. OMP lanes

OMP task dispatch selects an agent profile; that profile resolves its model through a role alias. Use these lanes:

| Lane | Agent | Default role | Use |
|---|---|---|---|
| lookup | `boss-errand` | `@smol` | read-only repository/docs/tool lookup; atomic factual questions |
| review-scout | `scout` | `@smol` | broad read-only review discovery, source/spec mapping, and evidence inventory |
| review-leaf | `boss-reviewer` | `@task` | one bounded substantive review area, hypothesis falsification, and candidate packets |
| fast | `boss-builder-fast` | `@smol` | mechanical, local, low-risk implementation |
| fast-deep | `boss-builder-fast-deep` | `@smol` | bounded but fiddly work on a cheap model with high thinking |
| standard | `boss-builder` | `@task` | normal implementation |
| standard-deep | `boss-builder-deep` | `@task` | normal model, hard contained reasoning |
| strong | `boss-builder-strong` | `@slow` | difficult/high-rework implementation |
| strong-deep | `boss-builder-strong-deep` | `@slow` | hardest bounded implementation |
| advisor | `boss-advisor` | `@advisor` | tool-free reasoning over a supplied evidence packet for a hard decision |
| debate-fast | `boss-advocate-fast` | `@smol` | cheap advocate in a structured coding design debate |
| debate | `boss-advocate` | `@task` | workhorse advocate |
| debate-strong | `boss-advocate-strong` | `@slow` | strongest advocate |

The profile's `model` alias is a routing request, not an automatic model switch. Use the preserved preflight result's `agents`, `roles`, and `models` values for routing. The root/supervisor should match the model resolved for `@slow`; a mismatch follows the warning-and-consent gate in the deterministic preflight section. Boss cannot switch an already-running model and must never claim that it did. Do not silently replace an unavailable cheap lane with an expensive one. The preflight validates the bundled profiles' declared tool boundaries, including packet-only advisors and read-only review leaves; if the harness later rewrites an effective allowlist, do not dispatch the affected profile unless that runtime allowlist remains explicit and safe.

## 3. Effort and cost control

Interpret user language as follows unless they explicitly configure something else:
- "cheap", "fast", "flash", "scout" -> `@smol` lanes;
- "normal", "workhorse" -> `@task` lanes;
- "strong", "deep", "careful", "frontier" -> `@slow` lanes;
- advisor work -> `@advisor`.

Route by expected TOTAL cost, including likely rework. Do not use a strong worker merely because it exists. There is one strong review supervisor, not five expensive area supervisors.

For repository exploration, prefer several narrow `boss-errand` calls over making the frontier supervisor search broadly. Ask factual questions such as "where is this instantiated?", "which tests cover this branch?", or "what invariant is enforced around this type?" Require exact `file:line` evidence and synthesize the results yourself.

## 4. Coding intake and reconnaissance

Trace only enough yourself to route and bound the coding task. Budget the supervisor roughly two quick local reads before delegating broader repository search.

For additional facts, dispatch `boss-errand` with atomic asks. Batch independent lookups in one task call where useful. Child agents receive no conversational history, so put all necessary facts in `context` and each task description.

Do not ask a cheap lookup agent to make the final architectural or correctness judgment. It gathers evidence; the supervisor reasons.

If the user has separately installed/configured an external reviewer or coding agent, it may be used as an optional second opinion or rescue lane only when explicitly available. Treat it as an external dependency; do not make this Boss package depend on a particular plugin name.

## 5. C++ inference review protocol

This section routes reviews; the shared review contract remains canonical. Before any review, read:

- `.omp/cpp-review/references/review-process.md`;
- `.omp/cpp-review/references/finding-rubric.md`.

Do not duplicate those files' domain checklists here. Preserve their exact-commit scoping, whole-codebase coverage accounting, mandatory simplification pass, candidate rubric, synthesis requirements, task template, numbering, collision, duplicate, and no-intermediate-`review.md` rules.
The five substantive areas are `cpp-inference-contract-correctness`, `cpp-inference-gpu-stability`, `cpp-inference-backend-simplicity`, `cpp-inference-numerical-testing`, and `cpp-inference-performance`; `cpp-inference-review-synthesis` is always the final pass. Use these exact skill names when routing review work.

### 5.1 Root deployment and scope

For a review, the preferred deployment runs the root/supervisor on `@slow`; a different running model may proceed only after the user accepts the mismatch through the deterministic preflight consent gate. Configuring `modelRoles.slow` or the root model remains an operator concern, not a hidden fallback in this skill. The root owns scope resolution, reviewed-state identity, specification mapping, routing, candidate acceptance, the final assignment table, task materialization decisions, and all final verification.

Resolve exactly one requested scope. In selected-commit mode, compare the target with its parent (or the empty tree for a root commit) and accept only defects introduced or materially exposed/worsened by that target. In whole-codebase mode, record the reviewed working-tree state and sampled versus exhaustive coverage. Preserve standalone behavior: when a `cpp-inference-*` specialist skill is invoked directly, the `@slow` root owns its selected area and invokes mandatory synthesis.

The exact-range exception is not a routine escalation path: use `scout` or `boss-errand` for ordinary discovery and reserve `@slow`/`@advisor` work for the decisions described below.

The root may read an exact known `file:line` range when that is cheaper than another dispatch. It must not compensate by broad source scans or ingesting an entire diff when a scout or bounded leaf can gather the evidence.

### 5.2 Cheap discovery and bounded review

Use `scout` first for broad source discovery, search, specification mapping, affected interfaces/callers/counterparts, and a compact evidence inventory. Use `boss-errand` for atomic factual follow-ups. Share confirmed facts once through the dispatch context or an internal artifact reference; do not make every worker repeat the same broad search.

After scope and scout evidence are resolved, dispatch independent `boss-reviewer` leaves in one batch, one assigned substantive area per leaf. Each leaf runs in orchestrated candidate-only mode: it returns rubric-complete candidate packets and never invokes orchestration or `cpp-inference-review-synthesis`. There are no five `@slow` area supervisors. Keep leaves read-only and bounded; they do not edit implementation/tests/specifications or create `review.md`.

Every candidate packet extends, rather than replaces, the finding rubric. It must preserve reviewed state/scope and baseline-versus-target provenance; exact path, line, symbol, and decisive minimal excerpts; SOURCE FACTS separated from INFERENCE; callers, guards, counterparts, and negative evidence; search coverage and uninspected areas; actual validation and gaps; a falsifier; and a complete remediation seed. Bound output and never truncate away evidence or coverage.

All independent work within a phase belongs in one batch: scout discovery, then review leaves, then synthesis support. Do not dispatch gratuitous agents for trivial known facts.

### 5.3 Advisor packets

`boss-advisor` is a tool-free `@advisor` lane. Use it only for a genuinely hard semantic, lifetime, numerical, security, disputed-evidence, key acceptance, or high-risk remediation decision. The supervisor must supply compact packet CONTENT, including the relevant excerpts and evidence; a path, URI, or artifact reference alone is invalid because the advisor cannot inspect it.

The advisor reasons only over that packet. It has no source/filesystem/web/shell/investigation tools, never edits or delegates, and never runs verification. If a material fact is missing, it returns `NEED EVIDENCE` with one exact question. The supervisor sends a cheap source-reading worker for that question and may return only the evidence delta for one follow-up. No advisor assertion becomes verified merely because models agree; the supervisor remains accountable.

Use advocates only for a real design fork. Give each advocate the same closed FACTS and candidates plus one position, then give their cases to the packet-only advisor when a hard judgment is needed. Do not use debate or advisor lanes for routine exploration, formatting, or every candidate.

### 5.4 Mandatory synthesis and materialization

Synthesis is mandatory after every review, whether orchestrated or standalone. Run `cpp-inference-review-synthesis` only after candidate collection. Use bounded `@task` work for independent disproof or routine drafting when useful, and `@smol` work for existence, collision, and path checks; use `@advisor` only for hard decisions. The `@slow` supervisor adjudicates candidates, deduplicates root causes, assigns severity/confidence/priority/order/blockers, and freezes the complete assignment table before any file is written.

Materialize accepted tasks through `boss-builder-fast` (`@smol`) or `boss-builder` (`@task`) only with exact destinations, the frozen assignment table, the shared template, and explicit non-goals. Builders have no design authority and must not allocate numbers concurrently. The orchestrator alone reads and validates the generated set with the existing validator. Workers propose exact validation commands; the root runs them after candidate collection and integration. For accelerator commands, follow `csw-remote`.

If delegation is unavailable, report the routing and coverage limits. Never silently perform an entire unassigned frontier review, substitute an expensive profile for a missing cheap profile, or hide a materially costlier fallback; request explicit permission before such a fallback. If no candidate survives, preserve the shared protocol's explicit no-findings output and coverage/validation-gap reporting.

## 6. Advisor and debate for coding

Before an expensive-to-reverse coding decision, use `boss-advisor` with a self-contained evidence packet, not a request to inspect code. The advisor must attack the plan before endorsing it and return `NEED EVIDENCE` when the packet is insufficient.

For a genuine coding design fork:
1. Frame 2–3 genuinely different candidate approaches.
2. Give every advocate the shared FACTS and all candidates, plus exactly one position to defend.
3. Run advocates in parallel only when the extra diversity is worth it.
4. Give the cases, relevant excerpts, and explicit questions to `boss-advisor`.
5. If the decision remains close, perform one rebuttal round by messaging the existing advocate sessions rather than spawning a new debate from scratch.
6. Supervisor decides and records the rationale.

## 7. Dispatch coding implementation

Before dispatch, use the preserved preflight result's `git.status` as the baseline and note its pre-existing changes.

Parallelize only independent work with disjoint files. Never ask workers to run verification while sibling mutations could affect the result.

Every builder brief must be self-contained and use this contract:

```text
GOAL: one bounded outcome
FILES: files/directories the worker may inspect or change
VERIFY: exact commands/checks for the supervisor to run after integration; a worker may run one only when the brief explicitly says the workspace is isolated and no sibling mutation is in flight
UNTOUCHED: files/areas/contracts the worker must not alter
DONE WHEN: observable completion criteria
FACTS: repository facts, constraints, decisions, and evidence already established
```

Every read-only lookup brief should be:

```text
ASK: one or a few atomic questions
TOOLS: specific read-only commands/docs/resources if relevant
DETAIL: concise | full
```

For OMP `task`, give a concise shared `context` and one task item per independent worker. Prefer a single batch call for parallel independent work.

## 8. Review the actual work

A worker report is a claim, not evidence.

For coding units, inspect status and the bounded relevant diff after each integration unit; do not force a broad source reread unrelated to the change. Check requirements, unintended scope, error paths, compatibility, security, and repository idioms.

For reviews, the root checks packet completeness, scope relation, evidence quality, coverage, deduplication, and the frozen assignment table. Read an exact known range only when cheaper than dispatching a read-only worker; do not re-run a broad review or demand whole-diff/source ingestion after bounded workers have supplied evidence.

Useful review triad:
- **correctness:** does the change or remediation do exactly what was requested, including edge/error paths?
- **containment:** did it touch only what it needed to touch?
- **evidence:** were decisive checks actually run, and do their outputs support the claim?

## 9. Bounce-back policy

If review finds a tiny obvious coding defect, fix it inline when that is cheaper than another worker turn.

For a substantial but bounded defect, send the original worker a DELTA containing:
- the concrete problem;
- `file:line` evidence;
- the required correction;
- the verification gate for the supervisor to run after integration.

Use OMP collaboration/Agent Hub messaging to continue the existing worker when possible. If the second attempt is still wrong, take over or escalate to a stronger lane. Do not bounce the same defect a third time.

## 10. Verification

The supervisor runs decisive gates after integration. Add checks in proportion to risk:
- targeted tests first, then broader tests when justified;
- typecheck/lint/build when they can catch the relevant failure;
- UI or visual verification when the task changes rendered behavior;
- migration/downgrade/rollback checks for persistent-data changes;
- security review for trust-boundary changes.

Review workers and discovery workers propose exact commands or scenarios; they do not run build/test/benchmark gates during parallel candidate collection. Do not report green that nobody ran. If a requested tool is unavailable, use the nearest useful check and state exactly what was not verified.

## 11. Landing and safety

`scout`, `boss-reviewer`, `boss-errand`, and `boss-advisor` are read-only lanes. In particular, `scout` and `boss-reviewer` have explicit read-only tool allowlists and no task/delegation or editing capability. Advocates are critique-only lanes under their own profiles. Builders never commit, push, or add AI attribution unless explicitly required and the supervisor decides to perform it.

The advisor's `tools: []` removes investigative tools, but the harness may still inject completion/coordination transport (`yield`/`hub`). Packet-only reasoning is also an explicit behavioral restriction: the advisor must not use coordination to inspect sessions, fetch evidence, delegate, or execute processes. This profile is not an OS sandbox.

Builder agents are intentionally denied the `task` tool, preventing recursive delegation through their declared tool allowlists. Read-only review leaves must never invoke orchestration or synthesis recursively.

OMP cannot faithfully reproduce a builder-only destructive-command hook without also affecting the root agent, because extension lifecycle/tool-call context does not expose reliable root-vs-subagent identity. This package therefore does not auto-enable a global destructive-command hook. Treat the supervisor's scoped diff review and worker contracts as the default protection.

Only commit when the user asks. Never push without explicit authorization.

## 12. Queues and multi-step changes

For 3+ dependent coding subtasks, maintain one reviewed integration unit at a time:
- carry forward only confirmed facts;
- keep each brief bounded;
- re-check repository state after every unit;
- update the remaining plan when an earlier unit changes assumptions;
- close with a coherence pass across all touched files.

For large work, keep a small scratchpad/progress file only if it materially reduces lost context; remove it before final delivery unless the user wants it retained.

## 13. Failure handling

Before cleanup or recovery, inspect status, relevant diffs, and untracked files so you do not destroy pre-existing user work.

Stop and surface the problem when:
- the brief is impossible or internally contradictory;
- a required repository fact cannot be established;
- the chosen plan is invalidated by code evidence;
- landed work contains a substantive bug;
- permissions or required external systems are missing.

Do not conceal uncertainty by silently broadening scope.

## 14. Final report

Keep the final response compact and accountable:
- what changed or, for review, the reviewed scope and accepted remediation assignments;
- decisive verification performed and actual result;
- material deviations, rejected/duplicate candidates, and unresolved risks;
- any follow-up the user actually needs.

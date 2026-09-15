---
name: cpp-inference-review-synthesis
description: Adversarially verify, deduplicate, prioritize, and materialize candidate findings from full or specialist C++ inference-engine reviews as implementation-ready remediation subtasks. Use as the mandatory final pass for cpp-inference reviews or independently on structured candidate packets.
argument-hint: "[candidate packets] [review scope] [spec docs/changes/... optional]"
---

# Review Synthesis & Direct Task Materialization

Read `skill://boss` **first**, before reading any repository file, reference, or other skill, and apply its root-model approval and warning-and-consent gate. The visible/root session should run as `@slow`; `@csw-yoda` is also automatically approved, while any other mismatch requires explicit user consent. This skill cannot switch an already-running model. Then read:

- `.omp/cpp-review/references/finding-rubric.md`
- `.omp/cpp-review/references/review-process.md`
- `.omp/cpp-review/templates/remediation-task.md`

The shared review process is canonical. This skill is the mandatory final pass for both the full orchestrator and every independently invoked specialist. It owns complete synthesis without spawning a nested supervisor, creating `review.md`, or deferring task generation to another skill.

## Synthesis routing

The root session, preferably launched with `@slow` or the automatically approved `@csw-yoda` alias under Boss's consent policy, owns packet intake, scope/coverage accountability, candidate acceptance, root-cause reconciliation, the complete assignment table, task materialization, generated-set validation, and the completion response. Synthesis is not a second broad repository review.

Use separate, bounded lanes:

- `boss-reviewer @task` performs independent adversarial evidence checks on candidates. It receives packet content and decisive excerpts/known ranges, may make narrow exact reads when needed, and never delegates, scans a review frontier, writes tasks, or runs gates.
- `boss-advisor @advisor` is tool-free and packet-only. Use it only for a genuinely hard semantic, lifetime, numerical, disputed-evidence, or key/high-risk acceptance/remediation decision. Supply compact packet **CONTENT**; it must never open a URI/path or access source, filesystem, web, shell, or other tools. If effective plan-mode tools would broaden its tools beyond this contract, do not dispatch it; report the routing limit.
- `boss-errand @smol` performs existence, path/symbol, destination, collision, and equivalent-task checks against exact known paths. It does not perform routine semantic review or broad scans.
- `boss-builder-fast @smol` mechanically drafts accepted task files after the root freezes the table. Use `boss-builder @task` only when the root explicitly chooses it for a bounded drafting need. Builders have no design authority and do not allocate numbers concurrently.

Run independent evidence/path checks in one batch when possible. A hard-decision advisor is conditional, not a routine call for every candidate. If an advisor returns `NEED EVIDENCE`, the root sends one exact question to `boss-errand @smol`, appends only that evidence delta to the packet, and may return the self-contained delta to the same advisor. The advisor's agreement never turns an unverified claim into verified evidence.

Workers never run compiler, build, test, sanitizer, profiler, benchmark, or other gates. They may identify a focused command and expected observation. The root runs actual validation after candidate collection and records the command, environment, and result. Accelerator checks follow `csw-remote` rules and execute on the selected remote Linux host over SSH; local inspection is not represented as a remote accelerator result.

When called from the full orchestrator, synthesis receives all five area packets from one concurrent batch. When called from a standalone specialist, the specialist's running root supplies its complete area packets and resolved scope under Boss's root-model consent policy. In neither mode may synthesis invoke another orchestrator, specialist review, or synthesis pass. In standalone mode, unavailable lanes do not authorize widening the assigned area into an unbounded whole-repository `@slow` review; report the area-only routing and coverage limit. If a required lane is unavailable, request explicit permission before a materially costlier fallback; do not silently substitute an expensive frontier review or claim missing checks ran.

## Inputs

Require:

- one resolved review scope and reviewed-state identity;
- review coverage and validation performed (distinguish root-run results from proposed worker commands);
- the shared reconnaissance/specification map and its search coverage/gaps when available;
- optional destination beneath `docs/changes/`;
- zero or more candidate packets using the common rubric.

The packet contract extends `.omp/cpp-review/references/finding-rubric.md`; it does not replace any rubric field. Every packet must preserve reviewed state/scope, exact path:line/symbol and decisive minimal excerpts, source facts separated from inference, relevant callers/guards/counterparts/negative evidence, search coverage and uninspected areas, baseline-v-target provenance for commit findings, actual validation and gaps, a falsifier, and a full remediation seed. Reference internal artifacts for workers when useful, but place compact packet CONTENT in any advisor request; an advisor must never need to open a URI/path. Bound output, split work rather than dump repository/raw logs, and never truncate away evidence or coverage.

When called independently, do not invent candidates from vague prose. Ask for missing candidate fields only when they cannot be recovered from the supplied scope and repository evidence. A standalone finalizer owns the complete finalization of the supplied area, not an unassigned repository frontier.

## 0. Freeze inputs and establish evidence coverage

Before dispatching checks, `[ROOT @slow]` confirms one scope, reviewed-state identity, baseline/target provenance where applicable, affected backends, specification requirements, and area coverage. It records sampled versus exhaustive search, uninspected areas, actual validation, and gaps. It may read exact known file:line ranges when cheaper than another dispatch, but must not broad-scan the repository. A cheap path worker receives exact known paths for existence/collision/equivalence checks; it is not asked to rediscover the review.

The root supplies each evidence checker with packet content and the smallest useful excerpts. Candidate packets are internal handoffs, not report sections. No task file is written until the root completes the assignment table in Section 6.

## 1. Adversarial gate

For every candidate, `[EVIDENCE boss-reviewer @task]` and then `[ROOT @slow]` ask:

1. Is it inside the requested scope?
2. In selected-commit mode, was it introduced or materially exposed/worsened by the target commit?
3. Is the backend/API assumption true?
4. Does another guard, owner, dependency, test, or code path already preserve the invariant?
5. Is there one concrete root cause rather than a collection of symptoms?
6. Is the failure or maintenance mechanism concrete and observable?
7. Is evidence stronger than preference?
8. Is severity based on impact rather than ugliness?
9. Could an ordinary compiler/linter completely explain it? If so, reject unless there is a larger invariant-level impact.
10. For performance, is the effect measured or mechanically certain on a relevant critical path? Otherwise retain only as a hypothesis with a decisive experiment.
11. Does the proposed remediation reuse the repository's established mechanism instead of creating a second pattern?
12. What evidence would falsify the claim?

The evidence worker reports disagreements, missing facts, negative evidence, and exact questions; it does not make final acceptance decisions. Use the packet-only advisor only when a hard decision remains. Reject candidates that fail this gate. Missing evidence does not become certainty.

## 2. Complexity and overengineering gate

Complexity reduction is a first-class review outcome, not an aesthetic appendix. Accept a structural candidate when repository evidence shows at least one objective reduction:

- fewer independently mutable states or invalid combinations;
- fewer sources of truth for a contract/capability/policy;
- deletion of duplicated implementation, validation, dispatch, or cleanup logic;
- deletion of a special-case branch or representation;
- removal of a forwarding wrapper or abstraction with no owned invariant;
- reuse of an existing mechanism instead of a parallel framework;
- removal of dead compatibility, caching, registry, or configuration machinery.

The task must identify what is deleted or consolidated, which invariant remains, and why behavior, diagnostics, lifetime, and critical-path performance remain intact. Reject:

- subjective “cleaner code” claims;
- helper extraction that only relocates duplication;
- generic frameworks for one use case;
- lowest-common-denominator abstractions that erase real backend differences;
- speculative future-proofing;
- changes that add more concepts than they remove.

A verified structural problem may be low severity even without a current runtime failure. Its concrete failure mode is maintenance divergence, invalid state, or redundant work demonstrated by current code—not hypothetical taste.

## 3. Deduplicate and reconcile

`[ROOT @slow]` merges candidates when one cause explains multiple symptoms or review areas. Preserve:

- the strongest location and evidence;
- every affected invariant/backend;
- secondary correctness, stability, numerical, and performance consequences;
- the smallest complete remediation.

Keep separate root causes separate even when they touch one subsystem. If two tasks interact, make their ownership boundaries explicit and add a blocker only when one requires an interface or mechanism delivered by the other.

Assign final IDs using the root cause's primary area:

- `CC-###` — contract/correctness
- `ST-###` — C++/GPU stability
- `AR-###` — architecture/simplicity
- `NT-###` — numerical correctness/testing
- `PF-###` — performance

Preserve specialist IDs when unique. Renumber collisions deterministically by area and source order. A finding ID alone is never proof that two tasks are equivalent.

## 4. Severity, confidence, and verification

Use verification states exactly:

- `verified` — reproduced/measured or proven by deterministic code-path reasoning;
- `strongly-supported` — concrete mechanism and substantial evidence, but direct reproduction unavailable;
- `hypothesis` — plausible material risk requiring the stated falsification experiment.

Severity:

- `critical` — likely corruption, deadlock, widespread wrong results, security-relevant unsafe behavior, or catastrophic broad failure;
- `high` — material wrong results, race/lifetime defect, backend breakage, major measured/mechanically certain performance regression, or common-path reliability failure;
- `medium` — bounded correctness/compatibility/performance defect or significant structural risk tied to a concrete invariant;
- `low` — real localized defect, duplication, or overengineering with limited impact. Never use low for style preference.

Confidence is an independent integer from 0 to 100.

The root preserves the distinction between actual root-run validation and proposed verification. For hypotheses, the task outcome is measurement/falsification and the suspected effect is not stated as fact.

## 5. Choose task destination, order, and collision behavior

### Specification-linked output

For a supplied destination, `[ROOT @slow]` resolves it with:

```bash
python3 .omp/cpp-review/scripts/resolve_spec_path.py --repo . --spec '<spec-dir>'
```

The root verifies the destination remains beneath `docs/changes/`. The resolver consumes `task_ctl.list_tasks` for existing lifecycle order and reports canonical records, while independently reserving every occupied numbered child directory, including directories without `task.yml` or `spec.md`. `[PATH boss-errand @smol]` checks current paths/symbols and searches existing `spec.md` files for equivalent tasks. It returns existence, collision, and equivalence evidence only; it does not decide acceptance.

- First new order is one greater than the largest order from task_ctl metadata or occupied numeric directory; start at `01` if none exist.
- Never fill an earlier gap.
- Width is at least two and no narrower than existing/new order width.
- Assign consecutive orders to accepted, not-already-materialized tasks.
- Directory shape is `<NN>-<FINDING-ID>-<remediation-slug>`.
- Preserve the uppercase finding ID.
- Derive the lowercase kebab-case slug from the remediation outcome, not the defect wording. Avoid generic `fix`, `cleanup`, `issue`, and `misc` slugs.
- Never overwrite or repurpose an existing directory. Resolve a name collision with a more precise slug.

Detect an existing equivalent task by matching the reviewed scope, root cause, and remediation in its `spec.md`; a matching finding ID alone is insufficient because IDs can recur across review runs. Report the existing path and do not duplicate it.

### Inline output

Without a specification directory, do not invent one. Return each task using the remediation template and identify lifecycle metadata as task_ctl fields (`type: impl`, `status: new`, `priority`, canonical `blocked-by`, and `source`); filesystem order may be omitted or shown as `unassigned`. Do not provide handwritten YAML.

## 6. Assign priority and blockers

`[ROOT @slow]` builds the full assignment table before writing any file. The table contains:

- final ID and title;
- source area and candidate order;
- final directory/order;
- priority with reason;
- canonical blocker task IDs;
- source parent `spec.md`;
- duplicate/collision decision;
- exact root-cause boundary.

The table is an in-memory planning record, not a second metadata format. After freezing it, write evidence to `spec.md` from the template, then invoke `.omp/csw/bin/task_ctl` (or its runpy-loaded API) with `set_task(..., {"type": "impl", "status": "new", "order": ..., "priority": ..., "blocked-by": [...], "source": "<parent>/spec.md"})` for each generated task. All lifecycle validation and writes belong to task_ctl; do not hand-edit `task.yml`, infer dependencies from prose, or embed these controls in `spec.md`.

Use the task contract, not blind severity conversion:

- **P0** — prerequisite, public-contract, correctness, memory-safety, lifetime, or backend-availability work that gates broad progress;
- **P1** — required normal-path or bounded remediation that does not broadly gate other work;
- **P2** — required independent finishing work such as measurement or cross-cutting verification. P2 is not optional.

Critical/high correctness and stability findings normally map to P0. A medium finding can be P0 when it protects a public ownership or compatibility invariant. Purely structural simplification normally maps to P1 unless it is prerequisite to another accepted task.

A blocker exists only when a task requires an interface, invariant, or mechanism from another task. Thematic similarity is not a dependency. Blockers must be canonical repository-relative task IDs naming existing tasks or earlier newly assigned tasks; no forward edges or cycles. Only `done` satisfies a dependency.

After the table is frozen, the root gives the drafting worker exact destinations, table values, candidate packet content, template, and any evidence deltas. The worker has no design authority and writes no task not present in the table. There is no concurrent order allocation.

## 7. Write one self-contained task per accepted root cause

`[DRAFT boss-builder-fast @smol]` (or the explicitly selected `boss-builder @task`) uses `.omp/cpp-review/templates/remediation-task.md` for evidence-only `spec.md`, then uses task_ctl to create/update the sibling `task.yml`. Each task must stand alone; an implementer must not need review prose, parent conversation, or sibling tasks to learn its contract. The builder receives exact task IDs and metadata from the frozen table, has no design authority, does not allocate numbers, and never writes YAML directly.

Use canonical repository-relative IDs for every `blocked-by` entry, with no forward edges or cycles.

Required content:

- reviewed scope and specialist/orchestrator evidence provenance in `spec.md`; lifecycle `source` is written to task.yml as the supplied parent `spec.md`;
- finding ID, area, severity, verification state, confidence, backend scope, and smallest useful location;
- observable corrected outcome;
- invariant, current failure or objective complexity mechanism, evidence, and impact;
- exact scope and non-goals;
- current implementation symbols, callers/counterparts, established analogue, and test touchpoints;
- normative implementation requirements with relevant shapes, sizes, ownership, ordering, cleanup, errors, compatibility, and backend boundaries;
- observable acceptance criteria;
- focused build/test/tool/benchmark commands or runtime scenarios.

Translate the candidate's remediation seed directly. Recheck paths/symbols that may have changed, but do not perform a second broad review. Resolve low-risk details from repository conventions. Select the smallest design that restores the invariant; do not leave competing designs for the implementer.

For structural tasks, name concrete deletions/consolidations and the single mechanism that remains. For hypotheses, make measurement/falsification the task outcome and do not state the suspected effect as fact.

Do not modify implementation, tests, parent specs, existing task specs, or unrelated files. If a destination is supplied, write only newly assigned task directories, evidence-only `spec.md`, and task_ctl-managed `task.yml`; never write `review.md`, indexes, manifests, or TODO files.

## 8. Validate the generated set

After writing, `[ROOT @slow]` reads every new `spec.md` and checks:

- every accepted non-duplicate root cause has exactly one task;
- no rejected candidate, residual observation, or validation note became a task;
- task_ctl order values are consecutive and match directory prefixes;
- IDs and slugs match directories;
- task_ctl priorities and canonical backward-only blockers match the assignment table;
- task_ctl source points to the supplied parent `spec.md`, with `type: impl` and `status: new`;
- no lifecycle controls are embedded in `spec.md`;
- invariant/failure/evidence and affected backends are preserved;
- paths and important symbols are current;
- acceptance criteria are observable and verification is focused;
- tasks do not duplicate mechanisms or leave design choices unresolved;
- simplification tasks reduce net concept count rather than move code around.

Then the root runs one command over the exact new files:

```bash
python3 .omp/cpp-review/scripts/validate_review_tasks.py \
  --spec-dir '<spec-dir>' \
  --task-file '<first-task>/spec.md' \
  --task-file '<next-task>/spec.md'
```

Repeat `--task-file` for every generated task. If validation fails, the root corrects the task (or sends a precise mechanical delta to the drafting worker) and reruns. This is document validation, not an implementation build. If no specification directory exists, the root validates the inline template fields and assignment metadata directly and writes no files.

## Completion response

Report only:

- reviewed scope and material validation actually performed by the root;
- ordered generated tasks with ID, priority, path, and blockers;
- equivalent existing tasks that prevented duplicates;
- any material unresolved hypothesis or destination collision;
- routing/coverage limits and any explicitly requested but unavailable costlier fallback;
- `No material findings; no remediation tasks generated.` when applicable.

The task specifications are the primary deliverable. Never claim that a worker-run gate, advisor source check, or unavailable lane occurred.

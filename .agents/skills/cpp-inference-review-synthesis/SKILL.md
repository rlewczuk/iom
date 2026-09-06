---
name: cpp-inference-review-synthesis
description: Adversarially verify, deduplicate, prioritize, and materialize candidate findings from full or specialist C++ inference-engine reviews as implementation-ready remediation subtasks. Use as the mandatory final pass for cpp-inference reviews or independently on structured candidate packets.
argument-hint: "[candidate packets] [review scope] [spec docs/changes/... optional]"
---

# Review Synthesis & Direct Task Materialization

Turn reviewed candidate packets into the smallest trustworthy set of implementation-ready remediation tasks. This is the shared final pass for both the full orchestrator and every independently invoked specialist.

Do not create `review.md`. Do not defer task generation to another skill.

Read:

- `.agents/cpp-review/references/finding-rubric.md`
- `.agents/cpp-review/references/review-process.md`
- `.agents/cpp-review/templates/remediation-task.md`

## Inputs

Require:

- one resolved review scope and reviewed-state identity;
- review coverage and validation performed;
- optional destination beneath `docs/changes/`;
- zero or more candidate packets using the common rubric.

When called independently, do not invent candidates from vague prose. Ask for missing candidate fields only when they cannot be recovered from the supplied scope and repository evidence.

## 1. Adversarial gate

For every candidate, ask:

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

Reject candidates that fail this gate. Missing evidence does not become certainty.

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

Merge candidates when one cause explains multiple symptoms or review areas. Preserve:

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

Preserve specialist IDs when unique. Renumber collisions deterministically by area and source order.

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

## 5. Choose task destination, order, and collision behavior

### Specification-linked output

Resolve a supplied destination with:

```bash
python3 .agents/cpp-review/scripts/resolve_spec_path.py --repo . --spec '<spec-dir>'
```

Inspect direct child directories whose names start with digits followed by `-`.

- First new order is one greater than the largest existing numeric prefix; start at `01` if none exist.
- Never fill an earlier gap.
- Width is at least two and no narrower than existing/new order width.
- Assign consecutive orders to accepted, not-already-materialized tasks.
- Directory shape is `<NN>-<FINDING-ID>-<remediation-slug>`.
- Preserve the uppercase finding ID.
- Derive the lowercase kebab-case slug from the remediation outcome, not the defect wording. Avoid generic `fix`, `cleanup`, `issue`, and `misc` slugs.
- Never overwrite or repurpose an existing directory. Resolve a name collision with a more precise slug.

Detect an existing equivalent task by matching the reviewed scope, root cause, and remediation in its `spec.md`; a matching finding ID alone is insufficient because IDs can recur across review runs. Report the existing path and do not duplicate it.

### Inline output

Without a specification directory, do not invent one. Return each task using the remediation template. Use stable finding IDs and priority/blocker metadata; filesystem order may be omitted or shown as `unassigned`.

## 6. Assign priority and blockers

Use the task contract, not blind severity conversion:

- **P0** — prerequisite, public-contract, correctness, memory-safety, lifetime, or backend-availability work that gates broad progress;
- **P1** — required normal-path or bounded remediation that does not broadly gate other work;
- **P2** — required independent finishing work such as measurement or cross-cutting verification. P2 is not optional.

Critical/high correctness and stability findings normally map to P0. A medium finding can be P0 when it protects a public ownership or compatibility invariant. Purely structural simplification normally maps to P1 unless it is prerequisite to another accepted task.

A blocker exists only when a task requires an interface, invariant, or mechanism from another task. Thematic similarity is not a dependency. Blockers must name existing task directories or earlier newly assigned directories; no forward edges or cycles.

Build the full assignment table before writing any file:

- final ID and title;
- source area and candidate order;
- final directory/order;
- priority with reason;
- blocker list;
- duplicate/collision decision;
- exact root-cause boundary.

## 7. Write one self-contained task per accepted root cause

Use `.agents/cpp-review/templates/remediation-task.md`. Each task must stand alone; an implementer must not need review prose, parent conversation, or sibling tasks to learn its contract.

Required content:

- reviewed scope and specialist/orchestrator source;
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

Do not modify implementation, tests, parent specs, existing task specs, or unrelated files.

## 8. Validate the generated set

After writing, read every new `spec.md` and check:

- every accepted non-duplicate root cause has exactly one task;
- no rejected candidate, residual observation, or validation note became a task;
- order values are consecutive and match directory prefixes;
- IDs and slugs match directories;
- priorities and backward-only blockers match the assignment table;
- reviewed scope and source metadata are present;
- invariant/failure/evidence and affected backends are preserved;
- paths and important symbols are current;
- acceptance criteria are observable and verification is focused;
- tasks do not duplicate mechanisms or leave design choices unresolved;
- simplification tasks reduce net concept count rather than move code around.

Then run one command over the exact new files:

```bash
python3 .agents/cpp-review/scripts/validate_review_tasks.py \
  --spec-dir '<spec-dir>' \
  --task-file '<first-task>/spec.md' \
  --task-file '<next-task>/spec.md'
```

Repeat `--task-file` for every generated task. If validation fails, correct the task and rerun. This is document validation, not an implementation build.

## Completion response

Report only:

- reviewed scope and material validation;
- ordered generated tasks with ID, priority, path, and blockers;
- equivalent existing tasks that prevented duplicates;
- any material unresolved hypothesis or destination collision;
- `No material findings; no remediation tasks generated.` when applicable.

The task specifications are the primary deliverable.
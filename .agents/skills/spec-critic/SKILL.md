---
name: spec-critic
description: Review and refine a change specification against the current repository, user remarks, project conventions, edge cases, non-functional concerns, and automated testability. Use only through /spec-critic <spec-name> or when explicitly requested.
hide: true
---

# Spec Critic

Review one change specification until it is accurate, focused, testable, and mutually understood, then edit the specification in place.

The command supplies a **spec name**. Work from the project root and use exactly these paths:

Note that `<spec-name>` may refer either to directory directly in `doc/changes` or any subdirectory, for example `0123-some-changes/03-cleanups`.

- Specification: `docs/changes/<spec-name>/spec.md`
- User remarks, optional: `docs/changes/<spec-name>/spec-fixme.md`

Modify only `spec.md` unless the user explicitly requests another file change. Never modify or delete `spec-fixme.md`.

## Guardrails

- Treat `<spec-name>` as one directory name. Reject an empty name, `.` or `..`, path separators, or path traversal.
- If `spec.md` does not exist, report the expected path and stop.
- Read the whole specification and the whole fixme file when present.
- The repository is authoritative for claims about current behavior. Inspect code instead of relying on the specification's description of it.
- Answer questions that the repository can answer by reading the repository. Ask the user only for intent, policy, product behavior, or trade-offs that cannot be derived from evidence.
- Ask exactly one question at a time. Wait for the answer before asking the next one.
- Prefer a recommended answer, with a brief reason, when a sensible default exists.
- Keep the design minimal. Remove or challenge speculative abstractions, future-proofing, generic frameworks, optional extras, unrelated cleanup, and complementary features not required by the stated change.
- Do not broaden scope merely because another improvement would be useful. Mention an adjacent issue only when it blocks correctness of this specification.
- Apply clear factual and editorial corrections directly. Use questions for genuine decisions, not for permission to fix obvious errors.
- Do not implement the change. The output of this workflow is the revised `spec.md` and a concise review summary.
- Ignore previous changes to `spec.md` and `spec-fixme.md` (if visible in git history), focus on current version. 

## Workflow

### 1. Establish the review target

Resolve the two paths from the supplied spec name. Read `spec.md`; read `spec-fixme.md` if it exists. Treat fixme content as user-provided remarks and additional constraints, but reconcile it with the codebase and ask when it conflicts with repository reality or with another remark.

While reading, extract:

- the intended outcome and explicit non-goals;
- factual claims about existing files, symbols, APIs, defaults, data, configuration, and behavior;
- proposed behavior, interfaces, state transitions, and implementation touchpoints;
- assumptions, unresolved choices, examples, acceptance criteria, and testing claims;
- every actionable remark from `spec-fixme.md`.

### 2. Ground the specification in the current project

Inspect all referenced files and enough surrounding code to understand the actual conventions. Search for analogous features and recent related changes in the repository. Read existing tests, configuration, schemas, migrations, public interfaces, and documentation only where relevant.

Verify at least:

- referenced paths, symbols, signatures, defaults, ownership, and current behavior;
- whether proposed names and file placement follow local conventions;
- whether all necessary touchpoints are present and unnecessary ones are removed;
- compatibility with current architecture, dependencies, data model, configuration, and supported platforms;
- whether the proposal duplicates an existing facility or bypasses an established extension point;
- whether examples and pseudocode are consistent with the actual project.

Do not infer current code behavior from names alone. Read implementations and call sites where the distinction matters.

### 3. Critique quality and scope

Review the specification as an implementation contract, not as prose alone.

Check for:

- contradictions, stale claims, missing prerequisites, circular dependencies, and impossible ordering;
- vague terms such as “appropriate”, “simple”, “fast”, “normally”, “etc.”, or examples standing in for required rules;
- unspecified inputs, outputs, errors, side effects, ownership, lifecycle, persistence, and rollback behavior;
- hidden scope or design that is more general than the requested change needs;
- requirements that prescribe internals without a demonstrated constraint;
- missing non-goals that would otherwise invite implementation drift.

Prefer the smallest design that satisfies the stated outcome and fits existing project patterns. Reuse existing mechanisms before introducing new layers. A focused specification should say what must change, what must remain unchanged, and how success is observed.

### 4. Check corner cases proportionally

Investigate functional surfaces relevant to the change, including where applicable:

- empty, missing, malformed, boundary, duplicate, and unsupported inputs;
- partial failure, retries, cancellation, cleanup, idempotency, and recovery;
- concurrency, ordering, races, atomicity, and repeated invocation;
- compatibility, migration, downgrade, rollback, and mixed-version behavior;
- permissions, trust boundaries, validation, secrets, and information exposure;
- state transitions, stale state, cache invalidation, and persistence;
- interactions with existing callers, configuration, tooling, and tests.

Check non-functional surfaces only when the change can materially affect them: performance and resource bounds, reliability, security and privacy, observability and diagnosability, accessibility, portability, operability, and maintainability. Require measurable behavior where a non-functional requirement matters; do not inflate the spec with a generic checklist.

### 5. Audit automated testability

For every normative behavior and acceptance criterion, determine whether an automated test can reliably distinguish success from failure.

Require the specification to identify, explicitly or by clear implication:

- observable inputs, outputs, state changes, errors, or emitted events;
- deterministic control of time, randomness, concurrency, environment, and external services where relevant;
- the appropriate test level: unit, integration, contract, end-to-end, migration, performance, or security;
- fixtures, seams, fakes, or dependency boundaries needed for reliable automation;
- negative and boundary cases, not only the happy path.

Rewrite requirements that are subjective or non-verifiable into observable criteria. Do not demand a unit test when the behavior can only be meaningfully verified at another automated level. Flag genuinely manual-only requirements and ask whether that limitation is acceptable.

### 6. Resolve ambiguities with the user

First apply straightforward corrections supported by repository evidence and unambiguous fixme remarks. Then build a private list of unresolved decisions ordered by dependency and implementation risk.

For each unresolved decision:

1. State the concrete ambiguity and why it matters.
2. Ask one focused question.
3. Give the recommended answer when evidence supports one, plus the main trade-off.
4. Wait for the user's response.
5. Incorporate the answer into `spec.md` promptly, in the project's terminology.
6. Re-check affected sections and continue with the next unresolved decision.

Do not dump a questionnaire. Do not ask the user to decide facts available in the code. Do not manufacture choices merely to prolong the interview. Stop questioning only when no material ambiguity remains and the user and model have a common, written understanding.

### 7. Resolve unnecessary complexities

Look for ways to shorten and simplify dsign and the spec whenever possible.
  
For each accidental complexity or non-essential/redundant feature found:

1. State the concrete ambiguity and why it matters.
2. Assess if feature should be kept or removed.
3. Check if removing feature will affect other features or aspects of design.
4. If unsure, ask user to decide what to do with it; wait for user response if asked.
5. If feature is decided to be removed or simplified, edit `spec.md`.
6. Re-check affected sections and spec for consistency. 

### 7. Finalize the specification

After the last answer, re-read the entire edited `spec.md` against:

- the current repository;
- every applicable fixme remark;
- decisions made in the conversation;
- minimal-scope and testability requirements.

Ensure the final document is internally consistent and implementation-ready. Preserve useful existing structure, but reorganize when necessary for clarity. Remove resolved questions, stale alternatives, duplicated text, speculative extensions, and instructions superseded by later decisions.

The final specification should contain, as appropriate to the change:

- purpose, scope, and explicit non-goals;
- current behavior and verified constraints;
- precise proposed behavior and affected interfaces;
- error and edge-case behavior;
- relevant non-functional requirements;
- implementation touchpoints without unnecessary implementation micromanagement;
- observable, automatable acceptance criteria and test strategy;
- migration, compatibility, rollout, or rollback details when required.

Do not leave `TBD`, contradictory alternatives, or unresolved questions unless the user explicitly accepts them as a documented blocker. In that case, label the specification as blocked and do not claim it is implementation-ready.

## Completion response

When the specification is finalized, report only:

- the path edited;
- the most important corrections and decisions;
- how automated testability is represented;
- any explicit blocker or residual risk.

Keep the report concise. The edited specification is the primary deliverable.

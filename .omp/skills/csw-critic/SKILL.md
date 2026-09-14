---
name: csw-critic
description: Review and refine a task specification against the current repository, user remarks, project conventions, edge cases, non-functional concerns, and automated testability. Use only through /csw-critic <task-name> or when explicitly requested.
hide: true
---

# CSW Critic

Review one change specification until it is accurate, focused, testable, and mutually understood, then edit the specification in place.

The command supplies a **task name**. Work from the project root and use exactly these paths:

`<task-name>` may refer either to a directory directly in `.cswd/tasks` or to any nested task directory, for example `0123-some-changes/03-cleanups`.

- Specification: `.cswd/tasks/<task-name>/spec.md`
- User remarks, optional: `.cswd/tasks/<task-name>/spec-fixme.md`

Modify only `spec.md` unless the user explicitly requests another file change. Never modify or delete `spec-fixme.md`.

## Guardrails

- Treat `<task-name>` as a repository-relative task path beneath `.cswd/tasks`. Reject an empty name, absolute paths, empty path components, `.` or `..` components, backslashes, and any path that escapes `.cswd/tasks`.
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

## Boss orchestration protocol

Before this workflow, the root MUST read `skill://boss` and follow Boss's canonical preflight, dispatch, brief, and verification rules. This skill adds only csw-critic routing and does not import any C++ review protocol. The root supervisor MUST already match the preflight-resolved `@slow` model; the skill MUST NOT switch models, edit model configuration, silently fall back, or inherit an expensive parent when a requested lane is unavailable. Use only the preserved Boss preflight's `roles`, `models`, and required `agents` entries to establish the `@slow`, `@task`, `@smol`, and `@advisor` routes, profile tool restrictions, and passive-advisor state. Do not repeat or manually reconstruct that discovery. A failed preflight is reported and dispatch stops.

The `@slow` root owns scope, complete `spec.md`/`spec-fixme.md` intake, decomposition, evidence synthesis, user questions, architectural and final decisions, integration, and verification. It asks the user only about intent, policy, product behavior, or trade-offs that evidence cannot settle.

- `boss-builder` (`@task`) handles bounded substantive source-fact gathering: current behavior, callers, conventions, edge cases, and testability. This explicitly overrides Boss's default `@smol` broad discovery for substantive facts. Each brief sets `UNTOUCHED` to all files and forbids edit/write/`ast_edit`/`bash`, delegation, and gates; it returns a facts packet in its normal report. Read-only is a brief restriction, not a sandbox or tool-allowlist guarantee.
- `boss-errand` (`@smol`, flash-class) handles batched, atomic mechanical lookups such as path/existence checks. It has no design authority.
- `boss-builder-fast` (`@smol`, flash-class) handles all simple mechanical editorial/materialization work after root approval of exact edits to `spec.md`. It has no design authority, never edits `spec-fixme.md`, and never runs implementation work.
- `boss-advisor` (`@advisor`) handles only key architectural decisions from supplied packet content. It is tool-free: it never searches, uses tools, or delegates.

Every substantive packet labels `SOURCE FACTS` separately from `INFERENCE`, gives exact `file:line` references with minimal excerpts, and records contradictions, missing evidence, and inspected versus uninspected coverage. Batch independent areas and reuse confirmed packets rather than repeating discovery. Never dispatch one worker for a known trivial fact when the root can answer it; batch independent work when delegation is useful.

Advisor `CONTENT` must inline the goals and non-goals, verified relevant excerpts, alternatives, constraints, trade-offs, and the exact decision requested; an artifact path alone is invalid. The advisor reasons only over that content. On missing material facts it returns `NEED EVIDENCE` with one exact missing-fact question; the root gets substantive answers from `boss-builder` (`@task`) and only mechanical lookups from `boss-errand` (`@smol`), then supplies the advisor an evidence delta. The root remains accountable for the decision.

Only `spec.md` is materialized as workflow output; the completion response carries its concise review summary. `spec-fixme.md` is immutable. Workers skip build, test, lint, and format gates; the root verifies the spec artifact and its consistency, never an implementation.

## Workflow

The eight substantive stages below are one Boss-owned workflow; delegation never creates a separate review process.

### 1. Establish the review target

The `@slow` root performs the complete intake before decomposition. It may batch independent, purely mechanical path checks through `boss-errand` (`@smol`), but does not delegate known trivial facts gratuitously.

Resolve the two paths from the supplied task name. Read `spec.md`; read `spec-fixme.md` if it exists. Treat fixme content as user-provided remarks and additional constraints, but reconcile it with the codebase and ask when it conflicts with repository reality or with another remark.

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

The root decomposes the remaining factual work into bounded, independent packets and batches `boss-builder` (`@task`) workers for substantive source evidence. Workers receive explicitly read-only briefs and report behavior, callers, conventions, edge cases, and testability with the packet evidence contract; they never edit, orchestrate recursively, or run gates. The root reuses packets across later stages.

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

The root synthesizes the evidence and critique, keeping `SOURCE FACTS`, `INFERENCE`, contradictions, missing evidence, and coverage distinct. For a genuinely key architectural decision, it sends a compact inline `CONTENT` packet to the tool-free `boss-advisor` (`@advisor`) and retains the final decision.

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

When corner-case evidence is not already known, the root batches bounded substantive questions to read-only-briefed `boss-builder` (`@task`) workers; only mechanical path or existence checks go to `boss-errand` (`@smol`). Neither lane decides design or edits files.

Check non-functional surfaces only when the change can materially affect them: performance and resource bounds, reliability, security and privacy, observability and diagnosability, accessibility, portability, operability, and maintainability. Require measurable behavior where a non-functional requirement matters; do not inflate the spec with a generic checklist.

### 5. Audit automated testability

For every normative behavior and acceptance criterion, determine whether an automated test can reliably distinguish success from failure.

Require the specification to identify, explicitly or by clear implication:

- observable inputs, outputs, state changes, errors, or emitted events;
- deterministic control of time, randomness, concurrency, environment, and external services where relevant;
- the appropriate test level: unit, integration, contract, end-to-end, migration, performance, or security;
- fixtures, seams, fakes, or dependency boundaries needed for reliable automation;
- negative and boundary cases, not only the happy path.

The root owns the testability judgment and acceptance wording. It may ask a read-only-briefed `boss-builder` (`@task`) worker for substantive evidence about existing tests, seams, fixtures, or observable behavior, then incorporates the facts into the specification rather than delegating the requirement decision.

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

The root alone asks the user, exactly one question at a time, and only for intent, policy, product behavior, or trade-offs. It may consult `boss-advisor` (`@advisor`) on a key architectural choice using inline packet content; an advisor `NEED EVIDENCE` response is resolved by the root through the prescribed `@task` or mechanical `@smol` lookup and a delta packet.

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

After a simplification or decision is settled, the root supplies `boss-builder-fast` (`@smol`) only exact, root-approved editorial/materialization edits. One writer at a time may edit `spec.md`; no worker may edit `spec-fixme.md`, make design decisions, or implement the change.

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

The `@slow` root integrates all approved edits and performs the final whole-spec consistency review against the repository, fixme remarks, user decisions, scope, and testability requirements. Workers skip validation gates; the root verifies the resulting `spec.md` artifact, not an implementation, and may use only exact editorial edits or batched mechanical checks to close findings.
Do not leave `TBD`, contradictory alternatives, or unresolved questions unless the user explicitly accepts them as a documented blocker. In that case, label the specification as blocked and do not claim it is implementation-ready.

## Completion response

When the specification is finalized, report only:

- the path edited;
- the most important corrections and decisions;
- how automated testability is represented;
- any explicit blocker or residual risk.

Keep the report concise. The edited specification is the primary deliverable.

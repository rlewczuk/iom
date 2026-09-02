---
name: review-to-task
description: Convert explicit findings in a change review into consecutively numbered, self-contained remediation task specifications. Use only through /review-to-task <change-name> [finding filter] or when explicitly requested.
argument-hint: "<change-name> [finding filter]"
hide: true
---

# Review to Task

Turn accepted findings from a change's `review.md` into implementation-ready task specifications. Create one task per selected finding; do not implement the fixes.

The command accepts:

```text
<change-name> [finding filter]
```

`<change-name>` names a directory relative to `docs/changes/`. The optional remainder narrows the findings to convert. Examples:

```text
/review-to-task 0001-tensor-view
/review-to-task 0001-tensor-view CC-001 ST-003
/review-to-task 0001-tensor-view high-severity ROCm findings
/review-to-task allocator/gpu-layout findings except performance hypotheses
```

For target `<target>`, use exactly:

- review source: `docs/changes/<target>/review.md`;
- parent specification, when present: `docs/changes/<target>/spec.md`;
- existing task specifications: `docs/changes/<target>/<NN>-<task-slug>/spec.md`;
- new review tasks: `docs/changes/<target>/<NN>-<FINDING-ID>-<remediation-slug>/spec.md`.

The numeric prefix records the task's position in the change. The finding ID preserves traceability to the review. The generated directories plus the completion response are the task list; do not create an index, manifest, or repository TODO file unless the user requests one.

## Guardrails

- Require one non-empty target argument. Reject absolute paths, `.` or `..` components, empty path components, backslashes, and any target that escapes `docs/changes/`.
- Allow safe nested targets such as `allocator/gpu-layout`; the argument names the exact directory containing `review.md`.
- If `review.md` does not exist, report its expected path and stop. Read the complete review, not selected snippets.
- Treat direct conversation instructions as highest priority, including an explicit finding filter. Treat `review.md` as the authoritative set of accepted findings. Use the repository as authority for current paths, symbols, behavior, and test commands while expanding a finding.
- Do not reinterpret synthesis prose, residual risks, validation gaps, suggested follow-ups, or general observations as findings. Only explicit finding sections are eligible.
- Do not modify `review.md`, a parent `spec.md`, existing task specifications, implementation files, tests, or unrelated files.
- Do not silently overwrite or repurpose an existing task directory. A prior task for the same finding makes that finding already converted unless the user explicitly requests revision.
- Keep each task limited to the finding's root cause and the smallest complete remediation. Exclude adjacent cleanup, speculative abstractions, complementary features, and unrelated defects.
- Preserve the finding's invariant, failure mode, affected backends, error behavior, and verification burden. Do not weaken a verified finding into a vague cleanup task.
- Do not implement the change. The deliverables are final `spec.md` files written by same-model subagents and a concise completion list.

## Workflow

### 1. Resolve the target and optional filter

Split the invocation into the target and the optional filtering instruction. The first argument is always the target; all remaining words form one filter expression. Normalize the target before reading any file.

If no filter is supplied, select every eligible finding. If a filter is supplied, it is restrictive: never add findings merely because they are related to a match.

A filter may name or combine:

- exact finding IDs, such as `CC-001` or `ST-003`;
- review areas;
- severity, verification state, confidence, backend scope, or scope relation;
- words or phrases in finding titles or bodies;
- exclusions such as `except`, `exclude`, or `not`.

Use exact metadata from each finding section. Combine constraints according to the user's natural-language intent; for example, `high ROCm` requires both high severity and ROCm scope, while `CC-001 or ST-003` selects either ID. Matching is case-insensitive except for emitted finding IDs, which retain their source spelling.

When two materially different selections remain plausible, ask one focused clarification and show the candidate IDs. Do not ask for approval when the filter has one reasonable interpretation. If nothing matches, create no directories and report the filter plus the available finding IDs.

### 2. Extract explicit finding sections

Parse the complete review structurally. In the standard review format, an eligible finding is a level-three heading of this form inside the review's finding areas:

```markdown
### CC-001 — Concrete finding title
```

The section runs from that heading to the next heading of the same or higher level. The heading must carry a stable finding ID such as `CC-001`, `ST-002`, `AR-001`, `NT-004`, or `PF-003`, and its body must be presented as a finding with review metadata and remediation evidence. Normally this includes `Severity`, `Location`, `Recommended fix`, and `Verification method` fields.

Accept an equivalent explicitly labeled finding heading when a review uses a different area taxonomy, but require both a stable ID and a concrete remediation section. Do not infer a finding from an ordinary paragraph or bullet.

Never select:

- review metadata;
- area headings;
- `No material findings.` text;
- overall assessment or synthesis subsections;
- cross-area root causes that summarize already listed findings;
- residual risks, verification gaps, or suggested validation sequences without their own explicit finding ID and finding section.

Record for every eligible finding:

- ID and title;
- enclosing area and source order;
- severity, verification state, confidence, scope relation, and backend scope;
- locations, invariant, failure mode, evidence, impact, recommended fix, and verification method.

Preserve the review's source order as the stable default order.

### 3. Detect existing tasks and choose new numbers

Inspect only direct child directories of `docs/changes/<target>/` whose names begin with digits followed by `-`.

- Parse every numeric prefix as an integer.
- Let the first new number be one greater than the largest existing prefix. If none exist, start at `01`.
- Never fill an earlier numbering gap.
- Use a width of at least two digits and never narrower than the widest existing numeric prefix or the new number.
- Assign consecutive numbers to the selected, not-already-converted findings.

Before assigning a number, detect prior conversion by checking existing task directory names and task `spec.md` source metadata for the finding ID. If an existing task names the ID or explicitly cites the same review finding as its source, skip it and report its existing path. Do not duplicate it under a new number.

Use directory names of the form:

```text
<NN>-<FINDING-ID>-<remediation-slug>
```

Keep the source ID uppercase. Derive a short lowercase kebab-case slug from the concrete remediation outcome, not merely the defect wording. Prefer `17-CC-001-validate-ttnn-native-extents` over `17-CC-001-ttnn-bug`. Avoid `fix`, `cleanup`, `issue`, `misc`, and other generic slugs unless they are made specific by the rest of the slug.

If a proposed destination already exists but is not a prior conversion of the same finding, choose a distinct precise slug. Never overwrite the collision.

### 4. Establish order, priority, and blockers

Use review order unless a real implementation dependency requires another selected finding to come first. A blocker exists only when one remediation needs an interface, invariant, or mechanism delivered by another task; thematic similarity is not a dependency. Every blocker must name an existing task directory or an earlier newly assigned task directory.

Assign priorities using the task contract rather than a blind severity conversion:

- **P0** — prerequisite, public-contract, correctness, memory-safety, lifetime, or backend-availability work that gates broad progress;
- **P1** — required normal-path or bounded defect remediation that does not broadly gate other work;
- **P2** — required finishing work such as independent cross-cutting verification or documentation. P2 is not optional.

Critical and high correctness or stability findings will normally be P0. A medium finding may also be P0 when it protects a public ownership or compatibility invariant. Preserve the review severity separately in the task metadata.

Build the complete assignment table before spawning any subagent. Each row must contain the finding ID, exact source heading, final directory, order, priority and reason, blockers, and any user filter constraint that affects its scope. This prevents parallel writers from making conflicting naming or dependency decisions.

### 5. Delegate one complete task per finding

Writing independent task specifications is mandatory subagent work. Launch one general-purpose writing subagent per selected finding in one parallel batch, up to the harness concurrency cap. If the selection exceeds that cap, use the fewest parallel waves needed. Give every subagent exclusive ownership of one destination `spec.md`; no two agents may edit the same file.

#### Same-model requirement

Every writer must use the exact model running the top-level skill invocation.

- Use the Task facility's inherited current-session model. In OMP, omit a specialist `agent` selection so the general-purpose task child inherits the parent model, and do not specify any different model or profile.
- If the spawn API exposes a `model` or `inherit-model` field, explicitly set it to the parent session's exact model identifier or exact inheritance mode.
- Do not use `sonic`, `scout`, a flash/small/cheap profile, `completion(..., model="smol")`, automatic model routing, or any specialist whose configured model may differ.
- If the API reports child model identity, verify it matches the parent before accepting output. If exact inheritance or pinning cannot be guaranteed, stop and report that blocker rather than silently allowing a downgrade.

Do not have the top-level agent pre-write drafts. Each subagent investigates and writes its final `spec.md` in one pass; this keeps finding-local repository context out of the parent context. Do not spawn a separate scout and writer for the same finding.

Use this assignment shape for every child:

```text
# Target
- Finding: <ID> — <exact review heading>
- Source: docs/changes/<target>/review.md, exact <ID> section
- Destination: docs/changes/<target>/<NN>-<ID>-<slug>/spec.md
- Exclusive ownership: destination spec only; do not modify implementation or other specs

# Change
- Read the complete <ID> finding section, the parent spec when relevant, and existing task specs needed for established contracts.
- Inspect only the referenced implementation, callers, tests, build targets, and nearest analogues needed to make this remediation executable without broad exploration.
- Use the assigned order, priority, and blockers exactly.
- Write the final self-contained spec at the destination using the required template and rules.
- Treat the review finding as the accepted root cause; refine stale paths or line references from current repository evidence without broadening its scope.
- Skip formatters, linters, builds, and tests; verification commands belong in the spec and project-wide validation happens only after all writers finish.

# Acceptance
- Exactly one final spec.md exists at the assigned destination.
- It traces to <ID>, contains concrete current paths and symbols, and gives a flash-class implementer all decisions needed to fix and verify the finding.
- It contains no implementation edits, unresolved design choices, placeholders, or requirements outside the finding.
```

Pass the assigned priority, blocker list, and filter constraint in the child task. Point the child to the review path and finding ID instead of copying the complete review into its prompt.

### 6. Write a self-contained remediation spec

Each child writes this structure, omitting only sections that truly do not apply:

```markdown
# <Action-oriented remediation title>

**Order:** <NN>
**Priority:** <P0|P1|P2> — <brief reason>
**Blocked by:** <task directory names, or “None”>
**Source:** `docs/changes/<target>/review.md` — `<FINDING-ID>`
**Review severity:** <critical|high|medium|low>
**Review verification:** <verified|strongly-supported|hypothesis>, confidence <0–100>

## Outcome

<One short paragraph describing the corrected, observable result.>

## Current failure

<The invariant, concrete failing path or scenario, and impact needed to understand why the change is required.>

## Scope

- <Precise behavior this task must change.>
- <Affected backends, interfaces, state transitions, errors, or compatibility rules.>

## Implementation references

- **Modify:** `path/to/file` — `<symbol or section>`; <why it is the touchpoint>.
- **Read:** `path/to/analogue` — `<symbol or section>`; <specific convention to reuse>.
- **Tests:** `path/to/test` — <existing suite, fixture, or nearest pattern>.

## Requirements

- <Normative, unambiguous implementation requirement.>
- <Relevant boundary, cleanup, ordering, failure, or compatibility behavior.>

## Non-goals

- <Nearby work explicitly excluded from this task.>

## Acceptance criteria

- [ ] <Observable criterion that distinguishes the remediation from the reviewed failure.>
- [ ] <Relevant negative, boundary, or repeated-operation criterion.>

## Verification

- `<specific focused build/test/static command>`
- <Required hardware, sanitizer, fault-injection, benchmark, or runtime scenario and its expected observation.>
```

Mini-spec writing rules:

- The spec must stand alone. The implementer must not need to read the full review, parent spec, conversation, or sibling specs to learn the finding's contract. The source link provides traceability, not omitted requirements.
- Translate the finding into an action-oriented task while retaining its exact invariant, failure mechanism, and affected scope. Do not merely paste the finding or say “apply the recommended fix.”
- Verify repository paths and important symbols against the current tree. Replace stale line numbers with current symbols or sections; do not ask the implementer to rediscover touchpoints.
- Read enough callers, counterparts, tests, configuration, and local analogues to state the smallest coherent fix. Reuse established repository mechanisms and exception/test conventions instead of inventing a second pattern.
- Resolve low-risk implementation details from repository evidence. Do not leave choices between alternative designs when one established pattern satisfies the finding.
- Where the review recommends alternatives, select the smallest design that fully restores the invariant. Record rejected nearby alternatives under non-goals only when doing so prevents likely scope drift.
- State inputs, outputs, shapes, sizes, ownership, ordering, cleanup, error categories, compatibility, and backend boundaries when they materially constrain the fix.
- Keep regression tests with the remediation. Tests must fail for the reviewed bug and defend observable behavior, not source text or incidental implementation.
- Convert the review's verification method into exact repository commands and scenarios. For accelerator work, require the repository's remote-development procedure and name the focused backend targets; do not substitute CPU-only evidence.
- A `hypothesis` finding must retain the measurement or falsification experiment and must not state the suspected effect as established fact.
- Include non-goals that prevent likely expansion beyond the finding. Do not introduce future-proofing, general frameworks, unrelated cleanup, or extra features.
- Do not use vague language such as “handle appropriately,” “update relevant files,” “as needed,” or “add tests.”
- Do not include placeholders, unresolved decisions, or a generic edge-case checklist.

### 7. Integrate and verify the generated task set

After all writers complete, the top-level agent reads every generated `spec.md` and performs a consistency pass. This is document verification, not an implementation build.

Confirm:

- every selected, non-duplicate explicit finding has exactly one generated task;
- no unselected finding, synthesis note, or residual risk became a task;
- numbering starts after the previous maximum, remains consecutive, and agrees with every `**Order:**` field;
- every directory includes its exact finding ID and a precise remediation slug;
- every `**Source:**` names the review path and correct finding ID;
- priorities and blockers match the precomputed assignment table, blockers point only backward, and there is no cycle;
- every spec preserves the source invariant and failure mode, names current implementation/test touchpoints, sets explicit non-goals, and includes observable acceptance criteria and focused verification;
- parallel writers did not duplicate mechanisms or assign the same implementation work to multiple findings. Keep distinct accepted root causes as distinct tasks; if two findings interact, clarify boundaries and blockers rather than silently merging them;
- no generated spec asks the implementer to perform broad exploration or choose unresolved alternatives.

If a spec fails this pass, send it back to its same-model owning subagent with concrete corrections. The subagent revises the final file; the parent should not replace it with a context-heavy rewrite unless subagent execution is unavailable. Re-read corrected files before completion.

## Completion response

Report only:

- the applied filter, or `all explicit findings`;
- the ordered new task list with finding ID, priority, directory path, and blockers;
- already-converted findings and their existing paths;
- any material ambiguity, collision, or model-pinning blocker.

Keep the response concise. The generated task specifications are the primary deliverable.

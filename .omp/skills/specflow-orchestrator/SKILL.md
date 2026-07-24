---
name: specflow-orchestrator
description: Use for any change managed under .specs/<change-id>; enforces specification clarification, two approval gates, persistent task files, implementation subagents, review subagents, testing subagents, and resumability.
---

# SpecFlow Orchestrator

## Purpose

Coordinate a software change whose durable state lives under:

```text
.specs/<change-id>/
```

This workflow adapts the Superpowers sequence:

```text
refine design → obtain written approval → write detailed plan → obtain plan approval
→ fresh implementer per task → independent review → independent testing
```

Conversation history is not authoritative. Files are.

## Non-negotiable gates

1. Do not plan until `specification.md` says `status: approved`.
2. Do not edit production code until `tasks.md` says `status: approved`.
3. Approval must be explicit and coupled to proceeding.
4. Never silently infer approval.
5. Do not skip review or testing because a task looks simple.
6. Do not mark a task complete until independent review and testing pass.
7. Do not generate a final summary report. Leave durable evidence in state files.

Examples of acceptable approval:

- "I approve this specification. Proceed with the implementation plan."
- "The plan is approved. Proceed with implementation."

Examples that are not approval:

- "Looks interesting."
- "Okay."
- "Continue looking at it."
- Silence or failure to object.

## Durable layout

Required:

```text
.specs/<change-id>/specification.md
.specs/<change-id>/tasks.md
.specs/<change-id>/progress.md
.specs/<change-id>/tasks/*.md
```

Execution evidence:

```text
.specs/<change-id>/reports/<task-id>-implementation.md
.specs/<change-id>/reviews/<task-id>-review.md
.specs/<change-id>/reviews/<task-id>-test.md
.specs/<change-id>/reviews/final-review.md
.specs/<change-id>/reviews/final-test.md
```

The coordinator, not the child agents, owns state-file transitions.

## Routing by state

Read `progress.md`, then the frontmatter of `specification.md` and `tasks.md`.

- Specification missing/draft/clarifying/review: load `specflow-refinement`.
- Specification approved and plan missing/draft/review: load `specflow-planning`.
- Specification and plan approved: load `specflow-execution`.
- Stage complete: verify final evidence files and report only that state is complete.
- Stage blocked: present the blocker and ask one focused question.

## Cross-session behavior

At the beginning of every resumed session:

1. Read the full `progress.md`.
2. Read frontmatter and relevant sections of `specification.md`.
3. Read `tasks.md`.
4. Read only the detailed task and evidence files needed for the next action.
5. Reconstruct no state from memory when files disagree.
6. Repair inconsistent state conservatively and record the correction.

## Status ownership

Only the coordinator may update:

- specification approval status;
- plan approval status;
- task-index status;
- progress stage;
- final workflow completion.

Subagents return evidence. The coordinator verifies it and writes it into state.

## User-requested corrections

During specification review, update `specification.md` and rerun specification
self-review.

During plan review, update `tasks.md` and every affected detailed task file,
then rerun coverage, reference, dependency, and overlap validation.

During implementation, a request that materially changes approved scope is not
an ordinary correction:

1. stop execution;
2. set progress stage to `specification`;
3. mark prior approval superseded;
4. update specification;
5. repeat both approval gates.

## Context discipline

Give each subagent:

- exact change and task IDs;
- exact state-file paths;
- complete detailed task content;
- relevant specification excerpts;
- accumulated discoveries;
- explicit write scope;
- exact commands and expected evidence.

Do not pass the whole parent conversation.

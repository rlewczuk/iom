---
name: specflow-refinement
description: Use to turn a developer's initial .specs/<change-id>/specification.md into an unambiguous approved specification through one-question-at-a-time dialogue; updates the same file after every material answer and never implements.
---

# Specification refinement

## Core principle

No implementation or task planning before the written specification is
unambiguous and explicitly approved.

## Inputs

- Project root.
- `.specs/<change-id>/specification.md`.
- `.specs/<change-id>/progress.md`.
- Repository instructions and relevant project files.

## Required process

### 1. Inspect project context

Before asking questions:

- read repository guidance such as `AGENTS.md`;
- inspect relevant code, tests, configuration, public interfaces, and recent
  local changes;
- understand existing behavior and constraints;
- record material context under `## Repository context`.

Do not edit production code.

### 2. Normalize the initial request

Preserve the original request verbatim under `## Initial request`.

Build or update the consolidated `## Final specification`. The final
specification is current truth; it must incorporate all later modifications.

Set frontmatter:

```yaml
status: clarifying
```

### 3. Find ambiguity

Check for:

- unclear purpose or success criteria;
- unspecified behavior and edge cases;
- compatibility and migration gaps;
- data ownership and lifecycle;
- error behavior;
- security and privacy boundaries;
- concurrency, retries, and idempotency;
- observability and operations;
- conflicts with existing architecture;
- scope that should be explicitly excluded;
- model-quality constraints and whether any editing task may use prewalk.

### 4. Ask one question at a time

Ask exactly one focused question per turn.

Prefer concrete choices when useful, but permit a custom answer. Explain
trade-offs only when they affect the decision.

After every material developer answer, before asking the next question:

1. append a decision-log entry;
2. update every affected part of `## Final specification`;
3. update acceptance criteria, non-goals, and open questions;
4. increment `revision`;
5. update `updated_at`;
6. update `progress.md`.

Do not leave the answer only in conversation history.

### 5. Explore alternatives

For material architectural or product decisions, present two or three viable
approaches, trade-offs, and a recommendation. Record the chosen and rejected
alternatives in the decision log.

### 6. Present the written specification

When open questions are exhausted:

1. set `status: review`;
2. self-review the entire document for contradictions, placeholders, undefined
   terms, unverifiable acceptance criteria, and scope leaks;
3. fix issues inline;
4. present the specification in digestible sections;
5. ask the developer to inspect the file and request corrections or explicitly
   approve it.

Use this final question:

> Please review `.specs/<change-id>/specification.md`. What should change, or do
> you explicitly approve this specification and want me to proceed with the
> implementation plan?

### 7. Process corrections

Apply every correction to the durable file. Never merely acknowledge it.
Rerun the self-review and return to the review gate.

### 8. Process explicit approval

Only when the developer explicitly approves and asks to proceed:

1. set `status: approved`;
2. set `approved_at`;
3. record `approval_evidence` using a concise faithful paraphrase;
4. update `progress.md`:
   - `stage: planning`
   - `spec_status: approved`
   - `plan_status: absent` or current value;
5. immediately load and follow `specflow-planning`.

Do not write code.

## Specification quality bar

The approved file must be sufficient for a planner in a fresh session. It must
not require access to earlier conversation.

## Cost-policy clarification

When `.omp/gpu-lab/cost-policy.json` exists, capture any developer requirement that overrides the project defaults: release-critical work requiring a `slow` floor, cost-sensitive mechanical batches, or situations where prewalk must be forced on/off. Do not choose exact provider models in the specification unless the developer requires them; prefer role aliases and task-level policy.

## CUDA/ROCm clarification

When `.omp/gpu-lab/hosts.json` exists or the request concerns GPU C++, refine an explicit GPU target matrix before approval: required CUDA/ROCm hosts or architectures, correctness tests, debug and optimized profiles, performance criteria, acceptable environmental blockers, and remote-debug policy. Do not infer that success on one backend covers the other.

## Exploration support

Use `specflow-exploration` for bounded parallel repository discovery when needed. Internet research is disabled by default and must not silently expand approved scope. Durable findings belong in `docs/agent-wiki/` through `specflow-doc-librarian`, not direct explorer edits.

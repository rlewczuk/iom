---
name: specflow-doc-librarian
description: Maintain a Karpathy-style project agent wiki under docs/agent-wiki from verified SpecFlow changes, without weakening existing implementation, review, test, or GPU gates.
---

# Project agent-wiki maintenance

## Layout

```text
docs/agent-wiki/
├── raw/       # immutable source/evidence notes
├── wiki/      # durable synthesized pages
├── index.md   # global contents/search map
└── log.md     # append-only update history
```

Initialize non-destructively:

```bash
.omp/gpu-lab/bin/gpu-lab-explore init-docs
```

## Workflow placement

The programming implementation agent remains the only production-source writer.
When a task has `docs_update: on`, or `auto` and produces durable project
knowledge, run `specflow-doc-librarian` serially after implementation and before
independent review. The reviewer and tester therefore inspect the final task
diff, including docs.

For a whole change, run a final docs synchronization before final whole-change
review. Never update docs after the final approval without rerunning affected
review and tests.

Skip the librarian when no durable knowledge changed. Record
`NO_DURABLE_CHANGE`; do not create filler pages.

## Sources

Use current source/tests, approved specification/tasks, actual diffs, and durable
reports. External research must be produced by the native `librarian` while the
session web gate is enabled, then recorded as cited raw evidence. The docs
librarian itself has no web access.

## Integrity

Raw evidence is append-only. Compiled wiki pages may be revised. Maintain
relative links, index coverage, and an append-only operation log. Code is the
authority; mark stale or contradictory wiki claims rather than preserving them
for narrative consistency.

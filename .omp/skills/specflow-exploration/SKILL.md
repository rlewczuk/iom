---
name: specflow-exploration
description: Use OMP-native scout/librarian subagents for bounded parallel codebase or optional internet exploration, with cost routing, session gates, and project agent-wiki reuse.
---

# Dedicated exploration

## Native OMP reuse

OMP already provides `scout` and `librarian` task agents. This package installs
a project-scoped `scout` override because project agents take precedence over
bundled agents with the same name. It preserves the native read-only exploration
role while making it offline-first and agent-wiki-aware. The native `librarian`
is retained unchanged for external library/API research.

## Session gates

```text
/explorers status
/explorers on
/explorers off
/explorer-web status
/explorer-web on
/explorer-web off
/explore-once <focused local question>
/explore-once --web <focused external question>
```

Internet research is disabled by default. `librarian` calls are blocked unless
internet is enabled or a one-shot web allowance is active. One-shot commands do
not change session defaults.

Persistent defaults live in `.omp/gpu-lab/exploration-policy.json` and are
managed with `.omp/gpu-lab/bin/gpu-lab-explore`. Session commands override only
the active OMP session/branch.

## Cost routing

Resolve every spawn explicitly:

```bash
.omp/gpu-lab/bin/gpu-lab-cost recommend \
  --role explorer --model-class smol --risk low \
  --situation read-only --prewalk off

.omp/gpu-lab/bin/gpu-lab-cost recommend \
  --role external_librarian --model-class smol --risk low \
  --situation read-only --prewalk off
```

The developer can override either role with `gpu-lab-cost role`. Exploration
agents are read-only and never use prewalk.

## Parallel fan-out

Use a single `task` batch with two to four independent questions, each assigned
to `scout` and the explicitly routed model. Distinguish search angles, such as:

- architecture/call path;
- tests and failure boundaries;
- build/toolchain and feature flags;
- CUDA/ROCm backend-specific assumptions.

Do not launch duplicate scouts with the same prompt. Consolidate their evidence
and resolve contradictions against current source. The configured maximum is a
budget ceiling, not a target.

## Caller rules

Planning, implementation, review, and verification agents may spawn explorers
only when missing context materially affects their assigned decision. Explorers
must not replace caller reasoning, implementation, independent review, or
verification. Never assign writes to `scout` or native `librarian`.

Search `docs/agent-wiki/` first, then verify against source. Flag durable new
facts for `specflow-doc-librarian` rather than editing docs from an explorer.

---
description: Execute an approved SpecFlow plan with implementation, review, and testing subagents. Usage: /specflow-execute <change-id>
---

Load and follow the `specflow-execution`, `specflow-cost-routing`, `specflow-exploration`, and `specflow-doc-librarian` skills for change `$1`.

Resume from `.specs/$1/progress.md` and `.specs/$1/tasks.md`. Refuse to
implement unless both specification and plan statuses are approved. Dispatch
fresh implementation subagents, followed by independent review and testing
subagents. Do not create a final summary report.

Resolve every spawned role through `.omp/gpu-lab/cost-policy.json`, pass the model explicitly, and persist `COST_ROUTING` evidence. Use the approved task's `prewalk_policy` and `prewalk_situation` to choose the standard or prewalk implementer; never prewalk read-only agents.

When approved tasks declare GPU targets, load `specflow-gpu-orchestrator`, use one source writer, and require durable CUDA/ROCm matrix evidence before completion.

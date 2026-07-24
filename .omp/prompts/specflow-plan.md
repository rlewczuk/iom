---
description: Create or correct a durable SpecFlow task plan. Usage: /specflow-plan <change-id>
---

Load and follow the `specflow-planning`, `specflow-cost-routing`, and `specflow-exploration` skills for change `$1`.

Use:

```text
.specs/$1/specification.md
.specs/$1/tasks.md
.specs/$1/tasks/
.specs/$1/progress.md
```

Do not edit production code. Refuse to plan unless the specification is
approved. Create a task index and one self-contained detailed Markdown file per
task, classify every task with `model_class`, `prewalk_policy`, and `prewalk_situation`, then stop at the explicit plan-review gate. The developer must be able to change those choices before approval.

For GPU-relevant scope, add host/profile fields to detailed tasks and plan the read-only multi-host validator wave through `specflow-gpu-orchestrator`.

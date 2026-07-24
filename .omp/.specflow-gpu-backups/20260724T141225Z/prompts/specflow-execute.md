---
description: Execute an approved SpecFlow plan with implementation, review, and testing subagents. Usage: /specflow-execute <change-id>
---

Load and follow the `specflow-execution` skill for change `$1`.

Resume from `.specs/$1/progress.md` and `.specs/$1/tasks.md`. Refuse to
implement unless both specification and plan statuses are approved. Dispatch
fresh implementation subagents, followed by independent review and testing
subagents. Do not create a final summary report.

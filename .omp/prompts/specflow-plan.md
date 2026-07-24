---
description: Create or correct a durable SpecFlow task plan. Usage: /specflow-plan <change-id>
---

Load and follow the `specflow-planning` skill for change `$1`.

Use:

```text
.specs/$1/specification.md
.specs/$1/tasks.md
.specs/$1/tasks/
.specs/$1/progress.md
```

Do not edit production code. Refuse to plan unless the specification is
approved. Create a task index and one self-contained detailed Markdown file per
task, then stop at the explicit plan-review gate.

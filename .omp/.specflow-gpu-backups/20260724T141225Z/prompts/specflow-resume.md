---
description: Route a SpecFlow change to its next durable workflow stage. Usage: /specflow-resume <change-id>
---

Load and follow the `specflow-orchestrator` skill for change `$1`.

Read `.specs/$1/progress.md`, specification frontmatter, plan frontmatter, and
the next relevant task/evidence files. Reconstruct state from files, not
conversation memory, and continue at the appropriate approval or execution
stage.

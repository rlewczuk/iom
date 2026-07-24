---
description: Route a SpecFlow change to its next durable workflow stage. Usage: /specflow-resume <change-id>
---

Load and follow the `specflow-orchestrator`, `specflow-cost-routing`, `specflow-exploration`, and `specflow-doc-librarian` skills for change `$1`.

Read `.specs/$1/progress.md`, specification frontmatter, plan frontmatter, and
the next relevant task/evidence files. Reconstruct state from files, not
conversation memory, and continue at the appropriate approval or execution
stage.

Before spawning, resolve the active cost profile, model fallback chain, and approved prewalk choice. If the next approved task declares GPU targets, also load `specflow-gpu-orchestrator` and resume from collected run IDs and artifacts.

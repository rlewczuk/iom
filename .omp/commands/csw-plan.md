---
description: Split .cswd/tasks/<task-name>[/subdirectory]/spec.md into typed tasks; optional --impl or --hld
---

Read `skill://csw-plan` with the read tool and follow it exactly.

Arguments (one task target and optional `--impl` or `--hld`): `$ARGUMENTS`

Explicit flags select leaf implementation or component design generation. Without a flag, a parent control type `hld` returned by `.omp/csw/bin/task_ctl get '<task-id>' --type` selects design mode; otherwise split logically and classify each child as `impl` or `hld` by flash-model complexity, as defined by the skill. All metadata is managed through `task_ctl`, never Markdown headers or direct `task.yml` edits.

---
description: Split .cswd/tasks/<task-name>[/subdirectory]/spec.md into typed tasks; optional --impl or --hld
---

Read `skill://csw-plan` with the read tool and follow it exactly.

Arguments (one task target and optional `--impl` or `--hld`): `$ARGUMENTS`

Explicit flags select leaf implementation or component design generation. Without a flag, a parent control type `hld` returned by `.omp/csw/bin/task_ctl get '<task-id>' --type` selects design mode; otherwise split logically and classify each child as `impl` or `hld` by flash-model complexity, as defined by the skill. All metadata is managed through `task_ctl`, never Markdown headers or direct `task.yml` edits.

**Mandatory decomposition policy:** strongly target 10–20 meaningful immediate children and no more than two levels below the original change root. Prefer direct `impl` leaves; Design mode does not require HLD children. Fewer than 10 require genuinely insufficient specification, never convenience or narrow grouping. Flatten one- or two-leaf HLD wrappers; every retained HLD must pass the skill's admission check, and deeper nesting requires explicit evidence that shallower alternatives cannot work. Never pad scope or weaken leaf readiness to meet a count.

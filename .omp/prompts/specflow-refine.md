---
description: Refine or resume a durable SpecFlow specification. Usage: /specflow-refine <change-id>
---

Load and follow the `specflow-refinement` and `specflow-exploration` skills for change `$1`.

Project-relative state directory:

```text
.specs/$1
```

Read `specification.md` and `progress.md`. Do not plan or implement. Inspect the
repository, ask exactly one ambiguity-resolving question at a time, and update
the durable specification after every material answer.

Also refine any cost-quality constraints: required model floor, whether editing prewalk is allowed, and tasks where it must be disabled.

When the change affects GPU C++, refine an explicit CUDA/ROCm target matrix and remote-debug policy before approval.

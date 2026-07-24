---
name: specflow-gpu-orchestrator
description: Coordinate approved SpecFlow tasks across configured CUDA and ROCm SSH hosts with one source writer, cost-routed parallel read-only validators, durable artifacts, bounded repair loops, and serial debugger escalation.
---
# Cost-aware multi-host GPU orchestration

Use after normal SpecFlow approval gates. `.omp/gpu-lab/hosts.json`,
`project.json`, and `cost-policy.json` define execution; `.specs/<change-id>`
remains authoritative workflow state.

Load `specflow-cost-routing` before creating the validator batch.

## Rules

- The coordinator or assigned implementer is the only source writer.
- Remote hosts are execution targets, never independent editable workspaces.
- Synchronize only through `.omp/gpu-lab/bin/gpu-labctl`; it creates per-run snapshots, host-local builds, GPU locks, and collected evidence.
- Use one shared run ID for every backend in a validation wave.
- Dispatch one CUDA/ROCm validator per enabled target plus one portability reviewer in a single `task` batch.
- Resolve each role from the active cost profile and pass its model fallback chain explicitly.
- Validators are read-only and always run with prewalk disabled.
- Cheap validator models execute deterministic commands and summarize structured output; they do not assert speculative root causes.
- Require every approved target unless explicitly optional.
- Bound repair cycles by `.omp/gpu-lab/project.json`; default is three.
- Do not treat environmental failures as code failures, but do not waive them silently.
- Debug one target at a time only after a stable failure is reproduced twice; debugger routing is strong-model, non-prewalk.

## Wave

1. Read the approved task's `gpu_targets`, profile, benchmark, debug policy, model floor, and prewalk decision.
2. Run local format/static/CPU checks.
3. Generate a collision-resistant shared run ID.
4. Resolve `cuda_validator` and/or `rocm_validator` routes plus `portability_reviewer`.
5. Batch-dispatch matching validator agents with explicit models and required `COST_ROUTING` output.
6. Read collected JSON/log evidence and persist concise evidence under `.specs`.
7. If any required target fails, dispatch a fresh implementation agent using the task's effective prewalk policy; then rerun review and the entire required target matrix.
8. On stable runtime failure, resolve role `debugger` and call `specflow-gpu-debugger` serially with `gpu-labctl debug-plan` output.
9. For performance requirements, run separate release/benchmark profiles and resolve role `performance_reviewer`.
10. Complete only when ordinary review/testing and the required GPU matrix pass.

OMP task concurrency is for validators/reviewers; hardware concurrency is also
controlled by remote `flock` locks. Do not spend a strong model on copying logs
or running known commands when a cheaper configured validator can produce the
same evidence.

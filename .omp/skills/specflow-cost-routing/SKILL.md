---
name: specflow-cost-routing
description: Resolve the lowest-cost adequate OMP model and optional edit-triggered prewalk variant for each SpecFlow subagent from the project cost policy, task risk, and developer overrides.
---

# Cost-aware subagent routing

Use this skill before every SpecFlow `task` spawn. The authoritative policy is:

```text
.omp/gpu-lab/cost-policy.json
```

Validate or inspect it with:

```bash
.omp/gpu-lab/bin/gpu-lab-cost validate
.omp/gpu-lab/bin/gpu-lab-cost show
```

## Required routing record

Before spawning, resolve and retain:

```text
profile
role
agent
model selector or fallback chain
thinking level
prewalk mode and target
reason for any escalation
```

Append the resolved thinking level as the OMP model-selector suffix (for example `@task:medium`) and pass the resulting selector/fallback chain explicitly as the task item `model`. Agent frontmatter and
`.omp/gpu-lab/omp-config.yml` are defensive defaults, not a substitute for
consulting the active policy.

Record the decision in the durable implementation/review/test evidence:

```text
COST_ROUTING:
- profile: ...
- role: ...
- agent: ...
- model: ...
- thinking: ...
- prewalk: on|off
- prewalk_target: ...|none
- rationale: ...
```

## Role map

- implementation without prewalk → `implementer` / `specflow-implementer`;
- implementation with prewalk → `implementer_prewalk` / `specflow-implementer-prewalk`;
- task review → `reviewer`;
- ordinary or final testing → `tester`;
- whole-change review → `final_reviewer`;
- CUDA execution → `cuda_validator`;
- ROCm execution → `rocm_validator`;
- CUDA/ROCm semantic review → `portability_reviewer`;
- optimized benchmark review → `performance_reviewer`;
- serial DAP diagnosis → `debugger`.

Use the policy's fallback chain exactly and apply the configured reasoning suffix to every selector. Do not replace a configured concrete provider/model with a guessed model name. The shell helper
`.omp/gpu-lab/bin/gpu-lab-cost recommend ...` returns the exact task-ready `model` value.

## Task metadata

Detailed task frontmatter supplies:

```yaml
model_class: smol|task|slow
prewalk_policy: auto|on|off
prewalk_situation: mechanical|repository-heavy-small-edit|normal|cross-cutting|high-risk|debugging
```

The developer may alter these values before plan approval. Explicit task
frontmatter is a deliberate per-task choice and takes precedence over an
automatic recommendation.

`model_class` is an adequacy floor:

- `smol`: routine commands, narrow mechanical checks, simple test execution;
- `task`: ordinary coding/review requiring repository reasoning;
- `slow`: architecture, security, migrations, difficult concurrency, numerical
  correctness, unstable compiler behavior, or debugger work.

When the role policy is weaker than the task's approved floor, escalate to the
corresponding `@task` or `@slow` role. Never downgrade an approved floor merely
to save tokens.

## Prewalk decision

OMP task prewalk starts an editing subagent on the resolved starting model and
hands off to the configured target at the first successful `edit` or `write`.
Use it only for an agent that is expected to edit.

Resolution order for implementation tasks:

1. `prewalk_policy: on` → use `specflow-implementer-prewalk`.
2. `prewalk_policy: off` → use `specflow-implementer`.
3. `prewalk_policy: auto` → consult the active profile's situation policy and
   developer overrides.
4. If the resolved mode remains `auto`, apply these conservative rules:
   - enable for `mechanical`;
   - enable for `repository-heavy-small-edit` when discovery is substantial but
     the approved edit is localized and routine;
   - disable for `cross-cutting`, `high-risk`, and `debugging`;
   - for `normal`, enable only when the implementation plan is stable before the
     first edit and the cheaper target is adequate for the remaining TDD loop.

Always disable prewalk when:

- the agent is read-only;
- no edit/write is expected;
- the task is primarily interactive debugging;
- the first edit is likely to reveal information that requires continued
  high-capability reasoning;
- the work is architecture-heavy, security-sensitive, migration-critical,
  numerically delicate, or broadly cross-cutting;
- the developer explicitly chose `off`.

The active prewalk target is synchronized into `.omp/gpu-lab/omp-config.yml` by `gpu-lab-cost apply`, profile/role changes, and the `omp-gpu` launcher.

Do not arm the parent session's `/prewalk` command on behalf of a task. This
package uses separate standard/prewalk implementer definitions so the choice is
bounded to one editing subagent. Validators, reviewers, testers, performance
agents, and debugger agents remain explicitly non-prewalk.

## Escalation

Start with the policy-selected model. Escalate one tier only after concrete
evidence such as:

- a model cannot understand the task or repository contract;
- repeated malformed tool use;
- review misses confirmed defects;
- a compiler/runtime failure needs nontrivial diagnosis;
- the task risk floor requires a stronger role.

Do not retry the same failed prompt on the same model unchanged. Add decisive
evidence, narrow the question, or escalate. After escalation, retain the new
model for that repair/review step only unless the policy is explicitly changed.

## Cheap-model boundaries

Cheap validator/tester models may execute deterministic commands and summarize
structured evidence. They must not invent root causes. Route semantic CUDA/ROCm
analysis to the portability reviewer and difficult failure diagnosis to the
reviewer/debugger roles.

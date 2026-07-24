---
name: specflow-gpu-testing
description: Execute and interpret deterministic CUDA/ROCm remote validation through gpu-labctl, preserving structured evidence in SpecFlow state.
---
# GPU testing

Load `specflow-cost-routing` before spawning validators. Use role `cuda_validator` or `rocm_validator`, pass the resolved model explicitly, require a `COST_ROUTING` block, and keep prewalk off. Cheap validators execute deterministic commands and summarize structured evidence; semantic diagnosis belongs to portability review or debugging.

Preflight:

```bash
.omp/gpu-lab/bin/gpu-labctl list
.omp/gpu-lab/bin/gpu-labctl doctor --all
```

Single target:

```bash
.omp/gpu-lab/bin/gpu-labctl validate HOST --run-id RUN --change CHANGE --task TASK --phase all
```

All enabled targets:

```bash
.omp/gpu-lab/bin/gpu-labctl validate-all --run-id RUN --change CHANGE --task TASK --phase all
```

The remote source snapshot, build tree, logs, and result JSON are immutable per run ID. Read the locally collected `result.json`; require expected phase statuses and zero exit codes. Record environment fingerprints and exact artifact paths.

Classify failures as code, backend-specific portability, test nondeterminism, host configuration, connectivity, lock timeout, or inconclusive. Reproduce runtime/test failures twice before debugger escalation. Use debug profiles for diagnosis and release/benchmark profiles for performance; CUDA `-G` builds are not benchmark evidence.

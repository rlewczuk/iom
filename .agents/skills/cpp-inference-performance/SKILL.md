---
name: cpp-inference-performance
description: Independently review C++ multi-backend inference performance using evidence-first analysis of synchronization, transfers, fallback, allocations, scheduling, overlap, memory, capture/compilation, launch count, redundant work, and material kernels, then emit direct remediation subtasks. Also serves as Area 5 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [workload/backend focus optional]"
---

# Performance Review

Performance is a correctness-like requirement for inference engines, but claims require workload relevance and evidence.

Read `.agents/cpp-review/references/review-process.md`, `.agents/cpp-review/references/finding-rubric.md`, `.agents/cpp-review/references/performance.md`, and applicable backend checklists.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope and return only `PF-###` candidate packets. Do not write task files before cross-area synthesis.
- **Standalone:** resolve scope and optional destination, perform this performance pass, then invoke `cpp-inference-review-synthesis` for adversarial verification and direct task output.

In selected-commit mode, accept only regressions introduced or materially exposed/worsened by the target.

## Evidence classes

Classify every candidate:

1. **Measured regression** — controlled benchmark/profile data shows material degradation.
2. **Mechanically clear critical-path regression** — code proves an unconditional relevant synchronization, transfer, fallback, allocation, recompilation, or work increase.
3. **Hypothesis** — plausible mechanism without enough evidence; the task must be the exact measurement/falsification experiment and must not state the suspected effect as fact.

Reject claims such as “virtual calls are slow” or “this loop may be expensive” without material workload evidence.

## Review system behavior before kernels

### Timeline/system pass

Inspect:

- host/device and device/device copies;
- CPU or alternate-backend fallback boundaries;
- layout/conversion kernels;
- device-wide, stream, queue, and host synchronization;
- lost overlap or unintended serialization;
- allocator churn and synchronization-inducing allocation APIs;
- module/kernel/pipeline recompilation;
- graph capture invalidation;
- launch fragmentation and lost fusion;
- collective/communication overhead and load imbalance;
- persistent/scratch/peak memory growth;
- cache invalidation/rebuild and capability queries on hot paths.

Separate model load/compile, prompt/prefill, token/decode, batch throughput, and multi-device scaling. One-time setup cost is not per-token cost.

### Kernel pass

Only after proving a kernel is material, inspect bandwidth/coalescing, arithmetic intensity, register spills, shared/local memory, divergence, geometry, cache behavior, redundant computation, and precision conversions. Do not optimize isolated counters.

## Benchmark discipline

Use representative shapes/workloads, warmups, repeated samples, variance, controlled hardware/software/power state, recorded backend/device identity, comparable profiler instrumentation, and explicit latency/throughput/memory thresholds.

A semantically correct fallback can be a severe performance defect when it adds critical-path transfers, device changes, conversions, or prevents fusion/partitioning.

## Mandatory simplicity pass

Prefer removing work over managing it:

- delete redundant transfers, conversions, synchronization, validation, or recomputation;
- hoist invariant work out of hot paths;
- reuse existing allocation/capture/cache mechanisms before creating another cache;
- challenge caches whose invalidation/state cost exceeds recomputation;
- reject schedulers, batching frameworks, tuning layers, or configuration knobs without present measured need;
- consolidate duplicated backend launch/setup code only when semantics and generated work remain equivalent;
- remove instrumentation or bookkeeping that executes unconditionally without a current consumer.

Do not introduce a framework to solve a local hotspot. The smallest remediation should remove the proven mechanism.

## Cross-area ownership

If a lifetime/order defect belongs to stability, keep one stability root cause and attach performance as secondary impact. If duplicated dispatch or conversion policy is the cause, let architecture own it. Do not produce parallel tasks for the same remediation.

## Candidate acceptance

Use IDs `PF-###` and the full packet. Include workload/critical path, baseline and target evidence when measured, exact mechanism, expected metric, current symbols, remediation, and falsifier.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
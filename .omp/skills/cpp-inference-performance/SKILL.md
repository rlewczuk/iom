---
name: cpp-inference-performance
description: Independently review C++ multi-backend inference performance using evidence-first analysis of synchronization, transfers, fallback, allocations, scheduling, overlap, memory, capture/compilation, launch count, redundant work, and material kernels, then emit direct remediation subtasks. Also serves as Area 5 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [workload/backend focus optional]"
---

# Performance Review

Performance is a correctness-like requirement for inference engines, but claims require workload relevance and evidence.

Read `skill://boss` first. Then read `.omp/cpp-review/references/review-process.md`, `.omp/cpp-review/references/finding-rubric.md`, `.omp/cpp-review/references/performance.md`, and applicable backend checklists. Follow the common routing, evidence-packet, synthesis, and verification contract in `review-process.md`.

## Boss routing contract

The canonical routing, evidence, scope, synthesis, and verification contract is `.omp/cpp-review/references/review-process.md`; this skill adds only performance-specific routing and evidence jobs. The visible/root session should run as `@slow`, but a different running model may continue after Boss displays the mismatch warning and the user explicitly consents; a skill cannot switch an already-running model. The root owns scope, routing, acceptance, the assignment table, and verification.

One root supervisor schedules one shared `scout (project read-only) @smol`, atomic `boss-errand @smol` follow-ups, and bounded read-only `boss-reviewer @task` analysis/falsification. Area leaves do not delegate, invoke orchestration or synthesis, write tasks, or run gates. Dispatch independent work in one batch, use exact known ranges when cheaper than another dispatch, and do not make gratuitous calls.

### Invocation modes

- **Orchestrated:** use the supplied scope/map as a `boss-reviewer @task` leaf and return only `PF-###` candidate packets. Request factual follow-ups through the root; do not delegate, synthesize, or materialize tasks.
- **Standalone:** the root invocation, under Boss's root-model consent policy, owns the complete area, resolves scope, performs this pass, and always invokes `cpp-inference-review-synthesis` for adversarial verification and direct task output.

Workers may propose exact benchmark, profiler, trace, or runtime commands/scenarios, but only the root runs gates after collection and records actual results and gaps. If a lane is unavailable, report routing/coverage limits and request permission before a materially costlier fallback; never silently replace cheap profiles or run an unbounded frontier review.

## Task metadata

When a destination is supplied, task lifecycle controls belong exclusively to task_ctl-managed `task.yml`: generated remediation records use `type: impl`, `status: new`, assigned `order`, P0–P2 `priority`, canonical `blocked-by` IDs, and the parent `spec.md` as `source`. Use `.omp/csw/bin/task_ctl` CLI/API (`task_dir`, `get_task`, `set_task`, `list_tasks`) for paths, ordering, metadata, and dependencies; never parse or hand-write YAML. Keep all review evidence in `spec.md`.

## Performance evidence classes

Classify every candidate:

1. **Measured regression** — controlled benchmark/profile data shows material degradation on a representative workload.
2. **Mechanically clear critical-path regression** — code proves an unconditional relevant synchronization, transfer, fallback, allocation, recompilation, or work increase.
3. **Hypothesis** — plausible mechanism without enough evidence; the task must be the exact measurement/falsification experiment and must not state the suspected effect as fact.

Reject claims such as “virtual calls are slow” or “this loop may be expensive” without material workload evidence. Do not turn an uninspected path, missing profile, or model agreement into verification or confidence.

## Concrete evidence jobs

### Cheap `@smol` jobs

- Inventory the selected path's model-load/compile, prompt/prefill, token/decode, batch-throughput, and multi-device phases; map host/device transfers, synchronization, fallback, allocation, launch, capture, and cache boundaries to exact symbols.
- For one suspected hotspot, trace callers, guards, backend counterparts, stream/queue and device transitions, allocator/cache invalidation, and relevant negative evidence; identify whether the mechanism is unconditional and critical-path.
- Locate representative benchmark/profiler harnesses and report workload shapes, warmups, samples/variance, hardware/software/power identity, instrumentation comparability, metrics, and thresholds; propose commands only.
- For a selected commit, compare baseline and target provenance at the smallest decisive ranges and classify introduced versus pre-existing work; do not ingest an entire diff when focused ranges suffice.

### Advisor hard-decision triggers

Use `boss-advisor @advisor` only for a genuinely hard decision: whether evidence establishes a material regression versus a hypothesis; whether a synchronization, transfer, fallback, allocation, cache, or launch mechanism is actually on a relevant critical path; disputed workload/baseline/threshold interpretation; ownership between performance, stability, and architecture; or a high/critical remediation choice such as deleting versus retaining a cache/scheduler/fusion layer.

The advisor is tool-free and reasons only over a compact packet containing the needed source facts, exact minimal excerpts, measurements or proposed falsifier, negative evidence, coverage, validation, and alternatives. It must never open a URI/path, search files, run commands, delegate, or supply missing repository facts. If facts are missing, it returns `NEED EVIDENCE` with one exact question; the root sends a cheap source-reading worker and returns only the evidence delta. Advisor agreement does not verify a claim and the supervisor remains accountable.

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

Use representative shapes/workloads, warmups, repeated samples, variance, controlled hardware/software/power state, recorded backend/device identity, comparable profiler instrumentation, and explicit latency/throughput/memory thresholds. Workers propose the exact command and evidence requirements; the root runs gates after collection and records actual results and gaps.

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

Use IDs `PF-###` and the full packet in `finding-rubric.md`. Include workload/critical path, baseline and target evidence when measured, exact mechanism, expected metric, current symbols, remediation, and falsifier. Preserve the measured/mechanically-clear/hypothesis distinction in `Verification`, `Confidence`, `Evidence`, and `Verification method`.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
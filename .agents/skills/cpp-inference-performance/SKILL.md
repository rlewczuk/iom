---
name: cpp-inference-performance
description: Review performance of C++ multi-GPU inference engines with evidence-first analysis of synchronization, transfers, fallback, allocations, scheduling, overlap, memory footprint, graph capture/compilation, launch count, and material kernel bottlenecks. Use independently or as Area 5 of cpp-inference-code-review.
argument-hint: "[scope] [workload/backend optional]"
---

# Performance Review

Performance is a correctness-like requirement for inference engines, but claims must be disciplined.

Read `.agents/cpp-review/references/performance.md`, `.agents/cpp-review/references/finding-rubric.md`, and relevant backend checklists.

## Evidence classes

Classify every performance candidate as one of:

1. **Measured regression** — benchmark/profile data demonstrates material degradation.
2. **Mechanically clear critical-path regression** — e.g. a newly unconditional device-wide synchronization or host round-trip in a per-token path. Benchmarking should quantify it, but the mechanism is certain enough to report.
3. **Performance hypothesis** — plausible mechanism without sufficient evidence. Never state it as an established regression; include an exact verification experiment.

Do not report subjective claims such as “virtual calls are slow” or “this loop may be expensive” without workload relevance and evidence.

## Review order: system first, kernel second

### System/timeline pass

Before micro-optimizing kernels, inspect whether the change causes:

- new host/device/device-device copies;
- CPU fallback or larger fallback regions;
- layout/conversion kernels;
- device-wide/stream synchronization;
- reduced overlap between independent work;
- allocator churn or synchronization-inducing allocation APIs;
- kernel/module/pipeline recompilation;
- graph-capture invalidation/loss;
- many tiny launches replacing fused/coarser work;
- serialized queues/streams/devices;
- communication/collective overhead;
- load imbalance;
- larger persistent/scratch memory footprint;
- cache invalidation/rebuild on hot paths.

For autoregressive inference, explicitly consider prompt processing and token generation separately; a one-time setup cost and a per-token cost have very different impact.

### Kernel pass

Only after establishing a material kernel matters, inspect:

- memory bandwidth and coalescing;
- arithmetic intensity/roofline position;
- occupancy only insofar as it limits throughput;
- register pressure/spills;
- shared/local memory;
- warp/wave divergence;
- launch/workgroup geometry;
- cache behavior;
- redundant computation;
- precision/conversion overhead.

Avoid optimizing metrics in isolation.

## Benchmark discipline

Prefer:

- representative model/operator shapes;
- warmups;
- repeated samples;
- variance/statistical context;
- same hardware/software/power settings;
- backend identity recorded;
- machine-readable baseline/current results;
- separate latency/throughput/memory dimensions;
- configured regression thresholds rather than eyeballing noise.

Do not compare results collected under materially different profiler instrumentation as if they were normal execution. Some kernel profilers serialize or perturb execution.

## Fallback rule

A semantically correct fallback can still be a severe performance finding when it introduces critical-path host transfers, device changes, conversion boundaries, or prevents graph fusion/partitioning.

## Cross-area deduplication

If a synchronization/lifetime defect already belongs to stability, avoid duplicating it. Attach performance consequence as secondary evidence and let synthesis choose one root-cause finding.

Use IDs `PF-###`.

# Backend Architecture & Simplicity Reference

## Desired boundary

The common layer expresses semantic requirements and backend-neutral scheduling decisions. Backends own backend-specific representation, compilation, execution, synchronization primitives, and device capabilities.

Representative architectures to study selectively:

- ggml/llama.cpp backend interfaces for buffers, async copies, events, support predicates, and graph execution;
- ONNX Runtime Execution Providers for capability discovery and graph partitioning/fallback;
- OpenVINO plugins/infer requests for a common inference API over device-specific compiled models and async execution.

External designs are evidence, not templates. Reuse the reviewed repository's established mechanisms before importing another architecture.

## Simplicity dimensions

A change is structurally simpler when it reduces:

- independently mutable states and invalid combinations;
- duplicated sources of truth;
- special-case branches;
- representations and conversions;
- ownership/lifetime modes;
- backend-specific knowledge in common code;
- wrappers/layers without responsibility;
- duplicated validation, dispatch, fallback, cleanup, build, or test policy;
- caches, registries, configuration, or extension points without current value;
- facts a maintainer must synchronize mentally and in code.

Line count is a weak proxy. Net concept and responsibility count are stronger.

## Responsibility test

An abstraction is justified when it owns a stable semantic responsibility or invariant. It is suspect when it only:

- forwards or renames parameters;
- wraps one implementation;
- hides synchronization or performance differences;
- generalizes a one-off case;
- creates configuration for a fixed current choice;
- extracts repeated syntax without consolidating the underlying decision.

Ask what becomes impossible or single-sourced because the abstraction exists. If the answer is “nothing,” deletion is the default candidate.

## Duplication test

Inventory decisions, not just identical text. Two code paths are duplicates when they independently determine the same semantic fact even if their syntax differs. High-risk examples:

- support constraints repeated in capability, dispatch, and kernel setup;
- equivalent lifetime/release protocols implemented independently;
- layout/shape rules copied across backends;
- backend matrices repeated in build and test registration;
- state derived once but also cached as mutable flags.

A valid consolidation task names the canonical remaining owner and proves real backend differences remain expressible.

## Capability contract

Capability/support predicates must stay consistent with operator semantics, dtype/layout/alignment limits, device features, backend implementation, graph partitioning, fallback, tests, and benchmarks. Prefer one typed/queryable source with a clear lifetime/cache key.

## Deletion proof

Before recommending removal or consolidation, establish:

1. all current callers/registrations/dynamic entry points;
2. the invariant the code claims to protect;
3. the existing mechanism that already protects it, or why the invariant is obsolete;
4. behavior/error/lifetime/performance equivalence after deletion;
5. focused verification that would catch a mistaken deletion.

Do not preserve dead weight solely for hypothetical future backends.
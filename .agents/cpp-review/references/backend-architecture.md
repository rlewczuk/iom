# Backend Architecture & Simplicity Reference

## Desired boundary

The common layer should express semantic requirements and backend-neutral scheduling decisions. Backends should own backend-specific representation, compilation, execution, synchronization primitives, and device capabilities.

Representative architectures to study:

- ggml/llama.cpp backend interface: buffers, async copies, events, support predicates, graph execution;
- ONNX Runtime Execution Providers: provider capability discovery and graph partitioning/fallback;
- OpenVINO plugins/infer requests: common inference API with device-specific compiled models and asynchronous execution pipelines.

## Simplicity dimensions

A change is structurally simpler when it reduces one or more of:

- independently mutable states;
- duplicated sources of truth;
- special-case branches;
- representations/conversions;
- ownership modes;
- backend-specific knowledge in common code;
- wrapper layers with no responsibility;
- invalid state combinations;
- duplicated validation/dispatch policy.

Line count is only a weak proxy.

## Good abstraction test

An abstraction is justified when it owns a stable semantic responsibility or invariant. It is suspect when it only forwards parameters, hides important synchronization/performance differences, or generalizes a one-off case without removing duplicated responsibility.

## Capability contract

Capability/support predicates are executable architectural contracts. They must stay consistent with:

- operator semantics;
- dtype/layout/alignment limitations;
- device generation/features;
- backend implementation;
- graph partitioning;
- fallback behavior;
- tests and benchmarks.

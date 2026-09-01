---
name: cpp-inference-backend-simplicity
description: Review multi-backend C++ inference-engine architecture for clean common/backend boundaries, capability modeling, duplicated dispatch/state, abstraction altitude, special cases, maintainability, and behavior-preserving simplification. Use independently or as Area 3 of cpp-inference-code-review.
argument-hint: "[scope] [backend(s) optional]"
---

# Backend Architecture & Simplicity Review

Optimize **concept count and sources of truth**, not line count. Do not simplify away semantics needed for correctness, debuggability, backend differences, or measured performance.

Read `.agents/cpp-review/references/backend-architecture.md`, `.agents/cpp-review/references/finding-rubric.md`, and `.agents/cpp-review/checklists/common.md`.

## Core architectural rule

Prefer:

```text
common semantic operation
→ backend-neutral capability/policy decision
→ backend interface
→ backend implementation
```

over scattered concrete-backend branching in common graph/tensor/runtime logic.

However, do **not** demand a lowest-common-denominator abstraction. Expose backend differences when they materially affect synchronization, memory placement, compilation, execution, or performance.

A useful rule is:

> Unify concepts with the same semantic contract; expose differences that change correctness or performance.

## Structural review

### Backend leakage

Flag concrete backend knowledge in common code when it indicates a missing capability, policy, or backend interface. Look for repeated `if CUDA / else HIP / else Vulkan` logic that evolves independently.

Do not flag a deliberate backend check whose semantics genuinely cannot be expressed by the existing common contract.

### Capability modeling

Prefer canonical typed/queryable capabilities over:

- duplicated boolean flags;
- magic device-name checks;
- scattered version checks;
- operator-specific ad-hoc backend tests;
- state inferred differently at different call sites.

Each capability should have one source of truth and a clear lifetime/cache key.

### State complexity

Search for:

- redundant derived state;
- multiple representations of the same condition;
- flags whose combinations create invalid states;
- caches with unclear invalidation;
- temporary compatibility paths becoming permanent architecture;
- backend-specific state stored in generic tensor/graph objects without a clear semantic need.

Ask: **How many additional facts must a maintainer keep simultaneously in mind after this change?**

### Abstraction altitude

Prefer fixing the lowest stable abstraction that owns the invariant. Flag fixes that add a new special case at every caller when one canonical capability/interface/representation change would remove the condition.

Also reject over-generalization that introduces indirection without eliminating duplicated responsibility.

### Reuse and duplication

Look for:

- duplicate implementations that differ only superficially;
- existing utilities/canonical helpers bypassed;
- repeated shape/layout validation;
- repeated transfer/fallback policy;
- wrapper layers that only forward arguments and do not own a semantic responsibility.

### Simplicity safety gate

A simplification recommendation is publishable only if it:

1. removes a concept/state/branch/source of truth or makes an invariant explicit;
2. preserves required behavior and backend-specific semantics;
3. does not introduce extra synchronization/copies/allocations on a critical path, unless measured and acceptable;
4. keeps debugging and failure attribution at least as good;
5. can be described as a concrete structural change, not subjective preference.

Use IDs `AR-###`.

---
name: cpp-inference-contract-correctness
description: Independently review a C++ inference-engine scope for specification compliance, semantic/operator/tensor contracts, capability and fallback correctness, error behavior, compatibility, and duplicated contract enforcement; finish with direct remediation subtasks. Also serves as Area 1 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [operator/backend focus optional]"
---

# Contract & Correctness Review

Review **what the system promises and whether the implementation preserves it**. Ignore style unless it obscures a semantic defect or creates duplicated sources of truth.

Read `.agents/cpp-review/references/review-process.md`, `.agents/cpp-review/references/finding-rubric.md`, and `.agents/cpp-review/checklists/common.md`.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope/specification map; return only `CC-###` candidate packets to the orchestrator. Do not write task files.
- **Standalone:** resolve scope and optional `docs/changes/...` destination using the shared process, perform this area only, then invoke `cpp-inference-review-synthesis`. The final deliverable is direct remediation subtasks, not a review report.

In selected-commit mode, inspect necessary surrounding code and backend counterparts but accept only defects introduced or materially exposed/worsened by the target commit.

## Establish the contract map

Identify:

- authoritative specification/design requirements;
- public and internal interfaces plus every affected caller;
- operator/graph transformation semantics;
- tensor representation and ownership rules;
- backend capability, dispatch, partitioning, and fallback;
- established error categories and compatibility guarantees;
- relevant behavioral tests and unsupported-case tests.

Separate required observable behavior from suggested implementation structure.

## Review questions

### Specification and observable behavior

- Are all mandatory requirements implemented?
- Do defaults, errors, feature gates, and compatibility behavior match the contract?
- Did an implementation detail accidentally become observable?
- Does validation happen before work or partial state mutation?
- Is unsupported input rejected or routed intentionally?

### Operator and tensor semantics

Check relevant assumptions about:

- shapes, ranks, zero/empty and dynamic dimensions;
- dtype and quantization metadata combinations;
- strides, non-contiguity, layout, tiling, and alignment;
- broadcasting and alias/in-place behavior;
- device placement and identity;
- output ownership and mutation;
- arithmetic overflow before allocation, indexing, or narrowing.

### Capability, dispatch, and fallback

Treat this as one contract:

```text
semantic operation
→ capability/support predicate
→ dispatch/partitioning
→ backend implementation constraints
→ fallback/error behavior
→ tests and benchmarks
```

Flag support predicates that overclaim implementation constraints or reject supported paths. Trace hidden fallback, representation conversions, and changed graph partitioning.

### Compatibility

Inspect effects on:

- other accelerators and CPU/reference paths;
- builds with each optional backend enabled or disabled;
- serialized models/caches/configurations when relevant;
- promised public ABI/API behavior.

## Mandatory simplicity pass

Contract logic is especially vulnerable to duplication. Look for:

- the same shape/dtype/layout condition enforced differently at multiple layers;
- capability facts duplicated between common dispatch and backend implementation;
- repeated fallback/error policy with divergent categories or messages;
- stored derived flags that can disagree with canonical tensor/device state;
- new caller-side special cases that belong in the lowest stable owner;
- parallel helpers that duplicate an established validation or capability mechanism.

Prefer one canonical contract owner. Do not centralize backend-specific constraints that materially differ.

## Candidate acceptance

Use IDs `CC-###` and the complete candidate packet contract. Include exact scope relation, current symbols/callers/tests, smallest structural remediation, acceptance seed, non-goals, and falsifier.

Do not emit a generic “missing test” candidate without an unprotected observable behavior. Do not turn multiple symptoms of one contract source-of-truth defect into separate candidates.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
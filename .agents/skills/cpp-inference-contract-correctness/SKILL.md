---
name: cpp-inference-contract-correctness
description: Review C++ inference-engine changes or codebases for specification compliance, semantic correctness, operator and tensor contracts, backend capability/fallback correctness, error behavior, and compatibility. Use independently or as Area 1 of cpp-inference-code-review.
argument-hint: "[scope] [spec path optional]"
---

# Contract & Correctness Review

Review **what the system promises and whether the implementation actually preserves it**. Do not spend review budget on style unless it obscures a semantic defect.

Read `.agents/cpp-review/references/review-process.md`, `.agents/cpp-review/references/finding-rubric.md`, and `.agents/cpp-review/checklists/common.md`.

## Inputs

Establish before reviewing:

- exact review scope (whole codebase or one resolved commit);
- authoritative specification/design documents, if supplied;
- changed/public interfaces and call sites;
- affected operators/graph transformations;
- backend capability and fallback paths;
- relevant existing tests.

For selected-commit mode, inspect surrounding code and counterpart backends as needed, but only report defects introduced or materially exposed by the selected commit.

## Review questions

### Specification and observable behavior

- What externally observable behavior is required?
- Which requirements are mandatory, optional, or explicitly out of scope?
- Does the implementation satisfy the requirement rather than merely resemble the proposed structure?
- Did implementation details accidentally become user-visible behavior?
- Are defaults, compatibility behavior, failure modes, and feature gating correct?

### Operator/tensor semantics

Check all affected assumptions about:

- shapes and ranks;
- dtype combinations;
- strides/non-contiguous inputs;
- layout and alignment;
- broadcasting;
- aliasing/in-place behavior;
- zero/empty dimensions;
- dynamic dimensions;
- quantization metadata;
- device placement;
- output ownership and mutation.

### Capability declaration and dispatch

Treat these as a coupled unit:

```text
semantic operation
→ capability/support predicate
→ dispatch/partitioning
→ backend implementation constraints
→ fallback/error behavior
→ tests
→ benchmark coverage
```

Flag capability predicates that claim support beyond implementation constraints or reject newly supported paths. Trace whether incorrect support changes graph partitioning or creates hidden CPU/device fallback.

### Fallback and error behavior

- Is unsupported input rejected or routed intentionally?
- Can a supposedly supported path fail only at execution time?
- Does fallback preserve semantics and tensor representation?
- Is failure explicit rather than silent partial execution?
- Are errors attributed to the correct operation/backend?

### Compatibility

Inspect likely effects on:

- other GPU backends;
- CPU/reference backend;
- builds with individual backends disabled;
- old serialized models/caches/configurations when relevant;
- public ABI/API if the project promises stability.

## Candidate finding format

Use IDs `CC-###` and the common finding schema. Include `Scope relation: introduced by selected commit | materially exposed by selected commit | whole-codebase`.

If no material defect survives verification, return `No material findings.`

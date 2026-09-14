---
name: cpp-inference-backend-simplicity
description: Independently find redundant, duplicated, overengineered, special-cased, or poorly layered code in multi-backend C++ inference engines and emit behavior-preserving remediation subtasks. Reviews common/backend boundaries, capabilities, state, dispatch, abstractions, and dead machinery. Also serves as Area 3 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [backend/subsystem focus optional]"
---

# Backend Architecture & Simplicity Review

Optimize **concept count, responsibility count, and sources of truth**, not line count. The strongest result often deletes code, state, branches, or an abstraction.

Do not simplify away correctness, debuggability, real backend differences, asynchronous lifetime, or measured performance.

Before any repository work, read `skill://boss` first. Then read `.omp/cpp-review/references/review-process.md`, `.omp/cpp-review/references/backend-architecture.md`, `.omp/cpp-review/references/finding-rubric.md`, and `.omp/cpp-review/checklists/common.md`.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope and return only `AR-###` candidate packets. This is candidate-only `boss-reviewer` execution at `@task`: do not resolve a new frontier, delegate recursively, invoke synthesis, write task files, or run validation gates.
- **Standalone:** run as the root orchestration for this area after applying Boss's root-model warning-and-consent gate. Resolve scope and the optional task destination using the shared process, own the complete area review and acceptance, then invoke `cpp-inference-review-synthesis` to cross-check and directly materialize tasks. The skill cannot switch an already-running model; `@slow` remains preferred, while an explicitly accepted mismatch may continue.

A request such as “find redundant code,” “remove overengineering,” or “simplify backend architecture” is a valid standalone invocation. In selected-commit mode, only accept complexity introduced or materially worsened/exposed by that commit.

## Boss routing for this area

Follow the canonical review process rather than restating it. The standalone invocation is owned by the running root under Boss's root-model consent policy, which resolves scope, accepts candidates, freezes the assignment table, and invokes synthesis; the orchestrated invocation is a candidate-only `boss-reviewer` leaf at `@task`. Neither mode may claim an already-running model was switched, and the leaf may not delegate, recurse, write tasks, or broaden discovery.

- Project agent `scout` at `@smol` handles broad concept/state inventory, repeated-decision search, caller/counterpart collection, wrapper/factory/registry tracing, dynamic-entry-point checks, and dead-machinery discovery. Use `boss-errand` at `@smol` only for atomic factual followups. Return exact `path:line`/symbol evidence, source facts versus inference, negative evidence, search coverage, and uninspected areas.
- `boss-reviewer` may inspect the bounded assigned source ranges and callpaths directly, alongside the scout evidence, to perform simplification reasoning and independent falsification; it returns candidate packets only.
- For a genuinely hard or disputed responsibility boundary, backend-semantic distinction, source-of-truth choice, or deletion/consolidation risk, the root may send compact packet **content** to `boss-advisor` at `@advisor`. The advisor is tool-free, packet-only, never delegates, and may answer `NEED EVIDENCE` with one exact question; agreement never verifies a claim.
- The root reads only cheap exact-known ranges when cheaper than another dispatch, not broad scans or whole-diff ingestion. Batch independent work and make no gratuitous calls.
- Workers skip builds, tests, benchmarks, formatters, and other validation; they propose exact gates. The root executes gates and mandatory synthesis. If delegation is unavailable, disclose routing/coverage limits and request permission before any materially costlier fallback.

### Cheap evidence assignments

Ask the `@smol` scout to inventory owners, states/transitions, representations/conversions, capability and policy sources, dispatch/fallback branches, validation/error paths, caches/registries/factories/adapters/wrappers, common/backend implementations, dynamic entry points, and backend counterparts; search repeated support, placement, allocation, conversion, error, registration, and version decisions and report canonical versus duplicated sources. Ask the `@task` reviewer to inspect only the assigned bounded ranges/callpaths and packet, identifying objective redundancy, invalid state combinations, backend leakage, special-case accumulation, weightless layers, dead machinery, or accidental work while preserving semantics, diagnostics, async lifetime, and performance. Escalate to the advisor only for a hard ownership boundary, disputed real backend distinction, or high-risk deletion/consolidation choice after evidence is complete; otherwise the root decides.

## Architectural rule

Prefer:

```text
common semantic operation
→ backend-neutral capability/policy decision
→ backend interface
→ backend implementation
```

over scattered concrete-backend branches in common tensor/model/runtime logic.

Do not force a lowest-common-denominator interface. Expose backend differences when they change representation, synchronization, memory placement, compilation, execution, failure, or performance.

> Unify responsibilities with the same semantic contract; expose differences that change correctness or performance.

## Review method

### 1. Inventory concepts before code

Map the subsystem's:

- owners and non-owning views;
- states and transitions;
- representations and conversions;
- capability/policy sources;
- dispatch and fallback decisions;
- validation and error paths;
- caches, registries, factories, adapters, and wrappers;
- common and backend-specific implementations.

Ask which entries protect current invariants and which merely duplicate another entry.

### 2. Trace repeated decisions

Search semantically, not only textually, for the same fact decided in multiple places:

- shape/dtype/layout support;
- backend selection and fallback;
- queue/device identity;
- allocation and release policy;
- conversion/staging decisions;
- error translation;
- test/backend registration;
- feature/version capability.

Different syntax can still encode duplicated responsibility.

### 3. Challenge every layer and state variable

For each abstraction, wrapper, cache, flag, or registry ask:

1. Which stable invariant does it own?
2. What invalid state or duplicated decision disappears because it exists?
3. Could an existing type/factory/queue/device interface own this responsibility?
4. Is the abstraction used by current distinct cases, or only a hypothetical future case?
5. Does deleting it preserve failure attribution and performance?

Forwarding, renaming, or future-proofing alone is not a responsibility.

## High-value findings

### Backend leakage

Flag concrete backend knowledge in common code when a capability, policy, or backend interface should own it. Repeated `if backend-kind` branches that evolve independently are strong leads.

Do not flag a deliberate backend distinction that the common contract cannot represent without hiding semantics.

### Duplicated sources of truth

Look for:

- derived flags stored beside canonical state;
- support predicates repeated in dispatch, backend code, and tests;
- several registries/caches for the same resource identity;
- repeated shape/layout validation;
- common and backend implementations that differ only syntactically;
- mirrored cleanup/quarantine/event protocols with the same invariant;
- build/test target boilerplate that encodes the same backend matrix repeatedly.

A task should name the surviving source of truth and everything made redundant.

### Invalid state space

Flag independent booleans/enums/pointers whose combinations permit impossible states, temporary compatibility state that became permanent, unclear cache invalidation, or backend state stored generically without semantic need.

Prefer representations where invalid combinations are unrepresentable or derived rather than synchronized manually.

### Special-case accumulation

Flag caller-level branches that repeat because the lowest stable abstraction lacks one owned capability or operation. Prefer one correction at that owner over scattered exceptions.

Do not generalize one special case into a framework. Sometimes deletion or one explicit backend override is simpler.

### Weightless abstraction

Flag layers that:

- only forward arguments;
- wrap one implementation without owning lifetime, policy, or translation;
- hide critical synchronization/performance behavior;
- add configuration or factories for one fixed current choice;
- introduce extension points with no present consumer;
- extract helpers without removing duplicated decisions.

### Dead and obsolete machinery

Verify and remove unused compatibility paths, feature flags, state fields, adapters, caches, registrations, and test infrastructure. Account for generated/dynamic entry points before calling code dead.

### Accidental work

Complexity can execute. Look for redundant conversions, repeated validation/shape walks, duplicate capability queries, needless copies, repeated allocations, and caches whose maintenance costs more than recomputation. Coordinate performance consequences without duplicating the root cause.

## Simplification acceptance gate

Publish an `AR-###` candidate only when it:

1. identifies objective current redundancy, invalid state, needless indirection, or misplaced responsibility;
2. names the concept/state/branch/source of truth to delete or consolidate;
3. names the single existing or minimal mechanism that remains;
4. preserves required behavior, backend-specific semantics, diagnostics, and async lifetime;
5. adds no avoidable synchronization, copy, allocation, or critical-path computation;
6. reduces net concepts rather than moving code;
7. includes current locations/symbols, acceptance criteria, non-goals, and a falsifier.

Structural risk may be low severity without a present runtime failure. State the concrete divergence or maintenance mechanism and inventory the duplicated locations. Reject subjective style claims.

## Candidate output

Use IDs `AR-###` and the full packet in `finding-rubric.md`. Remediation seeds must be deletion-oriented and implementation-ready. Prefer `remove X and use existing Y` over `introduce abstraction Z` unless Z replaces multiple present responsibilities with one stable owner.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
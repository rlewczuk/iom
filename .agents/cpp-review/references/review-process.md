# Review Process

## Goal

Produce a small number of high-value findings about correctness, stability, simplicity, numerical integrity, and performance of a C++ inference engine with multiple execution backends.

## Evidence hierarchy

Prefer evidence in this order:

1. authoritative change specification / explicit project contract;
2. executable tests, sanitizer results, profiler/benchmark evidence;
3. implementation plus call-path/data-flow reasoning;
4. public backend/runtime API documentation;
5. repository history/comments/conventions;
6. general engineering heuristic.

A heuristic alone should rarely justify a high-severity final finding.

## Scope discipline

### Selected commit

Review target commit against its parent (or empty tree for a root commit). Read surrounding code as needed, but report only issues introduced or materially exposed/worsened by the target.

Inspect at minimum:

- complete diff;
- modified interfaces and call sites;
- related tests;
- corresponding implementations in other backends;
- capability/dispatch/fallback logic;
- relevant history when intent is unclear.

### Whole codebase

Risk-rank the repository before deep review. Prioritize:

1. backend-neutral interfaces and graph scheduler/partitioner;
2. tensor/buffer ownership and allocation abstractions;
3. async execution/synchronization primitives;
4. capability/fallback paths;
5. core operators and backend registration;
6. numerical differential test infrastructure;
7. benchmark/profiling infrastructure;
8. backend-specific hot paths.

State which areas were sampled versus exhaustively inspected.

## Specification mapping

When a `docs/changes/...` directory is supplied:

1. enumerate relevant documents recursively;
2. extract required observable behavior and constraints;
3. distinguish requirements from suggested implementation details;
4. map each requirement to implementation locations and tests;
5. identify requirements with no implementation/test evidence;
6. use this map during contract review.

Do not treat an existing `review.md` as authoritative evidence unless asked to re-review prior findings.

## Invariant catalogue

- **Semantic** — model/operator behavior remains correct.
- **Tensor** — shape/dtype/stride/layout/alignment/aliasing assumptions remain valid.
- **Numerical** — errors remain within justified bounds.
- **Ownership** — resources have unambiguous owners and release points.
- **Lifetime** — resources remain live through asynchronous use.
- **Ordering** — every required dependency exists and unnecessary global ordering is avoided.
- **Visibility** — writes are visible before consumers use them.
- **Capability** — support declarations match actual constraints.
- **Fallback** — unsupported cases fail or fall back intentionally.
- **Concurrency** — supported parallel callers and devices do not race shared state.
- **Error propagation** — enqueue and deferred device failures are surfaced correctly.
- **Performance** — critical-path latency/throughput/memory remain within project budget.
- **Compatibility** — other enabled/disabled backends and supported configurations remain valid.

## Review anti-patterns

Suppress:

- line-by-line style commentary;
- demands for abstraction without a protected invariant;
- shorter-code recommendations that hide lifetime/synchronization semantics;
- performance guesses with no hot-path mechanism;
- duplicate symptoms;
- unrelated legacy defects during commit review;
- test-coverage comments with no untested behavior/invariant;
- “could be cleaner” observations that do not reduce conceptual complexity.

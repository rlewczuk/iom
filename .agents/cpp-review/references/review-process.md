# Review Process

## Goal

Produce a small number of trustworthy, implementation-ready remediation subtasks for correctness, stability, simplicity, numerical integrity, and performance problems in a multi-backend C++ inference engine. Never create an intermediate review document.

## Invocation modes

### Orchestrated specialist pass

The orchestrator supplies one resolved scope, reviewed-state identity, specification map, affected backends, and the candidate packet contract. Review only the assigned area. Return candidate packets to the orchestrator; do not write tasks because cross-area deduplication has not happened yet.

### Standalone specialist pass

The specialist owns scope/specification resolution, reviews only its area, and then invokes `cpp-inference-review-synthesis` on its candidate packets. The finalizer adversarially checks and directly materializes task specs. A standalone specialist is a complete workflow, not a lead generator.

### Shared finalizer

`cpp-inference-review-synthesis` is mandatory after either mode. It rejects weak candidates, reconciles roots, and emits self-contained tasks. No caller writes `review.md` or invokes a separate conversion skill.

## Evidence hierarchy

Prefer evidence in this order:

1. authoritative change specification or explicit project contract;
2. executable tests, sanitizer results, profiler/benchmark evidence;
3. implementation plus call-path/data-flow reasoning;
4. public backend/runtime API documentation;
5. repository history/comments/conventions;
6. general engineering heuristic.

A heuristic alone should rarely justify high severity. It can justify a low-severity structural task only when current code objectively demonstrates duplicated responsibility, redundant state, dead machinery, or needless indirection and the remediation safely removes it.

## Scope discipline

### Selected commit

Review the target commit against its parent, or the empty tree for a root commit. Read surrounding code as needed, but accept only problems introduced or materially exposed/worsened by the target.

Inspect at minimum:

- complete diff;
- modified interfaces and call sites;
- related tests;
- corresponding implementations in other backends;
- capability, dispatch, and fallback logic;
- relevant history when intent is unclear.

### Whole codebase

Risk-rank before deep review:

1. backend-neutral interfaces and scheduling/partitioning;
2. tensor/buffer ownership and allocation;
3. async execution and synchronization;
4. capability and fallback paths;
5. core operators and backend factories/registration;
6. differential numerical tests;
7. benchmark/profiling infrastructure;
8. backend-specific critical paths.

Record sampled versus exhaustive coverage and the reviewed working-tree state.

## Specification mapping

For a supplied `docs/changes/...` directory:

1. enumerate relevant documents recursively;
2. extract observable requirements and constraints;
3. distinguish requirements from suggested implementation details;
4. map requirements to implementation and tests;
5. identify requirements without evidence;
6. inspect existing numbered task specs for established contracts, numbering, dependencies, and equivalent remediations.

Prior review prose is not authoritative evidence unless the user explicitly requests its re-verification. Never create or update `review.md`.

## Invariant catalogue

- **Semantic** — model/operator behavior remains correct.
- **Tensor** — shape/dtype/stride/layout/alignment/aliasing assumptions remain valid.
- **Numerical** — error remains within justified bounds.
- **Ownership** — resources have unambiguous owners and release points.
- **Lifetime** — resources remain live through asynchronous use.
- **Ordering** — required dependencies exist and unnecessary global ordering is avoided.
- **Visibility** — writes are visible before consumers use them.
- **Capability** — support declarations match implementation constraints.
- **Fallback** — unsupported cases fail or fall back intentionally.
- **Concurrency** — supported callers/devices do not race shared state.
- **Error propagation** — enqueue and deferred failures remain observable.
- **Performance** — critical-path latency, throughput, memory, and overlap remain within contract.
- **Compatibility** — enabled/disabled backends and supported configurations remain valid.
- **Simplicity** — one responsibility has one source of truth; state, representations, branches, and layers exist only for a current invariant.

## Mandatory simplification pass

Every specialist asks:

- Can an existing canonical mechanism replace new or duplicate code?
- Are the same facts stored, derived, validated, or dispatched more than once?
- Does an abstraction own an invariant, or merely forward and rename?
- Did a local fix add branches at callers instead of fixing the lowest stable owner?
- Is compatibility/caching/registry/configuration machinery still used?
- Can the remediation delete code, state, or concepts instead of adding another layer?

Preserve backend differences that change correctness or performance. Net concept count matters more than line count.

## Candidate handoff

For every material candidate, return the complete packet required by `finding-rubric.md`, including an implementation-ready remediation seed. This packet is the only handoff to synthesis; there is no prose report conversion step.

If no candidate survives area-level checks, return `No material findings.` plus inspected coverage, verification performed, and material verification gaps.

## Review anti-patterns

Suppress:

- line-by-line style commentary;
- abstraction demands without a protected invariant;
- helper extraction that only moves duplicated code;
- generalized frameworks for a single current case;
- shorter-code recommendations that hide lifetime/synchronization semantics;
- performance guesses without a relevant critical-path mechanism;
- duplicate symptoms;
- unrelated legacy defects during commit review;
- test-coverage comments without an untested behavior/invariant;
- “could be cleaner” observations that do not reduce objective conceptual complexity.
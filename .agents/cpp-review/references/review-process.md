# Review Process

## Goal

Produce a small number of trustworthy, implementation-ready remediation subtasks for correctness, stability, simplicity, numerical integrity, and performance problems in a multi-backend C++ inference engine. Never create an intermediate review document.

This document is the canonical protocol for the five review areas and their synthesis pass. The skill files may add area-specific checks, but they MUST NOT weaken or replace this routing, evidence, scope, synthesis, or output contract.

## Roles, routing, and permissions

The visible/root session is the accountable supervisor and MUST run as `@slow`. A skill cannot switch the model of an already-running session: use the user's role configuration or model override to select `@slow`, and never claim that a skill auto-switched it.

| Lane | Profile and role | Allowed work | Prohibited work |
|---|---|---|---|
| Root supervisor | visible session, `@slow` | resolve scope, route work, adjudicate invariants and candidates, freeze the assignment table, materialize/validate the final set, and report coverage | silently delegating accountability or claiming unrun validation |
| Shared reconnaissance / bounded discovery | `scout`, `@smol` | read-only broad source discovery, search, specification mapping, backend/build map, and compact coverage mapping; this profile intentionally has no bash | design judgments, task drafting, broad repeated scans, or implementation changes |
| Atomic lookup / named command | `boss-errand`, `@smol` | read-only atomic factual lookups, named read-only git/script commands and artifacts, exact followups, and path/collision/equivalence checks | broad source/spec discovery, frontier scans, design judgments, task drafting, or implementation changes |
| Area review | `boss-reviewer`, `@task` | read-only bounded substantive analysis and independent falsification for exactly one area; return candidate packets | delegation, nested orchestration, task-file writes, or build/test/benchmark gates |
| Hard-decision advisor | `boss-advisor`, `@advisor` | tool-free reasoning over compact packet CONTENT for genuinely hard semantic, lifetime, numerical, disputed-evidence, or key/high-risk acceptance/remediation decisions | source/filesystem/web/shell/tool access, opening an artifact URI/path, delegation, or treating model agreement as verification |
| Task drafting | `boss-builder-fast`, `@smol` (or `boss-builder`, `@task` when needed) | mechanical materialization of an already frozen assignment table into exact destinations using the template | design authority, new review, candidate selection, concurrent number allocation, or implementation/test changes |

The root MUST make the routing and coverage decision. There are no five expensive area supervisors: one `@slow` root schedules one shared cheap reconnaissance phase/map and all five bounded `boss-reviewer` leaves. The shared phase MAY use independent bounded `scout @smol` shards batched once and merged once; it is not required to be one monolithic agent. All five area leaves MUST be dispatched concurrently in one batch when delegation is available. Narrow `boss-errand` lookups/followups are dispatched only for explicit factual questions or named read-only commands. Do not perform gratuitous calls for trivial known facts.

If delegation is unavailable, report the routing and coverage limitation and request explicit permission before any materially costlier fallback. Do not silently run a sequential five-frontier review, replace missing cheap profiles with expensive work, or imply equivalent coverage. In standalone mode, a specialist may complete only its already assigned area and mandatory synthesis; it MUST NOT widen into an unbounded `@slow` whole-repository review or silently substitute missing lanes, and it MUST disclose unavailable lanes and uninspected areas.

Workers never run build, test, sanitizer, profiler, benchmark, or other verification gates. They may propose exact commands, hardware, workloads, and expected observations in packets. The root runs actual verification after candidate collection and records the command and result. Any accelerator build/test/benchmark must follow `remote-development` rules: the selected remote Linux host is the execution environment, local source edits remain in the local workspace, and no local accelerator result is represented as remote evidence.

## Invocation modes

### Full orchestrated review

The code-review orchestrator is a complete workflow. It reads `skill://boss` before any other skill or repository material, runs as the `@slow` root, resolves one scope, performs one shared cheap reconnaissance phase/map (using one batched set of bounded `scout @smol` shards when useful), retrieves any complete selected diff through a named read-only `boss-errand @smol` command/artifact for those scouts and area leaves to inspect, dispatches all five area leaves concurrently, collects narrow errand followups as needed, and invokes synthesis once. It owns final acceptance, assignment ordering, task writing, generated-set validation, and the response.

The root supplies every area leaf with the same resolved scope, reviewed-state identity, specification map, affected backend list, reconnaissance map, search coverage, and candidate packet contract. Area leaves review only their assigned area and return candidate packets; they do not invoke another review skill, synthesis, or supervisor.

### Orchestrated specialist pass

The orchestrator supplies one resolved scope, reviewed-state identity, specification map, affected backends, reconnaissance map, and the candidate packet contract. Review only the assigned area. Return candidate packets to the orchestrator; do not write tasks because cross-area deduplication has not happened yet. Use exact known path:line/symbol ranges from the map and narrow factual followups rather than a new broad scan.

### Standalone specialist pass

When a specialist skill is invoked directly, its running `@slow` session owns the complete selected area: read `skill://boss` first, resolve scope/specification, establish coverage, perform the area review, record validation gaps, and invoke `cpp-inference-review-synthesis` as its mandatory final pass. It MUST NOT spawn a nested supervisor, widen to an unbounded whole-repository `@slow` fallback, or pretend that another area was reviewed. The standalone finalizer receives that area's complete candidate packets and emits tasks directly. Preserve the same selected-commit and whole-codebase rules and disclose area-only coverage.

### Shared finalizer

`cpp-inference-review-synthesis` is mandatory after either mode. It rejects weak candidates, reconciles roots, and emits self-contained tasks. It MUST NOT create `review.md`, invoke a separate conversion skill, or recursively invoke the full orchestrator. In standalone synthesis mode, its `@slow` root owns the finalizer workflow and may dispatch only bounded evidence-check, advisor, existence/collision, and drafting lanes described below.

## Evidence hierarchy

Prefer evidence in this order:

1. authoritative change specification or explicit project contract;
2. executable tests, sanitizer results, profiler/benchmark evidence;
3. implementation plus call-path/data-flow reasoning;
4. public backend/runtime API documentation;
5. repository history/comments/conventions;
6. general engineering heuristic.

A heuristic alone should rarely justify high severity. It can justify a low-severity structural task only when current code objectively demonstrates duplicated responsibility, redundant state, dead machinery, or needless indirection and the remediation safely removes it.

## Scope resolution and reconnaissance

The root resolves exactly one review scope and records its identity before area dispatch. Scope resolver/specification scripts are deterministic root work; broad reading, search, and mapping belong to `scout @smol`; atomic named git/script commands, exact followups, and path checks belong to `boss-errand @smol`.

For selected commits, `[ROOT @slow]` runs without changing checkout state:

```bash
python3 .agents/cpp-review/scripts/resolve_review_scope.py --repo . --commit-hash '<hash>'
```

or:

```bash
python3 .agents/cpp-review/scripts/resolve_review_scope.py --repo . --commit-message '<message>'
```

The root resolves hashes to exactly one commit, matches complete messages (then exact subjects), never chooses among multiple matches or fuzzy matches, and uses the empty tree for a root commit. The root captures target hash/subject, parent/baseline, changed and renamed/copied files, affected builds/backends, and supplied specification path.

For an optional specification directory, `[ROOT @slow]` runs:

```bash
python3 .agents/cpp-review/scripts/resolve_spec_path.py --repo . --spec '<spec-dir>'
```

The root verifies the resolved path remains beneath `docs/changes/` and passes the resolved destination to reconnaissance and synthesis. No specification directory is invented.

After scope resolution, the root starts one shared read-only reconnaissance phase/map. Independent bounded `scout @smol` shards MAY be batched and merged once; they cover broad source, callers, tests, backend counterparts, capability/dispatch/fallback paths, build/backend configuration, whole-codebase risk ranking, and specification mapping. For selected commits, an `[ERRAND boss-errand @smol]` worker first retrieves the complete diff (including rename/copy metadata) with a named read-only git command/artifact; the scouts and relevant area leaves then inspect that artifact and the exact source ranges it identifies. The scout phase reports a compact map with exact `path:line`/symbol anchors, exhaustive versus sampled search coverage, and uninspected areas. It MUST not turn raw logs or a repository dump into the packet.

For whole-codebase reviews, `[ROOT @slow]` records `HEAD` and working-tree state, while `[SCOUT scout @smol]` shards risk-rank before deep reading. Use this order unless an explicit scope requires otherwise:

1. backend-neutral interfaces and scheduling/partitioning;
2. tensor/buffer ownership and allocation;
3. async execution and synchronization;
4. capability and fallback paths;
5. core operators and backend factories/registration;
6. differential numerical tests;
7. benchmark/profiling infrastructure;
8. backend-specific critical paths.

For selected commits, full-diff review still occurs in the shared scout map and in the relevant area leaves. Each area leaf checks its exact changed hunks and necessary surrounding interfaces/callers/tests/counterparts; it accepts only a root cause introduced or materially exposed/worsened by the target. The root does not broad-scan the source or ingest an unbounded diff: it may read exact known file:line ranges when cheaper than another dispatch, then records that coverage.

If the reconnaissance map has a factual gap, `[ROOT @slow]` sends a narrow, atomic `[ERRAND boss-errand @smol]` followup naming the exact question, paths/symbols, and desired `path:line` evidence. Followups do not repeat the frontier review and their evidence is merged into the shared map before invariant decisions. Area leaves may request a followup through the root; they do not dispatch it themselves.

## Specification mapping

For a supplied `docs/changes/...` directory:

[SCOUT scout @smol] performs these specification-map reads:

1. enumerate relevant documents recursively;
2. extract observable requirements and constraints;
3. distinguish requirements from suggested implementation details;
4. map requirements to implementation and tests using exact anchors;
5. identify requirements without evidence;
6. inspect existing numbered task specs for established contracts, numbering, dependencies, and equivalent remediations.

The root supplies this compact map to every area leaf and synthesis. Prior review prose is not authoritative evidence unless the user explicitly requests its re-verification. Never create or update `review.md`. If the map omits a needed fact, request a narrow followup rather than asking an advisor to read a path.

## Invariant catalogue

Before dispatch, `[ROOT @slow]` identifies affected:

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

Area leaves use the collected map and these invariants to choose where to inspect. They MUST separate source facts from inference and state negative evidence when a guard, owner, counterpart, or test preserves an invariant. The root, not a worker or advisor, decides whether an invariant is established.

## Selected-commit discipline

Review the target commit against its parent, or the empty tree for a root commit. Read surrounding code as needed, but accept only problems introduced or materially exposed/worsened by the target.

At minimum, the shared map and area evidence cover:

- complete diff;
- modified interfaces and call sites;
- related tests;
- corresponding implementations in other backends;
- capability, dispatch, and fallback logic;
- relevant history when intent is unclear.

A candidate must identify baseline-v-target provenance. Do not report an unrelated legacy defect merely because the target area was read.

## Whole-codebase discipline

Review the checked-out working tree as a system. Use the risk order in reconnaissance before bounded area analysis. Record the reviewed `HEAD`, working-tree state, sampled versus exhaustive coverage, search paths/patterns where material, and explicitly uninspected areas. Never imply every file was inspected when it was not.

## Mandatory simplification pass

Every specialist asks:

- Can an existing canonical mechanism replace new or duplicate code?
- Are the same facts stored, derived, validated, or dispatched more than once?
- Does an abstraction own an invariant, or merely forward and rename?
- Did a local fix add branches at callers instead of fixing the lowest stable owner?
- Is compatibility/caching/registry/configuration machinery still used?
- Can the remediation delete code, state, or concepts instead of adding another layer?

Preserve backend differences that change correctness or performance. Net concept count matters more than line count. A simplification candidate still needs objective evidence of duplicated responsibility, redundant state, invalid combinations, dead machinery, unnecessary representation, special-case branching, or an abstraction without an owned invariant.

## Candidate packet and evidence handoff

For every material candidate, return the complete packet required by `.agents/cpp-review/references/finding-rubric.md`, including an implementation-ready remediation seed. The packet contract is extended with the following evidence fields; these fields do not replace any rubric field:

- reviewed state and exact scope identity;
- exact `path:line`/symbol location and decisive minimal excerpts (not a raw repository dump);
- source facts separated from inference;
- relevant callers, guards, owners, counterparts, and negative evidence;
- search coverage, sampled/exhaustive status, and uninspected areas;
- baseline-v-target provenance for selected-commit findings;
- actual validation performed and verification gaps (workers state proposed commands only; root records actual results);
- a concrete falsifier and a full remediation seed.

Reference internal artifacts for workers when useful, but advisor packets MUST include compact packet CONTENT. An advisor must never be required to open a URI or path. Bound output and split work rather than dumping repository text or raw tool logs; never truncate away decisive evidence or coverage. If no candidate survives area-level checks, return `No material findings.` plus inspected coverage, validation performed, and material verification gaps.

The packet is the only handoff to synthesis; there is no prose report-conversion step. In orchestrated mode, area leaves return packets only. In standalone mode, the specialist passes packets directly to the mandatory synthesis finalizer.

## Synthesis routing and gates

Synthesis is an adversarial final pass, not a second broad review. `[ROOT @slow]` first normalizes packets and freezes the input scope/map. It then separates work as follows:

1. `[EVIDENCE boss-reviewer @task]` independently checks every candidate's scope, mechanism, guards/callers/counterparts, evidence, falsifier, and remediation seed. These workers receive packet content and relevant exact excerpts; they do not delegate, perform broad frontier scans, write tasks, or run gates.
2. `[ADVISOR boss-advisor @advisor]` is used only for genuinely hard semantic/lifetime/numerical questions, disputed evidence, or key/high-risk acceptance/remediation choices. It receives a compact self-contained packet CONTENT, with facts, inference, alternatives, and the precise question. It has no tools and cannot open paths. If it returns `NEED EVIDENCE`, `[ROOT @slow]` asks `[ERRAND boss-errand @smol]` one exact source question, appends only the evidence delta, and may return that delta to the same advisor. Agreement is not verification.
3. `[PATH boss-errand @smol]` checks existence, current path/symbol validity, destination shape, collisions, and equivalent existing task specs using exact known paths. It does not perform routine semantic review or broad scans.
4. `[ROOT @slow]` adjudicates acceptance/rejection, merges same-root symptoms, preserves separate roots, assigns final IDs/severity/confidence/priority/order/blockers/slugs, and freezes the complete assignment table before any file is written.
5. `[DRAFT boss-builder-fast @smol]` (or explicitly selected `boss-builder @task`) mechanically writes one self-contained task per accepted root cause to the exact table destinations using `.agents/cpp-review/templates/remediation-task.md`. It has no design authority and cannot allocate numbers concurrently. The root may perform equivalent deterministic drafting only when no builder lane exists and must report that routing limit.
6. `[ROOT @slow]` reads every generated `spec.md`, checks the entire generated set against the table and synthesis gates, and runs the validator over exactly the new files. This is document validation, not an implementation gate.

No candidate is accepted solely because an advisor agrees with it. No task is written before the root freezes the table. If evidence is insufficient, reject or retain a hypothesis with its decisive falsification outcome; do not upgrade uncertainty by repetition.

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
- “could be cleaner” observations that do not reduce objective conceptual complexity;
- routine broad scans by area leaves, advisors, or synthesis;
- raw diagnostic dumps in place of evidence;
- claims of verification for commands no root actually ran.

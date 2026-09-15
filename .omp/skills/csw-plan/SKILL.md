---
name: csw-plan
description: Split a change specification into parallel-ready, prioritized implementation or design mini-specifications with only genuine dependencies. Use only through /csw-plan <task-name>[/subdirectory] [--impl|--hld] or when explicitly requested.
hide: true
---

# CSW Plan

Turn one change specification into a compact, broad, shallow set of implementation or design tasks. **Target 10–20 meaningful immediate subtasks and at most two task levels below the original change specification. Prefer direct `impl` leaves; `hld` is an exception that must earn its extra planning pass.** Write one focused, self-contained mini-specification per task; do not implement the change. Implementation leaves must remain safe for a cheap flash-class model. Avoid repeated intermediate specifications: each extra handoff risks losing requirements and weakening leaf implementation context.

**Maximize safe parallel implementation across the entire change. Start with independent tasks and add only proven prerequisites; never turn the ordered task list into a serial execution plan. This applies to every useful boundary, including independent tensor operations, backend implementations, and unrelated outcomes within the same component.**

The command supplies a path relative to `.cswd/tasks/` and an optional generation mode:

```text
<task-name>[/subdirectory...] [--impl|--hld]
```

For target `<target>`, resolve paths and metadata through `.omp/csw/bin/task_ctl`:

- canonical task ID: `.cswd/tasks/<target>`;
- parent specification: `task_ctl get '<task-id>' --spec`;
- user remarks, optional: sibling `spec-fixme.md`;
- generated children: `.cswd/tasks/<target>/<NN>-<task-slug>/spec.md`, with sibling `task.yml`;
- final planning summary: `.cswd/tasks/<target>/summary.md`, beside the parent `spec.md`.

`task.yml` is the only task metadata authority. Never read, create, edit, or parse it directly; use `task_ctl` for every metadata operation. `spec.md` contains the task contract, not control fields. The numeric directory prefix remains a human-readable order label; `order` in the control file is authoritative. `summary.md` is a derived, human-readable planning report, not a metadata authority or lifecycle tracker. Do not create any additional TODO or index file.

Below, `task_ctl` always means `.omp/csw/bin/task_ctl --repo '<project-root>'`; use that executable, not an assumed PATH installation. It requires Python 3 and PyYAML; its module docstring and `--help` contain the complete CLI contract.

## Task types and generation modes

Every generated task MUST have `type: impl` or `type: hld` in its control file, set through `task_ctl`:

- `impl` — a leaf task ready for direct implementation by an implementer agent, without further design or decomposition.
- `hld` — a high-level design for a substantial cohesive group of related areas that genuinely needs one further planning pass. Its expected children are `impl` leaves; another `hld` generation requires the depth exception below. Plan it again with `/csw-plan <target>/<NN>-<task-slug>`.

Resolve the mode after applying conversation and fixme precedence:

| Invocation / effective parent control type | Generation mode |
| --- | --- |
| Explicit `--hld`, regardless of parent type | Design |
| Explicit `--impl`, regardless of parent type | Implementation |
| No flag, parent `type: hld` | Design |
| No flag, parent `type: impl` | Automatic |

An explicit flag overrides the parent type; a parent marked `impl` does not force implementation mode. Use `task_ctl get '<task-id>' --type`. New controls created with `set` default to `hld` and `new`. Missing or invalid controls are script errors; correct the reported input through `task_ctl` rather than guessing.

- **Design:** treat the input as high-level design and group related areas along component, responsibility, or contract boundaries, subject to the breadth/depth policy below. Resolve design far enough in this pass to emit `impl` wherever practical. Use `hld` only for substantial groups that pass the HLD admission check; neither `--hld` nor an `hld` parent requires mostly `hld` children or authorizes another design layer. Planning a first-level `hld` should normally produce 10–20 direct `impl` leaves.
- **Implementation:** keep decomposing until every generated task passes the leaf-readiness check and is `impl`. Resolve necessary design decisions in this planning pass; never relabel complex work as `impl` just to satisfy the flag. If a material decision cannot be resolved, follow the ambiguity workflow before writing affected tasks; do not silently fall back to design mode.
- **Automatic:** identify logical boundaries, then actively refine candidates toward direct `impl` leaves in this pass. A failed initial leaf-readiness check means investigate, settle decisions, or split into sibling leaves before considering `hld`; it is not an automatic HLD classification. Preserve a substantial related group as `hld` only when it passes the HLD admission check and fits the two-level target.

Generate only the immediate children for this invocation, not a nested task tree. An `hld` child must narrow the parent's scope or establish concrete component contracts and a useful next decomposition; never copy the parent into an unchanged design wrapper. The type records readiness, not priority or implementation completion.

## Breadth and depth policy — mandatory planning gate

**Optimize for 10–20 useful siblings, not a small elegant-looking outline with a deep tree beneath it. One level is better than two when leaves are ready; two levels are the strong default maximum, not a target to fill. This policy applies in every generation mode and on every recursive invocation.**

- **Breadth:** the planner MUST strive for 10–20 immediate subtasks containing real, required outcomes. Fewer than 10 are allowed **only when the remaining specification does not support 10 meaningful tasks** after complete requirement accounting and a serious search for useful boundaries. Explain that shortage concretely; convenience, limited initial exploration, broad component names, or deferring work into HLD wrappers do not qualify. Never pad the count with invented scope, mechanical fragments, separate tests for each behavior, or already completed work. One genuinely indivisible leaf remains one `impl`; no remaining work remains zero tasks.
- **Compact groups:** group related areas into substantial cohesive scopes, not one HLD per minor concern, abstraction layer, file, or naming category. If a proposed HLD would yield only one or two IMPL leaves, resolve and promote those leaves into the current sibling set instead. A group of only a few leaves is likewise a strong signal to flatten or regroup related work, not to introduce an intermediate specification.
- **Depth:** count the original change specification as level 0, its immediate children as level 1, and their children as level 2. Aim for `root → impl` or `root → hld → impl`, never routine `root → hld → hld → impl`. Ten to twenty substantial groups with ten to twenty leaves each already accommodate roughly 100–400 leaves; do not create deeper levels just to mirror a subsystem taxonomy. A nested invocation does not reset the depth count.
- **Readiness still matters:** flatten by doing the planning now and promoting bounded leaves, not by relabeling an unresolved complex task `impl`. Resolve uncertainty through repository evidence and the ambiguity workflow. Missing facts or a difficult decision alone do not justify HLD.
- **Exceptions are evidence-backed, not habitual:** before exceeding two levels, the root MUST demonstrate why settling decisions now, promoting leaves, and regrouping related areas cannot produce safe leaves within two levels. Record the exact scope/coupling obstacle, alternatives rejected, and the shortest concrete route to `impl`. Prefer slightly more than 20 useful siblings to another HLD layer when that preserves cohesion and readiness; justify any count above 20. A depth exception never permits a one- or two-leaf HLD wrapper.
- **Reject and revise before writing:** a draft with unexplained fewer-than-10 children, thin HLD wrappers, or unjustified depth beyond level 2 fails planning. Rework the decomposition before freezing assignments. Do not merely mention these defects as residual risks in the final summary.

## Guardrails

- Require one non-empty target argument and at most one mode flag, `--impl` or `--hld`, in either position. Reject conflicting or repeated flags, unknown flags, and extra positional arguments before writing. Parse flags separately from the target; never include them in a task path. Use `task_ctl` for task path validation and resolution; do not reconstruct or bypass its traversal and containment checks.
- Allow safe nested targets such as `allocator/gpu-layout`; the argument names the exact directory containing the parent `spec.md`.
- If the parent `spec.md` does not exist, report its expected path and stop. A missing `spec-fixme.md` is normal.
- Read the complete parent specification and complete fixme file when present.
- Treat direct conversation instructions as highest priority, then `spec-fixme.md` as user corrections or additions, then `spec.md`. For claims about current repository behavior, the repository is authoritative. Ask only when a material intent or policy conflict remains.
- Do not modify parent requirements, `spec-fixme.md`, implementation files, or unrelated task directories. The only parent metadata change is `status: planned` after successful child generation.
- Keep the design minimal. Exclude speculative abstractions, future-proofing, generic frameworks, optional extras, unrelated cleanup, and complementary features not required by the source specification.
- Do not invent functionality to make a task feel complete. Every requirement in every mini-spec must trace to the parent spec, the fixme remarks, a direct user instruction, or a repository constraint necessary to implement them correctly.
- Reach the 10–20 breadth target through required, meaningful outcomes, never task-count padding. Fewer tasks require the specification-shortage rationale in the breadth/depth policy. If the specification is already one cohesive, indivisible leaf-ready unit, create one `impl` task in any mode.
- Do not create tasks for work that is already complete and conforms to the specification.
- Do not implement the change. The deliverables are task mini-specifications, their script-managed controls, the parent-directory `summary.md`, and the concise ordered list in the completion response.

## Boss orchestration contract

This seven-step workflow is one Boss process. Before starting it, the root MUST read `skill://boss`, run and preserve its deterministic preflight, then apply this skill's explicit override: substantive repository fact-gathering uses the read-only `csw-plan-facts` profile at `@task`, rather than Boss's default cheap exploration lane. The root should use `@slow`; `@csw-yoda` is also automatically approved, while any other mismatch follows Boss's warning-and-consent gate and may continue only if the user explicitly agrees. Use only the preflight's `roles`, `models`, and required `agents` entries for model routing, overrides, advisor state, availability, and profile tool restrictions; do not manually inspect or reconstruct them. Report a failed preflight and stop; never switch models, inherit an expensive parent, or silently fall back.

The `@slow` root owns intake, mode resolution, complete requirement accounting, decomposition, per-task type classification, priorities, the dependency DAG, architectural and ambiguity decisions, the frozen complete assignment table, verification, and final output:

- `[FACTS csw-plan-facts @task]` performs scoped, parallel, read-only factual discovery using exactly the profile's `read`, `grep`, `glob`, `lsp`, and `ast_grep` tools (`advisor: false`, `spawns: []`). It has no design authority, writing, delegation, or gates.
- `[DRAFT boss-builder-fast @smol]` writes frozen mini-specs and performs simple mechanical edits only after the root fixes destinations, types, order, blockers, requirements, and acceptance criteria. It has no design, type-classification, or numbering authority.
- `[PATH boss-errand @smol]` may invoke `task_ctl` for mechanical metadata, path, existence, and collision checks against exact known paths; it MUST NOT parse control files, duplicate script logic, perform substantive source discovery, or make design decisions.
- `[ADVISOR boss-advisor @advisor]` handles only genuinely key architectural decisions from supplied facts. It is tool-free and packet-only: it cannot search, fetch artifacts, edit, delegate, or run gates.

Fact packets MUST provide exact `file:line` and symbol evidence, decisive minimal excerpts, a `done`/`partial`/`missing`/`discrepant` mapping, relevant conventions and focused test commands, inspected and uninspected areas, gaps, and separate `SOURCE FACTS` from `INFERENCE`. Batch truly independent discovery and disjoint writers, collect each phase before consuming its output, and do not split output tasks merely for parallelism. No worker delegates recursively; workers skip gates, tests, linters, builds, and formatters.

These are planning-time discovery and writing barriers, not dependencies between the generated implementation tasks. Disjoint-writer rules, shared verification resources, and serialized integration protect execution; they MUST NOT become `blocked-by` edges without a required producer output.

Every brief is self-contained: it states exact scope and files, established facts and decisions, required output, non-goals, acceptance criteria, and any blockers. Advisor `CONTENT` must inline the requirements and non-goals, decisive excerpts and facts, constraints, alternatives with trade-offs, and one exact question; a path or URI alone is invalid. If the advisor returns `NEED EVIDENCE`, route the exact factual question back to `csw-plan-facts`, append only the evidence delta, and keep the final decision at the root.

Fact briefs MUST request evidence for both required producer/consumer relationships and independence, not just a list of plausible dependencies. Frozen writer briefs MUST include the required outputs behind each approved edge, the settled contracts that let independent siblings proceed, and ownership of any shared edit regions.

This contract preserves the existing generation contract: precedence and path guards, minimal scope, omission of completed work, the task template, collision protection, self-contained mini-specs, ambiguity handling, and the concise completion response remain authoritative.

## Workflow

The seven stages below are Boss-owned; role dispatch supplements the stage and does not create a second workflow.

### 1. Read and normalize the requested change
The `@slow` root completes argument parsing, intake, precedence and mode resolution before any delegation, including all requirements, non-goals, assumptions, and unresolved decisions.

Resolve the parent with `task_ctl get '<task-id>' --spec` and read its complete output file, then `spec-fixme.md` if present. Read metadata with `task_ctl get '<task-id>' --all`; stop on missing or invalid control metadata. Extract:

- required outcomes and observable behavior;
- explicit scope and non-goals;
- interfaces, data shapes, state transitions, errors, compatibility requirements, and constraints;
- named files, symbols, commands, tests, examples, and external or local references;
- every actionable fixme remark;
- unresolved decisions and assumptions.
- the parent control type returned by the script and the effective generation mode.

Build one coherent requirement set using the precedence rules above. A fixme correction replaces the conflicting parent requirement; do not preserve both alternatives. Do not propagate brainstorming, rejected alternatives, or editorial commentary as implementation work.

Establish the target's level relative to the original change specification from canonical task ancestry and `task_ctl get '<task-id>' --source` / `--spec` as needed; do not treat each nested target as a new level 0. Read relevant original/ancestor requirement sections when intermediate specs omit or compress essential context, and reconcile them with current precedence and scope. Carry the original root, current level, and remaining depth allowance into decomposition and writer briefs. Existing deep trees are not a precedent: finish their remaining planning directly into `impl` leaves wherever possible without moving or rewriting unrelated tasks.

### 2. Ground the work in the repository
The root dispatches the exact `csw-plan-facts` (`@task`) profile for scoped parallel repository facts, then reconciles its evidence rather than delegating requirement or design authority.

Explore only the project areas needed to decompose and anchor the change. Read referenced files and enough surrounding implementation, call sites, tests, configuration, schemas, and project design documentation to establish:

- what is already implemented and correct;
- what is partial, missing, or inconsistent with the specification;
- the actual files and symbols each remaining change touches;
- local implementation and test conventions to reuse;
- existing facilities or extension points that avoid new machinery;
- required producer outputs and the exact consumers that need them;
- evidence of independence across outcomes, including tensor operations and backends that share an interface but not implementation outputs;
- shared edit regions versus read-only analogues, and whether mutation ownership can be partitioned without a semantic prerequisite.

Classify each source requirement as **done**, **partial**, **missing**, or **discrepant**. Generate tasks only for the remaining work. Do not trust a path, symbol, signature, or current-behavior claim until checked when it affects a task.

Repository exploration is for reducing downstream search, not for proposing adjacent improvements. Ground `impl` tasks down to concrete implementation and test touchpoints; ground `hld` tasks at their component boundaries, existing facilities, interfaces, constraints, and verification strategy without inventing leaf-level detail. Ignore unrelated defects unless one directly blocks the specified behavior; if it does, include only the smallest necessary correction.

### 3. Decide whether and how to decompose

The root owns the decomposition judgment and any architectural decision. Consult `boss-advisor` only with self-contained packet `CONTENT`; never ask it to investigate the repository.

First inventory the remaining outcomes across the whole specification, then identify logical boundaries and apply the selected mode and mandatory breadth/depth gate. Do not freeze a handful of broad headings and reflexively label each HLD. Sketch the likely leaf outcomes before choosing groups so the root can compare a flat sibling set with substantial two-level groups. This is a sizing/coverage sketch, not generation of descendant specs. Every task must be:

- one concern or component with a clear outcome and explicit scope;
- independently verifiable at its declared level after its blockers are complete;
- large enough to avoid handoff-only tasks and meaningless scaffolding;
- complete across the layers needed for its outcome, rather than a horizontal list such as “all types,” “all logic,” then “all tests.”

For each candidate, apply this **leaf-readiness check**. Assign `impl` only when all of these hold for a fresh cheap flash-class implementer (for example `gpt-5.6-luna`; this is a complexity benchmark, not a model-routing override):

- one bounded behavior or mechanical change, with a small coherent set of verified touchpoints and enough context to fit one fresh context window;
- required interfaces, data formats, errors, ownership, ordering, and compatibility decisions are settled, with no architectural choice or further decomposition left to the implementer;
- local implementation and test patterns are identified, so no broad repository search or cross-subsystem discovery is needed;
- focused acceptance and verification can demonstrate the whole outcome after blockers complete;
- no tightly coupled multi-backend change, wide migration, unfamiliar algorithm, or difficult concurrency/lifetime reasoning remains beyond what the supplied decisions and established patterns make routine.

Judge reasoning burden and coupling, not just file count, line count, or whether the prose fits in context. **When uncertain, investigate and refine now; do not default to `hld`.** In every mode, first settle decisions and try a useful split into direct sibling leaves. In Implementation mode, continue until all pass; in Automatic or Design mode, an HLD candidate must additionally pass every admission condition below.

**HLD admission check — the root MUST record all of these before assigning `hld`:**

- A substantial, cohesive group of related required outcomes remains; the root has sketched its likely direct leaves and actively targeted 10–20 of them. A smaller group is admissible only because its specification supports fewer meaningful outcomes, never merely because the first grouping was narrow; one or two leaves are never sufficient.
- A further planning pass adds necessary design/decomposition value that cannot reasonably be completed now. Name that work and why promoting leaf candidates or regrouping related areas is inferior; “complex,” “multi-file,” “needs more detail,” and planner uncertainty are not sufficient reasons.
- Boundary contracts with siblings are already fixed, and the next pass is expected to produce direct `impl` leaves within the original root's two-level allowance. Any exception has the concrete evidence and shortest leaf route required by the depth policy.
- The HLD contract will preserve all applicable source requirements, settled decisions, acceptance criteria, and necessary repository context so its leaves do not inherit a progressively weaker summary.

For example: 12 ready outcomes become 12 `impl` siblings, not a few HLD headings; 12 substantial related groups each needing about 15 leaves can justify `root → hld → impl`; a proposed HLD with two leaves is replaced by those two sibling `impl` tasks. These are decomposition examples, not task quotas or permission to invent outcomes.

For an `hld` task, state the component's eventual behavior, responsibilities, inputs/outputs, boundary contracts, constraints, non-goals, and integration obligations. Identify the remaining decomposition/design questions and the criteria its future implementation leaves must satisfy. Internal decisions may remain for the next planning pass only when they do not leave sibling contracts ambiguous. An `hld` task is not permission to implement a large change directly.

Prefer vertical slices for `impl` tasks that leave the repository in a coherent state. Include focused tests with the behavior they cover. Create a separate test task only for genuinely cross-cutting integration, end-to-end, migration, performance, or compatibility verification spanning multiple slices.

Create a prerequisite refactor task only when the current structure makes the required change unsafe or impractical. Keep it mechanical and no broader than necessary. For an unavoidable wide refactor, use the smallest dependency-safe expand–migrate–contract sequence; do not introduce compatibility layers otherwise.

Split independent outcomes, unrelated file groups, and different blockers when they form useful boundaries. In every mode, try refining failed leaf candidates into direct sibling leaves; retain a complex group as `hld` only after the admission check. Merge candidates when one only scaffolds another, neither is useful alone, or separating them duplicates substantial context. Rebalance the whole sibling set toward 10–20 meaningful tasks without sacrificing readiness; do not solve an over-narrow grouping by adding another design layer.

If no implementation work remains, create no task directories and report that the specification is already satisfied, with the repository evidence that supports that conclusion. Still write the final `summary.md` described in stage 7, with an empty generated-task table and the supporting evidence; leave the parent status unchanged.

### 4. Build and prioritize the task graph

The root owns priorities and genuine dependency edges. `task_ctl plan` only validates and orders the graph supplied by the root: it performs the topological sort, numbering, exact blocker-ID substitution, and collision checks. It cannot detect unnecessary semantic dependencies or discover missed parallelism. `boss-errand` may invoke it against root-approved candidates; it does not make those judgments.

#### Dependency-minimization gate — mandatory before numbering

**Default to `blocked-by: []`. Every edge needs proof; every independent pair MUST remain unordered by the dependency graph, with neither a direct edge nor an artificial transitive path between them. An acyclic graph is not enough: an unnecessary serial chain fails planning.**

Before freezing candidates, the root MUST:

1. **Inventory inputs and outputs.** For each candidate, identify its delivered behavior/artifacts, the precise interfaces or state it consumes, its implementation and focused verification touchpoints, and shared edit ownership. Distinguish existing facilities and contracts settled during this planning pass from missing implementation outputs. Resolve shared design decisions now rather than requiring one implementation to serve as the next task's design input.
2. **Search the whole decomposition for independence.** Compare candidates across and within components, operations, backends, formats, and test scopes—not just adjacent rows. Split independently implementable outcomes and different prerequisite sets where they form meaningful leaf-ready tasks; do not keep unrelated work in a serial bundle merely because it shares a component or file. Preserve the breadth/depth and no-padding rules; do not invent a task for every operation/backend combination.
3. **Prove each proposed edge.** For “B is blocked by A,” name the exact output B consumes, evidence that it is unavailable without A, and what in B's scoped implementation or focused acceptance would be impossible or incorrect without it. Ask: **with the existing repository and frozen boundary contracts, can B be implemented and independently verified without waiting for A's output?** If yes, omit the edge. If evidence is missing, investigate or resolve the blocking ambiguity; neither guess a dependency nor discard a real prerequisite to widen the graph. Preserve source-mandated sequencing. Every retained edge MUST have non-empty `remarks` naming the required output and why this consumer needs it.
4. **Reject incidental sequencing.** Input order, directory numbers, priority, conceptual similarity, an implementation analogue, “do one backend/operation first,” shared files, a shared test suite or accelerator, limited workers, review order, and serialized integration are not producer/consumer dependencies. Settle reusable contracts/algorithms in the mini-specs instead of making later leaves wait to copy an earlier implementation. Partition shared edit regions or assign the common mutation to one owner; coordinate unavoidable resource contention during execution. If a task genuinely consumes another task's changed code or state, retain that precise prerequisite—not a chain across every task touching the file.
5. **Use minimal prerequisites and real joins.** When required common implementation is missing, isolate that cohesive outcome if necessary and let every consumer depend directly on it, not on one another. If the common contract/facility already exists, do not create a prerequisite task. Keep focused verification with its behavior; only genuinely required cross-cutting verification belongs in a join depending on the outputs it actually checks. Do not weaken acceptance or add scaffolding to manufacture concurrency. Attach blockers at the narrowest scope needing them: an `hld` blocker gates all its descendants, so regroup instead of lifting one branch's prerequisite onto unrelated leaves.
6. **Falsify the graph before writing.** Challenge every edge with the counterfactual above, remove unnecessary edges and redundant transitive restatements, then check reachability between every pair identified as independent. Inspect serial chains for missed fan-out and joins for unrelated inputs; a chain through an intermediate task still serializes its endpoints. Reassess inherited edges on replans rather than copying a historical chain. Every task may proceed once its own genuine blockers are complete; never require a whole numbered group or parallel frontier to finish before an unrelated branch proceeds.

For example, when a shared API implementation is genuinely missing, use `shared-api → {CUDA, ROCm, SYCL, TTNN}` and, only if required, `{CUDA, ROCm, SYCL, TTNN} → cross-backend-verification`. Never use `CUDA → ROCm → SYCL → TTNN` merely as a preferred implementation order. With the API already available, all four backend tasks have empty blocker lists.

Likewise, independently implementable tensor operations such as add, multiply, activation, and reduction remain siblings with no mutual dependencies, even within one backend or a shared dispatch/test file. A genuinely composed operation that consumes new add and activation implementations depends on those outputs, not on unrelated multiply or reduction work. These are examples of the general rule, not backend- or operation-specific exceptions.

Writing or decomposing an `hld` spec does not satisfy a blocker that requires its implemented behavior; that blocker remains until the required descendants are implemented and verified. **Reject and revise an over-serialized draft before calling `task_ctl plan`; do not defer dependency minimization to writers or the execution scheduler.**

Assign a priority:

- **P0** — required prerequisite, public contract, migration, or correctness/safety work that gates broad progress;
- **P1** — required feature behavior on the normal implementation path;
- **P2** — required finishing work such as cross-cutting verification or documentation that does not gate implementation.

All generated tasks are required; P2 never means optional. The control schema also accepts P3, but this workflow's priority policy remains P0–P2. Supply root-approved candidates to `task_ctl plan '<task-id>' '<candidate-list-as-yaml>'` in parent requirement order. Each candidate supplies a precise lowercase kebab-case `slug`, `type`, `priority`, and an explicit `blocked-by` list. For an edge to another proposed candidate, use its slug as the planning input's `task-id`; the script replaces it with the exact generated task ID. Use canonical IDs for existing external prerequisites. Although `remarks` is optional in the general control schema, this planner MUST supply the output-specific rationale required by the dependency-minimization gate for every edge.

The script returns dependency-first order, breaking ties by priority and then input order, assigns orders starting at 1 and `<NN>-<slug>` destinations, and reports occupied destinations. Preserve its returned IDs, orders, source paths, and blocker records verbatim. Do not hand-sort, allocate numbers, or infer dependencies from prefixes. This is a stable display/scheduling order, not a serial execution mandate: numbering or priority never adds blockers. Independent consumers may share a genuine prerequisite but have no mutual blockers; tasks with no genuine prerequisites keep empty lists.

Use short slugs describing delivered behavior, not `setup`, `misc`, `changes`, or `cleanup`. Never silently overwrite an occupied destination. If it is the same source and outcome, revise carefully; otherwise choose a distinct precise slug, rerun `plan`, and report the collision. Equivalence is a root judgment; collision detection is script-owned.

### 5. Write one self-contained mini-spec per task

Only after the root passes the breadth/depth, HLD admission, and dependency-minimization gates and freezes the complete script-returned assignment table, type-readiness rationale, requirements, and acceptance does `boss-builder-fast` (`@smol`) write mini-spec prose and invoke `task_ctl set` for each task's control. The writer cannot redesign tasks, change types or blockers, allocate numbers, or directly write `task.yml`.

Each mini-spec is either a direct implementation contract (`impl`) or a bounded input to a later design/decomposition pass (`hld`). It must be understandable without reading the parent specification, fixme file, conversation, or sibling task specs. Repeat the few shared decisions needed by the task instead of saying “follow the parent spec” or “same as the previous task.” References to blockers provide sequencing, not missing requirements.

Do not reintroduce hidden dependencies in prose, references, or acceptance criteria: no “after the previous task,” reliance on an unfinished sibling as the only implementation recipe, or unrelated sibling-completion gate. Carry the frozen shared contract and owned edit scope into each independent leaf. If writing reveals an actual missing prerequisite or overlapping mutation contract, report it to the root for resolution and a revised frozen graph; do not silently add sequencing.

Create or update each child's control with `task_ctl set '<exact-child-id>' --type '<impl|hld>' --status new --order '<integer>' --priority '<P0|P1|P2>' --blocked '<YAML-list-of-task-id-and-optional-remarks-records>' --source '<parent-spec-path>'`. Use the frozen table's values verbatim. Preserve a revised existing task's lifecycle unless the root explicitly determines the revision invalidates that state; never reset completion incidentally. A whole configuration may instead be passed to `set --all`; this is script input, never file content written by the model.

Use this shared Markdown structure; omit only body sections that truly do not apply. Do not duplicate type, status, order, priority, blockers, or source as Markdown control headers. Keep any substantive priority rationale in relevant requirements, not a second metadata field. Apply the type-specific rules below.

```markdown
# <Task title>

## Outcome

<One short paragraph describing the complete, observable result.>

## Scope

- <Precise behavior this task must implement or design for later decomposition.>
- <Inputs, outputs, state transitions, errors, or compatibility rules needed here.>

## Implementation references

- **Modify:** `path/to/file` — `<symbol or section>`; <why it is the touchpoint>.
- **Read:** `path/to/analogue` — `<symbol or section>`; <specific convention to reuse>.
- **Tests:** `path/to/test` — <existing suite, fixture, or nearest pattern>.

## Requirements

- <Normative, unambiguous requirement.>
- <Relevant boundary or failure behavior.>

## Non-goals

- <Nearby work explicitly excluded from this task.>

## Acceptance criteria

- [ ] <Observable criterion that distinguishes success from failure.>
- [ ] <Relevant negative or boundary criterion.>

## Verification

- `<specific existing test/build/format command>`
- <Any focused manual or runtime scenario required to observe the result.>
```

Mini-spec writing rules:

- Choose the writing contract by each child's Type, not by the invocation mode. An `impl` child generated in Design mode is still ready for direct implementation: use `Implementation references`, implementation acceptance, and executable verification; never require another planning pass or use design-completion acceptance for it.
- For `hld`, rename `Implementation references` to `Design references`; cite verified component/interface touchpoints and useful analogues, not speculative modify lists. Keep Outcome and Requirements about the eventual delivered behavior. Add `## Decomposition requirements` describing the next logical boundaries, internal decisions still to settle, contracts to preserve, and how to reach flash-sized leaves. Acceptance criteria must distinguish the completed design breakdown from eventual implementation acceptance and carry both forward. Verification describes checks of coverage, contracts, dependency order, and leaf readiness plus the eventual integration/runtime strategy; label implementation commands as future checks, not work executed by the design planner.
- An `hld` spec must explicitly require further planning before its own implementation. In `Decomposition requirements`, carry the original change root, this task's level, the 10–20 meaningful-child target, the expected direct `impl` leaf boundaries, and the requirement to finish within two levels below that root. Carry any justified breadth/depth exception and its evidence; nested planning MUST NOT reset the depth allowance. The next pass writes immediate child specs under this directory using the same template, normally all `impl`; an `hld` child requires a fresh admission check and explicit depth justification, not automatic recursion. Only `impl` descendants can be implemented after blockers complete; siblings need not all reach leaves first. Do not pre-generate descendants or introduce a separate design-document format.
- Carry the dependency-minimization gate into every `hld` task's `Decomposition requirements`: independent descendant outcomes must have no mutual dependencies, shared prerequisites must have output-specific evidence, and planning/integration phases must not become implementation barriers. Preserve independence across HLD boundaries as well as within each group.

- Keep only information needed to implement or further decompose this task. Do not copy the entire parent specification.
- State concrete behavior, not vague directions such as “handle appropriately,” “support as needed,” or “update relevant files.”
- Name exact verified repository paths and important symbols. Mark a path as planned when the file does not exist yet. Include the nearest useful implementation and test analogues when they materially reduce searching.
- Explain current behavior only where the implementer or subsequent planner needs it to work safely.
- Preserve precise interface, format, error, ordering, lifecycle, and compatibility decisions from the source requirement, together with applicable acceptance criteria and verification obligations. Check against the original requirement set, not just the intermediate spec's summary; every relevant obligation must survive into the eventual leaf contract.
- Include non-goals that prevent likely scope drift, especially tempting complementary features or generalization.
- Keep code snippets out unless a compact type, schema, state machine, or algorithm fragment captures a decision more precisely than prose.
- Make acceptance criteria observable and automatable where possible. Include meaningful happy-path, negative, and boundary behavior proportionally; do not add a generic edge-case checklist.
- Name exact focused verification commands when the repository provides them. Do not prescribe a project-wide suite when a narrower command proves the task.
- Keep tests in the same task as non-trivial behavior. Do not create test-only busywork or test implementation details.
- Do not require an `impl` implementer to rediscover decisions, search broadly, or choose among unresolved alternatives. For `hld`, distinguish fixed boundary contracts from explicitly scoped internal design decisions delegated to the next planning pass.

### 6. Resolve only blocking ambiguity

The root resolves material ambiguity. For a key architectural choice it may use packet-only `boss-advisor`; a `NEED EVIDENCE` response routes one exact question to `csw-plan-facts`, after which the root decides.

Use repository evidence and established project conventions for factual and low-risk implementation details. If a source ambiguity materially changes behavior, task boundaries, or dependency order and cannot be resolved from the repository, ask one focused question at a time with a recommended minimal answer. Wait for the answer before writing affected mini-specs.

Do not ask the user to approve an otherwise clear breakdown. Do not manufacture choices, expand the product design, or turn decomposition into a design interview.

### 7. Final consistency pass

The root collects all phase results before consuming them, performs the final consistency pass, and verifies the focused generated artifacts without building or inspecting untouched implementation.

Before completing:

- account for every required source behavior exactly once, except intentional repetition needed to keep mini-specs self-contained;
- confirm completed behavior has no task and every remaining requirement has one;
- use `task_ctl plan` output and `task_ctl list '<task-id>' --full` to verify the generated set matches the frozen dependency/priority order; the script validates types, records, paths, and cycles rather than the model parsing metadata;
- repeat the dependency-minimization gate against the final controls and prose: every edge has a concrete required-output rationale; no hidden sequencing, unnecessary direct edge, or artificial transitive path orders independent tasks;
- explicitly check backend siblings, tensor-operation siblings, shared-file ownership, HLD ancestor blockers, and cross-cutting joins where present; all useful safe parallelism must survive, and every retained serial chain must be necessary rather than merely convenient;
- confirm each script-returned type satisfies the selected mode and leaf-readiness check;
- in Implementation mode, confirm all tasks are `impl`; in Design and Automatic modes, confirm refinement toward direct leaves was attempted and every retained `hld` passes the admission check; mode selection alone never justifies an HLD layer;
- enforce the breadth/depth gate: target 10–20 immediate children, permit fewer only with a concrete specification-shortage rationale, justify more than 20, and normally end all branches at level 1 or 2 relative to the original root;
- reject every one- or two-leaf HLD wrapper and every unjustified extra level; verify each HLD's projected useful leaf count, necessary remaining planning work, and shortest route to implementation before accepting the set;
- verify original requirements and settled contracts survive any intermediate specification and each HLD explicitly carries the original root, current level, breadth target, and remaining depth allowance;
- verify each mini-spec contains sufficient repository references and type-appropriate acceptance/verification; each `hld` has actionable decomposition requirements and preserves eventual implementation acceptance;
- remove duplicated work, scaffolding-only tasks, speculative improvements, and optional extras;
- confirm acceptance criteria collectively cover delivery of the parent specification and applicable fixme remarks, while distinguishing design completion from implemented behavior.

After all generated children and their contracts pass the consistency check, write or refresh `summary.md` beside the resolved parent `spec.md` as the final generated artifact. Summarize the effective task goal and scope, generation mode, immediate child count, original-root-relative current and expected leaf depth, and granularity rationale. Include each HLD's admission rationale and projected direct leaf count, concrete reasons for fewer than 10 or more than 20 children, and any justified depth exception with rejected shallower alternatives. Distinguish projections from generated descendants. Report material decisions, collisions, or residual risks. Include this table, with exactly one row per immediate child generated or revised by this invocation, in the verified script-returned order:

```markdown
# <Parent task title> — planning summary

<Brief task goal, scope, effective mode, and decomposition rationale.>

## Parallelism and dependency rationale

<Identify tasks that can proceed concurrently, their genuine shared prerequisites if any, and the required outputs justifying remaining serial chains and joins. Describe opportunities, not execution waves: each task unlocks after its own blockers, without waiting for unrelated branches. This is a derived explanation of the verified graph, not additional dependency metadata.>

## Generated tasks

| Order | Task | Type | Priority | Dependencies | Description |
| --- | --- | --- | --- | --- | --- |
| <order> | [<NN>-<task-slug>](<NN>-<task-slug>/spec.md) | <impl or hld> | <P0, P1, or P2> | <canonical blocker IDs and required outputs, or None> | <One-line outcome of this task.> |
```

Derive order, task IDs, types, priorities, and dependencies from the verified `task_ctl` output, and descriptions from the final mini-specs; never parse `task.yml` or invent new requirements for the summary. Preserve exact canonical blocker IDs, including external prerequisites, and summarize their required outputs from blocker remarks or the established dependency rationale. Link each task to its mini-spec relative to `summary.md`. Do not include unrelated existing children or ungenerated descendants. On a successful replan, refresh the summary to describe the current invocation rather than appending stale rows. When no work remains, retain the table headers without a placeholder task row and explain why no tasks were generated, with repository evidence.

Verify that `summary.md` exists at the resolved target, its rows and dependencies match the verified generated set, its task links resolve, and its descriptions agree with the mini-spec outcomes. Do not publish a success summary for incomplete generation. Only after the generated children and summary pass these checks, run `task_ctl set '<parent-task-id>' --status planned`. Do not mark the parent planned when no children were generated or when generation remains incomplete. Planning never marks implementation done or satisfies a dependency.

## Completion response

Report only:

- the ordered task list with order, type, priority, directory path, and blockers;
- the path to the generated `summary.md`;
- one sentence giving the effective mode, immediate child count, expected original-root-relative leaf depth, and granularity/type rationale; explicitly explain fewer than 10 children or any depth exception;
- one sentence identifying the available parallel branches and genuine prerequisite/join boundaries; task numbering does not imply serial execution;
- any material ambiguity, collision, or residual risk.

Keep the response concise. The generated mini-specifications are the primary deliverable.

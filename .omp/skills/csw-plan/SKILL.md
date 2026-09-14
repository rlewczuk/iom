---
name: csw-plan
description: Split a change specification into dependency-ordered, prioritized implementation or design mini-specifications. Use only through /csw-plan <task-name>[/subdirectory] [--impl|--hld] or when explicitly requested.
hide: true
---

# CSW Plan

Turn one change specification into the smallest useful set of implementation or design tasks needed to deliver it. Write one focused mini-specification per task; do not implement the change. Implementation leaves must be safe for a cheap flash-class model; design tasks preserve cohesive components that need further decomposition.

The command supplies a path relative to `.cswd/tasks/` and an optional generation mode:

```text
<task-name>[/subdirectory...] [--impl|--hld]
```

For target `<target>`, use exactly:

- parent specification: `.cswd/tasks/<target>/spec.md`;
- user remarks, optional: `.cswd/tasks/<target>/spec-fixme.md`;
- generated task specifications: `.cswd/tasks/<target>/<NN>-<task-slug>/spec.md`.

The numeric prefix is part of the task directory name and records dependency order. The ordered directories plus the completion response are the task list; do not create a separate TODO or index file unless the user requests one.

## Task types and generation modes

Every generated `spec.md` MUST contain exactly one header field `**Type:** impl` or `**Type:** hld`:

- `impl` — a leaf task ready for direct implementation by an implementer agent, without further design or decomposition.
- `hld` — a high-level design for a cohesive component that must be split into subtasks before implementation. Its children may be `impl` or `hld`; plan it again with `/csw-plan <target>/<NN>-<task-slug>`.

Resolve the mode after applying conversation and fixme precedence:

| Invocation / effective parent header | Generation mode |
| --- | --- |
| Explicit `--hld`, regardless of parent type | Design |
| Explicit `--impl`, regardless of parent type | Implementation |
| No flag, parent `**Type:** hld` | Design |
| No flag, parent `**Type:** impl` or no Type header | Automatic |

An explicit flag overrides the parent type; a parent marked `impl` does not force implementation mode. Older parent specs without Type remain valid. Reject an unsupported or duplicate parent Type header instead of guessing.

- **Design:** treat the input as high-level design and split along component, responsibility, or contract boundaries. Generate mostly lower-level `hld` tasks; do not exhaustively flatten components into implementation leaves. A naturally small, fully decided component may be `impl` if it passes the leaf-readiness check below. Do not force a type quota or wrap an already leaf-ready input in a redundant `hld` task.
- **Implementation:** keep decomposing until every generated task passes the leaf-readiness check and is `impl`. Resolve necessary design decisions in this planning pass; never relabel complex work as `impl` just to satisfy the flag. If a material decision cannot be resolved, follow the ambiguity workflow before writing affected tasks; do not silently fall back to design mode.
- **Automatic:** first split logically along identifiable boundaries, without forcing leaf-sized pieces. Then assess each candidate independently: assign `impl` only when it passes the leaf-readiness check; otherwise assign `hld`. Do not recursively flatten every complex candidate in this mode.

Generate only the immediate children for this invocation, not a nested task tree. An `hld` child must narrow the parent's scope or establish concrete component contracts and a useful next decomposition; never copy the parent into an unchanged design wrapper. The type records readiness, not priority or implementation completion.

## Guardrails

- Require one non-empty target argument and at most one mode flag, `--impl` or `--hld`, in either position. Reject conflicting or repeated flags, unknown flags, and extra positional arguments before writing. Parse flags separately from the target; never include them in a task path. Reject absolute paths, `.` or `..` components, empty path components, backslashes, and any target that escapes `.cswd/tasks/`.
- Allow safe nested targets such as `allocator/gpu-layout`; the argument names the exact directory containing the parent `spec.md`.
- If the parent `spec.md` does not exist, report its expected path and stop. A missing `spec-fixme.md` is normal.
- Read the complete parent specification and complete fixme file when present.
- Treat direct conversation instructions as highest priority, then `spec-fixme.md` as user corrections or additions, then `spec.md`. For claims about current repository behavior, the repository is authoritative. Ask only when a material intent or policy conflict remains.
- Do not modify the parent `spec.md`, `spec-fixme.md`, implementation files, or unrelated task directories.
- Keep the design minimal. Exclude speculative abstractions, future-proofing, generic frameworks, optional extras, unrelated cleanup, and complementary features not required by the source specification.
- Do not invent functionality to make a task feel complete. Every requirement in every mini-spec must trace to the parent spec, the fixme remarks, a direct user instruction, or a repository constraint necessary to implement them correctly.
- Do not split work merely to produce more tasks. If the specification is already one cohesive, leaf-ready unit, create one `impl` task in any mode.
- Do not create tasks for work that is already complete and conforms to the specification.
- Do not implement the change. The deliverables are the task mini-specifications and the concise ordered list in the completion response.

## Boss orchestration contract

This seven-step workflow is one Boss process. Before starting it, the root MUST read `skill://boss`, run and preserve its deterministic preflight, then apply this skill's explicit override: substantive repository fact-gathering uses the read-only `csw-plan-facts` profile at `@task`, rather than Boss's default cheap exploration lane. The root should match the preflight-resolved `@slow` model; on mismatch, follow Boss's warning-and-consent gate and continue only if the user explicitly agrees. Use only the preflight's `roles`, `models`, and required `agents` entries for model routing, overrides, advisor state, availability, and profile tool restrictions; do not manually inspect or reconstruct them. Report a failed preflight and stop; never switch models, inherit an expensive parent, or silently fall back.

The `@slow` root owns intake, mode resolution, complete requirement accounting, decomposition, per-task type classification, priorities, the dependency DAG, architectural and ambiguity decisions, the frozen complete assignment table, verification, and final output:

- `[FACTS csw-plan-facts @task]` performs scoped, parallel, read-only factual discovery using exactly the profile's `read`, `grep`, `glob`, `lsp`, and `ast_grep` tools (`advisor: false`, `spawns: []`). It has no design authority, writing, delegation, or gates.
- `[DRAFT boss-builder-fast @smol]` writes frozen mini-specs and performs simple mechanical edits only after the root fixes destinations, types, order, blockers, requirements, and acceptance criteria. It has no design, type-classification, or numbering authority.
- `[PATH boss-errand @smol]` may perform mechanical metadata, path, existence, and collision checks against exact known paths; it MUST NOT perform substantive source discovery or design work.
- `[ADVISOR boss-advisor @advisor]` handles only genuinely key architectural decisions from supplied facts. It is tool-free and packet-only: it cannot search, fetch artifacts, edit, delegate, or run gates.

Fact packets MUST provide exact `file:line` and symbol evidence, decisive minimal excerpts, a `done`/`partial`/`missing`/`discrepant` mapping, relevant conventions and focused test commands, inspected and uninspected areas, gaps, and separate `SOURCE FACTS` from `INFERENCE`. Batch truly independent discovery and disjoint writers, collect each phase before consuming its output, and do not split output tasks merely for parallelism. No worker delegates recursively; workers skip gates, tests, linters, builds, and formatters.

Every brief is self-contained: it states exact scope and files, established facts and decisions, required output, non-goals, acceptance criteria, and any blockers. Advisor `CONTENT` must inline the requirements and non-goals, decisive excerpts and facts, constraints, alternatives with trade-offs, and one exact question; a path or URI alone is invalid. If the advisor returns `NEED EVIDENCE`, route the exact factual question back to `csw-plan-facts`, append only the evidence delta, and keep the final decision at the root.

This contract preserves the existing generation contract: precedence and path guards, minimal scope, omission of completed work, the task template, collision protection, self-contained mini-specs, ambiguity handling, and the concise completion response remain authoritative.

## Workflow

The seven stages below are Boss-owned; role dispatch supplements the stage and does not create a second workflow.

### 1. Read and normalize the requested change
The `@slow` root completes argument parsing, intake, precedence and mode resolution before any delegation, including all requirements, non-goals, assumptions, and unresolved decisions.

Read `spec.md`, then `spec-fixme.md` if present. Extract:

- required outcomes and observable behavior;
- explicit scope and non-goals;
- interfaces, data shapes, state transitions, errors, compatibility requirements, and constraints;
- named files, symbols, commands, tests, examples, and external or local references;
- every actionable fixme remark;
- unresolved decisions and assumptions.
- the parent Type header, if present, and the effective generation mode.

Build one coherent requirement set using the precedence rules above. A fixme correction replaces the conflicting parent requirement; do not preserve both alternatives. Do not propagate brainstorming, rejected alternatives, or editorial commentary as implementation work.

### 2. Ground the work in the repository
The root dispatches the exact `csw-plan-facts` (`@task`) profile for scoped parallel repository facts, then reconciles its evidence rather than delegating requirement or design authority.

Explore only the project areas needed to decompose and anchor the change. Read referenced files and enough surrounding implementation, call sites, tests, configuration, schemas, and project design documentation to establish:

- what is already implemented and correct;
- what is partial, missing, or inconsistent with the specification;
- the actual files and symbols each remaining change touches;
- local implementation and test conventions to reuse;
- existing facilities or extension points that avoid new machinery;
- dependency edges between pieces of work.

Classify each source requirement as **done**, **partial**, **missing**, or **discrepant**. Generate tasks only for the remaining work. Do not trust a path, symbol, signature, or current-behavior claim until checked when it affects a task.

Repository exploration is for reducing downstream search, not for proposing adjacent improvements. Ground `impl` tasks down to concrete implementation and test touchpoints; ground `hld` tasks at their component boundaries, existing facilities, interfaces, constraints, and verification strategy without inventing leaf-level detail. Ignore unrelated defects unless one directly blocks the specified behavior; if it does, include only the smallest necessary correction.

### 3. Decide whether and how to decompose

The root owns the decomposition judgment and any architectural decision. Consult `boss-advisor` only with self-contained packet `CONTENT`; never ask it to investigate the repository.

First identify cohesive outcomes and logical boundaries, then apply the selected mode. Every task must be:

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

Judge reasoning burden and coupling, not just file count, line count, or whether the prose fits in context. When uncertain, choose `hld` in Automatic or Design mode; in Implementation mode, refine the design and split further until the check passes.

For an `hld` task, state the component's eventual behavior, responsibilities, inputs/outputs, boundary contracts, constraints, non-goals, and integration obligations. Identify the remaining decomposition/design questions and the criteria its future implementation leaves must satisfy. Internal decisions may remain for the next planning pass only when they do not leave sibling contracts ambiguous. An `hld` task is not permission to implement a large change directly.

Prefer vertical slices for `impl` tasks that leave the repository in a coherent state. Include focused tests with the behavior they cover. Create a separate test task only for genuinely cross-cutting integration, end-to-end, migration, performance, or compatibility verification spanning multiple slices.

Create a prerequisite refactor task only when the current structure makes the required change unsafe or impractical. Keep it mechanical and no broader than necessary. For an unavoidable wide refactor, use the smallest dependency-safe expand–migrate–contract sequence; do not introduce compatibility layers otherwise.

Split independent outcomes, unrelated file groups, and different blockers when they form useful boundaries. In Implementation mode also split any candidate that fails leaf readiness; in Automatic or Design mode preserve a cohesive complex component as `hld` for later planning. Merge candidates when one only scaffolds another, neither is useful alone, or separating them duplicates substantial context.

If no implementation work remains, create no task directories and report that the specification is already satisfied, with the repository evidence that supports that conclusion.

### 4. Build and prioritize the task graph

The root owns priorities, the dependency DAG, and numbering. `boss-errand` may check exact destination metadata, paths, and collisions mechanically; it does not discover substantive source facts.

For every candidate task, identify only genuine blocking edges. A blocker is work whose output is required before the task can be planned, implemented, or verified; conceptual similarity is not a dependency. State which output is needed. Writing or decomposing an `hld` spec does not satisfy a blocker that requires its implemented behavior; that blocker remains until the required descendants are implemented and verified.

Assign a priority:

- **P0** — required prerequisite, public contract, migration, or correctness/safety work that gates broad progress;
- **P1** — required feature behavior on the normal implementation path;
- **P2** — required finishing work such as cross-cutting verification or documentation that does not gate implementation.

All generated tasks are required; P2 never means optional. Order tasks with a stable topological sort: blockers first, then higher priority, then the order requirements appear in the parent specification. When tasks are independent, say so instead of adding a false edge.

Number the sorted tasks from `01`. Use short lowercase kebab-case slugs that describe the delivered behavior, for example `01-load-manifest`. Avoid generic slugs such as `setup`, `misc`, `changes`, or `cleanup`.

Before writing, check every destination. Never silently overwrite an unrelated file or user-authored task specification. If a destination already contains a task for the same source and outcome, revise it carefully; otherwise choose a distinct precise slug and report the collision.

### 5. Write one self-contained mini-spec per task

Only after the root freezes destinations, types and their readiness rationale, order, blockers, requirements, and acceptance does `boss-builder-fast` (`@smol`) mechanically write the mini-specs. The writer cannot redesign tasks, change types, or allocate numbers.

Each mini-spec is either a direct implementation contract (`impl`) or a bounded input to a later design/decomposition pass (`hld`). It must be understandable without reading the parent specification, fixme file, conversation, or sibling task specs. Repeat the few shared decisions needed by the task instead of saying “follow the parent spec” or “same as the previous task.” References to blockers provide sequencing, not missing requirements.

Use this shared structure. All header fields, including Type, are mandatory; omit only body sections that truly do not apply. Apply the type-specific rules below.

```markdown
# <Task title>

**Order:** <NN>
**Type:** <impl|hld>
**Priority:** <P0|P1|P2> — <brief reason>
**Blocked by:** <task directory names, or “None”>
**Source:** `.cswd/tasks/<target>/spec.md`

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
- An `hld` spec must explicitly require further planning before its own implementation. Its next pass writes child specs under its own directory using this same template and mode rules; those immediate children may again mix `impl` and `hld`. Only `impl` descendants can be implemented, after their blockers complete; do not require all sibling branches to reach leaves first. Do not pre-generate descendants or introduce a separate design-document format.

- Keep only information needed to implement or further decompose this task. Do not copy the entire parent specification.
- State concrete behavior, not vague directions such as “handle appropriately,” “support as needed,” or “update relevant files.”
- Name exact verified repository paths and important symbols. Mark a path as planned when the file does not exist yet. Include the nearest useful implementation and test analogues when they materially reduce searching.
- Explain current behavior only where the implementer or subsequent planner needs it to work safely.
- Preserve precise interface, format, error, ordering, lifecycle, and compatibility decisions from the source requirement.
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
- verify each blocker points to an earlier generated task and no dependency cycle exists;
- verify numeric directory order matches the dependency and priority order;
- verify every generated header has exactly one Type with value `impl` or `hld`, and every classification satisfies the selected mode and leaf-readiness check;
- in Implementation mode, confirm all tasks are `impl`; in Design mode, confirm useful lower-level decomposition rather than forced flattening or unchanged wrappers; in Automatic mode, confirm complexity was assessed independently after the logical split;
- verify each mini-spec contains sufficient repository references and type-appropriate acceptance/verification; each `hld` has actionable decomposition requirements and preserves eventual implementation acceptance;
- remove duplicated work, scaffolding-only tasks, speculative improvements, and optional extras;
- confirm acceptance criteria collectively cover delivery of the parent specification and applicable fixme remarks, while distinguishing design completion from implemented behavior.

## Completion response

Report only:

- the ordered task list with order, type, priority, directory path, and blockers;
- one sentence explaining the effective mode and chosen granularity/type rationale, especially when the spec remained one task;
- any material ambiguity, collision, or residual risk.

Keep the response concise. The generated mini-specifications are the primary deliverable.

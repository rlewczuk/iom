# Candidate and Remediation Rubric

## Candidate packet contract

Each specialist returns zero or more candidate packets. Packets may be structured data or Markdown, but every packet must carry:

```markdown
### <AREA-ID> — <concise root-cause title>

- **Severity:** critical | high | medium | low
- **Verification:** verified | strongly-supported | hypothesis
- **Confidence:** <0-100>
- **Scope relation:** introduced by selected commit | materially exposed by selected commit | whole-codebase
- **Review scope:** <exact commit hash/baseline or whole-codebase reviewed-state identity>
- **Backend scope:** common | cpu | cuda | rocm | hip | vulkan | sycl | metal | opencl | multi-backend | <project backend>
- **Location:** `path/to/file.cpp:<smallest useful symbol or line/range>`
- **Invariant:** <what must remain true>
- **Failure mode:** <concrete runtime scenario or objective complexity/divergence mechanism>
- **Evidence:** <code path, test, tool, documentation, benchmark, profile, or duplication inventory>
- **Impact:** <observable consequence>
- **Remediation:** <smallest complete structural correction>
- **Verification method:** <exact test/tool/benchmark/scenario>
- **Affected symbols:** <current implementation, caller, counterpart, and test touchpoints>
- **Acceptance seed:** <observable conditions that distinguish success from the current problem>
- **Non-goals:** <nearby work excluded from the task>
- **Falsifier:** <evidence that would disprove the candidate>
```

A packet is an internal review handoff, not a report section. Include enough repository evidence and decisions for synthesis to produce a self-contained task without re-running a broad review.

## Compact evidence handoff

This section extends the packet above; it does not replace or relax any existing field. Keep evidence bounded and source-grounded:

- **Reviewed state and scope provenance:** identify the exact working-tree/commit state, selected-commit baseline and target when applicable, specification identity, and affected backend/workload scope.
- **Decisive excerpts:** provide the smallest useful `path:line`/symbol excerpts that establish the mechanism or contract. Do not paste whole files, whole diffs, or raw logs.
- **Source facts:** state what the code, test, benchmark, profile, or authoritative specification directly shows.
- **Inference and decision:** state the conclusion drawn from those facts, its verification class, and any assumptions; never present inference as a source fact.
- **Callers, guards, counterparts, and negative evidence:** name relevant call paths, capability/path-selection guards, backend/reference counterparts, existing tests or protections, and inspected evidence that argues against the candidate.
- **Search coverage:** state searches/areas inspected, sampled versus exhaustive coverage, and explicit uninspected areas.
- **Baseline-v-target provenance:** for selected-commit findings, identify the parent/target symbols, ranges, and behavior or evidence that changed; pre-existing observations are not findings.
- **Actual validation and gaps:** record gates actually run by the root, their exact results, unavailable tools, and remaining verification gaps. Proposed worker commands are not validation.
- **Falsifier and remediation seed:** make the disproof condition exact and keep the remediation structurally complete, including ownership boundaries, affected symbols, acceptance conditions, and non-goals.

Shared facts MAY be stored once in a compact evidence ledger and referenced by stable internal identifier in candidate packets. A packet must include its candidate-specific delta. When an advisor is consulted, the root MUST supply the advisor the needed packet content—including decisive excerpts, facts, inference, coverage, negative evidence, validation, and gaps—not merely a URI, path, or artifact reference. The advisor is not permitted to open references. Missing or uninspected evidence must be labeled as such and must never increase confidence or be silently treated as support.

## Area ID prefixes

- Contract & correctness: `CC-###`
- C++/GPU stability: `ST-###`
- Backend architecture & simplicity: `AR-###`
- Numerical correctness & tests: `NT-###`
- Performance: `PF-###`

IDs need only be unique within one review run. Synthesis resolves collisions before task materialization.

## Severity

### critical

Likely corruption, deadlock, widespread wrong inference results, security-relevant unsafe behavior, or catastrophic broad production failure.

### high

Material common-path correctness defect, race/lifetime issue, backend compatibility break, normal-path resource leak, or major measured/mechanically certain performance regression.

### medium

Bounded correctness/compatibility/performance defect, meaningful numerical gap tied to a failure mechanism, or significant structural complexity that threatens a concrete invariant.

### low

Localized real defect or an objectively demonstrated duplication/overengineering problem with limited current impact. Never use low for naming/style preference.

## Verification state

- **verified** — directly reproduced/measured or deterministically demonstrated from code semantics.
- **strongly-supported** — concrete mechanism and substantial evidence, but direct reproduction is unavailable.
- **hypothesis** — plausible material risk with insufficient evidence; requires a decisive verification task or method.

## Confidence

Confidence is independent of severity:

- `90–100`: direct evidence; alternatives implausible.
- `70–89`: strong evidence; minor unresolved assumptions.
- `50–69`: plausible with important assumptions; usually hypothesis.
- `<50`: normally reject unless impact is exceptional and falsification is cheap and important.

Confidence reflects supplied evidence and coverage, not missing work, advisor agreement, or the number of prose assertions.

## Root-cause and simplification rules

One cause with many symptoms becomes one task. Cross-reference secondary evidence rather than duplicating remediation.

A structural candidate is valid when it demonstrates current redundant responsibility, duplicated sources of truth, dead state, invalid combinations, unnecessary representation, special-case branching, or abstraction without an owned invariant. Its remediation must reduce net concept count and preserve behavior, failure attribution, async lifetime, and critical-path performance.

Do not accept an abstraction that only relocates code or future-proofs hypothetical requirements.

## Task materialization contract

Every accepted root cause becomes one evidence-only remediation `spec.md` using `templates/remediation-task.md`, plus a sibling lifecycle record written by `.omp/csw/bin/task_ctl`. The record uses `type: impl`, `status: new`, the assigned `order`, review `priority`, canonical `blocked-by` IDs, and the supplied parent `spec.md` as `source`. `Order`, `Type`, `Priority`, `Blocked by`, and `Source` must not be embedded in `spec.md`; use task_ctl's API/CLI for all path, metadata, ordering, and dependency operations.

A task must preserve the candidate's invariant, mechanism, evidence, backend scope, error/compatibility behavior, and verification burden while replacing review language with normative implementation requirements.

A task is not complete if it says to investigate, choose among designs, update relevant files, handle cases “as needed,” or read the review for missing context.
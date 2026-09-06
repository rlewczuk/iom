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

## Root-cause and simplification rules

One cause with many symptoms becomes one task. Cross-reference secondary evidence rather than duplicating remediation.

A structural candidate is valid when it demonstrates current redundant responsibility, duplicated sources of truth, dead state, invalid combinations, unnecessary representation, special-case branching, or abstraction without an owned invariant. Its remediation must reduce net concept count and preserve behavior, failure attribution, async lifetime, and critical-path performance.

Do not accept an abstraction that only relocates code or future-proofs hypothetical requirements.

## Task materialization contract

Every accepted root cause becomes one remediation task using `templates/remediation-task.md`. The task must preserve the candidate's invariant, mechanism, evidence, backend scope, error/compatibility behavior, and verification burden while replacing review language with normative implementation requirements.

A task is not complete if it says to investigate, choose among designs, update relevant files, handle cases “as needed,” or read the review for missing context.
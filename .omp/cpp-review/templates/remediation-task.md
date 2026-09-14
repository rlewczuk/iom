# <Action-oriented remediation title>

**Finding:** `<CC|ST|AR|NT|PF>-###`
**Review area:** <Contract & correctness | C++/GPU stability | Backend architecture & simplicity | Numerical correctness & tests | Performance>
**Review severity:** <critical|high|medium|low>
**Review verification:** <verified|strongly-supported|hypothesis>, confidence <0-100>
**Review scope:** <introduced by selected commit | materially exposed by selected commit | whole-codebase>
**Backend scope:** <common/backend list>
**Location:** `<smallest useful current path and symbol>`
**Review source:** `<cpp-inference skill>` — `<exact selected commit or whole-codebase reviewed-state identity>`

## Outcome

<One short paragraph describing the corrected observable result or, for a hypothesis, the decisive measured conclusion.>

## Current problem

<State the invariant, concrete runtime failure or objective complexity/divergence mechanism, evidence, and impact.>

## Scope

- <Precise behavior or structure this task must change.>
- <Affected backends, interfaces, state transitions, errors, or compatibility rules.>

## Implementation references

- **Modify:** `path/to/file` — `<symbol or section>`; <why it owns the change>.
- **Read:** `path/to/analogue` — `<symbol or section>`; <existing convention to reuse>.
- **Tests:** `path/to/test` — <suite, fixture, or nearest behavioral pattern>.

## Requirements

- <Normative, unambiguous implementation requirement.>
- <Required deletion or consolidation; identify the single mechanism/source of truth that remains when applicable.>
- <Relevant boundary, lifetime, ordering, cleanup, error, compatibility, or performance behavior.>

## Non-goals

- <Nearby work explicitly excluded to prevent scope drift or overengineering.>

## Acceptance criteria

- [ ] <Observable criterion that distinguishes remediation from the reviewed problem.>
- [ ] <Relevant negative, boundary, repeated-operation, cross-backend, or complexity criterion.>

## Verification

- `<specific focused build/test/static-analysis command>`
- <Required hardware, sanitizer, fault-injection, benchmark, or runtime scenario and expected observation.>

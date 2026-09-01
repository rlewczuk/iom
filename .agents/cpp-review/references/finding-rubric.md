# Finding Rubric

## Required finding fields

Each final finding must use this structure:

```markdown
### <AREA-ID> — <concise root-cause title>

- **Severity:** critical | high | medium | low
- **Verification:** verified | strongly-supported | hypothesis
- **Confidence:** <0-100>
- **Scope relation:** introduced by selected commit | materially exposed by selected commit | whole-codebase
- **Backend scope:** common | cpu | cuda | hip | vulkan | sycl | metal | opencl | multi-backend | <project backend>
- **Location:** `path/to/file.cpp:<smallest useful line/range>`
- **Invariant:** <what must remain true>
- **Failure mode:** <concrete scenario>
- **Evidence:** <code path, test, tool, documentation, benchmark, or profile evidence>
- **Impact:** <observable consequence>
- **Recommended fix:** <smallest structural correction>
- **Verification method:** <exact test/tool/benchmark that proves the correction>
```

For a whole-codebase review, `Scope relation` is `whole-codebase`.

## Area ID prefixes

- Contract & correctness: `CC-###`
- C++/GPU stability: `ST-###`
- Backend architecture & simplicity: `AR-###`
- Numerical correctness & tests: `NT-###`
- Performance: `PF-###`

## Severity calibration

### critical

Use sparingly for likely memory corruption, deadlock, widespread wrong inference results, security-relevant unsafe behavior, or production failure with broad reach.

### high

Material common-path correctness defect, race/lifetime issue, major backend compatibility break, significant resource leak under normal use, or major measured/mechanically certain performance regression.

### medium

Bounded edge-case correctness problem, limited backend/configuration failure, meaningful numerical/testing gap tied to a bug mechanism, or structural complexity likely to cause concrete maintenance/stability problems.

### low

Localized real defect or structural risk with limited impact. Never use `low` merely for style/naming preferences.

## Verification state

- **verified** — directly reproduced/measured or deterministically demonstrated from code semantics.
- **strongly-supported** — concrete failure mechanism with substantial evidence but no direct reproduction.
- **hypothesis** — plausible material risk with insufficient evidence; must contain a decisive verification method.

## Confidence

Confidence is independent of severity:

- `90–100`: evidence is direct and alternative explanations are implausible.
- `70–89`: strong evidence; minor unresolved assumptions.
- `50–69`: plausible but important assumptions remain; usually `hypothesis` unless deterministic reasoning is strong.
- `<50`: normally do not publish unless impact would be exceptional and verification is cheap/important.

## Root-cause rule

One cause with many symptoms is one finding. Cross-reference secondary evidence rather than creating multiple findings.

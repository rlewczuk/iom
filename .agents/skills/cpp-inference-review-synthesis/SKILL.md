---
name: cpp-inference-review-synthesis
description: Verify, reconcile, deduplicate, prioritize, and format candidate findings from C++ multi-backend inference-engine reviews. Separates severity, confidence, and verification state; rejects weak/speculative findings; produces the six-area final review. Use independently to reconcile review results or as Area 6 of cpp-inference-code-review.
argument-hint: "[candidate findings] [scope/spec context]"
---

# Review Synthesis & Adversarial Verification

Do not generate new broad review observations until the existing candidates have been challenged. The primary job is to **reduce false positives and duplicate symptoms** while preserving important uncertain risks as explicit hypotheses.

Read `.agents/cpp-review/references/finding-rubric.md`, `.agents/cpp-review/references/review-process.md`, and `.agents/cpp-review/templates/review.md`.

## Verification states

Use exactly:

- `verified` — reproduced by test/tool/benchmark or proven by direct deterministic reasoning from the code path;
- `strongly-supported` — evidence is substantial and the failure mechanism is concrete, but direct reproduction is unavailable;
- `hypothesis` — material plausible risk requiring the stated verification experiment.

Do not convert disagreement or missing evidence into certainty.

## Adversarial gate for every candidate

Ask:

1. Is the claim inside the requested scope?
2. In selected-commit mode, was it introduced or materially exposed/worsened by the target commit?
3. Is the underlying assumption actually true on the affected backend/API?
4. Did another code path, guard, ownership rule, test, or synchronization point already account for it?
5. Can the finding name one concrete invariant?
6. Can it describe a plausible failure/performance scenario rather than a vague smell?
7. Is the evidence stronger than personal preference?
8. Is the severity based on observable impact rather than how ugly the code looks?
9. Is this just a symptom of another root cause?
10. Could a mechanical linter/compiler report it completely without engineering interpretation? If yes, suppress unless there is larger impact.
11. For a performance claim, is there measured evidence or a mechanically certain hot-path mechanism? If not, mark hypothesis and prescribe measurement.
12. What evidence would falsify this claim?

Reject candidates that fail the gate.

## Deduplicate by root cause

Merge findings when one causal defect explains multiple symptoms across backends/tools/areas. Preserve the strongest location and evidence, and list secondary consequences under `Impact` or `Evidence`.

Example: a missing queue dependency causing stale output, allocator reuse, sanitizer warnings, and performance recovery synchronization is one root cause, not four findings.

## Severity

Use:

- `critical` — likely memory corruption, exploitable unsafe behavior, widespread wrong results, deadlock, or catastrophic production failure with broad reach;
- `high` — material wrong results, race/lifetime defects, backend breakage, major performance regression, or common-path reliability failure;
- `medium` — bounded correctness/compatibility/performance defect, significant maintainability risk tied to a concrete invariant, or edge-case failure;
- `low` — real localized defect or structural problem with limited impact. Do not use `low` for style preferences.

## Confidence

Use integer `0–100` independently of severity.

High severity may have medium confidence; a trivial issue may have 100 confidence. Do not conflate the dimensions.

## Final ordering

Keep the six required areas in this order:

1. Contract & correctness
2. C++/GPU stability
3. Backend architecture & simplicity
4. Numerical correctness & tests
5. Performance
6. Synthesis / overall assessment

Within Areas 1–5, order findings by severity, then confidence. In Area 6 summarize cross-area root causes, residual unverified risks, validation gaps, and overall quality assessment. Do not create a second copy of every finding.

Use the exact report shape from `.agents/cpp-review/templates/review.md`. Run `.agents/cpp-review/scripts/validate_review.py` when writing a file.

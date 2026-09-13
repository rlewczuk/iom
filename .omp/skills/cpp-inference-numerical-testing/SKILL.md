---
name: cpp-inference-numerical-testing
description: Independently review multi-backend C++ inference numerical correctness and behavioral tests—reference parity, tolerances, precision, quantization, reductions, edge cases, differential coverage, and redundant test machinery—then emit direct remediation subtasks. Also serves as Area 4 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [operator/backend focus optional]"
---

# Numerical Correctness & Testing Review

The oracle must be stronger than “passes on one GPU.” Prefer independent reference/differential checks with an explicit numerical policy.

Read `skill://boss` first. Then read `.omp/cpp-review/references/review-process.md`, `.omp/cpp-review/references/finding-rubric.md`, `.omp/cpp-review/references/numerical-testing.md`, and applicable backend checklists. Follow the common routing, evidence-packet, synthesis, and verification contract in `review-process.md`.

## Boss routing contract

The canonical routing, evidence, scope, synthesis, and verification contract is `.omp/cpp-review/references/review-process.md`; this skill adds only numerical-specific routing and evidence jobs. The visible/root session must be configured as `@slow` because a skill cannot switch an already-running model. The root owns scope, routing, acceptance, the assignment table, and verification.

One `@slow` root schedules one shared `scout (project read-only) @smol`, atomic `boss-errand @smol` follow-ups, and bounded read-only `boss-reviewer @task` analysis/falsification. Area leaves do not delegate, invoke orchestration or synthesis, write tasks, or run gates. Dispatch independent work in one batch, use exact known ranges when cheaper than another dispatch, and do not make gratuitous calls.

### Invocation modes

- **Orchestrated:** use the supplied scope/map as a `boss-reviewer @task` leaf and return only `NT-###` candidate packets. Request factual follow-ups through the root; do not delegate, synthesize, or materialize tasks.
- **Standalone:** the `@slow` invocation owns the complete area, resolves scope, performs this pass, and always invokes `cpp-inference-review-synthesis` for adversarial verification and direct task output.

Workers may propose exact differential tests, benchmarks, or scenarios, but only the root runs gates after collection and records actual results and gaps. If a lane is unavailable, report routing/coverage limits and request permission before a materially costlier fallback; never silently replace cheap profiles or run an unbounded frontier review.

## Concrete evidence jobs

### Cheap `@smol` jobs

- Inventory affected operators, backend/reference implementations, conformance tests, tolerance policies, and fast/fallback paths; return exact symbols and sampled versus exhaustive search coverage.
- For one operator/path, map callers, dispatch and capability guards, backend counterparts, reference construction, input/output and accumulation dtypes, reduction order, and relevant negative evidence.
- Locate test/benchmark commands and report the exact shapes, strides/layouts, dtypes, backend/device selection, and baseline/target provenance needed to validate a candidate; propose commands only.
- For a selected commit, compare the target with its parent at the smallest decisive ranges and identify whether the behavior is introduced or materially exposed/worsened.

### Advisor hard-decision triggers

Use `boss-advisor @advisor` only for a genuinely hard semantic decision: disputed mathematical or operator contract; tolerance legitimacy under reduction order, fast math, accumulation precision, quantization, NaN/Inf, or scale; whether a reference is independent enough; conflicting backend/reference evidence; cross-area ownership; or a high/critical acceptance or remediation choice.

The advisor is tool-free and reasons only over a compact packet containing the needed source facts, exact minimal excerpts, inference, negative evidence, coverage, validation, and alternatives. It must never open a URI/path, search files, run commands, delegate, or supply missing repository facts. If facts are missing, it returns `NEED EVIDENCE` with one exact question; the root sends a cheap source-reading worker and returns only the evidence delta. Advisor agreement does not verify a claim and the supervisor remains accountable.

## Numerical model

For every affected operator/path determine:

- mathematical contract;
- input/output and accumulation dtype;
- reduction ordering sensitivity;
- fast-math and approximation behavior;
- quantization/dequantization rules;
- NaN, Inf, and subnormal expectations where relevant;
- deterministic versus intentionally nondeterministic behavior;
- capability-dependent implementation/fallback path.

Do not demand bitwise equality across valid parallel floating-point implementations. Do not accept one unexplained global epsilon.

## Reference and differential testing

Prefer:

```text
independent trusted/reference implementation
  ├─ CPU
  ├─ CUDA
  ├─ ROCm/HIP
  ├─ SYCL/Vulkan/other
  └─ backend-specific fast/fallback paths
        ↓
operator/dtype/shape-aware comparison
```

Check that the oracle is independent enough to catch backend-specific defects and that tests observe the intended backend path rather than accidentally validating fallback.

## Tolerance review

Justify tolerance using dtype, operation, accumulation precision, reduction length/order, output scale, quantization error, and backend math mode. Flag bounds wide enough to hide plausible regressions or strict enough to reject legitimate reordering.

## Edge and transition coverage

Inspect relevant:

- zero, small, odd, and very large dimensions;
- dimensions around tile/workgroup/index boundaries;
- non-contiguous strides, offsets, broadcasts, and alignment;
- mixed precision and quantized extrema;
- NaN/Inf behavior when contractual;
- batch/sequence extremes;
- capability boundaries and fast/slow paths;
- repeated operations and multiple hardware generations when behavior differs.

A permanent regression test is justified only when it fails for the plausible bug and defends observable behavior.

## Mandatory simplicity pass

Testing infrastructure also accumulates duplication. Look for:

- multiple tolerance policies for the same operator/dtype contract;
- copied backend test bodies instead of shared conformance behavior;
- reference implementations that duplicate the production algorithm and therefore share its bugs;
- parameter matrices that add rows without new boundaries or invariants;
- tests that assert wiring, source text, field forwarding, or incidental defaults;
- separate helpers/fixtures that model the same backend capability differently;
- obsolete golden data or compatibility paths.

Prefer one independent oracle, one explicit numerical policy, and shared conformance tests with backend-specific runtime drivers. Delete low-value duplicate tests rather than adding another layer.

## Candidate acceptance

Use IDs `NT-###` and the full packet in `finding-rubric.md`. Tie every test gap to a concrete unprotected behavior or plausible numerical defect. Include exact shapes/dtypes/tolerances/path-selection evidence, current test symbols, remediation, and falsifier.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
---
name: cpp-inference-numerical-testing
description: Independently review multi-backend C++ inference numerical correctness and behavioral tests—reference parity, tolerances, precision, quantization, reductions, edge cases, differential coverage, and redundant test machinery—then emit direct remediation subtasks. Also serves as Area 4 of cpp-inference-code-review.
argument-hint: "[whole codebase | commit <hash|message>] [spec path optional] [operator/backend focus optional]"
---

# Numerical Correctness & Testing Review

The oracle must be stronger than “passes on one GPU.” Prefer independent reference/differential checks with an explicit numerical policy.

Read `.agents/cpp-review/references/review-process.md`, `.agents/cpp-review/references/finding-rubric.md`, `.agents/cpp-review/references/numerical-testing.md`, and applicable backend checklists.

## Invocation modes

- **Orchestrated:** use the supplied resolved scope and return only `NT-###` candidate packets. Do not materialize tasks before cross-area synthesis.
- **Standalone:** resolve scope and optional task destination, perform this numerical/testing pass, then invoke `cpp-inference-review-synthesis` for final verification and direct task output.

In selected-commit mode, accept only problems introduced or materially exposed/worsened by the target commit.

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

Use IDs `NT-###` and the full packet. Tie every test gap to a concrete unprotected behavior or plausible numerical defect. Include exact shapes/dtypes/tolerances/path-selection evidence, current test symbols, remediation, and falsifier.

When standalone, always finish through `cpp-inference-review-synthesis`. If no candidate survives, return `No material findings; no remediation tasks generated.` with coverage and verification limits.
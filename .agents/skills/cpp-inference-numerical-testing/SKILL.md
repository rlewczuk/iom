---
name: cpp-inference-numerical-testing
description: Review multi-backend C++ inference-engine numerical correctness and tests: reference parity, floating-point tolerance, mixed precision, quantization, reductions, edge cases, backend differential tests, and regression coverage. Use independently or as Area 4 of cpp-inference-code-review.
argument-hint: "[scope] [operator/backend optional]"
---

# Numerical Correctness & Testing Review

The test oracle for a multi-backend inference engine must be stronger than “passes on one GPU”. Prefer reference/differential testing across supported implementations with explicit numerical policy.

Read `.agents/cpp-review/references/numerical-testing.md`, `.agents/cpp-review/references/finding-rubric.md`, and relevant backend checklists.

## Numerical model

For every affected operator/path determine:

- mathematical contract;
- input/output dtype;
- accumulation dtype;
- reduction ordering sensitivity;
- fast-math/approximation behavior;
- quantization/dequantization rules;
- NaN/Inf/subnormal expectations where relevant;
- deterministic vs intentionally nondeterministic behavior.

Do not demand universal bitwise equality across CPU/GPU/backends. Do not accept one global epsilon without justification.

## Differential/reference testing

Prefer a structure like:

```text
trusted/reference implementation
  ├─ CUDA
  ├─ HIP
  ├─ Vulkan
  ├─ SYCL/other
  └─ CPU/vector backend
        ↓
operator/dtype/shape-aware comparison
```

Check whether the reference itself is independent enough to catch backend-specific defects.

## Tolerance review

Tolerance should be justified by some combination of:

- dtype;
- operation type;
- accumulation precision;
- reduction length/order;
- expected output scale;
- quantization error;
- backend math mode.

Flag tests whose tolerance is so wide that meaningful regressions pass, or so strict that valid parallel floating-point reorderings fail.

## Edge-case matrix

Inspect coverage for relevant cases:

- zero/small/odd dimensions;
- dimensions around tile/workgroup boundaries;
- non-contiguous tensors and unusual strides;
- broadcasts;
- alignment-relevant offsets;
- very large sizes/index limits;
- mixed precision;
- quantized extrema;
- NaN/Inf when contractually meaningful;
- batch/sequence extremes;
- capability-dependent fast/slow paths;
- multiple hardware generations where behavior differs.

## Cross-backend coupling

When one backend implementation changes, ask whether:

- shared operator tests run against all registered backends;
- unsupported cases are explicitly skipped with a capability reason;
- a changed capability predicate expands the required test matrix;
- fallback is being tested instead of the intended accelerated path;
- tests verify actual backend assignment/execution when that matters.

## Regression test acceptance

A test is valuable when it demonstrates the invariant violated by a plausible bug and would fail before the fix. Avoid demanding tests merely to increase line coverage.

Use IDs `NT-###`.

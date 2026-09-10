# Assert CUDA and ROCm broadcast-tail values

**Order:** 05
**Priority:** P2 — required focused falsification/coverage of a concrete mapping branch
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** NT-002
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** hypothesis, confidence 87
**Review scope:** whole-codebase
**Backend scope:** CUDA and ROCm standard-tiled GPU paths
**Location:** `test/backend/backend_conformance_add_gpu.hpp` — `run_gpu_eltwise_mapping_conformance`; production counterpart `src/shared/standard_tiled_add.inl` — `binary_body`; drivers in `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp`

## Outcome

The focused CUDA/ROCm mapped case proves every logical output value for singleton-axis broadcasts and final tiled tails against the independent oracle, rather than proving only allocation size and token success. This task is a coverage hypothesis: it must establish whether a deliberate broadcast/tail mapping mutation is observable, without claiming that current production mapping is already wrong.

## Current problem

The invariant is that a successful mapped binary request returns the independent elementwise result at every logical coordinate, including singleton-axis broadcasts and final row/column tails. `run_gpu_eltwise_mapping_conformance` submits U8 `[2,1,33]` and `[1,17,1]` into `[2,17,33]`, waits, and checks only `read_logical(out).size() == out_spec.logical_nbytes()`. A wrong `lhs_brow`/`lhs_bcol`, plane index, or tail slot in `binary_body` can preserve output size and token success while returning wrong values. Equal-shape GPU coverage compares values but does not execute these broadcast branches. The shared CPU/oracle helper already computes the same independent U8 broadcast expectations, and TTNN mapped tests provide a positive value-check counterpart. Root validation passed CUDA/ROCm smoke/conformance/coexistence, but no mapping mutation or candidate-specific mapped-value run was performed; no current runtime failure is claimed.

## Scope

- Replace the size-only mapped U8 assertion with all-element independent expected-value comparisons for ADD, MUL, and SUB, covering both singleton axes and both final tails.
- Add a valid nonzero-pattern F32 DIV mapped case with existing one-ULP/special-class comparison; retain U8 DIV Unsupported coverage.
- Keep the mapped matrix small and operation-neutral, and preserve equal-shape all-dtype coverage separately.

## Implementation references

- **Modify:** `test/backend/backend_conformance_add_gpu.hpp` — `run_gpu_eltwise_mapping_conformance`; own the value-sensitive mapped assertion and independent packing/comparison reuse.
- **Read:** `test/backend/backend_conformance_add.hpp` — existing independent U8 broadcast expected values and `add_oracle::binary`; do not call production mapping or codec.
- **Read:** `src/shared/standard_tiled_add.inl` — `binary_body` broadcast flags, plane/tail coordinate mapping, and metadata fields that the test must observe.
- **Tests:** `test/cuda/test_cuda_conformance.cpp` and `test/rocm/test_rocm_conformance.cpp` mapped driver cases; preserve their U8 DIV Unsupported branch.

## Requirements

- Compare all `2*17*33` U8 values for ADD/MUL/SUB against independently calculated results using distinct patterned operands; include singleton row, singleton column, final row tail, and final column tail.
- Add F32 DIV with nonzero patterned operands and compare finite/special classes under the existing one-ULP policy; retain the intentional U8 DIV Unsupported assertion.
- The test oracle remains structurally independent from `binary_body`, `add_plane_slot`, and codec implementation. Do not enlarge every dtype matrix for this shape.

## Non-goals

- Do not claim a current production mapping defect without a failing mutation or runtime observation.
- Do not duplicate all dtype cases for every mapped shape, require bitwise finite floating equality, or force TTNN native mapping into the standard GPU harness.
- Do not change production mapping or common validation in this test task.

## Acceptance criteria

- [ ] CUDA and ROCm mapped ADD/MUL/SUB compare every logical U8 value, including both singleton broadcasts and both tiled tails; size/token-only success is insufficient.
- [ ] Mapped F32 DIV uses nonzero operands and independently checks values/classes under one ULP; U8 DIV Unsupported remains covered.
- [ ] A temporary mutation forcing one broadcast flag or tail coordinate false preserves output size/token success but fails the new logical-value assertion on both backends.
- [ ] The helper continues to use an independent oracle and does not duplicate the full conformance matrix.

## Verification

- `(remote-development: CUDA host)` build/run the focused CUDA mapped conformance cases, then apply a temporary broadcast/tail mapping mutation and confirm the value assertion fails before removing it.
- `(remote-development: ROCm host)` run the corresponding ROCm mapped cases and the same negative mutation; assert all logical outputs and existing Unsupported behavior.
- Root-supplied remote CUDA/ROCm smoke/conformance/coexistence passed. No candidate-specific mapped-value test, mutation, sanitizer, or profiler has run; because this is a hypothesis, the decisive output is the proof/falsification result rather than a claimed current runtime failure.

# Cover real-queue MUL, SUB, and DIV values

**Order:** 06
**Priority:** P2 — closes required cross-backend observable coverage
**Blocked by:** `03-NT-001-encode-inf-capable-overflow-as-infinity`
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** NT-003
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** strongly-supported, confidence 88
**Review scope:** whole-codebase
**Backend scope:** CPU, SYCL, and TTNN real queues
**Location:** `test/cpu/test_cpu_conformance.cpp` and `test/cpu/test_cpu.cpp`; `test/sycl/test_sycl_conformance.cpp`; `test/ttnn/test_ttnn_conformance.cpp`; production dispatch counterparts in `src/cpu/device.cpp`, `src/sycl/copy.cpp`, and `src/ttnn/copy.cpp`

## Outcome

CPU, SYCL, and TTNN real queues compare every supported MUL, SUB, and DIV result (floating DIV only) to one independent oracle, including noncommutative operand order and backend-specific traversal/staging cases. This is a missing-coverage hypothesis: the permanent tests must prove or falsify an operation-dispatch defect with a wrong-operation mutation, not claim that current production dispatch is already wrong.

## Current problem

The conformance invariant requires each enabled backend to read real-queue output and compare every addressed element for every supported operation. CPU's all-operation test submits ADD/MUL/SUB/DIV but checks only positive tokens, waits, and sequence behavior; its shared conformance call narrows data types and has no real-queue value helper. SYCL invokes only `run_add_value_conformance`. TTNN's detailed compact, wide, broadcast, transformed-view, tail, and alias numerical cases all submit `queue->add`; no real-queue MUL/SUB/DIV value calls were found. The production paths have distinct operation dispatch surfaces, so an ADD-for-MUL selection or SUB/DIV operand reversal could pass current backend gates. Existing root CPU tests and remote SYCL/TTNN smoke/conformance/coexistence passed, but no candidate-specific wrong-dispatch mutation or missing-operation value gate ran; no current runtime failure is claimed.

## Scope

- Generalize `run_add_value_conformance` into one operation-neutral real-queue value helper using the existing independent oracle, packing, special-class, and ULP policy.
- Invoke it for ADD/MUL/SUB and floating DIV on CPU, SYCL, and TTNN across supported operation/dtype leaves, with compact ordered pairs, wide boundary/special vectors, and representative noncommutative broadcast/tail/transformed cases.
- Upgrade or retire CPU's token-only arithmetic-positive test once value coverage owns arithmetic; retain protocol/lifetime checks separately. Preserve backend-specific storage, staging, ownership, and CUDA/ROCm helper behavior.

## Implementation references

- **Modify:** shared test helpers around `run_add_value_conformance`, `add_oracle::binary`, packing, and comparison; keep one independent oracle.
- **Modify:** `test/cpu/test_cpu_conformance.cpp`, `test/sycl/test_sycl_conformance.cpp`, and `test/ttnn/test_ttnn_conformance.cpp`; register real-queue operation cases in each backend driver.
- **Read:** `test/backend/backend_conformance_add_gpu.hpp` — existing all-operation independent-oracle helper and comparison policy.
- **Read:** `src/cpu/device.cpp` — `CpuQueue` operation selection; `src/sycl/copy.cpp` — operation-specialized `binary_elements`; `src/ttnn/copy.cpp` — `binary_planes` traversal and operation selection.
- **Tests:** CPU, SYCL, and TTNN conformance CMake/CTest targets; preserve existing ADD/storage/ownership/failure coverage.

## Requirements

- Each backend must submit every supported MUL/SUB/DIV leaf through its real queue and compare all logical output elements independently; skip only contractually unsupported integer DIV, BOOL/F8_E8M0, and non-NONE cases.
- Use reversed unequal operands for SUB/DIV, compact raw pairs, wide boundaries/specials, and at least one backend-path-specific broadcast/tail/transformed case. The helper must reuse corrected NT-001 overflow policy and oracle rather than inventing another.
- A temporary mutation selecting ADD for MUL or swapping SUB/DIV operands must fail the affected backend target; no production storage/staging or tolerance policy changes are part of this task.

## Non-goals

- Do not duplicate complete dtype-by-shape suites per backend, replace the independent oracle with production arithmetic, or change operation eligibility/tolerance.
- Do not alter CUDA/ROCm coverage, backend storage/staging, queue ownership, or operation implementation.

## Acceptance criteria

- [ ] CPU, SYCL, and TTNN each execute supported MUL/SUB and floating DIV real-queue cases and compare every logical result to the independent oracle.
- [ ] Reversed unequal operands expose SUB/DIV order; compact, wide special/boundary, and representative mapping/staging cases are covered without a copied test framework.
- [ ] A temporary wrong-operation mutation fails in the affected backend target while existing ADD, owner, wait, and storage tests remain valid.
- [ ] The independent overflow oracle/policy from `03-NT-001-encode-inf-capable-overflow-as-infinity` is in place before this cross-backend matrix is accepted.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — run the focused CPU operation-value cases; on configured SYCL and TTNN hosts run `iom_sycl_conformance_tests` and `iom_ttnn_conformance_tests` with the same scenarios.
- Apply temporary ADD-for-MUL and SUB/DIV operand-order mutations independently and require each affected backend target to fail; then remove mutations and run the complete enabled backend conformance targets.
- Root-supplied CPU build/focused tests and remote SYCL/TTNN smoke/conformance/coexistence passed. No candidate-specific wrong-dispatch mutation or operation-value gate has run; as a hypothesis, the permanent falsification result is required rather than a claim of present runtime failure.

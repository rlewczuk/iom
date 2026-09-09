# Cover GPU wide ADD dtypes

**Order:** 08
**Priority:** P1 — unprotected device codec
**Blocked by:** None
**Review source:** `cpp-inference-numerical-testing` — `whole-codebase checked-out main/HEAD ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb ("Update remote hosts"), clean working tree at review start`
**Finding:** NT-004
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** strongly-supported, confidence 92
**Review scope:** whole-codebase
**Backend scope:** CUDA, ROCm/HIP
**Location:** `test/backend/backend_conformance_add_gpu.hpp:44-193` — shared GPU ADD conformance helpers and `test/cuda/test_cuda_conformance.cpp:341-353` / `test/rocm/test_rocm_conformance.cpp:493-507` drivers

## Outcome

CUDA and ROCm GPU ADD conformance share one wide-dtype helper that checks I8/U8/I16/U16/I32/U32/I64/U64 and F16/BF16/F32/F64 on a `[1,33,17]` tail shape. Boundary, high-bit, overflow, subnormal, infinity, NaN, zero-sign, and finite one-ULP behavior are compared against an independent oracle, while the existing low-width and mapping helpers remain single, non-duplicated sources of their existing coverage.

## Current problem

`test/backend/backend_conformance_add_gpu.hpp` currently provides `run_gpu_add_low_width_conformance` for compact leaves and `run_gpu_add_mapping_conformance` for broadcast/tail/alias mapping, and the CUDA/ROCm drivers invoke only those helpers. No shared GPU ADD case exercises the standard wide integer leaves or F16/BF16/F32/F64 device arithmetic and codec paths. The existing tests therefore can pass while a device-specific wide integer or floating codec kernel is wrong, and duplicating separate CUDA and ROCm matrices would create divergent coverage.

## Scope

- Add one backend-neutral wide-dtype ADD conformance helper in `test/backend/backend_conformance_add_gpu.hpp` and invoke it from both CUDA and ROCm conformance drivers.
- Cover exactly I8, U8, I16, U16, I32, U32, I64, U64, F16, BF16, F32, and F64 with a `[1,33,17]` shape whose final dimensions exercise row and column tails.
- Keep compact low-width exhaustive coverage and existing broadcast/transform/alias mapping coverage in their current helpers; this task adds only wide dtype arithmetic and boundary values.

## Implementation references

- **Modify:** `test/backend/backend_conformance_add_gpu.hpp` — add one `run_gpu_add_wide_conformance(iom::Device&)`-style helper beside `run_gpu_add_low_width_conformance`; this is the shared owner for CUDA and ROCm wide arithmetic cases.
- **Modify:** `test/cuda/test_cuda_conformance.cpp` — CUDA ADD conformance driver near the existing low-width and mapping test cases; invoke the shared wide helper after device setup and preserve the traffic-gate assertion.
- **Modify:** `test/rocm/test_rocm_conformance.cpp` — ROCm ADD conformance driver near its existing low-width and mapping cases; invoke the same shared wide helper with no ROCm-only duplicate matrix.
- **Read:** `test/backend/backend_conformance_add.hpp` — independent `add_oracle` decoding/encoding and comparison conventions; expected values must not come from the GPU implementation.
- **Read:** `src/shared/scalar_add.hpp` and `src/shared/standard_tiled_add.inl` — production arithmetic and device implementation only as behavior under test, never as the expected-value source.

## Requirements

- Use one shared helper, called by both accelerator drivers, and do not add parallel CUDA- and ROCm-specific wide matrices. The helper must create independent LHS, RHS, and output tensors for each dtype, submit ADD, wait for completion, read back logical values, and compare each element with the independent oracle.
- Exercise exactly the twelve dtypes I8/U8/I16/U16/I32/U32/I64/U64/F16/BF16/F32/F64 with `TensorShape{{1, 33, 17}}`. Do not change the tail shape to a tile-aligned shape or hide tail errors with padded storage reads.
- Include deterministic boundary vectors for every dtype: zero, signed integer extrema and high-bit patterns for signed/unsigned integers, and for floating leaves positive/negative zero, minimum subnormal, maximum finite, overflow-producing sums, positive/negative infinity, and NaN. Include cancellation and sign-changing combinations where the encoding supports them.
- Require exact integer raw results under the existing modulo-width contract. For floating leaves, require exact special-value classes including infinity sign and signed-zero sign, and permit at most one type-aware raw-order ULP for finite results; use the same corrected policy as NT-002 rather than a host-value epsilon.
- Keep the expected values independent: derive them from the test oracle and raw logical encodings, never by invoking `detail::scalar_add`, a CUDA/HIP kernel, or a result read from another GPU backend.
- Do not copy or reimplement `run_gpu_add_low_width_conformance`'s compact enumeration or `run_gpu_add_mapping_conformance`'s broadcast/transform/alias cases. The new helper must report its own dtype and boundary case when an element fails.
- Retain asynchronous token and wait checks and ensure only logical elements are compared, leaving padding outside the assertion unless an existing storage-oracle case owns it.

## Non-goals

- Do not alter CUDA, ROCm/HIP, or shared production arithmetic, dtype mappings, kernels, queue/lifetime behavior, or public APIs.
- Do not move or duplicate the existing low-width compact matrix, broadcast/transform/tail mapping matrix, exact alias matrix, or CPU/SYCL/TTNN tests.
- Do not add unsupported dtypes, mixed-type ADD, tolerance wider than one ULP, or a production fallback to make a conformance case pass.

## Acceptance criteria

- [ ] Both CUDA and ROCm conformance drivers invoke one shared wide helper, and no second backend-specific copy of that matrix exists.
- [ ] The helper tests all twelve specified dtypes on `[1,33,17]`, observes logical row and column tails, and checks every logical output element against an independent oracle.
- [ ] Integer boundary/high-bit/overflow cases are exact; floating finite results are within one ULP only, and wrong NaN, infinity-sign, or signed-zero classes fail.
- [ ] The existing low-width and mapping helpers remain present and are not duplicated or weakened; the wide helper does not silently replace their distinct coverage.
- [ ] A deliberate device output perturbation, wrong special class, or tail-index mutation is detected by the focused CUDA and ROCm cases.

## Verification

- `cmake --build <configured CUDA build> --target iom_cuda_conformance_tests` and `cmake --build <configured ROCm build> --target iom_rocm_conformance_tests` — proposed focused builds; not run for this specification-writing task.
- **Remote CUDA:** use `.agents/skills/remote-development/scripts/remote-sync cuda <unique-task-id>`, then `.agents/skills/remote-development/scripts/remote-exec cuda <unique-task-id> 'cmake --build <configured-build> --target iom_cuda_conformance_tests && ctest --test-dir <configured-build> --output-on-failure -R "^iom_cuda_conformance_tests$"'`; expected observation is a passing CUDA run with all twelve wide dtypes and tail elements checked.
- **Remote ROCm:** use `.agents/skills/remote-development/scripts/remote-sync rocm <unique-task-id>`, then `.agents/skills/remote-development/scripts/remote-exec rocm <unique-task-id> 'cmake --build <configured-build> --target iom_rocm_conformance_tests && ctest --test-dir <configured-build> --output-on-failure -R "^iom_rocm_conformance_tests$"'`; expected observation is the equivalent ROCm result. Accelerator checks must run remotely; no local GPU result substitutes for them.

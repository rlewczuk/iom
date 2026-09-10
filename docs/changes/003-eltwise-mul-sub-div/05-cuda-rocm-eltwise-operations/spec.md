# CUDA and ROCm elementwise operations

**Order:** 05
**Priority:** P1 — generalizes the shared accelerator kernel and queue while preserving policy separation in both drivers.
**Blocked by:** `01-common-binary-boundary`, `02-scalar-binary-arithmetic`, `03-shared-eltwise-conformance`
**Source:** `docs/changes/003-eltwise-mul-sub-div/spec.md`

## Outcome

Generalize the one vendor-neutral standard-tiled GPU source and policy-templated queue so CUDA and ROCm execute ADD, MUL, SUB, and floating DIV with identical metadata/mapping/lifetime semantics. Keep runtime primitives isolated in each policy, remove duplicated ADD-only/raw paths, and exercise every enabled driver through its real queue.

## Scope

- Refactor `src/shared/gpu_queue.hpp:84-96,234-350,458+` to carry operation identity, neutral immutable metadata, owner registrations, queue outcomes, completion, retained failures, and checked pre-acceptance resource setup for all four operations.
- Refactor `src/shared/standard_tiled_add.inl:89-531` into the sole standard-tiled arithmetic source. Share format/packed mapping and kernel body; select an operation-specialized arithmetic entry point once, with no CUDA/HIP arithmetic copies.
- The GPU translation-unit path must not contain a second copied format decoder/encoder or arithmetic implementation: reuse the operation-neutral scalar codec from task 02 through device-compatible thin adapters, with only policy-specific launch/runtime primitives local to CUDA or ROCm.
- Update `src/cuda/copy.cu:48-56`, `src/cuda/copy.hpp`, `src/rocm/copy.hip:48-56`, and `src/rocm/copy.hpp` to dispatch the neutral path using separate CUDA/HIP policy runtime primitives. Keep one policy queue, one source body, and existing target names; adjust CMake only if source registration requires it.
- Update `test/cuda/test_cuda_conformance.cpp:305-370` and `test/rocm/test_rocm_conformance.cpp:450-540` to configure context/ordinal and invoke shared four-operation real-queue cases.
- Support ADD/MUL/SUB on `I2,U2,I4,U4,I8,U8,I16,U16,I32,U32,I64,U64,F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`; support DIV on `F4_E2M1,F6_E2M3,F6_E3M2,F8_E4M3FN,F8_E5M2,F16,BF16,F32,F64`, regardless of native SDK dtype availability.

## Implementation references

- **Modify:** `src/shared/gpu_queue.hpp:84-96,234-350,458+` — current policy queue/task/outcome and completion path.
- **Modify:** `src/shared/standard_tiled_add.inl:89-531` — current shared codec/metadata/kernel; this remains the only vendor-neutral implementation.
- **Modify:** `src/cuda/copy.cu:48-56`, `src/cuda/copy.hpp` — CUDA policy instantiation and declarations.
- **Modify:** `src/rocm/copy.hip:48-56`, `src/rocm/copy.hpp` — ROCm policy instantiation and declarations.
- **Modify:** `test/cuda/test_cuda_conformance.cpp:305-370`, `test/rocm/test_rocm_conformance.cpp:450-540` — runtime setup, context/ordinal activation, and driver coverage.
- **Read:** `test/backend/backend_conformance_add_gpu.hpp:23-331` — shared GPU harness from task 03.

## Requirements

- Stage all metadata, packed mapping, kernel launch resources, and bounded allocations before positive OID acceptance. A required supported leaf must not fail with Unsupported merely due to an absent CUDA/HIP SDK dtype.
- Preserve standard-tiled broadcast/view/row-column-tail/padding mapping, exact aliases and read overlap, no caller operand/output allocation or relocation, three-owner deduplication, cleanup exactly once, and repeatable retained post-acceptance failure.
- Preserve ordered OIDs, pre-acceptance no-effect/error mapping, queue usability after failures, context and device ordinal activation, and separate CUDA/ROCm runtime synchronization/visibility policies.
- Run arithmetic in the shared codec/domain: integer MUL/SUB modulo-$2^w$ without signed-overflow UB; floating one-step encoding with RNE, gradual underflow, special values, and operand order for SUB/DIV.
- Keep ADD numerical/mapping/lifetime/fault behavior green and use the shared conformance harness; do not add fallback APIs, global dispatch state, or a second queue.

## Non-goals

- Do not create CUDA-only or HIP-only arithmetic copies, vendor-specific kernel source, a universal traversal, or a generic fallback framework.
- Do not touch CPU, SYCL, or TTNN implementations.
- Do not change public signatures, validation precedence, dtype domains, ownership policy, or unrelated kernels/operations.

## Acceptance criteria

- [ ] CUDA and ROCm real queues execute all required ADD/MUL/SUB leaves and floating DIV leaves with independent-oracle values, including representative U8 MUL/SUB and F32 DIV.
- [ ] Context/ordinal activation, mapping/tails/transforms, aliases, owner lifetime, pre/post failure retention, repeat waits, and queue recovery are covered in both enabled drivers.
- [ ] Source review finds exactly one standard-tiled kernel/codec/mapping body and one policy-templated queue, with runtime-specific primitives confined to CUDA or ROCm policy code.
- [ ] Existing CUDA/ROCm ADD tests remain compatible and valid supported operations are not rejected because of SDK dtype coverage.

## Verification

- `cmake --build build --target iom_cuda_conformance_tests iom_rocm_conformance_tests` — not run per assignment.
- `ctest --test-dir build -R '^(iom_cuda_conformance_tests|iom_rocm_conformance_tests)$' --output-on-failure` — not run per assignment.
- Focused accelerator scenario on each configured host: interleave all four operations with two queues, inject pre/post failures, repeat waits, and verify retained owners and later queue work; not run per assignment.

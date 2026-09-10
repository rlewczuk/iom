# Unify the host and GPU binary scalar codec

**Order:** 07
**Priority:** P1 — deletes duplicate format/decode/encode/arithmetic semantics
**Blocked by:** `03-NT-001-encode-inf-capable-overflow-as-infinity`
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** AR-002
**Review area:** Backend architecture & simplicity
**Review severity:** medium
**Review verification:** verified, confidence 95
**Review scope:** whole-codebase
**Backend scope:** all backends through the host/emulation codec and CUDA/ROCm device instantiation
**Location:** `src/shared/scalar_add.hpp` — `Format`, `format`, decode/encode/arithmetic, `scalar_binary`; duplicate `src/shared/standard_tiled_add.inl` — `AddFormat`, `add_format`, `add_decode`, `add_encode`, `add_integer`, `add_value`

## Outcome

One compile-time shared semantic codec owns format classification, decode/encode, RNE and special values, integer low-bit arithmetic, and all four operation semantics. Host and CUDA/ROCm adapters provide only explicit carrier, math, and qualifier differences: host retains long-double/std behavior and device code remains device-callable with double/intrinsics. No runtime framework, transfer, allocation, or per-element dispatch is added, and the independent test oracle remains separate.

## Current problem

The invariant is one production semantic source for every DataType and operation while preserving backend-specific execution carriers. `scalar_add.hpp` defines `Format`/`format`, decode/encode, special-value/RNE rules, integer modulo, and operation arithmetic used by CPU/SYCL/TTNN. `standard_tiled_add.inl` independently defines `AddFormat`/`add_format`, `add_decode`, `add_encode`, `add_integer`, and `add_value` for CUDA/ROCm. The device copy is not a trivial alias: it uses `IOM_GPU_DEVICE`, device intrinsics, and double/math facilities, while the host uses long double/std. This makes blind inclusion invalid, but does not justify two mutable semantic policies. A future dtype or edge-case update can diverge across paths. Root scalar/public CPU overflow probing reproduced the pre-correction saturation policy; remote backend smoke/conformance/coexistence passed, but no device compile/candidate parity gate ran.

## Scope

- Create `src/shared/scalar_binary_codec.hpp` as the sole compile-time semantic implementation, parameterized by an operation tag and a minimal traits type that supplies the real carrier, math/bit primitives, and host/device qualifier behavior.
- Make `scalar_add.hpp` provide only the long-double/std host traits and public scalar wrappers, and make `standard_tiled_add.inl` provide only the double/device-intrinsic traits and kernel-facing wrappers. Delete the duplicate `AddFormat`/`add_*` semantic implementation while preserving backend traversal, packed storage, queue, and policy boundaries.
- Carry the corrected NT-001 overflow/infinity policy as the single semantic result; keep the test oracle independently implemented.

## Implementation references

- **Create:** `src/shared/scalar_binary_codec.hpp` — sole templated owner of format classification, decode/encode, RNE/special values, integer low-bit arithmetic, and operation-specialized semantics; the traits contract is limited to carrier type, primitive math/bit operations, and compilation qualifier.
- **Read:** `src/cpu/device.cpp`, `src/sycl/copy.cpp`, and `src/ttnn/copy.cpp`; preserve their existing scalar selection and backend traversal/staging.
- **Read:** `src/shared/gpu_queue.hpp`, `src/cuda/copy.cu`, and `src/rocm/copy.hip`; preserve operation submission, packed-word mapping, and CUDA/HIP policy launch boundaries.
- **Tests:** `test/backend/test_scalar_add.cpp`, `test/backend/backend_conformance_add.hpp`, and enabled backend conformance targets; `add_oracle` remains independent.

## Requirements

- `src/shared/scalar_binary_codec.hpp` must be the one compile-time implementation of DataType format mapping, decode/encode, RNE, special values, integer modulo, and operation-specialized arithmetic. Its traits surface is limited to carrier type, primitive math/bit operations, and host/device compilation qualification; do not introduce runtime dispatch or a general framework.
- Delete `AddFormat`, `add_format`, `add_decode`, `add_encode`, `add_integer`, and `add_value` as a second semantic source. Preserve device-callable compilation, host long-double behavior, CUDA/ROCm device double/intrinsic behavior, and the corrected inf-capable overflow policy.
- Keep storage/traversal, synchronization, ownership, validation, launch geometry, TTNN native storage, and independent oracle boundaries unchanged. Select operation outside loops as required by AR-003, not with a per-element function pointer.

## Non-goals

- Do not force long double/std facilities into device code, merge backend storage or queue semantics, or use the independent oracle as production code.
- Do not alter operation eligibility, numerical contract beyond NT-001, owner registration, launch behavior, TTNN native mapping, or unrelated operators.
- Do not retain two format tables or semantic implementations under renamed helpers.

## Acceptance criteria

- [ ] Source review finds one production codec/format/arithmetic semantic implementation instantiated for host and CUDA/ROCm; no duplicate `AddFormat`/`add_decode`/`add_encode`/`add_integer`/`add_value` body remains.
- [ ] NVCC/HIP and host compilation succeed without host-only long-double/std calls in device instantiations; explicit carrier/math/qualifier adaptation is compile-time and minimal.
- [ ] Scalar, CPU, SYCL, TTNN, CUDA, and ROCm raw-pair/conformance outputs retain corrected overflow, RNE, underflow, special-value, integer modulo, and operation behavior; the independent oracle remains separate.
- [ ] No runtime allocation, transfer, synchronization, metadata reconstruction, or per-element operation function pointer is introduced.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_scalar_add_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^(iom_scalar_add_tests|iom_backend_conformance_cpu_tests)$'` — then build both CUDA and HIP translation units remotely and run all enabled operation conformance with compact raw pairs, RNE ties, underflow, special values, F64 overflow, and all operation domains.
- Inspect source and generated device compilation to confirm one shared semantic owner and no duplicate `add_*` codec bodies; mutate one shared format/rounding rule and require every production instantiation to change while independent oracle checks detect it.
- Actual evidence is the supplied scalar/public CPU overflow reproduction and root CPU/remote smoke/conformance/coexistence passes. No candidate-specific codec cutover compile, parity mutation, sanitizer, profiler, or benchmark has run.

# Physical storage oracle reuses production tile-slot arithmetic

**Order:** 06
**Priority:** P1
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** NT-001
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** strongly-supported, confidence 90
**Review scope:** whole-codebase
**Backend scope:** multi-backend
**Location:** `test/backend/backend_conformance_oracle.hpp:71-180` (`standard_layout_view_slot` at `:101-102`, `encode_standard_tiled_storage` at `:140-142`, `apply_standard_tiled_view` at `:149-180`); duplicate CPU model `test/cpu/test_cpu.cpp:365-443` (`view_slot_at`/`StorageModel`); production/common mapper `src/iom.cpp:147-203` (`iom::detail::standard_plane_slot`, `iom::detail::standard_layout_slot`)

## Outcome

The physical-storage conformance oracle derives every expected physical byte from a test-only canonical 16×16 tile-slot encoder that shares no arithmetic with production mapping (`iom::detail::standard_layout_slot`/`standard_plane_slot`), so a production or coordinated helper/kernel tile-map regression fails physical conformance instead of being mirrored by the expected-value generator.

## Current problem

A physical-storage oracle must independently encode the canonical logical-to-physical 16×16 tile mapping; a production or shared implementation defect must not be reproduced by the expected-value generator.

The expected-value path in `test/backend/backend_conformance_oracle.hpp` delegates physical slot computation for transformed views (`:101-102`) and for every standard tiled storage slot (`:140-142`) to the production helper `iom::detail::standard_layout_slot`. The duplicate CPU `StorageModel` at `test/cpu/test_cpu.cpp:378` independently computes bit values but also calls that same helper. The helper's tile arithmetic is implemented in `src/iom.cpp:147-173`; CPU production traversal calls its counterpart `standard_plane_slot` (`src/cpu/device.cpp:215-218,313-321`), and accelerator kernels duplicate equivalent arithmetic (`src/shared/standard_tiled_copy.inl:76-90`).

A defect in the shared tile-row, tile-column, in-tile, or plane-slot arithmetic is therefore mirrored by the oracle, and logical host round trips use the same effective mapping, so a backend kernel carrying the same mapping defect can pass both its logical and physical conformance. `PermutingStorageOracle` (`backend_conformance_oracle.hpp:215-260`) only perturbs an already generated map; it does not establish that the expected map is independent. Impact: conformance can report green while validating a wrong physical tile layout, weakening the suite's ability to detect corruption in odd/padded dimensions, later tile rows/columns, and rank/plane combinations — precisely where physical storage is meant to be checked independently of logical round-trip cancellation.

## Scope

- Replace the expected-value path in `test/backend/backend_conformance_oracle.hpp` with one test-only canonical slot encoder that does not call `iom::detail::standard_layout_slot` or any other production mapper.
- Replace (or remove) the duplicate CPU `StorageModel` expected-value path in `test/cpu/test_cpu.cpp` with the same independent encoder where retained.
- Keep production layout semantics, logical host-byte contracts, transformed-view offsets/strides, and the deliberately perturbed-oracle negative fixture unchanged; keep the production helper's own unit checks intact.

## Implementation references

- **Modify:** `test/backend/backend_conformance_oracle.hpp` — `standard_layout_view_slot`, `encode_standard_tiled_storage`, `apply_standard_tiled_view` (`:71-180`); owns the shared expected physical map consumed by every backend conformance wrapper (CPU `:197-229`, CUDA `:278-290`, ROCm `:376-397`, SYCL `:650-661`, TTNN `:468-478`).
- **Modify:** `test/cpu/test_cpu.cpp` — `view_slot_at`/`StorageModel` (`:365-443`); duplicate CPU expected-value path must stop calling `iom::detail::standard_layout_slot` (`:378`).
- **Read:** `src/iom.cpp` — `standard_plane_slot`/`standard_layout_slot` (`:147-203`); the production semantics the encoder must independently encode (row-major owner-plane numbering, `ceil(rows/16)`/`ceil(columns/16)` tile coordinates, 16×16 in-tile coordinates, bit/byte placement for every supported leaf width).
- **Read:** `test/backend/backend_conformance_common.hpp` — `:38-125`; the existing independent host bit encoding to reuse as the companion of the new slot encoder.
- **Tests:** `test/test_iom.cpp` — hard-coded layout assertions (`:575-627,652-744`); keep as separate production-helper unit checks; `PermutingStorageOracle` (`test/backend/backend_conformance_oracle.hpp:215-260`) stays as the negative fixture.

## Requirements

- Implement one test-only canonical slot encoder in `test/backend/backend_conformance_oracle.hpp` that explicitly derives: row-major owner-plane numbering; `ceil(rows/16)` and `ceil(columns/16)` tile coordinates; 16×16 in-tile coordinates; and bit/byte placement for every supported leaf width (BOOL and the integer/float widths already produced by the host encoder).
- Remove every call to `iom::detail::standard_layout_slot` (and `standard_plane_slot`) from the expected-value paths of `backend_conformance_oracle.hpp` and the CPU `StorageModel`; the production helper remains the source of truth for production code only and keeps its own unit checks.
- Keep transformed-view offsets/strides handling and the deliberately perturbed-oracle negative fixture working against the new independent encoder.
- Coverage must include the existing matrices — `{16,16}`, `{1,17}`, `{17,33}`, `{16,32}`, `{17,16}`, `{2,3,17,33}`, rank-six `{2,2,2,3,17,33}`, and accelerator rank-boundary shapes — for every backend whose wrappers consume the oracle. Do not alter production layout semantics or invent compute kernels.

## Non-goals

- No compute/operator implementation, arithmetic tolerance policy, quantization support, or backend-specific performance change.
- No production tile-layout redesign or change to logical host-byte contracts; preserve real TTNN native-layout differences.

## Acceptance criteria

- [ ] No call from `test/backend/backend_conformance_oracle.hpp` or the CPU `StorageModel` expected-value path reaches `iom::detail::standard_layout_slot`/`standard_plane_slot`; expected bytes for every listed odd/padded/rank/view matrix come from the independent encoder.
- [ ] A deterministic test-only fault that changes one production/backend slot mapping at an unhardcoded padded tile coordinate (or one oracle coordinate) makes the physical conformance fail before logical round-trip checks can cancel it; correct production backends still match the independent full-storage bytes and logical readbacks, and CPU hard-coded layout tests (`test/test_iom.cpp:575-627,652-744`) remain intact and passing.

## Verification

`actual validation: none (read-only review); proposed gates below`.

Root-run fact (remote bv2 host, CPU-only, at review time): configure OK; `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` pass on the reviewed HEAD. No accelerator gates were run.

- CPU (any host): configure with testing enabled, build, then run `ctest --test-dir <build> -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure`.
- Accelerator backends (remote-host work per remote-development; CUDA/ROCm/SYCL/TTNN each require the corresponding hardware host): configure and build with the backend enabled, then run `ctest --test-dir <build> -R '^iom_(cuda|rocm|sycl|ttnn)_conformance_tests$' --output-on-failure`; expected: every registered backend wrapper passes storage and async-copy conformance against the independent oracle.
- Mutation gate: add a deterministic fault scenario that alters one production/backend slot mapping at an unhardcoded padded tile coordinate and require the independent physical oracle to fail that backend; no such gate was run during the review.

- `ctest --test-dir <build> -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure`
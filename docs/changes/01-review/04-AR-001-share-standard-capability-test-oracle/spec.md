# Share one independent standard capability test oracle

**Order:** 04
**Priority:** P1 — structural dedup, independent test policy preserved
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` — whole-codebase review of checked-out main HEAD `ff5e2ba32f21ffba6a5e05b2b9c2a01c050c0cdb` (`Update remote hosts`), clean tree at review start
**Finding:** AR-001
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** strongly-supported, confidence 92
**Review scope:** whole-codebase
**Backend scope:** multi-backend (CPU, CUDA, ROCm, SYCL test machinery; TTNN intentionally excluded)
**Location:** `test/cpu/test_cpu_conformance.cpp:153-174`; `test/cuda/test_cuda_conformance.cpp:219-241`; `test/rocm/test_rocm_conformance.cpp:267-288`; `test/sycl/test_sycl_conformance.cpp:360-378`; shared test include `test/backend/backend_conformance_common.hpp:1-103`

## Outcome

CPU, CUDA, ROCm, and SYCL capability conformance tests compare their backend-owned 23-entry standard span against one shared, independently maintained constexpr oracle. The values remain literal test policy rather than an alias of `standard_supported_data_types()`, so missing, extra, reordered, or changed production leaves still fail with index-specific diagnostics while four duplicate arrays disappear.

## Current problem

Each of the four standard backend conformance translation units repeats the same 23 `DataType` values and order, beginning with `BOOL/I2/U2` and ending with `F16/BF16/F32/F64`, then performs the same size and per-index comparison. `src/shared/standard_tiled_copy.hpp:13-18` identifies one immutable standard capability policy and explicitly expects test arrays to remain independent so narrowing or reordering is detected. The duplicated arrays create four maintenance points: a future edit can leave one backend's test oracle inconsistent, producing divergent coverage or false confidence. All four tests already include `test/backend/backend_conformance_common.hpp`; all four production factories return the shared standard table. Root-run CPU, CUDA, ROCm, and SYCL conformance gates passed, but no deliberate capability perturbation was run; current passing behavior therefore does not remove the source duplication risk.

## Scope

- Add one independent constexpr 23-entry standard capability oracle and one comparison helper in `test/backend/backend_conformance_common.hpp`.
- Replace only the four local arrays/loops in CPU, CUDA, ROCm, and SYCL standard capability tests, preserving backend construction, test names, size checks, and index diagnostics.
- Keep production capability declarations and the independence of expected values unchanged.

## Implementation references

- **Modify:** `test/backend/backend_conformance_common.hpp` — shared constexpr oracle and comparison helper; this becomes the sole owner of the duplicated standard test policy.
- **Modify:** `test/cpu/test_cpu_conformance.cpp`, `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, and `test/sycl/test_sycl_conformance.cpp` — delete only their local arrays/loops and invoke the helper after each backend's own `supported_data_types()` call.
- **Read:** `src/shared/standard_tiled_copy.hpp` — production `kStandardSupportedDataTypes/standard_supported_data_types`; do not reference it from the test oracle.
- **Read:** `src/ttnn/device.cpp` and `test/ttnn/test_ttnn_conformance.cpp` — TTNN's distinct 22-entry native mapping; preserve it unchanged.
- **Read:** `test/test_iom.cpp` — mutable fake full/shrunk capability fixtures; preserve their separate behavior.

## Requirements

- Define exactly one independently maintained `constexpr std::array<iom::DataType, 23>` containing the current literal ordered values: `BOOL`, I2/U2, I4/U4, I8/U8, I16/U16, I32/U32, I64/U64, F4_E2M1, F6_E2M3/F6_E3M2, F8_E4M3FN/F8_E5M2/F8_E8M0, F16/BF16, F32/F64.
- The helper must compare the candidate span size to 23 and check each index against the independent oracle with the existing REQUIRE/CHECK diagnostic style. It must not call or derive from `standard_supported_data_types()` or any production capability accessor.
- Replace all four local copies and loops, preserving each backend's own factory initialization and the existing test case name. No production source changes are permitted.
- Leave TTNN's 22-entry test and core fake-device mutation fixtures untouched; they represent deliberate capability differences and negative checks.

## Non-goals

- Do not change production capability tables, ADD support policy, backend representation/lifetime, TTNN native mapping, or unrelated capability coverage.
- Do not add a generated source, runtime synchronization, allocation, or broad test-harness abstraction.
- Do not replace the independent oracle with the production table, even though all four standard backends currently return it.

## Acceptance criteria

- [ ] The 23 expected values and order occur once in shared test machinery, and all four standard backend tests invoke the shared comparison while retaining backend-specific construction and test names.
- [ ] A missing, extra, reordered, or changed advertised leaf fails the corresponding backend test with size or index-specific diagnostics; this remains true without consulting production table data.
- [ ] TTNN's 22-entry native capability test and `test/test_iom.cpp` fake full/shrunk capability fixtures remain unchanged and independently effective.
- [ ] Source search finds no second copy of the 23-entry standard oracle in the four backend conformance translation units and no helper call to `standard_supported_data_types()`.

## Verification

- `cmake -S . -B build/review-cpu -G Ninja -DBUILD_TESTING=ON && cmake --build build/review-cpu --target iom_cpu_conformance_tests && ctest --test-dir build/review-cpu --output-on-failure -R '^iom_cpu_conformance_tests$'` — CPU capability test passes and index diagnostics remain compiled.
- `(remote-development: CUDA host)` sync and run the README CUDA configure/build commands plus `ctest --test-dir build --output-on-failure -R '^iom_cuda_conformance_tests$'`; `(remote-development: ROCm host)` do the corresponding README ROCm commands and `^iom_rocm_conformance_tests$`; `(remote-development: SYCL host)` sync and run the configured oneAPI/icpx `iom_sycl_conformance_tests` target — all compare the shared independent oracle.
- In a throwaway test-only perturbation, remove/reorder one candidate advertised leaf and run each applicable conformance target; the test must fail on size/index mismatch. A source search must show one 23-entry literal oracle and no production accessor use.

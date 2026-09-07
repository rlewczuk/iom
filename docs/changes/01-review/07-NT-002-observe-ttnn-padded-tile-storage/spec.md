# TTNN physical oracle ignores native padded TILE storage

**Order:** 07
**Priority:** P1
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** NT-002
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** strongly-supported, confidence 88
**Review scope:** whole-codebase
**Backend scope:** ttnn
**Location:** `test/ttnn/test_ttnn_conformance.cpp:148-165,238-270` (`observe_plane_values`, `TtnnStorageOracle::observe`); callers `test/ttnn/test_ttnn_conformance.cpp:468-478`; production logical/padded transfer counterparts `src/ttnn/copy.cpp:78-116,211-280`

## Outcome

The TTNN storage oracle observes the complete native owner-plane TILE allocation — padded rows/columns and every owner plane — into an independent byte representation seeded with a nonzero padded-cell sentinel, so native padding writes or physical tile-placement regressions fail TTNN conformance instead of being hidden by logical-only readback.

## Current problem

The TTNN accelerator storage oracle must observe every native owner-plane TILE element, including padded rows/columns and untouched owner planes, when claiming complete physical-storage conformance; it must not substitute a zero-filled logical projection for unobserved native bytes.

`observe_plane_values` (`test/ttnn/test_ttnn_conformance.cpp:148-165`) reads `plane.to_vector<T>()`, requires only `values.size() >= rows * columns` (`:150`), and copies only logical `row < rows`, `column < columns` into the result (`:154-162`). `TtnnStorageOracle::observe` zero-initializes the full expected-sized result (`:226-227`) and runs that logical-only loop for each native plane (`:240-270`); padded extents are never copied or compared. Production upload allocates and zero-fills the padded host buffer (`src/ttnn/copy.cpp:78-116`), and download assembly copies only logical rows/columns (`src/ttnn/copy.cpp:211-280`), so nothing on the logical path exposes padded cells. If `to_vector<T>()` is semantic/logical rather than raw physical TILE order, its conversion further normalizes away native placement.

A regression that writes nonzero garbage into padded TILE cells, clears/overwrites padding unexpectedly, or changes native tile placement while preserving logical values can pass TTNN storage and async-copy conformance. CUDA/ROCm/SYCL oracles raw-copy the complete owner allocation (`test/cuda/test_cuda_conformance.cpp:203-211`, `test/rocm/test_rocm_conformance.cpp:328-336`, `test/sycl/test_sycl_conformance.cpp:343-351`), so this is a TTNN-specific oracle weakness. Impact: on the only backend with a native per-plane representation, the advertised full owner-storage/padding protection is not actually enforced.

## Scope

- Implement a TTNN-specific full-storage observer in `test/ttnn/test_ttnn_conformance.cpp` that reads the complete native owner-plane TILE allocation, including padded rows and columns, into an independent byte representation and compares every expected padded slot.
- If TT-Metal's `to_vector<T>()` is logical, use the native buffer/readback API that preserves physical TILE ordering; if it returns physical TILE order, require the full padded extent rather than only `rows * columns` and map all padded coordinates.
- Keep the view-plane mapping independent from `src/ttnn/copy.cpp` and preserve the existing logical readback checks.

## Implementation references

- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — `observe_plane_values` (`:148-165`) and `TtnnStorageOracle::observe`/`seed` (`:148-270`); owns the TTNN physical oracle and its padded seeding.
- **Read:** `test/{cuda,rocm,sycl}/test_*_conformance.cpp` — `StorageOracle::observe` raw owner-allocation copies (`test/cuda/test_cuda_conformance.cpp:203-211`, `test/rocm/test_rocm_conformance.cpp:328-336`, `test/sycl/test_sycl_conformance.cpp:343-351`); the complete-allocation convention to mirror.
- **Read:** `src/ttnn/copy.cpp` — `upload_plane`/`submit_download_plane`/`region_from_host`/`region_to_host` (`:78-280`); the padded-extent policy the oracle must verify (zero policy on upload) and the logical download path whose semantics must not change.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — oracle callers and supported-type/transformed-view suites (`:443-478`; supported leaves `:337-355`).

## Requirements

- The full-storage observer must cover: all native padded rows and columns of every owner plane; every owner plane, including untouched ones; all nine TTNN-supported leaves (BOOL, U8, I8, U16, I16, U32, I32, BF16, F32); and shapes `{16,16}`, `{1,17}`, `{17,33}`, `{16,32}`, `{2,3,17,33}`, `{2,3,16,48}`, `{17,16}`, `{1,1}`, rank-six `{2,2,2,3,17,33}`, plus transformed leading views.
- Seed padded cells with a nonzero sentinel so accidental zeroing/overwriting cannot cancel against the current zero-initialized expected result; the observer must report the exact expected untouched/zero policy per the documented upload behavior.
- Do not alter the public logical transfer contract; existing logical readback checks remain as-is.

## Non-goals

- No generic future operator coverage, floating-point tolerance policy, grouped-quantization support, CUDA/ROCm/SYCL oracle redesign, or change to TTNN logical host-transfer semantics; preserve legitimate TTNN native-layout differences.

## Acceptance criteria

- [ ] For every supported TTNN leaf and the exact odd/padded/rank/view shapes above, the observer compares all padded native rows/columns and all owner planes; a deliberate nonzero padded-cell mutation or native tile-slot permutation fails the oracle even when `copy_to_host` logical bytes remain unchanged.
- [ ] An unmodified implementation matches the documented zero/preservation policy for every transformed view, and the existing logical storage/async-copy conformance remains green.

## Verification

`actual validation: none (read-only review); proposed gates below`.

- TTNN (remote-host work per remote-development; requires TTNN hardware): configure/build with the TTNN backend enabled and run `ctest --test-dir <build> -R '^iom_ttnn_conformance_tests$' --output-on-failure`.
- Decisive scenario: seed one padded native row/column cell with a nonzero sentinel, perform a logical host write or transformed-plane copy, and require the full-storage observer to report the exact expected untouched/zero policy; intentionally corrupting a padded cell must fail while logical readback stays unchanged.

- `ctest --test-dir <build> -R '^iom_ttnn_conformance_tests$' --output-on-failure`
# Four standard-layout backends duplicate one 23-leaf capability source

**Order:** 16
**Priority:** P2
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** AR-003
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** strongly-supported, confidence 91
**Review scope:** whole-codebase
**Backend scope:** common, cpu, cuda, rocm, sycl
**Location:** `src/cpu/device.cpp:27-41 (kCpuSupportedDataTypes); src/cuda/device.cpp:19-41 (kCudaSupportedDataTypes); src/rocm/device.cpp:19-41 (kRocmSupportedDataTypes); src/sycl/device.cpp:29-51 (kSyclSupportedDataTypes)`

## Outcome

One immutable `constexpr std::array` in the shared standard-layout utility owns the 23-leaf capability list. The CPU, CUDA, ROCm, and SYCL `supported_data_types()` overrides return a read-only span into that single array, so the advertised leaf set and its order cannot drift between standard backends. The four independent per-backend test-expected arrays and TTNN's native nine-leaf mapping remain separate, so accidental narrowing, reordering, or divergence stays detectable.

## Current problem

Every standard 16x16 tiled backend must advertise exactly the set of unquantized leaf encodings its shared standard-layout transfer implementation can represent, and that set and its order must not drift accidentally. Today each of the four standard backends independently spells the identical ordered 23-entry `DataType` list — `BOOL, I2/U2, I4/U4, I8/U8, I16/U16, I32/U32, I64/U64, F4_E2M1, F6_E2M3/F6_E3M2, F8_E4M3FN/F8_E5M2/F8_E8M0, F16/BF16/F32/F64`:

- `kCpuSupportedDataTypes` (`src/cpu/device.cpp:27-41`), returned at `:435-438`;
- `kCudaSupportedDataTypes` (`src/cuda/device.cpp:19-41`), returned at `:97-101`;
- `kRocmSupportedDataTypes` (`src/rocm/device.cpp:19-41`), returned at `:82-86`;
- `kSyclSupportedDataTypes` (`src/sycl/device.cpp:29-51`), returned at `:111-115`.

Adding, removing, or renaming a leaf requires four production edits and can leave one enabled backend advertising a different capability while all four still consume the same common bit-width/layout machinery: `leaf_bits` (`src/iom.cpp:118-146`) covers all 23 leaves from the canonical `DataType` enum (`include/iom/tensor.hpp:13-34`), and the CPU and CUDA/ROCm/SYCL standard tiled transfer paths (`src/cpu/device.cpp:76-175, 502-675`; `src/shared/standard_tiled_copy.inl:370-445, 574-718`) share that same leaf-bit/layout model with no backend-specific type dispatch or subset. The per-backend conformance test that would catch a drift is built only when that backend is enabled, so backend configuration can hide divergence. Four backend tests do repeat independent expected 23-entry arrays (`test/cpu/test_cpu_conformance.cpp:153-183`, `test/cuda/test_cuda_conformance.cpp:218-250`, `test/rocm/test_rocm_conformance.cpp:266-306`, `test/sycl/test_sycl_conformance.cpp:359-399`) — these are useful regression checks and must be kept, not replaced by the production source. TTNN separately derives nine supported keys from its native mapping (`src/ttnn/device.cpp:49-107, 570-572`) and is materially different.

## Scope

- Add one `kStandardSupportedDataTypes` `constexpr std::array` (and a read-only span helper) to the existing common standard-layout utility `src/shared/standard_tiled_copy.hpp`; this becomes the single production source of the 23-leaf capability policy.
- Remove the four local `k*SupportedDataTypes` arrays; have CPU/CUDA/ROCm/SYCL `supported_data_types()` return the common span.
- Keep the `Device::supported_data_types()` virtual contract, each backend's factory, the four independent test-expected arrays, `test/test_iom.cpp`'s all-leaf oracle, and TTNN's native mapping unchanged.

## Implementation references

- **Modify:** `src/shared/standard_tiled_copy.hpp` — add `kStandardSupportedDataTypes` (`constexpr std::array<DataType, 23>`) plus a `span`-returning read-only helper; the existing common standard-layout utility is the natural owner.
- **Modify:** `src/cpu/device.cpp` — delete `kCpuSupportedDataTypes` (`:27-41`); `supported_data_types()` (`:435-438`) returns the common span. Same for `src/cuda/device.cpp` (`:19-41`, return at `:97-101`), `src/rocm/device.cpp` (`:19-41`, return at `:82-86`), and `src/sycl/device.cpp` (`:29-51`, return at `:111-115`).
- **Read:** `include/iom/tensor.hpp:13-34` — canonical `DataType` enum (the 23-leaf declaration); `src/iom.cpp:118-146` — `leaf_bits` covers all leaves.
- **Read:** `src/ttnn/device.cpp:49-107, 570-572` — `kSupportedToNative`/`kSupportedKeys` (nine native-mapped leaves; the separate counterpart that must stay).
- **Tests:** `test/cpu/test_cpu_conformance.cpp:153-183`, `test/cuda/test_cuda_conformance.cpp:218-250`, `test/rocm/test_rocm_conformance.cpp:266-306`, `test/sycl/test_sycl_conformance.cpp:359-399` — independent expected arrays retained so tests still detect an accidentally narrowed or reordered backend span; `test/backend/backend_conformance_copy_storage.hpp` — the shared 23-leaf parameterization driven by `supported_data_types()`.

## Requirements

- Exactly one production capability array remains: the common immutable `constexpr` array in `src/shared/standard_tiled_copy.hpp`, requiring no allocation, conversion, synchronization, or runtime query.
- All four standard `supported_data_types()` overrides MUST return spans pointing at that single array with identical value, order, and length (23 leaves); each backend's virtual contract and factory stay unchanged.
- Independent test expectations MUST NOT be weakened: the four per-backend expected arrays and the all-leaf oracle keep detecting a backend that returns a narrowed or reordered span.
- TTNN MUST continue to return exactly its nine native-mapped leaves; no TTNN behavior changes.

## Non-goals

- Do not derive production capability from test arrays or delete independent test expectations/the all-leaf oracle.
- Do not change `DataType` enum values, `leaf_bits`, quantization policy, TTNN support, or standard-layout/native transfer behavior.
- Do not add runtime capability discovery or backend registry machinery; no per-backend opt-out or subset is introduced in this task.

## Acceptance criteria

- [ ] CPU/CUDA/ROCm/SYCL `supported_data_types()` spans contain the same 23 values in the same order and point at the one common immutable array; `grep` shows no local `k<Backend>SupportedDataTypes` production arrays remain.
- [ ] CPU conformance and every enabled accelerator conformance suite still iterate all 23 leaves with full storage/copy matrices passing; TTNN conformance still iterates exactly its nine leaves.
- [ ] `test/test_iom.cpp`'s all-leaf oracle and the four independent expected-array tests pass unchanged.

## Verification

- `actual validation: none (read-only review)`; proposed gates below.
- CPU (local or remote per `remote-development`): `cmake --build <build> --target iom_cpu_conformance_tests` and `ctest --test-dir <build> -R '^iom_cpu_conformance_tests$' --output-on-failure`.
- CUDA, ROCm, SYCL on their respective remote hosts: `cmake --build <build> --target iom_cuda_conformance_tests iom_rocm_conformance_tests iom_sycl_conformance_tests` and `ctest --test-dir <build> -R '^iom_(cuda|rocm|sycl)_conformance_tests$' --output-on-failure`; run `iom_backend_coexistence_tests` when multiple backends are enabled.
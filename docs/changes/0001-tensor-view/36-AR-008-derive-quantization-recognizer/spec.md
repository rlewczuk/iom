# Derive quantization rejection from the supported boundary

**Order:** 36
**Priority:** P1 — required bounded common validation maintenance defect
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-008`
**Review severity:** low
**Review verification:** verified, confidence 80

## Outcome

`TensorSpec::validate()` has one authoritative supported quantization state (`QuantizationFormat::NONE`) and rejects every other declared format as unsupported without maintaining a second 24-entry vocabulary list. Values outside the declared enum remain invalid input, so callers retain the distinction between `std::invalid_argument` for invalid values and `std::runtime_error` for declared-but-unsupported formats.

## Current failure

`src/iom.cpp` maintains `is_recognized(QuantizationFormat)` as a switch duplicating every enumerator in `include/iom/tensor.hpp`. `TensorSpec::validate()` then unconditionally rejects all non-`NONE` formats; the switch only chooses the exception category. Adding a declared format without updating the switch misclassifies it as `std::invalid_argument` instead of the required unsupported-format `std::runtime_error`, while adding any format requires synchronized edits to two lists. The current contract in `include/iom/tensor.hpp`/the parent specification is that `NONE` is the only supported format, declared grouped formats are rejected before allocation, and invalid enum values are invalid input.

## Scope

- Change common quantization validation in `src/iom.cpp`, specifically the anonymous `is_recognized` helper and `TensorSpec::validate()`.
- Treat `QuantizationFormat::NONE` as the sole supported value; every declared non-`NONE` value continues to throw `std::runtime_error("grouped quantization formats are not supported")` before any size calculation or backend allocation.
- Retain `std::invalid_argument("unknown QuantizationFormat value")` for values outside the declared contiguous enum range, and retain invalid `DataType` behavior from `detail::leaf_bits`.
- Preserve all existing behavior for `logical_nbytes()` and `tiled_storage_nbytes()`, which validate before calculating sizes, and for CPU/CUDA/ROCm/SYCL/TTNN callers that rely on validation before construction.

## Implementation references

- **Modify:** `src/iom.cpp` — anonymous `is_recognized` and `TensorSpec::validate()`; remove the duplicated enumerator switch and implement a direct non-`NONE` unsupported check with a compact declared-range validity check.
- **Read:** `include/iom/tensor.hpp` — `QuantizationFormat` declaration (the authoritative contiguous `NONE` through `TT_BFP8A` vocabulary) and `TensorSpec::validate()` contract.
- **Read:** `docs/changes/0001-tensor-view/spec.md` — section 3 quantization contract: only `NONE` is supported; declared grouped formats are `runtime_error`, invalid enum values are `invalid_argument`.
- **Tests:** `test/test_iom.cpp` — `TensorSpec validate rejects every recognized grouped quantization format` and invalid-enum cases around lines 156–184; these are the existing validation-category tests to preserve unchanged.
- **Tests:** `test/cpu/test_cpu.cpp` — CPU tensor construction rejects grouped and unknown quantization before allocator activity (around lines 427–436), confirming the common validation boundary used by a backend.

## Requirements

- Remove `is_recognized` and do not add another per-enumerator switch/table. Use `quantization != QuantizationFormat::NONE` as the unsupported-format decision after validating whether the underlying value lies in the declared enum range.
- Keep the declared-range check independent of the supported check: an out-of-range cast must throw `std::invalid_argument`, while each declared non-`NONE` enumerator must throw `std::runtime_error` with the existing unsupported-grouped-format message.
- Ensure validation order remains unchanged: validate `data_type` first, then quantization; when both are invalid, `std::invalid_argument` remains observable from the data-type validation as covered by the existing test.
- Do not alter `QuantizationFormat` enumerator names or values, tensor size arithmetic, backend construction, allocation ownership, or exception behavior for any other validation rule.
- Preserve the existing tests and their coverage of all current non-`NONE` values; the implementation must continue to reject the full declared grouped-format range before allocation.

## Non-goals

- Do not add storage, scale metadata, packing, grouped layout, serialization, or computation support for any quantization format.
- Do not redesign the enum, add a public count/sentinel, or introduce a general enum-reflection framework; the fix is limited to eliminating the duplicated recognizer.
- Do not change backend capability tables, tensor allocation, SafeTensors parsing, or unrelated validation categories.
- Do not weaken or remove existing validation-category tests or convert unsupported grouped formats into successful tensor construction.

## Acceptance criteria

- [ ] There is no second 24-entry `QuantizationFormat` vocabulary in `src/iom.cpp`; validation derives the unsupported decision from `quantization != QuantizationFormat::NONE` and a compact validity boundary.
- [ ] Every declared non-`NONE` format still makes `validate()`, `logical_nbytes()`, and `tiled_storage_nbytes()` throw `std::runtime_error` before any allocation; `NONE` continues to validate successfully for every valid `DataType`.
- [ ] An invalid cast outside the declared quantization range still throws `std::invalid_argument`, and a spec with both invalid data type and grouped quantization still reports `std::invalid_argument`.
- [ ] Existing CPU construction and `test_iom.cpp` validation-category behavior remain unchanged.

## Verification

- `cmake --build cmake-build-debug --target iom_tests iom_cpu_tests`
- `ctest --test-dir cmake-build-debug --output-on-failure -R '^(iom_tests|iom_cpu_tests)$'`
- Confirm the focused `test_iom.cpp` cases exercise all declared non-`NONE` values as `std::runtime_error`, out-of-range enum values as `std::invalid_argument`, and `NONE` as accepted; confirm the CPU grouped-format case performs zero allocator events.

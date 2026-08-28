# Repair safetensors leaf dtype mappings

**Order:** 01
**Priority:** P0 — the stale enum names currently prevent the project from compiling.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

Safetensors metadata maps every accepted dtype string to the repository's authoritative `iom::DataType` leaf enumerator, and deterministic tests exercise the mapping without external model files or skip paths.

## Scope

- Replace the stale `F4` and `F8_E4M3` enum references while preserving the existing public `SafeTensorView::dtype()` API.
- Cover all accepted standard dtype strings with generated minimal safetensors files and reject every unrecognized or nonstandard string.

## Implementation references

- **Modify:** `src/safetensors.cpp` — anonymous `parse_dtype`; map `F4` to `DataType::F4_E2M1` and `F8_E4M3` to `DataType::F8_E4M3FN`.
- **Modify:** `test/test_safetensors.cpp` — replace dtype coverage that depends on `SAFETENSORS_TEST_DATA` with deterministic temporary-file cases for the parser's public behavior.
- **Read:** `include/iom/safetensors.hpp` — `SafeTensorView::dtype()` and `SafeTensorsFile`; reuse the existing public parsing surface rather than exposing `parse_dtype`.
- **Tests:** `test/test_safetensors.cpp` — existing doctest conventions and mapped-file construction.

## Requirements

- Accept `BOOL`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`, `F16`, `BF16`, `F32`, `F64`, `F8_E5M2`, `F8_E8M0`, `F6_E2M3`, and `F6_E3M2` as their same-named `DataType` values.
- Accept safetensors `F8_E4M3` as `DataType::F8_E4M3FN` and safetensors `F4` as `DataType::F4_E2M1`.
- Reject every other dtype string, including `I2`, `U2`, `I4`, and `U4`; the existence of those leaf types does not define nonstandard safetensors strings.
- Generate valid minimal headers, offsets, and payloads inside the tests. Dtype mapping coverage must not read a configured model path, silently skip, or depend on files outside the test process.
- Preserve existing file and directory behavior unrelated to dtype parsing.

## Non-goals

- Adding grouped `QuantizationFormat` metadata to safetensors.
- Adding new dtype strings or numeric conversion.
- Refactoring mapped-file or multi-shard directory behavior.

## Acceptance criteria

- [ ] Every accepted dtype string is observed through `SafeTensorsFile` as the exact required `DataType`.
- [ ] `F4` and `F8_E4M3` compile against the current enum and map to `F4_E2M1` and `F8_E4M3FN` respectively.
- [ ] Unknown and nonstandard sub-byte integer strings fail deterministically.
- [ ] Dtype tests run with no external model data and contain no skip branch.

## Verification

- `c++ -std=c++20 -Iinclude -DDOCTEST_CONFIG_IMPLEMENT_WITH_MAIN src/mmap.cpp src/safetensors.cpp test/test_safetensors.cpp -o /tmp/iom-safetensors-tests && /tmp/iom-safetensors-tests --test-case="SafeTensors dtype*"`

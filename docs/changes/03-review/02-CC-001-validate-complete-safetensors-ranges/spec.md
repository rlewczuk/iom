# Validate complete SafeTensors data coverage independently of key order

**Order:** 02
**Priority:** P0 — reject unindexed data and accept valid physical order
**Blocked by:** None
**Review source:** `cpp-inference-contract-correctness` — `whole-codebase reviewed state: branch main, clean HEAD 86533aef347935405cb86d465cc5489f5c3d530a (Remove review files.)`
**Finding:** `CC-001`
**Review area:** Contract & correctness
**Review severity:** medium
**Review verification:** verified, confidence 99
**Review scope:** whole-codebase
**Backend scope:** common SafeTensors parser and all callers
**Location:** `src/safetensors.cpp` — `SafeTensorsFile::SafeTensorsFile`

## Outcome

SafeTensors parsing validates physical ranges in a separate `(begin,end)`-sorted view, with a cursor starting at zero and ending exactly at `data_size`, while preserving original header iteration order for insertion and public keys. Valid nonlexical physical order, empty data, and valid zero-sized ranges are accepted; leading, intermediate, or trailing holes, a nonempty data buffer with no tensor ranges, overlaps, and out-of-range ranges are rejected before any records are inserted.

## Current problem

`SafeTensorsFile::SafeTensorsFile` iterates `header.items()` and compares each `begin` only with the preceding key's `end`. Since JSON object iteration is key-ordered, a physically contiguous file whose lexical key order differs from physical offset order is rejected. The same loop does not require the first range to begin at zero or the final range to end at `data_size`, so leading/intermediate/trailing holes and a nonempty data-only file are accepted. The official SafeTensors format requires the byte buffer to be entirely indexed, and the reference parser sorts by `data_offsets` before requiring exact cursor coverage. Existing insertion into `SafeTensorsStoreBase` makes key order observable and must not be replaced by physical order.

## Scope

- Parse every non-metadata tensor into a temporary record in original `header.items()` order, retaining name, dtype, shape, begin, end, and the data/view information needed after validation.
- Validate a separate index of those records sorted lexicographically by `(begin,end)` with `cursor = 0`; retain range bounds and payload-size checks, reject `begin != cursor`, advance the cursor to `end`, and require final `cursor == data_size`, including empty and zero-sized cases.
- Insert views into `base_` only after all validation succeeds and in original header iteration order. Preserve dtype/shape/offset overflow, lifetime, shard, and duplicate-key behavior.

## Implementation references

- **Modify:** `src/safetensors.cpp` — `SafeTensorsFile::SafeTensorsFile`, lines around the header loop and `base_.insert`; it owns common parsing, validation, and insertion.
- **Read:** `include/iom/safetensors.hpp` — `SafeTensorsStoreBase::insert`/`keys`; preserve current insertion-order behavior for the public keys API.
- **Read:** `src/mmap.cpp` and `include/iom/mmap.hpp` — mapped file/data lifetime; do not alter mapping or non-owning view semantics.
- **Read:** SafeTensors reference parser `read_metadata`/range validation; it is the format counterpart establishing sorted physical coverage and no holes.
- **Tests:** `test/test_safetensors.cpp` — existing mismatched/overlapping/out-of-order fixtures around `SafeTensorsFile rejects mismatched, overlapping, and out-of-order entries`; replace the faulty key-order assumption and add complete-coverage cases.

## Requirements

- Collection order MUST remain the original header iteration order, and no `SafeTensorView` may be inserted into `base_` until every tensor record has passed all range and payload validation.
- The validation view MUST be sorted by `(begin,end)` and walked with `cursor = 0`; reject leading gaps, intermediate gaps, overlaps, and trailing gaps, and reject a nonempty `data_size` when there are no tensor ranges. Retain `begin <= end <= data_size` and existing overflow/payload-size checks.
- Allow an empty data buffer and valid zero-sized ranges, including multiple zero-sized ranges sharing a boundary, when the sorted cursor still covers exactly `data_size`. Keep error behavior as stable `std::runtime_error` for invalid input.
- After validation, construct views from `data_begin + begin` and insert records in original key order. Do not sort or expose the physical validation order through `keys()`.

## Non-goals

- Do not change dtype mappings, shape/payload encoding, duplicate JSON-key policy, shard duplicate-key policy, mmap ownership, non-owning view lifetime, or backend materialization.
- Do not impose lexical, JSON insertion, or physical order on the public keys API beyond preserving its current header-iteration behavior.
- Do not redesign SafeTensors metadata, offset types, or parser abstractions beyond the temporary validation records required for this invariant.

## Acceptance criteria

- [ ] A valid fixture whose lexical key order differs from physical range order is accepted, returns exact payload bytes for every key, and preserves the original header iteration order in `SafeTensorsFile::keys()`.
- [ ] Leading-hole, intermediate-hole, trailing-hole, and nonempty-data-with-no-tensors fixtures throw before any store is externally usable; overlapping and out-of-range fixtures continue to throw.
- [ ] Empty data and valid zero-sized ranges, including boundary-sharing zero ranges, are accepted when complete coverage is satisfied, while ordinary contiguous files retain current behavior.
- [ ] No record is inserted before complete validation, and error results remain `std::runtime_error` without changing unrelated metadata, dtype, mmap, or shard behavior.

## Verification

- `cmake --build build --target iom_tests && ctest --test-dir build -R '^iom_tests$' --output-on-failure`
- Run the focused SafeTensors fixtures (or the repository's exact doctest filter) for nonlexical physical order, each hole position, empty/zero-sized coverage, overlap, out-of-range, and payload mismatch. Confirm payload bytes and `keys()` order on the valid nonlexical fixture; the root reproduction already demonstrated the pre-fix false rejection and hole acceptance, while no post-remediation gate has been run.
- Baseline validation ledger only (not CC-001 coverage): local GNU 15.2 CPU build/tests/bench/conformance passed 4/4; remote CUDA 13.2.78, ROCm HIP Clang 23, SYCL IntelLLVM 2026.1 (on two enumerated Arc Pro B60 Level Zero GPUs), and TTNN smoke+conformance each passed 2/2. The root throwaway probe reproduced the nonlexical-order rejection and hole acceptance; no post-remediation range-fixture gate has been run.

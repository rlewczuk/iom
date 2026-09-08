# Validate every native TTNN padded cell in conformance tests

**Order:** 04
**Priority:** P1 — close the physical-storage oracle gap without changing production transfer code
**Blocked by:** None
**Review source:** `cpp-inference-numerical-testing` — `whole-codebase reviewed state: branch main, clean HEAD 86533aef347935405cb86d465cc5489f5c3d530a (Remove review files.)`
**Finding:** `NT-001`
**Review area:** Numerical correctness & tests
**Review severity:** medium
**Review verification:** strongly-supported, confidence 93
**Review scope:** whole-codebase
**Backend scope:** TTNN
**Location:** `test/ttnn/test_ttnn_conformance.cpp` — `observe_plane_storage`; `TtnnPlaneLayout` and full-native readback path

## Outcome

The TTNN test-only physical oracle validates every cell in the complete native padded allocation, including native rows/columns beyond the standard 16x16-padded rectangle. U8 `{17,33}` exercises native cell `(0,48)` in the extra columns and U8 `{33,17}` exercises `(48,0)` in the extra rows; deliberate native-only mutations are detected, a logical host upload restores those cells to zero, and logical bytes remain exact.

## Current problem

`observe_plane_storage` verifies that the readback has the full native byte count but loops only through `owner.standard_padded_shape()`. For `{17,33}`, the native plane is 32x64 while the standard rectangle is 32x48; for `{33,17}`, native storage is 64x32 while the standard rectangle is 48x32. Thus native-only cells are downloaded but never inspected. The production `upload_plane` explicitly zero-fills through native padded extents, yet a stale native-only cell could pass the current logical readback and oracle. The current mutation test plants only standard padding at `(17,0)` and cannot demonstrate the missing region.

## Scope

- Change only `test/ttnn/test_ttnn_conformance.cpp`. Keep the existing standard-region mapping into canonical storage and add an independent native-allocation validator over every native coordinate outside the standard rectangle, requiring zero after a host upload.
- Extend the focused mutation scenario with U8 `{17,33}` native `(0,48)` and U8 `{33,17}` native `(48,0)`. Deliberately mutate each native-only cell, prove the independent check detects it, then call the production host upload and prove all native-only cells are zero.
- Preserve exact logical-byte and standard-storage assertions, retained-staging reuse coverage, native tile geometry, and the production `src/ttnn/copy.cpp:upload_plane` implementation as read-only reference.

## Implementation references

- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — `TtnnPlaneLayout`, `observe_plane_storage`, `TtnnStorageOracle::observe`, and the focused full-storage mutation test; these own the independent native readback and assertions.
- **Read:** `src/ttnn/copy.cpp` — `upload_plane`; production zero-padding behavior is the invariant under test and must not be modified.
- **Read:** `test/backend/backend_conformance_copy_storage.hpp` — storage-oracle shape/copy cases; reuse existing conformance conventions without changing the common oracle.
- **Read:** `src/iom.cpp` — standard 16x16 padding and `include/iom/tensor.hpp` — native handle exposure; distinguish standard canonical storage from TTNN native allocation.
- **Tests:** `test/ttnn/test_ttnn_conformance.cpp` — existing `TTNN conformance: full-storage oracle exposes native padding mutations` test and `TtnnStorageOracle` seed/observe paths.

## Requirements

- Inspect every native coordinate in `0 <= row < native.padded_rows` and `0 <= column < native.padded_columns`. Map standard-covered cells into the existing canonical storage exactly as today; independently assert that every cell outside the standard padded rectangle is zero after production host upload.
- Keep the validator independent of `upload_plane` and the common oracle's implementation. A helper/predicate MAY expose native-only failures without immediately asserting so the mutation test can prove detection before the upload, then assert zero afterward.
- Add both exact U8 shapes and coordinates: logical `{17,33}` with native extra-column cell `(0,48)` and logical `{33,17}` with native extra-row cell `(48,0)`. Exercise retained staging after a deliberate nonzero mutation and verify the full native allocation, not only logical projection.
- Do not change production transfer behavior, standard 16x16 storage semantics, native 32x32 tile geometry, or bytes outside `plane.padded_shape()`.

## Non-goals

- Do not modify `src/ttnn/copy.cpp` or any production source, common storage oracle, other backend, or vendor SDK behavior.
- Do not add arithmetic/operator coverage; TTNN operators remain unsupported.
- Do not specify or inspect bytes beyond the native allocation returned by `plane.padded_shape()`, and do not alter the standard logical layout model.

## Acceptance criteria

- [ ] For U8 `{17,33}`, the test identifies native 32x64 versus standard 32x48, detects a deliberate nonzero mutation at `(0,48)`, and after `copy_from_host()` observes every native-only column 48..63 as zero while logical bytes and standard canonical storage match exactly.
- [ ] For U8 `{33,17}`, the test identifies native 64x32 versus standard 48x32, detects `(48,0)`, and after `copy_from_host()` observes every native-only row 48..63 as zero while logical bytes and standard canonical storage match exactly.
- [ ] The independent check reads the complete native allocation and cannot hide a native-only mutation through standard-coordinate normalization; existing TTNN conformance behavior remains unchanged.

## Verification

- `cmake --build build/ttnn --target iom_ttnn_conformance_tests && ctest --test-dir build/ttnn -R '^iom_ttnn_conformance_tests$' --output-on-failure`
- On supported TTNN hardware, run both shapes, mutation-before-upload, retained-staging reuse, zero-after-`copy_from_host()`, and logical readback parity. The supplied ledger's TTNN smoke/conformance 2/2 result does not cover this extra-padding mutation; no focused native-only-padding gate has been run.
- Baseline validation ledger only (not NT-001 coverage): local GNU 15.2 CPU build/tests/bench/conformance passed 4/4; remote CUDA 13.2.78 smoke+conformance, ROCm HIP Clang 23 smoke+conformance, SYCL IntelLLVM 2026.1 smoke+conformance on two enumerated Arc Pro B60 Level Zero GPUs, and TTNN smoke+conformance each passed 2/2. No extra-padding mutation test, focused native-padding oracle gate, sanitizer, or static-analysis run has been performed.

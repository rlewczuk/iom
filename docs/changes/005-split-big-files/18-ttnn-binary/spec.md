# Extract TTNN binary planes

**Order:** 18
**Priority:** P1 — separates binary execution from retained host-transfer/layout code.
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

Split TTNN binary-plane execution into the planned `src/ttnn/binary.cpp` translation unit while retaining host transfer, native layout conversion, and plane-copy responsibilities in `src/ttnn/copy.cpp`. The extraction changes only source organization and build membership: native per-plane storage, carrier conversion, view addressing, broadcasting, scalar arithmetic, staging ownership, completion, failures, and linkage remain exactly as they are.

## Scope

- Retain in `src/ttnn/copy.cpp` the transfer/layout helpers and APIs in the current `:78-441` region, including `view_plane_count`, `owner_plane_at`, carrier-size and staging-slot mapping, tile-major `padded_cell_index`, upload/download conversion helpers, `view_rows`, `view_columns`, `region_from_host`, `region_to_host`, and `copy_planes`.
- Move the complete `binary_planes` template and its explicit `add`, `mul`, `sub`, and `div` instantiations from the current `src/ttnn/copy.cpp:423-608` region into a new planned `src/ttnn/binary.cpp`.
- Keep `src/ttnn/copy.hpp:12-35` as the private declaration boundary for `BinarySnapshot`, `BinaryRequest`, `BinaryFinish`, and the `binary_planes` declaration. If the moved implementation needs private tile/carrier helper visibility, expose only the minimum private declarations needed for the existing definitions; do not expose them through a public header or change their behavior.
- Update the TTNN `iom_ttnn` source list in `CMakeLists.txt` to compile and link `src/ttnn/binary.cpp` in addition to the retained `src/ttnn/copy.cpp`. Retain the existing TTNN support sources and private-header entries; do not change test source lists or public factory headers.
- Keep the retained copy translation unit at or below 430 physical lines and the new binary translation unit at or below 225 physical lines after normal formatting. Every touched or new production source/header remains at or below 499 physical lines.

## Implementation references

- `src/ttnn/copy.cpp:78-441` — retained plane-count/owner mapping, native carrier and tile-layout arithmetic, host upload/download conversion, direct region transfers, and `copy_planes`.
- `src/ttnn/copy.cpp:423-608` — `binary_planes` and the four explicit operation instantiations to relocate.
- `src/ttnn/copy.hpp:12-35` — private binary request/snapshot types, finish callback, and template declaration; preserve the declaration and its signature.
- `src/ttnn/staging.hpp:41-114` — `TtnnHostStaging::UploadLease` ownership, pin keepalive, release, and retirement semantics consumed by binary output uploads.
- `src/shared/scalar_add.hpp:52-63` — `scalar_binary<Op>` dispatch used for the existing host-side binary numerics.
- `CMakeLists.txt:114-140` — TTNN library target and source list to update.
- `test/CMakeLists.txt:245-255` — existing TTNN smoke/conformance targets and their unchanged source lists.
- `test/ttnn/test_ttnn_conformance.cpp:606-922` — TTNN storage/transfer, lifetime/failure, and real-queue add/mul/sub/div conformance coverage.
- `test/ttnn/test_ttnn_smoke.cpp:15-69` — TTNN hardware/context smoke coverage.

## Requirements

1. `src/ttnn/copy.cpp` must retain the complete host-transfer and layout path. `region_from_host` continues to map every logical view plane through `owner_plane_at`, acquire one retained upload lease per submitted plane using the native carrier dtype, perform the existing bit unpacking/tile-major placement, finish the native queue once the uploads are submitted, and release leases only after that completion proof. `region_to_host` continues to finish before consuming nonblocking native downloads, assemble carrier cells back into the logical bitstream, and preserve its retired-download drain/discard/retire behavior.
2. `copy_planes` remains the direct per-plane copy path with the existing view-plane count and source/destination owner-plane mapping, `any_submitted` transitions, injected-failure point, and exception behavior. It must not acquire binary responsibilities or a new abstraction.
3. `binary.cpp` must contain one definition of `template <detail::scalar_add_detail::BinaryOp Op> binary_planes` and exactly four explicit instantiations for `add`, `mul`, `sub`, and `div`. The declaration in `copy.hpp` and all callers retain their current signatures and visibility. There must be no duplicate template body or instantiation left in `copy.cpp`, and the split must preserve static-library compile/link and ODR behavior.
4. Preserve TTNN's native storage model: each leading-plane allocation remains an independent native `ttnn::Tensor`, and reads/writes use the existing carrier byte width, 64-bit two-carrier factor, padded dimensions, and physical tile-major `padded_cell_index` mapping. Do not flatten planes, introduce a row-major assumption, repack native storage, or allocate a new native representation.
5. Preserve result-plane addressing for all supported ranks. The binary loop must continue to enumerate every result leading-plane coordinate and matrix element, map operand coordinates right-aligned with singleton leading dimensions broadcast from coordinate zero, apply singleton row/column broadcasting, and select owner planes through each snapshot's offset and logical plane strides. Transformed views and multi-plane results must address the same owner planes as before.
6. Preserve the existing host-side scalar numerics exactly by continuing to call `detail::scalar_binary<Op>(request.out.spec.data_type, left, right)`. Do not substitute TTNN arithmetic, alter codecs or rounding, add supported dtypes, or change operation-specific validation/capability behavior elsewhere.
7. Preserve blocking native downloads and cache behavior: each distinct input owner plane is copied from TTNN with the existing blocking `ttnn::copy_to_host(..., /*blocking=*/true)` call once per invocation, then reused for all mapped coordinates. Do not make source downloads asynchronous or add a synchronization scheme.
8. Preserve output staging and completion ordering. Output plane images continue to use retained `TtnnHostStaging::UploadLease` instances and pinned host buffers, with existing retired-upload reclamation and native dtype dispatch. Submit every output upload first, invoke the supplied `BinaryFinish` exactly once after all output uploads, set `native_drained` only after that callback returns, and call `UploadLease::release()` only after the completion proof. If completion is not proven, lease destruction/retirement must keep possibly-read staging alive.
9. Preserve `any_submitted` and `native_drained` state transitions and the first-failure/error order. Exceptions from cache downloads, staging acquisition, host-buffer construction, native upload, the finish callback, or lease handling must retain their current propagation and must not be replaced, reordered, swallowed, or converted. Existing queue outcome retention, repeatable waits/failures, owner/native lifetime registration, and quarantine behavior remain owned by their current callers and are not redesigned by this extraction.
10. Keep all existing public headers, public signatures, capabilities, validation order, ownership/lifetimes, asynchronous in-order queue behavior, context behavior, and allocation behavior unchanged. This task is source factoring only and introduces no backend switch, global registry, cross-backend abstraction, compatibility alias/shim, synchronization redesign, or extra capability.
11. Do not add or modify tests. Existing TTNN smoke and conformance tests remain the behavior proof, and hardware absence/failure remains a test failure rather than a skip. Do not change test target source lists, test registration, or public factory headers.

## Non-goals

- Do not change `region_from_host`, `region_to_host`, `copy_planes`, host-transfer layout conversion, native carrier mapping, or their testing/failure seams beyond what is mechanically required to compile after the binary definition moves.
- Do not change binary arithmetic, dtype support, scalar encoding/decoding, broadcasting, rank/view semantics, alias or validation rules, queue submission, completion/failure policy, staging policy, or native allocation ownership.
- Do not move TTNN device, queue, testing, registry, or public API responsibilities; those are separate factoring boundaries. Do not create `queue_fence.cpp`, a factory-only file, a public binary header, or an additional shard for this extraction.
- Do not add new tests, permanent line-count tests, compatibility wrappers, or unrelated formatting/refactoring.

## Acceptance criteria

- `src/ttnn/copy.cpp` retains the named transfer/layout helpers, `region_from_host`, `region_to_host`, and `copy_planes`; the binary template body and all four explicit instantiations are absent from it.
- Planned `src/ttnn/binary.cpp` contains exactly one `binary_planes` definition and exactly one explicit instantiation each for `add`, `mul`, `sub`, and `div`, with the unchanged private declaration boundary in `src/ttnn/copy.hpp` and no ODR/link duplicate.
- Native per-plane storage/carrier conversion, rank and transformed-view owner-plane mapping, right-aligned leading broadcast, singleton matrix broadcast, blocking source downloads, scalar numerics, retained output `UploadLease` staging, one finish after all output uploads, release-after-proof, and failure/error ordering are observably unchanged by the split.
- The TTNN CMake target compiles and links `src/ttnn/binary.cpp` while retaining `src/ttnn/copy.cpp`; no test source list or public factory header changes.
- After normal formatting, `src/ttnn/copy.cpp` is at most 430 physical lines, `src/ttnn/binary.cpp` is at most 225 physical lines, and every touched/new production `.cpp`, `.cu`, `.hip`, `.hpp`, or `.inl` under `src/` or `include/` is at most 499 physical lines.
- Existing TTNN smoke, conformance, and coexistence coverage passes on the configured remote TTNN hardware, with no hardware test skipped and no new test introduced.

## Verification

Do not run gates while writing this mini-spec. After implementation, use the `remote-development` workflow on a configured TTNN-capable Linux host:

- Configure or reuse a TTNN-enabled build with `TTNN_ENABLED=ON` and `BUILD_TESTING=ON`, then build `iom_ttnn`, `iom_ttnn_smoke_tests`, `iom_ttnn_conformance_tests`, and `iom_backend_coexistence_tests`.
- Run `ctest --test-dir <remote-build> --output-on-failure -R '^(iom_ttnn_smoke_tests|iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$'` and require all three registered tests to pass; a missing device or hardware/runtime failure is a failure, not a skip.
- Run a deterministic one-time physical-line scanner over `src/ttnn/copy.cpp`, `src/ttnn/binary.cpp`, and any private production header touched by this extraction, explicitly checking `copy.cpp <=430`, `binary.cpp <=225`, and the universal `<=499` limit. Task 21 performs the final repository-wide scan. The scoped scanner is verification only and must not become a permanent test.
- Confirm the remote static-library build resolves the `binary_planes` declaration to one definition and the four requested explicit instantiations, while `copy.cpp` still supplies the retained transfer/layout APIs; compile/link success must cover `copy.hpp` consumers without changing its public availability.

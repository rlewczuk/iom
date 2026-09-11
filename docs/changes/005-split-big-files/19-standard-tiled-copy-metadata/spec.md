# Split standard tiled copy metadata

**Order:** 19
**Priority:** P1 — factors the oversized shared template fragment without changing consumers
**Blocked by:** None
**Source:** `docs/changes/005-split-big-files/spec.md`

## Outcome

`src/shared/standard_tiled_copy.inl` is split by responsibility without changing any included consumer. Its prologue, low-level field/tile codecs, scatter/gather kernels, host launch path, synchronous transfer implementation, and exported synchronous-transfer template remain in the original fragment. The metadata descriptor, checked metadata layout/writers, tiled-word copy, grid-stride bodies/kernels, and launch overloads move to the planned `src/shared/standard_tiled_copy_metadata.inl`. CUDA, ROCm, and SYCL consumers continue to include and instantiate the same private templates with the same symbols, visibility, macro configuration, launch behavior, and ODR properties.

## Scope

This task owns only the responsibility-complete factoring of the metadata/grid-stride block currently at `src/shared/standard_tiled_copy.inl:313-591`, the include at that exact boundary, and the corresponding CUDA/ROCm/SYCL target-list entries. Retain `src/shared/standard_tiled_copy.inl:1-312,594-739` unchanged in behavior and responsibility. The resulting retained fragment is expected to be about 461 physical lines and the planned metadata fragment about 279 physical lines; both, and every touched production source fragment, must be at most 499 physical lines after normal formatting.

## Implementation references

- Retained shared template fragment: `src/shared/standard_tiled_copy.inl`; keep the required macro checks/prologue, `iom::detail` and anonymous namespace setup, constants and low-level codecs through `gather_plane_kernel` at current lines 1-312, then retain `launch_view_transfer`, `synchronous_transfer_impl`, the exported `synchronous_transfer`, and the namespace closures at current lines 594-739.
- Planned metadata fragment: `src/shared/standard_tiled_copy_metadata.inl`; it is an included private implementation fragment, not a standalone translation unit. It owns the current lines 313-591: `CopyMetadataHeader`, `kInlineMetadataMaxRank`, `InlineCopyMetadata`, all metadata layout assertions, `checked_metadata_mul`, `checked_metadata_add`, `metadata_u64`, `padded_dimension`, `CopyMetadataLayout`, `copy_metadata_layout`, both `write_copy_metadata` overloads, `copy_one_tiled_word`, `grid_stride_copy_body`, `grid_stride_copy_kernel`, `grid_stride_copy_inline_kernel`, and both `launch_grid_stride_copy` overloads.
- Include boundary: in `src/shared/standard_tiled_copy.inl`, include `"standard_tiled_copy_metadata.inl"` exactly where the old lines 313-591 began—inside the already-open `namespace iom::detail { namespace {` scope and immediately after the retained low-level codecs. The new file must contain no namespace, include, macro, or namespace-closing wrapper.
- Backend target lists: `CMakeLists.txt` targets `iom_cuda`, `iom_rocm`, and `iom_sycl`; add both `src/shared/standard_tiled_copy.inl` and `src/shared/standard_tiled_copy_metadata.inl` to each corresponding source/header list while retaining all existing backend entries. Do not alter `src/shared/standard_tiled_add.inl` or test target source lists.
- Consumers and related private implementation: `src/cuda/copy.cu`, `src/rocm/copy.hip`, and `src/sycl/copy.cpp` include the standard tiled-copy implementation through their existing backend macro/include arrangement. Preserve those consumers, `src/shared/standard_tiled_copy.hpp`, and the existing `src/shared/standard_tiled_add.inl` contract.

## Requirements

1. **Exact ownership cut.** Move the complete current metadata block `src/shared/standard_tiled_copy.inl:313-591` and no other implementation. `CopyMetadataHeader` and `InlineCopyMetadata` must remain trivially copyable with the same field order, sizes, alignment, and fixed-slot assertions. Keep every checked arithmetic error condition and message, metadata dimension/stride/offset conversion, writer layout, and `copy_one_tiled_word` calculation unchanged.

2. **Exact retained block.** Keep `src/shared/standard_tiled_copy.inl:1-312,594-739` responsibility-complete: macro preconditions and defaults; `kTile`, `kTileSlots`, `kThreads`, and `kMaxBlocks`; field readers/writers and bit/tile coordinate helpers; `copy_tiled_to_tiled_word`, `copy_logical_to_tiled_word`, `copy_tiled_to_logical_word`; `scatter_plane_kernel` and `gather_plane_kernel`; `launch_view_transfer`; `synchronous_transfer_impl`; and exported `synchronous_transfer`. Do not duplicate any definition in either fragment.

3. **Include and visibility shape.** The retained file includes the planned metadata fragment at the former block location, after the low-level codecs and before `launch_view_transfer`. Because the new file is textually included, it must rely on the prior symbols and macros already established by the retained file and must not add wrappers or alter anonymous/private visibility. Preserve the existing include order, `IOM_GPU_*` macro preconditions/defaults, `Policy` launch/check calls, and namespace/ODR behavior for every backend.

4. **Consumer compatibility.** CUDA, ROCm, and SYCL must continue to compile and link their existing standard tiled-copy consumers without changing public headers, signatures, capabilities, backend switches, or include contracts. Keep all metadata and grid-stride symbols private to the existing implementation scope; no public declaration, compatibility alias, shim, global registry, cross-backend abstraction, or standalone compilation unit is permitted.

5. **Kernel and launch behavior.** Preserve `copy_one_tiled_word`, `grid_stride_copy_body`, `grid_stride_copy_kernel`, `grid_stride_copy_inline_kernel`, and both `launch_grid_stride_copy` overloads exactly in observable behavior: same descriptor interpretation, rank/stride/plane arithmetic, tail and padding handling, grid/block calculation, stream argument, kernel selection, and checked launch-count overflow behavior. Do not change synchronization, allocation, queue ordering, or failure behavior in the callers.

6. **Build integration.** Add the retained and planned `.inl` paths to each of the `iom_cuda`, `iom_rocm`, and `iom_sycl` target lists in `CMakeLists.txt`, without removing existing copy, driver, queue, or shared entries and without adding either fragment to test source lists. The new fragment is an included implementation section; target-list entries exist so IDE/build dependency tracking sees both files and do not imply separate linking units.

7. **Universal factoring invariants.** This is source factoring only. Preserve public headers/signatures/capabilities, error categories and validation order, ownership and lifetimes, allocation behavior, asynchronous in-order queues, repeatable waits/failures, registry/quarantine semantics, workspace/staging lease completion proof, context behavior, numerics, private visibility, and ODR. Use no new tests, backend switches, global registries, cross-backend abstractions, synchronization redesign, extra capability, changed public factory headers, or permanent line-count test.

8. **Line cap and untouched add fragment.** After normal formatting, `src/shared/standard_tiled_copy.inl` and `src/shared/standard_tiled_copy_metadata.inl` must each be at most 499 physical lines (approximately 461 and 279 respectively). `src/shared/standard_tiled_add.inl` is out of scope and must not be edited, reformatted, or otherwise changed.

## Non-goals

- Changing field/tile codecs, scatter/gather behavior, metadata representation, arithmetic, kernel arithmetic, launch geometry, synchronization, allocation, error behavior, or numerical results.
- Changing any CUDA, ROCm, or SYCL consumer, public header, backend macro contract, or `standard_tiled_copy.hpp` API.
- Moving code from `src/shared/standard_tiled_add.inl`, `src/shared/gpu_queue.hpp`, or any other production file into this fragment.
- Adding namespace/include wrappers, new helper layers, compatibility aliases, alternate backend paths, standalone modules, public metadata types, or a new test/line-count target.
- Modifying tests, test source lists, public factory headers, or unrelated CMake entries.

## Acceptance criteria

- `src/shared/standard_tiled_copy.inl` retains exactly the prologue/codecs/scatter-gather block and host transfer/export responsibilities described above, with the metadata include inserted at the former `:313` boundary.
- `src/shared/standard_tiled_copy_metadata.inl` contains exactly the former `:313-591` metadata, writer, tiled-word, grid-stride, kernel, and launch-overload responsibilities, with no namespace/include wrappers or duplicated symbols.
- Existing CUDA, ROCm, and SYCL consumers compile and link with their prior macro/include order, private symbol visibility, ODR behavior, launch behavior, and all observable copy, padding, tail, error, and numerical semantics unchanged.
- `CMakeLists.txt` lists both standard tiled-copy `.inl` paths under each of `iom_cuda`, `iom_rocm`, and `iom_sycl`, while retaining existing entries; `src/shared/standard_tiled_add.inl`, tests, and public factory headers are unchanged.
- Both tiled-copy fragments are at most 499 physical lines after normal formatting, with no new production file outside the planned destination.

## Verification

Proposed gates (not run by this specification writer):

- Through the `remote-development` workflow, configure or use the project’s existing CUDA build, build/link `iom_cuda` and its existing standard tiled-copy consumers, then run `ctest --test-dir <cuda-build> --output-on-failure -R '^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$'` on configured CUDA hardware. Hardware failures must fail; do not skip.
- Through the `remote-development` workflow, configure or use the project’s existing ROCm build, build/link `iom_rocm` and its existing standard tiled-copy consumers, then run `ctest --test-dir <rocm-build> --output-on-failure -R '^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$'` on configured ROCm hardware. Hardware failures must fail; do not skip.
- Through the `remote-development` workflow, configure or use the project’s existing SYCL build, build/link `iom_sycl` and its existing standard tiled-copy consumers, then run `ctest --test-dir <sycl-build> --output-on-failure -R '^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$'` on configured SYCL hardware. Hardware failures must fail; do not skip.
- Inspect compile/link proof for the CUDA, ROCm, and SYCL included-fragment consumers, including repeated inclusion/instantiation and private anonymous-namespace visibility; confirm no ODR or macro-order regression and no change to `standard_tiled_add.inl`.
- Run a deterministic one-time scanner over `src/shared/standard_tiled_copy.inl`, `src/shared/standard_tiled_copy_metadata.inl`, and any production file touched by this extraction, asserting no file exceeds 499 physical lines. Task 21 performs the final repository-wide scan. This scoped scanner is verification only and must not become a permanent test.

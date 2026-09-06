# Delete the SYCL copy-metadata twin and consume the shared `standard_tiled_copy.inl` header, layout, writer, and per-word decode

**Order:** 78
**Priority:** P1 — highest-value SYCL consolidation: ~150 lines of re-derived shared code deleted with no behavior change; bounded remediation that gates no other work
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` (Area 3 of `cpp-inference-code-review`) — whole-codebase working tree at `a9d8d0ccc3fb0d03e4746082481669dc969e7111`
**Finding:** `AR-010`
**Review area:** Backend architecture & simplicity
**Review severity:** medium
**Review verification:** verified, confidence 90
**Review scope:** whole-codebase
**Backend scope:** sycl
**Location:** `src/sycl/copy.cpp:53-150` (`SyclCopyMetadataHeader`, `checked_metadata_mul/add`, `metadata_u64`, `padded_dimension`, `SyclMetadataLayout`, `metadata_layout`), `:524-560` (`write_sycl_metadata`), `:739-785` (the `parallel_for` per-word plane decode)

## Outcome

`src/sycl/copy.cpp` stops re-implementing the queued-copy metadata contract that `src/shared/standard_tiled_copy.inl` already owns and that the SYCL translation unit already includes (line 32). The SYCL queued device-to-device copy computes its metadata layout with `detail::copy_metadata_layout`, writes it with `detail::write_copy_metadata` into a `detail::CopyMetadataHeader`, and its `parallel_for` body decodes one destination word by calling one shared device-side per-word helper. `SyclCopyMetadataHeader`, `SyclMetadataLayout`, `metadata_layout`, `write_sycl_metadata`, the four `checked_metadata_*`/`metadata_u64`/`padded_dimension` helpers, and the inline plane-decode block are deleted. Observable SYCL copy behavior — accepted shapes, overflow diagnostics, the single kernel launch, byte-exact tiled output — is unchanged.

## Current problem

- **Invariant:** one semantic operation has one source of truth. The queued-copy metadata representation (header field order/sizes), its checked layout arithmetic, the host writer that serializes strides/dimensions after the header, and the device-side per-word plane decode are backend-neutral facts owned by `src/shared/standard_tiled_copy.inl` (created by 63-AR-004, consumed by CUDA at `src/cuda/copy.cu:228-273` and ROCm at `src/rocm/copy.hip:227-272`).
- **Failing path:** `src/sycl/copy.cpp` includes that file (line 32) yet re-declares a byte-identical twin of nearly all of its host-side metadata machinery and re-derives its device-side decode:
  - `SyclCopyMetadataHeader` (`:53-61`) is field-for-field identical to `detail::CopyMetadataHeader` (`.inl:312-320`): `source_plane_offset`, `destination_plane_offset`, `rows`, `columns`, `plane_count` (all `uint64_t`), `bits`, `leading_rank` (both `uint32_t`).
  - `checked_metadata_mul`, `checked_metadata_add`, `metadata_u64`, `padded_dimension` (`:64-94`) are identical to `.inl:334-363`.
  - `SyclMetadataLayout` + `metadata_layout` (`:96-150`) are identical to `detail::CopyMetadataLayout` + `detail::copy_metadata_layout` (`.inl:365-419`); a normalized diff of the two host helper blocks scores 0.91 similarity, and the only residual differences are the `Sycl`-prefixed type names, the `detail::leaf_bits` vs unqualified `leaf_bits` call (the same `iom::detail::leaf_bits`), and the `total_words` vs `total_words_size` local name.
  - `write_sycl_metadata` (`:524-560`) is identical to `detail::write_copy_metadata` (`.inl:421-457`).
  - The `parallel_for` lambda body (`:741-785`) re-derives, word for word, the per-word plane decode that `detail::grid_stride_copy_body` owns (`.inl:466-511`): `logical_plane = word / words_per_plane`, `word_in_plane = word % words_per_plane`, the leading-coordinate stride walk producing `source_plane`/`destination_plane`, then `detail::copy_tiled_to_tiled_word(...)`. The shared body wraps that step in a grid-stride loop; SYCL executes it once per work item. The arithmetic is the same decision written twice.
- **Evidence:** normalized `difflib` comparison of `src/sycl/copy.cpp:53-150,524-560` against `src/shared/standard_tiled_copy.inl:312-463` returns 0.91 similarity with the differences listed above; `grep` for `detail::(copy_metadata_layout|write_copy_metadata|CopyMetadataHeader|grid_stride_copy_body)` in `src/sycl/copy.cpp` returns zero matches, confirming the shared owners are included but unused.
- **Impact:** every change to the copy-metadata contract — a new header field, a corrected overflow bound, a layout fix — must be applied in two places that no test ties together. A drift in the SYCL twin (e.g. a header field reordered, a `words_per_plane` rounding changed) silently produces wrong SYCL copies while CUDA/ROCm stay correct, and the divergence is invisible until SYCL hardware runs.

## Scope

- Replace the SYCL host-side metadata path with the shared owners: `detail::CopyMetadataHeader`, `detail::copy_metadata_layout`, `detail::write_copy_metadata`. Delete `SyclCopyMetadataHeader`, `SyclMetadataLayout`, `metadata_layout`, `write_sycl_metadata`, `checked_metadata_mul`, `checked_metadata_add`, `metadata_u64`, and `padded_dimension` from `src/sycl/copy.cpp`.
- Extract the single-word decode step from `detail::grid_stride_copy_body` into one shared `IOM_GPU_DEVICE` helper that both the shared grid-stride loop and the SYCL `parallel_for` lambda call. This is the only `src/shared/` change and it preserves the CUDA/ROCm expansion byte-for-byte in behavior.
- Affected backend: sycl only at runtime. The one shared-file edit must not change CUDA or ROCm behavior.

## Implementation references

- **Modify:** `src/sycl/copy.cpp` — delete `:53-150` and `:524-560`; in `SyclQueue::execute` (`:719-729`) replace `metadata_layout(...)` with `detail::copy_metadata_layout(*task.source, *task.destination)` (returns `detail::CopyMetadataLayout{bytes, total_words}`) and `write_sycl_metadata(...)` with `detail::write_copy_metadata(metadata_pool_.host_data(metadata_slot), *task.source, *task.destination)`; change the device-metadata cast at `:736-738` from `const SyclCopyMetadataHeader*` to `const detail::CopyMetadataHeader*`; rewrite the `parallel_for` lambda (`:741-785`) to call the new shared per-word helper once with `item[0]`.
- **Modify:** `src/shared/standard_tiled_copy.inl` — factor `grid_stride_copy_body` (`:466-511`) into `IOM_GPU_DEVICE void copy_one_tiled_word(const unsigned char* source, unsigned char* destination, const CopyMetadataHeader& metadata, const std::uint64_t* values, std::uint64_t word)` holding the per-word decode + `copy_tiled_to_tiled_word` call, and have `grid_stride_copy_body` loop `for (word = IOM_GPU_GLOBAL_INDEX; word < total_words; word += stride) copy_one_tiled_word(source, destination, metadata, values, word);`. `total_words` and the stride loop stay in `grid_stride_copy_body`.
- **Read:** `src/cuda/copy.cu:228-273`, `src/rocm/copy.hip:227-272` — the established consumption of `detail::copy_metadata_layout` / `detail::write_copy_metadata` / `detail::launch_grid_stride_copy`; SYCL mirrors the host-side calls and keeps its own `parallel_for` launch.
- **Read:** `docs/changes/0001-tensor-view/60-AR-001-converge-sycl-copy-architecture/spec.md` — the landed SYCL queued-copy shape (single metadata-driven `parallel_for`, `SyclFenceState`, fault seam at the single post-launch boundary) this task preserves; only the metadata *provenance* changes, not the submission/fence protocol.
- **Tests:** `test/sycl/test_sycl_conformance.cpp` — `run_async_copy_conformance`, `run_storage_oracle_conformance`, and `"SYCL queued copy submits one kernel regardless of plane count"` (`test/sycl/test_sycl_smoke.cpp:270-339`) exercise the changed path; assertions unchanged.

## Requirements

- `src/sycl/copy.cpp` contains no `SyclCopyMetadataHeader`, `SyclMetadataLayout`, `metadata_layout`, `write_sycl_metadata`, `checked_metadata_mul`, `checked_metadata_add`, `metadata_u64`, or `padded_dimension` definition; the queued-copy metadata is produced exclusively by `detail::copy_metadata_layout` and `detail::write_copy_metadata` over a `detail::CopyMetadataHeader`.
- The device-side metadata buffer layout is unchanged: a `detail::CopyMetadataHeader` (48 bytes: five `uint64_t` then two `uint32_t`) followed immediately by `3 * leading_rank` `uint64_t` values (source strides, destination strides, leading dimensions), exactly as `detail::write_copy_metadata` writes and `detail::grid_stride_copy_kernel` reads. The SYCL `parallel_for` reads the values array at `reinterpret_cast<const std::uint64_t*>(metadata + 1)`.
- `src/shared/standard_tiled_copy.inl` gains exactly one new `IOM_GPU_DEVICE` function (`copy_one_tiled_word`); `grid_stride_copy_body` becomes a grid-stride loop over it. The CUDA and ROCm kernel expansions (`grid_stride_copy_kernel`, `grid_stride_copy_inline_kernel`) produce identical device behavior: same `words_per_plane`, same plane decode, same `copy_tiled_to_tiled_word` arguments, same loop bounds.
- The SYCL `parallel_for` issues exactly one work item per destination storage word over `sycl::range<1>(layout.total_words)` and calls `copy_one_tiled_word(source_handle, destination_handle, *metadata, values, item[0])`; it contains no inline plane-decode arithmetic.
- Overflow diagnostics are preserved: `detail::copy_metadata_layout` throws the same `std::overflow_error` messages the deleted `metadata_layout` threw, before any enqueue, because the shared helper is the surviving owner of that arithmetic.
- The `SyclQueue::execute` submission/fence protocol — metadata-slot acquire, `ensure_slot_capacity`, in-order `queue_.memcpy` of the metadata, single `parallel_for`, `set_event`, the `submitted_any`/`metadata_enqueued` rollback boundary, and the `SubmissionFault` seam — is unchanged.

## Non-goals

- The synchronous host-transfer path (`region_from_host`/`region_to_host`, `launch_view_transfer`, `launch_scatter_words`/`launch_gather_words` at `src/sycl/copy.cpp:428-522,879-933`) and its relationship to the shared `synchronous_transfer<Policy>`/`launch_view_transfer<Policy>` templates — separate concern; SYCL's single in-order transfer queue (no `TransferStreamPool`, no `Policy::activate`) is a deliberate divergence pinned by 60-AR-001.
- `SyclMetadataSlotPool` (`:152-284`) vs the shared `detail::MetadataSlotPool<Policy>` — the SYCL pool allocates USM device+host mirrors against an explicit `sycl::context`, which the context-free shared policy seam cannot express; not unified here.
- `SyclFenceState`, the `StagedWorker` callbacks, the registry/outcome machinery, and the `SubmissionFault` enumerators — owned by 55-ST-001/60-AR-001; untouched.
- The SYCL `StagingSlotPool` (separate finding AR-011).
- Any CUDA, ROCm, CPU, or TTNN runtime behavior change; the only shared edit is the behavior-preserving `copy_one_tiled_word` extraction.

## Acceptance criteria

- [ ] `grep -rn "SyclCopyMetadataHeader\|SyclMetadataLayout\|metadata_layout\|write_sycl_metadata\|checked_metadata_\|metadata_u64\|padded_dimension" src/sycl/copy.cpp` returns zero matches.
- [ ] `grep -n "detail::copy_metadata_layout\|detail::write_copy_metadata\|detail::CopyMetadataHeader\|copy_one_tiled_word" src/sycl/copy.cpp` shows the shared owners consumed in `SyclQueue::execute` and the `parallel_for` lambda.
- [ ] `grep -n "copy_one_tiled_word" src/shared/standard_tiled_copy.inl` shows one definition and one call site inside `grid_stride_copy_body`; `grid_stride_copy_body` retains the `IOM_GPU_GLOBAL_INDEX`/`IOM_GPU_GLOBAL_STRIDE` loop.
- [ ] `git diff --stat src/cuda src/rocm src/cpu src/ttnn` is empty; `git diff --stat src/shared/standard_tiled_copy.inl` shows only the `copy_one_tiled_word` extraction.
- [ ] On SYCL hardware, `iom_sycl_conformance_tests` and `iom_sycl_smoke_tests` pass unchanged, including `"SYCL queued copy submits one kernel regardless of plane count"` (launch count still 1 for a multi-plane shape) and the full-leaf-type asynchronous-copy and storage-oracle cases (byte-exact tiled output).
- [ ] On CUDA and ROCm hardware, `iom_cuda_conformance_tests` / `iom_rocm_conformance_tests` pass unchanged, proving the shared `copy_one_tiled_word` extraction did not alter their kernel behavior.

## Verification

- `cd /home/rlew/iom/src/iom && cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir build --output-on-failure -R 'iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests'` — the shared `.inl` extraction must not break the CPU/common build (the file is header-only and parsed by every GPU TU; CPU does not include it, so this confirms no accidental core coupling).
- SYCL via `.agents/skills/remote-development`, `sycl` profile from `.remote-hosts.conf` (DPC++ through `source /opt/intel/oneapi/setvars.sh`): `remote-exec sycl <task-id> 'cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF'` then `cmake --build build/sycl -j --target iom_sycl_smoke_tests iom_sycl_conformance_tests iom_tests` then `ctest --test-dir build/sycl --output-on-failure -R '^iom_sycl_(smoke|conformance)_tests$|^iom_tests$'`.
- CUDA and ROCm regression for the shared-file edit, each on its own host per `.remote-hosts.conf`: build `iom_cuda_conformance_tests` / `iom_rocm_conformance_tests` and run the asynchronous-copy and storage-oracle cases; expect identical pass/fail to the pre-change baseline.
- Static audits: the `grep` criteria under Acceptance criteria, plus `git diff --stat src/shared/standard_tiled_copy.inl` reviewed line-by-line to confirm `grid_stride_copy_body`'s loop bounds and `copy_tiled_to_tiled_word` arguments are unchanged.

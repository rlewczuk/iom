# CPU byte-aligned tile rows use one tiny memcpy per element

**Order:** 12
**Priority:** P1
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** PF-005
**Review area:** Performance
**Review severity:** medium
**Review verification:** strongly-supported, confidence 85
**Review scope:** whole-codebase
**Backend scope:** cpu
**Location:** `src/cpu/device.cpp:89-106` (`copy_tile_row_byte_aligned`); callers `src/cpu/device.cpp:502-675` (`CpuTensor` host transfers) and `:709-769` (`CpuQueue::copy_elements`)

## Outcome

For disjoint contiguous byte-aligned tile row runs, CPU storage/host copies perform one bulk memory move per row run instead of up to 16 per-element memcpy calls; the element-wise path is retained for overlapping/self ranges so no alias behavior changes.

## Current problem

For disjoint contiguous byte-aligned tile row runs, storage/host copies should perform the minimum equivalent bulk memory work while preserving behavior for self/overlapping windows.

`copy_tile_row_byte_aligned<1|2|4|8>` (`src/cpu/device.cpp:99-106`) loops over up to 16 elements and invokes `std::memcpy` separately for each element; the only avoided case is an exact same-element pointer. The helper is called for every byte-aligned row run during CPU uploads, downloads, and queued copies (`CpuTensor::region_from_host`/`region_to_host` dispatch widths 8/16/32/64 at `:552-571`/`:644-663`; `CpuQueue::copy_elements` at `:735-758`). A full 4096×4096 F32 tensor has 4096 rows × 256 tile columns = 1,048,576 row runs × 16 element iterations ≈ 16.8M tiny copy calls per pass in source-level work. The shared copy test matrix explicitly chooses non-overlapping source/destination windows (`test/backend/backend_conformance_copy_storage.hpp:186-188`), while public `TensorView` documentation does not promise non-overlap. Impact: avoidable tiny-copy loop overhead and branch work on the CPU F32/U8/U16/U32/U64 transfer and queue-copy paths; compiler inlining may reduce actual call overhead, so the magnitude must be measured.

## Scope

- In `copy_tile_row_byte_aligned`, detect whether the contiguous source and destination byte ranges are disjoint.
- Use one bulk copy for disjoint ranges; retain the existing element-wise path for overlapping/self ranges so no alias behavior is silently changed.
- Keep row-run/tile traversal and all sub-byte paths unchanged.

## Implementation references

- **Modify:** `src/cpu/device.cpp` — `copy_tile_row_byte_aligned` (`:89-106`); owns every byte-aligned row-run copy for both transfers and queue copies.
- **Read:** `src/cpu/device.cpp` — callers `CpuTensor::region_from_host`/`region_to_host` (`:502-675`) and `CpuQueue::copy_elements` (`:709-769`); the row-run contract the helper serves.
- **Read:** `test/backend/backend_conformance_copy_storage.hpp:186-188` — the non-overlap convention exercised by the shared suite.
- **Tests:** `test/cpu/test_cpu_bench.cpp:132-143` — the 4096×4096 F32 host/queued benchmark; CPU/shared conformance storage/copy tests for behavior preservation.

## Requirements

- For disjoint byte-aligned row runs, perform one equivalent bulk memory move per run in each of the four instantiations (element widths 8/16/32/64 bits).
- Overlapping/self ranges must keep the current per-element behavior bit-for-bit; no alias semantics change.
- All logical/padded oracle checks and existing CPU conformance must remain byte-identical.

## Non-goals

- No rewrite of tiled traversal, sub-byte bit packing, GPU kernels, or alias semantics; no global copy framework/cache; no change to the benchmark harness itself.

## Acceptance criteria

- [ ] An instrumentation or generated-code check shows one bulk copy call per disjoint byte-aligned row run (down from up to 16 per-element memcpys), for U8/U16/U32/U64/F32 host uploads/downloads and queued copies.
- [ ] Self/overlap scenarios produce bytes identical to the pre-change behavior, the CPU/shared conformance byte oracle passes unchanged, and the 4096×4096 F32 benchmark throughput improves or does not regress.

## Verification

`actual validation: none (read-only review); proposed gates below`.

Root-run facts (remote bv2 host, CPU-only, at review time): configure OK; `iom_cpu_tests` and `iom_backend_conformance_cpu_tests` pass on the reviewed HEAD; `iom_cpu_bench` fails on the hard-coded one-host throughput floors (PF-006 verified). Because the current `iom_cpu_bench` CTest registration cannot pass on a generic host, use its measured samples as a relative before/after comparison rather than its absolute floors.

- CPU (any host): configure with testing enabled, build, and run `ctest --test-dir <build> -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure`; expected: byte-identical conformance and all existing tests passing.
- Benchmark: run `iom_cpu_bench` before and after the change and compare median throughput samples for aligned/disjoint F32/U8/U16/U32/U64 host uploads/downloads and queued copies; expected: improved or non-regressed throughput, plus a temporary aligned-row bulk-copy count confirming one bulk copy per disjoint row run.

- `ctest --test-dir <build> -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure`
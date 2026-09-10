# Share the canonical standard-tile mapping

**Order:** 08
**Priority:** P1 — deletes duplicate production mapping
**Blocked by:** `05-NT-002-assert-gpu-broadcast-tail-values`
**Review source:** `cpp-inference-code-review` — mandatory `cpp-inference-review-synthesis`; whole-codebase reviewed state: clean main at HEAD `1fc680892b8b08fd528edf965c669d68ed0bb993`, 99 commits ahead of `origin/main`
**Finding:** AR-001
**Review area:** Backend architecture & simplicity
**Review severity:** medium
**Review verification:** verified, confidence 98
**Review scope:** whole-codebase
**Backend scope:** CUDA and ROCm/HIP standard-tiled paths
**Location:** `src/shared/standard_tiled_add.inl` — `kAddTile`, `kAddTileSlots`, `add_plane_slot`, and duplicate slot-to-coordinate arithmetic; canonical counterpart `src/shared/standard_tiled_copy.inl` — `plane_slot` and `physical_coordinate`; inclusion units `src/cuda/copy.cu` and `src/rocm/copy.hip`

## Outcome

CUDA and ROCm binary kernels use the same canonical standard 16x16 plane-slot and slot-to-coordinate helpers as standard transfers. Exactly one production mapping owner remains; packed-word ownership, broadcast/tail behavior, transformed views, padding, launch geometry, and backend policy behavior remain unchanged.

## Current problem

The invariant is that every CUDA/ROCm standard-layout operation maps logical coordinates to the canonical physical slot and preserves padding/untouched storage. `standard_tiled_copy.inl` owns `plane_slot` and `physical_coordinate`, while `standard_tiled_add.inl` independently owns `kAddTile`/`kAddTileSlots`, `add_plane_slot`, and slot-to-row/column reconstruction. The formulas currently match, so no current numerical divergence is claimed; the defect is two mutable production sources. Both `cuda/copy.cu` and `rocm/copy.hip` include standard copy before standard add, making the canonical helpers available. Root CUDA/ROCm smoke/conformance/coexistence passed, but no candidate-specific mapping mutation or post-cutover compile ran.

## Scope

- Delete the duplicate add mapping constants/helper and duplicate slot-coordinate arithmetic.
- Make binary use `standard_tiled_copy.inl`'s existing `plane_slot` and `physical_coordinate` in both CUDA and ROCm translation units.
- Preserve packed-word merge/ownership, broadcast flags, tails, transformed leading views, aliases, padding, launch geometry, and policy-specific async behavior.

## Implementation references

- **Modify:** `src/shared/standard_tiled_add.inl` — remove `kAddTile`, `kAddTileSlots`, `add_plane_slot`, and duplicate coordinate expansion; call canonical helpers.
- **Read:** `src/shared/standard_tiled_copy.inl` — `plane_slot` and `physical_coordinate` are the one surviving production standard mapping.
- **Read:** `src/cuda/copy.cu` and `src/rocm/copy.hip` — preserve include order proving helper visibility in both translation units.
- **Read:** `src/shared/gpu_queue.hpp` — `GpuQueue::binary_impl`/`execute` metadata and launch callers; do not alter request or policy behavior.
- **Tests:** CUDA/ROCm binary conformance, storage/padding oracles, and the value-sensitive mapped cases from NT-002.

## Requirements

- Use the already included `plane_slot` for operand/output slot computation and `physical_coordinate` for slot-to-row/column expansion; no second formula or constants may remain in production CUDA/ROCm standard mapping.
- Preserve the existing packed-word merge/exclusive ownership behavior, broadcast coordinate handling, tails, transformed views, aliases, and untouched physical planes.
- Do not reuse production mapping helpers in independent test oracles, and do not unify TTNN's intentional native 32x32 mapping with standard layout.

## Non-goals

- Do not change common validation, codec/arithmetic semantics, owner/lifetime behavior, kernel geometry, allocations, synchronization, launch count, CUDA/HIP policy, or TTNN storage.
- Do not merge this mapping task with AR-002 codec or AR-003 operation-dispatch refactors.

## Acceptance criteria

- [ ] Each CUDA/ROCm translation unit has one production standard plane-slot and slot-coordinate implementation supplied by `standard_tiled_copy.inl`; no add mapping symbols or duplicate arithmetic remain.
- [ ] Binary and copy outputs remain correct for equal shapes, broadcasts, final tails, transformed views, aliases, padding, and untouched physical storage under independent oracles.
- [ ] The value-sensitive NT-002 mapped tests pass before and after the cutover, and a temporary mutation of canonical `plane_slot` affects both copy and binary checks.
- [ ] No allocation, synchronization, launch, storage representation, or policy behavior is added.

## Verification

- `cmake --build <cuda-build> --target iom_cuda_conformance_tests && ctest --test-dir <cuda-build> --output-on-failure -R '^iom_cuda_conformance_tests$'` — run on the configured CUDA host with tails, broadcasts, transformed views, aliases, padding, and full physical-storage checks.
- `cmake --build <rocm-build> --target iom_rocm_conformance_tests && ctest --test-dir <rocm-build> --output-on-failure -R '^iom_rocm_conformance_tests$'` — run the equivalent cases on the configured ROCm host.
- Apply a throwaway mutation to canonical `plane_slot`; require both standard copy and binary conformance to fail, then remove it. Run the NT-002 focused mapping-value cases as the prerequisite guard.
- Root-supplied remote CUDA/ROCm smoke/conformance/coexistence passed. No candidate-specific mapping mutation or post-cutover compile has run; generated artifacts and vendor behavior remain gaps.

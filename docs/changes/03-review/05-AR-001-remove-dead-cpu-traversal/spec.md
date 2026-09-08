# Remove dead CPU coordinate traversal machinery

**Order:** 05
**Priority:** P1 — leave one active traversal mechanism and remove obsolete worker residue
**Blocked by:** None
**Review source:** `cpp-inference-backend-simplicity` — `whole-codebase reviewed state: branch main, clean HEAD 86533aef347935405cb86d465cc5489f5c3d530a (Remove review files.)`
**Finding:** `AR-001`
**Review area:** Backend architecture & simplicity
**Review severity:** low
**Review verification:** verified, confidence 99
**Review scope:** whole-codebase
**Backend scope:** CPU
**Location:** `src/cpu/device.cpp` — anonymous-namespace `for_each_coordinate` and `plane_at`, plus direct worker-only includes

## Outcome

The CPU translation unit contains no dead `for_each_coordinate` or `plane_at` traversal and no direct unused `<condition_variable>`, `<deque>`, `<exception>`, or `<thread>` include. `for_each_tile` and `for_each_tile_lockstep` remain the sole active CPU tiled traversal mechanisms, with all copy, overlap, view, allocator, and inline-completion behavior unchanged.

## Current problem

`for_each_coordinate` and `plane_at` have anonymous-namespace linkage and no callers anywhere in the repository. They describe a competing coordinate/plane mapping path while active CPU transfers use `for_each_tile` and `for_each_tile_lockstep`. The CPU worker/task path was deleted, but `device.cpp` still directly includes the worker-only headers `<condition_variable>`, `<deque>`, `<exception>`, and `<thread>`. This is objective dead machinery and redundant dependency residue: it adds sources of truth and can mislead future maintenance without changing current runtime output.

## Scope

- Delete only the anonymous CPU `for_each_coordinate` and `plane_at` definitions from `src/cpu/device.cpp`.
- Remove only the direct unused includes `<condition_variable>`, `<deque>`, `<exception>`, and `<thread>` from that file.
- Retain `for_each_tile`, `for_each_tile_lockstep`, `CpuTensor::region_from_host`, `CpuTensor::region_to_host`, `CpuQueue::copy_elements`, inline queue completion, layout arithmetic, and all other backend code.

## Implementation references

- **Modify:** `src/cpu/device.cpp` — four includes and the anonymous-namespace helpers around lines 365–413; this is the only file containing the dead definitions and direct include residue.
- **Read:** `src/cpu/device.cpp` — `for_each_tile`, `for_each_tile_lockstep`, and active `CpuTensor`/`CpuQueue` callers; preserve these as the sole traversal owners.
- **Read:** `test/cpu/test_cpu_bench.cpp` — note documenting removal of the old CPU worker path; confirms the deleted worker-only dependencies are not active CPU queue machinery.
- **Tests:** `test/cpu/test_cpu.cpp` and `test/cpu/test_cpu_conformance.cpp` — existing transformed-view, partial-tile, overlap, sub-byte, lifetime, inline completion, and unsupported-operation behavior.

## Requirements

- Delete `for_each_coordinate` and `plane_at` completely; do not replace them with aliases, wrappers, or another traversal abstraction.
- Remove direct includes `<condition_variable>`, `<deque>`, `<exception>`, and `<thread>` only when they have no direct use in `src/cpu/device.cpp`; retain every include needed by the active implementation.
- Preserve `for_each_tile` and `for_each_tile_lockstep` and all callers, including CPU transformed-view/stride arithmetic, overlap-safe copies, synchronous completion, allocator ownership, and error behavior.
- Do not move CPU traversal into shared accelerator code: backend-specific asynchronous/native traversal mechanisms remain unchanged.

## Non-goals

- Do not change `TensorView` layout arithmetic, `detail::standard_plane_slot`, CPU queue token/failure semantics, allocator/lifetime handling, host-transfer behavior, or overlap handling.
- Do not alter CUDA, ROCm, SYCL, TTNN, shared kernels, the common staged worker, or the live TTNN `owner_plane_at` helper.
- Do not consolidate the separate active byte-width dispatches or perform style-only cleanup beyond the four named includes and two named helpers.

## Acceptance criteria

- [ ] Repository search finds no CPU definition or reference to `for_each_coordinate` or `plane_at`, and `src/cpu/device.cpp` retains only `for_each_tile` and `for_each_tile_lockstep` as traversal owners.
- [ ] `src/cpu/device.cpp` has no direct unused `<condition_variable>`, `<deque>`, `<exception>`, or `<thread>` include, and the normal include graph still compiles.
- [ ] Existing CPU tests preserve transformed/stepped views, partial tiles, sub-byte and byte-aligned transfers, overlapping windows, inline completion, address recycling, allocator behavior, and unsupported-operation diagnostics.

## Verification

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cpu --target iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/cpu -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$' --output-on-failure`
- Run an exhaustive source search for `for_each_coordinate`, `plane_at`, and the four include names, then confirm the CPU-only build/tests pass. The supplied CPU ledger passed its existing 4/4 gates before this deletion; no post-deletion gate has been run.
- Baseline validation ledger only (not AR-001 coverage): local GNU 15.2 CPU build/tests/bench/conformance passed 4/4; remote CUDA 13.2.78 smoke+conformance, ROCm HIP Clang 23 smoke+conformance, SYCL IntelLLVM 2026.1 smoke+conformance on two enumerated Arc Pro B60 Level Zero GPUs, and TTNN smoke+conformance each passed 2/2. No post-deletion CPU build/test, sanitizer, static-analysis, or IWYU gate has been run.

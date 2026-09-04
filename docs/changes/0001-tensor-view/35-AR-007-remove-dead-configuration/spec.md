# Remove still-dead configuration surface (hello-world executable, unconsumed backend macros, unused allocator predicate)

**Order:** 35
**Priority:** P2 — required finishing cleanup of dead configuration surface; no correctness, lifetime, or memory-safety invariant is involved.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-007`
**Review severity:** low
**Review verification:** verified, confidence 98

## Outcome

Every review-listed dead configuration surface is removed: the `iom` "hello world" executable and its single source file are deleted; the four unconsumed `IOM_*_ENABLED` `PRIVATE` compile definitions (`IOM_TTNN_ENABLED`, `IOM_CUDA_ENABLED`, `IOM_ROCM_ENABLED`, `IOM_SYCL_ENABLED`) are removed from the `CMakeLists.txt` blocks that attach them to nothing; and `SingleBufferAllocatorBase::owns(void*)` (declaration plus definition) is removed because no caller exists. Active SYCL configuration — the `SYCL_ENABLED` option, the `SYCL_COMPILER` discovery path, the `iom_sycl` static library, the `BackendKind::SYCL` enumerator, the `src/sycl/` factory, and the SYCL test drivers — is preserved because it is consumed by code that is live today. `__HIP_PLATFORM_AMD__` is preserved because it is consumed by AMD's HIP headers and is not a project-private macro.

## Current failure

`docs/changes/0001-tensor-view/review.md` AR-007 enumerates four dead-configuration items at the time of review. Repository evidence confirms three remain dead and one is already implemented:

1. `src/main.cpp` (lines 1–5) prints `Hello world!` and exits. The root `CMakeLists.txt:257-264` defines `add_executable(iom src/main.cpp)` and `target_link_libraries(iom PRIVATE libiom)`. No other target, install rule, or test references the `iom` binary. The build produces an executable that does nothing the library does not already cover. The review reports the binary was executed with exit 0.
2. The four `target_compile_definitions(... PRIVATE IOM_*_ENABLED=1)` calls in `CMakeLists.txt:141-144` (`IOM_TTNN_ENABLED=1`), `:175-178` (`IOM_CUDA_ENABLED=1`), `:213-217` (`IOM_ROCM_ENABLED=1`), and `:251-254` (`IOM_SYCL_ENABLED=1`) set macros that are referenced by no `.cpp`, `.hpp`, `.cu`, `.hip`, or test file in the current tree. Repo-wide greps of `IOM_CUDA_ENABLED|IOM_TTNN_ENABLED|IOM_ROCM_ENABLED|IOM_SYCL_ENABLED` under `src/`, `include/`, and `test/` return zero matches. The macros create the false impression that per-backend source switches exist.
3. `include/iom/alloc.hpp:26` declares `[[nodiscard]] bool owns(void* ptr) const;` on `SingleBufferAllocatorBase`; `src/alloc.cpp:53-56` defines it. Repo-wide search for `.owns(` returns only the declaration site and the definition site — no caller. The predicate duplicates a `begin_..raw_end_` containment check that `offset_of(void*)` already performs (it throws `std::invalid_argument` for the same out-of-range case at `src/alloc.cpp:60-62`).
4. `include/iom/tensor.hpp:78` exposes `BackendKind::SYCL`. At review time this was dangling. **Resolved** since: `src/sycl/device.cpp:77` returns `BackendKind::SYCL`, `test/sycl/test_sycl_smoke.cpp:90` asserts on it, `CMakeLists.txt:219-255` builds the `iom_sycl` static library behind `-DSYCL_ENABLED=ON`, and `test/CMakeLists.txt:138-207` builds `iom_sycl_smoke_tests` and `iom_sycl_conformance_tests` with that option. The enumerator is active configuration, not a dangling declaration. Per the AR-007 note ("the SYCL enumerator's fate is decided together with CC-002") and CC-002's review note ("SYCL backend has been implemented since, please skip this point"), this item is preserved untouched.

Maintainers and users currently infer behavior (a CLI, per-backend `#ifdef IOM_*_ENABLED` switches, a `BackendKind` switch dispatching to a missing SYCL backend) that does not exist. After this remediation, declared build surface corresponds one-to-one to real behavior.

## Scope

- Delete the `iom` executable target and its sole source file.
- Drop the four `target_compile_definitions(... PRIVATE IOM_*_ENABLED=1)` lines that attach `IOM_TTNN_ENABLED`, `IOM_CUDA_ENABLED`, `IOM_ROCM_ENABLED`, and `IOM_SYCL_ENABLED` to their backend static libraries.
- Delete the `owns(void*)` declaration and definition; no other allocator API is touched.
- Do **not** touch `BackendKind::SYCL`, the SYCL factory, `iom_sycl`, the SYCL test drivers, the `SYCL_ENABLED` option, the `SYCL_COMPILER` discovery path, or any `find_package(IOM_SYCL_*)` invocation.
- Do **not** touch `__HIP_PLATFORM_AMD__` (consumed by AMD's HIP headers, not project-private).
- Do **not** introduce new abstractions, reorganize allocators, or modify the `Allocator` interface.

## Implementation references

- **Modify:** `CMakeLists.txt:257-264` — delete the `add_executable(iom src/main.cpp)` block and its `target_link_libraries(iom PRIVATE libiom)`. The block is the sole producer of the `iom` binary; nothing else links the name.
- **Delete:** `src/main.cpp` (the entire 5-line file). With the executable removed, the source has no consumer.
- **Modify:** `CMakeLists.txt:141-144` — remove `target_compile_definitions(iom_ttnn PRIVATE IOM_TTNN_ENABLED=1)`. The surrounding `if(TTNN_ENABLED)` block, the `find_package(tt-nn CONFIG REQUIRED)`, the `add_library(iom_ttnn …)`, the include/link directives, and the `add_test` registration stay.
- **Modify:** `CMakeLists.txt:175-178` — remove `target_compile_definitions(iom_cuda PRIVATE IOM_CUDA_ENABLED=1)`. Preserve the `CUDA_STANDARD` properties, the include/link directives, and the `enable_language(CUDA)` switch.
- **Modify:** `CMakeLists.txt:213-217` — remove `IOM_ROCM_ENABLED=1` from the `target_compile_definitions(iom_rocm PRIVATE …)` block. Keep `__HIP_PLATFORM_AMD__`. The HIP version check (`hip_VERSION VERSION_LESS 7.2`), `enable_language(HIP)`, and the include/link directives stay.
- **Modify:** `CMakeLists.txt:251-254` — remove `target_compile_definitions(iom_sycl PRIVATE IOM_SYCL_ENABLED=1)`. Preserve the `add_library(iom_sycl …)`, the include directives (including the SYCL include path), `-fsycl` compile/link options, and the `target_link_libraries(iom_sycl PRIVATE libiom)` link.
- **Modify:** `include/iom/alloc.hpp:26` — delete the `[[nodiscard]] bool owns(void* ptr) const;` declaration inside `SingleBufferAllocatorBase`. Preserve the surrounding protected helpers (`ptr_from_addr`, `addr_from_ptr`, `offset_of`, `align_up_addr`), the public `capacity()`/`align()` accessors, and every member (`raw_begin_`, `raw_end_`, `begin_`, `align_`).
- **Modify:** `src/alloc.cpp:53-56` — delete the `SingleBufferAllocatorBase::owns(void* ptr) const` definition. No include changes; `addr_from_ptr` remains in use by `offset_of`.
- **Read:** `src/alloc.cpp:58-64` — `SingleBufferAllocatorBase::offset_of` already throws `std::invalid_argument` for the same out-of-range case `owns()` was a predicate for, so callers do not lose the containment check.
- **Read:** `test/test_alloc.cpp:90`, `:202`, `:314` — existing `std::bad_alloc` and `std::runtime_error` exhaustion assertions. None depend on `owns()`.
- **Tests:** `test/CMakeLists.txt` — no change required.

## Requirements

- The `iom` executable and `src/main.cpp` are removed; no replacement CLI is introduced.
- `CMakeLists.txt` no longer contains the strings `IOM_TTNN_ENABLED`, `IOM_CUDA_ENABLED`, `IOM_ROCM_ENABLED`, or `IOM_SYCL_ENABLED`. `__HIP_PLATFORM_AMD__` still appears at `CMakeLists.txt:213-217`.
- `SingleBufferAllocatorBase::owns(void*)` and its definition are removed. Every other member and accessor of `SingleBufferAllocatorBase`, `LinearAllocator`, `ListAllocator`, and `FixedSizeAllocator` is unchanged.
- No new `target_compile_definitions`, no new include, no new test case.
- The `iom_sycl` static library and the SYCL smoke/conformance test executables continue to be declared and continue to depend on `SYCL_ENABLED=ON`. `BackendKind::SYCL` remains exposed; `src/sycl/device.cpp:77`, `test/sycl/test_sycl_smoke.cpp:90`, and the rest of the SYCL surface are untouched.
- Build-time invariants: `find_package(CUDAToolkit REQUIRED MODULE)` and `enable_language(CUDA)` remain; `find_package(hip CONFIG REQUIRED PATHS "${ROCM_PATH}" NO_DEFAULT_PATH)`, the `hip_VERSION VERSION_LESS 7.2` guard, and `enable_language(HIP)` remain; `find_package(tt-nn CONFIG REQUIRED)` remains.
- `Allocator` interface, alignment contract, and backend tensor-construction behavior are unchanged.

## Non-goals

- The SYCL enumerator (`include/iom/tensor.hpp:78`), the `SYCL_ENABLED` option, the `SYCL_COMPILER` cache variable, the `IOM_SYCL_COMPILER` discovery path, the `iom_sycl` static library, the SYCL factory in `src/sycl/`, and `test/sycl/` smoke and conformance drivers are all preserved as active configuration.
- `__HIP_PLATFORM_AMD__` is preserved because it is required by AMD's HIP headers (`<hip/hip_runtime.h>` and friends) and is not a project-private flag.
- No structural changes to `Allocator`, `LinearAllocator`, `ListAllocator`, `FixedSizeAllocator`, or their backend consumers in `src/cpu/device.cpp`, `src/cuda/device.cpp`, `src/rocm/device.cpp`. Alignment behavior is unchanged.
- No new test, no test rename, no test deletion. AR-003 task `31-AR-003-expose-device-capabilities` owns the migration from hardcoded conformance driver leaf-type lists and any private `IOM_*_ENABLED` references to configure-time target presence. That migration is out of scope for this finding and is not a prerequisite for it.
- No changes to the `llama.cpp` / `llama.hpp` scratchpad (excluded by the in-file instruction and by AR-007's source list).
- No removal of the `if(SYCL_ENABLED)`, `if(TTNN_ENABLED)`, `if(CUDA_ENABLED)`, or `if(ROCM_ENABLED)` top-level gates in either `CMakeLists.txt` or `test/CMakeLists.txt`; those gates configure their respective backends and tests.

## Acceptance criteria

- [ ] `src/main.cpp` does not exist in the repository.
- [ ] No `add_executable(iom …)` and no `target_link_libraries(iom …)` block exists in `CMakeLists.txt`. A repo-wide search for `add_executable(iom\b` returns zero matches.
- [ ] `CMakeLists.txt` no longer contains the strings `IOM_TTNN_ENABLED`, `IOM_CUDA_ENABLED`, `IOM_ROCM_ENABLED`, or `IOM_SYCL_ENABLED`. The strings `IOM_SYCL_COMPILER`, `SYCL_ENABLED`, and `__HIP_PLATFORM_AMD__` still appear at their current locations.
- [ ] `include/iom/alloc.hpp` no longer declares `owns(void*)`. `src/alloc.cpp` no longer defines `SingleBufferAllocatorBase::owns`. Repo-wide search for `SingleBufferAllocatorBase::owns` returns zero matches.
- [ ] `include/iom/tensor.hpp:78` still enumerates `SYCL`. `BackendKind::SYCL` is still referenced by `src/sycl/device.cpp:77` and `test/sycl/test_sycl_smoke.cpp:90`.
- [ ] `Allocator`, `SingleBufferAllocatorBase`, `LinearAllocator`, `ListAllocator`, `FixedSizeAllocator` remain source-compatible for every existing caller in `src/cpu/device.cpp`, `src/cuda/device.cpp`, `src/rocm/device.cpp`.
- [ ] The `iom_sycl` static library and the SYCL smoke/conformance test executables continue to be declared in `CMakeLists.txt` and `test/CMakeLists.txt` and continue to depend on `SYCL_ENABLED=ON`.

## Verification

- `rg -n 'IOM_TTNN_ENABLED|IOM_CUDA_ENABLED|IOM_ROCM_ENABLED|IOM_SYCL_ENABLED' CMakeLists.txt src include test` returns zero matches.
- `rg -n 'add_executable\(iom\b' CMakeLists.txt test/CMakeLists.txt` returns zero matches.
- `rg -n 'SingleBufferAllocatorBase::owns' src include` returns zero matches.
- `rg -n '\.owns\(' src include test` returns zero matches.
- `rg -n 'BackendKind::SYCL' src include test` continues to return `src/sycl/device.cpp:77`, `test/sycl/test_sycl_smoke.cpp:90`, and any other SYCL backend reference points; `include/iom/tensor.hpp:78` enumerates `SYCL`.
- `rg -n '__HIP_PLATFORM_AMD__|IOM_SYCL_COMPILER|SYCL_ENABLED' CMakeLists.txt test/CMakeLists.txt` continues to return their existing matches.
- Local CPU configure and build (no accelerators enabled):
  - `cmake -S . -B build -DBUILD_TESTING=ON` succeeds without "unused variable" warnings on `IOM_TTNN_ENABLED`, `IOM_CUDA_ENABLED`, `IOM_ROCM_ENABLED`, or `IOM_SYCL_ENABLED`.
  - `cmake --build build -j` succeeds. The `iom` executable is not produced; only `libiom.a` and the test binaries (`iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`).
  - `ctest --test-dir build --output-on-failure` runs the three CPU tests green, with the same or higher assertion count than before the change.
- SYCL-enabled configure attempt (when the SDK is unavailable on the host, the existing `FATAL_ERROR` paths at `CMakeLists.txt:9-11,17-21,33-37,44-48` fire as before):
  - `cmake -S . -B build-sycl -DSYCL_ENABLED=ON -DSYCL_COMPILER=/nonexistent` fails with the existing "SYCL compiler does not exist" diagnostic.
  - When a real oneAPI compiler is present: `cmake --build build-sycl -j` produces `libiom_sycl.a`, `iom_sycl_smoke_tests`, `iom_sycl_conformance_tests`, and the rest of the active configuration; `ctest --test-dir build-sycl -R iom_sycl_smoke_tests --output-on-failure` runs the SYCL smoke driver against the real SYCL device with the existing assertions unchanged.
- CUDA-enabled and ROCm-enabled configure attempts follow the existing `find_package(CUDAToolkit REQUIRED MODULE)` / `find_package(hip CONFIG REQUIRED)` paths. The library targets `iom_cuda` and `iom_rocm` build as before with no `IOM_CUDA_ENABLED` / `IOM_ROCM_ENABLED` definitions attached.

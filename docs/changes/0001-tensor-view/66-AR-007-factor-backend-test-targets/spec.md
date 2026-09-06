# Factor the four per-backend CMake test blocks into one add_iom_backend_tests() function

**Order:** 66
**Priority:** P2 — required finishing work that removes the transcription pattern before the 0002 per-op test split multiplies it; prevents build-wiring drift, not optional.
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `AR-007`
**Review severity:** low
**Review verification:** verified, confidence 80

## Outcome

A single CMake function, `add_iom_backend_tests()`, creates each backend's smoke/conformance test-target pair and registers both tests in CTest. The four hand-transcribed gated regions in `test/CMakeLists.txt` become four gated calls carrying exactly the target names, sources, include directories, compile definitions, compile/link options, and libraries they carry today. The generated target and test set is identical for every configuration — `ctest -N` output diffs empty before vs after — and the incoming 0002 per-op test files grow a source list in one call instead of a new transcription block per backend.

## Current failure

- **Invariant:** build wiring for structurally identical targets is generated, not transcribed.
- **Failing path:** `test/CMakeLists.txt` contains four hand-written gated regions with byte-for-byte identical structure — `add_executable` pair, doctest source property, per-backend definitions/includes/options, `target_link_libraries`, `add_test` pair: ROCm `:52-107`, CUDA `:109-155`, SYCL `:158-227`, TTNN `:250-293` (~170 lines). Only target names, source files, and per-backend values vary. `docs/changes/0002-eltwise-binops/spec.md:108-109` requires per-operation conformance tests with "single file does not exceed 500 lines", which adds per-backend test files; under the current layout each new file is wired by hand-editing four regions, and any missed definition, include directory, or library fails only that backend's configure/build — exactly the inconsistency class the duplication invites.
- **Counting note:** the review title says "five" blocks; its evidence enumerates the four gated pair regions above. The fifth structurally similar block is the coexistence target (`:295-365`), which the review's recommended fix and this task explicitly keep hand-written.
- **Impact:** low at current scale; grows linearly with the 0002 test inventory.

## Scope

- Add one `add_iom_backend_tests()` function to `test/CMakeLists.txt` and replace the four per-backend gated regions with gated calls whose arguments reproduce today's wiring exactly (table below).
- Affected targets (names unchanged): `iom_rocm_smoke_tests`, `iom_rocm_conformance_tests`, `iom_cuda_smoke_tests`, `iom_cuda_conformance_tests`, `iom_sycl_smoke_tests`, `iom_sycl_conformance_tests`, `iom_ttnn_smoke_tests`, `iom_ttnn_conformance_tests`, and their eight same-named CTest registrations.
- Backend gates stay exactly `if(ROCM_ENABLED)`, `if(CUDA_ENABLED)`, `if(SYCL_ENABLED)`, `if(TTNN_ENABLED)` around the calls, in the current file positions; the regions are replaced in place, so CTest enumeration order (rocm pair, cuda pair, sycl pair, CPU conformance, ttnn pair, coexistence) is preserved.
- The change is a behavior-preserving extraction: no compile flag, link list, source property, gate, target name, or test name may change.

## Implementation references

- **Modify:** `test/CMakeLists.txt` — insert the function immediately above the `if(ROCM_ENABLED)` gate (line 52) so the CPU targets at `:1-50` are untouched; replace the four regions `:52-107`, `:109-155`, `:158-227`, `:250-293` with the gated calls below.
- **Read:** the four regions themselves — they are the authoritative source of every per-backend value; the values table in Requirements is transcribed from them.
- **Read:** `test/rocm/test_rocm_conformance.cpp:675` — the driver defines its own `main()` (watchdog child re-exec followed by `doctest::Context`), which is why the ROCm conformance source property is `DOCTEST_CONFIG_IMPLEMENT` instead of `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`; this is load-bearing (a generated second main would be a duplicate-symbol link error) and must survive the extraction as an explicit parameter.
- **Read:** `docs/changes/0002-eltwise-binops/spec.md:108-109` — the per-op test split is the growth vector; the function exposes multi-value `SMOKE_SOURCES`/`CONFORMANCE_SOURCES` for it, with explicit `SMOKE_MAIN_SOURCE`/`CONFORMANCE_MAIN_SOURCE` to keep the doctest `main()` on the driver TU only when per-op files join the executable.
- **Tests:** verification is CTest enumeration identity (`ctest -N`) per configuration, per the review's verification method; no new test code.

## Requirements

- Define one function at the top of the replaced region:

  ```cmake
function(add_iom_backend_tests name)
    cmake_parse_arguments(PARSE_ARGV 1 arg
        ""
        "CONFORMANCE_DOCTEST_CONFIG"
        "SMOKE_MAIN_SOURCE;CONFORMANCE_MAIN_SOURCE;SMOKE_SOURCES;CONFORMANCE_SOURCES;COMPILE_DEFINITIONS;SMOKE_INCLUDE_DIRS;CONFORMANCE_INCLUDE_DIRS;COMPILE_OPTIONS;LINK_OPTIONS")
  ```

- Default: when `CONFORMANCE_DOCTEST_CONFIG` is not passed (`if(NOT DEFINED arg_CONFORMANCE_DOCTEST_CONFIG)`), it defaults to `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`. It exists solely because the ROCm conformance driver supplies its own `main()`.
- Hard-coded structure the function applies to every caller (not parameterizable):
  - Smoke target `iom_${name}_smoke_tests` from `${arg_SMOKE_SOURCES}`; conformance target `iom_${name}_conformance_tests` from `${arg_CONFORMANCE_SOURCES}`.
  - `set_source_files_properties` is applied only to the driver TU that supplies the doctest `main()`: `${arg_SMOKE_MAIN_SOURCE}` gets `COMPILE_DEFINITIONS DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`; `${arg_CONFORMANCE_MAIN_SOURCE}` gets `COMPILE_DEFINITIONS ${arg_CONFORMANCE_DOCTEST_CONFIG}`. Every additional source in `${arg_SMOKE_SOURCES}` / `${arg_CONFORMANCE_SOURCES}` (the per-operation files the 0002 split will append) is added to the executable *without* a doctest main definition; otherwise a second TU that `#include`s `<doctest/doctest.h>` would pull a duplicate `main()` into the target and break the link. Every call must therefore pass the driver TU explicitly via `SMOKE_MAIN_SOURCE` and `CONFORMANCE_MAIN_SOURCE` even though each list currently contains only that one source.
  - Both targets link `doctest::doctest`, `${arg_BACKEND_LIBRARY}`, then the runtime libraries; the conformance target links `libiom` between the backend library and the runtime libraries, exactly as the four current conformance targets do. The composed link lists are therefore byte-identical to the four current `target_link_libraries` lists:
    - rocm smoke = `doctest::doctest, iom_rocm, hip::host`; rocm conformance = `doctest::doctest, iom_rocm, libiom, hip::host` (`test/CMakeLists.txt:70-75, 97-103`).
    - cuda smoke = `doctest::doctest, iom_cuda, CUDA::cuda_driver`; cuda conformance = `doctest::doctest, iom_cuda, libiom, CUDA::cuda_driver` (`:123-128, 145-151`).
    - sycl smoke = `doctest::doctest, iom_sycl`; sycl conformance = `doctest::doctest, iom_sycl, libiom` (`:183-187, 218-223`).
    - ttnn smoke = `doctest::doctest, iom_ttnn, TT::Metalium, TTNN::TTNN`; ttnn conformance = `doctest::doctest, iom_ttnn, libiom, TT::Metalium, TTNN::TTNN` (`:259-265, 282-289`).
  Concretely: smoke links `doctest::doctest ${arg_BACKEND_LIBRARY} ${arg_RUNTIME_LIBRARIES}` and conformance links `doctest::doctest ${arg_BACKEND_LIBRARY} libiom ${arg_RUNTIME_LIBRARIES}`. `BACKEND_LIBRARY` is a single-value keyword (one backend lib per call); `RUNTIME_LIBRARIES` is a multi-value keyword. SYCL has no runtime libraries, so the SYCL call passes `RUNTIME_LIBRARIES` as empty; the function must therefore guard the runtime-libraries append with `if(arg_RUNTIME_LIBRARIES) target_link_libraries(... ${arg_RUNTIME_LIBRARIES}) endif()` to keep the composed list byte-identical (a trailing empty item in `target_link_libraries` is not guaranteed accepted across CMake versions and would differ from today's text).
  - `add_test(NAME <target> COMMAND <target>)` for both targets: test registration is owned by the function, and test names equal target names exactly as today.
  - Apply `COMPILE_DEFINITIONS`, `SMOKE_INCLUDE_DIRS`/`CONFORMANCE_INCLUDE_DIRS`, `COMPILE_OPTIONS`, `LINK_OPTIONS`, and `RUNTIME_LIBRARIES` as `PRIVATE` `target_compile_definitions` / `target_include_directories` / `target_compile_options` / `target_link_options` / `target_link_libraries` to the corresponding target(s). Guard each application with `if(<list>)` so an omitted/empty list is legal (CMake errors on a bare `target_*` keyword with no values) — CUDA/TTNN pass no definitions, TTNN passes no smoke include dirs or options, SYCL passes no runtime libraries.
  - Relative source and include paths resolve against the calling directory (`test/`), same as today; the function is only ever called from `test/CMakeLists.txt`.
- Replace each region in place with exactly these gated calls (keyword order canonical; values are the current wiring, transcribed verbatim):
```cmake
if(ROCM_ENABLED)
    add_iom_backend_tests(rocm
        SMOKE_MAIN_SOURCE rocm/test_rocm_smoke.cpp
        CONFORMANCE_MAIN_SOURCE rocm/test_rocm_conformance.cpp
        SMOKE_SOURCES rocm/test_rocm_smoke.cpp
        CONFORMANCE_SOURCES rocm/test_rocm_conformance.cpp
        COMPILE_DEFINITIONS __HIP_PLATFORM_AMD__
        SMOKE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/rocm
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
        BACKEND_LIBRARY iom_rocm
        RUNTIME_LIBRARIES hip::host
        CONFORMANCE_DOCTEST_CONFIG DOCTEST_CONFIG_IMPLEMENT
    )
endif()

if(CUDA_ENABLED)
    add_iom_backend_tests(cuda
        SMOKE_MAIN_SOURCE cuda/test_cuda_smoke.cpp
        CONFORMANCE_MAIN_SOURCE cuda/test_cuda_conformance.cpp
        SMOKE_SOURCES cuda/test_cuda_smoke.cpp
        CONFORMANCE_SOURCES cuda/test_cuda_conformance.cpp
        SMOKE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/cuda
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
        BACKEND_LIBRARY iom_cuda
        RUNTIME_LIBRARIES CUDA::cuda_driver
    )
endif()

if(SYCL_ENABLED)
    add_iom_backend_tests(sycl
        SMOKE_MAIN_SOURCE sycl/test_sycl_smoke.cpp
        CONFORMANCE_MAIN_SOURCE sycl/test_sycl_conformance.cpp
        SMOKE_SOURCES sycl/test_sycl_smoke.cpp
        CONFORMANCE_SOURCES sycl/test_sycl_conformance.cpp
        SMOKE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/sycl ${IOM_SYCL_INCLUDE_DIR}
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/sycl ${IOM_SYCL_INCLUDE_DIR}
        COMPILE_OPTIONS -fsycl
        LINK_OPTIONS -fsycl
        BACKEND_LIBRARY iom_sycl
        RUNTIME_LIBRARIES
    )
endif()

if(TTNN_ENABLED)
    add_iom_backend_tests(ttnn
        SMOKE_MAIN_SOURCE ttnn/test_ttnn_smoke.cpp
        CONFORMANCE_MAIN_SOURCE ttnn/test_ttnn_conformance.cpp
        SMOKE_SOURCES ttnn/test_ttnn_smoke.cpp
        CONFORMANCE_SOURCES ttnn/test_ttnn_conformance.cpp
        CONFORMANCE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        BACKEND_LIBRARY iom_ttnn
        RUNTIME_LIBRARIES TT::Metalium TTNN::TTNN
    )
endif()
```
- Per-value provenance, so any future divergence is visible: `__HIP_PLATFORM_AMD__` from `:61-64`/`:86-89`; `${PROJECT_SOURCE_DIR}/src/rocm` from `:65-68`; `${PROJECT_SOURCE_DIR}/src/cuda` from `:119-122`; SYCL `-fsycl` compile+link options from `:173-181`/`:208-216`; `${IOM_SYCL_INCLUDE_DIR}` from `:167-171`/`:200-206` (the variable is only defined under `SYCL_ENABLED` at the top-level `CMakeLists.txt:27-29`, which is why the SYCL call must remain inside its gate); TTNN's empty smoke include set from `:251-267`; each `BACKEND_LIBRARY`/`RUNTIME_LIBRARIES` pair from the corresponding `target_link_libraries` blocks (`:70-75, 97-103, 123-128, 145-151, 183-187, 218-223, 259-265, 282-289`).
- No other line of `test/CMakeLists.txt` changes: the CPU targets (`:1-50`), `iom_backend_conformance_cpu_tests` (`:229-248`), and the coexistence block (`:295-365`) are untouched.
- `cmake_minimum_required` is 3.21 (`CMakeLists.txt:1`); `cmake_parse_arguments(PARSE_ARGV ...)` needs no version bump.

## Non-goals

- The coexistence target stays hand-written and explicit; extending it to SYCL is CC-003's remediation (`53-CC-003-add-sycl-backend-coexistence`), and this task must not absorb or pre-apply it.
- No change to the CPU-side targets (`iom_tests`, `iom_cpu_tests`, `iom_cpu_bench`, `iom_backend_conformance_cpu_tests`): they have no smoke/conformance pair and do not fit the backend-pair shape.
- No compile-flag, link-list, source-property, gate, or test-name changes beyond relocating them into function arguments; in particular the ROCm `DOCTEST_CONFIG_IMPLEMENT` and TTNN's empty smoke include set are preserved as-is, not "normalized".
- Do not create the 0002 per-op test files or wire any future source list; the multi-value source parameters are the prepared extension point, and consuming it belongs to the 0002 tasks.
- No CMake version bump, no new helper modules, no second abstraction over the coexistence branches or the CPU targets.

## Acceptance criteria

- [ ] `test/CMakeLists.txt` defines exactly one `add_iom_backend_tests()` function; the four per-backend regions are single gated calls matching the verbatim blocks in Requirements; all other lines of the file are unchanged.
- [ ] All eight backend target names, all eight CTest registrations, and the per-target `target_link_libraries` item lists are byte-identical to before. Verified mechanically by `diff` of `build*/test/CMakeFiles/iom_<backend>_<kind>_tests.dir/link.txt` (the link command CMake generates) before vs after the refactor; the file is regenerated by every CMake configure and contains exactly the library order CMake will pass to the linker.
- [ ] All eight backend target names and all eight CTest registrations appear in the same order as before in `ctest -N` output (i.e., rocm smoke/conformance, cuda smoke/conformance, sycl smoke/conformance, CPU conformance, ttnn smoke/conformance, coexistence — preserved by replacing the four regions in place).
- [ ] For each accelerator configuration (CUDA, ROCm, SYCL, TTNN enabled), configure and build succeed on the corresponding remote profile, the `ctest -N` name list (order preserved, unsorted) is identical to the pre-refactor configuration's list on the same host, and the corresponding `link.txt` files are identical (no library-order drift).
- [ ] CPU-only local configure, build, and `ctest -N` are unchanged (the four always-on tests), and `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` still pass — `iom_cpu_bench` is excluded as the known NT-001 red gate, not affected by this task.
- [ ] Each backend's smoke and conformance executables still link and run on their profile (run `iom_<profile>_smoke_tests` per remote profile), proving the composed link lists resolve.
## Verification

Local, CPU-only (no options) — proves the always-on wiring is untouched, the file still configures, and the link command for the always-on targets is byte-identical:

```text
cmake -S . -B /tmp/iom-66-before
# enumerate tests in registration order (order preserved, no sort)
ctest --test-dir /tmp/iom-66-before -N \
  | awk '/Test #[0-9]+:/ {sub(/^.*Test #[0-9]+: /,""); print}' \
  > /tmp/tests-order-before.txt
# enumerate tests as an unordered set (sorted, for set equality)
ctest --test-dir /tmp/iom-66-before -N \
  | awk '/Test #[0-9]+:/ {sub(/^.*Test #[0-9]+: /,""); print}' \
  | sort > /tmp/tests-set-before.txt
# capture the generated link command for the always-on CPU targets
for t in iom_tests iom_cpu_tests iom_cpu_bench iom_backend_conformance_cpu_tests; do
    cp /tmp/iom-66-before/test/CMakeFiles/${t}.dir/link.txt /tmp/link-${t}-before.txt
done
# apply the refactor
cmake -S . -B /tmp/iom-66-after
ctest --test-dir /tmp/iom-66-after -N \
  | awk '/Test #[0-9]+:/ {sub(/^.*Test #[0-9]+: /,""); print}' \
  > /tmp/tests-order-after.txt
ctest --test-dir /tmp/iom-66-after -N \
  | awk '/Test #[0-9]+:/ {sub(/^.*Test #[0-9]+: /,""); print}' \
  | sort > /tmp/tests-set-after.txt
for t in iom_tests iom_cpu_tests iom_cpu_bench iom_backend_conformance_cpu_tests; do
    cp /tmp/iom-66-after/test/CMakeFiles/${t}.dir/link.txt /tmp/link-${t}-after.txt
done
# set equality
diff /tmp/tests-set-before.txt /tmp/tests-set-after.txt        # must be empty
# order preservation (unsorted list)
diff /tmp/tests-order-before.txt /tmp/tests-order-after.txt    # must be empty
# link command byte-identity (always-on targets)
for t in iom_tests iom_cpu_tests iom_cpu_bench iom_backend_conformance_cpu_tests; do
    diff /tmp/link-${t}-before.txt /tmp/link-${t}-after.txt    # must be empty
done
cmake --build /tmp/iom-66-after -j
ctest --test-dir /tmp/iom-66-after -R 'iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests'   # must pass
```

Command-to-claim map:
- `ctest -N | awk … > /tmp/tests-order-*.txt` then `diff` (unsorted, line-for-line) — proves the **names-and-order** acceptance criterion (set of names unchanged AND `ctest -N` enumeration order unchanged).
- `ctest -N | awk … | sort > /tmp/tests-set-*.txt` then `diff` — proves **set equality** (names present; weak — only used as a secondary check).
- `for t in …; cp …/link.txt …; diff …` — proves **link-order byte-identity** for each `target_link_libraries` invocation (the only command that can prove "library list byte-identical to today", since CMake regenerates `link.txt` on every configure and `ctest -N` says nothing about link order).
- `cmake --build` + `ctest -R …` — proves **build + run** of the always-on targets is unaffected.

Accelerator configurations — the review's verification method (`ctest -N` target/test sets identical before and after) requires the real gates, which never open on a CPU-only host. Follow `.agents/skills/remote-development` per profile (`cuda`, `rocm`, `sycl`, `ttnn` from `.remote-hosts.conf`; `remote-sync` → `remote-exec` → `remote-clean`, one task directory per backend, unique task id). No CPU-only evidence substitutes for this. `ctest -N` enumerates without running, but configure/build need each toolchain. Per profile, with a unique task id:

1. `remote-sync <profile> <task-id>` with the unmodified tree; `remote-exec`: configure (`-DCUDA_ENABLED=ON`, `-DROCM_ENABLED=ON`, `-DSYCL_ENABLED=ON` after `source /opt/intel/oneapi/setvars.sh`, or `-DTTNN_ENABLED=ON` respectively), build all targets, then capture `ctest -N` *order*, `ctest -N` *set*, and the `link.txt` for every enabled backend target:
   ```text
   cmake -S . -B /tmp/iom-66-before -D<profile flag>
   cmake --build /tmp/iom-66-before -j
   ctest --test-dir /tmp/iom-66-before -N \
     | awk '/Test #[0-9]+:/ {sub(/^.*Test #[0-9]+: /,""); print}' \
     > /tmp/tests-order-before.txt
   ctest --test-dir /tmp/iom-66-before -N \
     | awk '/Test #[0-9]+:/ {sub(/^.*Test #[0-9]+: /,""); print}' \
     | sort > /tmp/tests-set-before.txt
   for t in iom_<profile>_smoke_tests iom_<profile>_conformance_tests; do
       cp /tmp/iom-66-before/test/CMakeFiles/${t}.dir/link.txt /tmp/link-${t}-before.txt
   done
   if [ -f /tmp/iom-66-before/test/CMakeFiles/iom_backend_coexistence_tests.dir/link.txt ]; then
       cp /tmp/iom-66-before/test/CMakeFiles/iom_backend_coexistence_tests.dir/link.txt \
          /tmp/link-iom_backend_coexistence_tests-before.txt
   fi
   ```
2. Land the refactor; `remote-sync` again; `remote-exec`: configure into a fresh build dir with the same flag, build all targets, capture the three artifacts the same way into `*-after.*`, and diff:
   ```text
   diff /tmp/tests-order-before.txt /tmp/tests-order-after.txt                 # empty
   diff /tmp/tests-set-before.txt /tmp/tests-set-after.txt                     # empty
   for t in iom_<profile>_smoke_tests iom_<profile>_conformance_tests; do
       diff /tmp/link-${t}-before.txt /tmp/link-${t}-after.txt                 # empty
   done
   if [ -f /tmp/iom-66-after/test/CMakeFiles/iom_backend_coexistence_tests.dir/link.txt ]; then
       diff /tmp/link-iom_backend_coexistence_tests-before.txt \
            /tmp/link-iom_backend_coexistence_tests-after.txt                  # empty
   fi
   ```
   Each of the four `diff`s must be empty. The first proves order preservation; the second proves set equality; the third proves per-backend link-order byte-identity; the fourth proves coexistence link-order byte-identity (the coexistence block was untouched, so this last diff is a sanity check that the refactor had no transitive effect).
3. Run the backend smoke suite once per profile (`ctest --test-dir <dir> -R iom_<profile>_smoke_tests` with exclusive device access, e.g. `flock /tmp/agent-gpu0.lock`) to confirm the extracted link wiring resolves at runtime.
4. `remote-clean <profile> <task-id>`.

Expected observation: every `diff` above empty, every build clean, every smoke suite passing — the refactor is invisible to CTest enumeration (names and order), to the generated link command for every backend target, and to runtime behavior.

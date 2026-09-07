# iom_tests cannot build: missing iom/llama.hpp include and iom::models test

**Order:** 01
**Priority:** P0 — build break gates test verification
**Blocked by:** None
**Review source:** `cpp-inference-code-review` — `whole-codebase, clean main HEAD de82efc588ba0247849cd8a6567f741eb0c3308f`
**Finding:** CC-003
**Review area:** Contract & correctness
**Review severity:** high
**Review verification:** verified, confidence 95
**Review scope:** whole-codebase
**Backend scope:** common
**Location:** `test/test_iom.cpp:31` (`#include "iom/llama.hpp"`); `test/test_iom.cpp:2144-2257` (`TEST_CASE "DeviceOps queue drives the llama models through owner views"`)

## Outcome

The `iom_tests` target compiles and links in any testing-enabled configure and its remaining suite passes; the repository contains no reference to `iom::models` symbols that have no production definition.

## Current problem

With `BUILD_TESTING` on, `iom_tests` is registered unconditionally (`test/CMakeLists.txt:3-6`) with `test_iom.cpp` as its only source. `libiom`'s sole PUBLIC include root is `${PROJECT_SOURCE_DIR}/include` (`CMakeLists.txt:104-107`), and `include/iom/` contains no `llama.hpp`, so the include at `test/test_iom.cpp:31` is a guaranteed compile failure. The `TEST_CASE` at `test/test_iom.cpp:2144-2257` additionally uses `iom::models::LlamaRoPE` (`:2154`), `LlamaAttention`/`LlamaMlp`/`LlamaDecoder` (`:2212-2214`), and `Llama2Model` (`:2216`); no definition exists anywhere in `src/` or `include/` (repo-wide grep for `namespace models|LlamaRoPE|LlamaAttention|LlamaDecoder|Llama2Model|LlamaMlp` over `src;include` returns no matches), and `libiom` builds only `iom.cpp`, `cpu/device.cpp`, `mmap.cpp`, `safetensors.cpp`, `alloc.cpp` (`CMakeLists.txt:75-80`) with no generated-file step — so even with the include deleted the target fails at link. The only matches for `**/*llama*` are stale build artifacts `build/*/CMakeFiles/libiom.dir/src/llama.cpp.o` and `.o.d`, showing a `src/llama.cpp` existed historically and was removed. No other llama/models reference exists in `CMakeLists.txt`, `docs`, `.agents`, `include/`, `src/`, `test/`, or `README.md`, and no pending spec restores the model. Impact: any developer/CI enabling tests hits an unconditional compile break, and the dead test obscures the current contract (only `Block::forward` remains of the model layer, `include/iom/iom.hpp:386-388`).

## Scope

- Delete the stale include at `test/test_iom.cpp:31` and the entire llama-model `TEST_CASE` at `test/test_iom.cpp:2144-2257` — the only `iom::models` references in the repository.
- Leave the `iom_tests` target registration (`test/CMakeLists.txt:3-6`) and the `libiom` target unchanged.
- Result: no llama/models reference remains anywhere outside build artifacts.

## Implementation references

- **Modify:** `test/test_iom.cpp` — delete line 31 and the `TEST_CASE` block at `:2144-2257`; clean cutover with no shim, stub header, or conditional compilation.
- **Read:** `test/CMakeLists.txt:3-6` — confirms `iom_tests` is registered unconditionally from `test_iom.cpp`; `include/iom/` directory listing — confirms no `llama.hpp` exists; `src/iom.cpp` — current model-layer surface is only `Block::forward`.
- **Tests:** the remaining `iom_tests` suite (`test/test_iom.cpp`) — must compile, link, and pass unchanged after the deletion.

## Requirements

- Remove the `#include "iom/llama.hpp"` line at `test/test_iom.cpp:31`.
- Remove the entire `TEST_CASE "DeviceOps queue drives the llama models through owner views"` body at `test/test_iom.cpp:2144-2257`.
- Do not add a placeholder model layer, stub header, conditional compilation guard, or macro gate: the deleted symbols must simply not be referenced.
- After deletion, a repo-wide search (excluding build artifacts) must find no `llama`/`models` reference in `test/`.

## Non-goals

- Re-adding or vendoring a model implementation.
- Gating the include or `TEST_CASE` behind a macro or backend condition.
- Converting the test into a stub that asserts nothing.
- Touching `test/CMakeLists.txt`, the `libiom` target, or any production source.

## Acceptance criteria

- [ ] `cmake -S . -B build -DIOM_ENABLE_TESTING=ON && cmake --build build --target iom_tests` succeeds (compile and link).
- [ ] `ctest --test-dir build -R iom_tests` passes with the remaining suite.
- [ ] `grep -rn llama test/` returns nothing; no `iom::models` symbol reference remains anywhere in `test/`.

## Verification

Actual validation (root-run, recorded): on remote host `bv2` with a CPU-only configure, CMake configure succeeded and `cmake --build` failed exactly at `test/test_iom.cpp:31` with a fatal error on `#include "iom/llama.hpp"` (no such file). The same configure successfully built `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, and `iom_cpu_bench` (the break is specific to the common `iom_tests` target).

Proposed gates (implementer, after the edit): run the configure/build/ctest commands in the acceptance criteria on a CPU-only build (local host or remote `bv2` when available), then the repo-wide `grep -rn llama test/` check.

- `cmake -S . -B build -DBUILD_TESTING=ON && cmake --build build --target iom_tests && ctest --test-dir build -R '^iom_tests$' --output-on-failure`
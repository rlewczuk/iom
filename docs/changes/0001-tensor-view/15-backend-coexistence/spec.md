# Verify independent backend builds and coexistence

**Order:** 15
**Priority:** P2 — this is the required cross-cutting build/link/runtime completion gate after all individual backends pass.
**Blocked by:** `08-rocm-storage-copy`, `10-cuda-storage-copy`, `12-sycl-storage-copy`, `14-ttnn-storage-copy`
**Source:** `docs/changes/0001-tensor-view/spec.md`

## Outcome

Core-only, each individual optional backend, and the all-backend configuration build as independent targets; one executable links and uses CPU, CUDA, ROCm, SYCL, and TTNN devices together without duplicate symbols or shared active-backend state.

## Scope

- Finish the CMake option matrix and add a combined compile/link/runtime coexistence test.
- Re-run each backend's already-defined hardware conformance target individually before the combined test; do not merge their implementations or tests.

## Implementation references

- **Modify:** `CMakeLists.txt` — ensure `CUDA_ENABLED`, `ROCM_ENABLED`, `SYCL_ENABLED`, and `TTNN_ENABLED` are four independent options and each controls only its own library/dependencies.
- **Modify:** `test/CMakeLists.txt` — compose a coexistence executable from exactly the enabled backend libraries.
- **Create (planned):** `test/backend/test_backend_coexistence.cpp` — include all enabled public factory headers, construct devices together, and interleave independent tensor/queue use.
- **Read:** `include/iom/{cpu,cuda,rocm,sycl,ttnn}/device.hpp` — use factories directly; do not add a registry, singleton, or `BackendKind` switch.
- **Tests:** `test/backend/backend_conformance.hpp` and each `test/<backend>/test_<backend>_conformance.cpp` — retain individual hardware gates rather than duplicating conformance logic in the coexistence test.

## Requirements

- `libiom` always contains common code and CPU only. Each accelerator is a separate optional library with its own public factory, runtime links, compiler settings, and tests.
- No option disables another option and no `if/elseif` chain makes backend selection exclusive. Disabling one backend removes only that backend's library, factory symbols, dependencies, and tests.
- Common sources and public common headers include no CUDA, HIP, SYCL, or TTNN types. Linking several optional libraries introduces no duplicate common symbols or static initialization that selects an active backend.
- Preserve the staged verification history: core-plus-CPU and each single-backend configuration must still configure, build, link, and run its applicable tests before the all-enabled configuration.
- The combined executable links `libiom`, `iom_cuda`, `iom_rocm`, `iom_sycl`, and `iom_ttnn` directly and compile-checks all five public factories in one translation unit.
- On hardware with all runtimes enabled, construct at least one device and queue from every backend in the same process. Use a common supported `BF16` tensor shape, perform host transfer and same-device asynchronous copy on each backend, interleave submissions, wait on each originating queue, and compare logical results.
- Construct multiple devices/queues where hardware permits and prove each reports its own kind/ordinal. A view from another `Device`, including the same backend and ordinal, must be rejected by the receiving queue.
- Queue IDs remain process-unique across backend types and can be released/reused without creating a backend-global selector. Per the direct token decision, stale-token collision after queue destruction remains a caller-invalid case and is not a generation check.
- Enabled runtime/hardware tests must fail rather than silently skip. Core-only and single-backend builds must not require unavailable disabled runtimes.
- Do not add persisted tensor formats, rollback paths, dispatch abstractions, or compatibility overloads as part of build integration.

## Non-goals

- Cross-device or cross-backend `DeviceOps::copy`; explicit host staging remains the caller path.
- Numerical compute conformance, performance comparison, or automatic backend selection.
- A global device registry, plugin loader, borrowed runtime context, or monolithic accelerator target.

## Acceptance criteria

- [ ] Core-only and each of the four single-backend configurations still pass their focused targets with other backends disabled.
- [ ] All four options configure simultaneously and build five distinct libraries plus `iom_backend_coexistence_tests` without duplicate symbols.
- [ ] The combined hardware test constructs and uses every backend in one process, interleaves queues, and returns bit-identical `BF16` logical results.
- [ ] Backend kind, ordinal, exact-device validation, and process-wide queue-ID uniqueness remain correct under coexistence.
- [ ] Turning any one option off removes only that backend's targets/dependencies and leaves every remaining enabled combination linkable.
- [ ] No enabled test silently skips because a requested runtime or device is absent.

## Verification

- `cmake -S . -B build/core-only -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/core-only --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/core-only --output-on-failure`
- `cmake -S . -B build/all-backends -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=ON -DSYCL_ENABLED=ON -DTTNN_ENABLED=ON -DCUDA_PATH=/usr/local/cuda -DROCM_PATH=/opt/rocm`
- `cmake --build build/all-backends --target iom_backend_coexistence_tests`
- `ctest --test-dir build/all-backends --output-on-failure -R '^iom_backend_coexistence_tests$'`

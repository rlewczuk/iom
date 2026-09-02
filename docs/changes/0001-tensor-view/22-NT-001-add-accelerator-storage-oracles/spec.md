# Add independent accelerator storage/native-plane oracles to the conformance harness

**Order:** 22
**Priority:** P2 — required finishing work: independent cross-cutting verification of accelerator physical/native layout (P2 is not optional)
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-001`
**Review severity:** medium
**Review verification:** strongly-supported, confidence 93

## Outcome

Each accelerator conformance driver (CUDA, ROCm, TTNN) seeds and observes candidate storage through a path that does not share code with the candidate's `copy_from_host` / `copy_to_host`. A negative fixture that perturbs the candidate transfer map identically in both directions now fails the harness, and the corrected oracle covers every leaf width, padded shape, and transformed view already exercised by the shared suite.

## Current failure

`test/backend/backend_conformance_copy_storage.hpp:274-326` (`run_storage_and_transfer_conformance`) and `:332-445` (`run_async_copy_conformance`) seed and read each candidate tensor through the same backend transfer pair. The shared `require_logical_bytes` (`test/backend/backend_conformance_common.hpp:158-166`) always calls the candidate view's own `copy_to_host`. A consistently wrong plane map, tile map, sub-byte packing, or endian transform in both directions cancels through a round trip. CUDA and ROCm derive their scatter/gather planes from the backend's `view_planes` (`src/cuda/copy.cu:62`, `src/rocm/copy.hip:81`); TTNN uses `ttnn_detail::owner_plane_at` for both directions (`src/ttnn/copy.cpp:37, 172, 194, 207`). The CPU-only perturbation fixture at `test/cpu/test_cpu_conformance.cpp:167-212` flips a byte of the correct CPU allocation but never exercises an accelerator whose transfer map is itself wrong. A backend can pass every shared logical case while violating the standard layout (`docs/changes/0001-tensor-view/spec.md:380-385`) or the TTNN logical-plane mapping (`docs/changes/0001-tensor-view/14-ttnn-storage-copy/spec.md:31`); the defect surfaces later when a compute kernel or external native operation follows the specified mapping rather than the matching erroneous transfer implementation.

## Scope

- Add an independent storage observer/seeder per accelerator backend driver that the shared harness can invoke alongside the existing logical round-trip helpers.
- Add one negative fixture that swaps two adjacent elements of the candidate's logical-to-native map, applies the same swap to both directions, and must be detected by the new oracle while still passing the existing logical round trip.
- Run the corrected oracle over the existing leaf-width table, the padded owner shapes (`{17, 33}`, `{2, 3, 16, 16}`, `{2, 3, 4, 17, 33}`, `{2, 2, 2, 3, 17, 33}`), and every transformed view from `view_cases_for`.
- Do not modify `copy_from_host`, `copy_to_host`, or `DeviceOps::copy` semantics; do not weaken any accepted accelerator transfer contract. This task strengthens the harness only.

## Implementation references

- **Modify:** `test/backend/backend_conformance_copy_storage.hpp` — add `run_storage_oracle_conformance(devices, supported_types, observer)` that, for every `(type, dimensions)` pair from `transfer_owner_shapes()` and every `ViewCase` from `view_cases_for(spec)`, writes the candidate through its own `copy_from_host`, then reads the candidate through the new per-backend oracle and compares bytes bit-for-bit against `encode_logical(view.spec(), salt)` exactly as `require_logical_bytes` does today.
- **Create:** `test/backend/backend_conformance_oracle.hpp` — declare a backend-neutral `AcceleratorStorageOracle` abstract base with `seed(view, encoded)` and `observe(view) -> std::vector<std::byte>` virtuals. Provide a test-side standard-layout encoder `encode_standard_tiled_storage(const TensorSpec&)` that walks every padded coordinate in the order documented in `docs/changes/0001-tensor-view/02-core-metadata-layout/spec.md:34` and emits `tiled_storage_nbytes()` bytes (host-side, no accelerator header). Reuse `iom::detail::standard_layout_slot` (`src/iom.cpp:208`, exposed at `include/iom/tensor.hpp:119`) for every slot address; do not re-derive tile math in tests.
- **Modify:** `test/cuda/test_cuda_conformance.cpp` — implement `CudaStorageOracle` over the `iom::make_cuda_device(0, ...)` candidate. For `seed`, copy the encoded host buffer through `cudaMemcpy` into `view.native_handle()` spanning exactly `view.spec().tiled_storage_nbytes()` (the CUDA allocator at `src/cuda/device.cpp:101` returns the full standard allocation). For `observe`, allocate one host buffer of `tiled_storage_nbytes()`, `cudaMemcpy` it back from `view.native_handle()`, then compare against the test-side encoder. Synchronize the CUDA context (`cudaStreamSynchronize(0)`) before and after the device-to-host copy.
- **Modify:** `test/rocm/test_rocm_conformance.cpp` — implement `HipStorageOracle` identically using `hipMemcpy` over the full `tiled_storage_nbytes()` allocation from `iom::make_rocm_device(0, ...)`. Synchronize with `hipDeviceSynchronize()` before and after.
- **Modify:** `test/ttnn/test_ttnn_conformance.cpp` — implement `TtnnStorageOracle` that does not call `copy_from_host`/`copy_to_host`. For `seed`, iterate the owner planes exposed via `std::vector<ttnn::Tensor>* planes = static_cast<std::vector<ttnn::Tensor>*>(view.native_handle())` (the TTNN owner stores its planes there at `src/ttnn/device.cpp:166-168, 187`), decompose the encoded bytes per `ttnn_detail::owner_plane_at(view, i)` (`src/ttnn/copy.cpp:37`), and use `ttnn::Tensor::to_vector<float>()`-style direct host-buffer round-trip via `tt::tt_metal::HostBuffer` (the helper `make_host_buffer` at `src/ttnn/copy.cpp:72` already builds typed host buffers without conversion). For `observe`, do the inverse, then compare the resulting `tiled_storage_nbytes()`-aligned buffer to the test-side encoder. The oracle must not share code with `ttnn_detail::region_from_host`/`region_to_host` (`src/ttnn/copy.cpp`).
- **Modify:** `test/cpu/test_cpu_conformance.cpp` — extend the existing negative fixture (`:167-212`) into a parameterized template that the accelerator drivers reuse. Replace the direct byte flip with a permutation hook: accept a callable that maps `std::size_t logical_index -> std::size_t physical_slot` and apply it for both seed and observe. Use `iom::detail::standard_layout_slot` to materialize the correct map, then build the perturbed map by swapping the slots of logical elements `0` and `1` for the chosen leaf width. The CPU run keeps passing; the accelerator runs must fail the new oracle and still pass `require_logical_bytes`.
- **Read:** `test/cpu/test_cpu.cpp:200-310` — reuse `snapshot_storage` / `expect_storage_matches` / `first_mismatch` helpers for the test-side standard-layout encoder when comparing full allocations; keep the standard-layout encoder free of any backend include.
- **Tests:** `test/cuda/test_cuda_conformance.cpp`, `test/rocm/test_rocm_conformance.cpp`, `test/ttnn/test_ttnn_conformance.cpp` — add one `TEST_CASE` per driver named `"<backend> conformance: storage oracle identifies perturbed transfer map"` and one named `"<backend> conformance: storage oracle covers every leaf width and padded shape"` that calls `run_storage_oracle_conformance` with the driver's supported-type table. The first case must fail before the fix and pass after; the second must pass after the fix and exercise every shape in `transfer_owner_shapes()` plus every `ViewCase`.

## Requirements

- The new oracle path must not include `src/cuda/copy.cu`, `src/rocm/copy.hip`, or `src/ttnn/copy.cpp` symbols. The CPU reference oracle in `test/backend/backend_conformance_oracle.hpp` and its standard-layout encoder must compile without any accelerator header, mirroring the rules in `docs/changes/0001-tensor-view/06-shared-backend-conformance/spec.md:28`.
- `encode_standard_tiled_storage` must produce exactly `tiled_storage_nbytes()` bytes per call and agree byte-for-byte with the CPU backend's allocation for every shape in `copy_owner_shapes()` and every `ViewCase`. Verify against `snapshot_storage` from `test/cpu/test_cpu.cpp` in a CPU-only test case before the accelerator cases run.
- The CUDA and ROCm oracles must read and write the full native allocation through `view.native_handle()` and `tiled_storage_nbytes()`; they must not call `iom::TensorView::copy_from_host` / `copy_to_host` and must not consult `view_planes`. The synchronization calls (`cudaStreamSynchronize` / `hipDeviceSynchronize`) must happen on the factory-owned context/stream already used by the corresponding `iom::make_<cuda|rocm>_device` call.
- The TTNN oracle must access owner planes through the vector stored at the owner's storage handle (`src/ttnn/device.cpp:166-168`) and must not call `ttnn_detail::region_from_host`, `region_to_host`, `upload_plane`, or `download_plane` (`src/ttnn/copy.cpp`). It must compute its own plane index per plane using the same `view.spec()`, `view.plane_offset()`, and `view.plane_strides()` inputs as `ttnn_detail::owner_plane_at` and a permutation hook supplied by the negative fixture; the production `owner_plane_at` is the cross-check target, not the implementation.
- The negative fixture's permutation hook swaps two adjacent logical elements for every leaf width. Its `seed` and `observe` paths apply the same permutation. The fixture must fail `run_storage_oracle_conformance` (the new oracle must identify a wrong slot or plane within `tiled_storage_nbytes()`), and it must still pass `require_logical_bytes` against the existing logical round-trip path because the same wrong map writes and reads back.
- The accelerator sweeps must cover every type in the driver's `supported_types` table, every shape in `transfer_owner_shapes()` and `copy_owner_shapes()`, and every `ViewCase` returned by `view_cases_for`. Each case must complete before the harness advances to the next; the existing `TrafficGate` allocator rule (no allocations inside transfers/transforms/operations) remains in force.
- Failure messages must name the byte position or plane index where the oracle first diverges from the expected standard-layout encoding, mirroring the precision of `require_logical_bytes` and `first_logical_mismatch`.

## Non-goals

- Adding, removing, or modifying any production transfer or copy path. `copy_from_host`, `copy_to_host`, `DeviceOps::copy`, `ttnn_detail::region_from_host`, `region_to_host`, `upload_plane`, `download_plane`, and the CUDA/ROCm `view_planes` call sites are unchanged.
- New performance benchmarks, profiler integration, or device-sanitizer instrumentation.
- Adding or strengthening the existing CPU physical-layout tests; the CPU oracle already exists and the negative fixture extension simply generalizes the parameterization.
- A general "oracle framework" abstraction beyond `AcceleratorStorageOracle`. The shared header exposes exactly the three virtuals required for seed, observe, and the standard-layout encoder helper; do not introduce multi-tier dispatch.
- SYCL backend coverage. Spec 12 (`docs/changes/0001-tensor-view/12-sycl-storage-copy`) remains unimplemented; once landed, this task's `AcceleratorStorageOracle` interface is the contract it must satisfy, but the SYCL implementation is out of scope here.

## Acceptance criteria

- [ ] `test/backend/backend_conformance_oracle.hpp` defines `AcceleratorStorageOracle`, the standard-layout encoder, and a CPU implementation that passes `encode_standard_tiled_storage` round-trip against `snapshot_storage` for every `transfer_owner_shapes()` and `copy_owner_shapes()` shape and every leaf width.
- [ ] `run_storage_oracle_conformance` exists in `test/backend/backend_conformance_copy_storage.hpp` and is invoked by the CUDA, ROCm, and TTNN conformance drivers after the existing logical round-trip cases.
- [ ] CUDA and ROCm conformance drivers each define a `*StorageOracle` that uses `cudaMemcpy` / `hipMemcpy` directly against `view.native_handle()` for the full `tiled_storage_nbytes()` range, with explicit context/stream/device synchronization, and never calls `copy_from_host` / `copy_to_host` / `view_planes`.
- [ ] TTNN conformance driver defines `TtnnStorageOracle` that reads and writes the owner plane vector through `view.native_handle()` and constructs/deconstructs `tt::tt_metal::HostBuffer` via `make_host_buffer` without going through `region_from_host` / `region_to_host`.
- [ ] A negative fixture with an identical nontrivial permutation in both directions fails the corrected oracle on each accelerator (byte position or plane index reported), and the same fixture still passes `require_logical_bytes`.
- [ ] The corrected oracle passes on real CUDA, ROCm, and TTNN hardware for every leaf width, every shape in `transfer_owner_shapes()` and `copy_owner_shapes()`, and every transformed view in `view_cases_for`.
- [ ] `test/cpu/test_cpu_conformance.cpp` continues to pass `run_storage_and_transfer_conformance`, `run_async_copy_conformance`, and `run_storage_oracle_conformance` end-to-end; the existing CPU perturbation test is generalized to accept a permutation hook and remains passing on CPU.

## Verification

- Build CPU-only:
  - `cmake -S . -B build/cpu -DBUILD_TESTING=ON`
  - `cmake --build build/cpu --target iom_backend_conformance_cpu_tests iom_cpu_tests`
  - `ctest --test-dir build/cpu --output-on-failure -R '^iom_(cpu|backend_conformance_cpu)_tests$'`
  - Expect every CPU case, including the generalized negative fixture, to pass. Confirm the test-side standard-layout encoder round-trips against `snapshot_storage` for `{17,33}` U8 and `{2,3,4,17,33}` BF16.
- Run the negative fixture locally on CPU first:
  - `ctest --test-dir build/cpu --output-on-failure -R 'conformance harness detects perturbed candidate bytes'`
  - Expect the existing CPU round-trip case to detect the perturbed byte at element-zero's slot.
- Use `.agents/skills/remote-development` for accelerator verification. Configure `.remote-hosts.conf` with one CUDA host alias and one ROCm host alias; pick a host with Tenstorrent runtime access for TTNN.
  - Sync once per backend:
    - `.agents/skills/remote-development/scripts/remote-sync cuda task-nt001-cuda`
    - `.agents/skills/remote-development/scripts/remote-sync rocm task-nt001-rocm`
    - `.agents/skills/remote-development/scripts/remote-sync ttnn task-nt001-ttnn`
  - Build on each remote with the focused backend target only:
    - `remote-exec cuda task-nt001-cuda 'cmake -S . -B build -DCUDA_ENABLED=ON -DBUILD_TESTING=ON && cmake --build build -j --target iom_cuda_conformance_tests'`
    - `remote-exec rocm task-nt001-rocm 'cmake -S . -B build -DROCM_ENABLED=ON -DBUILD_TESTING=ON && cmake --build build -j --target iom_rocm_conformance_tests'`
    - `remote-exec ttnn task-nt001-ttnn 'cmake -S . -B build -DTTNN_ENABLED=ON -DBUILD_TESTING=ON && cmake --build build -j --target iom_ttnn_conformance_tests'`
  - Run the negative fixture on each backend:
    - `remote-exec cuda task-nt001-cuda 'ctest --test-dir build --output-on-failure -R "CUDA conformance: storage oracle identifies perturbed transfer map"'`
    - `remote-exec rocm task-nt001-rocm 'ctest --test-dir build --output-on-failure -R "ROCm conformance: storage oracle identifies perturbed transfer map"'`
    - `remote-exec ttnn task-nt001-ttnn 'ctest --test-dir build --output-on-failure -R "TTNN conformance: storage oracle identifies perturbed transfer map"'`
    - Expect each to fail and report the first divergent slot (CUDA/ROCm) or plane index (TTNN).
  - Run the full corrected sweep on each backend:
    - `remote-exec cuda task-nt001-cuda 'ctest --test-dir build --output-on-failure -R "CUDA conformance: storage oracle covers every leaf width and padded shape"'`
    - `remote-exec rocm task-nt001-rocm 'ctest --test-dir build --output-on-failure -R "ROCm conformance: storage oracle covers every leaf width and padded shape"'`
    - `remote-exec ttnn task-nt001-ttnn 'ctest --test-dir build --output-on-failure -R "TTNN conformance: storage oracle covers every leaf width and padded shape"'`
    - Expect every shape and view case to pass with no `TrafficGate` allocations inside transfers.
  - Cleanup: `remote-clean cuda task-nt001-cuda`, `remote-clean rocm task-nt001-rocm`, `remote-clean ttnn task-nt001-ttnn`.

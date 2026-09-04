# Run the negative storage-oracle fixture on CPU

**Order:** 39
**Priority:** P2 — Required finishing work: discharges the unmet negative-fixture acceptance item of prerequisite `22-NT-001-add-accelerator-storage-oracles` on the only broadly runnable backend (CPU-only CI); no production change required. Also depends on `31-AR-003-expose-device-capabilities` for the `iom::Device::supported_data_types()` predicate that supplies the leaf-type span after the per-driver `kCpuLeafTypes` constant is deleted.
**Blocked by:** 22-NT-001-add-accelerator-storage-oracles, 31-AR-003-expose-device-capabilities
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-002`
**Review severity:** low
**Review verification:** verified, confidence 90

## Outcome

`iom_backend_conformance_cpu_tests` executes a CPU-only `TEST_CASE` that wraps `iom_conformance::CpuStorageOracle` in `iom_conformance::PermutingStorageOracle(swap_first_adjacent_slots)`, drives `iom_conformance::run_storage_oracle_conformance` with a single-leaf span derived from `devices.candidate->supported_data_types()` (the `iom::Device` capability predicate introduced by `31-AR-003-expose-device-capabilities`), `require_match=false, verify_seed=false`, and `CHECK_FALSE`s the harness result. The same case asserts the positive polarity by also running the harness with an identity permutation and `REQUIRE`ing it returns `true`. The two assertions are local; no GPU is required.

## Current failure

The detection machinery (`iom_conformance::PermutingStorageOracle`, `iom_conformance::swap_first_adjacent_slots`, the `require_match=false` polarity on `iom_conformance::run_storage_oracle_conformance`) is defined once in `test/backend/backend_conformance_oracle.hpp` and reached by every accelerator driver's negative fixture — `test/cuda/test_cuda_conformance.cpp:278-284`, `test/rocm/test_rocm_conformance.cpp:248-253`, `test/ttnn/test_ttnn_conformance.cpp:434-439`, `test/sycl/test_sycl_conformance.cpp:287-288`. The CPU driver (`test/cpu/test_cpu_conformance.cpp:121-127`) only contains the positive fixture; `PermutingStorageOracle`/`swap_first_adjacent_slots` are absent from `test/cpu/`. The CPU-only CI run is the only configuration broad contributors execute, so a regression that neuters detection (e.g. an identity slot map, `require_match` polarity flipped, the `PermutingStorageOracle` permute loop removed) is invisible until the next hardware run. The finding additionally notes that `22-NT-001-add-accelerator-storage-oracles`'s acceptance criterion requires the CPU perturbation fixture be generalized around the permutation hook — that item is unmet and this task closes it.

## Scope

- This task is the CPU-only completion of prerequisite `22-NT-001-add-accelerator-storage-oracles`, which delivered the `iom_conformance::PermutingStorageOracle`, `iom_conformance::swap_first_adjacent_slots`, and `iom_conformance::run_storage_oracle_conformance(devices, types, oracle, observer, require_match, verify_seed)` machinery plus accelerator negative fixtures; this task reuses those symbols unchanged and adds the missing CPU mirror.
- Add exactly one `TEST_CASE` to `test/cpu/test_cpu_conformance.cpp` mirroring the accelerator negative fixtures that `22-NT-001-add-accelerator-storage-oracles` introduced, using the existing `CpuDevices` harness and the existing transitive includes.
- Drive `iom_conformance::run_storage_oracle_conformance` (delivered by `22-NT-001-add-accelerator-storage-oracles` in `test/backend/backend_conformance_copy_storage.hpp:313-390`) exactly as the CUDA/ROCm/TTNN/SYCL drivers do: pass a one-entry span over `devices.candidate->supported_data_types()` (the public capability predicate added by `31-AR-003-expose-device-capabilities` to `iom::Device`), `&devices.gate`, `require_match=false`, `verify_seed=false`. The one-entry span is `devices.candidate->supported_data_types().subspan(0, 1)`, taken from the same public surface every other CPU `TEST_CASE` in this driver uses after `31-AR-003-expose-device-capabilities` deletes `kCpuLeafTypes`.
- No production code change; no new headers; no new files beyond this spec.

## Implementation references

- **Modify:** `test/cpu/test_cpu_conformance.cpp` — add one `TEST_CASE("CPU conformance: storage oracle identifies perturbed transfer map")` immediately after the existing positive oracle case at line 121-127. Reuse the existing `CpuDevices` fixture defined at `test/cpu/test_cpu_conformance.cpp:94-106`. Do not reference the deleted `kCpuLeafTypes` constant (`test/cpu/test_cpu_conformance.cpp:21-34`, removed by `31-AR-003-expose-device-capabilities`); the new case obtains its one-entry span from `devices.candidate->supported_data_types()` exactly like every neighbouring `TEST_CASE` after task 31 lands.
- **Read:** `test/backend/backend_conformance_oracle.hpp:215-274` — definitions of `iom_conformance::PermutingStorageOracle`, its `SlotMap` alias, `set_owner_spec`/`seed`/`observe` overrides, and `iom_conformance::swap_first_adjacent_slots`. Already in scope via the existing include of `backend/backend_conformance_copy_storage.hpp` in `test/cpu/test_cpu_conformance.cpp:12`.
- **Read:** `test/backend/backend_conformance_copy_storage.hpp:313-390` — `iom_conformance::run_storage_oracle_conformance` signature: `(const ConformanceDevices&, std::span<const iom::DataType>, AcceleratorStorageOracle&, ConformanceObserver* = nullptr, bool require_match = true, bool verify_seed = true)`. The negative case passes `(devices.conformance(), one_type, perturbed, &devices.gate, false, false)` exactly as the CUDA/ROCm/TTNN/SYCL drivers do.
- **Read:** `test/ttnn/test_ttnn_conformance.cpp:425-440` and `test/rocm/test_rocm_conformance.cpp:232-254` — confirm the pattern is backend-agnostic (no backend-specific seeding), proving the same fixture body is valid for CPU.
- **Tests:** `test/cpu/test_cpu_conformance.cpp` — existing `CpuDevices` (allocators, gate, devices) and the positive `CPU conformance: standard storage oracle covers every leaf width and padded shape` case to be paired with. The new case must end with the same `CHECK_FALSE(devices.gate.armed());` discipline as every neighbouring case.

## Requirements

1. The new `TEST_CASE` constructs `iom_conformance::CpuStorageOracle direct;` followed by `iom_conformance::PermutingStorageOracle perturbed(direct, iom_conformance::swap_first_adjacent_slots);` — same construction as `test/cuda/test_cuda_conformance.cpp:278-280`.
2. The case obtains the supported leaf set from `devices.candidate->supported_data_types()` (the pure-virtual method `[[nodiscard]] virtual std::span<const iom::DataType> supported_data_types() const noexcept = 0;` declared in `include/iom/device.hpp` between `backend_kind()` and `backend_device()` by `31-AR-003-expose-device-capabilities`, returning the 23-entry CPU span BOOL..F64 in the documented order). It binds that span to a local named `supported`, requires `supported.size() >= 1`, and forms `one_type = supported.subspan(0, 1)`. The case invokes `iom_conformance::run_storage_oracle_conformance(devices.conformance(), one_type, perturbed, &devices.gate, false, false)`. No reference to the deleted `kCpuLeafTypes` constant (`test/cpu/test_cpu_conformance.cpp:21-34`, removed by `31-AR-003-expose-device-capabilities`); no per-driver `constexpr std::initializer_list`.
3. The case wraps that call in `CHECK_FALSE(...)`, mirroring `test/cuda/test_cuda_conformance.cpp:281-283`. On success the harness returns `false` because the permuted storage diverges from the unpermuted expectation under `require_match=false`, satisfying the negative-polarity contract; on a regression that makes detection inert the call would return `true` and the assertion fires.
4. The case additionally exercises the positive polarity with an identity slot map to prove the harness returns `true` when the oracle sees what it expects. Concretely: construct `iom_conformance::PermutingStorageOracle identity(direct, [](std::size_t l) { return l; });` and `REQUIRE(iom_conformance::run_storage_oracle_conformance(devices.conformance(), one_type, identity, &devices.gate))` where `one_type` is the same `supported.subspan(0, 1)` from requirement 2. This is the polarity counterpart required by the invariant "must itself be proven to fail on a wrong map".
5. The case ends with `CHECK_FALSE(devices.gate.armed());` to match the surrounding convention.
6. No edits to `test/backend/*.hpp`, to `CpuStorageOracle`, to `PermutingStorageOracle`, to `swap_first_adjacent_slots`, to `run_storage_oracle_conformance`, or to any `iom::Device` override. No new headers, no new files beyond this directory. The `supported_data_types()` query is the only API added by a prerequisite (task 31); this task consumes it read-only.

## Non-goals

- Adding equivalent negative fixtures to other (non-CPU) drivers — they already exist at the cited lines.
- Modifying the `PermutingStorageOracle` permute loop or its slot map — the detection machinery is the authority under test.
- Generalizing `run_storage_oracle_conformance` to accept a `PermutingStorageOracle` natively — out of scope; the existing `AcceleratorStorageOracle&` parameter already accepts the wrapper.
- Iterating the supported leaf set more broadly than the established one-entry subspan — matching the accelerator drivers keeps the case cheap and symmetric; full-coverage identity sanity is already provided by the existing positive `CPU conformance: standard storage oracle covers every leaf width and padded shape`, which after `31-AR-003-expose-device-capabilities` iterates the full `supported_data_types()` span on the candidate device.
- Touching production code, `src/cpu/**`, `include/iom/**`, or any backend outside `test/cpu/`.

## Acceptance criteria

- [ ] A new `TEST_CASE` named `CPU conformance: storage oracle identifies perturbed transfer map` exists in `test/cpu/test_cpu_conformance.cpp` between the existing positive storage-oracle case and `CPU conformance: asynchronous copies against the CPU reference` (line 129).
- [ ] The case wraps `CpuStorageOracle` in `PermutingStorageOracle(swap_first_adjacent_slots)` and `CHECK_FALSE`s `run_storage_oracle_conformance(..., false, false)` — identical pattern to `test/cuda/test_cuda_conformance.cpp:278-284` and `test/rocm/test_rocm_conformance.cpp:248-253`.
- [ ] The case obtains its one-entry leaf span via `devices.candidate->supported_data_types().subspan(0, 1)`. `grep -n "kCpuLeafTypes" test/cpu/test_cpu_conformance.cpp` returns zero matches (the constant is deleted by `31-AR-003-expose-device-capabilities`). No per-driver `constexpr std::initializer_list<iom::DataType>` is reintroduced by this task.
- [ ] The case additionally `REQUIRE`s `run_storage_oracle_conformance` returns `true` for an identity-permuted oracle (using the same `supported.subspan(0, 1)`), proving positive polarity.

## Verification
This task has two prerequisites. `22-NT-001-add-accelerator-storage-oracles` must have landed `test/backend/backend_conformance_oracle.hpp` (defining `iom_conformance::PermutingStorageOracle` and `iom_conformance::swap_first_adjacent_slots`) and the `iom_conformance::run_storage_oracle_conformance` signature in `test/backend/backend_conformance_copy_storage.hpp:313-390`. Separately, `31-AR-003-expose-device-capabilities` must have added the pure-virtual `[[nodiscard]] virtual std::span<const iom::DataType> supported_data_types() const noexcept = 0;` to `include/iom/device.hpp` and its `CpuDevice::supported_data_types()` override in `src/cpu/device.cpp` — without that method the case cannot obtain the one-entry leaf span. If either prerequisite has not yet landed, the new case will fail to compile and that prerequisite must be merged first.

Run from the existing CPU build directory (already configured locally per the review's validation record):

```sh
cmake --build build --target iom_backend_conformance_cpu_tests
ctest --test-dir build -R iom_backend_conformance_cpu_tests --output-on-failure
```

Mutation sanity (manual, not part of the run set): temporarily replace `swap_first_adjacent_slots` with the identity lambda ` [](std::size_t l) { return l; }` and re-run only the new case — `CHECK_FALSE` must fire, proving the negative case is not a no-op.

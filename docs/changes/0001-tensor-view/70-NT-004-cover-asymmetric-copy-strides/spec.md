# Cover asymmetric source/destination strides and minimal-extent and row-tail owner shapes in the shared copy matrix

**Order:** 70
**Priority:** P1 — closes the accelerator-visible blind spot in independent source-plane versus destination-plane stride arithmetic and the minimal-extent/row-tail owner geometries; bounded shared-matrix remediation that does not broadly gate other work
**Blocked by:** 52-CC-002-cover-aligned-multitile-storage — task 52 appends the aligned multi-tile shapes to the same two initializer lists this task extends and establishes the post-CC-001 matrix whose settled shape set this task's guard reasoning and negative demonstration assume
**Source:** `docs/changes/0001-tensor-view/review.md` — `NT-004`
**Review severity:** medium
**Review verification:** verified, confidence 85

## Outcome

`copy_cases_for` emits two asymmetric-stride copy cases — "contiguous source to stepped destination" and "permuted source to contiguous destination" — whose source and destination views carry genuinely different plane-stride vectors while remaining valid, equal logical specs, and `transfer_owner_shapes()`/`copy_owner_shapes()` contain `{17, 16}` (row-tail-only padding) and `{1, 1}` (minimal extent). A queued-copy implementation that walks the destination with the source's stride vector now fails `run_async_copy_conformance` on every backend instead of silently passing the entire shared suite, and the smallest and row-tail owner geometries are covered by the transfer, copy, and storage-oracle sweeps. The CPU asynchronous-copy case additionally passes a `CpuStorageOracle`, so the asymmetric cases are physical-scope-checked (padding, out-of-window planes, untouched source allocation) on the always-run CPU suite; every accelerator driver's existing oracle argument covers them on hardware.

## Current failure

Invariant (review NT-004): conformance must exercise each backend's independent source-plane and destination-plane arithmetic. Every backend owns two stride vectors on a queued copy: CUDA/ROCm carry separate source/destination stride arrays in the copy metadata (`src/cuda/copy.cu:258-263` kernel decode, `:588-597` `write_cuda_metadata`; ROCm twin in `src/rocm/copy.hip`), SYCL walks each view independently (`view_planes`/`plane_pairs` at `src/sycl/copy.cpp:34-71` today; after 60-AR-001 the `SyclCopyMetadataHeader` stride arrays mirror CUDA), TTNN maps each view through `owner_plane_at` (`src/ttnn/copy.cpp:37-49`, consumed per view by `copy_planes` `:231-240`), and CPU walks both views in `for_each_tile_lockstep` (`src/cpu/device.cpp:264-382`).

Every `CopyCase` that `copy_cases_for` (`test/backend/backend_conformance_copy_storage.hpp:193-253`) emits builds source and destination with mirrored transforms — the same slice pair at different offsets (`:207-217`), the same stepped slice pair (`:218-228`), the same permutation (`:229-240`), the same select/slice pair (`:241-250`) — so the two views always carry identical plane-stride vectors and only the plane offset differs. A backend that substitutes `source.plane_strides()` for the destination walk is behaviorally identical on every shared case, on every backend, and ships silent plane misplacement the first time a user copies between differently-strided views. Secondary gaps in the same matrices: no `{1, 1}` minimal-extent owner (one padded tile, one logical element — the smallest queued copy, staging, and identical-window no-op instance) and no row-tail-only padded shape (`{17, 16}` — two tile rows whose second holds a single logical row, exactly one tile column). Differing-stride copies are exercised only CPU-locally (`test/cpu/test_cpu.cpp:981-995`, stepped source to dense destination inside "CPU copies move logical planes without touching padding"), which cannot observe accelerator-only stride bugs.

## Scope

- Add exactly two shared copy cases to `copy_cases_for` (`test/backend/backend_conformance_copy_storage.hpp:193-253`), appended after the existing cases before `return cases;`:
  - "contiguous source to stepped destination" — on the first leading axis whose dimension is `>= 4`: source `full.slice(axis, 0, count, 1)` (stride-preserving prefix, dense strides), destination `full.slice(axis, 1, count, 2)` (offset advanced one plane, stride doubled), with `count = dims[axis] / 2 >= 2`. Skip the case when no leading axis reaches 4.
  - "permuted source to contiguous destination" — on the first adjacent pair of leading axes with equal dimensions `>= 2`: source `full.permute(order)` with `order` the identity permutation except that pair swapped, destination the full view unchanged. Skip the case when no equal adjacent leading-axis pair exists.
  - Both cases satisfy the harness's `REQUIRE(candidate_source_view.spec() == candidate_destination_view.spec())` (`:539-540`) without modification: the stepped pair has identical shape with strides differing by a factor of two on one axis; the permuted pair swaps equal dimensions so the shape sequence is unchanged while the matching strides transpose.
- Append `{17, 16}` and `{1, 1}` to both `transfer_owner_shapes()` (`:168-178`) and `copy_owner_shapes()` (`:255-263`) with trailing geometry comments in the existing style. `storage_oracle_owner_shapes()` (`:297-306`) inherits both via its existing deduplication with no code change.
- Wire the existing `iom_conformance::CpuStorageOracle` into the CPU asynchronous-copy case ("CPU conformance: asynchronous copies against the CPU reference", `test/cpu/test_cpu_conformance.cpp:238-243`) as the fourth `run_async_copy_conformance` argument, mirroring how the CUDA, ROCm, and TTNN drivers already pass their oracles. No oracle machinery changes.
- Affected surfaces: the shared copy matrix and owner-shape tables consumed by `run_storage_and_transfer_conformance`, `run_async_copy_conformance`, and `run_storage_oracle_conformance` on all five backends; the CPU async-copy test case. Affected backends: all five (the matrix is shared); the new cases must pass on the production walks of every backend.

## Implementation references

- **Modify:** `test/backend/backend_conformance_copy_storage.hpp` — `copy_cases_for` (`:193-253`): append the two asymmetric cases; the builder bodies follow the existing captured-lambda style of the neighboring cases. A working form (the file already includes `<optional>` at `:20`, `<numeric>` at `:19` for `std::iota`, and `<utility>` at `:25` for `std::swap`; no new includes):
  ```cpp
  std::optional<std::size_t> stepped_axis;
  for (std::size_t axis = 0; axis < leading; ++axis) {
      if (dims[axis] >= 4) {
          stepped_axis = axis;
          break;
      }
  }
  if (stepped_axis.has_value()) {
      const std::size_t axis = *stepped_axis;
      const std::size_t count = dims[axis] / 2;
      cases.push_back(
              {"contiguous source to stepped destination",
               [axis, count](const iom::TensorView& full) {
                   return full.slice(axis, 0, count, 1);
               },
               [axis, count](const iom::TensorView& full) {
                   return full.slice(axis, 1, count, 2);
               }});
  }

  std::optional<std::size_t> swapped_axis;
  for (std::size_t axis = 0; axis + 1 < leading; ++axis) {
      if (dims[axis] == dims[axis + 1] && dims[axis] >= 2) {
          swapped_axis = axis;
          break;
      }
  }
  if (swapped_axis.has_value()) {
      const std::size_t axis = *swapped_axis;
      std::vector<std::size_t> order(leading);
      std::iota(order.begin(), order.end(), std::size_t{0});
      std::swap(order[axis], order[axis + 1]);
      cases.push_back(
              {"permuted source to contiguous destination",
               [order](const iom::TensorView& full) {
                   return full.permute(std::span<const std::size_t>{order});
               },
               [](const iom::TensorView& full) { return full; }});
  }
  ```
- **Modify:** `test/backend/backend_conformance_copy_storage.hpp` — `transfer_owner_shapes()` (`:168-178`) and `copy_owner_shapes()` (`:255-263`): append `{17, 16},  // row-tail only` and `{1, 1},  // minimal extent` to each list, after task 52's `{16, 32}`/`{2, 3, 16, 48}` entries. Append-only; keep every per-shape trailing comment.
- **Modify:** `test/cpu/test_cpu_conformance.cpp` — "CPU conformance: asynchronous copies against the CPU reference" (`:238-243`): construct `iom_conformance::CpuStorageOracle oracle;` and pass `&oracle` as the fourth argument of `run_async_copy_conformance`, exactly as the CUDA driver does at `test/cuda/test_cuda_conformance.cpp:319-322` (ROCm: `test/rocm/test_rocm_conformance.cpp:428-431`; TTNN: `test/ttnn/test_ttnn_conformance.cpp:475-477`).
- **Read:** `test/backend/backend_conformance_oracle.hpp` — `standard_layout_view_slot` (`:71-103`) maps each logical element through the view's own `plane_offset()`/`plane_strides()` into `iom::detail::standard_layout_slot`, and `apply_standard_tiled_view` (`:167-182`) is what builds the expected destination encoding and applies the source seed. Both are view-driven: the oracle-side expected-destination and source-unchanged checks in `run_async_copy_conformance` (`:549-620`) already handle asymmetric strides with zero changes.
- **Read:** `src/cpu/device.cpp` — `for_each_tile_lockstep` (`:264-382`, `source_strides`/`destination_strides` at `:285-288`, the per-axis plane recursion at `:298-316`) and `copy_elements` (`:891-910`). This is the reference/candidate queued walk the shared cases drive and the CPU mutation site for the negative demonstration below.
- **Read:** `src/cuda/copy.cu` — `write_cuda_metadata` (`:562-598`, destination stride array at `:592-594`) and `grid_stride_copy_kernel` (`:242-288`, `destination_plane += coordinate * destination_strides[axis]` at `:281`); the ROCm twin in `src/rocm/copy.hip`; the TTNN walk `copy_planes`/`owner_plane_at` (`src/ttnn/copy.cpp:231-240`, `:37-49`); the SYCL convergence target in `docs/changes/0001-tensor-view/60-AR-001-converge-sycl-copy-architecture/spec.md` (post-60 SYCL mirrors the CUDA metadata shape). These are the accelerated mutation sites for the hardware demonstration.
- **Read:** `docs/changes/0001-tensor-view/52-CC-002-cover-aligned-multitile-storage/spec.md` — the established append-only shape-table conventions (dual placement in both lists, trailing comments, dedup semantics) this task reuses.
- **Tests:** `iom_backend_conformance_cpu_tests` (case "CPU conformance: asynchronous copies against the CPU reference"), `iom_cpu_tests`, `iom_tests`, and each accelerator's `iom_<backend>_conformance_tests` (cases "<backend> conformance: asynchronous copies against the CPU reference" and "<backend> conformance: storage oracle covers every leaf width and padded shape"). No new test cases are created; the shared sweeps grow.

## Requirements

- The two new cases must produce genuinely different plane-stride vectors between source and destination with equal logical specs: the stepped pair differs by a factor of two on exactly one leading axis, and the permuted pair transposes the strides of one equal-dimension leading-axis pair. Neither case may degenerate to a single plane (`count >= 2` for the stepped pair; swapped dimensions `>= 2`), so the differing stride is always consumed by the walk.
- Both new cases must submit real copies on every backend, never identical-window no-ops: per spec §8 an identical window requires equal plane strides, which these pairs violate. A backend whose no-op predicate ignored strides fails these cases via the destination readback.
- The stepped destination slice must stay in bounds without harness changes: `first + (count - 1) * step = 2 * count - 1 <= dims[axis] - 1` holds for `count = dims[axis] / 2` on an axis of at least 4; the source prefix `slice(axis, 0, count, 1)` is always in bounds.
- On the post-task-52 matrix the guards must actually fire: "contiguous source to stepped destination" executes on `{2, 3, 4, 17, 33}` (axis 2, `count = 2`) and "permuted source to contiguous destination" executes on `{2, 2, 2, 3, 17, 33}` (axes 0 and 1 swapped). Do not reuse the existing mirrored stepped case's `dims[0] >= 4` guard (`:218-228`), which never fires on any copy-owner shape.
- `{17, 16}` must appear exactly once in each of `transfer_owner_shapes()` and `copy_owner_shapes()` with its trailing comment; `{1, 1}` likewise. No existing entry is moved, renamed, or removed; both lists remain strict supersets of their post-task-52 contents; `storage_oracle_owner_shapes()` returns each new shape exactly once with no code change.
- The new shapes must flow through the existing builders without per-shape branches: `view_cases_for({17, 16})` and `view_cases_for({1, 1})` return the rank-two case set ("full" plus the empty permutation), and `copy_cases_for` returns the rank-two set ("full to full" plus the identical-window no-op) plus, where guards allow, the new asymmetric cases (neither fires on rank two).
- The CPU asynchronous-copy case must pass a `CpuStorageOracle` so that every copy case — including the asymmetric pair on `{2, 3, 4, 17, 33}` and `{2, 2, 2, 3, 17, 33}` — is physical-scope-checked on CPU: the complete destination allocation equals its seed with only the addressed slots replaced, and the complete source allocation is unchanged (`require_storage_oracle_bytes` at `:596-620`). All harness-side seed/expect logic is reused unmodified.
- Every sweep that consumes `copy_owner_shapes()` and `transfer_owner_shapes()` — storage-and-transfer, asynchronous-copy, and storage-oracle on all five backends — must cover the new shapes for every `DataType` in the driver's supported-type table, with zero `TrafficGate` allocator calls inside transfers, transforms, and operations (the 06-shared-backend-conformance rule).
- The negative property must hold: with a mirrored-stride walk (destination planes derived from the source view's stride vector), the pre-extension shared suite passes on every backend and the extended suite fails at both new case labels — via `require_logical_bytes` on the destination readback everywhere, and additionally via `require_storage_oracle_bytes` on CPU and on every driver that passes an oracle.

## Non-goals

- Any production change: the copy/transfer walks of all five backends (`src/cpu/device.cpp`, `src/cuda/copy.cu`, `src/rocm/copy.hip`, `src/sycl/copy.cpp`, `src/ttnn/copy.cpp`, `src/shared/standard_tiled_copy.inl`) are untouched. Backend source edits exist only as temporary, reverted fault injections during verification.
- Task 52's territory: the aligned multi-tile-column shapes `{16, 32}` and `{2, 3, 16, 48}` and the oracle-side enforcement of the tile-aligned geometry.
- Task 68's territory (`68-NT-002-observe-sycl-queued-storage`): wiring `SyclStorageOracle` into SYCL's async-copy call. Once landed, SYCL's oracle argument covers the new cases automatically; until then the logical readback covers them.
- Task 69's territory (`69-NT-003`): lifetime/destruction regressions; no registry, destructor, or quarantine work.
- NT-001's territory: benchmark gates or throughput rebaselining.
- New `ViewCase` builders, transfer-side asymmetric view cases beyond the shape additions, new `DataType` coverage, new negative fixtures, new oracle virtuals, error-category changes, or any `test/CMakeLists.txt` change.
- Asymmetric variants beyond the two named cases (offset-only pairs, reshape-based pairs, deeper permutation enumerations): the review names exactly these two, and the API's no-last-two-dimension-transform rule makes plane strides the only copy-matrix degree of freedom beyond offset.

## Acceptance criteria

- [ ] `copy_cases_for` returns the two new cases appended after the existing six, with the specified builders and guards; no existing case is modified, and the file's include set is unchanged.
- [ ] "contiguous source to stepped destination" executes on `{2, 3, 4, 17, 33}` and "permuted source to contiguous destination" executes on `{2, 2, 2, 3, 17, 33}` in the post-task-52 matrix (observable via `CAPTURE(copy_case.label)` output or a debugger stop), and neither is reported as an identical-window no-op on any backend.
- [ ] `transfer_owner_shapes()` and `copy_owner_shapes()` each return `{17, 16}` and `{1, 1}` exactly once with trailing comments; both lists are strict supersets of their post-task-52 contents; `storage_oracle_owner_shapes()` contains each new shape exactly once with no code change to the union.
- [ ] "CPU conformance: asynchronous copies against the CPU reference" passes a `CpuStorageOracle`, and `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests` all pass on the reference host.
- [ ] With the CPU mirrored-stride injection (`destination.plane_strides()` replaced by `source.plane_strides()` at `src/cpu/device.cpp:287-288`) applied to the pre-extension tree, the async-copy sweep passes — demonstrating the blindness; with this task's extension and the injection still applied, the sweep fails at "contiguous source to stepped destination" and "permuted source to contiguous destination"; after reverting the injection everything is green.
- [ ] On at least one accelerator with available hardware (CUDA via `write_cuda_metadata`'s destination stride array, or the analogous ROCm/TTNN site), the same two-sided demonstration holds: the extended suite fails at the new case labels under the injection and passes after the revert.
- [ ] Every enabled backend's conformance target passes the extended async-copy and storage-oracle sweeps over the new shapes and cases for every supported `DataType` on hardware, with no `TrafficGate` violations.
- [ ] After verification, `git diff` shows changes only in `test/backend/backend_conformance_copy_storage.hpp` and `test/cpu/test_cpu_conformance.cpp`; no production file retains a fault injection.

## Verification

Local CPU evidence (reference host, no hardware required):

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON` (if not already configured) then `cmake --build build/cpu --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j`
- `ctest --test-dir build/cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — all pass, including the extended cases over `{17, 16}`, `{1, 1}`, `{2, 3, 4, 17, 33}`, and `{2, 2, 2, 3, 17, 33}` for every leaf type, with the CPU oracle active on the async-copy sweep.

Negative demonstration (CPU, deterministic — run before the accelerator rounds):

1. On the pre-extension tree, inject the mirrored-stride bug: in `for_each_tile_lockstep` (`src/cpu/device.cpp:287-288`) replace `destination.plane_strides()` with `source.plane_strides()`; rebuild and run `ctest --test-dir build/cpu --output-on-failure -R iom_backend_conformance_cpu_tests` — every case passes, proving the current matrix cannot see a mirrored-stride walk.
2. Apply this task's extension with the injection still in place; rebuild; rerun — expect `require_logical_bytes` failures labeled "contiguous source to stepped destination" and "permuted source to contiguous destination" (reference arm first), plus `require_storage_oracle_bytes` failures naming the first wrong destination byte now that the CPU oracle is wired.
3. Revert the injection; rebuild; the full local set from step one is green again.

Accelerator evidence (requires hardware; use `.agents/skills/remote-development` and `.remote-hosts.conf`; CPU-only evidence does not satisfy the finding's accelerator-visibility point, and enabled backends must not skip — fail-not-skip per parent spec §11.6):

- Sync once per backend with unique task ids: `.agents/skills/remote-development/scripts/remote-sync cuda task-nt004-cuda` (and `rocm task-nt004-rocm`, `sycl task-nt004-sycl`, `ttnn task-nt004-ttnn` as available).
- Configure and build each backend's conformance target only, e.g. `remote-exec cuda task-nt004-cuda 'cmake -S . -B build -DCUDA_ENABLED=ON -DBUILD_TESTING=ON && cmake --build build -j --target iom_cuda_conformance_tests'` (ROCm: `-DROCM_ENABLED=ON`; SYCL: `-DSYCL_ENABLED=ON` with the DPC++ environment from the skill; TTNN: `-DTTNN_ENABLED=ON`).
- Run each backend's conformance binary: `remote-exec cuda task-nt004-cuda 'ctest --test-dir build --output-on-failure -R "^iom_cuda_conformance_tests$"'` (mirrored per backend). Expect the extended async-copy sweep — including both asymmetric cases and the new shapes for every supported type — and the storage-oracle sweep to pass.
- Accelerated mirrored-stride demonstration (once, on CUDA): edit `write_cuda_metadata` (`src/cuda/copy.cu:592-594`) to fill the destination stride array from `source.plane_strides()[axis]`; rebuild; rerun — the async-copy case must fail at the two new case labels; revert and confirm green. The analogous single-edit demonstrations are the ROCm metadata twin and `owner_plane_at(destination, index)` → `owner_plane_at(source, index)` in `src/ttnn/copy.cpp:236-238`; post-60 SYCL's mirrored metadata writer is the SYCL site.
- Cleanup: `remote-clean cuda task-nt004-cuda` (and per-backend equivalents).

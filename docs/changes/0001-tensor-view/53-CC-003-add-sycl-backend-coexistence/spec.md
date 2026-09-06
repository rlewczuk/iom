# Add SYCL to the backend coexistence build, link, and interleaved-copy matrix

**Order:** 53
**Priority:** P1 — required combined-build behavior for the SYCL normal path: root spec goal 8, §6, §11.4, and §11.7 make "any combination of enabled backends coexists in one build and process" a completion criterion, and SYCL is the one optional backend the coexistence matrix excludes; the defect gates no other remediation task
**Blocked by:** None
**Source:** `docs/changes/0001-tensor-view/review.md` — `CC-003`
**Review severity:** medium
**Review verification:** verified, confidence 95

## Outcome

`iom_backend_coexistence_tests` exists and links `iom_sycl` whenever `SYCL_ENABLED=ON`, including the SYCL-only (CPU+SYCL) configuration where the target previously did not exist at all. The coexistence translation unit gains an `IOM_COEXIST_SYCL` participant that constructs a real SYCL device (ordinal 0) with a USM-backed allocator, interleave-submits same-device asynchronous copies across CPU and SYCL queues in one process, waits on each originating queue, and matches every destination bit-for-bit against the CPU reference. SYCL also joins the two focused coexistence cases the other accelerators already have: rejection of views from another device instance, and second-device ordinal reporting where hardware provides more than one eligible device. A short errata recorded in this spec resolves sub-spec 15's internally inconsistent requirements text so the documentation and the tree agree.

## Current failure

The invariant under review is root spec goal 8 ("any combination of enabled backends can coexist in one build and process", `docs/changes/0001-tensor-view/spec.md:18`), §6 (:439, "any number and combination of backend devices may coexist in one process"), §11.4, and §11.7 (:669, "a combined job enables and links all four optional backend libraries into one executable"), together with sub-spec 15's outcome line, which already names SYCL.

The tree violates it in both possible configurations (`test/CMakeLists.txt:302`, `:322-361`, `test/backend/test_backend_coexistence.cpp`):

1. With only `SYCL_ENABLED=ON`, the coexistence gate `if(CUDA_ENABLED OR ROCM_ENABLED OR TTNN_ENABLED)` is false, so `iom_backend_coexistence_tests` does not exist; a SYCL-enabled build has no coexistence coverage whatsoever.
2. With SYCL plus any other accelerator, the target exists but SYCL is not a participant: `IOM_COEXIST_SYCL` appears nowhere in the tree, `iom_sycl` is never linked into the target, and the SYCL branch is absent from all four coexistence `TEST_CASE`s. SYCL link-time symbol collisions, owned-context coexistence with another backend's context, and process-wide queue-id pool behavior with SYCL queues alive are entirely invisible to CI.

The root cause of the stale spec text is historical: sub-spec 15's requirements line ("The combined executable links `libiom`, `iom_cuda`, `iom_rocm` and `iom_ttnn` directly") and blocked-by list (`08-rocm-storage-copy`, `10-cuda-storage-copy`, `14-ttnn-storage-copy`) predate SYCL's implementation (`11-sycl-buildable-scaffold`, `12-sycl-storage-copy`); the build followed the stale lines.

## Scope

- Extend the coexistence target gate in `test/CMakeLists.txt` to `SYCL_ENABLED`, and add an `IOM_COEXIST_SYCL=1` compile-definition branch that also links `iom_sycl` and applies SYCL compile/link options and include directories to the single coexistence translation unit.
- Add the SYCL participant to `test/backend/test_backend_coexistence.cpp`: a USM shared-memory `Allocator` bound to the factory-owned context via the existing `iom::sycl_detail::context_calls.context_ready` test seam, a `make_sycl_participant` factory parallel to the CUDA/ROCm/TTNN ones, and participation in the main interleaved-copy case.
- Add the SYCL branch to the "queues reject views from another device" case (device plus independent foreign device, both directions rejected) and to the "second devices report their own ordinal" case (guarded by the eligible-device runtime count).
- Record the sub-spec 15 errata inside this spec (see below); `15-backend-coexistence/spec.md` itself is not edited.
- No production source changes: `src/sycl/**`, `include/iom/**`, and the root `CMakeLists.txt` are untouched. Affected backends: sycl (participant) plus the existing cpu/cuda/rocm/ttnn participants only through the shared gate expression.

## Implementation references

- **Modify:** `test/CMakeLists.txt:302` — change the coexistence gate to `if(CUDA_ENABLED OR ROCM_ENABLED OR TTNN_ENABLED OR SYCL_ENABLED)`; add an `if(SYCL_ENABLED)` branch beside the CUDA/ROCm/TTNN branches at `:322-361` that sets `IOM_COEXIST_SYCL=1`, adds `${PROJECT_SOURCE_DIR}/src/sycl` and `${IOM_SYCL_INCLUDE_DIR}` include directories, adds `-fsycl` to `target_compile_options` and `target_link_options`, and links `iom_sycl`. The option pattern mirrors `iom_sycl_smoke_tests`/`iom_sycl_conformance_tests` at `:158-227`.
- **Modify:** `test/backend/test_backend_coexistence.cpp` — add the `#ifdef IOM_COEXIST_SYCL` include block (`<sycl/sycl.hpp>`, `"iom/sycl/device.hpp"`, `"runtime.hpp"`), a `SyclUsmAllocator`, a `sycl_runtime_device_count()` guard, `make_sycl_participant`, and the SYCL branches of the rejection and second-device cases, beside the existing `IOM_COEXIST_CUDA`/`IOM_COEXIST_ROCM`/`IOM_COEXIST_TTNN` blocks (`:26-54`, `:80-138`, `:172-227`, `:377-450`).
- **Read:** `test/sycl/test_sycl_conformance.cpp:71-130,156-206` — the established `SyclAllocator` (`sycl::malloc_shared`/`sycl::free` bound to the owned context), the `active_context_allocator` + `capture_context` switch, and the `ContextCallsRestore` RAII over `iom::sycl_detail::context_calls`; reuse this mechanism, including per-device context binding around each `make_sycl_device` call and restoration afterward.
- **Read:** `test/sycl/test_sycl_smoke.cpp:72-99` — `eligible_device_count_from_runtime()` (filter `sycl::device::get_devices()` by `is_gpu() || is_accelerator()`) is the runtime device-count pattern for the second-device guard; the factory throwing `std::invalid_argument` on an unavailable ordinal (`src/sycl/device.cpp:199-207`) is the required-hardware failure path.
- **Read:** `src/sycl/runtime.hpp` — `ContextCalls::context_ready(const sycl::context&)` is the test-only observation seam the allocator binds through; it is invoked exactly once during `SyclDevice` construction (`src/sycl/device.cpp:73-78`).
- **Read:** `src/sycl/device.cpp:65-126,127-182` — `SyclDevice` owns its `sycl::context` (independent instances on the same physical device are legal, unlike TTNN), and `SyclTensor` rejects allocator pointers whose `sycl::get_pointer_type` is `unknown` in the owned context — which is why the participant allocator must allocate USM, not heap.
- **Tests:** `test/backend/test_backend_coexistence.cpp` — the existing case structure is the contract: `coexistence_spec()` (`{2,3,17,33}` `BF16`, padded tiles), two queues per participant, host transfer + `require_logical_bytes` readback, two interleaved rounds × two bursts round-robin, reverse-order waits plus one repeatable wait, `queue_ids.size() == queue_count`, forward-order second round. The SYCL participant slots into this unchanged flow; `iom_conformance::read_logical` reads through each backend's `region_to_host`, so no oracle machinery is added.

## Requirements

1. The gate at `test/CMakeLists.txt:302` becomes `if(CUDA_ENABLED OR ROCM_ENABLED OR TTNN_ENABLED OR SYCL_ENABLED)`. With `SYCL_ENABLED=ON` and every other accelerator off, the target is created and the test runs with CPU+SYCL participants only. With no accelerator enabled the target still does not exist.
2. The `SYCL_ENABLED` branch defines `IOM_COEXIST_SYCL=1` on `iom_backend_coexistence_tests`, links `iom_sycl`, adds `${PROJECT_SOURCE_DIR}/src/sycl` and `${IOM_SYCL_INCLUDE_DIR}` as include directories, and adds `-fsycl` to both compile and link options. The single-translation-unit design is preserved: when `SYCL_ENABLED=ON` the whole project already compiles under icpx/dpcpp (`CMakeLists.txt:39-52`), and `-fsycl` engages SYCL compilation/linking for this TU exactly as it does for the per-backend SYCL test targets. A separate SYCL participant TU must not be introduced.
3. The `IOM_COEXIST_SYCL` include block in the coexistence TU pulls in `<sycl/sycl.hpp>`, `"iom/sycl/device.hpp"`, and `"runtime.hpp"`. It does not include any CUDA/HIP/TTNN header beyond what the existing blocks already include.
4. `SyclUsmAllocator` implements `iom::Allocator` by allocating `sycl::malloc_shared(size, device_, *context_)` (throwing `std::bad_alloc` on null) and freeing `sycl::free(buffer, *context_)`. Its context/device are bound through `iom::sycl_detail::context_calls.context_ready` while the owning `make_sycl_device` call runs, mirroring `SyclAllocator`/`bind_context`/`capture_context` in `test/sycl/test_sycl_conformance.cpp:71-161`; the previous `context_calls` value and active-allocator pointer are restored after each construction (RAII guard equivalent to `ContextCallsRestore`).
5. `make_sycl_participant` follows the CUDA/ROCm participant shape exactly (`test/backend/test_backend_coexistence.cpp:172-189`): `name = "sycl"`, `kind = iom::BackendKind::SYCL`, `ordinal = 0`, `owned_device = iom::make_sycl_device(0, allocator)`, one `source`, two queues, one `destination` per queue, all on `coexistence_spec()`. It is appended to `participants` in the main interleaved-copy case.
6. The main interleaved-copy case requires no SYCL-specific code path: the SYCL participant flows through the shared kind/ordinal checks, host transfer, `copy_from_host`/`require_logical_bytes` readback, interleaved `copy` submissions, reverse-order per-originating-queue waits, the repeated wait, and the `queue_ids.size() == queue_count` process-uniqueness check with SYCL queue ids included.
7. The "queues reject views from another device" case gains an `IOM_COEXIST_SYCL` branch constructing `iom::make_sycl_device(0, allocator)` and an independent `iom::make_sycl_device(0, foreign_allocator)` (separate owned contexts, separate allocators), asserting `BackendKind::SYCL` and equal ordinals on both, then running the shared `check_rejection` lambda for both copy directions. SYCL permits multiple live contexts per physical device, so the TTNN-style CPU stand-in is not needed.
8. The "second devices report their own ordinal" case gains an `IOM_COEXIST_SYCL` branch: a `sycl_runtime_device_count()` helper filters `sycl::device::get_devices()` by `is_gpu() || is_accelerator()` and `REQUIRE`s a positive count; when the count exceeds one, `iom::make_sycl_device(1, allocator)` constructs and reports `BackendKind::SYCL` with `backend_device() == 1`.
9. Hardware is required, never skipped: when `SYCL_ENABLED=ON` and no eligible SYCL device exists, `make_sycl_device(0, allocator)` throws `std::invalid_argument` (`src/sycl/device.cpp:199-207`) and the test fails. No count-based `if` may bypass participant construction in the main or rejection cases.
10. Record the spec-15 errata (section below) as part of this task's documentation outcome; it is the only doc edit and lives in this spec file.

## Spec-15 errata

Sub-spec 15 (`docs/changes/0001-tensor-view/15-backend-coexistence/spec.md`) is internally inconsistent: its Outcome already reads "one executable links and uses CPU, CUDA, ROCm, SYCL, and TTNN devices together", but its requirements line "The combined executable links `libiom`, `iom_cuda`, `iom_rocm` and `iom_ttnn` directly" and its blocked-by list (`08-rocm-storage-copy`, `10-cuda-storage-copy`, `14-ttnn-storage-copy`) predate SYCL's implementation, and the build wiring followed the stale lines. Effective immediately, read sub-spec 15 as follows:

- The combined executable links `libiom`, `iom_cuda`, `iom_rocm`, `iom_sycl`, and `iom_ttnn` — every enabled optional backend library — and compile-checks all enabled public factories in one translation unit.
- Sub-spec 15's effective blocked-by set includes `12-sycl-storage-copy`.
- Limitation: the "every enabled optional backend library" reading is constrained by the configured hosts. The root compiler gate at `CMakeLists.txt:39-52` forces `icpx`/`dpcpp` for the whole build whenever `SYCL_ENABLED=ON`, which is not the CUDA (`nvcc`/`CUDA::cuda_driver`) or TTNN (`TT::Metalium`/`TTNN::TTNN`) toolchain; no configured profile exposes both DPC++ and the CUDA/TTNN toolchain on a single host (`sycl`/`rocm` on `bv2`, `cuda`/`ttnn` on `bv1`). This task proves combined coexistence for SYCL+ROCm only; SYCL+CUDA and SYCL+TTNN combined builds remain deferred to the multi-toolchain remediation (task60) and are unproven until a per-target language switch or a shared multi-toolchain host is configured.

This errata is recorded here; this task does not edit `15-backend-coexistence/spec.md`.

## Non-goals

- SYCL queue/teardown hardening: porting the 49-ST-003 destruction registry, migrating `SyclQueue` onto `detail::StagedWorker`, submission-order or transactional-OOM behavior — owned by ST-001. The coexistence test must not depend on or assert those behaviors.
- The SYCL convergence onto shared machinery (StagedWorker, `standard_tiled_copy.inl` kernels, pooled staging) — owned by AR-001. AR-001 must not absorb the coexistence wiring; folding SYCL into coexistence is exclusively this task.
- Wiring `SyclStorageOracle` into the SYCL async-copy conformance — owned by NT-002. The coexistence target keeps its existing logical-bytes comparisons and adds no oracle.
- Extending the "queue ids release, reuse, and stay unique" case beyond its CPU-only form; process-wide uniqueness with SYCL ids is already asserted by the main case's `queue_ids.size() == queue_count` check.
- Any change to `src/sycl/**`, `include/iom/**`, the root `CMakeLists.txt`, or `15-backend-coexistence/spec.md`.
- Combined SYCL+CUDA or SYCL+TTNN coexistence builds on a single host: the root compiler gate at `CMakeLists.txt:39-52` forces `CMAKE_CXX_COMPILER` to `icpx`/`dpcpp` whenever `SYCL_ENABLED=ON`, which is not the CUDA or TTNN toolchain; no configured profile provides both DPC++ and the CUDA/TTNN toolchain on one host. Combined coexistence is proven only with ROCm (the `rocm` profile on `bv2` carries oneAPI alongside ROCm). Multi-toolchain support is deferred to task60 and is not part of this remediation.
- Cross-backend `DeviceOps::copy`, automatic backend selection, or performance comparison (sub-spec 15 non-goals remain in force).

## Acceptance criteria

- [ ] `IOM_COEXIST_SYCL` appears in `test/CMakeLists.txt` as a compile definition gated on `SYCL_ENABLED`, and the coexistence gate at `:302` includes `SYCL_ENABLED`.
- [ ] A SYCL-only configure (`-DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF`) creates `iom_backend_coexistence_tests`; the target builds and runs, with CPU+SYCL interleaved copies returning bit-identical `BF16` results.
- [ ] A SYCL-plus-ROCm configure (`-DSYCL_ENABLED=ON -DROCM_ENABLED=ON`, other accelerators OFF) is the required combined-build attempt. Toolchain caveat: with `SYCL_ENABLED=ON`, the root compiler gate at `CMakeLists.txt:39-52` makes `icpx`/`dpcpp` the project-wide `CMAKE_CXX_COMPILER`, and `ROCM_ENABLED` additionally calls `enable_language(HIP)` plus the config-mode `hip::host` import (`CMakeLists.txt:185-194`). The HIP language toolchain cooperating with a DPC++ project compiler is plausible but unproven on `bv2`. If the configure or build fails for a toolchain reason (HIP language selection, `hip::host` import under a DPC++ project compiler, mixed-linker errors), the failure is recorded verbatim and the combined coexistence leg is classified as blocked-on-tooling rather than a pass; SYCL-only coexistence (the previous criterion) then stands as the delivered evidence for this task, and multi-toolchain combined builds remain deferred to task60.
- [ ] When the SYCL-plus-ROCm build succeeds on `bv2` (`rocm` profile), the SYCL device constructs in the same process as the ROCm context, both backends' queues interleave copies, every destination matches the CPU reference bit-for-bit, and `queue_ids.size() == queue_count` holds across cpu+rocm+sycl. These observations are conditional on the build succeeding; they are not asserted otherwise.
- [ ] Combined SYCL+CUDA and SYCL+TTNN builds are explicitly deferred to task60 (multi-toolchain remediation) and are not asserted by this spec. The configured host profiles (`sycl`/`rocm` on `bv2`, `cuda`/`ttnn` on `bv1`) do not provide both DPC++ and the CUDA/TTNN toolchain on a single host.
- [ ] The rejection case rejects SYCL foreign-device views in both directions with `std::invalid_argument` while the same-device path stays covered by the main case.
- [ ] With more than one eligible SYCL device, the second-device case reports `BackendKind::SYCL` at ordinal 1; with exactly one, the branch is inert without failing.
- [ ] With `SYCL_ENABLED=ON` on a host with no eligible SYCL device, the test fails rather than skips.
- [ ] Disabling `SYCL_ENABLED` removes only the SYCL branch: the target still builds for every remaining enabled combination, and no other backend's targets or dependencies change.

## Verification

Accelerator verification runs through the repository's remote-development procedure (`.agents/skills/remote-development`). Each invocation must use the exact configured profile that exposes the toolchain the build needs, and every GPU build/test step must hold an exclusive lock (`flock /tmp/agent-gpu0.lock`) so concurrent agents do not contend for the device. The configured profile map is fixed and authoritative:

- `sycl` → `bv2` (sources `/opt/intel/oneapi/setvars.sh`)
- `rocm` → `bv2` (sources `/home/rlew/rocm_env.sh`)
- `cuda` → `bv1` (sources `/home/rlew/cuda_env.sh`)
- `ttnn` → `bv1` (sources `/home/rlew/ttnn_env.sh`)

The local workspace is authoritative and is synced before every remote step. The focused target is `iom_backend_coexistence_tests` — not the full suite. The two required configurations (SYCL-only on the `sycl` profile and SYCL+ROCm on the `rocm` profile) both run on `bv2`; SYCL+CUDA and SYCL+TTNN combined builds are deferred to task60 (multi-toolchain remediation) and are not asserted by this spec.

SYCL-only on the `sycl` profile (proves the gate extension and the minimal CPU+SYCL process):

```bash
.agents/skills/remote-development/scripts/remote-sync sycl task-53-cc003
.agents/skills/remote-development/scripts/remote-exec sycl task-53-cc003 \
  'flock /tmp/agent-gpu0.lock -c "cmake -S . -B build/sycl-cc003 -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF"'
.agents/skills/remote-development/scripts/remote-exec sycl task-53-cc003 \
  'flock /tmp/agent-gpu0.lock -c "cmake --build build/sycl-cc003 --target iom_backend_coexistence_tests -j"'
.agents/skills/remote-development/scripts/remote-exec sycl task-53-cc003 \
  'flock /tmp/agent-gpu0.lock -c "ctest --test-dir build/sycl-cc003 --output-on-failure -R iom_backend_coexistence_tests"'
```

SYCL+ROCm on the `rocm` profile — required attempt, not a guaranteed pass; if configure/build fails for a toolchain reason the leg is blocked-on-tooling and SYCL-only coexistence is the delivered evidence. The `rocm` profile sources ROCm, the command additionally sources oneAPI so DPC++ is available alongside hipcc:

```bash
.agents/skills/remote-development/scripts/remote-sync rocm task-53-cc003
.agents/skills/remote-development/scripts/remote-exec rocm task-53-cc003 \
  'flock /tmp/agent-gpu0.lock -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>/dev/null; cmake -S . -B build/sycl-rocm-cc003 -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DTTNN_ENABLED=OFF"'
.agents/skills/remote-development/scripts/remote-exec rocm task-53-cc003 \
  'flock /tmp/agent-gpu0.lock -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>/dev/null; cmake --build build/sycl-rocm-cc003 --target iom_backend_coexistence_tests -j"'
.agents/skills/remote-development/scripts/remote-exec rocm task-53-cc003 \
  'flock /tmp/agent-gpu0.lock -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>/dev/null; ctest --test-dir build/sycl-rocm-cc003 --output-on-failure -R iom_backend_coexistence_tests"'
```
The combined SYCL+ROCm build is the required attempt, not a guaranteed pass. With `SYCL_ENABLED=ON`, the root compiler gate at `CMakeLists.txt:39-52` makes `icpx`/`dpcpp` the project-wide CXX compiler, and `ROCM_ENABLED` additionally calls `enable_language(HIP)` (`CMakeLists.txt:194`) plus the config-mode `hip::host` import (`CMakeLists.txt:185-194`). The HIP language toolchain cooperating with a DPC++ project compiler is plausible but unproven on `bv2`. If `cmake -S` or the build fails for a toolchain reason (HIP language/compiler selection under a DPC++ project compiler, `hip::host` import failure, mixed-linker errors), record the exact error and classify the combined leg as blocked-on-tooling: SYCL-only coexistence is the delivered evidence for this task, and multi-toolchain combined builds remain deferred to task60.

Required observations:

- SYCL-only on `bv2` (`sycl` profile) is the required pass: the test constructs one SYCL device and the CPU device in one process, interleaves two rounds of copies across both SYCL queues and both CPU queues without intermediate waits, waits in reverse participant order plus one repeatable wait, and every SYCL destination matches the CPU-encoded `expected` bytes bit-for-bit (`require_logical_bytes` reports no divergence).
- SYCL+ROCm on `bv2` (`rocm` profile) is the required attempt. If the configure and build succeed: the process-uniqueness check passes with queue ids from all participants, and the SYCL device coexists with the ROCm context for the whole case (no cross-backend symbol collisions at link time, no context interference at runtime). The DPC++ compiler is the project compiler; the HIP language target is enabled alongside CXX. If the configure or build fails for a toolchain reason, record the exact error and classify the combined leg as blocked-on-tooling: SYCL-only coexistence stands as the delivered evidence for this task, and multi-toolchain combined builds are deferred to task60.
- Negative evidence: on the pre-fix tree, the SYCL-only configure leaves `iom_backend_coexistence_tests` undefined (`cmake --build` fails with an unknown-target error) and the combined build's binary contains no SYCL participant; post-fix both commands succeed. Source inspection confirms `IOM_COEXIST_SYCL` exists only inside the `SYCL_ENABLED` branch and that `src/sycl/**`, `include/iom/**`, and `15-backend-coexistence/spec.md` are untouched by the change.
- Deferral note: SYCL+CUDA combined (`-DSYCL_ENABLED=ON -DCUDA_ENABLED=ON`) and SYCL+TTNN combined (`-DSYCL_ENABLED=ON -DTTNN_ENABLED=ON`) builds are not attempted in this task. No configured profile runs DPC++ alongside the CUDA/TTNN toolchain on one host; the root compiler gate at `CMakeLists.txt:39-52` currently forces a single project compiler, and that compiler is not the CUDA or TTNN toolchain. Multi-toolchain coexistence is owned by task60. A blocked-on-tooling SYCL+ROCm build falls under the same deferral: the combined toolchain is not converged, and task60 (multi-toolchain remediation) is where both the HIP/DPC++ and the CUDA-or-TTNN/DPC++ combinations are unblocked.

Clean up the remote mirrors when done:

```bash
.agents/skills/remote-development/scripts/remote-clean sycl task-53-cc003
.agents/skills/remote-development/scripts/remote-clean rocm task-53-cc003
```

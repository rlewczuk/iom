
**Status:** done

## Summary

Added the test-only `iom::sycl_detail::LaunchCalls` seam and instrumented all three SYCL `parallel_for` submission sites without changing kernel or queue behavior. Added the hardware smoke regression `SYCL queued copy submits one kernel regardless of plane count`, covering one-plane and 64-plane F32 copies, identical-window zero launch, and byte-exact readback. The throwaway `_local/` throughput driver and baseline compatibility shim were used for measurement and removed after cleanup; no CMake or permanent benchmark target was added.

## Verification

- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DCMAKE_BUILD_TYPE=Release` via remote-development profile `sycl`, task `pf001-71-20260906` — configured with IntelLLVM 2026.1.
- `cmake --build build/sycl --target iom_sycl_smoke_tests iom_sycl_conformance_tests iom_tests -j` via remote-development — all three targets built successfully.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"` — 2/2 tests passed.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"` — 1/1 test passed.
- `flock /tmp/agent-gpu0.lock ./build/sycl/test/iom_sycl_smoke_tests -tc="*one kernel regardless of plane count*"` — 1 case passed, 161/161 assertions passed; launch counts were 1 for the one-plane copy, 1 for the 64-plane copy, equal between those copies, and 0 for the identical window, with byte-exact readback checks passing.
- After rebasing onto integration commit `e208947...`, the same Release SYCL configure/build and `flock /tmp/agent-gpu0.lock ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"` plus `ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"` and focused launch case — all targets built, 2/2 SYCL tests passed, `iom_tests` passed, and 161/161 focused assertions passed.
- `sycl-ls` on both the post-fix task mirror `pf001-71-20260906` and baseline mirror `pf001-71-20260906-base` — identical devices: two `[level_zero:gpu]` Intel Arc Pro B60 entries, two `[opencl:gpu]` Intel Arc Pro B60 entries, and one `[opencl:cpu]` AMD Ryzen 7 9700X entry.
- Baseline source was commit `8a1e2f2fa31db36ec616c830cdcfbdb263833b50` (`spec-run-task(0001-tensor-view/63-AR-004-share-cuda-rocm-copy-infrastructure): share CUDA and ROCm copy infrastructure`), verified to contain the pre-convergence `PlanePair` loop. The same driver source, shapes, median-of-5 policy, and public operations were used on both sides; each side compiled once and ran three times under `flock /tmp/agent-gpu0.lock`.

| metric (GB/s) | baseline samples | baseline median | post-fix samples | post-fix median | delta |
| --- | --- | ---: | --- | ---: | ---: |
| F32 `{4096,4096}` `copy_from_host` | 5.3250357, 5.0453316, 4.9803761 | 5.0453316 | 9.2602102, 9.5318294, 9.6410379 | 9.5318294 | +4.4864978 |
| F32 `{4096,4096}` `copy_to_host` | 0.036614644, 0.036570990, 0.036524656 | 0.036570990 | 9.9032566, 9.9525225, 9.9418138 | 9.9418138 | +9.90524281 |
| F32 `{4096,4096}` queued `copy` + `wait` | 20.831552, 20.724555, 20.844712 | 20.831552 | 39.947750, 39.901813, 39.957502 | 39.947750 | +19.116198 |
| I4 `{2048,2048}` `copy_from_host` | 0.81789712, 0.81917440, 0.81944039 | 0.81917440 | 5.7039596, 5.7695378, 5.7527301 | 5.7527301 | +4.93355570 |
| I4 `{2048,2048}` `copy_to_host` | 0.019386016, 0.019337909, 0.019311360 | 0.019337909 | 5.9723418, 5.9733625, 5.9772953 | 5.9733625 | +5.95402459 |
| I4 `{2048,2048}` queued `copy` + `wait` | 0.79722826, 0.79575507, 0.79359509 | 0.79575507 | 9.2783252, 9.3462219, 9.3350308 | 9.3350308 | +8.53927573 |

- `git diff --check`, repository file-boundary audit, absence of `test/sycl/test_sycl_bench.cpp`, absence of `iom_sycl_bench`, and zero `CHECK_(GE|LE|GT|LT)` matches in the throwaway driver — passed while the artifacts existed.
- Remote mirrors, baseline worktree, local `_local/sycl_throughput.cpp`, `_local/sycl_baseline_compat.hpp`, and compiled throwaway binaries — removed after measurement.

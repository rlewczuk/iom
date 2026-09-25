# ROCm profiling and diagnostics on bv2

This guide uses the existing ROCm 10.0 environment; it does not install packages or alter the SDK. Host inventory is dated **2026-09-25**: bv2 is Ubuntu 26.04.1 / Linux 7.0.0-31 with Radeon AI PRO R9700 (`gfx1201`). `/opt/rocm` resolves to the ROCm 10.0 tree; observed HIP is 7.15.26333, `rocprofv3` 1.3.5, `rocprof-compute` 3.8.0 and ROCgdb 16.3. `/dev/kfd` exists (verified read-only 2026-09-25); the SSH user belongs to `video` and `render`. This verifies device-node presence/groups only, not profiler counter support. See [tool inventory](TOOLS.md).

## Sync and environment

All edits belong in the exact selected local checkout/worktree. Sync before every remote command using the absolute helper path from the integration checkout, not `.omp` under the selected worktree. Pass the canonical task directory and selected workspace explicitly, with `rocm` (`bv2`) and a unique mirror id:

```bash
CSW_REMOTE_TASK_DIR="/absolute/integration/.cswd/tasks/<task>" \
CSW_REMOTE_WORKSPACE="/absolute/path/to/selected/worktree" \
  /absolute/integration/.omp/csw/bin/csw-remote-sync rocm <unique-id>
```

Use the same environment with `/absolute/integration/.omp/csw/bin/csw-remote-exec rocm <unique-id> '<remote command>'`. Confirm the sync output reports the intended workspace. Only tracked files (plus `_local/`) are synced. The profile sources `/home/rlew/rocm_env.sh`; independently check that it succeeded because the configured source uses `|| true`:

```bash
source /home/rlew/rocm_env.sh
hipcc --version
rocprofv3 --version
rocminfo
```

The selected runtime and trace tool are already present under `/opt/rocm/bin`. For a different host, use AMD's version-matched [ROCprofiler-SDK installation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/install/install.html) and ROCm [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html); do not overlay or upgrade this working SDK merely to match a newer web page.

## Selected usable path: rocprofv3 dispatch trace

Prior committed IOM evidence on `gfx1201` used `rocprofv3 --kernel-trace --hip-trace --sys-trace -f csv` to attribute dispatches. It explicitly states `rocprofv3` 1.3.5 on this device did not support PC sampling or SPM counter collection and that host enqueue/completion observations are **not** native device time. See [ROCm official inference evidence](../BACKEND_CONTRACT/model-inference-rocm-official-evidence.md#bounded-profiler-payload). This is prior evidence, not a fresh profiling run or claim that every installed service works.

For future trace collection, first sync the exact workspace and prepare the already-selected workload. The example below is a template; replace `./build/test/iom_rocm_conformance_tests` with the intended bounded workload and choose a non-existing evidence directory. HIP/API tracing alone does not imply kernel/copy records, hence separate trace options. Locks are in required order and cover the entire bounded process:

```bash
flock -w 120 /tmp/iom-perf.lock \
  flock -w 120 /tmp/agent-gpu0.lock \
  timeout --kill-after=30s 1800s \
  rocprofv3 --kernel-trace --hip-trace --sys-trace -f csv \
    -d /tmp/iom-rocm-profile -- \
    ./build/test/iom_rocm_conformance_tests
```

Retain stdout/stderr, exit status and generated CSV alongside the task evidence. Do not compare trace-instrumented duration to unprofiled host latency. For general tool discovery only, `rocprofv3-avail` is a listed installed helper; its output is not proof a service/counter works on `gfx1201`.

## Explicit support limits

- **ROCm Compute Profiler is not a supported R9700 setup path in this guide.** Although `rocprof-compute` 3.8.0 is installed, its published supported-accelerator table omits Radeon AI PRO R9700 / `gfx1201`. Do not run profile/roofline as a harmless permission check or claim support. AMD's quickstart also requires AMDGPU, `/dev/kfd`, `/dev/dri`, video/render membership and Python; node existence alone cannot close the device-support gap.
- **Counters/PMC:** enumerate capabilities and check the exact device/service documentation before planning collection. Counter services are device-dependent; some RDNA3/4 services may require `STABLE_STD` performance level. Any clock/performance-policy change requires administrator approval and is outside this guide. No PC sampling, SPM or general counter capability is claimed.
- **ROCm Systems Profiler:** `/opt/rocm/bin/rocprof-sys-run` exists, but no working run was demonstrated. Its PAPI counters commonly require perf access (`perf_event_paranoid <=2` in the inspected docs); bv2 reports 4. Do not lower the sysctl. SELinux/instrumentation constraints may apply.
- **GPU ASan:** ROCm 10.0 documentation lists `gfx942`/`gfx950` packages, not `gfx1201`; do not install/use it as an R9700 recipe. Host Valgrind cannot substitute for GPU memory instrumentation.
- **Debugging:** ROCgdb 16.3 is installed; use matching [ROCgdb docs](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/index.html), but debugger presence/enumerated agent does not establish a profiler metric. `rocminfo` is device enumeration, not a profiler.

## Measurement hygiene

Every comparison requires a fresh exact-workspace sync, matched code/build/workload, same device/profile and recorded conditions. Take `/tmp/iom-perf.lock` before `/tmp/agent-gpu0.lock`, both with bounded waits; wrap all setup/timed work in `timeout --kill-after`. The locks coordinate IOM processes only. Record tool/runtime versions, exact source commits, workload, trace files and observed contention/thermal/frequency conditions. Dispatch attribution, host enqueue/completion timing and hardware counters are distinct claims. This guide changes no security policy, driver configuration, clocks, or installed SDK.

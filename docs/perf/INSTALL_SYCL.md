# Intel SYCL profiling and diagnostics on bv2

This uses the observed Intel oneAPI installation without installing or changing an SDK. Inventory date **2026-09-25**: bv2 is Ubuntu 26.04.1 / Linux 7.0.0-31, with two Intel Arc Pro B60 GPUs on Level Zero V2; `icpx` is 2026.1.0, `sycl-trace` and `gdb-oneapi` are from oneAPI 2026.1.0. After initialization, `sycl-ls` listed both B60 adapters (driver `1.15.38646+7`). Without initialization, absolute `sycl-ls` returned “No platforms found”; setup is required on every invocation. The remote recheck confirmed the installed tree and enumeration, not profiler execution. See [tool inventory](TOOLS.md).

## Sync and nounset-safe oneAPI setup

Sync the exact selected local checkout/worktree before each remote command. Use the integration checkout's absolute helper path (the selected worktree may not contain ignored `.omp` assets), and pass the canonical task directory and selected workspace explicitly:

```bash
CSW_REMOTE_TASK_DIR="/absolute/integration/.cswd/tasks/<task>" \
CSW_REMOTE_WORKSPACE="/absolute/path/to/selected/worktree" \
  /absolute/integration/.omp/csw/bin/csw-remote-sync sycl <unique-id>
```

Use the same values for every `/absolute/integration/.omp/csw/bin/csw-remote-exec sycl <unique-id> '<remote command>'`; check sync output identifies the intended workspace. Only tracked files (plus `_local/`) are synchronized. For remote-exec, the configured `sycl` profile must have an **empty optional setup field**: helper shell startup is nounset (`set -u`) and oneAPI `setvars.sh` reads unset variables. For supplemental commands, make an outside-checkout config override containing the same host/base (`sycl|bv2|agent-work/iom|`) and point `CSW_REMOTE_CONFIG` to that override for every helper invocation. Do not edit shared config for a one-off command.

At the **start of every remote command**, initialize the SDK by temporarily disabling nounset, then restore it before subsequent work:

```bash
set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls
```

This exact sequence is important; do not rely on a previous shell's environment. Check the setup log/exit status if initialization fails. The installed SDK is at `/opt/intel/oneapi/2026.1`; do not install/upgrade it. If setting up another host, check Intel's [oneAPI installation documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and the exact release's OS/device support before selecting packages.

## Selected usable path: sycl-trace dispatch attribution

Prior IOM evidence retained oneAPI 2026.1 `sycl-trace --ur.call` records tying Level Zero/B60 joint-matrix dispatches to the official inference workload. It reports call records only—no hardware counter, ISA instruction count, occupancy or device-duration claim. This is historical evidence, not a new profiler run: [SYCL official inference evidence](../BACKEND_CONTRACT/model-inference-sycl-official-evidence.md#sycl-ur-matrix-and-timing-evidence).

The trace launcher has an important observed constraint. In the retained testing, a wrapper target named `python3` failed to launch; `/usr/bin/python3` wrapper completed but did not trace the conformance grandchild owning device work. The supplemental trace targeted the conformance binary itself. Use an absolute executable path when wrapping an interpreter, and prefer tracing the actual device workload executable when attribution requires it. The example is a **future-use template**, not a run performed for this guide; replace the binary/arguments with the intended workload and use a fresh output/evidence location:

```bash
flock -w 120 /tmp/iom-perf.lock \
  flock -w 120 /tmp/agent-gpu0.lock \
  timeout --kill-after=30s 1800s \
  bash -lc 'set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-trace --print-format=verbose --ur.call ./build/test/iom_sycl_conformance_tests'
```

The enclosing remote command must itself be invoked with the no-setup profile override above, so the helper's nounset shell does not source setvars before this explicit initialization. For an evidence run, first sync and build the exact workspace, choose workload and test-case filters intentionally, and retain enumeration, trace output, stdout/stderr and exit status. The outer `flock` and `timeout` cover SDK setup as well as profiling. Traced time is not a comparable unprofiled benchmark score.

## Tools not established for bv2

- **VTune:** No `vtune` executable was found in checked PATH/top-level product locations (not proof it is absent elsewhere). Although the inspected 2026 release notes name Arc Pro B-Series, its GPU OS list includes Ubuntu 22.04/24.04/25.04, not bv2's Ubuntu 26.04. Do not install or promise GPU Hotspots/XPU Offload here without Intel-confirmed OS/device support and appropriate driver/perf access. The inspected VTune Linux kernel minimum is internally inconsistent (>=4.11 vs >=4.14); use the stricter requirement when assessing a candidate release.
- **Intel Distribution for GDB:** `gdb-oneapi` 2026.1 exists and B-Series is named, but its inspected tested-Ubuntu list ends at 24.04. GPU debug packages and Ubuntu 26.04/B60 compatibility remain unverified. Installed executable does not prove it can debug this device.
- **Advisor:** no executable found in checked locations. Advisor was removed from the default oneAPI Toolkit in 2026.0; the inspected 2025.4 Roofline device/OS requirements do not establish B60/Ubuntu 26.04. No install recipe is offered.
- **PTI `unitrace`:** not found in checked PATH/top-level locations and is a different tool from `sycl-trace`. Rolling PTI source documentation is not evidence of a pinned installation or B60 metric access. Do not call it installed.

## Measurement hygiene

Before each run, sync the precise local workspace and record commit, workload, compiler/runtime/driver and device. Use matched baseline/candidate builds and the same B60 selection. Take `/tmp/iom-perf.lock` first and `/tmp/agent-gpu0.lock` second with bounded waits; bound all setup/timing with `timeout --kill-after`. Cooperative locks cannot exclude unrelated system load. Preserve raw trace and observed contention. A SYCL host enqueue/completion interval is not GPU device duration; trace attribution, device counters and benchmark elapsed time remain separate evidence classes. No package, permission, driver, clock, or kernel security change is made here.

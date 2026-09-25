# CUDA profiling and diagnostics on bv1

Instructions for the **existing** installation; this does not install or upgrade a toolkit. Host inventory is dated **2026-09-25** (read-only, not a fresh profiler run): bv1 is Ubuntu 24.04.4, Linux 6.14.0-37, RTX 5090, NVIDIA driver 595.71.05. `nvidia-smi` reports driver CUDA compatibility 13.2; Toolkit `nvcc` is 13.2.78 (these are distinct). Nsight Systems 2026.4.1 is `/usr/local/bin/nsys`; Nsight Compute 2026.1.1, Compute Sanitizer 2026.1.1, and CUDA-GDB 13.2 are under `/usr/local/cuda/bin`. Their Toolkit commands were not in the unsourced login PATH. NVIDIA reports `RmProfilingAdminOnly: 1`, so do not assume Nsight Compute counters are accessible. See [inventory](TOOLS.md).

## Workspace sync and environment

Edit in the exact local checkout/worktree and sync it before every remote build, test, profile, or execution. Use the integration checkout's absolute helper path (not the selected worktree's ignored `.omp` path), canonical local task directory, selected workspace, and a unique mirror id with the configured `cuda` profile (`bv1`):

```bash
CSW_REMOTE_TASK_DIR="/absolute/integration/.cswd/tasks/<task>" \
CSW_REMOTE_WORKSPACE="/absolute/path/to/selected/worktree" \
  /absolute/integration/.omp/csw/bin/csw-remote-sync cuda <unique-id>
```

Use the same environment with `/absolute/integration/.omp/csw/bin/csw-remote-exec cuda <unique-id> '<remote command>'`; verify sync output identifies the intended workspace. `csw-remote` sends tracked files (and `_local/`), not arbitrary untracked files. The profile sources `/home/rlew/cuda_env.sh`, which sets `CUDA_HOME=/usr/local/cuda` and adds its `bin`; the remote setup is configured with `|| true`, so independently check initialization. For an interactive remote shell:

```bash
source /home/rlew/cuda_env.sh
nvcc --version
/usr/local/bin/nsys --version
/usr/local/cuda/bin/ncu --version
```

Current binaries are already installed. If another compatible host needs installation, use NVIDIA's [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and version-matched [Nsight Systems installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/); match OS, kernel, driver, GPU and tool release first. Do not replace this working SDK or assume a current guide describes the installed versions.

## Selected usable path: Nsight Systems timeline

Nsight Systems is the selected installed CUDA trace path. Prior committed IOM evidence successfully ran `nsys profile` and `nsys stats --report cuda_gpu_kern_sum`, attributing CUDA kernel launches and reporting device-kernel duration. This is **historical evidence**, not a new run; details and exact preserved payload are in [CUDA official inference evidence](../BACKEND_CONTRACT/model-inference-cuda-official-evidence.md#native-bf16-matrix-evidence). Nsight Systems timeline attribution is not the same as unprofiled end-to-end host latency. CPU sampling/backtraces may be limited by perf permissions: the inspected Nsight Systems 2025.3 Linux guide calls for `perf_event_paranoid <=2` for those sampling paths, whereas bv1 reports 4. CUDA timeline precedent does not establish those sampling features.

For a future controlled trace, first sync the exact workspace and build/run the already-selected workload as required by its own procedure. Set `TRACE_ID` to a unique run id for each trace; the command fails rather than reusing an unset id and does not request overwrite. Hold both host-wide measurement and GPU locks **in that order**, covering the full bounded profiling process:

```bash
flock -w 120 /tmp/iom-perf.lock \
  flock -w 120 /tmp/agent-gpu0.lock \
  timeout --kill-after=30s 1800s \
  /usr/local/bin/nsys profile \
    -o "/tmp/iom-cuda-profile-${TRACE_ID:?set TRACE_ID to a unique run id}" \
    ./build/test/iom_cuda_conformance_tests
```

Inspect an existing report separately, preserving the distinction between profiler trace output and benchmark timing:

```bash
flock -w 120 /tmp/iom-perf.lock \
  timeout --kill-after=10s 120s \
  /usr/local/bin/nsys stats --report cuda_gpu_kern_sum "/tmp/iom-cuda-profile-${TRACE_ID:?set TRACE_ID to the trace run id}.nsys-rep"
```

The earlier IOM run used an official inference wrapper and its exact command is linked above; this shorter command is a template, not a claim about the example executable's test selection or a profile that was run here. Reports can contain environment/workload details; keep them with task evidence and avoid overwriting retained results.

## Nsight Compute, sanitizer, debugger limits

`ncu` 2026.1.1 is present at `/usr/local/cuda/bin/ncu`, but driver's observed `RmProfilingAdminOnly: 1` restricts access to GPU profiling metrics. The support/permission check is an administrator decision before any counter collection. Do not change NVIDIA module settings, use elevated collection, or present installed `ncu` as proof counters work. NVIDIA [Nsight Compute requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html) and [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) are release-specific; replay/multiple passes can alter duration. No counter command is offered as an ordinary-user recipe.

For correctness faults, use version-matched [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) (installed 2026.1.1; online manual inspected was 2026.3). It provides memcheck/racecheck/initcheck/synccheck diagnostics, not timing. CUDA-GDB 13.2 is available at `/usr/local/cuda/bin/cuda-gdb`; the inspected online manual is 13.4, so consult matching version docs. Both need a compatible workload/debug information. Do not interpret instrumented or debug runs as representative performance.

## Measurement hygiene

Each measured run requires a fresh sync of the exact workspace, matching baseline/candidate source and build, a fixed workload, same host/GPU, and raw evidence. Use `/tmp/iom-perf.lock` before `/tmp/agent-gpu0.lock`; bound both lock waits and the complete operation with `timeout --kill-after`. These cooperative locks cannot exclude unrelated work. Record profiler and SDK versions, source revision, environment, output path, and contention. A profiler's device interval, host wrapper elapsed time, and ordinary benchmark latency are different quantities; never substitute one for another. No sysctl, driver security setting, or package installation is changed by this guide.

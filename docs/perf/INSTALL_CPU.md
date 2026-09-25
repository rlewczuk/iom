# CPU performance tools

This is a use guide, not an installer. Inventory is dated **2026-09-25**; the remote host evidence was read-only and did not run a profiler. `bv1` (`cpu` profile) is Ubuntu 24.04.4 / Linux 6.14.0-37 with Ryzen 9 9950X and `perf` 6.14.11, Valgrind 3.22.0. `bv2` (`rocm` or `sycl` profile) is Ubuntu 26.04.1 / Linux 7.0.0-31 with Ryzen 7 9700X, `perf` 7.0.14, Valgrind 3.26.0. Both report `/proc/sys/kernel/perf_event_paranoid=4`. See [tool inventory](TOOLS.md) for the complete CPU/GPU distinctions and limitations.

## Prepare the exact remote workspace

Edit in the selected local checkout/worktree. Before **each** remote command, sync that exact workspace; use the integration checkout's current absolute helper path, not a helper relative to the selected worktree. Supply the canonical local task directory and selected workspace explicitly. Use a unique task/mirror id and the configured `cpu` host profile (bv1):

```bash
CSW_REMOTE_TASK_DIR="/absolute/integration/.cswd/tasks/<task>" \
CSW_REMOTE_WORKSPACE="/absolute/path/to/selected/worktree" \
  /absolute/integration/.omp/csw/bin/csw-remote-sync cpu <unique-id>
```

Run the corresponding command through `/absolute/integration/.omp/csw/bin/csw-remote-exec cpu <unique-id> '<remote command>'` with the same `CSW_REMOTE_TASK_DIR` and `CSW_REMOTE_WORKSPACE`. Check the sync output names the intended workspace. `csw-remote` synchronizes tracked files (plus `_local/`); stage new project files first. The helper's configured CPU environment setup is `/home/rlew/cpu_env.sh` on this host. A profile setup line ending in `|| true` is not proof the setup succeeded; inspect versions/paths on the remote.

## Available paths and safe checks

No package installation is needed for the observed tools. Verify existing versions remotely without collecting events:

```bash
perf --version
valgrind --version
cat /proc/sys/kernel/perf_event_paranoid
```

If a tool is absent on another host, consult the official [Valgrind download/install page](https://valgrind.org/downloads/) or that OS release's supported package repository; do not assume the installed 3.22/3.26 version matches current manual 3.27.1 or all newer CPU instructions. OProfile is historical/optional (upstream download page lists source 1.4.0; docs describe 1.3.0); it was not found in checked PATH on either host and is not a workaround for perf-event policy.

### Native sampling with Linux perf

`perf record` samples host events statistically. Before running, check local policy and consult the host administrator if access is denied; both observed machines report `perf_event_paranoid=4`, so unprivileged hardware PMU collection is **not promised**. Linux documents narrower `CAP_PERFMON` access; no guide command changes sysctl or grants capabilities. Do not broaden privileges yourself.

For a host CPU profile, use a representative optimized CPU executable and bounded run. This example writes samples to `/tmp/iom-cpu-perf.data`; replace the output path, executable, and arguments with the task-local evidence path and frozen workload. It may fail for permissions or unavailable CPU events; preserve the diagnostic as a blocked capability check rather than changing policy.

```bash
flock -w 120 /tmp/iom-perf.lock \
  timeout --kill-after=30s 180s \
  perf record -o /tmp/iom-cpu-perf.data --call-graph dwarf -- ./build/test/iom_backend_conformance_cpu_tests
```

Then inspect the already captured data in a separate bounded, locked command:

```bash
flock -w 120 /tmp/iom-perf.lock \
  timeout --kill-after=10s 60s \
  perf report --stdio -i /tmp/iom-cpu-perf.data
```

A successful `perf --version` or `perf list` does not demonstrate sample permission. Samples are not GPU timing and require comparable workload, CPU affinity/frequency conditions, and event support.

### Valgrind attribution, allocation, and correctness

Valgrind tools instrument host execution and add substantial overhead. Use them to locate relative host instruction/call/cache costs (`callgrind`), heap allocation sites (`massif`), or host memory errors (`memcheck`), not as a performance score. Build the selected CPU target first using the project's normal optimized configuration; debug information improves symbol attribution. Memcheck often benefits from a lower-optimization diagnostic build, which is separate from the representative timed build. Keep profiler output separate from benchmark evidence.

Callgrind example for an already-built representative executable:

```bash
flock -w 120 /tmp/iom-perf.lock \
  timeout --kill-after=30s 300s \
  valgrind --tool=callgrind --callgrind-out-file=/tmp/iom-callgrind.out \
    ./build/test/iom_backend_conformance_cpu_tests
```

Inspect the resulting modeled call/event attribution with `callgrind_annotate`; it is simulated cost, not PMU time. Massif writes a heap profile (`--tool=massif --massif-out-file=...`); its defaults omit direct mmap/brk and do not report process RSS. Memcheck is correctness instrumentation (`--tool=memcheck --leak-check=full`), not a performance run. All workload runs must use bounded locks/timeouts as above.

## Compare and report

Freeze the workload, exact source revision, executable/build flags, host/profile, run conditions and metric before comparing. Run baseline/candidate on the same host and comparable conditions; report raw observations and variability. `perf` sampling is statistical, Valgrind is instrumented/model output, and ordinary elapsed host time is not accelerator device duration. Cooperative locks exclude only other cooperating IOM measurements, not arbitrary host jobs; record observed load/frequency drift and classify contaminated runs as inconclusive. Never alter `perf_event_paranoid` or install/enable OProfile to bypass access restrictions.

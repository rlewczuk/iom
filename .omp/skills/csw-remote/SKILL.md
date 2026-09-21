---
name: csw-remote
description: Perform implementation work in a local Git workspace while building, testing, and running exclusively on a selected remote Linux host over SSH.
---

# CSW Remote

Use this skill for CPU, CUDA, ROCm, and SYCL remote validation. All CPU-backend and general unit tests run on the configured `cpu` profile, not the local workstation.

## Core rule

The selected local Git workspace is authoritative.

- Edit files locally in the selected checkout or worktree.
- Before every remote build, test, benchmark, or execution, sync that exact workspace to a unique remote task directory.
- Run builds/tests/programs only through SSH on the selected remote host.
- Do not edit the shared remote checkout directly unless explicitly required.
- Do not sync `.git`, `.work`, `.worktrees`, or `.cswd` to the remote mirror.
- Keep one remote directory per concurrent task.

## Host configuration

Copy `references/hosts.example.conf` to a convenient project-local file such as:

```text
.remote-hosts.conf
```

Then edit host aliases, remote base directories, and optional setup commands.

The helper scripts read the config path from `CSW_REMOTE_CONFIG`. If unset, they first use `.remote-hosts.conf` in the selected workspace. A linked worktree without its own config falls back to `.remote-hosts.conf` in the primary checkout; the config is read from there, never synchronized.

## Workspace selection

The scripts resolve the selected workspace with Git:

- By default, they use the worktree containing the current directory, even when the command is invoked from a subdirectory.
- Set `CSW_REMOTE_WORKSPACE` to a path inside the intended checkout or worktree when invoking a script from elsewhere.
- They compare the worktree Git directory with the common Git directory and report `checkout` or `worktree` in `csw-remote-sync` output.

For a `csw-run-worker` task, run from its exact `.work/<task-path>` worktree. Every helper invocation requires both `CSW_REMOTE_TASK_DIR` set to the local directory containing the task's returned `spec_path` and `task.yml`, and `CSW_REMOTE_WORKSPACE` set to the exact assigned worktree. The scripts append the invocation and remote stdout/stderr to `remote.log` in the task directory. Do not point either variable into the remote mirror.

```bash
CSW_REMOTE_TASK_DIR="$(dirname "$spec_path")" \
CSW_REMOTE_WORKSPACE="$worktree" \
  /absolute/integration/.omp/csw/bin/csw-remote-sync rocm task-123
```

Use the integration checkout's current remote helper scripts while selecting the prepared worktree through `CSW_REMOTE_WORKSPACE`; a task worktree may contain an older helper version. Use the same two environment values for `csw-remote-exec` and `csw-remote-clean`. Never sync a `csw-run-worker` task from the primary checkout. Confirm that `csw-remote-sync` reports the exact assigned worktree before remote execution.

The worktree's `.cswd` link exposes local task specifications, lifecycle controls, and evidence only. Read specifications and update task status locally through the task helpers. Never run task-control helpers remotely, resolve remote paths through `.cswd`, or require task metadata in a remote build/test command. Pass the needed code paths and command arguments explicitly.

## Standard backend suites

Use the integration checkout's current executable scripts rather than assembling standard CMake/CTest commands:

| Script | Remote profile | Tests |
| --- | --- | --- |
| `.omp/csw/bin/test_cpu` | `cpu` | All CPU-only CTest entries: general/reference unit tests, CPU tests, CPU conformance, and the registered CPU benchmark. |
| `.omp/csw/bin/test_cuda` | `cuda` | CUDA unit/smoke/conformance, including nonmatrix SDPA, plus backend coexistence. |
| `.omp/csw/bin/test_rocm` | `rocm` | ROCm unit/smoke/conformance, including nonmatrix SDPA, plus backend coexistence. |
| `.omp/csw/bin/test_sycl` | `sycl` | SYCL unit/smoke/conformance plus backend coexistence, after Level Zero GPU enumeration. |

Each script accepts one unique mirror prefix and requires the same local task/workspace environment as the remote helpers:

```bash
CSW_REMOTE_TASK_DIR="/repo/.cswd/tasks/change/leaf" \
CSW_REMOTE_WORKSPACE="/repo/.work/change/leaf" \
  /repo/.omp/csw/bin/test_cpu leaf-attempt-7
```

The runner appends `-cpu`, `-cuda`, `-rocm`, or `-sycl` to the prefix, preventing mirror collisions even when profiles share an SSH host. It configures a Release build with only the requested accelerator enabled, builds, and runs the complete standard selection with CTest `--no-tests=error --output-on-failure --timeout 300`. General and CPU tests are not rerun on accelerator profiles. Artifact-dependent real-model tests remain opt-in; any specification requiring them needs an additional explicit gate.

The scripts own sync-before-exec, remote-side build/test deadlines, logs and SYCL's outside-checkout no-setup profile override/nounset-safe initialization. Every accelerator test phase automatically acquires the **same host-wide** `/tmp/agent-gpu0.lock` with bounded acquisition. This conservatively serializes accelerator tests across profiles/mirrors sharing a host, while builds remain concurrent. CPU tests take no device lock. Additional focused accelerator commands must cooperate with this lock.

Configure all four profiles in `.remote-hosts.conf`; the example includes a dedicated CPU alias. Existing installations may map CPU and CUDA to the same SSH host; CPU remains a separate profile and mirror.

## Typical workflow

Assume the selected workspace is the local task/worktree root.

1. Pick a configured host and a unique task id. Set `CSW_REMOTE_TASK_DIR` to the local task directory that owns the resulting evidence and `CSW_REMOTE_WORKSPACE` to the exact workspace.
2. Sync the local workspace:

```bash
.omp/csw/bin/csw-remote-sync rocm task-123
```

3. Build or test remotely with a remote-side deadline:

```bash
.omp/csw/bin/csw-remote-exec rocm task-123 \
  'timeout --kill-after=30s 1800s cmake --build build -j'
.omp/csw/bin/csw-remote-sync rocm task-123
.omp/csw/bin/csw-remote-exec rocm task-123 \
  'flock -w 120 /tmp/agent-gpu0.lock timeout --kill-after=30s 900s ctest --test-dir build --output-on-failure --timeout 300 --no-tests=error'
```

4. Continue editing locally and repeat sync + remote execution as needed.
5. Remove the remote mirror when done:

```bash
.omp/csw/bin/csw-remote-clean rocm task-123
```

Every helper failure prints the failed local operation, exit status, and local `remote.log` path to stderr. Preserve that line and the log; do not report only “remote failed.”

### `csw_verify` backend verification

For `csw-run` leaf verification, use one command from the assigned worktree:

```text
/repo/.omp/csw/bin/csw_verify --repo /repo verify change/leaf \
  --backend-tests leaf-attempt-7 --summary "Verified required behavior"
```

The helper supplies the prepared `CSW_REMOTE_TASK_DIR` and `CSW_REMOTE_WORKSPACE`, runs all four integration-checkout backend scripts **concurrently**, and waits for every runner before returning. Each backend has its own gate log and result; remote helper output also remains in the task's `remote.log`. All four must pass. The total attempt is bounded to 7200 seconds; remote builds and tests retain their own deadlines.

Do not write a manual standard-backend gate plan. Only specification-required scenarios absent from the scripts need an additional `--plan <outside-checkout-file>`; those gates run sequentially after all four scripts pass. A plan-only invocation remains available for workflow/helper checks that do not require backend suites, but general unit tests still execute on the CPU remote host.

Additional remote gates use absolute integration-checkout `csw-remote-sync`/`csw-remote-exec` paths and explicit prepared task/worktree environment. Every exec needs its own preceding sync for the same profile/mirror, remote-side `timeout --kill-after`, and bounded `/tmp/agent-gpu0.lock` acquisition for accelerator tests. `csw_verify` rejects mismatched paths or missing bounds. Supplemental SYCL commands require the no-setup profile override described below; the standard `test_sycl` script handles it automatically. Failure evidence includes every failed backend and a primary developer-presentable `problem` with operation/profile/mirror, exit/timeout, diagnostics, gate log, and `remote.log`.

## Parallel work

For concurrent tasks:

- use separate local Git worktrees or isolated agent workspaces;
- use a unique `task-id` for every task;
- never share the same remote task directory between agents;
- multiple tasks may target the same host if that host has enough CPU/GPU capacity;
- accelerator tests must acquire the same host-wide lock used by the standard scripts; bound both acquisition and execution, but do not lock CPU tests:

```bash
.omp/csw/bin/csw-remote-exec rocm task-123 \
  'flock -w 120 /tmp/agent-gpu0.lock timeout --kill-after=30s 900s ./build/tests/gpu_test'
```

## Long-running jobs

Use `tmux` or the site's scheduler on the remote host when a job should survive the SSH session, for example:

```bash
.omp/csw/bin/csw-remote-exec cuda task-9 \
  "tmux new-session -d -s agent-task-9 'cmake --build build -j && ctest --test-dir build'"
```

## Safety and consistency

- `csw-remote-sync` uses `rsync --delete-delay`; files removed locally are removed from the remote mirror.
- Only tracked files are synchronized. Untracked and ignored paths are excluded using Git's ignore rules, including nested `.gitignore` files. Stage a new project file before remote validation so it becomes part of the synchronized workspace.
- `_local/` is the sole non-versioned exception. Its contents are synchronized so agents can create and remotely run temporary helper programs and scripts, except that `.cswd` remains excluded even beneath `_local/`.
- `.git`, `.work`, `.worktrees`, and `.cswd` are always excluded. `.cswd` is excluded whether it is a directory or symbolic link, even if tracked; it is never copied or followed. The `.work/` container is therefore never copied when the primary checkout is selected, while selecting one of its worktrees still synchronizes that worktree's root.
- The remote path is derived from the configured base directory plus the task id. The base directory is relative to the SSH user's home directory.
- Task ids must contain only letters, digits, `.`, `_`, and `-`.
- Host aliases must exist in the config.
- SSH key authentication should already be configured in `~/.ssh/config` or via the normal OpenSSH key mechanisms.
- Prefer SSH host aliases so `ProxyJump`, usernames, ports, and keys remain in standard OpenSSH configuration.
- Remote setup commands should activate toolchains only; avoid destructive shell state changes.
- SYCL `remote-exec` profiles used by this repository must omit `REMOTE_SETUP`: the helper starts with nounset enabled, while oneAPI setup reads unset variables. `test_sycl` handles this automatically. For supplemental commands, create an outside-checkout config override containing the same host/base with an empty setup field, set `CSW_REMOTE_CONFIG` to it in every SYCL gate's `env`, then begin each remote command with `set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u;` before the bounded build/test command.

## Files

- `.omp/csw/bin/csw-remote-sync` — one-way rsync from local workspace to remote task directory.
- `.omp/csw/bin/csw-remote-exec` — execute a command inside the remote task directory.
- `.omp/csw/bin/csw-remote-clean` — delete the remote task directory.
- `.omp/csw/bin/test_{cpu,cuda,rocm,sycl}` — complete standard backend suites through the remote helpers; GPU test locking is automatic.
- `references/hosts.example.conf` — simple host profile format.

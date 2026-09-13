---
name: remote-development
description: Perform implementation work in a local Git workspace while building, testing, and running exclusively on a selected remote Linux host over SSH.
---

# Remote Development

Use this skill when implementation must be validated on a specific remote Linux machine, for example CUDA, ROCm, or SYCL hardware.

## Core rule

The selected local Git workspace is authoritative.

- Edit files locally in the selected checkout or worktree.
- Before every remote build, test, benchmark, or execution, sync that exact workspace to a unique remote task directory.
- Run builds/tests/programs only through SSH on the selected remote host.
- Do not edit the shared remote checkout directly unless explicitly required.
- Do not sync `.git`, `.work`, or `.worktrees` to the remote mirror.
- Keep one remote directory per concurrent task.

## Host configuration

Copy `references/hosts.example.conf` to a convenient project-local file such as:

```text
.remote-hosts.conf
```

Then edit host aliases, remote base directories, and optional setup commands.

The helper scripts read the config path from `REMOTE_DEV_CONFIG`. If unset, they first use `.remote-hosts.conf` in the selected workspace. A linked worktree without its own config falls back to `.remote-hosts.conf` in the primary checkout; the config is read from there, never synchronized.

## Workspace selection

The scripts resolve the selected workspace with Git:

- By default, they use the worktree containing the current directory, even when the command is invoked from a subdirectory.
- Set `REMOTE_DEV_WORKSPACE` to a path inside the intended checkout or worktree when invoking a script from elsewhere.
- They compare the worktree Git directory with the common Git directory and report `checkout` or `worktree` in `remote-sync` output.

For a `spec-run-task` task, run from its exact `.work/<task-path>` worktree:

```bash
cd .work/<task-path>
.omp/skills/remote-development/scripts/remote-sync rocm task-123
```

Alternatively, select it explicitly from the primary checkout:

```bash
REMOTE_DEV_WORKSPACE=.work/<task-path> \
  .omp/skills/remote-development/scripts/remote-sync rocm task-123
```

Never sync a `spec-run-task` task from the primary checkout. Confirm that the `remote-sync` output names the exact assigned worktree before remote execution.

## Typical workflow

Assume the selected workspace is the local task/worktree root.

1. Pick a configured host and a unique task id.
2. Sync the local workspace:

```bash
.omp/skills/remote-development/scripts/remote-sync rocm task-123
```

3. Build or test remotely:

```bash
.omp/skills/remote-development/scripts/remote-exec rocm task-123 'cmake -S . -B build && cmake --build build -j'
.omp/skills/remote-development/scripts/remote-exec rocm task-123 'ctest --test-dir build --output-on-failure'
```

4. Continue editing locally and repeat sync + remote execution as needed.
5. Commit or merge changes locally using normal Git tooling.
6. Remove the remote mirror when done:

```bash
.omp/skills/remote-development/scripts/remote-clean rocm task-123
```

## Parallel work

For concurrent tasks:

- use separate local Git worktrees or isolated agent workspaces;
- use a unique `task-id` for every task;
- never share the same remote task directory between agents;
- multiple tasks may target the same host if that host has enough CPU/GPU capacity;
- if tests require exclusive GPU access, wrap the remote command with `flock`, for example:

```bash
.omp/skills/remote-development/scripts/remote-exec rocm task-123 \
  'flock /tmp/agent-gpu0.lock ./build/tests/gpu_test'
```

## Long-running jobs

Use `tmux` or the site's scheduler on the remote host when a job should survive the SSH session, for example:

```bash
.omp/skills/remote-development/scripts/remote-exec cuda task-9 \
  "tmux new-session -d -s agent-task-9 'cmake --build build -j && ctest --test-dir build'"
```

## Safety and consistency

- `remote-sync` uses `rsync --delete-delay`; files removed locally are removed from the remote mirror.
- Only tracked files are synchronized. Untracked and ignored paths are excluded using Git's ignore rules, including nested `.gitignore` files. Stage a new project file before remote validation so it becomes part of the synchronized workspace.
- `_local/` is the sole non-versioned exception. Its contents are synchronized so agents can create and remotely run temporary helper programs and scripts.
- `.git`, `.work`, and `.worktrees` are always excluded. The `.work/` container is therefore never copied when the primary checkout is selected, while selecting one of its worktrees still synchronizes that worktree's root.
- The remote path is derived from the configured base directory plus the task id. The base directory is relative to the SSH user's home directory.
- Task ids must contain only letters, digits, `.`, `_`, and `-`.
- Host aliases must exist in the config.
- SSH key authentication should already be configured in `~/.ssh/config` or via the normal OpenSSH key mechanisms.
- Prefer SSH host aliases so `ProxyJump`, usernames, ports, and keys remain in standard OpenSSH configuration.
- Remote setup commands should activate toolchains only; avoid destructive shell state changes.

## Files

- `scripts/remote-sync` — one-way rsync from local workspace to remote task directory.
- `scripts/remote-exec` — execute a command inside the remote task directory.
- `scripts/remote-clean` — delete the remote task directory.
- `references/hosts.example.conf` — simple host profile format.

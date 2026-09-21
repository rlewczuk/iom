#!/usr/bin/env python3
"""Behavioral checks for backend runner failure handling."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest




BIN = Path(__file__).parents[1] / "bin"


class BackendRunnerTests(unittest.TestCase):
    def test_cancellation_reaches_local_remote_helper(self) -> None:
        with tempfile.TemporaryDirectory(prefix="backend-cancel-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            config = root / "hosts.conf"
            config.write_text(f"cpu|test-host|{root}|\n", encoding="utf-8")
            runner_bin = root / "bin"
            runner_bin.mkdir()
            for name in ("csw-backend-runner", "csw-remote-common.sh"):
                shutil.copy2(BIN / name, runner_bin / name)
            (runner_bin / "csw-remote-sync").write_text(
                '#!/bin/sh\nprintf "%s" "$$" >"$CSW_REMOTE_TASK_DIR/helper.pid"\nexec sleep 60\n',
                encoding="utf-8",
            )
            for name in ("csw-remote-exec", "csw-remote-clean"):
                (runner_bin / name).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            for helper in runner_bin.iterdir():
                helper.chmod(0o755)
            environment = os.environ | {
                "CSW_REMOTE_CONFIG": str(config),
                "CSW_REMOTE_TASK_DIR": str(root),
                "CSW_REMOTE_WORKSPACE": str(workspace),
            }
            process = subprocess.Popen(
                [str(runner_bin / "csw-backend-runner"), "cpu", "cancel"],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            marker = root / "helper.pid"
            helper_pid = None
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() or not marker.read_text():
                    self.assertIsNone(process.poll())
                    self.assertLess(time.monotonic(), deadline, "helper never started")
                    time.sleep(0.01)
                helper_pid = int(marker.read_text())
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=5)
                status = Path(f"/proc/{helper_pid}/stat")
                if status.exists():
                    self.assertEqual(status.read_text().split(") ", 1)[1][0], "Z")
            finally:
                # The pre-fix timeout created its own group. Clean up that
                # owned fixture too, so a failed regression leaves no sleeper.
                for pid in (helper_pid, process.pid):
                    if pid is not None:
                        try:
                            group = os.getpgid(pid)
                            if group != os.getpgrp():
                                os.killpg(group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                process.communicate(timeout=5)

    def test_remote_test_failure_propagates_and_retains_mirror(self) -> None:
        with tempfile.TemporaryDirectory(prefix="backend-runner-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            config = root / "hosts.conf"
            mirror = root / "remote-mirror"
            config.write_text(f"cpu|test-host|{root}|\n", encoding="utf-8")
            task = root / "task"
            task.mkdir()

            runner_bin = root / "bin"
            runner_bin.mkdir()
            shutil.copy2(BIN / "csw-backend-runner", runner_bin / "csw-backend-runner")
            shutil.copy2(BIN / "csw-remote-common.sh", runner_bin / "csw-remote-common.sh")
            (runner_bin / "csw-remote-sync").write_text(
                '#!/bin/sh\nmkdir -p "$IOM_RUNNER_MIRROR"\n',
                encoding="utf-8",
            )
            (runner_bin / "csw-remote-exec").write_text(
                "#!/bin/sh\n"
                "case \"$3\" in *ctest*) exit 23;; esac\n",
                encoding="utf-8",
            )
            (runner_bin / "csw-remote-clean").write_text(
                '#!/bin/sh\nrmdir "$IOM_RUNNER_MIRROR"\n',
                encoding="utf-8",
            )
            for helper in runner_bin.iterdir():
                helper.chmod(0o755)

            environment = os.environ | {
                "CSW_REMOTE_CONFIG": str(config),
                "CSW_REMOTE_TASK_DIR": str(task),
                "CSW_REMOTE_WORKSPACE": str(workspace),
                "IOM_RUNNER_MIRROR": str(mirror),
            }
            result = subprocess.run(
                [str(runner_bin / "csw-backend-runner"), "cpu", "failure-case"],
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertTrue(mirror.is_dir(), "failed remote mirror must remain available")




    def test_accelerator_lock_serializes_and_cpu_bypasses_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="backend-lock-") as temporary:
            root = Path(temporary)
            fake_commands = root / "commands"
            fake_commands.mkdir()
            (fake_commands / "cmake").write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = -S ]; then\n"
                "  while [ $# -gt 0 ]; do\n"
                "    if [ \"$1\" = -B ]; then mkdir -p \"$2\"; fi\n"
                "    shift\n"
                "  done\n"
                "fi\n",
                encoding="utf-8",
            )
            (fake_commands / "ctest").write_text(
                "#!/usr/bin/env python3\n"
                "import fcntl, os, pathlib, time\n"
                "with open(os.environ['IOM_RUNNER_LOCK'], 'a') as lock:\n"
                "    try:\n"
                "        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except BlockingIOError:\n"
                "        pass\n"
                "    else:\n"
                "        if os.environ['IOM_RUNNER_BACKEND'] != 'cpu':\n"
                "            raise SystemExit('accelerator test ran without the device lock')\n"
                "marker = pathlib.Path(os.environ['IOM_RUNNER_MARKER'])\n"
                "marker.mkdir()\n"
                "time.sleep(0.1)\n"
                "marker.rmdir()\n",
                encoding="utf-8",
            )
            for command in fake_commands.iterdir():
                command.chmod(0o755)

            config = root / "hosts.conf"
            config.write_text(
                "cpu|test-host|/tmp/iom-runner-test|\n"
                "rocm|test-host|/tmp/iom-runner-test|\n",
                encoding="utf-8",
            )
            marker = root / "ctest-in-use"
            runner_bins: list[Path] = []
            workspaces: list[Path] = []
            tasks: list[Path] = []
            for name in ("one", "two", "cpu"):
                workspace = root / f"workspace-{name}"
                workspace.mkdir()
                subprocess.run(["git", "init", "-q", str(workspace)], check=True)
                runner_bin = root / f"bin-{name}"
                runner_bin.mkdir()
                shutil.copy2(BIN / "csw-backend-runner", runner_bin / "csw-backend-runner")
                shutil.copy2(BIN / "csw-remote-common.sh", runner_bin / "csw-remote-common.sh")
                (runner_bin / "csw-remote-sync").write_text("#!/bin/sh\n", encoding="utf-8")
                (runner_bin / "csw-remote-clean").write_text("#!/bin/sh\n", encoding="utf-8")
                (runner_bin / "csw-remote-exec").write_text(
                    "#!/usr/bin/env python3\n"
                    "import os, shlex, sys\n"
                    "command = sys.argv[3].replace('/tmp/agent-gpu0.lock', shlex.quote(os.environ['IOM_RUNNER_LOCK']))\n"
                    "os.execl('/bin/bash', 'bash', '-c', command)\n",
                    encoding="utf-8",
                )
                for helper in runner_bin.iterdir():
                    helper.chmod(0o755)
                task = root / f"task-{name}"
                task.mkdir()
                runner_bins.append(runner_bin)
                workspaces.append(workspace)
                tasks.append(task)

            environment_base = os.environ | {
                "PATH": f"{fake_commands}{os.pathsep}{os.environ['PATH']}",
                "CSW_REMOTE_CONFIG": str(config),
                "IOM_RUNNER_MARKER": str(marker),
                "IOM_RUNNER_LOCK": str(root / "gpu.lock"),
                "IOM_RUNNER_BACKEND": "rocm",
            }
            accelerator_processes = []
            for index in (0, 1):
                accelerator_environment = environment_base | {
                    "CSW_REMOTE_TASK_DIR": str(tasks[index]),
                    "CSW_REMOTE_WORKSPACE": str(workspaces[index]),
                }
                accelerator_processes.append(
                    subprocess.Popen(
                        [str(runner_bins[index] / "csw-backend-runner"), "rocm", f"mirror-{index}"],
                        cwd=workspaces[index],
                        env=accelerator_environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            accelerator_results = [process.communicate(timeout=30) for process in accelerator_processes]
            self.assertEqual([process.returncode for process in accelerator_processes], [0, 0], accelerator_results)

            with (root / "gpu.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                cpu_environment = environment_base | {
                    "CSW_REMOTE_TASK_DIR": str(tasks[2]),
                    "CSW_REMOTE_WORKSPACE": str(workspaces[2]),
                    "IOM_RUNNER_BACKEND": "cpu",
                }
                cpu = subprocess.Popen(
                    [str(runner_bins[2] / "csw-backend-runner"), "cpu", "mirror-cpu"],
                    cwd=workspaces[2], env=cpu_environment, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                )
                try:
                    stdout, stderr = cpu.communicate(timeout=5)
                finally:
                    if cpu.poll() is None:
                        os.killpg(cpu.pid, signal.SIGKILL)
                        cpu.communicate(timeout=5)
            self.assertEqual(cpu.returncode, 0, stdout + stderr)


if __name__ == "__main__":
    unittest.main()
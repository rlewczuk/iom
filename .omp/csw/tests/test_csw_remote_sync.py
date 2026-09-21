#!/usr/bin/env python3
"""Behavioral tests for remote workspace synchronization filters."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import unittest


REMOTE_CLEAN = Path(__file__).parents[1] / "bin" / "csw-remote-clean"
REMOTE_EXEC = Path(__file__).parents[1] / "bin" / "csw-remote-exec"
REMOTE_SYNC = Path(__file__).parents[1] / "bin" / "csw-remote-sync"


class RemoteSyncTests(unittest.TestCase):
    def test_remote_exec_preserves_quoted_commands_and_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-exec-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            ssh = fake_bin / "ssh"
            ssh.write_text('#!/bin/sh\nexec /bin/sh -c "$2"\n', encoding="utf-8")
            ssh.chmod(0o755)
            remote_base = root / "remote host's directory"
            destination = remote_base / "quoted"
            destination.mkdir(parents=True)
            config = root / "hosts.conf"
            config.write_text(f"cpu|test-host|{remote_base}|\n", encoding="utf-8")
            environment = os.environ | {
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "CSW_REMOTE_CONFIG": str(config),
                "CSW_REMOTE_WORKSPACE": str(workspace),
                "CSW_REMOTE_TASK_DIR": str(root),
            }
            arguments = ["argument with spaces", "apostrophe's value", "$(touch injected)"]
            command = shlex.join(["printf", "%s\\n", *arguments])
            result = subprocess.run(
                [str(REMOTE_EXEC), "cpu", "quoted", command],
                cwd=workspace, env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "\n".join(arguments) + "\n")
            self.assertFalse((destination / "injected").exists())

    def test_cswd_directories_and_links_are_excluded_but_sources_sync(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remote-sync-") as temporary:
            temporary_root = Path(temporary)
            fake_bin = temporary_root / "bin"
            fake_bin.mkdir()
            (fake_bin / "ssh").write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "shift\n"
                "if [ $# -eq 1 ]; then\n"
                "  exec /bin/sh -c \"$1\"\n"
                "fi\n"
                "exec \"$@\"\n",
                encoding="utf-8",
            )
            (fake_bin / "ssh").chmod(0o755)
            remote_base = temporary_root / "remote"

            for mode in ("directory", "link"):
                with self.subTest(root_mode=mode):
                    workspace = temporary_root / mode
                    workspace.mkdir()
                    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
                    subprocess.run(
                        ["git", "-C", str(workspace), "config", "user.email", "remote-sync@example.invalid"],
                        check=True,
                    )
                    subprocess.run(
                        ["git", "-C", str(workspace), "config", "user.name", "Remote Sync"],
                        check=True,
                    )
                    (workspace / "src").mkdir()
                    (workspace / "src" / "input.txt").write_text("source\n", encoding="utf-8")
                    (workspace / "_local").mkdir()
                    (workspace / "_local" / "helper.sh").write_text("#!/bin/sh\n", encoding="utf-8")
                    (workspace / "_local" / ".cswd").symlink_to(
                        workspace / ".cswd", target_is_directory=True
                    )
                    metadata = temporary_root / f"{mode}-metadata"
                    (metadata / "tasks").mkdir(parents=True)
                    (metadata / "tasks" / "secret.yml").write_text("private\n", encoding="utf-8")
                    task_directory = metadata / "tasks" / f"filter-{mode}"
                    task_directory.mkdir()
                    if mode == "directory":
                        (workspace / ".cswd").mkdir()
                        (workspace / ".cswd" / "tasks").mkdir()
                        (workspace / ".cswd" / "tasks" / "tracked.yml").write_text(
                            "tracked metadata\n", encoding="utf-8"
                        )
                        (workspace / "nested").mkdir()
                        (workspace / "nested" / ".cswd").symlink_to(
                            metadata, target_is_directory=True
                        )
                        subprocess.run(
                            ["git", "-C", str(workspace), "add", "src", ".cswd", "nested"], check=True
                        )
                    else:
                        (workspace / ".cswd").symlink_to(metadata, target_is_directory=True)
                        (workspace / "nested").mkdir()
                        (workspace / "nested" / ".cswd").mkdir()
                        (workspace / "nested" / ".cswd" / "secret").write_text(
                            "private nested\n", encoding="utf-8"
                        )
                        subprocess.run(
                            ["git", "-C", str(workspace), "add", "src", ".cswd", "nested"], check=True
                        )
                    subprocess.run(
                        ["git", "-C", str(workspace), "commit", "-qm", "tracked source"], check=True
                    )
                    config = workspace / ".remote-hosts.conf"
                    config.write_text(f"local|test-host|{remote_base}|\n", encoding="utf-8")
                    environment = os.environ.copy()
                    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
                    environment["CSW_REMOTE_CONFIG"] = str(config)
                    environment["CSW_REMOTE_WORKSPACE"] = str(workspace)
                    environment["RSYNC_RSH"] = str(fake_bin / "ssh")
                    environment["CSW_REMOTE_TASK_DIR"] = str(task_directory)
                    result = subprocess.run(
                        [str(REMOTE_SYNC), "local", f"filter-{mode}"],
                        cwd=workspace,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    destination = remote_base / f"filter-{mode}"
                    for relative in (".cswd", "nested/.cswd", "_local/.cswd"):
                        self.assertFalse(os.path.lexists(destination / relative), relative)
                    self.assertEqual(
                        (destination / "src" / "input.txt").read_text(encoding="utf-8"),
                        "source\n",
                    )
                    self.assertEqual(
                        (destination / "_local" / "helper.sh").read_text(encoding="utf-8"),
                        "#!/bin/sh\n",
                    )
                    command = subprocess.run(
                        [str(REMOTE_EXEC), "local", f"filter-{mode}",
                         "printf remote-stdout; printf remote-stderr >&2"],
                        cwd=workspace,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
                    self.assertEqual(command.returncode, 0, command.stderr)
                    self.assertEqual(command.stdout, "remote-stdout")
                    self.assertEqual(command.stderr, "remote-stderr")
                    failure = subprocess.run(
                        [str(REMOTE_EXEC), "local", f"filter-{mode}",
                         "printf remote-failure >&2; exit 23"],
                        cwd=workspace,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
                    self.assertEqual(failure.returncode, 23)
                    self.assertIn("remote-failure", failure.stderr)
                    self.assertIn("ssh failed with exit 23; full output:", failure.stderr)
                    self.assertIn(str(task_directory / "remote.log"), failure.stderr)
                    clean = subprocess.run(
                        [str(REMOTE_CLEAN), "local", f"filter-{mode}"],
                        cwd=workspace,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=30,
                    )
                    self.assertEqual(clean.returncode, 0, clean.stderr)
                    remote_log = (task_directory / "remote.log").read_text(encoding="utf-8")
                    self.assertIn(f"argv: local filter-{mode}", remote_log)
                    self.assertIn("csw-remote-sync", remote_log)
                    self.assertIn("csw-remote-exec", remote_log)
                    self.assertIn("csw-remote-clean", remote_log)
                    self.assertIn("remote-stdout", remote_log)
                    self.assertIn("remote-stderr", remote_log)
                    self.assertIn("remote-failure", remote_log)
                    self.assertIn("exit: 23", remote_log)
                    self.assertIn("ssh stderr:", remote_log)


if __name__ == "__main__":
    unittest.main()

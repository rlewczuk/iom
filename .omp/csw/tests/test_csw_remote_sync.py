#!/usr/bin/env python3
"""Behavioral tests for remote workspace synchronization filters."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REMOTE_SYNC = Path(__file__).parents[1] / "bin" / "csw-remote-sync"


class RemoteSyncTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

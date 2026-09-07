#!/usr/bin/env python3
"""Behavioral checks for the spec-run-all CLI; all mutations use temporary repos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).with_name("spec_run_all.py")
TASK_HELPER = RUNNER.parents[2] / "spec-run-task" / "scripts" / "spec_run_task.py"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spec-run-all-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.target = self.repo / "docs" / "changes" / "example"
        self.target.mkdir(parents=True)
        self.state = self.root / "state.json"
        self.env = os.environ | {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Runner Test",
            "GIT_AUTHOR_EMAIL": "runner@example.invalid",
            "GIT_COMMITTER_NAME": "Runner Test",
            "GIT_COMMITTER_EMAIL": "runner@example.invalid",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.git("init", "-b", "main")
        (self.repo / ".gitignore").write_text(".work/\n", encoding="utf-8")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo), *args], env=self.env,
            text=True, capture_output=True, check=True,
        ).stdout.strip()

    def task(self, name, blocked="None", status=None):
        directory = self.target / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spec.md").write_text(
            f"# {name}\n\n**Blocked by:** {blocked}\n", encoding="utf-8",
        )
        if status is not None:
            (directory / "task.md").write_text(
                f"**Status:** {status}\n", encoding="utf-8",
            )
        return directory

    def run_cli(self, command, *, target="example", state=None, success=True):
        args = [sys.executable, str(RUNNER), "--repo", str(self.repo), command, target]
        if state is not None:
            self.state.write_text(json.dumps(state), encoding="utf-8")
            args.extend(["--state", str(self.state)])
        result = subprocess.run(args, env=self.env, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def helper(self, command, task, *args):
        result = subprocess.run(
            [sys.executable, str(TASK_HELPER), "--repo", str(self.repo),
             command, task, *args], env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_discovery_omits_only_done_and_caps_alphabetical_wave(self):
        self.task("00-done", status="done")
        self.task("01-ready", status="ready")
        self.task("02-failed", status="failed")
        self.task("03-blocked", status="blocked")
        for number in reversed(range(4, 10)):
            self.task(f"{number:02d}-new")
        (self.target / "notes").mkdir()
        scan = self.run_cli("scan")
        names = [task["name"] for task in scan["tasks"]]
        self.assertEqual(names, sorted(p.name for p in self.target.iterdir() if (p / "spec.md").is_file()))
        queue = self.run_cli("queue")
        self.assertEqual(queue["already_done"], ["00-done"])
        self.assertEqual(queue["ready"], names[1:6])
        self.assertFalse(queue["finished"])

    def test_dependencies_override_alphabet_and_only_done_unlocks(self):
        self.task("01-consumer", "`03-provider`, `02-provider`")
        self.task("02-provider", status="done")
        self.task("03-provider", status="ready")
        self.assertEqual(self.run_cli("queue")["ready"], ["03-provider"])
        self.task("03-provider", status="done")
        self.assertEqual(self.run_cli("queue")["ready"], ["01-consumer"])

    def test_cycles_and_missing_dependencies_do_not_stop_independent_tasks(self):
        self.task("01-cycle", "02-cycle")
        self.task("02-cycle", "01-cycle")
        self.task("03-missing", "does-not-exist")
        self.task("04-independent")
        queue = self.run_cli("queue")
        self.assertEqual(queue["ready"], ["04-independent"])
        diagnostics = {item["name"]: item["reason"] for item in queue["blocked"] + queue["waiting"]}
        self.assertEqual(set(diagnostics), {"01-cycle", "02-cycle", "03-missing"})
        self.assertIn("does-not-exist", diagnostics["03-missing"])

    def test_completed_task_breaks_historical_dependency_cycle(self):
        self.task("01-consumer", "02-provider")
        self.task("02-provider", "01-consumer", status="done")
        self.assertEqual(self.run_cli("queue")["ready"], ["01-consumer"])

    def test_run_outcomes_suppress_retry_but_do_not_satisfy_dependencies(self):
        self.task("01-provider")
        self.task("02-consumer", "01-provider")
        self.task("03-independent")
        state = {"outcomes": {"01-provider": {"status": "blocked", "reason": "GPU host unavailable"}}}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], ["03-independent"])
        self.assertIn("GPU host unavailable", json.dumps(queue["blocked"]))
        self.task("01-provider", status="done")
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["02-consumer", "03-independent"])

    def test_explicit_dependency_resolution_and_external_completion(self):
        self.task("01-consumer", "After the provider's API is available")
        self.task("02-provider", status="done")
        self.assertEqual(self.run_cli("queue")["ready"], [])
        state = {"dependencies": {"01-consumer": ["02-provider"]}}
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["01-consumer"])
        self.task("03-external", "docs/changes/other/provider/spec.md")
        external = self.repo / "docs" / "changes" / "other" / "provider"
        external.mkdir(parents=True)
        (external / "spec.md").write_text("# Provider\n**Blocked by:** None\n", encoding="utf-8")
        (external / "task.md").write_text("**Status:** done\n", encoding="utf-8")
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["01-consumer", "03-external"])

    def test_invalid_metadata_and_paths_fail_closed(self):
        malformed = self.task("01-invalid")
        (malformed / "task.md").write_text("**Status:** done\n**Status:** ready\n", encoding="utf-8")
        self.task("02-independent")
        queue = self.run_cli("queue")
        self.assertEqual(queue["ready"], ["02-independent"])
        self.assertIn("01-invalid", [item["name"] for item in queue["blocked"]])
        self.run_cli("scan", target="../example", success=False)
        self.run_cli("queue", state={"outcomes": {"unknown": {"status": "blocked", "reason": "missing"}}}, success=False)
        invalid_outcome = {"outcomes": {"02-independent": {"status": [], "reason": "invalid"}}}
        self.assertEqual(self.run_cli("queue", state=invalid_outcome)["ready"], [])

    def test_empty_and_completed_targets_are_terminal(self):
        self.assertTrue(self.run_cli("queue")["finished"])
        self.task("01-done", status="done")
        queue = self.run_cli("queue")
        self.assertTrue(queue["finished"])
        self.assertEqual(queue["ready"], [])

    def test_prepare_reuse_and_integration_unlock_canonical_dependency(self):
        self.task("01-consumer", "02-provider")
        self.task("02-provider")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        prepared = self.run_cli("prepare")["prepared"]
        self.assertEqual([entry["name"] for entry in prepared], ["02-provider"])
        provider = prepared[0]
        worktree = Path(provider["worktree"])
        self.assertNotEqual(worktree, self.repo)
        self.assertTrue(Path(provider["spec_path"]).is_file())
        self.assertEqual(self.run_cli("prepare")["prepared"][0]["worktree"], str(worktree))
        task_path = provider["task_path"]
        self.helper("annotate", task_path, "--status", "ready", "--summary", "Prepared fixture")
        self.helper("commit", task_path, "--status", "ready", "--outcome", "complete fixture")
        state = {"outcomes": {"02-provider": {"status": "ready", "reason": "Awaiting integration"}}}
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        self.helper("annotate", task_path, "--status", "done", "--summary", "Fixture paths verified",
                    "--verification", "Prepared spec path exists and worktree reused")
        self.helper("commit", task_path, "--status", "done")
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        self.helper("integrate", task_path)
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["01-consumer"])

    def test_preparation_failure_does_not_abort_independent_owner(self):
        self.task("01-collision")
        self.task("02-independent")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        collision = self.repo / ".work" / "example" / "01-collision"
        collision.mkdir(parents=True)
        retained = collision / "owned.txt"
        retained.write_text("preserve this state", encoding="utf-8")
        result = self.run_cli("prepare")
        self.assertEqual([entry["name"] for entry in result["prepared"]], ["02-independent"])
        self.assertIn("01-collision", [entry["name"] for entry in result["blocked"]])
        self.assertEqual(retained.read_text(encoding="utf-8"), "preserve this state")


if __name__ == "__main__":
    unittest.main()

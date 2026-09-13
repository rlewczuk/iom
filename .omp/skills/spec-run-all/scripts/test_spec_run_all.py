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
        args = [sys.executable, str(RUNNER), "--repo", str(self.repo), command]
        if command != "control":
            args.append(target)
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

    def test_discovery_queues_and_prepares_all_unfinished_tasks(self):
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
        self.assertEqual(queue["ready"], names[1:])
        self.assertFalse(queue["finished"])
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        prepared = self.run_cli("prepare")
        self.assertEqual([entry["name"] for entry in prepared["prepared"]], names[1:])

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

    def test_running_outcomes_wait_and_canonical_done_overrides_them(self):
        self.task("01-provider")
        self.task("02-consumer", "01-provider")
        state = {"outcomes": {"01-provider": {"status": "running", "reason": "owner dispatched"}}}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], [])
        self.assertEqual(queue["blocked"], [])
        self.assertFalse(queue["finished"])
        waiting = {item["name"]: item["reason"] for item in queue["waiting"]}
        self.assertIn("01-provider", waiting)
        self.assertIn("02-consumer", waiting)
        state["outcomes"]["01-provider"] = {"status": "ready", "reason": "owner completed"}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], [])
        self.assertEqual(queue["blocked"], [])
        self.assertIn("01-provider", {item["name"] for item in queue["waiting"]})
        self.task("01-provider", status="done")
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["already_done"], ["01-provider"])
        self.assertEqual(queue["ready"], ["02-consumer"])
        self.assertFalse(queue["finished"])

    def test_running_only_work_is_unfinished_and_waiting(self):
        self.task("01-owner")
        state = {"outcomes": {"01-owner": {"status": "running", "reason": "long-running owner"}}}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], [])
        self.assertEqual(queue["blocked"], [])
        self.assertEqual([item["name"] for item in queue["waiting"]], ["01-owner"])
        self.assertFalse(queue["finished"])

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
        invalid_running = {"outcomes": {"02-independent": {"status": "running", "reason": " "}}}
        queue = self.run_cli("queue", state=invalid_running)
        self.assertEqual(queue["ready"], [])
        self.assertEqual({item["name"] for item in queue["blocked"]}, {"01-invalid", "02-independent"})

    def test_empty_and_completed_targets_are_terminal(self):
        self.assertTrue(self.run_cli("queue")["finished"])
        self.task("01-done", status="done")
        queue = self.run_cli("queue")
        self.assertTrue(queue["finished"])
        self.assertEqual(queue["ready"], [])

    def test_prepare_reuse_integration_and_dependency_refill_while_owner_runs(self):
        self.task("01-consumer", "02-provider")
        self.task("02-provider")
        self.task("03-slow")
        self.task("04-second-consumer", "02-provider")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        initial = self.run_cli("prepare")["prepared"]
        self.assertEqual([entry["name"] for entry in initial], ["02-provider", "03-slow"])
        provider = initial[0]
        slow = initial[1]
        provider_worktree = Path(provider["worktree"])
        slow_worktree = Path(slow["worktree"])
        self.assertNotEqual(provider_worktree, self.repo)
        self.assertTrue(Path(provider["spec_path"]).is_file())
        preserved = slow_worktree / "owned.txt"
        preserved.write_text("slow owner work", encoding="utf-8")
        state = {"outcomes": {"03-slow": {"status": "running", "reason": "slow owner active"}}}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], ["02-provider"])
        self.assertEqual(queue["blocked"], [])
        self.assertIn("03-slow", {item["name"] for item in queue["waiting"]})
        prepared = self.run_cli("prepare", state=state)["prepared"]
        self.assertEqual([entry["name"] for entry in prepared], ["02-provider"])
        self.assertEqual(prepared[0]["worktree"], str(provider_worktree))
        self.assertEqual(preserved.read_text(encoding="utf-8"), "slow owner work")
        self.assertTrue(self.helper("show", "example/03-slow")["dirty"])

        state["outcomes"]["02-provider"] = {"status": "running", "reason": "provider dispatched"}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], [])
        self.assertEqual(queue["blocked"], [])
        self.assertIn("02-provider", {item["name"] for item in queue["waiting"]})
        self.assertIn("03-slow", {item["name"] for item in queue["waiting"]})
        self.assertEqual(self.run_cli("prepare", state=state)["prepared"], [])

        task_path = provider["task_path"]
        self.helper("annotate", task_path, "--status", "ready", "--summary", "Prepared fixture")
        self.helper("commit", task_path, "--status", "ready", "--outcome", "complete fixture")
        state["outcomes"]["02-provider"] = {"status": "ready", "reason": "Awaiting integration"}
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        self.helper("annotate", task_path, "--status", "done", "--summary", "Fixture paths verified",
                    "--verification", "Prepared spec path exists and worktree reused")
        self.helper("commit", task_path, "--status", "done")
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        integrated = self.helper("integrate", task_path)
        integration_head = integrated["integration_head"]
        control = self.run_cli("control")
        self.assertEqual(control["integration_head"], integration_head)
        self.assertEqual(preserved.read_text(encoding="utf-8"), "slow owner work")
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["already_done"], ["02-provider"])
        self.assertEqual(queue["ready"], ["01-consumer", "04-second-consumer"])
        self.assertIn("03-slow", {item["name"] for item in queue["waiting"]})
        refilled = self.run_cli("prepare", state=state)
        self.assertEqual(
            [entry["name"] for entry in refilled["prepared"]],
            ["01-consumer", "04-second-consumer"],
        )
        self.assertEqual(refilled["integration_head"], integration_head)
        self.assertTrue(all(entry["base"] == integration_head for entry in refilled["prepared"]))
        for name in ("01-consumer", "04-second-consumer"):
            state["outcomes"][name] = {"status": "running", "reason": "dependent dispatched"}
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        self.assertEqual(self.run_cli("prepare", state=state)["prepared"], [])
        self.assertEqual(preserved.read_text(encoding="utf-8"), "slow owner work")

    def test_control_refreshes_head_without_target_or_root_spec(self):
        self.target.rmdir()
        self.git("add", ".")
        self.git("commit", "-m", "initial")
        tracked = self.repo / "tracked.txt"
        tracked.write_text("advance integration", encoding="utf-8")
        self.git("add", "tracked.txt")
        self.git("commit", "-m", "advance")
        expected_head = self.git("rev-parse", "HEAD")
        control = self.run_cli("control")
        self.assertEqual(control["repo_root"], str(self.repo.resolve()))
        self.assertEqual(control["integration_branch"], "main")
        self.assertEqual(control["integration_head"], expected_head)
        self.assertEqual(self.git("status", "--short"), "")

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

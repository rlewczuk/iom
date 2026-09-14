#!/usr/bin/env python3
"""Behavioral checks for the csw-run CLI; all mutations use temporary repos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "bin" / "csw_run"
TASK_HELPER = RUNNER.with_name("csw_run_worker")
TASK_CTL_PATH = RUNNER.with_name("task_ctl")
TASK_CTL = runpy.run_path(str(TASK_CTL_PATH))
get_task = TASK_CTL["get_task"]
set_task = TASK_CTL["set_task"]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="csw-run-test-")
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

    def task(self, name, blocked=None, status="new", task_type="impl", order=None):
        directory = self.target / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spec.md").write_text(f"# {name}\n", encoding="utf-8")
        dependencies = []
        for dependency in blocked or []:
            task_id = dependency if dependency.startswith((".cswd/tasks/", "docs/changes/")) else (
                f"docs/changes/example/{dependency}"
            )
            dependencies.append({"task-id": task_id})
        updates = {
            "type": task_type,
            "status": status,
            "blocked-by": dependencies,
        }
        if order is not None:
            updates["order"] = order
        set_task(self.repo, f"docs/changes/example/{name}", updates)
        return directory

    def run_cli(self, command, *, target="example", state=None, success=True):
        args = [str(RUNNER), "--repo", str(self.repo), command]
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
            [str(TASK_HELPER), "--repo", str(self.repo),
             command, task, *args], env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_discovery_queues_and_prepares_all_unfinished_tasks(self):
        self.task("00-done", status="done")
        self.task("01-ready", status="ready")
        self.task("02-ready", status="ready")
        self.task("03-ready", status="ready")
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

    def test_task_ctl_numeric_order_is_preserved(self):
        self.task("01-canonical-first", order=2)
        self.task("99-ordered-first", order=1)
        self.task("02-unordered")
        scan = self.run_cli("scan")
        self.assertEqual(
            [task["name"] for task in scan["tasks"]],
            ["99-ordered-first", "01-canonical-first", "02-unordered"],
        )
        self.assertEqual(
            self.run_cli("queue")["ready"],
            ["99-ordered-first", "01-canonical-first", "02-unordered"],
        )

    def test_dependencies_override_task_order_and_only_done_unlocks(self):
        self.task("01-consumer", ["03-provider", "02-provider"])
        self.task("02-provider", status="done")
        self.task("03-provider", status="ready")
        self.assertEqual(self.run_cli("queue")["ready"], ["03-provider"])
        self.task("03-provider", status="done")
        self.assertEqual(self.run_cli("queue")["ready"], ["01-consumer"])

    def test_dependency_cycles_fail_closed(self):
        self.task("01-cycle", ["02-cycle"])
        self.task("02-cycle", ["01-cycle"])
        self.task("03-independent")
        queue = self.run_cli("queue")
        self.assertEqual(queue["ready"], ["03-independent"])
        self.assertEqual({task["name"] for task in queue["waiting"]}, {"01-cycle", "02-cycle"})

    def test_missing_dependency_does_not_stop_independent_tasks(self):
        self.task("01-missing", ["does-not-exist"])
        self.task("02-independent")
        queue = self.run_cli("queue")
        self.assertEqual(queue["ready"], ["02-independent"])
        diagnostics = {item["name"]: item["reason"] for item in queue["waiting"]}
        self.assertEqual(set(diagnostics), {"01-missing"})
        self.assertIn("does-not-exist", diagnostics["01-missing"])

    def test_completed_task_breaks_historical_dependency_cycle(self):
        self.task("01-consumer", ["02-provider"])
        self.task("02-provider", ["01-consumer"], status="done")
        self.assertEqual(self.run_cli("queue")["ready"], ["01-consumer"])

    def test_run_outcomes_suppress_retry_but_do_not_satisfy_dependencies(self):
        self.task("01-provider")
        self.task("02-consumer", ["01-provider"])
        self.task("03-independent")
        state = {"outcomes": {"01-provider": {"status": "blocked", "reason": "GPU host unavailable"}}}
        queue = self.run_cli("queue", state=state)
        self.assertEqual(queue["ready"], ["03-independent"])
        self.assertIn("GPU host unavailable", json.dumps(queue["blocked"]))
        self.task("01-provider", status="done")
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["02-consumer", "03-independent"])

    def test_running_outcomes_wait_and_canonical_done_overrides_them(self):
        self.task("01-provider")
        self.task("02-consumer", ["01-provider"])
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

    def test_semantic_and_external_dependencies_use_canonical_ids(self):
        self.task("01-consumer")
        self.task("02-provider")
        state = {
            "dependencies": {
                "01-consumer": ["docs/changes/example/02-provider"],
            },
        }
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["02-provider"])
        self.task("02-provider", status="done")
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["01-consumer"])
        self.task("03-external", ["docs/changes/other/provider"])
        external = self.repo / "docs" / "changes" / "other" / "provider"
        external.mkdir(parents=True)
        (external / "spec.md").write_text("# Provider\n", encoding="utf-8")
        set_task(
            self.repo,
            "docs/changes/other/provider",
            {"type": "impl", "status": "done"},
        )
        self.assertEqual(self.run_cli("queue", state=state)["ready"], ["01-consumer", "03-external"])

    def test_invalid_metadata_and_paths_fail_closed(self):
        malformed = self.task("01-invalid")
        (malformed / "task.yml").write_text("type: impl\nstatus: [ready]\n", encoding="utf-8")
        self.task("02-independent")
        queue = self.run_cli("queue")
        self.assertEqual(queue["ready"], ["02-independent"])
        self.assertEqual([task["name"] for task in queue["blocked"]], ["01-invalid"])
        set_task(
            self.repo,
            "docs/changes/example/01-invalid",
            {"type": "impl", "status": "ready"},
            replace=True,
        )
        self.run_cli("scan", target="../example", success=False)
        self.run_cli("queue", state={"outcomes": {"unknown": {"status": "blocked", "reason": "missing"}}}, success=False)
        invalid_outcome = {"outcomes": {"02-independent": {"status": [], "reason": "invalid"}}}
        self.assertEqual(self.run_cli("queue", state=invalid_outcome)["ready"], ["01-invalid"])
        invalid_running = {"outcomes": {"02-independent": {"status": "running", "reason": " "}}}
        queue = self.run_cli("queue", state=invalid_running)
        self.assertEqual(queue["ready"], ["01-invalid"])
        self.assertEqual({item["name"] for item in queue["blocked"]}, {"02-independent"})

    def test_missing_control_fails_closed(self):
        missing = self.target / "01-missing"
        missing.mkdir()
        (missing / "spec.md").write_text("# Missing control\n", encoding="utf-8")
        queue = self.run_cli("queue")
        self.assertEqual(queue["ready"], [])
        self.assertEqual([task["name"] for task in queue["blocked"]], ["01-missing"])
        self.assertFalse(queue["finished"])

    def test_failed_new_task_retains_one_commit_and_cannot_integrate(self):
        self.task("01-new", status="new")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        inspection = self.helper("inspect", "example/01-new")
        self.assertTrue(inspection["leaves"][0]["explicitly_ready"])
        base = self.git("rev-parse", "HEAD")
        prepared = self.helper(
            "prepare",
            "example/01-new",
            "--run-target",
            "example/01-new",
            "--integration-branch",
            "main",
            "--integration-base",
            base,
        )
        self.helper(
            "annotate",
            "example/01-new",
            "--outcome",
            "failed",
            "--summary",
            "No implementation was retained",
            "--error",
            "required input — unavailable",
        )
        retained = self.helper(
            "commit",
            "example/01-new",
            "--status",
            "new",
            "--outcome",
            "retain failed attempt",
        )
        self.assertEqual(retained["status"], "new")
        self.assertEqual(len(self.helper("show", "example/01-new")["task_commits"]), 1)
        self.assertEqual(
            get_task(Path(prepared["worktree"]), "docs/changes/example/01-new")["status"],
            "new",
        )
        integration = subprocess.run(
            [
                str(TASK_HELPER),
                "--repo",
                str(self.repo),
                "integrate",
                "example/01-new",
            ],
            env=self.env,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(integration.returncode, 0)
        self.assertEqual(get_task(self.repo, "docs/changes/example/01-new")["status"], "new")

    def test_empty_and_completed_targets_are_terminal(self):
        self.assertTrue(self.run_cli("queue")["finished"])
        self.task("01-done", status="done")
        queue = self.run_cli("queue")
        self.assertTrue(queue["finished"])
        self.assertEqual(queue["ready"], [])

    def test_prepare_reuse_integration_and_dependency_refill_while_owner_runs(self):
        self.task("01-consumer", ["02-provider"])
        self.task("02-provider")
        self.task("03-slow")
        self.task("04-second-consumer", ["02-provider"])
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
        self.helper("annotate", task_path, "--outcome", "ready", "--summary", "Prepared fixture")
        self.helper("commit", task_path, "--status", "ready", "--outcome", "complete fixture")
        state["outcomes"]["02-provider"] = {"status": "ready", "reason": "Awaiting integration"}
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        verified = self.helper(
            "annotate",
            task_path,
            "--outcome",
            "verified",
            "--summary",
            "Fixture paths verified",
            "--verification",
            "Prepared spec path exists and worktree reused",
        )
        verified_commit = self.helper("commit", task_path, "--status", "verified")["commit"]
        self.assertEqual(verified["lifecycle_status"], "verified")
        self.assertEqual(
            get_task(provider_worktree, "docs/changes/example/02-provider")["status"],
            "verified",
        )
        self.assertEqual(
            get_task(self.repo, "docs/changes/example/02-provider")["status"],
            "new",
        )
        self.assertEqual(self.run_cli("queue", state=state)["ready"], [])
        self.helper(
            "annotate", task_path, "--outcome", "failed", "--summary", "Later verification failed",
            "--error", "Combined verification exposed a failure",
        )
        self.helper("commit", task_path, "--status", "verified")
        rejected = subprocess.run(
            [str(TASK_HELPER), "--repo", str(self.repo), "integrate", task_path],
            env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(rejected.returncode, 2, rejected.stdout)
        self.assertEqual(get_task(self.repo, "docs/changes/example/02-provider")["status"], "new")
        self.helper(
            "annotate", task_path, "--outcome", "verified", "--summary", "Verification rerun succeeded",
            "--verification", "Fixture verification rerun passed",
        )
        verified_commit = self.helper("commit", task_path, "--status", "verified")["commit"]
        integrated = self.helper("integrate", task_path)
        self.assertEqual(len(integrated["commits"]), 1)
        self.assertEqual(integrated["commits"][0]["original_commit"], verified_commit)
        self.assertNotEqual(integrated["commits"][0]["commit"], verified_commit)
        integration_head = integrated["integration_head"]
        self.assertEqual(
            len(self.git("rev-list", f"{integrated['previous_head']}..{integration_head}").splitlines()),
            1,
        )
        self.assertEqual(
            get_task(self.repo, "docs/changes/example/02-provider")["status"],
            "done",
        )
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

    def test_stale_container_rollup_needs_no_new_implementation_task(self):
        (self.target / "spec.md").write_text("# Container\n", encoding="utf-8")
        set_task(self.repo, "docs/changes/example", {"type": "hld", "status": "planned"})
        self.task("01-completed", status="done")
        self.git("add", ".")
        self.git("commit", "-m", "completed leaves with pending container rollup")
        base = self.git("rev-parse", "HEAD")
        task = "example/01-completed"
        prepared = self.helper(
            "prepare", task, "--run-target", "example",
            "--integration-branch", "main", "--integration-base", base,
        )
        self.assertTrue(prepared["rollup_only"])
        for annotated in (task, "example"):
            self.helper(
                "annotate", task, "--for-task", annotated, "--outcome", "verified",
                "--summary", "All implementation leaves are already complete",
                "--verification", "Canonical task controls confirm all leaves done",
            )
        self.helper("commit", task, "--status", "verified", "--outcome", "complete container rollup")
        self.helper("integrate", task)
        self.assertEqual(get_task(self.repo, "docs/changes/example")["status"], "done")
        self.assertEqual(get_task(self.repo, "docs/changes/" + task)["status"], "done")
        self.assertEqual(self.git("rev-list", "--count", f"{base}..HEAD"), "1")

    def test_verified_train_and_rollup_finalize_without_completion_commits(self):
        (self.target / "spec.md").write_text("# Example container\n", encoding="utf-8")
        set_task(
            self.repo,
            "docs/changes/example",
            {"type": "hld", "status": "ready"},
        )
        self.task("01-first")
        self.task("02-second")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        base = self.git("rev-parse", "HEAD")

        for task in ("example/01-first", "example/02-second"):
            self.helper(
                "prepare",
                task,
                "--run-target",
                "example",
                "--integration-branch",
                "main",
                "--integration-base",
                base,
            )
            self.helper("annotate", task, "--outcome", "ready", "--summary", f"Implemented {task}")
            self.helper("commit", task, "--status", "ready", "--outcome", f"complete {task}")
            self.helper(
                "annotate",
                task,
                "--outcome",
                "verified",
                "--summary",
                f"Verified {task}",
                "--verification",
                f"{task} scenario — passed",
            )
            self.helper("commit", task, "--status", "verified")

        first_commit = self.helper("show", "example/01-first")["task_commits"][0]["commit"]
        self.helper("rebase", "example/02-second", "--onto", first_commit)
        self.helper(
            "annotate",
            "example/02-second",
            "--for-task",
            "example",
            "--outcome",
            "verified",
            "--summary",
            "All child tasks verified",
            "--verification",
            "Combined scenario — passed",
        )
        self.helper("commit", "example/02-second", "--status", "verified")
        integrated = self.helper("integrate", "example/02-second")

        self.assertEqual(len(integrated["commits"]), 2)
        self.assertEqual(
            len(self.git("rev-list", f"{base}..{integrated['integration_head']}").splitlines()),
            2,
        )
        for task_id in (
            "docs/changes/example",
            "docs/changes/example/01-first",
            "docs/changes/example/02-second",
        ):
            self.assertEqual(get_task(self.repo, task_id)["status"], "done")

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

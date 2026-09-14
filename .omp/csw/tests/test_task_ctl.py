#!/usr/bin/env python3
"""Behavioral tests for the task_ctl control-file owner."""

from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest

import yaml


TASK_CTL = Path(__file__).parents[1] / "bin" / "task_ctl"


class TaskCtlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="task-ctl-")
        self.repo = Path(self.temporary.name) / "repo"
        (self.repo / ".cswd" / "tasks").mkdir(parents=True)
        (self.repo / "docs" / "changes").mkdir(parents=True)
        self.api = runpy.run_path(str(TASK_CTL))
        self.TaskCtlError = self.api["TaskCtlError"]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(TASK_CTL), "--repo", str(self.repo), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def make_dir(self, task_id: str) -> Path:
        path = self.repo / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def set_task(self, task_id: str, config: dict, *, replace: bool = True) -> dict:
        return self.api["set_task"](self.repo, task_id, config, replace)

    def test_public_api_and_schema_reject_duplicate_unknown_and_invalid_values(self) -> None:
        for name in (
            "TaskCtlError",
            "task_dir",
            "get_task",
            "set_task",
            "list_tasks",
            "parse_config",
            "detect_cycle",
        ):
            self.assertIn(name, self.api)

        parse = self.api["parse_config"]
        with self.assertRaisesRegex(self.TaskCtlError, "duplicate key"):
            parse("type: impl\ntype: hld\nstatus: new\n")
        with self.assertRaisesRegex(self.TaskCtlError, "unknown task control"):
            parse("type: impl\nstatus: new\nextra: true\n")
        with self.assertRaisesRegex(self.TaskCtlError, "not bool"):
            parse("type: impl\nstatus: new\norder: true\n")
        with self.assertRaisesRegex(self.TaskCtlError, "duplicate task-id"):
            parse(
                "type: impl\nstatus: ready\nblocked-by:\n"
                "  - task-id: .cswd/tasks/a\n"
                "  - task-id: .cswd/tasks/a\n"
            )
        with self.assertRaisesRegex(self.TaskCtlError, "status must"):
            parse("type: impl\nstatus: running\n")
        with self.assertRaisesRegex(self.TaskCtlError, "source must end"):
            parse("type: impl\nstatus: new\nsource: docs/design.md\n")
        with self.assertRaisesRegex(self.TaskCtlError, "contains.*\\.\\."):
            parse("type: impl\nstatus: new\nsource: ../spec.md\n")
        with self.assertRaisesRegex(self.TaskCtlError, "exact path"):
            parse(
                "type: impl\nstatus: new\nblocked-by:\n"
                "  - task-id: sibling\n"
            )

    def test_create_defaults_and_requires_only_immediate_parent(self) -> None:
        self.make_dir(".cswd/tasks/parent")
        result = self.api["set_task"](
            self.repo,
            ".cswd/tasks/parent/child",
            {"priority": "P2"},
        )
        self.assertEqual(result, {"type": "hld", "status": "new", "priority": "P2"})
        self.assertEqual(
            self.api["get_task"](self.repo, ".cswd/tasks/parent/child"),
            result,
        )

        with self.assertRaisesRegex(self.TaskCtlError, "parent directory does not exist"):
            self.api["set_task"](
                self.repo,
                ".cswd/tasks/missing/child",
                {"type": "impl"},
            )
        self.assertFalse((self.repo / ".cswd/tasks/missing").exists())

        cli_default = self.run_cli("set", "cli-default")
        self.assertEqual(cli_default.returncode, 0, cli_default.stderr)
        self.assertEqual(
            yaml.safe_load(cli_default.stdout),
            {"type": "hld", "status": "new"},
        )

    def test_existing_whole_replacement_requires_required_fields_and_is_atomic(self) -> None:
        directory = self.make_dir(".cswd/tasks/item")
        original = self.set_task(
            ".cswd/tasks/item",
            {"type": "impl", "status": "ready", "order": 4},
        )
        before = (directory / "task.yml").read_bytes()
        with self.assertRaisesRegex(self.TaskCtlError, "missing required"):
            self.api["set_task"](
                self.repo,
                ".cswd/tasks/item",
                {"priority": "P0"},
                True,
            )
        self.assertEqual((directory / "task.yml").read_bytes(), before)
        self.assertEqual(self.api["get_task"](self.repo, ".cswd/tasks/item"), original)

        with self.assertRaisesRegex(self.TaskCtlError, "priority must"):
            self.api["set_task"](
                self.repo,
                ".cswd/tasks/item",
                {"priority": "urgent"},
            )
        self.assertEqual((directory / "task.yml").read_bytes(), before)

    def test_cli_whole_yaml_fields_and_conflicts(self) -> None:
        self.make_dir(".cswd/tasks/item")
        whole = "type: impl\nstatus: ready\nblocked-by: []\nsource: .cswd/tasks/item/spec.md"
        result = self.run_cli("set", "item", "--all", whole)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(yaml.safe_load(result.stdout)["type"], "impl")

        defaulted = self.run_cli("set", "new-whole", "--all", "priority: P3\n")
        self.assertEqual(defaulted.returncode, 0, defaulted.stderr)
        self.assertEqual(
            yaml.safe_load(defaulted.stdout),
            {"type": "hld", "status": "new", "priority": "P3"},
        )
        status = self.run_cli("get", "item", "--status")
        self.assertEqual((status.returncode, status.stdout), (0, "ready\n"))
        missing = self.run_cli("get", "item", "--priority")
        self.assertEqual((missing.returncode, missing.stdout), (0, "\n"))
        blockers = self.run_cli("get", "item", "--blocked")
        self.assertEqual(yaml.safe_load(blockers.stdout), [])
        all_fields = self.run_cli("get", "item", "--all")
        self.assertEqual(all_fields.returncode, 0, all_fields.stderr)
        self.assertEqual(yaml.safe_load(all_fields.stdout)["source"], ".cswd/tasks/item/spec.md")
        implicit_all = self.run_cli("get", "item")
        self.assertEqual(implicit_all.returncode, 0, implicit_all.stderr)
        self.assertEqual(implicit_all.stdout, all_fields.stdout)

        before = (self.repo / ".cswd/tasks/item/task.yml").read_bytes()
        conflict = self.run_cli(
            "set",
            "item",
            "type: hld\nstatus: new",
            "--status",
            "done",
        )
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("mutually exclusive", conflict.stderr)
        self.assertEqual((self.repo / ".cswd/tasks/item/task.yml").read_bytes(), before)

        duplicate = self.run_cli("set", "item", "--status", "new", "--status", "done")
        self.assertEqual(duplicate.returncode, 2)
        self.assertIn("only once", duplicate.stderr)
        self.assertEqual((self.repo / ".cswd/tasks/item/task.yml").read_bytes(), before)

    def test_cli_blocked_records_replace_the_complete_list(self) -> None:
        self.make_dir(".cswd/tasks/parent")
        self.make_dir(".cswd/tasks/parent/leaf")
        result = self.run_cli(
            "set",
            "parent/leaf",
            "--type",
            "impl",
            "--blocked",
            "{task-id: .cswd/tasks/provider, remarks: public API}",
            "--blocked",
            "[{task-id: docs/changes/external/provider}]",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load(result.stdout)
        self.assertEqual(
            config["blocked-by"],
            [
                {"task-id": ".cswd/tasks/provider", "remarks": "public API"},
                {"task-id": "docs/changes/external/provider"},
            ],
        )
        cleared = self.run_cli("set", "parent/leaf", "--blocked", "[]")
        self.assertEqual(cleared.returncode, 0, cleared.stderr)
        self.assertEqual(yaml.safe_load(cleared.stdout)["blocked-by"], [])

    def test_get_recreates_missing_control_only_for_task_directories(self) -> None:
        directory = self.make_dir("docs/changes/review/item")
        result = self.run_cli("get", "docs/changes/review/item", "--spec")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str((directory / "spec.md").resolve()))

        evidence = directory / "task.md"
        evidence.write_text("**Outcome:** done\n", encoding="utf-8")
        result = self.run_cli("get", "docs/changes/review/item", "--status")
        self.assertEqual((result.returncode, result.stdout), (0, "new\n"))
        self.assertEqual(
            yaml.safe_load((directory / "task.yml").read_text(encoding="utf-8")),
            {"type": "hld", "status": "new"},
        )

        self.make_dir("docs/changes/review/empty")
        empty = self.run_cli("get", "docs/changes/review/empty", "--status")
        self.assertEqual(empty.returncode, 2)
        self.assertIn("task Markdown not found", empty.stderr)
        missing = self.run_cli("get", "docs/changes/review/missing", "--status")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("task directory not found", missing.stderr)

    def test_list_is_direct_recreates_defaults_and_sorted(self) -> None:
        parent = self.make_dir(".cswd/tasks/parent")
        self.make_dir(".cswd/tasks/parent/20-late")
        self.set_task(
            ".cswd/tasks/parent/20-late",
            {"type": "impl", "status": "new", "order": 20},
        )
        self.make_dir(".cswd/tasks/parent/02-first")
        self.set_task(
            ".cswd/tasks/parent/02-first",
            {"type": "hld", "status": "planned", "order": 2},
        )
        self.make_dir(".cswd/tasks/parent/no-order")
        self.set_task(
            ".cswd/tasks/parent/no-order",
            {"type": "impl", "status": "ready"},
        )
        nested = self.make_dir(".cswd/tasks/parent/02-first/01-nested")
        self.set_task(
            ".cswd/tasks/parent/02-first/01-nested",
            {"type": "impl", "status": "new", "order": 1},
        )

        records = self.api["list_tasks"](self.repo, ".cswd/tasks/parent")
        self.assertEqual(
            [record["task-id"] for record in records],
            [
                ".cswd/tasks/parent/02-first",
                ".cswd/tasks/parent/20-late",
                ".cswd/tasks/parent/no-order",
            ],
        )
        self.assertNotIn(
            ".cswd/tasks/parent/02-first/01-nested",
            [record["task-id"] for record in records],
        )

        listed = self.run_cli("list", "parent")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        expected_ids = "".join(record["task-id"] + "\n" for record in records)
        self.assertEqual(listed.stdout, expected_ids)
        full = self.run_cli("list", "parent", "--full")
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertEqual(yaml.safe_load(full.stdout), records)

        (parent / "assets").mkdir()
        (parent / "009-reserved").mkdir()
        self.assertEqual(
            [record["task-id"] for record in self.api["list_tasks"](self.repo, ".cswd/tasks/parent")],
            [record["task-id"] for record in records],
        )

        partial = parent / "partial"
        partial.mkdir()
        (partial / "spec.md").write_text("# Partial\n", encoding="utf-8")
        updated = self.api["list_tasks"](self.repo, ".cswd/tasks/parent")
        self.assertEqual(
            updated[-1],
            {"task-id": ".cswd/tasks/parent/partial", "type": "hld", "status": "new"},
        )
        self.assertEqual(
            yaml.safe_load((partial / "task.yml").read_text(encoding="utf-8")),
            {"type": "hld", "status": "new"},
        )

    def test_impl_list_checks_done_dependencies_missing_dependencies_and_containers(self) -> None:
        self.make_dir("docs/changes/work")
        for name in ("01-provider", "02-consumer", "03-waiting", "04-container"):
            self.make_dir(f"docs/changes/work/{name}")
        self.set_task(
            "docs/changes/work/01-provider",
            {"type": "impl", "status": "done", "order": 1},
        )
        self.set_task(
            "docs/changes/work/02-consumer",
            {
                "type": "impl",
                "status": "ready",
                "order": 2,
                "blocked-by": [{"task-id": "docs/changes/work/01-provider"}],
            },
        )
        self.set_task(
            "docs/changes/work/03-waiting",
            {
                "type": "impl",
                "status": "ready",
                "order": 3,
                "blocked-by": [{"task-id": "docs/changes/missing"}],
            },
        )
        self.set_task(
            "docs/changes/work/04-container",
            {"type": "impl", "status": "ready", "order": 4},
        )
        self.make_dir("docs/changes/work/04-container/01-leaf")
        self.set_task(
            "docs/changes/work/04-container/01-leaf",
            {"type": "impl", "status": "ready", "order": 1},
        )

        ready = self.api["list_tasks"](self.repo, "docs/changes/work", True)
        self.assertEqual(
            [record["task-id"] for record in ready],
            ["docs/changes/work/02-consumer"],
        )

    def test_impl_list_rejects_existing_dependency_cycle(self) -> None:
        self.make_dir(".cswd/tasks/cycle")
        for name in ("a", "b"):
            self.make_dir(f".cswd/tasks/cycle/{name}")
        self.set_task(
            ".cswd/tasks/cycle/a",
            {
                "type": "impl",
                "status": "ready",
                "blocked-by": [{"task-id": ".cswd/tasks/cycle/b"}],
            },
        )
        self.set_task(
            ".cswd/tasks/cycle/b",
            {
                "type": "impl",
                "status": "ready",
                "blocked-by": [{"task-id": ".cswd/tasks/cycle/a"}],
            },
        )
        with self.assertRaisesRegex(self.TaskCtlError, "dependency cycle"):
            self.api["list_tasks"](self.repo, ".cswd/tasks/cycle")
        with self.assertRaisesRegex(self.TaskCtlError, "dependency cycle"):
            self.api["list_tasks"](self.repo, ".cswd/tasks/cycle", True)

        self.api["set_task"](
            self.repo,
            ".cswd/tasks/cycle/b",
            {"status": "done"},
        )
        ready = self.api["list_tasks"](self.repo, ".cswd/tasks/cycle", True)
        self.assertEqual([record["task-id"] for record in ready], [".cswd/tasks/cycle/a"])

    def test_both_canonical_roots_are_list_containers(self) -> None:
        self.make_dir("docs/changes/review")
        self.set_task(
            "docs/changes/review",
            {"type": "hld", "status": "critic", "order": 1},
        )
        records = self.api["list_tasks"](self.repo, "docs/changes")
        self.assertEqual(records[0]["task-id"], "docs/changes/review")
        self.assertEqual(records[0]["status"], "critic")

    def test_plan_is_stable_topological_read_only_and_reports_collisions(self) -> None:
        parent = self.make_dir(".cswd/tasks/parent")
        (parent / "spec.md").write_text("# Parent\n", encoding="utf-8")
        (parent / "01-urgent").mkdir()
        candidates = [
            {"slug": "slow", "type": "impl", "priority": "P2"},
            {"slug": "urgent", "type": "hld", "priority": "P0"},
            {
                "slug": "consumer",
                "type": "impl",
                "priority": "P0",
                "blocked-by": [{"task-id": "slow", "remarks": "needs output"}],
            },
        ]
        proposals = self.api["plan_tasks"](self.repo, ".cswd/tasks/parent", candidates)
        self.assertEqual(
            [record["task-id"] for record in proposals],
            [
                ".cswd/tasks/parent/01-urgent",
                ".cswd/tasks/parent/02-slow",
                ".cswd/tasks/parent/03-consumer",
            ],
        )
        self.assertTrue(proposals[0]["collision"])
        self.assertFalse(proposals[1]["collision"])
        self.assertEqual(
            proposals[2]["blocked-by"],
            [{"task-id": ".cswd/tasks/parent/02-slow", "remarks": "needs output"}],
        )
        self.assertEqual(proposals[2]["source"], ".cswd/tasks/parent/spec.md")
        self.assertFalse((parent / "02-slow").exists())

        cyclic = [
            {"slug": "a", "type": "impl", "priority": "P1", "blocked-by": [{"task-id": "b"}]},
            {"slug": "b", "type": "impl", "priority": "P1", "blocked-by": [{"task-id": "a"}]},
        ]
        with self.assertRaisesRegex(self.TaskCtlError, "dependency cycle"):
            self.api["plan_tasks"](self.repo, ".cswd/tasks/parent", cyclic)

    def test_plan_cli_accepts_block_yaml_and_requires_parent_spec(self) -> None:
        self.make_dir(".cswd/tasks/parent")
        candidates = "- slug: leaf\n  type: impl\n  priority: P1\n"
        missing = self.run_cli("plan", "parent", "--", candidates)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("parent specification not found", missing.stderr)
        (self.repo / ".cswd/tasks/parent/spec.md").write_text("# Parent\n", encoding="utf-8")
        result = self.run_cli("plan", "parent", "--", candidates)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(yaml.safe_load(result.stdout)[0]["order"], 1)


    def test_path_traversal_and_symlink_escape_are_rejected(self) -> None:
        with self.assertRaisesRegex(self.TaskCtlError, r"\.\."):
            self.api["task_dir"](self.repo, ".cswd/tasks/../outside")
        cli = self.run_cli("get", "../outside", "--spec")
        self.assertEqual(cli.returncode, 2)

        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.repo / ".cswd/tasks/escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(self.TaskCtlError, "escapes repository"):
            self.api["task_dir"](self.repo, ".cswd/tasks/escape")

        item = self.make_dir(".cswd/tasks/item")
        (self.repo / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(self.TaskCtlError, "source escapes"):
            self.api["set_task"](
                self.repo,
                ".cswd/tasks/item",
                {"type": "impl", "status": "new", "source": "linked/spec.md"},
                True,
            )
        self.assertFalse((item / "task.yml").exists())


if __name__ == "__main__":
    unittest.main()

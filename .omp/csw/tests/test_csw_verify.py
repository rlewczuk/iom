#!/usr/bin/env python3
"""Exercise verifier gates and recovery against real disposable Git worktrees."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


BIN = Path(__file__).resolve().parents[1] / "bin"
VERIFY = BIN / "csw_verify"
WORKER = BIN / "csw_run_worker"
API = SimpleNamespace(**runpy.run_path(str(VERIFY)))


class VerifierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="csw verifier ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.env = os.environ | {
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Verifier Test", "GIT_AUTHOR_EMAIL": "verifier@example.invalid",
            "GIT_COMMITTER_NAME": "Verifier Test", "GIT_COMMITTER_EMAIL": "verifier@example.invalid",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Verifier Test")
        self.git("config", "user.email", "verifier@example.invalid")
        (self.repo / ".gitignore").write_text(".work/\n.omp/\n", encoding="utf-8")
        self.git("add", ".gitignore")
        self.git("commit", "-qm", "initial")
        (self.repo / ".cswd").mkdir()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], env=self.env,
                              text=True, capture_output=True, check=True, timeout=15).stdout.strip()

    def command(self, program, *args, ok=True, expected=None):
        process = subprocess.run([str(program), "--repo", str(self.repo), *args], env=self.env,
                                 text=True, capture_output=True, timeout=20)
        expected = (0 if ok else 2) if expected is None else expected
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout)

    def advance_integration(self, name="integration.txt", content="advance\n"):
        path = self.repo / name
        path.write_text(content, encoding="utf-8")
        self.git("add", name)
        self.git("commit", "-qm", f"advance {name}")
        return self.git("rev-parse", "HEAD")

    def assessment_packet(self, verified, complexity="simple", **overrides):
        review = verified["review"]
        receipt = verified["receipt"]
        packet = {
            "commit": receipt["commit"],
            "base": receipt["base"],
            "worktree": receipt["worktree"],
            "reviewed_commit": review["reviewed_commit"],
            "rebase_digest": review["rebase_digest"],
            "complexity": complexity,
            "evidence": "Local value reconciliation preserves the value prefix and consumer contract.",
        }
        packet.update(overrides)
        return packet

    def ready(self, name="leaf"):
        task = "example/" + name
        directory = self.repo / ".cswd" / "tasks" / task
        directory.mkdir(parents=True)
        (directory / "spec.md").write_text("# Persist the task's value\n", encoding="utf-8")
        API.worker.task_ctl_set_task(self.repo, ".cswd/tasks/" + task, {"type": "impl", "status": "new"})
        prepared = self.command(WORKER, "prepare", task, "--run-target", task,
                                "--integration-branch", "main", "--integration-base", self.git("rev-parse", "HEAD"))
        wt = Path(prepared["worktree"])
        (wt / (name + ".txt")).write_text("value\n", encoding="utf-8")
        self.amend(task)
        return task, wt

    def amend(self, task):
        self.command(WORKER, "annotate", task, "--outcome", "ready", "--summary", "Implemented value")
        return self.command(WORKER, "commit", task, "--status", "ready", "--outcome", "persist value")

    def plan(self, task, code=None, timeout=5):
        code = code or f"from pathlib import Path; assert Path({(Path(task).name + '.txt')!r}).read_text().startswith('value')"
        return {"commands": [{"name": "behavior", "argv": [sys.executable, "-c", code],
                              "timeout_seconds": timeout}], "total_timeout_seconds": 10}

    def verify(self, task, plan=None, ok=True, expected=None):
        return self.command(VERIFY, "verify", task, "--plan", json.dumps(plan or self.plan(task)),
                            "--summary", "Persisted value verified", ok=ok, expected=expected)


    def backend_verify(self, task, mirror="mirror", plan=None, ok=True, expected=None):
        args = ["verify", task, "--backend-tests", mirror]
        if plan is not None:
            args.extend(["--plan", json.dumps(plan)])
        args.extend(["--summary", "Backend tests verified"])
        return self.command(VERIFY, *args, ok=ok, expected=expected)

    def install_backend_scripts(self, failures=None, children=False):
        failures = failures or {}
        directory = self.repo / ".omp" / "csw" / "bin"
        directory.mkdir(parents=True)
        for backend in ("cpu", "cuda", "rocm", "sycl"):
            fail = int(failures.get(backend, 0))
            child_code = ""
            child_write = ""
            if children:
                child_code = (
                    "p=subprocess.Popen([sys.executable,'-c',"
                    "\"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)\"]); "
                )
                child_write = f"Path(task / 'child-{backend}.pid').write_text(str(p.pid)); "
            finish = f"raise SystemExit({fail})\n" if not children else "time.sleep(60)\n"
            source = (
                "#!/usr/bin/env python3\n"
                "import os, subprocess, sys, time\n"
                "from pathlib import Path\n"
                f"backend={backend!r}; task=Path(os.environ['CSW_REMOTE_TASK_DIR'])\n"
                "assert len(sys.argv) == 2 and sys.argv[1] == 'mirror-id'\n"
                f"{child_code}{child_write}"
                "(task / ('start-' + backend)).write_text('started')\n"
                "while not all((task / ('start-' + item)).exists() for item in ('cpu','cuda','rocm','sycl')):\n"
                "    time.sleep(0.01)\n"
                f"{finish}"
            )
            path = directory / f"test_{backend}"
            path.write_text(source, encoding="utf-8")
            path.chmod(0o755)
    def status(self, task):
        return self.command(VERIFY, "status", task)

    def lifecycle(self, task):
        return API.worker.task_ctl_get_task(self.repo, ".cswd/tasks/" + task)["status"]

    def packet(self, task, receipt, findings=None):
        findings = findings or []
        counts = {severity: sum(item["severity"] == severity for item in findings)
                  for severity in ("critical", "high", "medium", "low")}
        issues = "\n".join(
            f"{item['source_id']}. **[{item['severity']}] {item['title']}** — `{item['location']}`\n"
            f"   - Evidence: {item['evidence']}\n   - Impact: {item['impact']}\n   - Remedy: {item['remedy']}"
            for item in findings
        ) or "**No actionable issues found**"
        raw = (f"## Commit review\n- Commit: {receipt['commit']} persist value\n"
               f"- Baseline: {receipt['base']}\n- Worktree: {receipt['worktree']}\n"
               "- Scope: complete task commit\n\n### Issues\n" + issues + "\n\n### Summary\n"
               + "- Counts: " + ", ".join(f"{severity} {count}" for severity, count in counts.items()) + ".\n"
               "- Assessment: reviewed task behavior\n- Verification: static inspection\n- Limitations: none\n")
        return {"commit": receipt["commit"], "base": receipt["base"], "worktree": receipt["worktree"],
                "round": self.status(task)["next_review_round"],
                "reviewers": [{"alias": alias, "status": "success", "raw_report": raw,
                               "findings": copy.deepcopy(findings)} for alias in sorted(API.ALIASES)],
                "ledger": {"items": [{"id": item["id"], "status": "open"} for item in findings]}}

    def review(self, task, packet, ok=True):
        return self.command(VERIFY, "review", task, "--packet", json.dumps(packet), ok=ok)

    def integrate(self, task, ok=True):
        return self.command(VERIFY, "integrate", task, ok=ok)
    def assess(self, task, packet, ok=True):
        return self.command(VERIFY, "assess-rebase", task, "--packet", json.dumps(packet), ok=ok)

    def counted_plan(self, task, marker, code=None, timeout=5):
        code = code or (
            f"from pathlib import Path; p=Path({str(marker)!r}); "
            "p.write_text((p.read_text() if p.exists() else '') + 'run\\n'); "
            f"assert Path({(Path(task).name + '.txt')!r}).read_text().startswith('value')"
        )
        return self.plan(task, code, timeout)

    @staticmethod
    def finding():
        return {"id": "F1", "source_id": "1", "severity": "high", "title": "Correct the value",
                "location": "leaf.txt:1", "evidence": "consumer receives a wrong value",
                "impact": "supported consumer fails", "remedy": "persist the corrected value"}

    def test_green_gate_review_reuse_and_idempotent_integration(self):
        task, _ = self.ready()
        verified = self.verify(task)
        receipt = verified["receipt"]
        packet = self.packet(task, receipt)
        self.review(task, packet)
        self.review(task, packet)  # Duplicate delivery must not append another round.
        repeated = self.verify(task)
        self.assertTrue(repeated["reused"])
        self.assertEqual(repeated["receipt"]["commit"], receipt["commit"])
        self.integrate(task)
        self.integrate(task)
        self.assertEqual(self.lifecycle(task), "done")
        self.assertEqual((self.repo / "leaf.txt").read_text(), "value\n")
        self.assertEqual(self.git("rev-list", "--count", f"{receipt['base']}..HEAD"), "1")
        evidence = self.status(task)["state"]["review_receipt"]["review_file"]
        self.assertEqual(Path(evidence).read_text().count("## Review round"), 1)
    def test_backend_runners_start_concurrently_and_receipt_reuses(self):
        task, _ = self.ready("backend-concurrent")
        self.install_backend_scripts()
        first = self.backend_verify(task, "mirror-id")
        self.assertTrue(first["ok"])
        self.assertEqual(
            [gate["name"] for gate in first["receipt"]["gates"]],
            ["backend-cpu", "backend-cuda", "backend-rocm", "backend-sycl"],
        )
        self.assertTrue(all(Path(gate["log"]).is_file() for gate in first["receipt"]["gates"]))
        second = self.backend_verify(task, "mirror-id")
        self.assertTrue(second["reused"])

    def test_backend_failure_retains_all_parallel_results_and_blocks_extras(self):
        task, _ = self.ready("backend-failure")
        self.install_backend_scripts({"cuda": 7})
        marker = self.root / "extra-ran"
        extra = self.plan(task, f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')")
        outcome = self.backend_verify(task, "mirror-id", extra, ok=False)
        self.assertEqual(outcome["code"], "gate_failed")
        self.assertEqual(len(outcome["receipt"]["gates"]), 4)
        self.assertEqual([gate["name"] for gate in outcome["receipt"]["gates"]], [
            "backend-cpu", "backend-cuda", "backend-rocm", "backend-sycl",
        ])
        self.assertEqual(outcome["problem"]["remote"]["profile"], "cuda")
        self.assertEqual(outcome["problem"]["remote"]["task_id"], "mirror-id-cuda")
        self.assertEqual(len(outcome["problems"]), 1)
        self.assertFalse(marker.exists())

    def test_backend_sigterm_stops_every_parallel_process_group(self):
        task, _ = self.ready("backend-interrupt")
        self.install_backend_scripts(children=True)
        process = subprocess.Popen(
            [str(VERIFY), "--repo", str(self.repo), "verify", task,
             "--backend-tests", "mirror-id", "--summary", "Interrupt backend tests"],
            env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        markers = [self.repo / ".cswd" / "tasks" / task / f"start-{backend}"
                   for backend in ("cpu", "cuda", "rocm", "sycl")]
        self.wait_files(markers, [process])
        process.terminate()
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 2, stdout + stderr)
        self.assertEqual(json.loads(stdout)["code"], "interrupted")
        for backend in ("cpu", "cuda", "rocm", "sycl"):
            self.assert_not_running(int((self.repo / ".cswd" / "tasks" / task /
                                         f"child-{backend}.pid").read_text()))


    def test_failed_gate_retains_failure_and_cannot_integrate(self):
        task, _ = self.ready()
        outcome = self.verify(task, self.plan(task, "print('real failure'); raise SystemExit(7)"), ok=False)
        self.assertEqual(outcome["code"], "gate_failed")
        gate = outcome["receipt"]["gates"][0]
        self.assertEqual(gate["returncode"], 7)
        self.assertIn("real failure", Path(gate["log"]).read_text())
        self.integrate(task, ok=False)
        self.assertNotEqual(self.lifecycle(task), "done")
        self.assertFalse((self.repo / "leaf.txt").exists())

    def test_gate_environment_and_remote_failure_are_visible(self):
        task, wt = self.ready("remote")
        task_directory = self.repo / ".cswd" / "tasks" / task
        helper_directory = self.repo / ".omp" / "csw" / "bin"
        helper_directory.mkdir(parents=True)
        sync = helper_directory / "csw-remote-sync"
        execute = helper_directory / "csw-remote-exec"
        sync.write_text("#!/bin/sh\nprintf 'remote unavailable\\n' >&2\nexit 19\n", encoding="utf-8")
        execute.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        sync.chmod(0o755)
        execute.chmod(0o755)
        environment = {
            "CSW_REMOTE_TASK_DIR": str(task_directory),
            "CSW_REMOTE_WORKSPACE": str(wt),
        }
        plan = {
            "commands": [
                {"name": "remote sync", "argv": [str(sync), "cuda", "remote-test"],
                 "env": environment, "timeout_seconds": 5},
                {"name": "remote test", "argv": [str(execute), "cuda", "remote-test",
                                                "timeout --kill-after=5s 30s true"],
                 "env": environment, "timeout_seconds": 5},
            ],
            "total_timeout_seconds": 10,
        }
        outcome = self.verify(task, plan, ok=False)
        self.assertEqual(outcome["problem"]["remote"]["profile"], "cuda")
        self.assertEqual(outcome["problem"]["exit"], 19)
        self.assertIn("remote unavailable", outcome["problem"]["diagnostic"])
        self.assertIn("remote verification gate remote sync failed", outcome["problem"]["summary"])
        self.assertEqual(
            outcome["problem"]["remote"]["remote_log"],
            str(task_directory.resolve() / "remote.log"),
        )

    def test_remote_plan_requires_exact_evidence_paths_and_bounded_execution(self):
        task, wt = self.ready("remote-plan")
        plan = self.plan(task)
        plan["commands"][0]["argv"] = ["csw-remote-exec", "cuda", "remote-test", "true"]
        with self.assertRaisesRegex(API.worker.TaskError, "CSW_REMOTE_TASK_DIR"):
            API.validate_remote_plan(API.command_plan(json.dumps(plan)), self.repo, task, wt)


    def assert_not_running(self, pid):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            self.assertEqual(stat.read_text().split(") ", 1)[1][0], "Z")

    def test_raw_backend_runner_plan_requires_integration_path_and_prepared_env(self):
        task, wt = self.ready("backend-plan")
        task_directory = self.repo / ".cswd" / "tasks" / task
        plan = {
            "commands": [{
                "name": "cpu runner",
                "argv": ["/tmp/test_cpu", "mirror-id"],
                "env": {
                    "CSW_REMOTE_TASK_DIR": str(task_directory),
                    "CSW_REMOTE_WORKSPACE": str(wt),
                },
                "timeout_seconds": 5,
            }],
            "total_timeout_seconds": 10,
        }
        with self.assertRaisesRegex(API.worker.TaskError, "integration test_cpu"):
            API.validate_remote_plan(API.command_plan(json.dumps(plan)), self.repo, task, wt)


    def test_test_timeout_stops_even_term_ignoring_descendants(self):
        task, _ = self.ready()
        pid_path = self.root / "child.pid"
        child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        code = ("import subprocess,sys,time; from pathlib import Path; "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
                f"Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(30)")
        outcome = self.verify(task, self.plan(task, code, timeout=0.4), ok=False)
        self.assertEqual(outcome["code"], "gate_timeout")
        self.assertTrue(outcome["receipt"]["gates"][0]["timed_out"])
        self.assert_not_running(int(pid_path.read_text()))
        self.integrate(task, ok=False)
        self.assertNotEqual(self.lifecycle(task), "done")

    def wait_files(self, paths, processes):
        deadline = time.monotonic() + 8
        while not all(path.exists() for path in paths):
            if any(process.poll() is not None for process in processes) or time.monotonic() >= deadline:
                self.fail("gate failed to reach its synchronization marker")
            time.sleep(0.01)

    def start_verify(self, task, plan):
        process = subprocess.Popen([str(VERIFY), "--repo", str(self.repo), "verify", task,
                                    "--plan", json.dumps(plan), "--summary", "Concurrent gate"],
                                   env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=15)
        self.addCleanup(cleanup)
        return process

    def test_sigterm_cancels_active_gate_and_retains_its_log_location(self):
        task, _ = self.ready()
        marker = self.root / "gate.pid"
        code = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
        process = self.start_verify(task, self.plan(task, code))
        self.wait_files([marker], [process])
        process.terminate()
        stdout, stderr = process.communicate(timeout=15)
        self.assertEqual(process.returncode, 2, stdout + stderr)
        self.assertEqual(json.loads(stdout)["code"], "interrupted")
        self.assert_not_running(int(marker.read_text()))
        self.assertTrue(Path(self.status(task)["state"]["active_attempt"]["log_dir"]).is_dir())
        self.integrate(task, ok=False)

    def test_both_aliases_and_authoritative_raw_findings_are_required(self):
        task, _ = self.ready()
        receipt = self.verify(task)["receipt"]
        packet = self.packet(task, receipt, [self.finding()])
        missing = copy.deepcopy(packet)
        missing["reviewers"].pop()
        self.review(task, missing, ok=False)
        omitted = copy.deepcopy(packet)
        omitted["reviewers"][0]["findings"] = []
        self.review(task, omitted, ok=False)
        self.integrate(task, ok=False)
        self.assertNotEqual(self.lifecycle(task), "done")

    def test_malformed_reviewer_outcome_returns_a_structured_blocker(self):
        task, _ = self.ready()
        receipt = self.verify(task)["receipt"]
        packet = self.packet(task, receipt)
        packet["reviewers"][0]["status"] = ["success"]
        rejected = self.review(task, packet, ok=False)
        self.assertFalse(rejected["ok"])
        self.integrate(task, ok=False)

    def test_raw_report_for_another_commit_cannot_approve(self):
        task, _ = self.ready()
        receipt = self.verify(task)["receipt"]
        packet = self.packet(task, receipt)
        packet["reviewers"][0]["raw_report"] = packet["reviewers"][0]["raw_report"].replace(receipt["commit"], "0" * 40)
        self.review(task, packet, ok=False)
        self.integrate(task, ok=False)

    def test_finding_history_survives_repair_and_requires_confirmed_closure(self):
        task, wt = self.ready()
        original = self.verify(task)["receipt"]
        blocked = self.review(task, self.packet(task, original, [self.finding()]), ok=False)
        self.assertEqual(blocked["blocking_ids"], ["F1"])
        (wt / "leaf.txt").write_text("value corrected\n")
        self.amend(task)
        fixed = self.verify(task)["receipt"]
        packet = self.packet(task, fixed)
        self.review(task, packet, ok=False)  # Prior F1 cannot vanish with an empty issues list.
        self.integrate(task, ok=False)
        packet["ledger"]["items"] = [{"id": "F1", "status": "solved", "original_commit": original["commit"],
                                         "fixing_commit": fixed["commit"], "evidence": "fixed value observed by gate and both current reviews"}]
        accepted = self.review(task, packet)
        self.assertEqual(accepted["ledger"]["F1"]["status"], "solved")
        self.assertEqual({source["alias"] for source in accepted["ledger"]["F1"]["sources"]}, API.ALIASES)
        self.integrate(task)
        text = Path(accepted["review_file"]).read_text()
        self.assertIn(original["commit"], text)
        self.assertIn(fixed["commit"], text)
    def test_parallel_gates_do_not_hold_repository_lock(self):
        tasks = [self.ready("first")[0], self.ready("second")[0]]
        release = self.root / "release"
        markers = [self.root / f"started-{index}" for index in range(2)]
        processes = []
        for task, marker in zip(tasks, markers):
            code = ("from pathlib import Path\nimport time\n"
                    f"Path({str(marker)!r}).write_text('started')\n"
                    f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n")
            processes.append(self.start_verify(task, self.plan(task, code)))
        self.wait_files(markers, processes)
        self.assertTrue(all(process.poll() is None for process in processes))
        release.write_text("continue")
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            self.assertEqual(json.loads(stdout)["code"], "verified")

    def test_concurrent_integration_reuses_review_after_clean_rebase(self):
        tasks = [self.ready("first")[0], self.ready("second")[0]]
        packets = {}
        plans = {}
        markers = {}
        conformance_markers = {}
        for task in tasks:
            marker = self.root / (Path(task).name + "-gates")
            markers[task] = marker
            plans[task] = self.counted_plan(task, marker)
            plans[task]["commands"][0]["name"] = "unit"
            conformance_markers[task] = self.root / (Path(task).name + "-conformance")
            conformance = self.counted_plan(task, conformance_markers[task])["commands"][0]
            conformance["name"] = "conformance"
            plans[task]["commands"].append(conformance)
            receipt = self.verify(task, plans[task])["receipt"]
            packets[task] = self.packet(task, receipt)
            self.review(task, packets[task])
        processes = [subprocess.Popen([str(VERIFY), "--repo", str(self.repo), "integrate", task],
                                      env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for task in tasks]
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertIn(process.returncode, (0, 2), stdout + stderr)
            outputs.append(json.loads(stdout))
        self.assertEqual({outcome["code"] for outcome in outputs}, {"integrated", "needs_verification"})
        stale = tasks[next(index for index, outcome in enumerate(outputs) if not outcome["ok"])]
        old_packet = packets[stale]
        receipt = self.verify(stale, plans[stale])
        self.assertEqual(receipt["review"]["action"], "reuse")
        rejected = self.review(stale, old_packet, ok=False)
        self.assertEqual(rejected["code"], "stale_review")
        self.assertTrue(self.status(stale)["state"]["reviews_done"])
        self.integrate(stale)
        self.assertEqual(self.lifecycle(stale), "done")
        self.assertEqual(len(markers[stale].read_text().splitlines()), 2)
        self.assertEqual(conformance_markers[stale].read_text().splitlines(), ["run", "run"])
        self.assertEqual(self.status(stale)["state"]["review_round"], 1)
        self.assertEqual((self.repo / "second.txt").read_text(), "value\n")

        self.assertEqual((self.repo / "first.txt").read_text(), "value\n")

    def test_unreviewed_candidate_is_blocked_and_first_dual_review_is_persistent(self):
        task, _ = self.ready()
        verified = self.verify(task)
        self.assertEqual(verified["review"]["action"], "review_required")
        self.integrate(task, ok=False)
        self.assertNotEqual(self.lifecycle(task), "done")
        state = self.status(task)["state"]
        self.assertFalse(state.get("reviews_done", False))
        accepted = self.review(task, self.packet(task, verified["receipt"]))
        self.assertTrue(accepted["ok"])
        state = self.status(task)["state"]
        self.assertTrue(state["reviews_done"])
        self.assertEqual(state["review_round"], 1)
        self.integrate(task)

    def test_plan_change_runs_fresh_gates_but_keeps_review_for_metadata_only_commit(self):
        task, _ = self.ready()
        marker = self.root / "plan-gates"
        original = self.counted_plan(task, marker)
        first = self.verify(task, original)
        self.review(task, self.packet(task, first["receipt"]))
        self.env["GIT_COMMITTER_DATE"] = "2001-01-01T00:00:00+0000"
        self.amend(task)
        changed = self.counted_plan(task, marker, code=(
            f"from pathlib import Path; p=Path({str(marker)!r}); "
            "p.write_text((p.read_text() if p.exists() else '') + 'changed\\n'); "
            f"assert Path({(Path(task).name + '.txt')!r}).read_text() == 'value\\n'"
        ))
        rebound = self.verify(task, changed)
        self.assertEqual(rebound["receipt"]["tree"], first["receipt"]["tree"])
        self.assertNotEqual(rebound["receipt"]["commit"], first["receipt"]["commit"])
        self.assertEqual(rebound["review"]["action"], "reuse")
        self.assertEqual(len(marker.read_text().splitlines()), 2)
        self.assertEqual(self.status(task)["state"]["review_round"], 1)
        self.integrate(task)

    def test_repeated_clean_rebases_retain_one_review_but_source_edits_require_review(self):
        task, wt = self.ready()
        first = self.verify(task)
        self.review(task, self.packet(task, first["receipt"]))
        self.advance_integration("advance-one.txt")
        replayed = self.verify(task)
        self.assertEqual(replayed["review"]["action"], "reuse")
        self.advance_integration("advance-two.txt")
        replayed_again = self.verify(task)
        self.assertEqual(replayed_again["review"]["action"], "reuse")
        history = self.status(task)["helper"]["rebase_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(self.status(task)["state"]["review_round"], 1)
        edited, edited_wt = self.ready("edited")
        approved = self.verify(edited)
        self.review(edited, self.packet(edited, approved["receipt"]))
        self.advance_integration("advance-edited.txt")
        (edited_wt / "edited.txt").write_text("value changed before replay\n", encoding="utf-8")
        self.amend(edited)
        changed = self.verify(edited)
        self.assertEqual(changed["review"]["action"], "review_required")
        self.integrate(edited, ok=False)

        post, post_wt = self.ready("post")
        approved = self.verify(post)
        self.review(post, self.packet(post, approved["receipt"]))
        self.advance_integration("advance-post.txt")
        replayed = self.verify(post)
        self.assertEqual(replayed["review"]["action"], "reuse")
        (post_wt / "post.txt").write_text("value changed after replay\n", encoding="utf-8")
        self.amend(post)
        changed = self.verify(post)
        self.assertEqual(changed["review"]["action"], "review_required")
        self.integrate(post, ok=False)


    def test_rebase_conflict_requires_assessment_and_simple_assessment_reuses_review(self):
        task, wt = self.ready()
        initial = self.verify(task)
        self.review(task, self.packet(task, initial["receipt"]))
        onto = self.advance_integration("leaf.txt", "integration version\n")
        conflict = self.verify(task, expected=3)
        self.assertEqual(conflict["code"], "repair_needed")
        self.assertEqual(conflict["onto"], onto)
        self.assertEqual(conflict["conflicts"], ["leaf.txt"])
        (wt / "leaf.txt").write_text("value resolved\n", encoding="utf-8")
        continued = self.command(WORKER, "continue-rebase", task)
        self.assertTrue(continued["rebased"])
        self.assertIsNone(self.status(task)["helper"]["pending_rebase"])
        self.amend(task)
        verified = self.verify(task)
        self.assertEqual(verified["review"]["action"], "assessment_required")
        history = self.status(task)["helper"]["rebase_history"]
        self.assertTrue(history[-1]["conflicted"])
        self.assertIn("leaf.txt", history[-1]["conflicts"])
        self.integrate(task, ok=False)
        self.assess(task, self.assessment_packet(verified, commit=initial["receipt"]["commit"]), ok=False)
        stale = self.assessment_packet(verified, reviewed_commit="0" * 40)
        self.assess(task, stale, ok=False)
        mismatched = self.assessment_packet(verified, rebase_digest="not-the-rebase")
        self.assess(task, mismatched, ok=False)
        self.integrate(task, ok=False)
        accepted = self.assess(task, self.assessment_packet(verified))
        self.assertEqual(accepted["review"]["action"], "reuse")
        self.assertEqual(self.status(task)["state"]["review_round"], 1)
        self.advance_integration("after-simple.txt")
        self.assertEqual(self.verify(task)["review"]["action"], "reuse")
        self.integrate(task)
        self.assertEqual((self.repo / "leaf.txt").read_text(), "value resolved\n")

    def test_complex_conflict_assessment_requires_fresh_pair(self):
        task, wt = self.ready("complex")
        initial = self.verify(task)
        self.review(task, self.packet(task, initial["receipt"]))
        self.advance_integration("complex.txt", "base advance\n")
        conflict = self.verify(task, expected=3)
        self.assertEqual(conflict["conflicts"], ["complex.txt"])
        (wt / "complex.txt").write_text("value restructured\n", encoding="utf-8")
        self.command(WORKER, "continue-rebase", task)
        self.amend(task)
        verified = self.verify(task)
        self.assertEqual(verified["review"]["action"], "assessment_required")
        complex_result = self.assess(task, self.assessment_packet(
            verified, "complex", evidence="Resolution changes the component's algorithm and execution structure."))
        self.assertEqual(complex_result["review"]["action"], "review_required")
        self.integrate(task, ok=False)
        self.advance_integration("after-complex.txt")
        self.assertEqual(self.verify(task)["review"]["action"], "review_required")
        current = self.status(task)["state"]["verification_receipt"]
        accepted = self.review(task, self.packet(task, current))
        self.assertTrue(accepted["ok"])
        self.integrate(task)

    def test_rebased_gate_failure_blocks_even_previously_approved_work(self):
        task, _ = self.ready()
        initial = self.verify(task)
        self.review(task, self.packet(task, initial["receipt"]))
        self.advance_integration()
        failed = self.verify(task, self.plan(task, "raise SystemExit(17)"), ok=False)
        self.assertEqual(failed["code"], "gate_failed")
        self.integrate(task, ok=False)
        self.assertNotEqual(self.lifecycle(task), "done")
        self.assertEqual(self.verify(task)["review"]["action"], "reuse")
        self.integrate(task)

    def test_exact_commit_exceptions_do_not_transfer_across_rebase(self):
        for reduced_coverage in (False, True):
            with self.subTest(reduced_coverage=reduced_coverage):
                task, _ = self.ready("reduced" if reduced_coverage else "waived")
                initial = self.verify(task)
                packet = self.packet(task, initial["receipt"])
                for reviewer in packet["reviewers"][1 if reduced_coverage else 0:]:
                    reviewer.update(status="quota", raw_report="provider quota exhausted",
                                    error="provider quota exhausted")
                if not reduced_coverage:
                    packet["waiver"] = {"commit": initial["receipt"]["commit"], "round": packet["round"],
                                       "consent": "Proceed without review for this commit",
                                       "source": "fixture developer response"}
                self.review(task, packet)
                self.assertFalse(self.status(task)["state"]["reviews_done"])
                self.advance_integration(Path(task).name + "-advance.txt")
                self.assertEqual(self.verify(task)["review"]["action"], "review_required")
                self.integrate(task, ok=False)

    def test_fast_forward_metadata_failure_recovers_without_another_merge(self):
        task, wt = self.ready()
        receipt = self.verify(task)["receipt"]
        self.review(task, self.packet(task, receipt))
        def fail_metadata(*args):
            raise API.worker.TaskError("injected metadata write failure")
        with mock.patch.dict(API.worker.integrate_task.__globals__, {"mark_tasks_done": fail_metadata}):
            failed = API.integrate(self.repo, task, wt)
        self.assertEqual(failed["code"], "integration_recovery_needed")
        self.assertEqual(self.git("rev-parse", "HEAD"), receipt["commit"])
        self.assertEqual(self.lifecycle(task), "verified")
        self.integrate(task)
        self.assertEqual(self.lifecycle(task), "done")
        self.assertEqual(self.git("rev-list", "--count", f"{receipt['base']}..HEAD"), "1")


if __name__ == "__main__":
    unittest.main()

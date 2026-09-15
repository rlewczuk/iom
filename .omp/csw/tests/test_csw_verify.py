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
        (self.repo / ".gitignore").write_text(".work/\n", encoding="utf-8")
        self.git("add", ".gitignore")
        self.git("commit", "-qm", "initial")
        (self.repo / ".cswd").mkdir()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], env=self.env,
                              text=True, capture_output=True, check=True, timeout=15).stdout.strip()

    def command(self, program, *args, ok=True):
        process = subprocess.run([str(program), "--repo", str(self.repo), *args], env=self.env,
                                 text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0 if ok else 2, process.stdout + process.stderr)
        return json.loads(process.stdout)

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

    def verify(self, task, plan=None, ok=True):
        return self.command(VERIFY, "verify", task, "--plan", json.dumps(plan or self.plan(task)),
                            "--summary", "Persisted value verified", ok=ok)

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

    def assert_not_running(self, pid):
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            self.assertEqual(stat.read_text().split(") ", 1)[1][0], "Z")

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

    def test_only_exact_round_user_waiver_unblocks_two_availability_failures(self):
        task, _ = self.ready()
        receipt = self.verify(task)["receipt"]
        packet = self.packet(task, receipt)
        for reviewer in packet["reviewers"]:
            reviewer.update(status="quota", raw_report="provider quota exhausted", error="provider quota exhausted")
        waiting = self.review(task, packet, ok=False)
        self.assertEqual(waiting["code"], "waiver_required")
        self.integrate(task, ok=False)
        packet["waiver"] = {"commit": receipt["commit"], "round": packet["round"],
                             "consent": "Proceed without review for this commit", "source": "fixture developer response"}
        self.review(task, packet)
        self.integrate(task)
        self.assertEqual(self.lifecycle(task), "done")

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

    def test_concurrent_integration_rejects_stale_base_then_replays(self):
        tasks = [self.ready("first")[0], self.ready("second")[0]]
        packets = {}
        for task in tasks:
            receipt = self.verify(task)["receipt"]
            packets[task] = self.packet(task, receipt)
            self.review(task, packets[task])
        processes = [subprocess.Popen([str(VERIFY), "--repo", str(self.repo), "integrate", task],
                                     env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for task in tasks]
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertIn(process.returncode, (0, 2), stdout + stderr)
            outputs.append(json.loads(stdout))
        self.assertEqual({outcome["code"] for outcome in outputs}, {"integrated", "needs_verification"})
        stale = tasks[next(index for index, outcome in enumerate(outputs) if not outcome["ok"])]
        receipt = self.verify(stale)["receipt"]
        self.review(stale, packets[stale], ok=False)
        self.integrate(stale, ok=False)
        self.review(stale, self.packet(stale, receipt))
        self.integrate(stale)
        self.assertTrue(all(self.lifecycle(task) == "done" for task in tasks))
        self.assertEqual((self.repo / "first.txt").read_text(), "value\n")
        self.assertEqual((self.repo / "second.txt").read_text(), "value\n")

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

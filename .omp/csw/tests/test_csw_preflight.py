#!/usr/bin/env python3
"""Behavioral tests for the shared csw_preflight executable."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


HELPER = Path(__file__).parents[1] / "bin" / "csw_preflight"
READ_ONLY = "read, grep, glob, lsp, ast_grep"
LOOKUP = f"{READ_ONLY}, bash, web_search"
BUILDER = f"{READ_ONLY}, ast_edit, bash, edit, write"
BOSS_PROFILES = {
    "boss-errand": ("@smol", LOOKUP),
    "scout": ("@smol", f"[{READ_ONLY}]"),
    "boss-reviewer": ("@task", f"[{READ_ONLY}]"),
    "boss-builder-fast": ("@smol", BUILDER),
    "boss-builder-fast-deep": ("@smol", BUILDER),
    "boss-builder": ("@task", BUILDER),
    "boss-builder-deep": ("@task", BUILDER),
    "boss-builder-strong": ("@slow", BUILDER),
    "boss-builder-strong-deep": ("@slow", BUILDER),
    "boss-advisor": ("@advisor", "[]"),
    "boss-advocate-fast": ("@smol", LOOKUP),
    "boss-advocate": ("@task", LOOKUP),
    "boss-advocate-strong": ("@slow", LOOKUP),
    "csw-plan-facts": ("@task", f"[{READ_ONLY}]"),
}


class CswPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="csw-preflight-")
        self.repo = Path(self.temporary.name) / "repo"
        (self.repo / ".omp" / "agents").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Preflight Test"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "preflight@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "--allow-empty", "-m", "initial"], check=True)
        self.write_profile("spec-run-all-implementer", "@implementer", BUILDER, "[spec-run-debug]")
        self.write_profile("spec-run-debug", "@slow", f"[{READ_ONLY}, bash]", "[]", advisor=False)
        self.omp = Path(self.temporary.name) / "omp"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_profile(
        self,
        name: str,
        model: str,
        tools: str,
        spawns: str | None = None,
        *,
        advisor: bool | None = None,
    ) -> None:
        lines = [
            "---",
            f"name: {name}",
            f"description: Test profile for {name}.",
            f"tools: {tools}",
        ]
        if spawns is not None:
            lines.append(f"spawns: {spawns}")
        if advisor is not None:
            lines.append(f"advisor: {'true' if advisor else 'false'}")
        lines.extend((f'model: "{model}"', "---", ""))
        (self.repo / ".omp" / "agents" / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")

    def write_boss_profiles(self) -> None:
        for name, (model, tools) in BOSS_PROFILES.items():
            advisor = False if name in ("scout", "boss-reviewer", "csw-plan-facts") else None
            spawns = "[]" if name in ("scout", "boss-reviewer", "boss-advisor", "csw-plan-facts") else None
            self.write_profile(name, model, tools, spawns, advisor=advisor)

    def write_omp(self, config: dict, models: list[dict], *, exit_status: int = 0) -> None:
        payload_config = json.dumps(config)
        payload_models = json.dumps({"models": models})
        self.omp.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f"config = json.loads({payload_config!r})\n"
            f"models = json.loads({payload_models!r})\n"
            f"status = {exit_status}\n"
            "if status:\n"
            "    print('fake omp failure', file=sys.stderr)\n"
            "    raise SystemExit(status)\n"
            "if sys.argv[1:4] == ['config', 'list', '--json']:\n"
            "    print(json.dumps(config))\n"
            "elif sys.argv[1:3] == ['models', '--json']:\n"
            "    print(json.dumps(models))\n"
            "else:\n"
            "    raise SystemExit(9)\n",
            encoding="utf-8",
        )
        self.omp.chmod(self.omp.stat().st_mode | stat.S_IXUSR)

    @staticmethod
    def record(value):
        return {"value": value, "type": "test"}

    def config(self, **changes):
        values = {
            "modelRoles": {
                "implementer": "openai/cheap",
                "slow": "openai/slow",
                "smol": "openai/cheap",
                "task": "openai/task",
                "advisor": "openai/advisor",
            },
            "task.agentModelOverrides": {},
            "task.agentAdvisor": {},
            "task.agentPrewalk": {},
            "task.disabledAgents": [],
            "task.maxRecursionDepth": 2,
        }
        values.update(changes)
        return {path: self.record(value) for path, value in values.items()}

    @staticmethod
    def models(*, cheap=True, slow=True):
        result = [
            {"selector": "openai/task", "provider": "openai", "id": "task", "thinking": ["high"]},
            {"selector": "openai/advisor", "provider": "openai", "id": "advisor", "thinking": ["high"]},
        ]
        if cheap:
            result.append({"selector": "openai/cheap", "provider": "openai", "id": "cheap", "thinking": ["low", "high"]})
        if slow:
            result.append({"selector": "openai/slow", "provider": "openai", "id": "slow", "thinking": ["high"]})
        return result

    def run_helper(self, workflow="csw-run"):
        return subprocess.run(
            [str(HELPER), "--repo", str(self.repo), "--workflow", workflow, "--omp", str(self.omp)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_healthy_spec_run_all_configuration_and_git_metadata(self):
        self.write_omp(self.config(), self.models())
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["workflow"], "csw-run")
        self.assertEqual(payload["git"]["repo_root"], str(self.repo.resolve()))
        self.assertTrue(payload["git"]["head"])
        self.assertFalse(payload["git"]["clean"])
        self.assertEqual(payload["agents"]["spec-run-all-implementer"]["resolved"], "openai/cheap")
        self.assertTrue(payload["agents"]["spec-run-debug"]["available"])
        self.assertEqual(payload["model_count"], 4)
        self.assertEqual(len(payload["models"]), 2)

    def test_detached_head_is_valid_git_metadata(self):
        subprocess.run(["git", "-C", str(self.repo), "checkout", "-q", "--detach"], check=True)
        self.write_omp(self.config(), self.models())
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["git"]["detached"])
        self.assertIsNone(payload["git"]["branch"])

    def test_csw_run_worker_checks_only_debugger_profile(self):
        (self.repo / ".omp" / "agents" / "spec-run-all-implementer.md").unlink()
        self.write_omp(self.config(), self.models())
        result = self.run_helper("csw-run-worker")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(set(payload["agents"]), {"spec-run-debug"})

    def test_boss_contract_validates_every_associated_profile(self):
        self.write_boss_profiles()
        self.write_omp(self.config(), self.models())
        result = self.run_helper("boss")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(set(payload["agents"]), set(BOSS_PROFILES))
        self.assertEqual(payload["agents"]["boss-advisor"]["tools"], [])
        self.assertEqual(payload["agents"]["csw-plan-facts"]["resolved"], "openai/task")

    def test_role_alias_chain_preserves_thinking_suffix(self):
        self.write_omp(
            self.config(modelRoles={"implementer": "@worker", "worker": "openai/cheap:low", "slow": "openai/slow"}),
            self.models(),
        )
        result = self.run_helper()
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["agents"]["spec-run-all-implementer"]["resolved"], "openai/cheap")
        self.assertEqual(payload["agents"]["spec-run-all-implementer"]["thinking"], "low")

    def test_missing_role_and_model_are_blockers(self):
        self.write_omp(
            self.config(modelRoles={"implementer": "@missing", "slow": "openai/slow"}),
            self.models(slow=False),
        )
        result = self.run_helper()
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("missing model role" in error for error in payload["errors"]))
        self.assertTrue(any("unavailable" in error for error in payload["errors"]))

    def test_override_and_profile_contract_are_rejected(self):
        self.write_profile("spec-run-debug", "@task", f"[{READ_ONLY}, bash]", "[]", advisor=False)
        self.write_omp(
            self.config(**{"task.agentModelOverrides": {"spec-run-all-implementer": "@slow"}}),
            self.models(),
        )
        payload = json.loads(self.run_helper().stdout)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("exact contract" in error for error in payload["errors"]))
        self.assertTrue(any("exact model alias" in error for error in payload["errors"]))

    def test_advisor_prewalk_disabled_and_depth_are_blockers(self):
        self.write_omp(
            self.config(
                **{
                    "task.agentAdvisor": {"spec-run-all-implementer": "on"},
                    "task.agentPrewalk": {"spec-run-all-implementer": True},
                    "task.disabledAgents": ["spec-run-debug"],
                    "task.maxRecursionDepth": 1,
                }
            ),
            self.models(),
        )
        payload = json.loads(self.run_helper().stdout)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("advisor" in error for error in payload["errors"]))
        self.assertTrue(any("prewalk" in error for error in payload["errors"]))
        self.assertTrue(any("disabled" in error for error in payload["errors"]))
        self.assertTrue(any("maxRecursionDepth" in error for error in payload["errors"]))

    def test_command_failure_is_json_blocker(self):
        self.write_omp(self.config(), self.models(), exit_status=7)
        result = self.run_helper()
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("command failed" in error for error in payload["errors"]))


if __name__ == "__main__":
    unittest.main()

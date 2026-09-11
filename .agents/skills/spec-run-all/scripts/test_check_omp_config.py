#!/usr/bin/env python3
"""Behavioral tests for check_omp_config.py using temporary repositories."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).with_name("check_omp_config.py")


class CheckOmpConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="check-omp-config-")
        self.repo = Path(self.temporary.name) / "repo"
        (self.repo / ".omp" / "agents").mkdir(parents=True)
        self.write_profile("spec-run-all-implementer", "@implementer", "[spec-run-debug]")
        self.write_profile("spec-run-debug", "@slow", "[]")
        self.omp = Path(self.temporary.name) / "omp"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_profile(self, name: str, model: str, spawns: str) -> None:
        (self.repo / ".omp" / "agents" / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: Test profile for {name}.\nmodel: \"{model}\"\nspawns: {spawns}\n---\n",
            encoding="utf-8",
        )

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
            "modelRoles": {"implementer": "openai/cheap", "slow": "openai/slow"},
            "task.agentModelOverrides": {},
            "task.agentAdvisor": {},
            "task.agentPrewalk": {},
            "task.disabledAgents": [],
            "task.maxRecursionDepth": 2,
        }
        values.update(changes)
        return {path: self.record(value) for path, value in values.items()}

    def models(self, *, cheap=True, slow=True):
        result = []
        if cheap:
            result.append({"selector": "openai/cheap", "provider": "openai", "id": "cheap", "thinking": ["low"]})
        if slow:
            result.append({"selector": "openai/slow", "provider": "openai", "id": "slow", "thinking": ["high"]})
        return result

    def run_helper(self):
        return subprocess.run(
            [sys.executable, str(HELPER), "--repo", str(self.repo), "--omp", str(self.omp)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_healthy_configuration(self):
        self.write_omp(self.config(), self.models())
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["agents"]["spec-run-all-implementer"]["resolved"], "openai/cheap")
        self.assertTrue(payload["agents"]["spec-run-debug"]["available"])


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
        self.write_omp(self.config(modelRoles={"implementer": "@missing", "slow": "openai/slow"}), self.models(slow=False))
        result = self.run_helper()
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("missing model role" in error for error in payload["errors"]))
        self.assertTrue(any("unavailable" in error for error in payload["errors"]))

    def test_override_and_profile_contract_are_rejected(self):
        self.write_profile("spec-run-debug", "@task", "[]")
        self.write_omp(
            self.config(**{"task.agentModelOverrides": {"spec-run-all-implementer": "@slow"}}),
            self.models(),
        )
        payload = json.loads(self.run_helper().stdout)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("exact contract" in error for error in payload["errors"]))
        self.assertTrue(any("exact model alias" in error for error in payload["errors"]))

    def test_advisor_prewalk_disabled_and_depth_blockers(self):
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

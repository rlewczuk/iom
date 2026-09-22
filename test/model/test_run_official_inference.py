#!/usr/bin/env python3
"""Standard-library regression tests for the official inference wrapper."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("run_official_inference.py")
SPEC = importlib.util.spec_from_file_location("official_wrapper", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def snapshot(prefix: list[int], selected: int, index: int) -> dict[str, object]:
    logits = [0.0] * WRAPPER.GEOMETRY["vocab_size"]
    logits[0] = 2.0
    start = 0 if index == 0 else len(prefix) - 1
    length = len(prefix) if index == 0 else 1
    return {
        "phase": "prefill" if index == 0 else "cached-decode",
        "prefix_ids": prefix,
        "position_start": start,
        "run_length": length,
        "absolute_positions": list(range(start, start + length)),
        "logits": logits,
        "greedy_id": 0,
        "selected_id": selected,
        "margin_stable": True,
    }


def reference_case(
    name: str,
    mode: str,
    policy: str,
    prompt: list[int],
    decode: list[int],
) -> dict[str, object]:
    rendered = (
        b"The capital of France is"
        if mode == "raw"
        else b"<|user|>\nHello.</s>\n<|assistant|>\n"
    )
    input_record: dict[str, object] = {
        "mode": mode,
        "selection_policy": policy,
        "prompt_ids": prompt,
        "rendered_utf8": list(rendered),
        "positions": list(range(len(prompt))),
        "decode_ids": decode,
        "max_new_tokens": 0 if not decode else 4,
        "bos_policy": {
            "add_special_tokens": mode == "raw",
            "require_bos": mode == "raw",
            "bos_token_id": 1,
        },
    }
    if mode == "raw":
        input_record["text"] = "The capital of France is"
    else:
        input_record["messages"] = [{"role": "user", "content": "Hello."}]
    snapshots = [
        snapshot(prompt + decode[:index], selected, index)
        for index, selected in enumerate(decode)
    ]
    return {
        "id": name,
        "input": input_record,
        "snapshots": snapshots,
        "expected_result": {
            "token_ids": decode,
            "stop_reason": "max_new_tokens",
            "initialized_kv_length": 0 if not decode else len(prompt) + len(decode) - 1,
        },
    }


def write_reference(model: Path, reference: Path, artifact_id: str) -> None:
    config = {
        **WRAPPER.GEOMETRY,
        "model_type": "llama",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "model_id": "TinyLlama/wrapper-regression",
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in WRAPPER.REQUIRED_MODEL_FILES - {"config.json"}:
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"verified model bytes")
    cases = [
        reference_case(
            "raw-production-greedy",
            "raw",
            "production-greedy",
            [1, 10],
            [0, 0, 0, 0],
        ),
        reference_case(
            "raw-fixed-reference-continuation",
            "raw",
            "fixed-reference-continuation",
            [1, 10],
            list(WRAPPER.FORCED_IDS),
        ),
        reference_case(
            "chat-production-greedy",
            "chat",
            "production-greedy",
            [10],
            [0, 0, 0, 0],
        ),
        reference_case(
            "chat-fixed-reference-continuation",
            "chat",
            "fixed-reference-continuation",
            [10],
            list(WRAPPER.FORCED_IDS),
        ),
        reference_case("zero-new-token", "raw", "none", [1, 10], []),
    ]
    canonical = json.dumps(
        cases,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    pack = {
        "schema_version": 1,
        "kind": "official",
        "provenance": {
            "runtime": WRAPPER.PINNED_RUNTIME,
            "packages": WRAPPER.PINNED_PACKAGES,
            "precision": WRAPPER.PINNED_PRECISION,
            "generator_sha256": "0" * 64,
            "case_payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "command": ["python3", "wrapper-regression"],
            "model_id": config["model_id"],
            "revision": artifact_id,
            "artifacts": WRAPPER.model_artifacts(model),
        },
        "tolerances": WRAPPER.TOLERANCES,
        "model_config": config,
        "artifact_id": artifact_id,
        "cases": cases,
    }
    reference.write_text(
        json.dumps(pack, allow_nan=False), encoding="utf-8"
    )


class WrapperRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="iom-official-wrapper-test-"
        )
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.reference = self.root / "reference.json"
        self.evidence = self.root / "evidence.json"
        self.artifact_id = "wrapper-regression"
        write_reference(self.model, self.reference, self.artifact_id)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "IOM_TEST_MODEL_DIR": str(self.model),
                "IOM_TEST_MODEL_ID": self.artifact_id,
                "IOM_TEST_MODEL_REFERENCE": str(self.reference),
                "IOM_TEST_MODEL_EVIDENCE": str(self.evidence),
            }
        )
        return environment

    def invoke(
        self,
        executable: Path,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--backend",
                "cpu",
                "--executable",
                str(executable),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment() if environment is None else environment,
            timeout=10,
            check=False,
        )

    def test_tampered_artifact_rejects_before_sentinel_launch(self) -> None:
        marker = self.root / "launched"
        sentinel = self.root / "sentinel.py"
        sentinel.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('launched')\n",
            encoding="utf-8",
        )
        sentinel.chmod(0o755)
        (self.model / "model.safetensors").write_bytes(b"tampered bytes")

        completed = self.invoke(sentinel)

        self.assertEqual(completed.returncode, 2)
        self.assertFalse(marker.exists())
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["stage"], "input-validation")
        self.assertIn("artifact identity mismatch", evidence["diagnostics"][0])
        self.assertEqual(
            evidence["argv"],
            [str(sentinel), "--test-case=*real model inference*"],
        )


    def test_old_tolerance_pack_rejects_before_sentinel_launch(self) -> None:
        marker = self.root / "launched"
        sentinel = self.root / "sentinel.py"
        sentinel.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('launched')\n",
            encoding="utf-8",
        )
        sentinel.chmod(0o755)
        pack = json.loads(self.reference.read_text(encoding="utf-8"))
        pack["tolerances"]["absolute"] = 0.25
        pack["tolerances"]["formula"] = (
            "abs(actual-ref) <= 0.25 + 0.02*abs(ref)"
        )
        self.reference.write_text(
            json.dumps(pack, allow_nan=False), encoding="utf-8"
        )

        completed = self.invoke(sentinel)

        self.assertEqual(completed.returncode, 2)
        self.assertFalse(marker.exists())
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["stage"], "input-validation")
        self.assertIn(
            "comparison policy differs from the frozen policy",
            evidence["diagnostics"][0],
        )

    def test_launch_oserror_preserves_machine_readable_evidence(self) -> None:
        invalid = self.root / "invalid-executable"
        invalid.write_text("not an executable format\n", encoding="utf-8")
        invalid.chmod(0o755)

        completed = self.invoke(invalid)

        self.assertEqual(completed.returncode, 2)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["stage"], "launch")
        self.assertIn("cannot launch backend test", evidence["diagnostics"][0])
        self.assertEqual(evidence["artifact_id"], self.artifact_id)
        self.assertEqual(evidence["environment"], {
            name: self.environment()[name]
            for name in (
                "IOM_TEST_MODEL_DIR",
                "IOM_TEST_MODEL_ID",
                "IOM_TEST_MODEL_REFERENCE",
                "IOM_TEST_MODEL_EVIDENCE",
            )
        })
    def test_missing_inputs_replace_stale_evidence_before_launch(self) -> None:
        marker = self.root / "launched"
        sentinel = self.root / "sentinel.py"
        sentinel.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('launched')\n",
            encoding="utf-8",
        )
        sentinel.chmod(0o755)
        artifact = self.model / "model.safetensors"

        for label, missing in (
            ("reference", self.reference),
            ("executable", sentinel),
            ("artifact", artifact),
        ):
            with self.subTest(label=label):
                original = missing.read_bytes()
                mode = missing.stat().st_mode
                self.evidence.write_text('{"status":"stale"}', encoding="utf-8")
                missing.unlink()
                try:
                    completed = self.invoke(sentinel)
                    self.assertEqual(completed.returncode, 2)
                    self.assertFalse(marker.exists())
                    recorded = json.loads(
                        self.evidence.read_text(encoding="utf-8")
                    )
                    self.assertEqual(recorded["status"], "failed")
                    self.assertEqual(recorded["stage"], "input-validation")
                    self.assertNotEqual(recorded, {"status": "stale"})
                finally:
                    missing.write_bytes(original)
                    missing.chmod(mode)

    def test_removed_executable_still_records_launch_failure(self) -> None:
        executable = self.root / "removed-before-popen.py"
        executable.write_text(
            "#!/usr/bin/env python3\nraise SystemExit(0)\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        self.evidence.write_text('{"status":"stale"}', encoding="utf-8")

        def disappear(*_args: object, **_kwargs: object) -> object:
            executable.unlink()
            raise FileNotFoundError("removed immediately before Popen")

        with mock.patch.dict(os.environ, self.environment(), clear=True):
            with mock.patch.object(
                WRAPPER.subprocess, "Popen", side_effect=disappear
            ):
                result = WRAPPER.main([
                    "--backend",
                    "cpu",
                    "--executable",
                    str(executable),
                ])

        self.assertEqual(result, 2)
        recorded = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(recorded["status"], "failed")
        self.assertEqual(recorded["stage"], "launch")
        self.assertIn("removed immediately before Popen", recorded["diagnostics"][0])

    def test_removed_reference_during_child_run_records_failure(self) -> None:
        executable = self.root / "remove-reference.py"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "Path(os.environ['IOM_TEST_MODEL_REFERENCE']).unlink()\n"
            "raise SystemExit(3)\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        self.evidence.write_text('{"status":"stale"}', encoding="utf-8")

        completed = self.invoke(executable)

        self.assertEqual(completed.returncode, 3)
        self.assertFalse(self.reference.exists())
        recorded = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(recorded["status"], "failed")
        self.assertEqual(recorded["returncode"], 3)
        self.assertNotEqual(recorded, {"status": "stale"})


    def test_protected_evidence_aliases_leave_inputs_unchanged(self) -> None:
        marker = self.root / "launched"
        sentinel = self.root / "sentinel.py"
        sentinel.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('launched')\n",
            encoding="utf-8",
        )
        sentinel.chmod(0o755)
        artifact = self.model / "model.safetensors"

        for label, protected in (
            ("reference", self.reference),
            ("artifact", artifact),
            ("executable", sentinel),
        ):
            with self.subTest(label=label):
                original = protected.read_bytes()
                environment = self.environment()
                environment["IOM_TEST_MODEL_EVIDENCE"] = str(protected)
                completed = self.invoke(sentinel, environment)
                self.assertEqual(completed.returncode, 2)
                self.assertFalse(marker.exists())
                self.assertEqual(protected.read_bytes(), original)

        hard_link = self.root / "artifact-hard-link"
        os.link(artifact, hard_link)
        artifact_bytes = artifact.read_bytes()
        environment = self.environment()
        environment["IOM_TEST_MODEL_EVIDENCE"] = str(hard_link)
        completed = self.invoke(sentinel, environment)
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(marker.exists())
        self.assertEqual(artifact.read_bytes(), artifact_bytes)
        self.assertEqual(hard_link.read_bytes(), artifact_bytes)

        temporary_target = self.root / "temporary-alias-evidence.json"
        temporary = temporary_target.with_name(
            f".{temporary_target.name}.tmp"
        )
        os.link(artifact, temporary)
        environment = self.environment()
        environment["IOM_TEST_MODEL_EVIDENCE"] = str(temporary_target)
        completed = self.invoke(sentinel, environment)
        self.assertEqual(completed.returncode, 2)
        self.assertFalse(marker.exists())
        self.assertFalse(temporary_target.exists())
        self.assertEqual(artifact.read_bytes(), artifact_bytes)
        self.assertEqual(temporary.read_bytes(), artifact_bytes)

        for destination in (
            self.model / "new-evidence.json",
            self.root / "model-alias" / "new-evidence.json",
        ):
            if destination.parent.name == "model-alias":
                destination.parent.symlink_to(
                    self.model, target_is_directory=True
                )
            environment = self.environment()
            environment["IOM_TEST_MODEL_EVIDENCE"] = str(destination)
            completed = self.invoke(sentinel, environment)
            self.assertEqual(completed.returncode, 2)
            self.assertFalse(marker.exists())
            self.assertFalse(destination.exists())

    def test_pack_model_id_allows_an_undeclared_caller_identity(self) -> None:
        pack = json.loads(self.reference.read_text(encoding="utf-8"))
        config_path = self.model / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("model_id")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        WRAPPER.validate_config(self.model, pack["model_config"])

        config["_name_or_path"] = "TinyLlama/conflicting-distribution"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(WRAPPER.HarnessError):
            WRAPPER.validate_config(self.model, pack["model_config"])

    def test_frozen_cases_are_id_only_and_exactly_typed(self) -> None:
        pack = json.loads(self.reference.read_text(encoding="utf-8"))
        WRAPPER.validate_cases(pack["cases"])

        mutations = []
        name_alias = json.loads(json.dumps(pack["cases"]))
        name_alias[0]["name"] = name_alias[0].pop("id")
        mutations.append(name_alias)
        boolean_position = json.loads(json.dumps(pack["cases"]))
        boolean_position[0]["input"]["positions"][0] = False
        mutations.append(boolean_position)
        integer_bos_flag = json.loads(json.dumps(pack["cases"]))
        integer_bos_flag[0]["input"]["bos_policy"]["add_special_tokens"] = 1
        mutations.append(integer_bos_flag)
        boolean_token = json.loads(json.dumps(pack["cases"]))
        boolean_token[0]["expected_result"]["token_ids"][0] = False
        mutations.append(boolean_token)
        boolean_kv = json.loads(json.dumps(pack["cases"]))
        boolean_kv[-1]["expected_result"]["initialized_kv_length"] = False
        mutations.append(boolean_kv)
        boolean_absolute_position = json.loads(json.dumps(pack["cases"]))
        boolean_absolute_position[0]["snapshots"][0][
            "absolute_positions"
        ][0] = False
        mutations.append(boolean_absolute_position)

        for index, malformed in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(WRAPPER.HarnessError):
                    WRAPPER.validate_cases(malformed)

        malformed_pack = json.loads(
            self.reference.read_text(encoding="utf-8")
        )
        malformed_pack["schema_version"] = True
        malformed_path = self.root / "boolean-schema.json"
        malformed_path.write_text(
            json.dumps(malformed_pack), encoding="utf-8"
        )
        with self.assertRaises(WRAPPER.HarnessError):
            WRAPPER.validate_reference(
                malformed_path, self.model, self.artifact_id
            )

        malformed_pack = json.loads(
            self.reference.read_text(encoding="utf-8")
        )
        malformed_pack["model_config"]["tie_word_embeddings"] = 0
        malformed_path = self.root / "integer-tie.json"
        malformed_path.write_text(
            json.dumps(malformed_pack), encoding="utf-8"
        )
        with self.assertRaises(WRAPPER.HarnessError):
            WRAPPER.validate_reference(
                malformed_path, self.model, self.artifact_id
            )

    def test_success_stdout_is_compact_and_evidence_retains_measurement(
        self,
    ) -> None:
        environment = self.environment()
        environment["OMP_NUM_THREADS"] = "3"
        reference_identity = WRAPPER.file_identity(
            self.reference, str(self.reference)
        )
        artifacts = WRAPPER.model_artifacts(self.model)
        selected_environment = {
            name: environment[name]
            for name in WRAPPER.MEASUREMENT_ENVIRONMENT
            if name in environment
        }
        measurement = {
            "schema_version": 1,
            "kind": "official-inference-evidence",
            "status": "passed",
            "backend": "cpu",
            "reference": {
                "size": reference_identity["size"],
                "sha256": reference_identity["sha256"],
                "artifact_id": self.artifact_id,
                "artifacts": artifacts,
            },
            "environment": selected_environment,
            "cases": [
                {"id": name} for name in WRAPPER.EXPECTED_CASES
            ],
        }
        executable = self.root / "successful-backend.py"
        encoded_measurement = json.dumps(measurement)
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            "from pathlib import Path\n"
            f"measurement = json.loads({encoded_measurement!r})\n"
            "Path(os.environ['IOM_TEST_MODEL_EVIDENCE']).write_text("
            "json.dumps(measurement), encoding='utf-8')\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

        completed = self.invoke(executable, environment)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(
            set(summary),
            {"status", "backend", "reference_sha256", "evidence"},
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["backend"], "cpu")
        self.assertEqual(
            summary["reference_sha256"], reference_identity["sha256"]
        )
        self.assertLess(len(completed.stdout.encode("utf-8")), 500)
        wrapper_evidence = json.loads(
            self.evidence.read_text(encoding="utf-8")
        )
        self.assertEqual(wrapper_evidence["status"], "passed")
        self.assertEqual(wrapper_evidence["measurement"], measurement)

    def test_registered_timeout_reaps_child_and_publishes_evidence(
        self,
    ) -> None:
        marker = self.root / "timeout-child.pid"
        sleeper = self.root / "timeout-backend.py"
        sleeper.write_text(
            "#!/usr/bin/env python3\n"
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            "while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        sleeper.chmod(0o755)
        environment = self.environment()
        stdout = io.StringIO()
        stderr = io.StringIO()

        started = time.monotonic()
        with mock.patch.dict(os.environ, environment, clear=True):
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    result = WRAPPER.main(
                        [
                            "--backend",
                            "cpu",
                            "--executable",
                            str(sleeper),
                        ],
                        execution_timeout=0.2,
                        terminate_timeout=0.1,
                        kill_timeout=0.5,
                    )
        elapsed = time.monotonic() - started

        self.assertEqual(result, 124, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertLess(elapsed, 2.0)
        child_pid = int(marker.read_text(encoding="utf-8"))
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "failed")
        self.assertLess(evidence["returncode"], 0)
        self.assertEqual(
            evidence["supervision"],
            {
                "execution_timeout_seconds": 0.2,
                "terminate_grace_seconds": 0.1,
                "kill_reap_seconds": 0.5,
                "timed_out": True,
                "signals_sent": ["SIGTERM", "SIGKILL"],
                "reap": "after-SIGKILL",
            },
        )
        self.assertIn("exceeded 0.2 seconds", evidence["diagnostics"][0])

    def test_registered_timeout_drains_residual_process_group(
        self,
    ) -> None:
        leader_marker = self.root / "timeout-leader.pid"
        descendant_marker = self.root / "timeout-descendant.pid"
        descendant_source = (
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"Path({str(descendant_marker)!r}).write_text(str(os.getpid()))\n"
            "while True: time.sleep(1)\n"
        )
        leader = self.root / "timeout-process-group.py"
        leader.write_text(
            "#!/usr/bin/env python3\n"
            "import os, signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "def stop(_signum, _frame): raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            f"descendant_source = {descendant_source!r}\n"
            "subprocess.Popen(\n"
            "    [sys.executable, '-c', descendant_source],\n"
            "    stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL,\n"
            "    stderr=subprocess.DEVNULL,\n"
            "    close_fds=True,\n"
            ")\n"
            f"descendant_marker = Path({str(descendant_marker)!r})\n"
            "deadline = time.monotonic() + 5\n"
            "while not descendant_marker.exists():\n"
            "    if time.monotonic() >= deadline: raise RuntimeError("
            "'descendant did not start')\n"
            "    time.sleep(0.01)\n"
            f"Path({str(leader_marker)!r}).write_text(str(os.getpid()))\n"
            "print('leader-timeout-output', flush=True)\n"
            "print('leader-timeout-error', file=sys.stderr, flush=True)\n"
            "while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        leader.chmod(0o755)
        environment = self.environment()
        stdout = io.StringIO()
        stderr = io.StringIO()

        started = time.monotonic()
        with mock.patch.dict(os.environ, environment, clear=True):
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(stderr):
                    result = WRAPPER.main(
                        [
                            "--backend",
                            "cpu",
                            "--executable",
                            str(leader),
                        ],
                        execution_timeout=1.0,
                        terminate_timeout=0.2,
                        kill_timeout=1.0,
                    )
        elapsed = time.monotonic() - started

        self.assertEqual(result, 124, stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertLess(elapsed, 3.0)
        leader_pid = int(leader_marker.read_text(encoding="utf-8"))
        descendant_pid = int(
            descendant_marker.read_text(encoding="utf-8")
        )
        for pid in (leader_pid, descendant_pid):
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        evidence = json.loads(self.evidence.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["returncode"], 0)
        self.assertEqual(evidence["stdout"], "leader-timeout-output\n")
        self.assertEqual(evidence["stderr"], "leader-timeout-error\n")
        self.assertEqual(
            evidence["supervision"],
            {
                "execution_timeout_seconds": 1.0,
                "terminate_grace_seconds": 0.2,
                "kill_reap_seconds": 1.0,
                "timed_out": True,
                "signals_sent": ["SIGTERM", "SIGKILL"],
                "reap": "after-SIGKILL",
            },
        )

    def test_post_sigkill_collection_and_wait_share_one_deadline(self) -> None:
        class DeadlineProcess:
            pid = 12345
            stdout = None
            stderr = None

            def __init__(self) -> None:
                self.communicate_calls = 0
                self.wait_timeouts: list[float] = []

            def communicate(
                self, timeout: float
            ) -> tuple[str, str]:
                self.communicate_calls += 1
                if self.communicate_calls == 1:
                    raise subprocess.TimeoutExpired("child", timeout)
                time.sleep(timeout + 0.03)
                raise subprocess.TimeoutExpired(
                    "child", timeout, output="", stderr=""
                )

            def wait(self, timeout: float) -> int:
                self.wait_timeouts.append(timeout)
                time.sleep(timeout)
                return 0

        process = DeadlineProcess()
        supervision = {"signals_sent": [], "reap": "normal"}

        def record_signal(
            _process: object,
            requested: object,
            details: dict[str, object],
        ) -> None:
            details["signals_sent"].append(requested.name)

        started = time.monotonic()
        with mock.patch.object(
            WRAPPER, "_signal_process_group", side_effect=record_signal
        ):
            with self.assertRaises(WRAPPER.ProcessFailure):
                WRAPPER._terminate_and_reap(
                    process,
                    supervision,
                    terminate_timeout=0.01,
                    kill_timeout=0.2,
                )
        elapsed = time.monotonic() - started
        self.assertEqual(process.wait_timeouts, [])
        self.assertEqual(supervision["reap"], "failed-after-SIGKILL")
        self.assertLess(elapsed, 0.35)
    def test_supervision_uses_official_default_and_bounded_term_kill(self) -> None:
        completed, normal = WRAPPER.run_backend_process(
            [sys.executable, "-c", "pass"], os.environ.copy()
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(
            normal["execution_timeout_seconds"],
            WRAPPER.EXECUTION_TIMEOUT_SECONDS,
        )
        self.assertEqual(WRAPPER.EXECUTION_TIMEOUT_SECONDS, 3600.0)

        sleeper = self.root / "ignore-term.py"
        sleeper.write_text(
            "#!/usr/bin/env python3\n"
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        sleeper.chmod(0o755)
        started = time.monotonic()
        timed, supervision = WRAPPER.run_backend_process(
            [str(sleeper)],
            os.environ.copy(),
            execution_timeout=0.1,
            terminate_timeout=0.1,
            kill_timeout=0.5,
        )
        elapsed = time.monotonic() - started
        self.assertTrue(supervision["timed_out"])
        self.assertIn("SIGTERM", supervision["signals_sent"])
        self.assertIn("SIGKILL", supervision["signals_sent"])
        self.assertEqual(supervision["reap"], "after-SIGKILL")
        self.assertLess(elapsed, 1.0)
        self.assertLess(timed.returncode, 0)


if __name__ == "__main__":
    unittest.main()

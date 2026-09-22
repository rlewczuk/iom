#!/usr/bin/env python3
"""Launch one opt-in official IOM inference test after offline verification.

This wrapper is deliberately standard-library only.  It verifies the caller's
model directory and reference pack, then invokes exactly one already-built
backend test through an argv list.  It never downloads, discovers a default
model, runs a shell, or substitutes another backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


GEOMETRY = {
    "num_hidden_layers": 22,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
}
TOLERANCES = {
    "absolute": 0.53125,
    "relative": 0.02,
    "formula": "abs(actual-ref) <= 0.53125 + 0.02*abs(ref)",
    "tie_policy": "lowest-id",
    "exact_token": "reference-margin-certified-only",
}
REQUIRED_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
FORCED_IDS = [3, 4, 5, 6]
EXPECTED_CASES = [
    "raw-production-greedy",
    "raw-fixed-reference-continuation",
    "chat-production-greedy",
    "chat-fixed-reference-continuation",
    "zero-new-token",
]

PINNED_PACKAGES = {
    "jinja2": "3.1.6",
    "numpy": "1.26.4",
    "safetensors": "0.4.3",
    "sentencepiece": "0.2.0",
    "tokenizers": "0.19.1",
    "torch": "2.3.1+cpu",
    "transformers": "4.41.2",
}
PINNED_RUNTIME = {"implementation": "CPython", "version": "3.12.3"}
PINNED_PRECISION = {
    "weights": "BF16",
    "activations": "BF16",
    "attention": "eager",
    "logits_storage": "BF16",
    "reference_arithmetic": "FP32",
    "device": "cpu",
}

EXECUTION_TIMEOUT_SECONDS = 3600.0
TERMINATION_GRACE_SECONDS = 30.0
KILL_REAP_SECONDS = 30.0
MEASUREMENT_ENVIRONMENT = (
    "IOM_TEST_MODEL_DIR",
    "IOM_TEST_MODEL_ID",
    "IOM_TEST_MODEL_REFERENCE",
    "IOM_TEST_MODEL_EVIDENCE",
    "IOM_TEST_MODEL_ARENA_BYTES",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
)


class HarnessError(RuntimeError):
    pass


class ProcessFailure(HarnessError):
    def __init__(
        self,
        stage: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = dict(details or {})


def fail(message: str) -> None:
    raise HarnessError(message)


def reject_constant(value: str) -> Any:
    fail(f"non-finite JSON constant {value!r} is forbidden")


def load_json(path: Path, label: str) -> Any:
    try:
        payload = path.read_bytes()
        return json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except HarnessError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {label} {path}: {error}")
    raise AssertionError("unreachable")


def object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{label} must be an object")
    return dict(value)


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty string")
    return value


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} must be an integer >= {minimum}")
    return value


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{label} must be a boolean")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be a finite number")
    result = float(value)
    if not (result == result and abs(result) != float("inf")):
        fail(f"{label} must be finite")
    return result


def valid_relative_path(value: Any, label: str) -> str:
    path = nonempty_string(value, label)
    parsed = Path(path)
    if parsed.is_absolute() or not parsed.parts or any(
        part in ("", ".", "..") for part in parsed.parts
    ):
        fail(f"{label} must be a normalized relative path")
    return parsed.as_posix()


def model_root(path: Path) -> Path:
    try:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            fail(f"caller model path {path} must be a directory")
        return path.resolve(strict=True)
    except HarnessError:
        raise
    except OSError as error:
        fail(f"caller model directory {path} is unavailable: {error}")
    raise AssertionError("unreachable")


def artifact_path(root: Path, relative: str) -> Path:
    candidate = root / Path(valid_relative_path(relative, "artifact path"))
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        fail(f"artifact {relative!r} is unavailable: {error}")
    if resolved != root and root not in resolved.parents:
        fail(f"artifact {relative!r} escapes the caller model directory")
    return candidate


def file_identity(path: Path, relative: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"cannot inspect artifact {relative!r}: {error}")
    if not stat.S_ISREG(info.st_mode):
        fail(f"artifact {relative!r} must be a regular file")
    digest = hashlib.sha256()
    count = 0
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                fail(f"artifact {relative!r} changed to a non-regular file")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
            final_size = os.fstat(stream.fileno()).st_size
    except HarnessError:
        raise
    except OSError as error:
        fail(f"cannot read artifact {relative!r}: {error}")
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if count != final_size:
        fail(f"artifact {relative!r} changed while hashing")
    return {"path": relative, "size": count, "sha256": digest.hexdigest()}


def model_artifact_paths(root: Path) -> list[tuple[str, Path]]:
    try:
        names = [entry.name for entry in root.iterdir()]
    except OSError as error:
        fail(f"cannot enumerate caller model directory {root}: {error}")
    shards = sorted(name for name in names if name.endswith(".safetensors"))
    if not shards:
        fail("caller model directory contains no SafeTensors shard")
    names_set = set(REQUIRED_MODEL_FILES) | set(shards)
    index = root / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        names_set.add(index.name)
    return [
        (name, artifact_path(root, name))
        for name in sorted(names_set)
    ]


def model_artifacts(root: Path) -> list[dict[str, Any]]:
    return [
        file_identity(path, name)
        for name, path in model_artifact_paths(root)
    ]


def validate_config(root: Path, pack_config: Mapping[str, Any]) -> None:
    config = object_value(load_json(root / "config.json", "model config"), "model config")
    if config.get("model_type") != "llama":
        fail("model config model_type must be 'llama'")
    actual = dict(config)
    actual["head_dim"] = config.get("hidden_size", 0) // max(
        1, config.get("num_attention_heads", 1)
    )
    actual["torch_dtype"] = (
        "bfloat16" if config.get("torch_dtype") in ("bfloat16", "torch.bfloat16")
        else config.get("torch_dtype")
    )
    for field, expected in GEOMETRY.items():
        if actual.get(field) != expected:
            fail(
                f"model config {field} mismatch: expected {expected}, "
                f"actual {actual.get(field)!r}"
            )
    if config.get("tie_word_embeddings") is not False:
        fail("model config must use untied BF16 lm_head")
    if actual.get("torch_dtype") != "bfloat16":
        fail("model config must declare BF16 torch_dtype")
    expected_pack = dict(pack_config)
    for field, expected in {
        **GEOMETRY,
        "model_type": "llama",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }.items():
        if field == "tie_word_embeddings":
            if boolean(
                expected_pack.get(field),
                "reference model_config tie_word_embeddings",
            ) is not expected:
                fail(
                    "reference model_config tie_word_embeddings does not "
                    "match official geometry"
                )
        elif expected_pack.get(field) != expected:
            fail(
                f"reference model_config {field} does not match official "
                "geometry"
            )
        if actual.get(field) != expected:
            fail(f"caller model_config {field} does not match official geometry")
    declared_model_id = config.get(
        "model_id", config.get("_name_or_path", config.get("name_or_path"))
    )
    if (
        "model_id" in expected_pack
        and declared_model_id is not None
        and expected_pack["model_id"] != declared_model_id
    ):
        fail("reference model_config model_id conflicts with the caller model config")
    for field, value in expected_pack.items():
        if field in GEOMETRY or field in (
            "model_type",
            "tie_word_embeddings",
            "torch_dtype",
            "model_id",
        ):
            continue
        if field not in actual or actual[field] != value:
            fail(f"reference model_config {field} does not match caller config")


def token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        fail(f"{label} must be a token-ID list")
    result = []
    for index, item in enumerate(value):
        item = integer(item, f"{label}[{index}]")
        if item >= GEOMETRY["vocab_size"]:
            fail(f"{label}[{index}] is outside the official vocabulary")
        result.append(item)
    return result


def check_snapshot(value: Any, label: str, prefix: Sequence[int], index: int) -> None:
    snapshot = object_value(value, label)
    expected_phase = "prefill" if index == 0 else "cached-decode"
    expected_start = 0 if index == 0 else len(prefix) - 1
    expected_length = len(prefix) if index == 0 else 1
    if snapshot.get("phase") != expected_phase:
        fail(f"{label}.phase mismatch")
    if token_ids(snapshot.get("prefix_ids"), f"{label}.prefix_ids") != list(prefix):
        fail(f"{label}.prefix_ids mismatch")
    if integer(snapshot.get("position_start"), f"{label}.position_start") != expected_start:
        fail(f"{label}.position_start mismatch")
    if integer(snapshot.get("run_length"), f"{label}.run_length", 1) != expected_length:
        fail(f"{label}.run_length mismatch")
    if "absolute_positions" in snapshot and token_ids(
        snapshot["absolute_positions"], f"{label}.absolute_positions"
    ) != list(range(expected_start, expected_start + expected_length)):
        fail(f"{label}.absolute_positions mismatch")
    logits = snapshot.get("logits")
    if not isinstance(logits, list) or len(logits) != GEOMETRY["vocab_size"]:
        fail(f"{label}.logits must contain the full official vocabulary")
    for position, item in enumerate(logits):
        finite(item, f"{label}.logits[{position}]")
    greedy = integer(snapshot.get("greedy_id"), f"{label}.greedy_id")
    if greedy >= GEOMETRY["vocab_size"]:
        fail(f"{label}.greedy_id is outside the official vocabulary")
    winner = max(range(len(logits)), key=lambda item: (float(logits[item]), -item))
    if greedy != winner:
        fail(f"{label}.greedy_id is not lowest-ID argmax")
    winner_error = 0.53125 + 0.02 * abs(float(logits[winner]))
    stable = all(
        float(logits[winner]) - winner_error
        > float(logits[other]) + 0.53125 + 0.02 * abs(float(logits[other]))
        for other in range(len(logits)) if other != winner
    )
    if snapshot.get("margin_stable") is not stable:
        fail(f"{label}.margin_stable is not reference-derived")


def validate_cases(cases: Any) -> None:
    if not isinstance(cases, list):
        fail("reference cases must be a list")
    expected_names = [
        "raw-production-greedy",
        "raw-fixed-reference-continuation",
        "chat-production-greedy",
        "chat-fixed-reference-continuation",
        "zero-new-token",
    ]
    if len(cases) != len(expected_names):
        fail("reference cases must contain the fixed five-case order")
    for case_index, case_value in enumerate(cases):
        case = object_value(case_value, f"case[{case_index}]")
        required = {"id", "input", "snapshots", "expected_result"}
        if set(case) != required:
            fail(f"case[{case_index}] fields are malformed")
        name = nonempty_string(case["id"], f"case[{case_index}].id")
        if name != expected_names[case_index]:
            fail(f"case[{case_index}] id/order mismatch")
        input_record = object_value(case["input"], f"case {name}.input")
        mode = input_record.get("mode")
        expected_input = {
            "mode",
            "selection_policy",
            "prompt_ids",
            "rendered_utf8",
            "positions",
            "decode_ids",
            "max_new_tokens",
            "bos_policy",
            "text" if mode == "raw" else "messages",
        }
        if set(input_record) != expected_input:
            fail(f"case {name}.input fields are malformed")
        if mode not in ("raw", "chat"):
            fail(f"case {name} has unsupported mode")
        if (name.startswith("chat-") and mode != "chat") or (
            name.startswith("raw-") and mode != "raw"
        ):
            fail(f"case {name} mode does not match its fixed name")
        prompt = token_ids(input_record["prompt_ids"], f"case {name}.input.prompt_ids")
        positions = token_ids(
            input_record["positions"], f"case {name}.input.positions"
        )
        if positions != list(range(len(prompt))):
            fail(f"case {name} positions are not contiguous")
        decode = token_ids(input_record["decode_ids"], f"case {name}.input.decode_ids")
        max_new = integer(input_record["max_new_tokens"], f"case {name}.input.max_new_tokens")
        policy = input_record["selection_policy"]
        if policy not in ("production-greedy", "fixed-reference-continuation", "none"):
            fail(f"case {name} selection policy is unsupported")
        bos_policy = object_value(input_record["bos_policy"], f"case {name}.input.bos_policy")
        if set(bos_policy) != {"add_special_tokens", "require_bos", "bos_token_id"}:
            fail(f"case {name} BOS policy fields are malformed")
        add_special_tokens = boolean(
            bos_policy["add_special_tokens"],
            f"case {name}.input.bos_policy.add_special_tokens",
        )
        require_bos = boolean(
            bos_policy["require_bos"],
            f"case {name}.input.bos_policy.require_bos",
        )
        bos_token_id = integer(
            bos_policy["bos_token_id"],
            f"case {name}.input.bos_policy.bos_token_id",
        )
        if bos_token_id != 1:
            fail(f"case {name} has an unsupported BOS token ID")
        rendered = input_record["rendered_utf8"]
        if not isinstance(rendered, list) or any(
            integer(item, f"case {name} rendered byte") > 255 for item in rendered
        ):
            fail(f"case {name} rendered_utf8 is malformed")
        if mode == "raw":
            if input_record["text"] != "The capital of France is":
                fail(f"case {name} raw text differs from policy")
            if rendered != list(input_record["text"].encode("utf-8")):
                fail(f"case {name} raw bytes differ from policy")
            if add_special_tokens is not True or require_bos is not True:
                fail(f"case {name} raw BOS policy mismatch")
            if not prompt or prompt[0] != 1:
                fail(f"case {name} raw prompt does not record the required BOS")
        else:
            if input_record["messages"] != [{"role": "user", "content": "Hello."}]:
                fail(f"case {name} chat messages differ from policy")
            if bytes(rendered).decode("utf-8") != "<|user|>\nHello.</s>\n<|assistant|>\n":
                fail(f"case {name} chat bytes differ from policy")
            if add_special_tokens is not False or require_bos is not False:
                fail(f"case {name} chat BOS policy mismatch")
            if not prompt or prompt[0] == 1:
                fail(f"case {name} chat prompt unexpectedly has a BOS")
        if name == "zero-new-token":
            if policy != "none" or decode or max_new != 0:
                fail("zero-new-token case policy is malformed")
        elif name.endswith("fixed-reference-continuation"):
            if policy != "fixed-reference-continuation" or decode != FORCED_IDS or max_new != 4:
                fail(f"case {name} fixed continuation is malformed")
        else:
            if policy != "production-greedy" or not decode or len(decode) > 4 or max_new != 4:
                fail(f"case {name} production policy is malformed")
            if 2 in decode[:-1]:
                fail(f"case {name} contains output after EOS")
        snapshots = case["snapshots"]
        if not isinstance(snapshots, list) or len(snapshots) != len(decode):
            fail(f"case {name} snapshot count does not match decode IDs")
        for index, snapshot_value in enumerate(snapshots):
            snapshot = object_value(snapshot_value, f"case {name}.snapshots[{index}]")
            if set(snapshot) != {
                "phase",
                "prefix_ids",
                "position_start",
                "run_length",
                "absolute_positions",
                "logits",
                "greedy_id",
                "selected_id",
                "margin_stable",
            }:
                fail(f"case {name}.snapshots[{index}] fields are malformed")
            check_snapshot(
                snapshot,
                f"case {name}.snapshots[{index}]",
                prompt + decode[:index],
                index,
            )
            if integer(snapshot["selected_id"], f"case {name}.selected_id") != decode[index]:
                fail(f"case {name}.selected_id mismatch")
        expected = object_value(case["expected_result"], f"case {name}.expected_result")
        if set(expected) != {"token_ids", "stop_reason", "initialized_kv_length"}:
            fail(f"case {name}.expected_result fields are malformed")
        expected_token_ids = token_ids(
            expected["token_ids"], f"case {name}.expected_result.token_ids"
        )
        if expected_token_ids != decode:
            fail(f"case {name} expected token IDs mismatch")
        if name == "zero-new-token":
            expected_stop = "max_new_tokens"
        elif policy == "fixed-reference-continuation":
            expected_stop = "max_new_tokens"
        elif decode[-1] == 2:
            expected_stop = "eos"
        elif len(decode) == max_new:
            expected_stop = "max_new_tokens"
        elif len(prompt) + len(decode) >= GEOMETRY["max_position_embeddings"]:
            expected_stop = "context_exhaustion"
        else:
            fail(f"case {name} has no valid stop condition")
        if expected["stop_reason"] != expected_stop:
            fail(f"case {name} stop reason mismatch")
        expected_kv = 0 if not decode else len(prompt) + len(decode) - 1
        initialized_kv_length = integer(
            expected["initialized_kv_length"],
            f"case {name}.expected_result.initialized_kv_length",
        )
        if initialized_kv_length != expected_kv:
            fail(f"case {name} initialized KV state mismatch")

def validate_provenance(
    provenance: Mapping[str, Any], artifact_id: str, model_id: str
) -> None:
    runtime = object_value(provenance.get("runtime"), "official provenance runtime")
    if runtime != PINNED_RUNTIME:
        fail("official provenance runtime pin mismatch")
    packages = object_value(provenance.get("packages"), "official provenance packages")
    if set(packages) != set(PINNED_PACKAGES):
        fail("official provenance package set mismatch")
    for package, expected in PINNED_PACKAGES.items():
        actual = nonempty_string(
            packages.get(package), f"official provenance package {package}"
        )
        if actual != expected:
            fail(f"official provenance package {package} is not pinned to {expected}")
    if object_value(provenance.get("precision"), "official provenance precision") != PINNED_PRECISION:
        fail("official provenance precision policy mismatch")
    digest = nonempty_string(
        provenance.get("generator_sha256"),
        "official provenance generator_sha256",
    )
    if len(digest) != 64 or digest.lower() != digest or any(
        character not in "0123456789abcdef" for character in digest
    ):
        fail("official provenance generator_sha256 is not a lowercase SHA-256")
    command = provenance.get("command")
    if not isinstance(command, list) or not command or any(
        not isinstance(value, str) or not value for value in command
    ):
        fail("official provenance command must be a non-empty argv list")
    if provenance.get("model_id") != model_id or provenance.get("revision") != artifact_id:
        fail("official provenance model/revision identity mismatch")


def validate_reference(reference_path: Path, model_dir: Path, artifact_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pack = object_value(load_json(reference_path, "official reference pack"), "official reference pack")
    if set(pack) != {"schema_version", "kind", "provenance", "tolerances", "model_config", "artifact_id", "cases"}:
        fail("official reference pack has an unexpected schema")
    if (
        integer(pack.get("schema_version"), "official schema_version") != 1
        or pack.get("kind") != "official"
    ):
        fail("official reference pack schema_version/kind mismatch")
    if pack.get("artifact_id") != artifact_id:
        fail("official artifact ID differs from the reference pack")
    if pack.get("tolerances") != TOLERANCES:
        fail("official comparison policy differs from the frozen policy")
    provenance = object_value(pack.get("provenance"), "official provenance")
    digest = nonempty_string(provenance.get("case_payload_sha256"), "case payload digest")
    canonical = json.dumps(pack["cases"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if digest != actual_digest:
        fail(f"case payload digest mismatch: expected {digest}, actual {actual_digest}")
    records = provenance.get("artifacts")
    if not isinstance(records, list) or not records:
        fail("official provenance artifacts are missing")
    expected: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(records):
        record = object_value(value, f"artifact[{index}]")
        if set(record) != {"path", "size", "sha256"}:
            fail(f"artifact[{index}] fields are malformed")
        relative = valid_relative_path(record.get("path"), f"artifact[{index}].path")
        if relative in expected:
            fail(f"duplicate official artifact {relative}")
        size = integer(record.get("size"), f"artifact {relative}.size")
        sha = record.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or sha.lower() != sha or any(char not in "0123456789abcdef" for char in sha):
            fail(f"artifact {relative} has an invalid SHA-256")
        expected[relative] = {"path": relative, "size": size, "sha256": sha}
    actual = model_artifacts(model_dir)
    if {record["path"] for record in actual} != set(expected):
        fail("official artifact inventory differs from the caller model directory")
    for record in actual:
        if record != expected[record["path"]]:
            fail(f"artifact identity mismatch for {record['path']}")
    pack_config = object_value(pack.get("model_config"), "pack model_config")
    validate_config(model_dir, pack_config)
    model_id = nonempty_string(pack_config.get("model_id"), "pack model_config.model_id")
    validate_provenance(provenance, artifact_id, model_id)
    validate_cases(pack.get("cases"))
    return pack, actual


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        fail(f"{name} must be set explicitly")
    return value


def evidence_path(value: str) -> Path:
    path = Path(value)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        fail(f"IOM_TEST_MODEL_EVIDENCE must name a regular file when present")
    if not path.parent.exists() or not path.parent.is_dir():
        fail(f"evidence parent does not exist: {path.parent}")
    return path


def normalized_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def resolved_if_available(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        return normalized_absolute(path)


def available_identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_dev, info.st_ino)


class EvidencePublication:
    def __init__(
        self,
        path: Path,
        model_dir: Path,
        reference: Path,
        executable: Path,
    ) -> None:
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp")
        self.destination_lexical = normalized_absolute(path)
        self.destination_resolved = resolved_if_available(path)
        self.temporary_lexical = normalized_absolute(self.temporary)
        self.temporary_resolved = resolved_if_available(self.temporary)
        self.model_dir = model_dir
        self.model_root_lexical = normalized_absolute(model_dir)
        self.model_root_resolved = resolved_if_available(model_dir)
        self.protected: dict[
            Path, tuple[str, Path, tuple[int, int] | None]
        ] = {}
        self.add_protected(reference, "reference")
        self.add_protected(executable, "backend executable")
        for name in (
            *sorted(REQUIRED_MODEL_FILES),
            "model.safetensors",
            "model.safetensors.index.json",
        ):
            self.add_protected(model_dir / name, f"model artifact {name}")
        self._retain_current_model_entries()
        self.validate()

    def add_protected(self, path: Path, label: str) -> None:
        lexical = normalized_absolute(path)
        if lexical not in self.protected:
            self.protected[lexical] = (
                label,
                resolved_if_available(path),
                available_identity(path),
            )

    def _retain_current_model_entries(self) -> None:
        try:
            entries = list(self.model_dir.iterdir())
        except OSError:
            return
        for entry in entries:
            self.add_protected(entry, f"model artifact {entry.name}")

    def _validate_candidate(
        self,
        candidate: Path,
        retained_lexical: Path,
        retained_resolved: Path,
    ) -> None:
        lexical = normalized_absolute(candidate)
        resolved = resolved_if_available(candidate)
        if (
            lexical == self.model_root_lexical
            or self.model_root_lexical in lexical.parents
            or retained_lexical == self.model_root_lexical
            or self.model_root_lexical in retained_lexical.parents
            or resolved == self.model_root_resolved
            or self.model_root_resolved in resolved.parents
            or retained_resolved == self.model_root_resolved
            or self.model_root_resolved in retained_resolved.parents
        ):
            fail(
                "IOM_TEST_MODEL_EVIDENCE must be outside the caller model "
                "directory"
            )
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            identity = None
        except OSError as error:
            fail(f"cannot inspect evidence destination {candidate}: {error}")
        else:
            if not stat.S_ISREG(info.st_mode):
                fail(
                    "evidence destination must name a regular non-symlink "
                    "file when present"
                )
            identity = (info.st_dev, info.st_ino)

        for protected_lexical, (
            label,
            protected_resolved,
            retained_identity,
        ) in self.protected.items():
            protected_identity = retained_identity
            protected_path = Path(protected_lexical)
            current_identity = available_identity(protected_path)
            current_resolved = resolved_if_available(protected_path)
            if current_identity is not None:
                protected_identity = current_identity
            if (
                lexical == protected_lexical
                or retained_lexical == protected_lexical
                or resolved == protected_resolved
                or retained_resolved == protected_resolved
                or resolved == current_resolved
                or retained_resolved == current_resolved
                or (
                    identity is not None
                    and protected_identity is not None
                    and identity == protected_identity
                )
                or (
                    identity is not None
                    and retained_identity is not None
                    and identity == retained_identity
                )
            ):
                fail(
                    "IOM_TEST_MODEL_EVIDENCE aliases protected "
                    f"{label}"
                )

    def validate(self) -> None:
        self._retain_current_model_entries()
        self._validate_candidate(
            self.path,
            self.destination_lexical,
            self.destination_resolved,
        )
        self._validate_candidate(
            self.temporary,
            self.temporary_lexical,
            self.temporary_resolved,
        )

    def clear_stale(self) -> None:
        self.validate()
        try:
            self.path.unlink(missing_ok=True)
        except OSError as error:
            fail(f"cannot clear prior evidence {self.path}: {error}")


def write_evidence(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except OSError:
            pass
        fail(f"cannot write evidence {path}: {error}")

def validate_arena_bytes() -> None:
    value = os.environ.get("IOM_TEST_MODEL_ARENA_BYTES")
    if value is None:
        return
    if not value or not value.isascii() or not value.isdecimal():
        fail(
            "IOM_TEST_MODEL_ARENA_BYTES must be a positive decimal byte "
            "count divisible by 32"
        )
    size = int(value)
    if size <= 0 or size % 32:
        fail(
            "IOM_TEST_MODEL_ARENA_BYTES must be a positive decimal byte "
            "count divisible by 32"
        )


def validate_measurement(
    value: Any,
    backend: str,
    artifact_id: str,
    reference_identity: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    selected_environment: Mapping[str, str],
) -> dict[str, Any]:
    measurement = object_value(value, "backend measurement")
    if measurement.get("schema_version") != 1:
        fail("backend measurement schema_version mismatch")
    if measurement.get("kind") != "official-inference-evidence":
        fail("backend measurement kind mismatch")
    if measurement.get("status") != "passed":
        fail("backend measurement did not report passed execution")
    if measurement.get("backend") != backend:
        fail(
            f"backend measurement selected {measurement.get('backend')!r}, "
            f"not requested backend {backend!r}"
        )
    reference = object_value(
        measurement.get("reference"), "backend measurement reference"
    )
    for field in ("size", "sha256"):
        if reference.get(field) != reference_identity[field]:
            fail(f"backend measurement reference {field} mismatch")
    if reference.get("artifact_id") != artifact_id:
        fail("backend measurement artifact ID mismatch")
    expected_artifacts = {
        record["path"]: dict(record)
        for record in artifacts
    }
    measured_records = reference.get("artifacts")
    if not isinstance(measured_records, list):
        fail("backend measurement artifact records are missing")
    measured_artifacts = {}
    for index, record_value in enumerate(measured_records):
        record = object_value(
            record_value, f"backend measurement artifact[{index}]"
        )
        path = record.get("path")
        if not isinstance(path, str) or path in measured_artifacts:
            fail("backend measurement artifact records are malformed")
        measured_artifacts[path] = record
    if measured_artifacts != expected_artifacts:
        fail("backend measurement artifact identity mismatch")
    environment = object_value(
        measurement.get("environment"), "backend measurement environment"
    )
    if environment != dict(selected_environment):
        fail("backend measurement environment differs from launched values")
    cases = measurement.get("cases")
    if not isinstance(cases, list) or [
        object_value(case, "backend measurement case").get("id")
        for case in cases
    ] != EXPECTED_CASES:
        fail("backend measurement does not contain all fixed logical cases")
    return measurement


def _signal_process_group(
    process: subprocess.Popen[str],
    requested: signal.Signals,
    supervision: dict[str, Any],
) -> bool:
    # start_new_session=True makes the leader PID the original process-group
    # ID. Keep using that immutable value even after the leader exits; never
    # discover or signal a replacement group through a surviving descendant.
    try:
        os.killpg(process.pid, requested)
        supervision["signals_sent"].append(requested.name)
        return True
    except ProcessLookupError:
        supervision["signals_sent"].append(f"{requested.name}:already-exited")
        return False
    except OSError as error:
        raise ProcessFailure(
            "termination",
            f"cannot send {requested.name} to backend process group: {error}",
            supervision,
        ) from error

def _process_group_exists(
    process: subprocess.Popen[str],
    supervision: dict[str, Any],
) -> bool:
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        raise ProcessFailure(
            "reap",
            f"cannot query backend process group after termination: {error}",
            supervision,
        ) from error


def _wait_for_process_group_exit(
    process: subprocess.Popen[str],
    supervision: dict[str, Any],
    kill_deadline: float,
    kill_timeout: float,
) -> None:
    while _process_group_exists(process, supervision):
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            supervision["reap"] = "failed-after-SIGKILL"
            raise ProcessFailure(
                "reap",
                "backend process group did not disappear within the single "
                f"bounded post-SIGKILL {kill_timeout:g}-second interval",
                supervision,
            )
        time.sleep(min(0.01, remaining))


def _close_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _terminate_and_reap(
    process: subprocess.Popen[str],
    supervision: dict[str, Any],
    terminate_timeout: float,
    kill_timeout: float,
) -> tuple[str, str]:
    _signal_process_group(process, signal.SIGTERM, supervision)
    try:
        stdout, stderr = process.communicate(timeout=terminate_timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass
    else:
        if not _process_group_exists(process, supervision):
            supervision["reap"] = "after-SIGTERM"
            return stdout, stderr
        kill_deadline = time.monotonic() + kill_timeout
        kill_sent = _signal_process_group(
            process, signal.SIGKILL, supervision
        )
        _wait_for_process_group_exit(
            process, supervision, kill_deadline, kill_timeout
        )
        supervision["reap"] = (
            "after-SIGKILL" if kill_sent else "after-SIGTERM"
        )
        return stdout, stderr

    kill_deadline = time.monotonic() + kill_timeout
    kill_sent = _signal_process_group(
        process, signal.SIGKILL, supervision
    )
    try:
        stdout, stderr = process.communicate(
            timeout=max(0.0, kill_deadline - time.monotonic())
        )
        _wait_for_process_group_exit(
            process, supervision, kill_deadline, kill_timeout
        )
        supervision["reap"] = (
            "after-SIGKILL" if kill_sent else "after-SIGTERM"
        )
        return stdout, stderr
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_output(error.stdout)
        stderr = _timeout_output(error.stderr)
        _close_process_pipes(process)
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            supervision["reap"] = "failed-after-SIGKILL"
            raise ProcessFailure(
                "reap",
                "backend process did not reap within the single bounded "
                f"post-SIGKILL {kill_timeout:g}-second interval",
                supervision,
            ) from error
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as wait_error:
            supervision["reap"] = "failed-after-SIGKILL"
            raise ProcessFailure(
                "reap",
                "backend process did not reap within the single bounded "
                f"post-SIGKILL {kill_timeout:g}-second interval",
                supervision,
            ) from wait_error
        _wait_for_process_group_exit(
            process, supervision, kill_deadline, kill_timeout
        )
        supervision["reap"] = "pipes-closed-after-SIGKILL"
        return stdout, stderr
    except OSError as error:
        _close_process_pipes(process)
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            supervision["reap"] = "failed-after-collection-error"
            raise ProcessFailure(
                "reap",
                "backend process did not reap after a collection error within "
                f"the single {kill_timeout:g}-second post-SIGKILL interval: "
                f"{error}",
                supervision,
            ) from error
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as wait_error:
            supervision["reap"] = "failed-after-collection-error"
            raise ProcessFailure(
                "reap",
                "backend process did not reap after a collection error within "
                f"the single {kill_timeout:g}-second post-SIGKILL interval: "
                f"{error}",
                supervision,
            ) from wait_error
        _wait_for_process_group_exit(
            process, supervision, kill_deadline, kill_timeout
        )
        supervision["reap"] = "pipes-closed-after-collection-error"
        return "", str(error)


def run_backend_process(
    command: Sequence[str],
    environment: Mapping[str, str],
    execution_timeout: float = EXECUTION_TIMEOUT_SECONDS,
    terminate_timeout: float = TERMINATION_GRACE_SECONDS,
    kill_timeout: float = KILL_REAP_SECONDS,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    if execution_timeout <= 0 or terminate_timeout <= 0 or kill_timeout <= 0:
        raise ValueError("backend supervision timeouts must be positive")
    supervision: dict[str, Any] = {
        "execution_timeout_seconds": execution_timeout,
        "terminate_grace_seconds": terminate_timeout,
        "kill_reap_seconds": kill_timeout,
        "timed_out": False,
        "signals_sent": [],
        "reap": "normal",
    }
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(environment),
            start_new_session=True,
        )
    except OSError as error:
        raise ProcessFailure(
            "launch", f"cannot launch backend test: {error}", supervision
        ) from error
    try:
        stdout, stderr = process.communicate(timeout=execution_timeout)
    except subprocess.TimeoutExpired:
        supervision["timed_out"] = True
        stdout, stderr = _terminate_and_reap(
            process, supervision, terminate_timeout, kill_timeout
        )
    except OSError as error:
        try:
            _terminate_and_reap(
                process, supervision, terminate_timeout, kill_timeout
            )
        except ProcessFailure as cleanup_error:
            raise ProcessFailure(
                "collection",
                f"backend output collection failed: {error}; "
                f"cleanup also failed: {cleanup_error}",
                cleanup_error.details,
            ) from error
        raise ProcessFailure(
            "collection",
            f"backend output collection failed and the child was reaped: {error}",
            supervision,
        ) from error
    return (
        subprocess.CompletedProcess(
            list(command), process.returncode, stdout, stderr
        ),
        supervision,
    )


def _failure_payload(
    backend: str,
    stage: str,
    diagnostic: str,
    context: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "official-inference-wrapper-evidence",
        "status": "failed",
        "backend": backend,
        "stage": stage,
        "diagnostics": [diagnostic],
        "native_capability": "unclaimed",
    }
    payload.update(context)
    if details:
        payload["supervision"] = dict(details)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="verify and launch one official IOM inference backend test")
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "sycl"), required=True)
    parser.add_argument("--executable", required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    execution_timeout: float = EXECUTION_TIMEOUT_SECONDS,
    terminate_timeout: float = TERMINATION_GRACE_SECONDS,
    kill_timeout: float = KILL_REAP_SECONDS,
) -> int:
    args = parse_args(argv)
    evidence: Path | None = None
    publication: EvidencePublication | None = None
    stage = "environment"
    failure_context: dict[str, Any] = {}
    try:
        model_dir_value = required_environment("IOM_TEST_MODEL_DIR")
        artifact_id = required_environment("IOM_TEST_MODEL_ID")
        reference_value = required_environment("IOM_TEST_MODEL_REFERENCE")
        evidence_candidate = evidence_path(
            required_environment("IOM_TEST_MODEL_EVIDENCE")
        )
        model_dir_candidate = Path(model_dir_value)
        reference = Path(reference_value)
        executable = Path(args.executable)
        command = [
            str(Path(args.executable)),
            "--test-case=*real model inference*",
        ]
        selected_environment = {
            name: os.environ[name]
            for name in MEASUREMENT_ENVIRONMENT
            if name in os.environ
        }
        failure_context.update(
            {
                "artifact_id": artifact_id,
                "argv": command,
                "environment": selected_environment,
            }
        )
        stage = "input-validation"
        publication = EvidencePublication(
            evidence_candidate,
            model_dir_candidate,
            reference,
            executable,
        )
        evidence = evidence_candidate
        publication.clear_stale()
        validate_arena_bytes()

        model_dir = model_root(model_dir_candidate)
        if reference.is_symlink() or not reference.is_file():
            fail(
                "IOM_TEST_MODEL_REFERENCE must name a regular file: "
                f"{reference}"
            )

        if (
            executable.is_symlink()
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
        ):
            fail(
                "backend executable is not an executable regular file: "
                f"{executable}"
            )
        artifact_paths = [
            path for _, path in model_artifact_paths(model_dir)
        ]
        for path in artifact_paths:
            publication.add_protected(path, f"model artifact {path.name}")
        publication.validate()
        reference_identity = file_identity(reference, str(reference))
        pack, artifacts = validate_reference(reference, model_dir, artifact_id)
        failure_context.update(
            {
                "reference": {
                    **reference_identity,
                    "case_payload_sha256": pack["provenance"][
                        "case_payload_sha256"
                    ],
                    "artifact_records": artifacts,
                },
                "model": {
                    "directory": str(model_dir),
                    "artifacts": artifacts,
                },
            }
        )

        # The child owns this path while it runs. The stale result was removed
        # as soon as a safe publication descriptor had been retained, so every
        # later input, launch, or runtime failure can replace it with fresh
        # machine-readable evidence.

        stage = "launch"
        completed, supervision = run_backend_process(
            command,
            os.environ.copy(),
            execution_timeout,
            terminate_timeout,
            kill_timeout,
        )
        timed_out = bool(supervision["timed_out"])

        stage = "measurement-validation"
        measurement: dict[str, Any] | None = None
        measurement_diagnostic: str | None = None
        try:
            raw_measurement = load_json(evidence, "backend measurement")
            if completed.returncode == 0:
                measurement = validate_measurement(
                    raw_measurement,
                    args.backend,
                    artifact_id,
                    reference_identity,
                    artifacts,
                    selected_environment,
                )
            else:
                measurement = object_value(
                    raw_measurement, "failed backend measurement"
                )
        except HarnessError as error:
            measurement_diagnostic = str(error)

        passed = (
            not timed_out
            and completed.returncode == 0
            and measurement_diagnostic is None
        )
        diagnostics = []
        if timed_out:
            diagnostics.append(
                f"backend test exceeded {execution_timeout:g} seconds and its "
                f"process group was terminated ({terminate_timeout:g}-second "
                f"TERM grace and bounded {kill_timeout:g}-second post-KILL "
                "reap)"
            )
        if completed.returncode != 0:
            diagnostics.append(
                f"backend test exited with status {completed.returncode}"
            )
        if measurement_diagnostic is not None:
            diagnostics.append(measurement_diagnostic)
        payload = {
            "schema_version": 1,
            "kind": "official-inference-wrapper-evidence",
            "status": "passed" if passed else "failed",
            "backend": args.backend,
            **failure_context,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "measurement": measurement,
            "supervision": supervision,
            "diagnostics": diagnostics,
            "native_capability": "unclaimed",
        }
        publication.validate()
        stage = "evidence-publication"
        write_evidence(evidence, payload)
        if not passed:
            print(
                "official inference backend test failed; "
                f"evidence: {evidence}; diagnostics: {diagnostics}",
                file=sys.stderr,
            )
            if timed_out:
                return 124
            if completed.returncode != 0:
                return completed.returncode if completed.returncode > 0 else 1
            return 2
        print(json.dumps(
            {
                "status": "passed",
                "backend": args.backend,
                "reference_sha256": reference_identity["sha256"],
                "evidence": str(evidence),
            },
            ensure_ascii=False,
            sort_keys=True,
        ))
        return 0
    except Exception as error:
        effective_stage = (
            error.stage if isinstance(error, ProcessFailure) else stage
        )
        details = (
            error.details if isinstance(error, ProcessFailure) else None
        )
        diagnostic = str(error)
        if not isinstance(error, (HarnessError, OSError)):
            diagnostic = f"unexpected failure: {error}"
        if publication is not None and evidence is not None:
            try:
                publication.validate()
                write_evidence(
                    evidence,
                    _failure_payload(
                        args.backend,
                        effective_stage,
                        diagnostic,
                        failure_context,
                        details,
                    ),
                )
            except HarnessError as evidence_error:
                print(
                    "official inference wrapper: failed to publish failure "
                    f"evidence: {evidence_error}",
                    file=sys.stderr,
                )
        print(f"official inference wrapper: {diagnostic}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

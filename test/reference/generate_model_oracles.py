#!/usr/bin/env python3
"""Generate and verify the pinned, offline synthetic TinyLlama corpus.

Generation is deliberately opt-in.  Verification and comparison use only the
Python standard library so ordinary tests never import a model package.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence


PINNED_RUNTIME = {"implementation": "CPython", "version": "3.11.16"}
PINNED_PACKAGES = {
    "jinja2": "3.1.2",
    "numpy": "1.26.4",
    "safetensors": "0.4.1",
    "sentencepiece": "0.1.99",
    "tokenizers": "0.14.1",
    "torch": "2.1.2",
    "transformers": "4.35.0",
}
SCHEMA_VERSION = 1
CHECKPOINT_ATOL = 0.05
CHECKPOINT_RTOL = 0.02
LOGITS_ATOL = 0.05
LOGITS_RTOL = 0.02

MODEL_SPECS = (
    {
        "id": "synthetic-h18-i22-hq3-hkv1-d6",
        "num_hidden_layers": 1,
        "hidden_size": 18,
        "intermediate_size": 22,
        "num_attention_heads": 3,
        "num_key_value_heads": 1,
        "head_dim": 6,
    },
    {
        "id": "synthetic-h8-i12-hq4-hkv2-d2",
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
    },
)
BASE_TOKENS = [1] + [3 + (index % 16) for index in range(16)]
PREFIX_LENGTHS = (1, 15, 16, 17)
CONTINUATION = [4, 5, 6]
PERTURBED_TOKEN_INDEX = 15
PERTURBED_TOKEN = 3


class CorpusError(RuntimeError):
    """A malformed corpus or an unavailable pinned reference environment."""


def _fail(message: str) -> None:
    raise CorpusError(message)


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError, OverflowError) as error:
        _fail(f"cannot serialize canonical corpus payload: {error}")
    raise AssertionError("unreachable")


def _pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as error:
        _fail(f"cannot serialize corpus: {error}")
    raise AssertionError("unreachable")


def _load_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail(f"cannot read corpus {path}: {error}")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=lambda value: _fail(
            f"corpus {path} contains non-finite JSON constant {value}"
        ))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"corpus {path} is not valid UTF-8 JSON: {error}")
    raise AssertionError("unreachable")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    try:
        data = path.read_bytes()
    except OSError as error:
        _fail(f"cannot hash {path}: {error}")
    return len(data), _sha256_bytes(data)


def _generator_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())[1]


def _actual_runtime() -> dict[str, str]:
    implementation = getattr(sys.implementation, "name", "<unknown>")
    version_info = sys.version_info
    version = ".".join(
        str(component)
        for component in (version_info.major, version_info.minor, version_info.micro)
    )
    return {"implementation": "CPython" if implementation == "cpython" else implementation,
            "version": version}


def _require_pinned_runtime() -> dict[str, str]:
    actual = _actual_runtime()
    if actual != PINNED_RUNTIME:
        _fail(
            "reference runtime mismatch: expected CPython 3.11.16, actual "
            f"{actual['implementation']} {actual['version']}"
        )
    return actual


def _semantic_version(version: str) -> str:
    return version.split("+", 1)[0]


def _require_pinned_packages() -> dict[str, str]:
    actual: dict[str, str] = {}
    for package, expected in sorted(PINNED_PACKAGES.items()):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            _fail(f"reference package {package!r} is missing; expected {expected}")
        except Exception as error:
            _fail(f"cannot inspect package {package!r}: {error}")
        if _semantic_version(version) != expected:
            _fail(
                f"reference package mismatch for {package!r}: expected semantic "
                f"version {expected!r}, actual {version!r}"
            )
        actual[package] = version
    return actual


def _canonical_command(output: Path) -> list[str]:
    return ["python3", "test/reference/generate_model_oracles.py", "--output", str(output)]


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        _fail(f"generator artifact is outside repository root: {path}")
    size, digest = _sha256_file(path)
    return {"path": relative, "size": size, "sha256": digest}


def _payload_view(corpus: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": corpus["schema_version"],
        "kind": corpus["kind"],
        "tolerances": corpus["tolerances"],
        "models": corpus["models"],
        "cases": corpus["cases"],
    }


def _payload_sha256(corpus: Mapping[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(_payload_view(corpus)))


def _normal_command(command: Any) -> Any:
    if not isinstance(command, list):
        return command
    normalized = list(command)
    for index, argument in enumerate(normalized[:-1]):
        if argument == "--output":
            normalized[index + 1] = "<OUTPUT>"
            break
    return normalized


def _comparison_view(corpus: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(corpus))
    provenance = result.get("provenance")
    if isinstance(provenance, dict) and "command" in provenance:
        provenance["command"] = _normal_command(provenance["command"])
    return result


def _expect_object(value: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{source} must be an object, actual {type(value).__name__}")
    return value


def _expect_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{source} must be a non-empty string")
    return value


def _expect_nonnegative_int(value: Any, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{source} must be a non-negative integer")
    return value


def _expect_positive_int(value: Any, source: str) -> int:
    value = _expect_nonnegative_int(value, source)
    if value == 0:
        _fail(f"{source} must be positive")
    return value


def _expect_finite_number(value: Any, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{source} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{source} must be finite")
    return number


def _expect_shape(value: Any, source: str) -> list[int]:
    if not isinstance(value, list) or not value:
        _fail(f"{source} must be a non-empty shape list")
    shape = [_expect_positive_int(item, f"{source}[{index}]") for index, item in enumerate(value)]
    return shape


def _shape_count(shape: Sequence[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _validate_tolerances(value: Any) -> dict[str, dict[str, float]]:
    document = _expect_object(value, "tolerances")
    if set(document) != {"checkpoint", "logits"}:
        _fail("tolerances must contain exactly checkpoint and logits")
    result: dict[str, dict[str, float]] = {}
    for name, expected in (
        ("checkpoint", (CHECKPOINT_ATOL, CHECKPOINT_RTOL)),
        ("logits", (LOGITS_ATOL, LOGITS_RTOL)),
    ):
        record = _expect_object(document[name], f"tolerances.{name}")
        if set(record) != {"atol", "rtol"}:
            _fail(f"tolerances.{name} must contain exactly atol and rtol")
        atol = _expect_finite_number(record["atol"], f"tolerances.{name}.atol")
        rtol = _expect_finite_number(record["rtol"], f"tolerances.{name}.rtol")
        if atol != expected[0] or rtol != expected[1] or atol < 0 or rtol < 0:
            _fail(f"tolerances.{name} does not match the frozen policy")
        result[name] = {"atol": atol, "rtol": rtol}
    return result


def _expected_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": spec["num_hidden_layers"],
        "hidden_size": spec["hidden_size"],
        "intermediate_size": spec["intermediate_size"],
        "num_attention_heads": spec["num_attention_heads"],
        "num_key_value_heads": spec["num_key_value_heads"],
        "vocab_size": 19,
        "max_position_embeddings": 17,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
    }


def _expected_weight_plan(config: Mapping[str, Any]) -> list[tuple[str, list[int]]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    kv_width = int(config["num_key_value_heads"]) * int(config["head_dim"])
    result: list[tuple[str, list[int]]] = [
        ("model.embed_tokens.weight", [19, hidden]),
        ("model.norm.weight", [hidden]),
        ("lm_head.weight", [19, hidden]),
    ]
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        result.extend(
            (
                (prefix + "input_layernorm.weight", [hidden]),
                (prefix + "post_attention_layernorm.weight", [hidden]),
                (prefix + "self_attn.q_proj.weight", [hidden, hidden]),
                (prefix + "self_attn.k_proj.weight", [kv_width, hidden]),
                (prefix + "self_attn.v_proj.weight", [kv_width, hidden]),
                (prefix + "self_attn.o_proj.weight", [hidden, hidden]),
                (prefix + "mlp.gate_proj.weight", [intermediate, hidden]),
                (prefix + "mlp.up_proj.weight", [intermediate, hidden]),
                (prefix + "mlp.down_proj.weight", [hidden, intermediate]),
            )
        )
    return result


def _bf16_hex_finite(payload: str, expected_count: int, source: str) -> None:
    if not isinstance(payload, str) or len(payload) != expected_count * 4:
        _fail(
            f"{source}.bf16_le_hex has wrong payload length; expected "
            f"{expected_count * 4} hex characters"
        )
    try:
        raw = bytes.fromhex(payload)
    except ValueError as error:
        _fail(f"{source}.bf16_le_hex is not hexadecimal: {error}")
    if len(raw) != expected_count * 2:
        _fail(f"{source}.bf16_le_hex has wrong byte length")
    for index in range(expected_count):
        word = struct.unpack_from("<H", raw, index * 2)[0]
        value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
        if not math.isfinite(value):
            _fail(f"{source}.bf16_le_hex contains non-finite element {index}")


def _validate_models(models: Any) -> list[dict[str, Any]]:
    if not isinstance(models, list) or len(models) != len(MODEL_SPECS):
        _fail("models must contain exactly the two pinned configurations")
    normalized: list[dict[str, Any]] = []
    for index, spec in enumerate(MODEL_SPECS):
        model = _expect_object(models[index], f"models[{index}]")
        if model.get("id") != spec["id"]:
            _fail(f"models[{index}].id does not match the pinned configuration")
        config = _expect_object(model.get("config"), f"models[{index}].config")
        if config != _expected_config(spec):
            _fail(f"models[{index}].config does not match the pinned configuration")
        weights = model.get("weights")
        plan = _expected_weight_plan({**spec, "head_dim": spec["head_dim"]})
        if not isinstance(weights, list) or len(weights) != len(plan):
            _fail(f"models[{index}].weights has the wrong number of records")
        for weight_index, (name, shape) in enumerate(plan):
            record = _expect_object(weights[weight_index], f"models[{index}].weights[{weight_index}]")
            if set(record) != {"name", "shape", "dtype", "bf16_le_hex"}:
                _fail(f"models[{index}].weights[{weight_index}] has malformed fields")
            if record["name"] != name or record["shape"] != shape or record["dtype"] != "BF16":
                _fail(f"models[{index}].weights[{weight_index}] has wrong identity or shape")
            _bf16_hex_finite(
                record["bf16_le_hex"],
                _shape_count(shape),
                f"models[{index}].weights[{weight_index}]",
            )
        normalized.append({"id": model["id"], "config": dict(config), "weights": list(weights)})
    return normalized


def _validate_tensor_record(
    record: Any,
    expected_shape: Sequence[int] | None,
    source: str,
) -> dict[str, Any]:
    document = _expect_object(record, source)
    if set(document) != {"shape", "values"}:
        _fail(f"{source} must contain exactly shape and values")
    shape = _expect_shape(document["shape"], f"{source}.shape")
    if expected_shape is not None and shape != list(expected_shape):
        _fail(f"{source}.shape is wrong; expected {list(expected_shape)}, actual {shape}")
    values = document["values"]
    if not isinstance(values, list) or len(values) != _shape_count(shape):
        _fail(f"{source}.values has the wrong payload length for its shape")
    normalized_values: list[float] = []
    for index, value in enumerate(values):
        normalized_values.append(_expect_finite_number(value, f"{source}.values[{index}]"))
    return {"shape": shape, "values": normalized_values}


def _validate_snapshots(snapshots: Any, model_config: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    if not isinstance(snapshots, list) or len(snapshots) != 2:
        _fail(f"{source}.snapshots must contain full and cached snapshots")
    hidden = int(model_config["hidden_size"])
    vocab = int(model_config["vocab_size"])
    result: list[dict[str, Any]] = []
    phases: set[str] = set()
    for index, snapshot in enumerate(snapshots):
        item = _expect_object(snapshot, f"{source}.snapshots[{index}]")
        required = {
            "phase", "prefix_ids", "position_start", "run_length", "positions",
            "last_layer_output", "final_norm", "logits", "greedy_id",
            "margin_stable", "margins",
        }
        if set(item) != required:
            _fail(f"{source}.snapshots[{index}] has malformed fields")
        phase = item["phase"]
        if phase not in {"full", "cached"} or phase in phases:
            _fail(f"{source}.snapshots[{index}].phase is not a unique full/cached phase")
        phases.add(phase)
        prefix_ids = item["prefix_ids"]
        if not isinstance(prefix_ids, list) or not prefix_ids:
            _fail(f"{source}.snapshots[{index}].prefix_ids must be non-empty")
        for token_index, token in enumerate(prefix_ids):
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab:
                _fail(f"{source}.snapshots[{index}].prefix_ids[{token_index}] is invalid")
        position_start = _expect_nonnegative_int(item["position_start"], f"{source}.snapshots[{index}].position_start")
        run_length = _expect_positive_int(item["run_length"], f"{source}.snapshots[{index}].run_length")
        positions = item["positions"]
        if positions != list(range(position_start, position_start + run_length)):
            _fail(f"{source}.snapshots[{index}].positions are not contiguous absolute positions")
        last = _validate_tensor_record(item["last_layer_output"], [run_length, hidden], f"{source}.snapshots[{index}].last_layer_output")
        norm = _validate_tensor_record(item["final_norm"], [run_length, hidden], f"{source}.snapshots[{index}].final_norm")
        logits = _validate_tensor_record(item["logits"], [run_length, vocab], f"{source}.snapshots[{index}].logits")
        for key in ("greedy_id", "margin_stable"):
            values = item[key]
            if not isinstance(values, list) or len(values) != run_length:
                _fail(f"{source}.snapshots[{index}].{key} has the wrong length")
        for token_index, token in enumerate(item["greedy_id"]):
            if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab:
                _fail(f"{source}.snapshots[{index}].greedy_id[{token_index}] is invalid")
        if any(not isinstance(value, bool) for value in item["margin_stable"]):
            _fail(f"{source}.snapshots[{index}].margin_stable must contain booleans")
        margins = item["margins"]
        if not isinstance(margins, list) or len(margins) != run_length:
            _fail(f"{source}.snapshots[{index}].margins has the wrong length")
        for margin_index, margin in enumerate(margins):
            margin_doc = _expect_object(margin, f"{source}.snapshots[{index}].margins[{margin_index}]")
            expected_fields = {"winner_id", "winner", "runner_up", "winner_error", "runner_up_error", "stable"}
            if set(margin_doc) != expected_fields:
                _fail(f"{source}.snapshots[{index}].margins[{margin_index}] is malformed")
            winner_id = margin_doc["winner_id"]
            if isinstance(winner_id, bool) or not isinstance(winner_id, int) or not 0 <= winner_id < vocab:
                _fail(f"{source}.snapshots[{index}].margins[{margin_index}].winner_id is invalid")
            for field in ("winner", "runner_up", "winner_error", "runner_up_error"):
                _expect_finite_number(margin_doc[field], f"{source}.snapshots[{index}].margins[{margin_index}].{field}")
            if not isinstance(margin_doc["stable"], bool):
                _fail(f"{source}.snapshots[{index}].margins[{margin_index}].stable is not boolean")
            if margin_doc["stable"] != item["margin_stable"][margin_index]:
                _fail(f"{source}.snapshots[{index}] has inconsistent margin stability")
        result.append({
            "phase": phase,
            "prefix_ids": list(prefix_ids),
            "position_start": position_start,
            "run_length": run_length,
            "positions": list(positions),
            "last_layer_output": last,
            "final_norm": norm,
            "logits": logits,
            "greedy_id": list(item["greedy_id"]),
            "margin_stable": list(item["margin_stable"]),
            "margins": list(margins),
        })
    return result

def _canonical_case_specs(model_id: str) -> dict[str, dict[str, Any]]:
    def input_record(
        mode: str,
        prompt_ids: Sequence[int],
        prefix_ids: Sequence[int],
        decode_ids: Sequence[int],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "selection_policy": "lowest_id_on_equal_logits",
            "prompt_ids": list(prompt_ids),
            "prefix_ids": list(prefix_ids),
            "decode_ids": list(decode_ids),
            "tokens": list(prompt_ids),
            "rendered": None,
            "positions": list(range(len(prompt_ids))),
            "max_new_tokens": max_new_tokens,
        }

    records: dict[str, dict[str, Any]] = {}
    for prefix_length in PREFIX_LENGTHS:
        prompt = BASE_TOKENS[:prefix_length]
        records[f"{model_id}-full-prefix-{prefix_length}"] = {
            "model_id": model_id,
            "input": input_record("full_prompt", prompt, prompt, [], 0),
            "token_ids": [],
            "stop_reason": "max_new_tokens",
            "initialized_kv_length": prefix_length,
        }
    for prefix_length in (1, 14):
        prompt = BASE_TOKENS[:prefix_length] + CONTINUATION
        records[f"{model_id}-teacher-forced-from-{prefix_length}"] = {
            "model_id": model_id,
            "input": input_record(
                "teacher_forced",
                prompt,
                BASE_TOKENS[:prefix_length],
                CONTINUATION,
                len(CONTINUATION),
            ),
            "token_ids": list(CONTINUATION),
            "stop_reason": "max_new_tokens",
            "initialized_kv_length": prefix_length,
        }
    perturbed = list(BASE_TOKENS)
    perturbed[PERTURBED_TOKEN_INDEX] = PERTURBED_TOKEN
    records[f"{model_id}-future-perturbation"] = {
        "model_id": model_id,
        "input": input_record(
            "future_perturbation",
            perturbed,
            BASE_TOKENS[:PERTURBED_TOKEN_INDEX],
            [],
            0,
        ),
        "token_ids": [],
        "stop_reason": "max_new_tokens",
        "initialized_kv_length": PERTURBED_TOKEN_INDEX,
    }
    return records


def _derived_margin(logits: Sequence[float]) -> dict[str, Any]:
    if not logits or any(not math.isfinite(value) for value in logits):
        _fail("stored logits must be finite and non-empty")
    winner_id = max(range(len(logits)), key=lambda index: (logits[index], -index))
    winner = float(logits[winner_id])
    runner_up = max(
        (float(value) for index, value in enumerate(logits) if index != winner_id),
        default=winner,
    )
    winner_error = LOGITS_ATOL + LOGITS_RTOL * abs(winner)
    runner_up_error = LOGITS_ATOL + LOGITS_RTOL * abs(runner_up)
    return {
        "winner_id": winner_id,
        "winner": winner,
        "runner_up": runner_up,
        "winner_error": winner_error,
        "runner_up_error": runner_up_error,
        "stable": winner - winner_error > runner_up + runner_up_error,
    }


def _validate_snapshot_policy(
    snapshots: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[int],
    source: str,
) -> tuple[list[int], list[int]]:
    if [snapshot["phase"] for snapshot in snapshots] != ["full", "cached"]:
        _fail(f"{source}.snapshots must be ordered full then cached")
    run_length = len(prompt_ids)
    derived_by_phase: dict[str, tuple[list[int], list[bool]]] = {}
    for snapshot in snapshots:
        phase = snapshot["phase"]
        if snapshot["prefix_ids"] != list(prompt_ids):
            _fail(f"{source}.snapshots.{phase}.prefix_ids do not match the canonical prompt")
        if snapshot["position_start"] != 0 or snapshot["run_length"] != run_length:
            _fail(f"{source}.snapshots.{phase} does not cover the canonical run length")
        logits = snapshot["logits"]["values"]
        rows = [
            logits[offset : offset + 19]
            for offset in range(0, len(logits), 19)
        ]
        margins = [_derived_margin(row) for row in rows]
        greedy_ids = [margin["winner_id"] for margin in margins]
        stable = [margin["stable"] for margin in margins]
        if snapshot["greedy_id"] != greedy_ids:
            _fail(f"{source}.snapshots.{phase}.greedy_id is not derived from logits")
        if snapshot["margin_stable"] != stable:
            _fail(f"{source}.snapshots.{phase}.margin_stable is not derived from logits")
        if snapshot["margins"] != margins:
            _fail(f"{source}.snapshots.{phase}.margins are not derived from logits")
        derived_by_phase[phase] = (greedy_ids, stable)

    full = snapshots[0]
    cached = snapshots[1]
    for key in ("last_layer_output", "final_norm", "logits"):
        full_values = full[key]["values"]
        cached_values = cached[key]["values"]
        if len(full_values) != len(cached_values):
            _fail(f"{source}.full and cached {key} payloads differ in length")
        atol, rtol = (
            (LOGITS_ATOL, LOGITS_RTOL)
            if key == "logits"
            else (CHECKPOINT_ATOL, CHECKPOINT_RTOL)
        )
        for index, (reference, actual) in enumerate(zip(full_values, cached_values)):
            bound = atol + rtol * abs(reference)
            if abs(actual - reference) > bound:
                _fail(f"{source}.full and cached {key} differ at element {index}")
    return derived_by_phase["full"]


def _validate_cases(cases: Any, models: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        _fail("cases must be a list")
    model_by_id = {model["id"]: model for model in models}
    canonical: dict[str, dict[str, Any]] = {}
    for model_id in model_by_id:
        canonical.update(_canonical_case_specs(model_id))
    if len(cases) != len(canonical):
        _fail("cases must contain exactly one record for every pinned case")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(cases):
        source = f"cases[{index}]"
        document = _expect_object(case, source)
        required = {"id", "model_id", "input", "snapshots", "expected_result"}
        if set(document) != required:
            _fail(f"{source} has malformed fields")
        case_id = _expect_string(document["id"], f"{source}.id")
        if case_id in seen:
            _fail(f"{source}.id is duplicated")
        seen.add(case_id)
        expected_case = canonical.get(case_id)
        if expected_case is None:
            _fail(f"{source}.id is not a pinned case")
        model_id = _expect_string(document["model_id"], f"{source}.model_id")
        if model_id != expected_case["model_id"]:
            _fail(f"{source}.model_id does not match its pinned case ID")
        model_config = model_by_id[model_id]["config"]

        input_doc = _expect_object(document["input"], f"{source}.input")
        if input_doc != expected_case["input"]:
            _fail(f"{source}.input does not match the frozen case matrix")
        snapshots = _validate_snapshots(document["snapshots"], model_config, source)
        full_greedy_ids, full_margin_stable = _validate_snapshot_policy(
            snapshots, input_doc["prompt_ids"], source
        )

        expected = _expect_object(document["expected_result"], f"{source}.expected_result")
        expected_fields = {
            "token_ids", "stop_reason", "initialized_kv_length", "production_greedy_ids",
            "reference_margin_certified_positions",
        }
        if set(expected) != expected_fields:
            _fail(f"{source}.expected_result has malformed fields")
        if expected["token_ids"] != expected_case["token_ids"]:
            _fail(f"{source}.expected_result.token_ids do not match the frozen case")
        if expected["stop_reason"] != expected_case["stop_reason"]:
            _fail(f"{source}.expected_result.stop_reason is not the frozen outcome")
        if expected["initialized_kv_length"] != expected_case["initialized_kv_length"]:
            _fail(f"{source}.expected_result.initialized_kv_length is not the prefix length")
        if expected["production_greedy_ids"] != full_greedy_ids:
            _fail(f"{source}.expected_result.production_greedy_ids do not match full logits")
        certified = [
            position
            for position, stable in enumerate(full_margin_stable)
            if stable
        ]
        if expected["reference_margin_certified_positions"] != certified:
            _fail(f"{source}.expected_result.reference_margin_certified_positions are not derived from margins")
        normalized.append({
            "id": case_id,
            "model_id": model_id,
            "input": dict(input_doc),
            "snapshots": snapshots,
            "expected_result": dict(expected),
        })
    if seen != set(canonical):
        _fail("cases do not contain exactly the pinned prefix, continuation, and perturbation records")
    return normalized


def _validate_provenance(value: Any, source: str) -> dict[str, Any]:
    document = _expect_object(value, f"{source}.provenance")
    required = {
        "runtime", "packages", "generator_sha256", "command", "precision", "artifacts",
        "sensitivity_recipe", "tie_policy", "payload_sha256", "bounds",
    }
    if set(document) != required:
        _fail(f"{source}.provenance has malformed fields")
    runtime = _expect_object(document["runtime"], f"{source}.provenance.runtime")
    if runtime != PINNED_RUNTIME:
        _fail(f"{source}.provenance.runtime does not match CPython 3.11.16")
    packages = _expect_object(document["packages"], f"{source}.provenance.packages")
    if set(packages) != set(PINNED_PACKAGES):
        _fail("provenance package set does not match the pinned package set")
    for package, expected in PINNED_PACKAGES.items():
        version = _expect_string(packages[package], f"provenance.packages.{package}")
        if _semantic_version(version) != expected:
            _fail(f"provenance package {package!r} is not semantically pinned to {expected}")
    generator_sha = document["generator_sha256"]
    if not isinstance(generator_sha, str) or len(generator_sha) != 64 or any(c not in "0123456789abcdef" for c in generator_sha):
        _fail("provenance.generator_sha256 must be a lowercase SHA-256 digest")
    if generator_sha != _generator_sha256():
        _fail("provenance.generator_sha256 does not match the checked-in generator")
    command = document["command"]
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
        _fail("provenance.command must be a non-empty list of strings")
    precision = _expect_object(document["precision"], "provenance.precision")
    expected_precision = {
        "weights": "BF16",
        "activations": "BF16",
        "stored_results": "BF16",
        "reductions": "FP32",
        "softmax": "FP32",
        "softmax_probability_storage": "BF16 before PV",
        "reference": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
    }
    if precision != expected_precision:
        _fail("provenance.precision does not match the frozen arithmetic policy")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        _fail("provenance.artifacts must contain at least the generator artifact")
    root = Path(__file__).resolve().parents[2]
    for index, artifact in enumerate(artifacts):
        record = _expect_object(artifact, f"provenance.artifacts[{index}]")
        if set(record) != {"path", "size", "sha256"}:
            _fail(f"provenance.artifacts[{index}] has malformed fields")
        path = _expect_string(record["path"], f"provenance.artifacts[{index}].path")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            _fail(f"provenance.artifacts[{index}].path must be repository-relative")
        actual_path = root / relative
        size = _expect_nonnegative_int(record["size"], f"provenance.artifacts[{index}].size")
        digest = _expect_string(record["sha256"], f"provenance.artifacts[{index}].sha256")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            _fail(f"provenance.artifacts[{index}].sha256 is not lowercase SHA-256")
        actual_size, actual_digest = _sha256_file(actual_path)
        if size != actual_size or digest != actual_digest:
            _fail(f"provenance artifact {path!r} identity changed")
    recipe = _expect_object(document["sensitivity_recipe"], "provenance.sensitivity_recipe")
    expected_recipe = {
        "non_normalization_multiplier": 8,
        "normalization": "BF16 RNE(1 + 0.015 * (i mod 3))",
        "seed": 17,
        "application": "once per initialized configuration before measurement",
    }
    if recipe != expected_recipe:
        _fail("provenance.sensitivity_recipe does not match the frozen recipe")
    tie_policy = _expect_object(document["tie_policy"], "provenance.tie_policy")
    if tie_policy != {
        "name": "lowest_id",
        "description": "exact equality chooses the lowest vocabulary ID; interval stability remains strict",
    }:
        _fail("provenance.tie_policy does not match the frozen policy")
    payload_sha = document["payload_sha256"]
    if not isinstance(payload_sha, str) or len(payload_sha) != 64 or any(c not in "0123456789abcdef" for c in payload_sha):
        _fail("provenance.payload_sha256 must be a lowercase SHA-256 digest")
    bounds = _expect_object(document["bounds"], "provenance.bounds")
    if bounds != {
        "checkpoint": {"atol": CHECKPOINT_ATOL, "rtol": CHECKPOINT_RTOL},
        "logits": {"atol": LOGITS_ATOL, "rtol": LOGITS_RTOL},
        "formula": "abs(actual-ref) <= atol + rtol * abs(ref)",
    }:
        _fail("provenance.bounds does not match the frozen policy")
    return dict(document)


def validate_corpus(path: Path) -> dict[str, Any]:
    corpus = _expect_object(_load_json(path), str(path))
    expected_top = {"schema_version", "kind", "provenance", "tolerances", "models", "cases"}
    if set(corpus) != expected_top:
        _fail(f"{path} has malformed top-level schema")
    if corpus["schema_version"] != SCHEMA_VERSION or corpus["kind"] != "synthetic":
        _fail(f"{path} has unsupported schema version or kind")
    provenance = _validate_provenance(corpus["provenance"], str(path))
    tolerances = _validate_tolerances(corpus["tolerances"])
    models = _validate_models(corpus["models"])
    cases = _validate_cases(corpus["cases"], models)
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synthetic",
        "provenance": provenance,
        "tolerances": tolerances,
        "models": models,
        "cases": cases,
    }
    actual_payload_sha = _payload_sha256(normalized)
    if provenance["payload_sha256"] != actual_payload_sha:
        _fail(f"{path} payload digest does not match its logical corpus")
    return normalized


def _set_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _import_reference_packages() -> tuple[Any, Any, Any]:
    _set_offline_environment()
    _require_pinned_runtime()
    _require_pinned_packages()
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as error:
        _fail(f"cannot import pinned reference packages: {error}")
    return torch, LlamaConfig, LlamaForCausalLM


def _round_bf16(torch: Any, tensor: Any) -> Any:
    return tensor.to(dtype=torch.bfloat16).to(dtype=torch.float32)


def _config_for(torch: Any, LlamaConfig: Any, spec: Mapping[str, Any]) -> Any:
    config = LlamaConfig(
        vocab_size=19,
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        num_key_value_heads=spec["num_key_value_heads"],
        max_position_embeddings=17,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
    )
    # Transformers 4.35 uses this private selector for explicit eager attention.
    config._attn_implementation = "eager"
    config.attention_bias = False
    config.mlp_bias = False
    return config


def _apply_sensitivity(torch: Any, model: Any) -> None:
    with torch.no_grad():
        model.to(dtype=torch.bfloat16)
        for name, parameter in model.named_parameters():
            if name.endswith("input_layernorm.weight") or name.endswith("post_attention_layernorm.weight") or name == "model.norm.weight":
                values = [1.0 + 0.015 * (index % 3) for index in range(parameter.numel())]
                replacement = torch.tensor(values, dtype=torch.float32).to(dtype=torch.bfloat16).reshape(parameter.shape)
                parameter.copy_(replacement)
            else:
                parameter.mul_(8)


def _tensor_hex(torch: Any, tensor: Any) -> str:
    # Torch 2.1 exposes BF16 storage but not a public uint16 dtype.  Viewing
    # the words as signed int16 preserves the exact bits; masking restores
    # their unsigned little-endian representation.
    values = tensor.detach().to(dtype=torch.bfloat16).cpu().contiguous().view(torch.int16).reshape(-1).tolist()
    return b"".join(struct.pack("<H", int(value) & 0xFFFF) for value in values).hex()


def _tensor_values(tensor: Any) -> list[float]:
    values = tensor.detach().to(dtype=tensor.dtype).cpu().reshape(-1).tolist()
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        _fail("reference produced a non-finite tensor value")
    return result


def _capture_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def _margin_info(logits: Sequence[float]) -> tuple[int, bool, dict[str, Any]]:
    if not logits or any(not math.isfinite(value) for value in logits):
        _fail("reference produced non-finite or empty logits")
    winner = max(range(len(logits)), key=lambda index: (logits[index], -index))
    winner_value = float(logits[winner])
    others = [value for index, value in enumerate(logits) if index != winner]
    runner_value = max(others) if others else winner_value
    winner_error = LOGITS_ATOL + LOGITS_RTOL * abs(winner_value)
    runner_error = LOGITS_ATOL + LOGITS_RTOL * abs(runner_value)
    stable = winner_value - winner_error > runner_value + runner_error
    return winner, stable, {
        "winner_id": winner,
        "winner": winner_value,
        "runner_up": float(runner_value),
        "winner_error": float(winner_error),
        "runner_up_error": float(runner_error),
        "stable": bool(stable),
    }


def _snapshot(phase: str, prefix_ids: Sequence[int], output: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    last_layer = output["last_layer_output"]
    final_norm = output["final_norm"]
    logits = output["logits"]
    run_length = int(logits.shape[0])
    hidden = int(last_layer.shape[-1])
    vocab = int(logits.shape[-1])
    logits_rows = logits.reshape(run_length, vocab).tolist()
    greedy_ids: list[int] = []
    margin_stable: list[bool] = []
    margins: list[dict[str, Any]] = []
    for row in logits_rows:
        row_values = [float(value) for value in row]
        winner, stable, margin = _margin_info(row_values)
        greedy_ids.append(winner)
        margin_stable.append(stable)
        margins.append(margin)
    return {
        "phase": phase,
        "prefix_ids": list(prefix_ids),
        "position_start": 0,
        "run_length": run_length,
        "positions": list(range(run_length)),
        "last_layer_output": {"shape": [run_length, hidden], "values": _tensor_values(last_layer)},
        "final_norm": {"shape": [run_length, hidden], "values": _tensor_values(final_norm)},
        "logits": {"shape": [run_length, vocab], "values": _tensor_values(logits)},
        "greedy_id": greedy_ids,
        "margin_stable": margin_stable,
        "margins": margins,
    }


def _forward_full(torch: Any, model: Any, token_ids: Sequence[int]) -> tuple[dict[str, Any], Any]:
    layer_values: list[Any] = []
    norm_values: list[Any] = []
    layer_hook = model.model.layers[-1].register_forward_hook(
        lambda _module, _inputs, output: layer_values.append(_capture_output(output).detach().clone())
    )
    norm_hook = model.model.norm.register_forward_hook(
        lambda _module, _inputs, output: norm_values.append(_capture_output(output).detach().clone())
    )
    try:
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor([list(token_ids)], dtype=torch.long),
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        layer_hook.remove()
        norm_hook.remove()
    if len(layer_values) != 1 or len(norm_values) != 1:
        _fail("reference hooks did not capture exactly one full forward")
    logits = _round_bf16(torch, output.logits[0])
    last_layer = _round_bf16(torch, layer_values[0][0])
    final_norm = _round_bf16(torch, norm_values[0][0])
    return {
        "last_layer_output": last_layer,
        "final_norm": final_norm,
        "logits": logits,
    }, output.past_key_values


def _forward_one(torch: Any, model: Any, token_id: int, past: Any, position: int) -> tuple[dict[str, Any], Any]:
    layer_values: list[Any] = []
    norm_values: list[Any] = []
    layer_hook = model.model.layers[-1].register_forward_hook(
        lambda _module, _inputs, output: layer_values.append(_capture_output(output).detach().clone())
    )
    norm_hook = model.model.norm.register_forward_hook(
        lambda _module, _inputs, output: norm_values.append(_capture_output(output).detach().clone())
    )
    try:
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor([[token_id]], dtype=torch.long),
                position_ids=torch.tensor([[position]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        layer_hook.remove()
        norm_hook.remove()
    if len(layer_values) != 1 or len(norm_values) != 1:
        _fail("reference hooks did not capture exactly one cached forward")
    return {
        "last_layer_output": _round_bf16(torch, layer_values[0][0]),
        "final_norm": _round_bf16(torch, norm_values[0][0]),
        "logits": _round_bf16(torch, output.logits[0]),
    }, output.past_key_values


def _forward_cached(torch: Any, model: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    rows: dict[str, list[Any]] = {"last_layer_output": [], "final_norm": [], "logits": []}
    past = None
    for position, token_id in enumerate(token_ids):
        output, past = _forward_one(torch, model, int(token_id), past, position)
        for key in rows:
            rows[key].append(output[key][0].detach().clone())
    return {
        key: torch.stack(values, dim=0)
        for key, values in rows.items()
    }


def _assert_self_consistent(full: Mapping[str, Any], cached: Mapping[str, Any]) -> None:
    for key in ("last_layer_output", "final_norm", "logits"):
        full_values = [float(value) for value in full[key].reshape(-1).tolist()]
        cached_values = [float(value) for value in cached[key].reshape(-1).tolist()]
        if len(full_values) != len(cached_values):
            _fail(f"full and past_key_values {key} shapes differ")
        for index, (reference, actual) in enumerate(zip(full_values, cached_values)):
            bound = CHECKPOINT_ATOL + CHECKPOINT_RTOL * abs(reference)
            if key == "logits":
                bound = LOGITS_ATOL + LOGITS_RTOL * abs(reference)
            if not math.isfinite(reference) or not math.isfinite(actual) or abs(actual - reference) > bound:
                _fail(f"full and past_key_values {key} differ at element {index}")


def _case(model_id: str, mode: str, prompt_ids: Sequence[int], prefix_ids: Sequence[int], decode_ids: Sequence[int], max_new_tokens: int, torch: Any, model: Any) -> dict[str, Any]:
    full, _past = _forward_full(torch, model, prompt_ids)
    cached = _forward_cached(torch, model, prompt_ids)
    _assert_self_consistent(full, cached)
    snapshots = [
        _snapshot("full", prompt_ids, full, torch),
        _snapshot("cached", prompt_ids, cached, torch),
    ]
    greedy_ids = list(snapshots[0]["greedy_id"])
    certified_positions = [
        index for index, stable in enumerate(snapshots[0]["margin_stable"])
        if stable
    ]
    if mode == "teacher_forced":
        token_ids = list(decode_ids)
    else:
        token_ids = []
    stop_reason = "max_new_tokens"
    initialized_kv_length = len(prefix_ids)
    return {
        "id": f"{model_id}-" + {
            "full_prompt": f"full-prefix-{len(prompt_ids)}",
            "teacher_forced": f"teacher-forced-from-{len(prefix_ids)}",
            "future_perturbation": "future-perturbation",
        }[mode],
        "model_id": model_id,
        "input": {
            "mode": mode,
            "selection_policy": "lowest_id_on_equal_logits",
            "prompt_ids": list(prompt_ids),
            "prefix_ids": list(prefix_ids),
            "decode_ids": list(decode_ids),
            "tokens": list(prompt_ids),
            "rendered": None,
            "positions": list(range(len(prompt_ids))),
            "max_new_tokens": max_new_tokens,
        },
        "snapshots": snapshots,
        "expected_result": {
            "token_ids": token_ids,
            "stop_reason": stop_reason,
            "initialized_kv_length": initialized_kv_length,
            "production_greedy_ids": greedy_ids,
            "reference_margin_certified_positions": certified_positions,
        },
    }


def _generate_payload(torch: Any, LlamaConfig: Any, LlamaForCausalLM: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    models: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        torch.manual_seed(17)
        config = _config_for(torch, LlamaConfig, spec)
        model = LlamaForCausalLM(config)
        model.eval()
        _apply_sensitivity(torch, model)
        config_document = _expected_config(spec)
        plan = _expected_weight_plan({**spec, "head_dim": spec["head_dim"]})
        state = model.state_dict()
        weights: list[dict[str, Any]] = []
        for name, shape in plan:
            if name not in state:
                _fail(f"HF model did not expose required weight {name}")
            tensor = state[name].detach().cpu()
            weights.append({
                "name": name,
                "shape": shape,
                "dtype": "BF16",
                "bf16_le_hex": _tensor_hex(torch, tensor),
            })
        models.append({"id": spec["id"], "config": config_document, "weights": weights})
        for prefix_length in PREFIX_LENGTHS:
            prompt = BASE_TOKENS[:prefix_length]
            cases.append(_case(spec["id"], "full_prompt", prompt, prompt, [], 0, torch, model))
        for prefix_length in (1, 14):
            prompt = BASE_TOKENS[:prefix_length] + CONTINUATION
            cases.append(_case(spec["id"], "teacher_forced", prompt, BASE_TOKENS[:prefix_length], CONTINUATION, 3, torch, model))
        perturbed = list(BASE_TOKENS)
        perturbed[PERTURBED_TOKEN_INDEX] = PERTURBED_TOKEN
        cases.append(_case(
                spec["id"], "future_perturbation", perturbed,
                BASE_TOKENS[:PERTURBED_TOKEN_INDEX], [], 0, torch, model))
    packages = _require_pinned_packages()
    return models, cases, packages


def build_corpus(output: Path) -> dict[str, Any]:
    torch, LlamaConfig, LlamaForCausalLM = _import_reference_packages()
    runtime = _require_pinned_runtime()
    models, cases, packages = _generate_payload(torch, LlamaConfig, LlamaForCausalLM)
    root = Path(__file__).resolve().parents[2]
    corpus: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synthetic",
        "provenance": {
            "runtime": runtime,
            "packages": packages,
            "generator_sha256": _generator_sha256(),
            "command": _canonical_command(output),
            "precision": {
                "weights": "BF16",
                "activations": "BF16",
                "stored_results": "BF16",
                "reductions": "FP32",
                "softmax": "FP32",
                "softmax_probability_storage": "BF16 before PV",
                "reference": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
            },
            "artifacts": [_artifact_record(Path(__file__).resolve(), root)],
            "sensitivity_recipe": {
                "non_normalization_multiplier": 8,
                "normalization": "BF16 RNE(1 + 0.015 * (i mod 3))",
                "seed": 17,
                "application": "once per initialized configuration before measurement",
            },
            "tie_policy": {
                "name": "lowest_id",
                "description": "exact equality chooses the lowest vocabulary ID; interval stability remains strict",
            },
            "payload_sha256": "0" * 64,
            "bounds": {
                "checkpoint": {"atol": CHECKPOINT_ATOL, "rtol": CHECKPOINT_RTOL},
                "logits": {"atol": LOGITS_ATOL, "rtol": LOGITS_RTOL},
                "formula": "abs(actual-ref) <= atol + rtol * abs(ref)",
            },
        },
        "tolerances": {
            "checkpoint": {"atol": CHECKPOINT_ATOL, "rtol": CHECKPOINT_RTOL},
            "logits": {"atol": LOGITS_ATOL, "rtol": LOGITS_RTOL},
        },
        "models": models,
        "cases": cases,
    }
    corpus["provenance"]["payload_sha256"] = _payload_sha256(corpus)
    return corpus


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except OSError:
            pass
        _fail(f"cannot write corpus {path}: {error}")


def generate(output: Path) -> None:
    corpus = build_corpus(output)
    _write_atomic(output, _pretty_bytes(corpus))
    validate_corpus(output)


def verify(path: Path) -> None:
    validate_corpus(path)


def compare(left: Path, right: Path) -> None:
    left_corpus = validate_corpus(left)
    right_corpus = validate_corpus(right)
    if _comparison_view(left_corpus) != _comparison_view(right_corpus):
        _fail(f"corpora differ beyond their invocation output paths: {left} vs {right}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or verify the offline synthetic Llama corpus")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path, metavar="PATH", help="generate a corpus at PATH")
    operation.add_argument("--verify", type=Path, metavar="PATH", help="verify a corpus with stdlib only")
    operation.add_argument("--compare", nargs=2, type=Path, metavar=("LEFT", "RIGHT"), help="compare two validated corpora")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output is not None:
            generate(args.output)
        elif args.verify is not None:
            verify(args.verify)
        else:
            compare(args.compare[0], args.compare[1])
    except CorpusError as error:
        print(f"synthetic reference corpus: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

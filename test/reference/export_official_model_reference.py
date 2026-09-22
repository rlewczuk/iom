#!/usr/bin/env python3
"""Export and verify the pinned offline TinyLlama HF reference pack.

The exporter is intentionally an explicit developer tool.  It measures only a
caller-selected local Hugging Face distribution and never discovers a model,
downloads weights, or consults an IOM implementation.  Verification is kept
stdlib-only so a recorded pack can be checked before any neural runtime is
loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tokenizer_reference as tokenizer_provenance
except ImportError:  # pragma: no cover - package-style invocation support.
    from test.reference import tokenizer_reference as tokenizer_provenance  # type: ignore


SCRIPT_PATH = Path(__file__).resolve()
REFERENCE_DIR = SCRIPT_PATH.parent
TOKENIZER_MANIFEST = REFERENCE_DIR / "tokenizer_reference_manifest.json"
TOKENIZER_ORACLES = REFERENCE_DIR / "tokenizer_oracles.json"

PINNED_RUNTIME: dict[str, str] = dict(tokenizer_provenance.PINNED_RUNTIME)
PINNED_PACKAGES: dict[str, str] = {
    **tokenizer_provenance.PINNED_PACKAGES,
    "numpy": "1.26.4",
    "safetensors": "0.4.3",
    "torch": "2.3.1+cpu",
}
PINNED_MODEL: dict[str, Any] = {
    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "artifacts": {
        "config.json": {
            "size": 608,
            "sha256": "486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9",
        },
        "model.safetensors": {
            "size": 2200119864,
            "sha256": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
        },
    },
}
GEOMETRY: dict[str, int] = {
    "num_hidden_layers": 22,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
}
TOKENIZER_ARTIFACTS: tuple[str, ...] = tuple(
    tokenizer_provenance.REQUIRED_ARTIFACTS
)
FORCED_IDS: tuple[int, ...] = (3, 4, 5, 6)
EOS_ID = 2
BOS_ID = 1


class OfficialReferenceError(RuntimeError):
    """An expected, contextual failure in export or verification."""


def _fail(message: str) -> None:
    raise OfficialReferenceError(message)


def _load_json(path: Path, label: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        _fail(f"cannot read {label} {path}: {error}")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError) as error:
        _fail(f"{label} {path} is not valid finite UTF-8 JSON: {error}")
    raise AssertionError("unreachable")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a JSON object; actual {type(value).__name__}")
    return dict(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string; actual {value!r}")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer; actual {value!r}")
    if minimum is not None and value < minimum:
        _fail(f"{label} must be >= {minimum}; actual {value}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number; actual {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite; actual {value!r}")
    return result


def _sha256_file(path: Path, relative: str | None = None) -> dict[str, Any]:
    label = relative or str(path)
    try:
        file_stat = path.lstat()
    except OSError as error:
        _fail(f"cannot inspect artifact {label!r} at {path}: {error}")
    if not stat.S_ISREG(file_stat.st_mode):
        if stat.S_ISLNK(file_stat.st_mode):
            kind = "a symbolic link"
        elif stat.S_ISDIR(file_stat.st_mode):
            kind = "a directory"
        else:
            kind = "a non-regular file"
        _fail(f"artifact {label!r} at {path} must be a regular file; actual {kind}")

    digest = hashlib.sha256()
    byte_count = 0
    descriptor = -1
    try:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            opened_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                _fail(f"artifact {label!r} changed to a non-regular file while opening")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
            final_stat = os.fstat(stream.fileno())
    except OfficialReferenceError:
        raise
    except OSError as error:
        _fail(f"cannot read artifact {label!r} at {path}: {error}")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if byte_count != final_stat.st_size:
        _fail(
            f"artifact {label!r} changed while reading; expected "
            f"{final_stat.st_size} bytes, read {byte_count}"
        )
    return {
        "path": label,
        "size": byte_count,
        "sha256": digest.hexdigest(),
    }


def _valid_relative_path(value: Any, label: str) -> str:
    path = _string(value, label)
    parsed = Path(path)
    if parsed.is_absolute() or not parsed.parts or any(
        part in ("", ".", "..") for part in parsed.parts
    ):
        _fail(f"{label} must be a normalized relative path; actual {path!r}")
    return parsed.as_posix()


def _model_directory(model_dir: Path) -> Path:
    try:
        directory_stat = model_dir.lstat()
    except OSError as error:
        _fail(f"caller-supplied model directory {model_dir} is unavailable: {error}")
    if not stat.S_ISDIR(directory_stat.st_mode):
        _fail(f"caller-supplied model path {model_dir} must be a directory")
    try:
        return model_dir.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve caller-supplied model directory {model_dir}: {error}")
    raise AssertionError("unreachable")


def _artifact_path(model_dir: Path, relative: str) -> Path:
    rel = _valid_relative_path(relative, "artifact path")
    path = model_dir / Path(rel)
    try:
        resolved = path.resolve(strict=True)
        root = model_dir.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve artifact {relative!r}: {error}")
    if resolved != root and root not in resolved.parents:
        _fail(f"artifact {relative!r} escapes the caller-supplied model directory")
    return path


def _check_generation_environment() -> dict[str, Any]:
    try:
        runtime, packages = tokenizer_provenance.check_environment(
            expected_runtime=PINNED_RUNTIME,
            expected_packages=PINNED_PACKAGES,
        )
    except tokenizer_provenance.ProvenanceError as error:
        _fail(str(error))
    return {"runtime": runtime, "packages": packages}


def _load_policy(path: Path) -> dict[str, Any]:
    policy = _object(_load_json(path, "official export policy"), "official export policy")
    expected_fields = {
        "artifacts",
        "digest",
        "exporter",
        "forced_continuation",
        "geometry",
        "kind",
        "model",
        "packages",
        "precision",
        "prompts",
        "runtime",
        "schema_version",
        "stop_policy",
        "tolerances",
    }
    if set(policy) != expected_fields:
        _fail(
            "official export policy fields mismatch: expected "
            f"{sorted(expected_fields)!r}, actual {sorted(policy)!r}"
        )
    if policy.get("schema_version") != 1:
        _fail(f"official export policy schema_version must be 1; actual {policy.get('schema_version')!r}")
    if policy.get("kind") != "official":
        _fail(f"official export policy kind must be 'official'; actual {policy.get('kind')!r}")

    runtime = _object(policy.get("runtime"), "official export policy runtime")
    if runtime != PINNED_RUNTIME:
        _fail(f"official export policy runtime is not pinned: expected {PINNED_RUNTIME!r}, actual {runtime!r}")
    packages = _object(policy.get("packages"), "official export policy packages")
    if packages != PINNED_PACKAGES:
        _fail(
            "official export policy packages must equal the frozen pins: "
            f"expected {PINNED_PACKAGES!r}, actual {packages!r}"
        )
    model = _object(policy.get("model"), "official export policy model")
    if model != PINNED_MODEL:
        _fail(
            "official export policy model identity is not pinned: "
            f"expected {PINNED_MODEL!r}, actual {model!r}"
        )
    geometry = _object(policy.get("geometry"), "official export policy geometry")
    if geometry != GEOMETRY:
        _fail(f"official export policy geometry mismatch: expected {GEOMETRY!r}, actual {geometry!r}")

    precision = _object(policy.get("precision"), "official export policy precision")
    expected_precision = {
        "weights": "BF16",
        "activations": "BF16",
        "logits_storage": "BF16",
        "reference_arithmetic": "FP32",
        "attention": "eager",
        "device": "cpu",
    }
    if precision != expected_precision:
        _fail(f"official export policy precision mismatch: expected {expected_precision!r}, actual {precision!r}")

    tolerances = _object(policy.get("tolerances"), "official export policy tolerances")
    expected_tolerances = {
        "absolute": 0.53125,
        "relative": 0.02,
        "formula": "abs(actual-ref) <= 0.53125 + 0.02*abs(ref)",
        "tie_policy": "lowest-id",
        "exact_token": "reference-margin-certified-only",
    }
    if tolerances != expected_tolerances:
        _fail(f"official export policy tolerances mismatch: expected {expected_tolerances!r}, actual {tolerances!r}")

    artifacts = _object(policy.get("artifacts"), "official export policy artifacts")
    expected_artifacts = {
        "tokenizer_manifest": "test/reference/tokenizer_reference_manifest.json",
        "tokenizer_oracles": "test/reference/tokenizer_oracles.json",
        "tokenizer_files": list(TOKENIZER_ARTIFACTS),
        "config": "config.json",
        "safetensors": "all-consumed-shards",
    }
    if artifacts != expected_artifacts:
        _fail(f"official export policy artifacts mismatch: expected {expected_artifacts!r}, actual {artifacts!r}")

    prompts = _object(policy.get("prompts"), "official export policy prompts")
    expected_prompts = {
        "raw": {
            "mode": "raw",
            "text": "The capital of France is",
            "add_special_tokens": True,
            "require_bos": True,
            "max_new_tokens": 4,
        },
        "chat": {
            "mode": "chat",
            "messages": [{"role": "user", "content": "Hello."}],
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "assistant_prefix": "<|assistant|>\n",
            "expected_rendered": "<|user|>\nHello.</s>\n<|assistant|>\n",
            "max_new_tokens": 4,
        },
        "zero": {
            "mode": "raw",
            "text": "The capital of France is",
            "add_special_tokens": True,
            "require_bos": True,
            "max_new_tokens": 0,
        },
    }
    if prompts != expected_prompts:
        _fail(f"official export policy prompts mismatch: expected {expected_prompts!r}, actual {prompts!r}")

    forced = _object(policy.get("forced_continuation"), "official export policy forced continuation")
    if forced != {
        "ids": list(FORCED_IDS),
        "selection_policy": "fixed-reference-continuation",
    }:
        _fail(f"official export policy forced continuation is not frozen: actual {forced!r}")

    stop_policy = _object(policy.get("stop_policy"), "official export policy stop_policy")
    expected_stop_policy = {
        "eos_before_max_new_tokens": True,
        "max_new_tokens_before_context_exhaustion": True,
        "terminal_token_no_kv_append": True,
        "zero_limit_no_forward": True,
    }
    if stop_policy != expected_stop_policy:
        _fail(f"official export policy stop_policy mismatch: expected {expected_stop_policy!r}, actual {stop_policy!r}")

    digest = _object(policy.get("digest"), "official export policy digest")
    expected_digest = {
        "field": "provenance.case_payload_sha256",
        "encoding": "json.dumps(cases, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)",
    }
    if digest != expected_digest:
        _fail(f"official export policy digest mismatch: expected {expected_digest!r}, actual {digest!r}")
    if policy.get("exporter") != "test/reference/export_official_model_reference.py":
        _fail("official export policy exporter must identify the committed exporter")
    return policy




def _collect_artifact_paths(model_dir: Path) -> list[str]:
    required = ["config.json", *TOKENIZER_ARTIFACTS]
    try:
        entries = list(model_dir.iterdir())
    except OSError as error:
        _fail(f"cannot enumerate caller-supplied model directory {model_dir}: {error}")
    safetensors = sorted(
        entry.name
        for entry in entries
        if entry.name.endswith(".safetensors")
    )
    if not safetensors:
        _fail("official distribution contains no SafeTensors shard")
    paths = [*required, *safetensors]
    index = model_dir / "model.safetensors.index.json"
    if index.exists() or index.is_symlink():
        paths.append(index.name)
    return sorted(set(paths))


def _collect_artifacts(model_dir: Path) -> list[dict[str, Any]]:
    paths = _collect_artifact_paths(model_dir)
    records = [_sha256_file(model_dir / relative, relative) for relative in paths]
    index_record = next(
        (record for record in records if record["path"] == "model.safetensors.index.json"),
        None,
    )
    if index_record is not None:
        index = _object(
            _load_json(model_dir / "model.safetensors.index.json", "SafeTensors index"),
            "SafeTensors index",
        )
        weight_map = _object(index.get("weight_map"), "SafeTensors index weight_map")
        listed = set()
        for key, shard in weight_map.items():
            _string(key, "SafeTensors index tensor name")
            shard_name = _valid_relative_path(shard, "SafeTensors index shard")
            if not shard_name.endswith(".safetensors"):
                _fail(f"SafeTensors index shard {shard_name!r} is not a SafeTensors file")
            listed.add(shard_name)
        actual_shards = {
            record["path"] for record in records if record["path"].endswith(".safetensors")
        }
        if not listed or not listed.issubset(actual_shards):
            _fail(
                "SafeTensors index references missing shards: "
                f"referenced {sorted(listed)!r}, available {sorted(actual_shards)!r}"
            )
    return records


def _validate_config(config: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    config_dict = dict(config)
    if config_dict.get("model_type") != "llama":
        _fail(
            "official model config model_type must be 'llama'; actual "
            f"{config_dict.get('model_type')!r}"
        )
    if config_dict.get("architectures") != ["LlamaForCausalLM"]:
        _fail(
            "official model config architectures must be ['LlamaForCausalLM']; "
            f"actual {config_dict.get('architectures')!r}"
        )
    for field, expected in GEOMETRY.items():
        if field == "head_dim":
            actual = config_dict.get(field, GEOMETRY["hidden_size"] // GEOMETRY["num_attention_heads"])
        else:
            actual = config_dict.get(field)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            _fail(f"official model config {field} must be {expected}; actual {actual!r}")
    if config_dict.get("tie_word_embeddings") is not False:
        _fail("official TinyLlama reference requires untied lm_head (tie_word_embeddings=false)")
    if config_dict.get("torch_dtype") not in ("bfloat16", "torch.bfloat16"):
        _fail(
            "official TinyLlama reference requires BF16 torch_dtype; actual "
            f"{config_dict.get('torch_dtype')!r}"
        )
    if config_dict.get("rope_scaling") not in (None, {}):
        _fail("official TinyLlama reference does not support rope_scaling")
    expected_model = _object(policy["model"], "official export policy model")
    expected_model_id = _string(expected_model.get("id"), "official model ID")
    declared_model_id = None
    for key in ("model_id", "_name_or_path", "name_or_path"):
        candidate = config_dict.get(key)
        if isinstance(candidate, str) and candidate:
            declared_model_id = candidate
            break
    if declared_model_id is not None and declared_model_id != expected_model_id:
        _fail(
            "official model config identity differs from the pinned artifact: "
            f"expected {expected_model_id!r}, actual {declared_model_id!r}"
        )
    normalized = dict(GEOMETRY)
    normalized.update(
        {
            "model_type": "llama",
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "model_id": expected_model_id,
        }
    )
    return normalized


def _load_tokenizer_oracle_configuration() -> dict[str, Any]:
    oracle = _object(_load_json(TOKENIZER_ORACLES, "tokenizer oracle"), "tokenizer oracle")
    chat = _object(oracle.get("chat"), "tokenizer oracle chat")
    configuration = _object(chat.get("configuration"), "tokenizer oracle chat configuration")
    for key in ("chat_template", "bos_token", "eos_token"):
        _string(configuration.get(key), f"tokenizer oracle chat configuration {key}")
    return configuration


def _load_generation_stack(model_dir: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    # Set both offline guards before importing any HF module.  Import-time
    # cache discovery must not be able to consult a network-backed hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        from transformers import AutoTokenizer
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    except Exception as error:
        _fail(f"cannot import pinned offline HF generation stack: {error}")
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, use_fast=True
        )
        model = LlamaForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    except Exception as error:
        _fail(f"cannot load the caller-supplied offline HF Llama distribution: {error}")
    try:
        model.eval()
        model.config._attn_implementation = "eager"
        if model.training:
            _fail("official HF Llama model remained in training mode")
        if torch.get_num_threads() != 1:
            _fail("official HF generation must use exactly one torch thread")
        if not torch.are_deterministic_algorithms_enabled():
            _fail("official HF generation did not enable deterministic algorithms")
        parameters = list(model.named_parameters())
        if not parameters:
            _fail("official HF Llama model has no parameters")
        for name, parameter in parameters:
            if parameter.device.type != "cpu":
                _fail(
                    f"official HF Llama parameter {name!r} is on "
                    f"{parameter.device.type!r}, expected CPU"
                )
            if parameter.dtype is not torch.bfloat16:
                _fail(
                    f"official HF Llama parameter {name!r} has dtype "
                    f"{parameter.dtype}, expected torch.bfloat16"
                )
        if (
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        ):
            _fail("official TinyLlama reference requires physically untied embeddings")
    except OfficialReferenceError:
        raise
    except Exception as error:
        _fail(f"cannot configure HF Llama for eager deterministic evaluation: {error}")

    oracle_configuration = _load_tokenizer_oracle_configuration()
    actual_template = getattr(tokenizer, "chat_template", None)
    if actual_template != oracle_configuration["chat_template"]:
        _fail("official tokenizer chat_template differs from the pinned tokenizer oracle")
    for key in ("bos_token", "eos_token"):
        if getattr(tokenizer, key, None) != oracle_configuration[key]:
            _fail(
                f"official tokenizer {key} differs from the pinned tokenizer oracle: "
                f"expected {oracle_configuration[key]!r}, actual {getattr(tokenizer, key, None)!r}"
            )
    return torch, tokenizer, model, oracle_configuration


def _token_ids(value: Any, label: str, vocabulary: int) -> list[int]:
    if not isinstance(value, (list, tuple)):
        _fail(f"{label} must be a list of token IDs")
    result = []
    for index, token in enumerate(value):
        token_id = _integer(token, f"{label}[{index}]")
        if token_id < 0 or token_id >= vocabulary:
            _fail(f"{label}[{index}]={token_id} is outside vocabulary [0,{vocabulary})")
        result.append(token_id)
    return result


def _render_prompts(tokenizer: Any, policy: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    prompts = _object(policy["prompts"], "policy prompts")
    raw_policy = _object(prompts["raw"], "raw prompt policy")
    chat_policy = _object(prompts["chat"], "chat prompt policy")
    zero_policy = _object(prompts["zero"], "zero prompt policy")

    raw_text = _string(raw_policy["text"], "raw prompt text")
    raw_bytes = list(raw_text.encode("utf-8"))
    try:
        raw_ids = list(tokenizer.encode(raw_text, add_special_tokens=True))
    except Exception as error:
        _fail(f"cannot tokenize fixed raw prompt: {error}")
    raw_ids = _token_ids(raw_ids, "raw prompt IDs", GEOMETRY["vocab_size"])
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id != BOS_ID or not raw_ids or raw_ids[0] != BOS_ID:
        _fail(
            "fixed raw prompt BOS policy failed: expected tokenizer BOS ID 1 at "
            f"the first position, actual tokenizer BOS {bos_token_id!r}, IDs {raw_ids[:4]!r}"
        )

    messages = chat_policy["messages"]
    if not isinstance(messages, list) or messages != [{"role": "user", "content": "Hello."}]:
        _fail("fixed chat messages are malformed")
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_ids_direct = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        chat_ids_encoded = tokenizer.encode(rendered, add_special_tokens=False)
    except Exception as error:
        _fail(f"cannot render fixed distribution chat prompt: {error}")
    if not isinstance(rendered, str):
        _fail("distribution chat template did not return UTF-8 text")
    expected_rendered = _string(chat_policy["expected_rendered"], "chat expected rendered text")
    if rendered != expected_rendered:
        _fail(
            "distribution chat rendering differs from the fixed assistant-prefix policy: "
            f"expected {expected_rendered!r}, actual {rendered!r}"
        )
    assistant_prefix = _string(chat_policy["assistant_prefix"], "chat assistant prefix")
    if not rendered.endswith(assistant_prefix):
        _fail("distribution chat rendering does not end with the fixed assistant prefix")
    if list(chat_ids_direct) != list(chat_ids_encoded):
        _fail("chat tokenizer direct template IDs differ from encoding its rendered bytes")
    chat_ids = _token_ids(chat_ids_direct, "chat prompt IDs", GEOMETRY["vocab_size"])
    if not chat_ids or chat_ids[0] == BOS_ID:
        _fail("chat prompt BOS policy requires add_special_tokens=false and no leading BOS")

    if zero_policy != raw_policy | {"max_new_tokens": 0}:
        _fail("zero-new-token policy must be the fixed raw prompt with max_new_tokens=0")

    return {
        "raw": {
            "mode": "raw",
            "selection_policy": "production-greedy",
            "prompt_ids": raw_ids,
            "rendered_utf8": raw_bytes,
            "text": raw_text,
            "positions": list(range(len(raw_ids))),
            "bos_policy": {
                "add_special_tokens": True,
                "require_bos": True,
                "bos_token_id": BOS_ID,
            },
            "max_new_tokens": 4,
        },
        "chat": {
            "mode": "chat",
            "selection_policy": "production-greedy",
            "prompt_ids": chat_ids,
            "rendered_utf8": list(rendered.encode("utf-8")),
            "messages": messages,
            "positions": list(range(len(chat_ids))),
            "bos_policy": {
                "add_special_tokens": False,
                "require_bos": False,
                "bos_token_id": BOS_ID,
            },
            "max_new_tokens": 4,
        },
        "zero": {
            "mode": "raw",
            "selection_policy": "none",
            "prompt_ids": list(raw_ids),
            "rendered_utf8": raw_bytes,
            "text": raw_text,
            "positions": list(range(len(raw_ids))),
            "bos_policy": {
                "add_special_tokens": True,
                "require_bos": True,
                "bos_token_id": BOS_ID,
            },
            "max_new_tokens": 0,
        },
    }


def _greedy_id(logits: Sequence[float]) -> int:
    if not logits:
        _fail("reference model returned an empty vocabulary")
    winner = 0
    winner_value = logits[0]
    for index, value in enumerate(logits[1:], 1):
        if value > winner_value:
            winner = index
            winner_value = value
    return winner


def _margin_stable(logits: Sequence[float], winner: int) -> bool:
    winner_value = logits[winner]
    winner_error = 0.53125 + 0.02 * abs(winner_value)
    for index, value in enumerate(logits):
        if index == winner:
            continue
        error = 0.53125 + 0.02 * abs(value)
        if not (winner_value - winner_error > value + error):
            return False
    return True


def _finite_logits(logits: Any, torch: Any, label: str) -> list[float]:
    try:
        if not bool(torch.isfinite(logits).all().item()):
            _fail(f"{label} contains non-finite logits")
        # The model's final values are explicitly rounded through BF16 RNE,
        # then represented as finite FP32 JSON numbers.
        stored = logits.detach().to(dtype=torch.float32).to(dtype=torch.bfloat16).to(dtype=torch.float32)
        values = [float(value) for value in stored.cpu().tolist()]
    except OfficialReferenceError:
        raise
    except Exception as error:
        _fail(f"cannot convert {label} to BF16-stored FP32 logits: {error}")
    if len(values) != GEOMETRY["vocab_size"]:
        _fail(f"{label} has {len(values)} entries; expected {GEOMETRY['vocab_size']}")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            _fail(f"{label}[{index}] is non-finite after BF16 storage")
    return values




def _forward_reference(
    model: Any,
    torch: Any,
    prompt_ids: Sequence[int],
    selection_policy: str,
    max_new_tokens: int,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    if max_new_tokens == 0:
        return [], [], {
            # A zero-limit request is an admitted max_new_tokens stop with no
            # forward and no initialized KV state.
            "stop_reason": "max_new_tokens",
            "kv_sequence_length": 0,
            "cached_decode_forwards": 0,
            "committed_decode_tokens": 0,
            "next_position": 0,
            "terminal_token_no_kv_append": True,
        }
    if len(prompt_ids) > GEOMETRY["max_position_embeddings"]:
        _fail("prompt exceeds the official context capacity")
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
    except Exception as error:
        _fail(f"official model prefill failed: {error}")

    forced = list(FORCED_IDS)
    decode_ids: list[int] = []
    snapshots: list[dict[str, Any]] = []
    past = outputs.past_key_values
    prefix = list(prompt_ids)
    next_logits = outputs.logits[0, -1, :]
    step = 0
    stop_reason = "max_new_tokens"
    while True:
        logits_for_step = next_logits
        if selection_policy == "production-greedy":
            # Convert once for the snapshot and select from exactly the stored
            # BF16 reference values, preserving the lowest-ID tie policy.
            stored_logits = _finite_logits(logits_for_step, torch, "reference logits")
            candidate = _greedy_id(stored_logits)
            snapshot = {
                "phase": "prefill" if step == 0 else "cached-decode",
                "prefix_ids": list(prefix),
                "position_start": 0 if step == 0 else len(prefix) - 1,
                "run_length": len(prefix) if step == 0 else 1,
                "absolute_positions": list(range(0, len(prefix)))
                if step == 0
                else [len(prefix) - 1],
                "logits": stored_logits,
                "greedy_id": candidate,
                "selected_id": candidate,
                "margin_stable": _margin_stable(stored_logits, candidate),
            }
        else:
            if step >= len(forced):
                _fail("fixed continuation exhausted before max_new_tokens")
            stored_logits = _finite_logits(logits_for_step, torch, "reference logits")
            candidate = forced[step]
            snapshot = {
                "phase": "prefill" if step == 0 else "cached-decode",
                "prefix_ids": list(prefix),
                "position_start": 0 if step == 0 else len(prefix) - 1,
                "run_length": len(prefix) if step == 0 else 1,
                "absolute_positions": list(range(0, len(prefix)))
                if step == 0
                else [len(prefix) - 1],
                "logits": stored_logits,
                "greedy_id": _greedy_id(stored_logits),
                "selected_id": candidate,
                "margin_stable": _margin_stable(stored_logits, _greedy_id(stored_logits)),
            }
        snapshots.append(snapshot)
        decode_ids.append(candidate)
        step += 1

        # EOS is considered before max_new_tokens.  The selected terminal ID
        # is committed to the result, but no cached forward appends it.
        if selection_policy == "production-greedy" and candidate == EOS_ID:
            stop_reason = "eos"
            break
        if step >= max_new_tokens:
            stop_reason = "max_new_tokens"
            break
        if len(prompt_ids) + step >= GEOMETRY["max_position_embeddings"]:
            stop_reason = "context_exhaustion"
            break

        token = torch.tensor([[candidate]], dtype=torch.long)
        prefix.append(candidate)
        decode_attention = torch.ones((1, len(prefix)), dtype=torch.long)
        position_ids = torch.tensor([[len(prefix) - 1]], dtype=torch.long)
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=token,
                    attention_mask=decode_attention,
                    position_ids=position_ids,
                    past_key_values=past,
                    use_cache=True,
                )
        except Exception as error:
            _fail(f"official model cached decode failed at token {step}: {error}")
        past = outputs.past_key_values
        next_logits = outputs.logits[0, -1, :]

    expected_state = {
        "stop_reason": stop_reason,
        "kv_sequence_length": len(prompt_ids) + max(0, len(decode_ids) - 1),
        "cached_decode_forwards": max(0, len(decode_ids) - 1),
        "committed_decode_tokens": len(decode_ids),
        "next_position": len(prompt_ids) + max(0, len(decode_ids) - 1),
        "terminal_token_no_kv_append": True,
    }
    if selection_policy == "fixed-reference-continuation":
        if decode_ids != forced:
            _fail(f"fixed continuation IDs changed: expected {forced!r}, actual {decode_ids!r}")
        expected_state["stop_reason"] = "max_new_tokens"
    return decode_ids, snapshots, expected_state

def _build_case(
    case_id: str,
    prompt: Mapping[str, Any],
    selection_policy: str,
    decode_ids: Sequence[int],
    snapshots: Sequence[Mapping[str, Any]],
    expected_state: Mapping[str, Any],
) -> dict[str, Any]:
    input_record = {
        "mode": prompt["mode"],
        "selection_policy": selection_policy,
        "prompt_ids": list(prompt["prompt_ids"]),
        "rendered_utf8": list(prompt["rendered_utf8"]),
        "positions": list(prompt["positions"]),
        "decode_ids": list(decode_ids),
        "max_new_tokens": prompt["max_new_tokens"],
        "bos_policy": dict(prompt["bos_policy"]),
    }
    if prompt["mode"] == "raw":
        input_record["text"] = prompt["text"]
    else:
        input_record["messages"] = prompt["messages"]
    return {
        "id": case_id,
        "input": input_record,
        "snapshots": [dict(snapshot) for snapshot in snapshots],
        "expected_result": {
            "token_ids": list(decode_ids),
            "stop_reason": expected_state["stop_reason"],
            "initialized_kv_length": expected_state["kv_sequence_length"],
        },
    }


def _case_payload_digest(cases: Any) -> str:
    try:
        payload = json.dumps(
            cases,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        _fail(f"cannot serialize official case payload for digest: {error}")
    return hashlib.sha256(payload).hexdigest()


def _script_digest() -> str:
    return _sha256_file(SCRIPT_PATH, SCRIPT_PATH.name)["sha256"]


def _publish(document: Mapping[str, Any], output: Path, model_dir: Path) -> None:
    try:
        output_resolved = output.resolve(strict=False)
        model_resolved = model_dir.resolve(strict=True)
    except OSError as error:
        _fail(f"cannot resolve official reference output {output}: {error}")
    if output_resolved == model_resolved or model_resolved in output_resolved.parents:
        _fail("official reference output must not be written inside the caller-supplied model directory")
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            _fail(f"official reference output {output} must be a regular file when replacing it")
    if not output.parent.exists() or not output.parent.is_dir():
        _fail(f"official reference output parent does not exist: {output.parent}")
    try:
        payload = (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        _fail(f"cannot serialize official reference pack: {error}")
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    except OSError as error:
        _fail(f"cannot atomically publish official reference pack {output}: {error}")
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _export(args: argparse.Namespace, policy: Mapping[str, Any]) -> None:
    model_dir = _model_directory(Path(args.model_dir))
    artifact_id = _string(args.artifact_id, "--artifact-id")
    expected_model = _object(policy["model"], "official export policy model")
    expected_revision = _string(
        expected_model.get("revision"), "official model revision"
    )
    if artifact_id != expected_revision:
        _fail(
            f"--artifact-id must be the pinned official revision "
            f"{expected_revision!r}; actual {artifact_id!r}"
        )
    output = Path(args.output)
    environment = _check_generation_environment()
    config = _object(
        _load_json(model_dir / "config.json", "official model config"),
        "official model config",
    )
    model_config = _validate_config(config, policy)
    artifacts = _collect_artifacts(model_dir)
    normalized_artifacts = {
        _string(record["path"], "official artifact path"): record
        for record in artifacts
    }
    _validate_tokenizer_manifest_records(normalized_artifacts)
    _validate_pinned_model_artifacts(normalized_artifacts, policy)
    torch, tokenizer, model, _ = _load_generation_stack(model_dir)
    prompts = _render_prompts(tokenizer, policy)

    raw_greedy_ids, raw_greedy_snapshots, raw_state = _forward_reference(
        model, torch, prompts["raw"]["prompt_ids"], "production-greedy", 4
    )
    raw_forced_ids, raw_forced_snapshots, raw_forced_state = _forward_reference(
        model, torch, prompts["raw"]["prompt_ids"], "fixed-reference-continuation", 4
    )
    chat_greedy_ids, chat_greedy_snapshots, chat_state = _forward_reference(
        model, torch, prompts["chat"]["prompt_ids"], "production-greedy", 4
    )
    chat_forced_ids, chat_forced_snapshots, chat_forced_state = _forward_reference(
        model, torch, prompts["chat"]["prompt_ids"], "fixed-reference-continuation", 4
    )
    _, _, zero_state = _forward_reference(
        model, torch, prompts["zero"]["prompt_ids"], "none", 0
    )

    cases = [
        _build_case(
            "raw-production-greedy",
            prompts["raw"],
            "production-greedy",
            raw_greedy_ids,
            raw_greedy_snapshots,
            raw_state,
        ),
        _build_case(
            "raw-fixed-reference-continuation",
            prompts["raw"],
            "fixed-reference-continuation",
            raw_forced_ids,
            raw_forced_snapshots,
            raw_forced_state,
        ),
        _build_case(
            "chat-production-greedy",
            prompts["chat"],
            "production-greedy",
            chat_greedy_ids,
            chat_greedy_snapshots,
            chat_state,
        ),
        _build_case(
            "chat-fixed-reference-continuation",
            prompts["chat"],
            "fixed-reference-continuation",
            chat_forced_ids,
            chat_forced_snapshots,
            chat_forced_state,
        ),
        _build_case(
            "zero-new-token",
            prompts["zero"],
            "none",
            [],
            [],
            zero_state,
        ),
    ]
    provenance = {
        "runtime": environment["runtime"],
        "packages": environment["packages"],
        "generator_sha256": _script_digest(),
        "command": list(sys.argv),
        "precision": dict(policy["precision"]),
        "model_id": model_config["model_id"],
        "revision": artifact_id,
        "artifacts": artifacts,
        "case_payload_sha256": _case_payload_digest(cases),
    }
    document = {
        "schema_version": 1,
        "kind": "official",
        "provenance": provenance,
        "tolerances": dict(policy["tolerances"]),
        "model_config": model_config,
        "artifact_id": artifact_id,
        "cases": cases,
    }
    _publish(document, output, model_dir)


def _validate_runtime_record(provenance: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    runtime = _object(provenance.get("runtime"), "pack provenance runtime")
    expected_runtime = _object(policy["runtime"], "policy runtime")
    if runtime != expected_runtime:
        _fail(f"pack runtime policy mismatch: expected {expected_runtime!r}, actual {runtime!r}")
    packages = _object(provenance.get("packages"), "pack provenance packages")
    if packages != PINNED_PACKAGES:
        _fail(
            "pack package policy mismatch: expected exact full versions "
            f"{PINNED_PACKAGES!r}, actual {packages!r}"
        )
    if _object(provenance.get("precision"), "pack provenance precision") != _object(policy["precision"], "policy precision"):
        _fail("pack precision policy mismatch")

def _resolve_recorded_path(value: Any, label: str) -> Path:
    recorded = _string(value, label)
    try:
        return Path(recorded).resolve(strict=True)
    except OSError as error:
        _fail(f"{label} cannot be resolved: {error}")
    raise AssertionError("unreachable")


def _validate_generation_command(
    value: Any,
    model_dir: Path,
    artifact_id: str,
    policy_path: Path,
    pack_path: Path,
) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(argument, str) or not argument for argument in value)
    ):
        _fail("pack provenance command must be a non-empty list of strings")
    command = list(value)
    expected_script = Path("test/reference") / SCRIPT_PATH.name
    recorded_script = Path(command[0]).as_posix()
    if (
        recorded_script != expected_script.as_posix()
        and not recorded_script.endswith(f"/{expected_script.as_posix()}")
    ):
        _fail(
            "pack provenance command does not name the pinned exporter: "
            f"actual {command[0]!r}"
        )
    if (len(command) - 1) % 2 != 0:
        _fail("pack provenance command options must be explicit flag/value pairs")
    options: dict[str, str] = {}
    for index in range(1, len(command), 2):
        flag = command[index]
        argument = command[index + 1]
        if flag not in ("--model-dir", "--artifact-id", "--policy", "--output"):
            _fail(f"pack provenance command contains unsupported option {flag!r}")
        if flag in options:
            _fail(f"pack provenance command repeats option {flag!r}")
        options[flag] = argument
    expected_flags = {"--model-dir", "--artifact-id", "--policy", "--output"}
    if set(options) != expected_flags:
        _fail(
            "pack provenance command options mismatch: expected "
            f"{sorted(expected_flags)!r}, actual {sorted(options)!r}"
        )
    if _resolve_recorded_path(
        options["--model-dir"], "pack provenance --model-dir"
    ) != model_dir:
        _fail("pack provenance --model-dir differs from the verified model directory")
    if options["--artifact-id"] != artifact_id:
        _fail("pack provenance --artifact-id differs from the verified revision")
    if _resolve_recorded_path(
        options["--policy"], "pack provenance --policy"
    ) != policy_path.resolve(strict=True):
        _fail("pack provenance --policy differs from the verified policy")
    if _resolve_recorded_path(
        options["--output"], "pack provenance --output"
    ) != pack_path.resolve(strict=True):
        _fail("pack provenance --output differs from the verified pack")


def _validate_tokenizer_manifest_records(
    normalized: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest = _object(
        _load_json(TOKENIZER_MANIFEST, "committed tokenizer reference manifest"),
        "committed tokenizer reference manifest",
    )
    manifest_artifacts = _object(
        manifest.get("artifacts"),
        "committed tokenizer reference manifest artifacts",
    )
    if set(manifest_artifacts) != set(TOKENIZER_ARTIFACTS):
        _fail(
            "committed tokenizer artifact set mismatch: expected "
            f"{sorted(TOKENIZER_ARTIFACTS)!r}, actual {sorted(manifest_artifacts)!r}"
        )
    if _object(manifest.get("runtime"), "committed tokenizer runtime") != tokenizer_provenance.PINNED_RUNTIME:
        _fail("committed tokenizer manifest runtime differs from the shared authority")
    if _object(manifest.get("packages"), "committed tokenizer packages") != tokenizer_provenance.PINNED_PACKAGES:
        _fail("committed tokenizer manifest packages differ from the shared authority")
    for relative in TOKENIZER_ARTIFACTS:
        expected = _object(
            manifest_artifacts[relative],
            f"committed tokenizer artifact {relative}",
        )
        if set(expected) != {"name", "size", "sha256"}:
            _fail(f"committed tokenizer artifact {relative!r} has malformed fields")
        if expected["name"] != relative:
            _fail(f"committed tokenizer artifact {relative!r} has a mismatched name")
        recorded = normalized.get(relative)
        if recorded is None:
            _fail(f"pack provenance omits pinned tokenizer artifact {relative!r}")
        if (
            recorded.get("size") != expected["size"]
            or recorded.get("sha256") != expected["sha256"]
        ):
            _fail(
                f"pack tokenizer artifact {relative!r} differs from the committed "
                "tokenizer reference manifest"
            )
def _validate_pinned_model_artifacts(
    normalized: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    model = _object(policy.get("model"), "official export policy model")
    expected_artifacts = _object(
        model.get("artifacts"),
        "official export policy model artifacts",
    )
    expected_safetensors = {
        relative
        for relative in expected_artifacts
        if relative.endswith(".safetensors")
    }
    actual_safetensors = {
        relative for relative in normalized if relative.endswith(".safetensors")
    }
    if actual_safetensors != expected_safetensors:
        _fail(
            "official SafeTensors artifact set mismatch: expected "
            f"{sorted(expected_safetensors)!r}, actual {sorted(actual_safetensors)!r}"
        )
    for relative, identity_value in expected_artifacts.items():
        identity = _object(
            identity_value,
            f"official export policy model artifact {relative}",
        )
        if set(identity) != {"size", "sha256"}:
            _fail(f"official model artifact policy for {relative!r} is malformed")
        recorded = normalized.get(relative)
        if recorded is None:
            _fail(f"official distribution omits pinned artifact {relative!r}")
        if (
            recorded.get("size") != identity["size"]
            or recorded.get("sha256") != identity["sha256"]
        ):
            _fail(
                f"official model artifact {relative!r} differs from revision "
                f"{model['revision']!r}"
            )






def _validate_artifact_records(
    provenance: Mapping[str, Any],
    model_dir: Path,
    policy: Mapping[str, Any],
) -> None:
    records = provenance.get("artifacts")
    if not isinstance(records, list) or not records:
        _fail("pack provenance artifacts must be a non-empty list")
    normalized: dict[str, dict[str, Any]] = {}
    for index, record_value in enumerate(records):
        record = _object(record_value, f"pack artifact[{index}]")
        if set(record) != {"path", "size", "sha256"}:
            _fail(f"pack artifact[{index}] has unexpected fields: {sorted(record)!r}")
        relative = _valid_relative_path(record["path"], f"pack artifact[{index}] path")
        if relative in normalized:
            _fail(f"pack provenance contains duplicate artifact {relative!r}")
        size = _integer(record["size"], f"pack artifact {relative} size", minimum=0)
        digest = record["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest or any(
            character not in "0123456789abcdef" for character in digest
        ):
            _fail(f"pack artifact {relative!r} has an invalid lowercase SHA-256 digest")
        normalized[relative] = {"path": relative, "size": size, "sha256": digest}

    expected_paths = set(_collect_artifact_paths(model_dir))
    if set(normalized) != expected_paths:
        _fail(
            "pack artifact inventory differs from the explicit model directory: "
            f"expected {sorted(expected_paths)!r}, actual {sorted(normalized)!r}"
        )
    _validate_tokenizer_manifest_records(normalized)
    _validate_pinned_model_artifacts(normalized, policy)
    for relative in sorted(normalized):
        actual = _sha256_file(_artifact_path(model_dir, relative), relative)
        expected = normalized[relative]
        if actual["size"] != expected["size"] or actual["sha256"] != expected["sha256"]:
            _fail(
                f"artifact identity mismatch for {relative!r}: expected "
                f"size {expected['size']} sha256 {expected['sha256']}, actual "
                f"size {actual['size']} sha256 {actual['sha256']}"
            )


def _validate_model_config(pack: Mapping[str, Any], model_dir: Path, policy: Mapping[str, Any]) -> None:
    config = _object(_load_json(model_dir / "config.json", "official model config"), "official model config")
    expected = _validate_config(config, policy)
    actual = _object(pack.get("model_config"), "pack model_config")
    if actual != expected:
        _fail(f"pack model_config does not match the explicit model directory: expected {expected!r}, actual {actual!r}")


def _validate_bytes(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        _fail(f"{label} must be a list of UTF-8 byte values")
    result = []
    for index, item in enumerate(value):
        byte = _integer(item, f"{label}[{index}]")
        if byte < 0 or byte > 255:
            _fail(f"{label}[{index}]={byte} is outside byte range")
        result.append(byte)
    return result


def _validate_snapshot(
    snapshot_value: Any,
    label: str,
    prefix: Sequence[int],
    expected_phase: str,
    expected_start: int,
    expected_length: int,
    expected_selected: int,
) -> tuple[int, bool]:
    snapshot = _object(snapshot_value, label)
    expected_fields = {
        "phase",
        "prefix_ids",
        "position_start",
        "run_length",
        "absolute_positions",
        "logits",
        "greedy_id",
        "selected_id",
        "margin_stable",
    }
    if set(snapshot) != expected_fields:
        _fail(f"{label} fields mismatch: expected {sorted(expected_fields)!r}, actual {sorted(snapshot)!r}")
    if snapshot["phase"] != expected_phase:
        _fail(f"{label} phase mismatch: expected {expected_phase!r}, actual {snapshot['phase']!r}")
    if _token_ids(snapshot["prefix_ids"], f"{label}.prefix_ids", GEOMETRY["vocab_size"]) != list(prefix):
        _fail(f"{label}.prefix_ids do not match its logical cached prefix")
    if _integer(snapshot["position_start"], f"{label}.position_start", minimum=0) != expected_start:
        _fail(f"{label}.position_start is inconsistent with its logical prefix")
    if _integer(snapshot["run_length"], f"{label}.run_length", minimum=1) != expected_length:
        _fail(f"{label}.run_length is inconsistent with its phase")
    expected_positions = list(range(expected_start, expected_start + expected_length))
    if snapshot["absolute_positions"] != expected_positions:
        _fail(f"{label}.absolute_positions mismatch: expected {expected_positions!r}, actual {snapshot['absolute_positions']!r}")
    logits_value = snapshot["logits"]
    if not isinstance(logits_value, list) or len(logits_value) != GEOMETRY["vocab_size"]:
        _fail(f"{label}.logits must contain exactly {GEOMETRY['vocab_size']} vocabulary values")
    logits = [_finite(value, f"{label}.logits[{index}") for index, value in enumerate(logits_value)]
    greedy = _integer(snapshot["greedy_id"], f"{label}.greedy_id", minimum=0)
    selected = _integer(snapshot["selected_id"], f"{label}.selected_id", minimum=0)
    if greedy >= GEOMETRY["vocab_size"] or selected >= GEOMETRY["vocab_size"]:
        _fail(f"{label} token ID exceeds vocabulary size")
    if selected != expected_selected:
        _fail(f"{label}.selected_id mismatch: expected {expected_selected}, actual {selected}")
    computed_greedy = _greedy_id(logits)
    if greedy != computed_greedy:
        _fail(f"{label}.greedy_id is not the lowest-ID greedy value")
    computed_margin = _margin_stable(logits, computed_greedy)
    if snapshot["margin_stable"] is not computed_margin:
        _fail(f"{label}.margin_stable is not derived from reference logits")
    return greedy, computed_margin


def _validate_case(case_value: Any, index: int, policy: Mapping[str, Any]) -> None:
    case = _object(case_value, f"case[{index}]")
    expected_case_fields = {"id", "input", "snapshots", "expected_result"}
    if set(case) != expected_case_fields:
        _fail(
            f"case[{index}] fields mismatch: expected {sorted(expected_case_fields)!r}, "
            f"actual {sorted(case)!r}"
        )
    case_id = _string(case["id"], f"case[{index}].id")
    expected_names = {
        "raw-production-greedy",
        "raw-fixed-reference-continuation",
        "chat-production-greedy",
        "chat-fixed-reference-continuation",
        "zero-new-token",
    }
    if case_id not in expected_names:
        _fail(f"case[{index}] has unsupported id {case_id!r}")

    input_record = _object(case["input"], f"case {case_id}.input")
    input_fields = {
        "mode",
        "selection_policy",
        "prompt_ids",
        "rendered_utf8",
        "positions",
        "decode_ids",
        "max_new_tokens",
        "bos_policy",
        "text",
    }
    if input_record.get("mode") == "chat":
        input_fields.remove("text")
        input_fields.add("messages")
    if set(input_record) != input_fields:
        _fail(
            f"case {case_id}.input fields mismatch: expected {sorted(input_fields)!r}, "
            f"actual {sorted(input_record)!r}"
        )
    mode = input_record["mode"]
    if mode not in ("raw", "chat"):
        _fail(f"case {case_id}.input.mode is unsupported: {mode!r}")
    expected_mode = "chat" if case_id.startswith("chat-") else "raw"
    if mode != expected_mode:
        _fail(f"case {case_id} must use the fixed {expected_mode!r} prompt mode")
    selection_policy = input_record["selection_policy"]
    if selection_policy not in (
        "production-greedy",
        "fixed-reference-continuation",
        "none",
    ):
        _fail(f"case {case_id}.input.selection_policy is unsupported")
    prompt_ids = _token_ids(
        input_record["prompt_ids"],
        f"case {case_id}.input.prompt_ids",
        GEOMETRY["vocab_size"],
    )
    rendered_bytes = _validate_bytes(
        input_record["rendered_utf8"], f"case {case_id}.input.rendered_utf8"
    )
    if input_record["positions"] != list(range(len(prompt_ids))):
        _fail(f"case {case_id}.input.positions must be contiguous absolute prompt positions")
    decode_ids = _token_ids(
        input_record["decode_ids"],
        f"case {case_id}.input.decode_ids",
        GEOMETRY["vocab_size"],
    )
    max_new_tokens = _integer(
        input_record["max_new_tokens"],
        f"case {case_id}.input.max_new_tokens",
        minimum=0,
    )
    bos_policy = _object(
        case["input"]["bos_policy"], f"case {case_id}.input.bos_policy"
    )
    if set(bos_policy) != {"add_special_tokens", "require_bos", "bos_token_id"}:
        _fail(f"case {case_id}.input.bos_policy fields are malformed")
    if _integer(
        bos_policy["bos_token_id"],
        f"case {case_id}.input.bos_policy.bos_token_id",
        minimum=0,
    ) != BOS_ID:
        _fail(f"case {case_id} has an unsupported BOS token ID")

    if mode == "raw":
        prompt_policy = (
            policy["prompts"]["zero"]
            if case_id == "zero-new-token"
            else policy["prompts"]["raw"]
        )
        if input_record["text"] != prompt_policy["text"]:
            _fail(f"case {case_id} raw text differs from the fixed policy")
        if rendered_bytes != list(prompt_policy["text"].encode("utf-8")):
            _fail(f"case {case_id} raw rendered bytes differ from fixed UTF-8 input")
        if bos_policy != {
            "add_special_tokens": True,
            "require_bos": True,
            "bos_token_id": BOS_ID,
        }:
            _fail(f"case {case_id} raw BOS policy mismatch")
        if not prompt_ids or prompt_ids[0] != BOS_ID:
            _fail(f"case {case_id} raw prompt does not record the required BOS")
    else:
        prompt_policy = policy["prompts"]["chat"]
        if input_record["messages"] != prompt_policy["messages"]:
            _fail(f"case {case_id} chat messages differ from the fixed policy")
        if bytes(rendered_bytes).decode("utf-8") != prompt_policy["expected_rendered"]:
            _fail(
                f"case {case_id} rendered chat bytes differ from the fixed "
                "distribution template"
            )
        if bos_policy != {
            "add_special_tokens": False,
            "require_bos": False,
            "bos_token_id": BOS_ID,
        }:
            _fail(f"case {case_id} chat BOS policy mismatch")
        if not prompt_ids or prompt_ids[0] == BOS_ID:
            _fail(f"case {case_id} chat prompt unexpectedly has a BOS")
    if max_new_tokens != prompt_policy["max_new_tokens"]:
        _fail(f"case {case_id} max_new_tokens differs from fixed policy")

    if case_id.endswith("fixed-reference-continuation"):
        if selection_policy != "fixed-reference-continuation" or decode_ids != list(FORCED_IDS):
            _fail(
                f"case {case_id} must carry the exact fixed-reference continuation "
                f"{list(FORCED_IDS)!r}"
            )
    elif case_id == "zero-new-token":
        if selection_policy != "none" or decode_ids:
            _fail("zero-new-token case must have selection_policy=none and empty output")
    else:
        if selection_policy != "production-greedy":
            _fail(f"case {case_id} must be labeled production-greedy")
        if len(decode_ids) > 4 or not decode_ids:
            _fail(f"case {case_id} must contain one through four production greedy IDs")
        if EOS_ID in decode_ids[:-1]:
            _fail(f"case {case_id} contains tokens after its EOS")

    snapshots = case["snapshots"]
    if not isinstance(snapshots, list):
        _fail(f"case {case_id}.snapshots must be a list")
    if len(snapshots) != len(decode_ids):
        _fail(
            f"case {case_id} must contain exactly one snapshot per selected output token"
        )
    for snapshot_index, snapshot in enumerate(snapshots):
        prefix = prompt_ids + decode_ids[:snapshot_index]
        phase = "prefill" if snapshot_index == 0 else "cached-decode"
        start = 0 if snapshot_index == 0 else len(prefix) - 1
        length = len(prefix) if snapshot_index == 0 else 1
        _validate_snapshot(
            snapshot,
            f"case {case_id}.snapshots[{snapshot_index}]",
            prefix,
            phase,
            start,
            length,
            decode_ids[snapshot_index],
        )

    result = _object(case["expected_result"], f"case {case_id}.expected_result")
    expected_result_fields = {"token_ids", "stop_reason", "initialized_kv_length"}
    if set(result) != expected_result_fields:
        _fail(
            f"case {case_id}.expected_result fields mismatch: expected "
            f"{sorted(expected_result_fields)!r}, actual {sorted(result)!r}"
        )
    if result["token_ids"] != decode_ids:
        _fail(f"case {case_id}.expected_result.token_ids do not match input.decode_ids")
    if case_id == "zero-new-token":
        expected_stop = "max_new_tokens"
        expected_kv = 0
    elif selection_policy == "fixed-reference-continuation":
        expected_stop = "max_new_tokens"
        expected_kv = len(prompt_ids) + len(decode_ids) - 1
    elif decode_ids[-1] == EOS_ID:
        expected_stop = "eos"
        expected_kv = len(prompt_ids) + len(decode_ids) - 1
    elif len(decode_ids) == max_new_tokens:
        expected_stop = "max_new_tokens"
        expected_kv = len(prompt_ids) + len(decode_ids) - 1
    elif len(prompt_ids) + len(decode_ids) >= GEOMETRY["max_position_embeddings"]:
        expected_stop = "context_exhaustion"
        expected_kv = len(prompt_ids) + len(decode_ids) - 1
    else:
        _fail(f"case {case_id} has no valid documented stop condition")
    if result["stop_reason"] != expected_stop:
        _fail(
            f"case {case_id} stop_reason mismatch: expected {expected_stop!r}, "
            f"actual {result['stop_reason']!r}"
        )
    if result["initialized_kv_length"] != expected_kv:
        _fail(f"case {case_id} initialized KV length violates terminal-token policy")


def _verify(args: argparse.Namespace, policy: Mapping[str, Any]) -> None:
    model_dir = _model_directory(Path(args.model_dir))
    pack_path = Path(args.verify)
    pack = _object(_load_json(pack_path, "official reference pack"), "official reference pack")
    expected_top = {"schema_version", "kind", "provenance", "tolerances", "model_config", "artifact_id", "cases"}
    if set(pack) != expected_top:
        _fail(f"official reference pack fields mismatch: expected {sorted(expected_top)!r}, actual {sorted(pack)!r}")
    if pack["schema_version"] != 1 or pack["kind"] != "official":
        _fail("official reference pack schema or kind mismatch")
    artifact_id = _string(pack["artifact_id"], "official reference pack artifact_id")
    expected_model = _object(policy["model"], "official export policy model")
    expected_revision = _string(
        expected_model.get("revision"), "official model revision"
    )
    if artifact_id != expected_revision:
        _fail(
            "official reference pack artifact_id mismatch: expected pinned revision "
            f"{expected_revision!r}, actual {artifact_id!r}"
        )
    provenance = _object(pack["provenance"], "official reference pack provenance")
    expected_provenance_fields = {
        "artifacts",
        "case_payload_sha256",
        "command",
        "generator_sha256",
        "model_id",
        "packages",
        "precision",
        "revision",
        "runtime",
    }
    if set(provenance) != expected_provenance_fields:
        _fail(
            "official reference provenance fields mismatch: expected "
            f"{sorted(expected_provenance_fields)!r}, actual {sorted(provenance)!r}"
        )
    _validate_runtime_record(provenance, policy)
    revision = _string(provenance.get("revision"), "pack provenance revision")
    expected_revision = _string(
        expected_model.get("revision"),
        "official model revision",
    )
    if revision != expected_revision:
        _fail(
            "pack provenance revision mismatch: expected "
            f"{expected_revision!r}, actual {revision!r}"
        )
    if _object(pack["tolerances"], "pack tolerances") != _object(policy["tolerances"], "policy tolerances"):
        _fail("official reference pack tolerance policy mismatch")
    generator_digest = _string(provenance.get("generator_sha256"), "pack generator_sha256")
    if generator_digest != _script_digest():
        _fail(f"official reference pack generator digest mismatch: expected {_script_digest()}")
    model_config = _object(pack.get("model_config"), "pack model_config")
    if _string(provenance.get("model_id"), "pack provenance model_id") != _string(model_config.get("model_id"), "pack model_config model_id"):
        _fail("pack provenance model_id does not match model_config")
    _validate_generation_command(
        provenance.get("command"),
        model_dir,
        artifact_id,
        Path(args.policy),
        pack_path,
    )
    _validate_artifact_records(provenance, model_dir, policy)
    _validate_model_config(pack, model_dir, policy)
    cases = pack["cases"]
    if not isinstance(cases, list) or len(cases) != 5:
        _fail("official reference pack must contain exactly five fixed cases")
    names: list[str] = []
    for index, case in enumerate(cases):
        _validate_case(case, index, policy)
        case_object = _object(case, f"case[{index}]")
        names.append(_string(case_object.get("id"), f"case[{index}].id"))
    expected_names = [
        "raw-production-greedy",
        "raw-fixed-reference-continuation",
        "chat-production-greedy",
        "chat-fixed-reference-continuation",
        "zero-new-token",
    ]
    if names != expected_names:
        _fail(f"official reference cases must be in fixed order: expected {expected_names!r}, actual {names!r}")
    expected_digest = _string(provenance.get("case_payload_sha256"), "pack case_payload_sha256")
    if len(expected_digest) != 64 or expected_digest.lower() != expected_digest or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        _fail("pack case_payload_sha256 is not a lowercase SHA-256 digest")
    actual_digest = _case_payload_digest(cases)
    if expected_digest != actual_digest:
        _fail(
            "official reference case payload digest mismatch: "
            f"expected {expected_digest}, actual {actual_digest}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export or stdlib-verify the pinned offline official TinyLlama reference pack."
    )
    parser.add_argument("--model-dir", metavar="PATH")
    parser.add_argument("--artifact-id", metavar="ID")
    parser.add_argument("--output", metavar="PATH")
    parser.add_argument("--policy", metavar="PATH")
    parser.add_argument("--verify", metavar="PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.policy is None:
            _fail("export and verification require an explicit --policy")
        policy = _load_policy(Path(args.policy))
        if args.verify is not None:
            if args.model_dir is None:
                _fail("--verify requires an explicit --model-dir")
            if args.artifact_id is not None or args.output is not None:
                _fail("--verify cannot be combined with --artifact-id or --output")
            _verify(args, policy)
        else:
            if args.model_dir is None or args.artifact_id is None or args.output is None:
                _fail("export requires explicit --model-dir, --artifact-id, and --output")
            _export(args, policy)
    except OfficialReferenceError as error:
        print(f"official reference export: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"official reference export: unexpected failure: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

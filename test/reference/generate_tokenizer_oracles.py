#!/usr/bin/env python3
"""Generate deterministic, offline tokenizer and chat reference oracles.

The committed tokenizer_reference module is the sole authority for the pinned
runtime, package, and distribution identity.  This generator only loads a
synthetic descriptor produced by the recipe below (and, when explicitly
requested, the caller's local tokenizer.json) after those checks succeed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from tokenizer_reference import (
        REQUIRED_ARTIFACTS,
        ProvenanceError,
        build_manifest,
    )
except ImportError:  # pragma: no cover - supports package-style imports.
    from test.reference.tokenizer_reference import (  # type: ignore
        REQUIRED_ARTIFACTS,
        ProvenanceError,
        build_manifest,
    )


GENERATOR_VERSION = "1"
DESCRIPTOR_RECIPE_VERSION = "synthetic-bpe-v1"
VOCAB_SIZE = 32_000
SPECIAL_PIECES: tuple[tuple[int, str], ...] = (
    (0, "<unk>"),
    (1, "<s>"),
    (2, "</s>"),
)
FIXED_TEST_PIECES: tuple[str, ...] = (
    "▁",
    "h",
    "e",
    "l",
    "o",
    "he",
    "ll",
    "llo",
    "hello",
    "▁hello",
    "w",
    "r",
    "d",
    "wo",
    "wor",
    "worl",
    "world",
    "▁world",
    "é",
    "ø",
    "世",
    "界",
    "▁héllø",
    "▁世界",
    "!",
    "?",
)
FIXED_MERGE_CHAIN: tuple[tuple[str, str, str], ...] = (
    ("h", "e", "he"),
    ("l", "l", "ll"),
    ("ll", "o", "llo"),
    ("he", "llo", "hello"),
    ("▁", "hello", "▁hello"),
    ("w", "o", "wo"),
    ("wo", "r", "wor"),
    ("wor", "l", "worl"),
    ("worl", "d", "world"),
    ("▁", "world", "▁world"),
)
FILLER_PREFIX = "<reserved_"
FILLER_SUFFIX = ">"
FILLER_WIDTH = 5
BYTE_PIECE_FIRST_ID = 3
BYTE_PIECE_COUNT = 256
TEST_PIECE_FIRST_ID = BYTE_PIECE_FIRST_ID + BYTE_PIECE_COUNT
FILLER_FIRST_ID = TEST_PIECE_FIRST_ID + len(FIXED_TEST_PIECES)
FILLER_COUNT = VOCAB_SIZE - FILLER_FIRST_ID

# These names and values intentionally remain caller-independent.  The default
# template is read from the pinned caller-supplied tokenizer_config.json.
CHAT_CASES: tuple[tuple[str, tuple[tuple[str, str], ...], bool], ...] = (
    ("empty_chat", (), False),
    ("empty_user", (("user", ""),), False),
    (
        "system_user_no_generation",
        (("system", "system"), ("user", "hello")),
        False,
    ),
    (
        "system_user_generation",
        (("system", "system"), ("user", "hello")),
        True,
    ),
    (
        "assistant_content",
        (("user", "hello"), ("assistant", "world")),
        False,
    ),
)
FIXED_OVERRIDE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'OVERRIDE_USER:' + message['content'] + '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'OVERRIDE_GENERATION:' }}"
    "{% endif %}"
)

RAW_TEXT_CASES: tuple[tuple[str, str, bool], ...] = (
    ("empty", "", True),
    ("one_ascii_space", " ", True),
    ("two_ascii_spaces", "  ", True),
    ("ascii_word", "hello", True),
    ("repeated_merge", "hello hello", True),
    ("tab", "\t", True),
    ("newline", "\n", True),
    ("unicode_héllø_世界", "héllø 世界", True),
    ("emoji_byte_fallback", "🙂", True),
    ("adjacent_added_tokens", "<unk>hello<s></s>", True),
    ("single_special_true", "hello", True),
    ("single_special_false", "hello", False),
)
PAIR_CASES: tuple[tuple[str, str, str], ...] = (
    ("pair_empty_empty", "", ""),
    ("pair_empty_text", "", "hello"),
    ("pair_text_empty", "hello", ""),
    ("pair_text_text", "hello", "world"),
)
INVALID_UTF8 = bytes((0xF0, 0x28, 0x8C, 0x28))


class OracleError(RuntimeError):
    """A contextual failure that must leave an existing oracle untouched."""


def _filler_name(index: int) -> str:
    return f"{FILLER_PREFIX}{index:0{FILLER_WIDTH}d}{FILLER_SUFFIX}"


def _byte_piece(value: int) -> str:
    return f"<0x{value:02X}>"


def _build_synthetic_descriptor() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the complete fast-tokenizer descriptor and its compact recipe."""

    vocab: dict[str, int] = {piece: identifier for identifier, piece in SPECIAL_PIECES}
    for value in range(BYTE_PIECE_COUNT):
        vocab[_byte_piece(value)] = BYTE_PIECE_FIRST_ID + value
    for offset, piece in enumerate(FIXED_TEST_PIECES):
        vocab[piece] = TEST_PIECE_FIRST_ID + offset
    for offset in range(FILLER_COUNT):
        vocab[_filler_name(offset)] = FILLER_FIRST_ID + offset

    added_tokens = [
        {
            "content": piece,
            "id": identifier,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for identifier, piece in SPECIAL_PIECES
    ]
    descriptor: dict[str, Any] = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {
                    "type": "Replace",
                    "pattern": {"String": " "},
                    "content": "▁",
                },
            ],
        },
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
            },
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {"String": "▁"},
                    "content": " ",
                },
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": [f"{left} {right}" for left, right, _ in FIXED_MERGE_CHAIN],
        },
    }
    recipe: dict[str, Any] = {
        "recipe_version": DESCRIPTOR_RECIPE_VERSION,
        "tokenizer_schema_version": "1.0",
        "vocab_size": VOCAB_SIZE,
        "special_tokens": [
            {"id": identifier, "piece": piece} for identifier, piece in SPECIAL_PIECES
        ],
        "byte_fallback": {
            "first_id": BYTE_PIECE_FIRST_ID,
            "count": BYTE_PIECE_COUNT,
            "piece_format": "<0xHH>",
            "values": "0x00 through 0xFF in ascending order",
        },
        "fixed_test_pieces": list(FIXED_TEST_PIECES),
        "filler": {
            "first_id": FILLER_FIRST_ID,
            "count": FILLER_COUNT,
            "prefix": FILLER_PREFIX,
            "suffix": FILLER_SUFFIX,
            "width": FILLER_WIDTH,
            "index_origin": 0,
        },
        "merges": [
            {
                "rank": rank,
                "left": left,
                "right": right,
                "result": result,
            }
            for rank, (left, right, result) in enumerate(FIXED_MERGE_CHAIN)
        ],
        "added_tokens": [
            {
                "id": identifier,
                "content": piece,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for identifier, piece in SPECIAL_PIECES
        ],
        "pre_tokenizer": None,
        "model_options": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
        },
        "normalizer": "Sequence(Prepend('▁'), Replace(' ', '▁'))",
        "post_processor": "TemplateProcessing(single='<s> A', pair='<s> A <s>:1 B:1')",
        "decoder": "Sequence(Replace('▁', ' '), ByteFallback, Fuse, Strip(' ', 1, 0))",
    }
    _validate_synthetic_descriptor(descriptor, recipe)
    return descriptor, recipe


def _validate_synthetic_descriptor(
    descriptor: Mapping[str, Any], recipe: Mapping[str, Any]
) -> None:
    try:
        vocab = descriptor["model"]["vocab"]  # type: ignore[index]
    except (KeyError, TypeError):
        raise OracleError("synthetic descriptor model.vocab is missing") from None
    if not isinstance(vocab, Mapping):
        raise OracleError("synthetic descriptor model.vocab is not an object")
    if len(vocab) != VOCAB_SIZE:
        raise OracleError(
            f"synthetic descriptor vocabulary size mismatch: expected {VOCAB_SIZE}, "
            f"actual {len(vocab)}"
        )
    identifiers = list(vocab.values())
    if any(isinstance(identifier, bool) or not isinstance(identifier, int) for identifier in identifiers):
        raise OracleError("synthetic descriptor vocabulary contains a non-integer ID")
    if sorted(identifiers) != list(range(VOCAB_SIZE)):
        raise OracleError("synthetic descriptor vocabulary IDs are not contiguous 0..31999")
    for identifier, piece in SPECIAL_PIECES:
        if vocab.get(piece) != identifier:
            raise OracleError(
                f"synthetic descriptor special piece {piece!r} has ID {vocab.get(piece)!r}; "
                f"expected {identifier}"
            )
    for value in range(BYTE_PIECE_COUNT):
        piece = _byte_piece(value)
        expected = BYTE_PIECE_FIRST_ID + value
        if vocab.get(piece) != expected:
            raise OracleError(
                f"synthetic descriptor byte piece {piece!r} has ID {vocab.get(piece)!r}; "
                f"expected {expected}"
            )
    for offset, piece in enumerate(FIXED_TEST_PIECES):
        expected = TEST_PIECE_FIRST_ID + offset
        if vocab.get(piece) != expected:
            raise OracleError(
                f"synthetic descriptor test piece {piece!r} has ID {vocab.get(piece)!r}; "
                f"expected {expected}"
            )
    merges = descriptor["model"].get("merges")  # type: ignore[index]
    expected_merges = [f"{left} {right}" for left, right, _ in FIXED_MERGE_CHAIN]
    if merges != expected_merges:
        raise OracleError(
            f"synthetic descriptor merge chain mismatch: expected {expected_merges!r}, "
            f"actual {merges!r}"
        )
    for rank, (left, right, result) in enumerate(FIXED_MERGE_CHAIN):
        if left not in vocab or right not in vocab or result not in vocab:
            raise OracleError(
                f"synthetic descriptor merge {rank} references missing operand/result: "
                f"{left!r}, {right!r} -> {result!r}"
            )
    if recipe.get("recipe_version") != DESCRIPTOR_RECIPE_VERSION:
        raise OracleError("synthetic descriptor recipe version is not recorded")


def _load_reference_stack() -> tuple[Any, Any]:
    try:
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
    except Exception as error:  # pragma: no cover - depends on caller environment.
        raise OracleError(f"cannot import pinned tokenizer stack after provenance checks: {error}") from error
    return Tokenizer, PreTrainedTokenizerFast


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} {path} must contain a JSON object")
    return value


def _load_chat_configuration(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "tokenizer_config.json"
    configuration = _load_json_object(path, "authoritative tokenizer configuration")
    template = configuration.get("chat_template")
    if not isinstance(template, str) or not template:
        raise OracleError(
            f"authoritative tokenizer configuration {path} has no usable chat_template"
        )
    for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
        if not isinstance(configuration.get(name), str):
            raise OracleError(
                f"authoritative tokenizer configuration {path} has invalid {name!r}"
            )
    return {
        "chat_template": template,
        "bos_token": configuration["bos_token"],
        "eos_token": configuration["eos_token"],
        "unk_token": configuration["unk_token"],
        "pad_token": configuration["pad_token"],
        "clean_up_tokenization_spaces": configuration.get(
            "clean_up_tokenization_spaces", False
        ),
    }


def _build_tokenizer_from_descriptor(
    descriptor: Mapping[str, Any], Tokenizer: Any
) -> Any:
    try:
        descriptor_json = json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        tokenizer = Tokenizer.from_str(descriptor_json)
    except Exception as error:
        raise OracleError(f"cannot load generated synthetic tokenizer descriptor: {error}") from error
    return tokenizer


def _build_official_tokenizer(model_dir: Path, Tokenizer: Any) -> Any:
    path = model_dir / "tokenizer.json"
    try:
        return Tokenizer.from_file(str(path))
    except Exception as error:
        raise OracleError(f"cannot load official tokenizer.json {path}: {error}") from error


def _build_fast_wrapper(
    tokenizer: Any, configuration: Mapping[str, Any], PreTrainedTokenizerFast: Any
) -> Any:
    try:
        wrapper = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token=configuration["bos_token"],
            eos_token=configuration["eos_token"],
            unk_token=configuration["unk_token"],
            pad_token=configuration["pad_token"],
            clean_up_tokenization_spaces=bool(
                configuration.get("clean_up_tokenization_spaces", False)
            ),
        )
        wrapper.chat_template = configuration["chat_template"]
    except Exception as error:
        raise OracleError(f"cannot construct fast tokenizer wrapper: {error}") from error
    return wrapper


def _utf8_bytes(text: str, context: str) -> list[int]:
    try:
        return list(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise OracleError(f"{context} returned text that is not UTF-8 encodable: {error}") from error


def _decode_observations(tokenizer: Any, ids: Sequence[int], context: str) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for skip_special_tokens in (False, True):
        try:
            decoded = tokenizer.decode(
                list(ids), skip_special_tokens=skip_special_tokens
            )
        except Exception as error:
            raise OracleError(
                f"{context} decode failed for skip_special_tokens="
                f"{skip_special_tokens}: {error}"
            ) from error
        if not isinstance(decoded, str):
            raise OracleError(
                f"{context} decode returned {type(decoded).__name__}, expected str"
            )
        observations.append(
            {
                "decode_options": {
                    "skip_special_tokens": skip_special_tokens,
                },
                "decoded_bytes": _utf8_bytes(decoded, f"{context} decode"),
            }
        )
    return observations


def _measure_encoding(
    tokenizer: Any,
    first: str,
    second: str | None,
    add_special_tokens: bool,
    context: str,
    designated_lossless: bool,
    expected_lossless_bytes: Sequence[int] | None,
) -> dict[str, Any]:
    try:
        if second is None:
            encoding = tokenizer.encode(
                first, add_special_tokens=add_special_tokens
            )
        else:
            encoding = tokenizer.encode(
                first, pair=second, add_special_tokens=add_special_tokens
            )
    except Exception as error:
        raise OracleError(
            f"{context} encode failed with add_special_tokens={add_special_tokens}: {error}"
        ) from error

    ids = list(encoding.ids)
    type_ids_value = getattr(encoding, "type_ids", None)
    if type_ids_value is None:
        type_ids: list[int] | None = None
        type_ids_available = False
    else:
        type_ids = list(type_ids_value)
        type_ids_available = True
    try:
        special_tokens_mask = list(encoding.special_tokens_mask)
    except Exception as error:
        raise OracleError(f"{context} did not expose special_tokens_mask: {error}") from error
    observations = _decode_observations(tokenizer, ids, context)
    round_trip: dict[str, Any] = {
        "designated_lossless": designated_lossless,
        "checked_decode_options": None,
        "matches_input": None,
    }
    if designated_lossless:
        if expected_lossless_bytes is None:
            raise OracleError(f"{context} has no designated lossless input bytes")
        checked = observations[1]
        matches = checked["decoded_bytes"] == list(expected_lossless_bytes)
        round_trip["checked_decode_options"] = {"skip_special_tokens": True}
        round_trip["matches_input"] = matches
        if not matches:
            raise OracleError(
                f"{context} designated-lossless round trip mismatch: expected "
                f"{list(expected_lossless_bytes)!r}, actual {checked['decoded_bytes']!r}"
            )
    measured: dict[str, Any] = {
        "add_special_tokens": add_special_tokens,
        "ids": ids,
        "type_ids": type_ids,
        "type_ids_available": type_ids_available,
        "special_tokens_mask": special_tokens_mask,
        "decode_observations": observations,
        "round_trip": round_trip,
    }
    if second is not None and not type_ids_available:
        measured["type_ids_unavailable_reason"] = (
            "the pinned fast Encoding API did not expose type_ids"
        )
    return measured


def _measure_raw_corpus(tokenizer: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, text, selected_add_special_tokens in RAW_TEXT_CASES:
        input_bytes = _utf8_bytes(text, f"raw case {name} input")
        designated_lossless = name != "adjacent_added_tokens"
        encodings = [
            _measure_encoding(
                tokenizer,
                text,
                None,
                add_special_tokens,
                f"raw case {name}",
                designated_lossless=designated_lossless,
                expected_lossless_bytes=input_bytes if designated_lossless else None,
            )
            for add_special_tokens in (False, True)
        ]
        selected = next(
            record
            for record in encodings
            if record["add_special_tokens"] == selected_add_special_tokens
        )
        records.append(
            {
                "name": name,
                "input": text,
                "input_bytes": input_bytes,
                "is_pair": False,
                "encode_options": {"add_special_tokens": selected_add_special_tokens},
                "encode": selected,
                "encodings": encodings,
            }
        )

    invalid_encodings: list[dict[str, Any]] = []
    for add_special_tokens in (False, True):
        try:
            # Passing bytes intentionally avoids ever replacing invalid input
            # with a Python str using an error handler.
            tokenizer.encode(INVALID_UTF8, add_special_tokens=add_special_tokens)
        except (TypeError, ValueError, UnicodeError) as error:
            invalid_encodings.append(
                {
                    "add_special_tokens": add_special_tokens,
                    "status": "rejected",
                    "error_category": "invalid_utf8",
                    "error_type": type(error).__name__,
                }
            )
        except Exception as error:
            raise OracleError(
                "raw case invalid_utf8 received an unexpected reference exception "
                f"{type(error).__name__}: {error}"
            ) from error
        else:
            raise OracleError(
                "raw case invalid_utf8 unexpectedly accepted invalid UTF-8 bytes; "
                "the generator must not replace them"
            )
    records.append(
        {
            "name": "invalid_utf8",
            "input_bytes": list(INVALID_UTF8),
            "input_encoding": "raw_bytes",
            "is_pair": False,
            "encode_options": {"add_special_tokens": [False, True]},
            "status": "rejected",
            "expected_error_category": "invalid_utf8",
            "encodings": invalid_encodings,
        }
    )

    for name, first, second in PAIR_CASES:
        first_bytes = _utf8_bytes(first, f"raw case {name} first input")
        second_bytes = _utf8_bytes(second, f"raw case {name} second input")
        encodings = [
            _measure_encoding(
                tokenizer,
                first,
                second,
                add_special_tokens,
                f"raw case {name}",
                designated_lossless=False,
                expected_lossless_bytes=None,
            )
            for add_special_tokens in (False, True)
        ]
        selected = next(
            record for record in encodings if record["add_special_tokens"] is True
        )
        records.append(
            {
                "name": name,
                "input": {"first": first, "second": second},
                "input_bytes": {"first": first_bytes, "second": second_bytes},
                "is_pair": True,
                "encode_options": {"add_special_tokens": True},
                "encode": selected,
                "encodings": encodings,
            }
        )
    return records


def _messages_for_chat(
    entries: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in entries]


def _measure_chat_case(
    fast_tokenizer: Any,
    name: str,
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
    template: str,
    context: str,
    template_label: str,
) -> dict[str, Any]:
    try:
        rendered = fast_tokenizer.apply_chat_template(
            list(messages),
            chat_template=template,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        ids = fast_tokenizer.apply_chat_template(
            list(messages),
            chat_template=template,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
    except Exception as error:
        raise OracleError(f"{context} chat rendering failed: {error}") from error
    if not isinstance(rendered, str):
        raise OracleError(f"{context} chat rendering returned {type(rendered).__name__}")
    if not isinstance(ids, list) or any(
        isinstance(identifier, bool) or not isinstance(identifier, int) for identifier in ids
    ):
        raise OracleError(f"{context} chat tokenization returned malformed IDs: {ids!r}")
    try:
        direct = fast_tokenizer._tokenizer.encode(  # noqa: SLF001 - independent fast reference
            rendered, add_special_tokens=False
        )
    except Exception as error:
        raise OracleError(f"{context} direct fast-reference encoding failed: {error}") from error
    if list(direct.ids) != ids:
        raise OracleError(
            f"{context} chat ID mismatch between apply_chat_template and direct encoding: "
            f"apply={ids!r}, direct={list(direct.ids)!r}"
        )
    return {
        "name": name,
        "messages": [dict(message) for message in messages],
        "add_generation_prompt": add_generation_prompt,
        "template_label": template_label,
        "template_source": template,
        "encode_options": {"add_special_tokens": False},
        "rendered_bytes": _utf8_bytes(rendered, f"{context} rendered chat"),
        "ids": ids,
    }


def _measure_chat_corpus(
    fast_tokenizer: Any, configuration: Mapping[str, Any]
) -> dict[str, Any]:
    default_template = configuration["chat_template"]
    cases: list[dict[str, Any]] = []
    for name, entries, add_generation_prompt in CHAT_CASES:
        cases.append(
            _measure_chat_case(
                fast_tokenizer,
                name,
                _messages_for_chat(entries),
                add_generation_prompt,
                default_template,
                f"chat case {name}",
                "distribution",
            )
        )

    override_messages = _messages_for_chat((("user", "hello"),))
    default_override = _measure_chat_case(
        fast_tokenizer,
        "fixed_override_default_reference",
        override_messages,
        True,
        default_template,
        "chat case fixed_override default reference",
        "distribution",
    )
    override = _measure_chat_case(
        fast_tokenizer,
        "fixed_override",
        override_messages,
        True,
        FIXED_OVERRIDE_TEMPLATE,
        "chat case fixed_override",
        "fixed_override",
    )
    override_bytes = override["rendered_bytes"]
    default_bytes = default_override["rendered_bytes"]
    if override_bytes == default_bytes:
        raise OracleError(
            "chat case fixed_override did not change rendered bytes from distribution template"
        )
    override_text = bytes(override_bytes).decode("utf-8")
    if "OVERRIDE_USER:" not in override_text or "OVERRIDE_GENERATION:" not in override_text:
        raise OracleError(
            "chat case fixed_override markers are absent; override may have fallen back"
        )
    override["override_selected"] = True
    override["fallback_detected"] = False
    override["default_rendered_bytes"] = default_bytes
    override["selection_proof"] = {
        "user_literal_present": "OVERRIDE_USER:" in override_text,
        "generation_literal_present": "OVERRIDE_GENERATION:" in override_text,
        "differs_from_distribution": True,
    }
    cases.append(override)
    return {
        "configuration": {
            "chat_template": default_template,
            "bos_token": configuration["bos_token"],
            "eos_token": configuration["eos_token"],
            "unk_token": configuration["unk_token"],
            "pad_token": configuration["pad_token"],
            "clean_up_tokenization_spaces": configuration.get(
                "clean_up_tokenization_spaces", False
            ),
        },
        "cases": cases,
        "fixed_override_template": FIXED_OVERRIDE_TEMPLATE,
    }


def _measure_stably(
    measure: Callable[[], Any], context: str
) -> Any:
    first = measure()
    second = measure()
    if first != second:
        raise OracleError(
            f"{context} produced inconsistent repeated measurements: "
            f"first={first!r}, second={second!r}"
        )
    return first


def _identity_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OracleError("provenance manifest has no artifacts identity")
    return {
        "runtime": dict(manifest["runtime"]),
        "packages": dict(manifest["packages"]),
        "artifacts": {
            name: dict(artifacts[name]) for name in REQUIRED_ARTIFACTS
        },
    }


def _build_oracle_document(
    model_dir: Path,
    manifest: Mapping[str, Any],
    configuration: Mapping[str, Any],
    Tokenizer: Any,
    PreTrainedTokenizerFast: Any,
    official: bool,
) -> dict[str, Any]:
    descriptor, recipe = _build_synthetic_descriptor()
    synthetic_tokenizer = _build_tokenizer_from_descriptor(descriptor, Tokenizer)
    synthetic_fast = _build_fast_wrapper(
        synthetic_tokenizer, configuration, PreTrainedTokenizerFast
    )
    synthetic_raw = _measure_stably(
        lambda: _measure_raw_corpus(synthetic_tokenizer),
        "synthetic raw corpus",
    )
    synthetic_chat = _measure_stably(
        lambda: _measure_chat_corpus(synthetic_fast, configuration),
        "synthetic chat corpus",
    )

    if official:
        official_tokenizer = _build_official_tokenizer(model_dir, Tokenizer)
        official_fast = _build_fast_wrapper(
            official_tokenizer, configuration, PreTrainedTokenizerFast
        )
        official_raw = _measure_stably(
            lambda: _measure_raw_corpus(official_tokenizer),
            "official raw corpus",
        )
        official_chat = _measure_stably(
            lambda: _measure_chat_corpus(official_fast, configuration),
            "official chat corpus",
        )
        official_section: dict[str, Any] = {
            "enabled": True,
            "manifest_identity": _identity_from_manifest(manifest),
            "tokenizer_artifact": "tokenizer.json",
            "raw_text": {"cases": official_raw},
            "chat": official_chat,
        }
    else:
        official_section = {"enabled": False}

    return {
        "generator": {
            "name": "tinyllama-tokenizer-oracle-generator",
            "version": GENERATOR_VERSION,
            "serialization": {
                "encoding": "UTF-8",
                "newline": "\\n",
                "indent": 2,
                "sort_keys": True,
                "ensure_ascii": False,
            },
        },
        "provenance": {
            "manifest_identity": _identity_from_manifest(manifest),
            "manifest_generation": dict(manifest["generation"]),
        },
        "descriptor": recipe,
        "raw_text": {
            "tokenizer_source": "generated synthetic descriptor",
            "cases": synthetic_raw,
        },
        "chat": synthetic_chat,
        "official_distribution": official_section,
    }


def _serialize_oracle(document: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise OracleError(f"cannot serialize tokenizer oracle JSON: {error}") from error


def _reject_artifact_output(model_dir: Path, output: Path) -> None:
    try:
        output_resolved = output.resolve(strict=False)
        for name in REQUIRED_ARTIFACTS:
            artifact_resolved = (model_dir / name).resolve(strict=False)
            if output_resolved == artifact_resolved:
                raise OracleError(
                    f"oracle output {output} must not replace required artifact {name!r}"
                )
    except OSError as error:
        raise OracleError(f"cannot resolve oracle output {output}: {error}") from error


def _publish_atomically(document: Mapping[str, Any], output: Path, model_dir: Path) -> None:
    _reject_artifact_output(model_dir, output)
    payload = _serialize_oracle(document)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    except OSError as error:
        raise OracleError(f"cannot atomically publish tokenizer oracle {output}: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic offline TinyLlama tokenizer oracles."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        metavar="PATH",
        help="caller-supplied local distribution directory (never discovered)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=str(Path(__file__).resolve().with_name("tokenizer_oracles.json")),
        help="oracle destination (default: beside this script)",
    )
    parser.add_argument(
        "--expected-manifest",
        "--manifest",
        dest="expected_manifest",
        metavar="PATH",
        default=str(Path(__file__).resolve().with_name("tokenizer_reference_manifest.json")),
        help="task 01 manifest whose complete identity must match",
    )
    parser.add_argument(
        "--official-distribution",
        "--official",
        dest="official",
        action="store_true",
        help="also measure the caller's local tokenizer.json distribution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    expected_manifest = Path(args.expected_manifest)
    try:
        # This is deliberately the first operation that can inspect caller
        # artifacts or import reference dependencies.  task 01 performs the
        # runtime/package checks and hashes all five files before this point.
        manifest = build_manifest(
            model_dir,
            output=expected_manifest,
            expected_manifest=expected_manifest,
        )
        _reject_artifact_output(model_dir, output)
        configuration = _load_chat_configuration(model_dir)
        Tokenizer, PreTrainedTokenizerFast = _load_reference_stack()
        document = _build_oracle_document(
            model_dir,
            manifest,
            configuration,
            Tokenizer,
            PreTrainedTokenizerFast,
            args.official,
        )
        _publish_atomically(document, output, model_dir)
    except (OracleError, ProvenanceError) as error:
        print(f"tokenizer oracle generation: {error}", file=os.sys.stderr)
        return 2
    except Exception as error:  # Keep all reference failures contextual and nonzero.
        print(
            f"tokenizer oracle generation: unexpected reference failure: {error}",
            file=os.sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

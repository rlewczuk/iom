#!/usr/bin/env python3
"""Offline comparison of the fast TinyLlama tokenizer and tokenizer.model.

The fast ``tokenizer.json`` is the distribution authority.  The
SentencePiece model is loaded only as a diagnostic cross-check and is never
used as a runtime or fallback tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import tokenizer_reference as provenance


GENERATOR_NAME = "crosscheck_tokenizer_model"
GENERATOR_VERSION = "1"
REPORT_VERSION = "tinyllama-tokenizer-model-equivalence-v1"
CORPUS_VERSION = "tinyllama-tokenizer-corpus-v1"

# This is deliberately the same ordered corpus contract used by the oracle
# generator.  The bool is the fast tokenizer's add_special_tokens option.
TEXT_CASES: tuple[tuple[str, str, bool], ...] = (
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

INVALID_CASE = ("invalid_utf8", bytes((0xF0, 0x28, 0x8C, 0x28)))
PAIR_CASES: tuple[tuple[str, str, str], ...] = (
    ("pair_empty_empty", "", ""),
    ("pair_empty_text", "", "hello"),
    ("pair_text_empty", "hello", ""),
    ("pair_text_text", "hello", "world"),
)

# SentencePiece has no added-token table, pair encoder, or explicit
# skip-special-token decode policy.  These are named limitations rather than
# unexplained semantic failures.
KNOWN_LIMITATIONS: tuple[dict[str, Any], ...] = (
    {
        "code": "added-token-atomicity",
        "affected_cases": ["adjacent_added_tokens"],
        "description": (
            "tokenizer.json recognizes <unk>, <s>, and </s> as atomic added "
            "tokens; SentencePieceProcessor exposes only protobuf pieces and "
            "cannot reproduce that added-token segmentation"
        ),
    },
    {
        "code": "invalid-utf8-api-surface",
        "affected_cases": ["invalid_utf8"],
        "description": (
            "the fast tokenizer rejects the original invalid UTF-8 bytes while "
            "the pinned SentencePiece binding accepts a bytes object; both raw "
            "outcomes are recorded as not-comparable and neither is treated as "
            "a replacement-decoded equivalence"
        ),
    },
    {
        "code": "sentencepiece-bos-policy",
        "affected_cases": [
            "empty",
            "one_ascii_space",
            "two_ascii_spaces",
            "ascii_word",
            "repeated_merge",
            "tab",
            "newline",
            "unicode_héllø_世界",
            "emoji_byte_fallback",
            "adjacent_added_tokens",
            "single_special_true",
            "pair_empty_empty",
            "pair_empty_text",
            "pair_text_empty",
            "pair_text_text",
        ],
        "description": (
            "SentencePiece does not add BOS by default; this report selects "
            "the explicit bos encode option for the comparable special=true "
            "case and records the selected option per observation"
        ),
    },
    {
        "code": "sentencepiece-decode-policy",
        "affected_cases": [
            "empty",
            "one_ascii_space",
            "two_ascii_spaces",
            "ascii_word",
            "repeated_merge",
            "tab",
            "newline",
            "unicode_héllø_世界",
            "emoji_byte_fallback",
            "adjacent_added_tokens",
            "single_special_true",
            "pair_empty_empty",
            "pair_empty_text",
            "pair_text_empty",
            "pair_text_text",
        ],
        "description": (
            "SentencePieceProcessor has no skip_special_tokens argument and "
            "drops control symbols during DecodeIds; fast retained and skipped "
            "decodes are both recorded, with skipped output used for comparison"
        ),
    },
    {
        "code": "sentencepiece-no-pair-api",
        "affected_cases": [
            "pair_empty_empty",
            "pair_empty_text",
            "pair_text_empty",
            "pair_text_text",
        ],
        "description": (
            "SentencePieceProcessor exposes single-sequence encoding only; "
            "pair IDs and type IDs therefore cannot be compared"
        ),
    },
    {
        "code": "sentencepiece-no-type-ids",
        "affected_cases": [
            "pair_empty_empty",
            "pair_empty_text",
            "pair_text_empty",
            "pair_text_text",
        ],
        "description": (
            "TemplateProcessing type IDs are a fast-tokenizer surface and are "
            "not exposed by SentencePieceProcessor"
        ),
    },
    {
        "code": "sentencepiece-padding-metadata",
        "affected_cases": [],
        "description": (
            "A visible <pad> protobuf lookup or SentencePiece pad_id is not the "
            "tokenizer padding semantic; this distribution uses tokenizer pad "
            "ID 2 (EOS)"
        ),
    },
    {
        "code": "generation-padding-separation",
        "affected_cases": [],
        "description": (
            "generation_config.json pad_token_id 0 is generation metadata and is "
            "not tokenizer-model equivalence evidence"
        ),
    },
)


class CrosscheckError(RuntimeError):
    """A contextual, expected failure while creating the report."""


def _as_int_list(values: Any, context: str) -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise CrosscheckError(
            f"{context} returned {type(values).__name__}; expected a sequence of IDs"
        )
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CrosscheckError(
                f"{context} returned non-integer ID at index {index}: {value!r}"
            )
        result.append(int(value))
    return result


def _validate_ids(ids: Sequence[int], vocabulary_size: int, context: str) -> None:
    for index, token_id in enumerate(ids):
        if token_id < 0 or token_id >= vocabulary_size:
            raise CrosscheckError(
                f"{context} returned out-of-range ID at index {index}: {token_id}; "
                f"vocabulary size {vocabulary_size}"
            )


def _bytes(value: str, context: str) -> list[int]:
    try:
        return list(value.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise CrosscheckError(f"{context} returned text that is not UTF-8: {error}") from error


def _safe_message(error: BaseException, model_dir: Path) -> str:
    """Keep caller paths out of the deterministic report payload."""

    message = str(error)
    for candidate in (str(model_dir), str(model_dir.resolve(strict=False))):
        if candidate:
            message = message.replace(candidate, "<MODEL_DIR>")
    return message


def _decode_observation(
    tokenizer: Any,
    ids: Sequence[int],
    *,
    skip_special_tokens: bool,
    vocabulary_size: int,
    side: str,
    model_dir: Path,
) -> dict[str, Any]:
    """Decode without converting failures into an empty byte string."""

    _validate_ids(ids, vocabulary_size, f"{side} decode")
    try:
        text = tokenizer.decode(
            list(ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        # SentencePieceProcessor has no HF keyword surface.  This branch is
        # intentionally never used for the slow side; retaining it here makes
        # the failure explicit if a future API changes unexpectedly.
        raise
    except Exception as error:  # pragma: no cover - defensive API boundary
        return {
            "status": "rejected",
            "category": "decode-error",
            "message": _safe_message(error, model_dir),
            "bytes": None,
        }
    if not isinstance(text, str):
        return {
            "status": "rejected",
            "category": "decode-type-error",
            "message": f"{side} decode returned {type(text).__name__}, expected str",
            "bytes": None,
        }
    return {
        "status": "ok",
        "text": text,
        "bytes": _bytes(text, f"{side} decode"),
    }


def _fast_decode_observation(
    tokenizer: Any,
    ids: Sequence[int],
    *,
    skip_special_tokens: bool,
    vocabulary_size: int,
    model_dir: Path,
) -> dict[str, Any]:
    return _decode_observation(
        tokenizer,
        ids,
        skip_special_tokens=skip_special_tokens,
        vocabulary_size=vocabulary_size,
        side="fast",
        model_dir=model_dir,
    )


def _slow_decode_observation(
    processor: Any,
    ids: Sequence[int],
    *,
    vocabulary_size: int,
    model_dir: Path,
) -> dict[str, Any]:
    _validate_ids(ids, vocabulary_size, "slow decode")
    try:
        text = processor.decode(list(ids))
    except Exception as error:  # pragma: no cover - defensive API boundary
        return {
            "status": "rejected",
            "category": "decode-error",
            "message": _safe_message(error, model_dir),
            "bytes": None,
        }
    if not isinstance(text, str):
        return {
            "status": "rejected",
            "category": "decode-type-error",
            "message": f"slow decode returned {type(text).__name__}, expected str",
            "bytes": None,
        }
    return {
        "status": "ok",
        "text": text,
        "bytes": _bytes(text, "slow decode"),
    }


def _fast_encode(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
    model_dir: Path,
) -> tuple[list[int], list[int]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(
            f"fast encode failed for text case {text!r}: {_safe_message(error, model_dir)}"
        ) from error
    if not isinstance(encoded, Mapping):
        raise CrosscheckError(
            f"fast encode for text case {text!r} returned {type(encoded).__name__}"
        )
    ids = _as_int_list(encoded.get("input_ids"), "fast encode input_ids")
    type_ids = _as_int_list(
        encoded.get("token_type_ids", [0] * len(ids)),
        "fast encode token_type_ids",
    )
    if len(type_ids) != len(ids):
        raise CrosscheckError(
            f"fast encode token_type_ids length {len(type_ids)} does not match "
            f"input_ids length {len(ids)}"
        )
    return ids, type_ids


def _slow_encode(
    processor: Any,
    text: str,
    *,
    add_bos: bool,
    model_dir: Path,
) -> list[int]:
    # SetEncodeExtraOptions is the pinned SentencePiece API for the declared
    # BOS behavior.  Clearing it is important because the processor is reused.
    options = "bos" if add_bos else ""
    try:
        processor.SetEncodeExtraOptions(options)
        ids = processor.encode(text, out_type=int)
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(
            f"slow encode failed for text {text!r} with options {options!r}: "
            f"{_safe_message(error, model_dir)}"
        ) from error
    return _as_int_list(ids, "slow encode input_ids")


def _successful_decode(observation: Mapping[str, Any], context: str) -> list[int]:
    if observation.get("status") != "ok" or not isinstance(observation.get("bytes"), list):
        raise CrosscheckError(
            f"{context} did not produce comparable decoded bytes: {dict(observation)!r}"
        )
    return [int(value) for value in observation["bytes"]]


def _text_case(
    tokenizer: Any,
    processor: Any,
    *,
    name: str,
    text: str,
    add_special_tokens: bool,
    fast_vocabulary_size: int,
    slow_vocabulary_size: int,
    model_dir: Path,
) -> dict[str, Any]:
    input_bytes = _bytes(text, f"text case {name} input")
    fast_ids, fast_type_ids = _fast_encode(
        tokenizer,
        text,
        add_special_tokens=add_special_tokens,
        model_dir=model_dir,
    )
    slow_ids = _slow_encode(
        processor,
        text,
        add_bos=add_special_tokens,
        model_dir=model_dir,
    )
    _validate_ids(fast_ids, fast_vocabulary_size, f"fast encode case {name}")
    _validate_ids(slow_ids, slow_vocabulary_size, f"slow encode case {name}")

    fast_decode = {
        "skip_special_tokens_false": _fast_decode_observation(
            tokenizer,
            fast_ids,
            skip_special_tokens=False,
            vocabulary_size=fast_vocabulary_size,
            model_dir=model_dir,
        ),
        "skip_special_tokens_true": _fast_decode_observation(
            tokenizer,
            fast_ids,
            skip_special_tokens=True,
            vocabulary_size=fast_vocabulary_size,
            model_dir=model_dir,
        ),
    }
    slow_decode = _slow_decode_observation(
        processor,
        slow_ids,
        vocabulary_size=slow_vocabulary_size,
        model_dir=model_dir,
    )

    base: dict[str, Any] = {
        "name": name,
        "kind": "text",
        "input_text": text,
        "input_bytes": input_bytes,
        "options": {
            "add_special_tokens": add_special_tokens,
            "decode_policy": {
                "fast": [False, True],
                "slow": "SentencePieceProcessor.DecodeIds default",
            },
            "input_encoding": "utf-8",
            "slow_encode_extra_options": "bos" if add_special_tokens else "",
        },
        "fast": {
            "ids": fast_ids,
            "type_ids": fast_type_ids,
            "decode": fast_decode,
        },
        "slow": {
            "encode_options": {
                "extra_options": "bos" if add_special_tokens else "",
                "adds_bos": add_special_tokens,
            },
            "ids": slow_ids,
            "decode": slow_decode,
        },
    }

    # The added-token case is intentionally diagnostic-only: SentencePiece
    # cannot expose the fast tokenizer's atomic added-token table.
    if name == "adjacent_added_tokens":
        base["status"] = "not-comparable"
        base["comparison"] = {
            "status": "not-comparable",
            "reason_code": "added-token-atomicity",
            "reason": (
                "fast added-token segmentation and slow protobuf-piece "
                "segmentation are different API surfaces"
            ),
            "fast_ids": fast_ids,
            "slow_ids": slow_ids,
            "decoded_bytes": {
                "fast_skip_special_tokens_false": _successful_decode(
                    fast_decode["skip_special_tokens_false"],
                    f"fast case {name}",
                ),
                "fast_skip_special_tokens_true": _successful_decode(
                    fast_decode["skip_special_tokens_true"],
                    f"fast case {name}",
                ),
                "slow": _successful_decode(slow_decode, f"slow case {name}"),
            },
        }
        return base

    if fast_ids != slow_ids:
        base["status"] = "disagree"
        base["comparison"] = {
            "status": "disagree",
            "reason_code": "ids-mismatch",
            "fast_ids": fast_ids,
            "slow_ids": slow_ids,
        }
        raise CrosscheckError(
            f"required comparable case {name!r} has ID disagreement: "
            f"fast {fast_ids!r}, slow {slow_ids!r}"
        )

    # With explicit BOS the comparable decode is the fast API's skipped
    # special-token result.  With no special IDs both fast policies are
    # comparable to SentencePiece's default decode.
    if add_special_tokens:
        decode_policy = "fast.skip_special_tokens=true"
        fast_bytes = _successful_decode(
            fast_decode["skip_special_tokens_true"], f"fast case {name}"
        )
    else:
        decode_policy = "fast.skip_special_tokens=false"
        fast_bytes = _successful_decode(
            fast_decode["skip_special_tokens_false"], f"fast case {name}"
        )
    slow_bytes = _successful_decode(slow_decode, f"slow case {name}")
    if fast_bytes != slow_bytes:
        base["status"] = "disagree"
        base["comparison"] = {
            "status": "disagree",
            "reason_code": "decoded-bytes-mismatch",
            "policy": decode_policy,
            "fast_decoded_bytes": fast_bytes,
            "slow_decoded_bytes": slow_bytes,
        }
        raise CrosscheckError(
            f"required comparable case {name!r} has decoded-byte disagreement: "
            f"fast {fast_bytes!r}, slow {slow_bytes!r}"
        )

    base["status"] = "agree"
    base["comparison"] = {
        "status": "agree",
        "policy": decode_policy,
        "fast_ids_equal_slow_ids": True,
        "fast_decoded_bytes": fast_bytes,
        "slow_decoded_bytes": slow_bytes,
    }
    return base


def _invalid_utf8_case(
    tokenizer: Any,
    processor: Any,
    *,
    fast_vocabulary_size: int,
    slow_vocabulary_size: int,
    model_dir: Path,
) -> dict[str, Any]:
    name, raw = INVALID_CASE

    def fast_observation() -> dict[str, Any]:
        try:
            encoded = tokenizer(
                raw,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=True,
            )
            if not isinstance(encoded, Mapping):
                raise CrosscheckError(
                    f"fast invalid input returned {type(encoded).__name__}, "
                    "expected a mapping"
                )
            ids = _as_int_list(
                encoded.get("input_ids"),
                "fast invalid input input_ids",
            )
            _validate_ids(ids, fast_vocabulary_size, "fast invalid input")
            decode = {
                "skip_special_tokens_false": _fast_decode_observation(
                    tokenizer,
                    ids,
                    skip_special_tokens=False,
                    vocabulary_size=fast_vocabulary_size,
                    model_dir=model_dir,
                ),
                "skip_special_tokens_true": _fast_decode_observation(
                    tokenizer,
                    ids,
                    skip_special_tokens=True,
                    vocabulary_size=fast_vocabulary_size,
                    model_dir=model_dir,
                ),
            }
            return {
                "status": "accepted",
                "category": "bytes-accepted",
                "ids": ids,
                "decode": decode,
                "decoded_bytes": {
                    policy: observation.get("bytes")
                    for policy, observation in decode.items()
                },
            }
        except Exception as error:
            return {
                "status": "rejected",
                "category": "invalid-utf8",
                "exception_type": type(error).__name__,
                "message": _safe_message(error, model_dir),
                "ids": None,
                "decoded_bytes": None,
            }

    def slow_observation() -> dict[str, Any]:
        try:
            processor.SetEncodeExtraOptions("")
            ids = _as_int_list(
                processor.encode(raw, out_type=int),
                "slow invalid input input_ids",
            )
            _validate_ids(ids, slow_vocabulary_size, "slow invalid input")
            decode = _slow_decode_observation(
                processor,
                ids,
                vocabulary_size=slow_vocabulary_size,
                model_dir=model_dir,
            )
            return {
                "status": "accepted",
                "category": "bytes-accepted",
                "ids": ids,
                "decode": decode,
                "decoded_bytes": decode.get("bytes"),
            }
        except Exception as error:
            return {
                "status": "rejected",
                "category": "invalid-utf8",
                "exception_type": type(error).__name__,
                "message": _safe_message(error, model_dir),
                "ids": None,
                "decoded_bytes": None,
            }

    fast = fast_observation()
    slow = slow_observation()
    pair = (fast["status"], slow["status"])
    if pair == ("rejected", "accepted"):
        reason_code = "raw-bytes-fast-rejects-slow-accepts"
    elif pair == ("accepted", "rejected"):
        reason_code = "raw-bytes-fast-accepts-slow-rejects"
    elif pair == ("rejected", "rejected"):
        reason_code = "raw-bytes-both-reject"
    else:
        reason_code = "raw-bytes-both-accept"

    return {
        "name": name,
        "kind": "text",
        "input_bytes": list(raw),
        "options": {
            "add_special_tokens": False,
            "input_encoding": "utf-8",
            "invalid_input_policy": (
                "pass the original bytes to each API; comparator never "
                "replacement-decodes them"
            ),
            "raw_bytes_passed_to_apis": True,
        },
        "fast": fast,
        "slow": slow,
        "status": "not-comparable",
        "comparison": {
            "status": "not-comparable",
            "reason_code": reason_code,
            "reason": (
                "invalid UTF-8 bytes are outside the comparable text contract; "
                "each API outcome is recorded independently and no acceptance "
                "or rejection is certified as equivalent"
            ),
            "fast_status": fast["status"],
            "slow_status": slow["status"],
        },
    }


def _pair_case(
    tokenizer: Any,
    processor: Any,
    *,
    name: str,
    first: str,
    second: str,
    fast_vocabulary_size: int,
    slow_vocabulary_size: int,
    model_dir: Path,
) -> dict[str, Any]:
    try:
        encoded = tokenizer(
            first,
            text_pair=second,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(
            f"fast pair encode failed for {name!r}: {_safe_message(error, model_dir)}"
        ) from error
    if not isinstance(encoded, Mapping):
        raise CrosscheckError(
            f"fast pair encode for {name!r} returned {type(encoded).__name__}"
        )
    fast_ids = _as_int_list(encoded.get("input_ids"), f"fast pair {name} input_ids")
    fast_type_ids = _as_int_list(
        encoded.get("token_type_ids", [0] * len(fast_ids)),
        f"fast pair {name} token_type_ids",
    )
    if len(fast_type_ids) != len(fast_ids):
        raise CrosscheckError(
            f"fast pair {name} token_type_ids length {len(fast_type_ids)} does not "
            f"match input_ids length {len(fast_ids)}"
        )
    _validate_ids(fast_ids, fast_vocabulary_size, f"fast pair {name}")
    fast_decode = {
        "skip_special_tokens_false": _fast_decode_observation(
            tokenizer,
            fast_ids,
            skip_special_tokens=False,
            vocabulary_size=fast_vocabulary_size,
            model_dir=model_dir,
        ),
        "skip_special_tokens_true": _fast_decode_observation(
            tokenizer,
            fast_ids,
            skip_special_tokens=True,
            vocabulary_size=fast_vocabulary_size,
            model_dir=model_dir,
        ),
    }

    # There is no SentencePiece pair operation.  Record each independently
    # available sequence measurement, but never invent a pair/type-ID result.
    slow_first = _slow_encode(
        processor, first, add_bos=True, model_dir=model_dir
    )
    slow_second = _slow_encode(
        processor, second, add_bos=True, model_dir=model_dir
    )
    _validate_ids(slow_first, slow_vocabulary_size, f"slow pair {name} first")
    _validate_ids(slow_second, slow_vocabulary_size, f"slow pair {name} second")
    slow_first_decode = _slow_decode_observation(
        processor,
        slow_first,
        vocabulary_size=slow_vocabulary_size,
        model_dir=model_dir,
    )
    slow_second_decode = _slow_decode_observation(
        processor,
        slow_second,
        vocabulary_size=slow_vocabulary_size,
        model_dir=model_dir,
    )

    return {
        "name": name,
        "kind": "pair",
        "input_text": [first, second],
        "input_bytes": [_bytes(first, f"pair {name} first"), _bytes(second, f"pair {name} second")],
        "options": {
            "add_special_tokens": True,
            "decode_policy": {
                "fast": [False, True],
                "slow": "SentencePieceProcessor.DecodeIds default per sequence",
            },
            "input_encoding": "utf-8",
        },
        "fast": {
            "ids": fast_ids,
            "type_ids": fast_type_ids,
            "decode": fast_decode,
        },
        "slow": {
            "pair_encoding": {
                "status": "not-exposed",
                "category": "pair-api-unavailable",
                "message": (
                    "SentencePieceProcessor exposes single-sequence encode only"
                ),
                "ids": None,
                "type_ids": None,
                "decoded_bytes": None,
            },
            "first": {
                "encode_options": {
                    "extra_options": "bos",
                    "adds_bos": True,
                },
                "ids": slow_first,
                "decode": slow_first_decode,
            },
            "second": {
                "encode_options": {
                    "extra_options": "bos",
                    "adds_bos": True,
                },
                "ids": slow_second,
                "decode": slow_second_decode,
            },
        },
        "status": "not-comparable",
        "comparison": {
            "status": "not-comparable",
            "reason_code": "sentencepiece-no-pair-api",
            "reason": (
                "fast pair output is recorded exactly; SentencePiece has no "
                "comparable pair operation"
            ),
            "fast_ids": fast_ids,
            "fast_type_ids": fast_type_ids,
            "slow_ids": None,
            "slow_type_ids": None,
            "fast_decoded_bytes": {
                "skip_special_tokens_false": _successful_decode(
                    fast_decode["skip_special_tokens_false"], f"fast pair {name}"
                ),
                "skip_special_tokens_true": _successful_decode(
                    fast_decode["skip_special_tokens_true"], f"fast pair {name}"
                ),
            },
            "slow_decoded_bytes": {
                "first": _successful_decode(
                    slow_first_decode, f"slow pair {name} first"
                ),
                "second": _successful_decode(
                    slow_second_decode, f"slow pair {name} second"
                ),
            },
            "type_ids": {
                "status": "not-exposed",
                "fast": fast_type_ids,
                "slow": None,
            },
        },
    }


def _load_generation_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "generation_config.json"
    try:
        with path.open("rb") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrosscheckError(
            f"cannot load generation metadata from required artifact {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise CrosscheckError(
            f"generation metadata {path} must be a JSON object; "
            f"actual {type(value).__name__}"
        )
    return dict(value)


def _destination_is_safe(model_dir: Path, output: Path) -> None:
    try:
        output_resolved = output.resolve(strict=False)
        for name in provenance.REQUIRED_ARTIFACTS:
            artifact = (model_dir / name).resolve(strict=False)
            if output_resolved == artifact:
                raise CrosscheckError(
                    f"report output must not replace required artifact {name!r}"
                )
    except OSError as error:
        raise CrosscheckError(
            f"cannot resolve report output {output}: {error}"
        ) from error


def _publish(report: Mapping[str, Any], output: Path) -> None:
    try:
        payload = (
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise CrosscheckError(f"cannot serialize equivalence report: {error}") from error

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    except OSError as error:
        raise CrosscheckError(
            f"cannot atomically publish equivalence report {output}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _build_report(model_dir: Path, output: Path) -> dict[str, Any]:
    _destination_is_safe(model_dir, output)

    # This call performs the task-01 CPython/package checks and all five
    # artifact reads/hashes before either tokenizer import or construction.
    expected_manifest_path = Path(__file__).resolve().with_name(
        "tokenizer_reference_manifest.json"
    )
    manifest = provenance.build_manifest(
        model_dir,
        expected_manifest=expected_manifest_path,
    )

    # Imports are intentionally delayed until provenance has succeeded.
    try:
        from transformers import PreTrainedTokenizerFast
        import sentencepiece as spm
    except Exception as error:  # pragma: no cover - provenance normally catches pins
        raise CrosscheckError(
            f"cannot import pinned tokenizer APIs after provenance validation: {error}"
        ) from error

    tokenizer_path = model_dir / "tokenizer.json"
    model_path = model_dir / "tokenizer.model"
    try:
        fast = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="</s>",
            clean_up_tokenization_spaces=False,
        )
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(
            f"cannot load fast tokenizer from caller artifact tokenizer.json: "
            f"{_safe_message(error, model_dir)}"
        ) from error
    try:
        slow = spm.SentencePieceProcessor(model_file=str(model_path))
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(
            f"cannot load slow SentencePiece tokenizer from caller artifact "
            f"tokenizer.model: {_safe_message(error, model_dir)}"
        ) from error

    try:
        fast_vocabulary_size = len(fast.get_vocab())
        slow_vocabulary_size = int(slow.vocab_size())
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(f"cannot inspect tokenizer vocabulary sizes: {error}") from error
    if fast_vocabulary_size <= 0 or slow_vocabulary_size <= 0:
        raise CrosscheckError(
            f"invalid tokenizer vocabulary sizes: fast {fast_vocabulary_size}, "
            f"slow {slow_vocabulary_size}"
        )

    generation_config = _load_generation_config(model_dir)
    try:
        generation_pad_id = int(generation_config["pad_token_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise CrosscheckError(
            "generation_config.json does not expose an integer pad_token_id"
        ) from error

    try:
        tokenizer_pad_id = int(fast.pad_token_id)
        tokenizer_pad_token = str(fast.pad_token)
        tokenizer_bos_id = int(fast.bos_token_id)
        tokenizer_eos_id = int(fast.eos_token_id)
        sentencepiece_pad_id = int(slow.pad_id())
        sentencepiece_pad_piece_id = int(slow.piece_to_id("<pad>"))
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise CrosscheckError(f"cannot inspect tokenizer special-token metadata: {error}") from error

    text_records = [
        _text_case(
            fast,
            slow,
            name=name,
            text=text,
            add_special_tokens=add_special_tokens,
            fast_vocabulary_size=fast_vocabulary_size,
            slow_vocabulary_size=slow_vocabulary_size,
            model_dir=model_dir,
        )
        for name, text, add_special_tokens in TEXT_CASES
    ]
    text_records.append(
        _invalid_utf8_case(
            fast,
            slow,
            fast_vocabulary_size=fast_vocabulary_size,
            slow_vocabulary_size=slow_vocabulary_size,
            model_dir=model_dir,
        )
    )
    pair_records = [
        _pair_case(
            fast,
            slow,
            name=name,
            first=first,
            second=second,
            fast_vocabulary_size=fast_vocabulary_size,
            slow_vocabulary_size=slow_vocabulary_size,
            model_dir=model_dir,
        )
        for name, first, second in PAIR_CASES
    ]

    # Keep the fixed list in source order, while ensuring accidental edits do
    # not introduce nondeterministic ordering into the report.
    limitations = sorted(
        (dict(item) for item in KNOWN_LIMITATIONS),
        key=lambda item: str(item["code"]),
    )
    disagreements: list[str] = []
    for record in [*text_records, *pair_records]:
        if record["status"] == "disagree":
            disagreements.append(str(record["name"]))
    if disagreements:
        raise CrosscheckError(
            "required tokenizer-model disagreements: " + ", ".join(disagreements)
        )

    text_order = [name for name, _, _ in TEXT_CASES] + [INVALID_CASE[0]]
    pair_order = [name for name, _, _ in PAIR_CASES]
    return {
        "corpus": {
            "pair_cases": pair_records,
            "pair_order": pair_order,
            "text_cases": text_records,
            "text_order": text_order,
            "version": CORPUS_VERSION,
        },
        "generation": {
            "command": [
                "python3",
                "test/reference/crosscheck_tokenizer_model.py",
                "--model-dir",
                "<MODEL_DIR>",
                "--output",
                "<OUTPUT>",
            ],
            "generator_version": GENERATOR_VERSION,
        },
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
        },
        "interpretation": {
            "fast_vocabulary_size": fast_vocabulary_size,
            "generation_pad_id": generation_pad_id,
            "sentencepiece_pad_id": sentencepiece_pad_id,
            "sentencepiece_pad_piece_lookup_id": sentencepiece_pad_piece_id,
            "sentencepiece_pad_piece_lookup_is_declared_padding": False,
            "tokenizer_bos_id": tokenizer_bos_id,
            "tokenizer_eos_id": tokenizer_eos_id,
            "tokenizer_pad_id": tokenizer_pad_id,
            "tokenizer_pad_token": tokenizer_pad_token,
            "notes": [
                "tokenizer pad ID 2 is EOS (</s>) for this distribution",
                "generation pad ID 0 is independent generation metadata",
                "SentencePiece pad_id -1 and <pad> lookup are protobuf metadata, not tokenizer padding",
            ],
            "slow_vocabulary_size": slow_vocabulary_size,
        },
        "known_limitations": limitations,
        "provenance": manifest,
        "report_version": REPORT_VERSION,
        "result": {
            "disagreements": [],
            "status": "pass",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a caller-selected tokenizer.json and tokenizer.model "
            "without network or cache discovery."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        metavar="PATH",
        help="caller-supplied local distribution directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="equivalence report destination",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    try:
        report = _build_report(model_dir, output)
        _publish(report, output)
    except (provenance.ProvenanceError, CrosscheckError) as error:
        print(f"tokenizer model cross-check: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - final contextual boundary
        print(f"tokenizer model cross-check: unexpected failure: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

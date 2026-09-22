#!/usr/bin/env python3
"""Exercise the committed corpus verifier against policy-preserving mutations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "test" / "reference" / "generate_model_oracles.py"
CORPUS = ROOT / "test" / "model" / "synthetic_reference.json"


def refresh_payload_digest(document: dict) -> None:
    payload = {
        key: document[key]
        for key in ("schema_version", "kind", "tolerances", "models", "cases")
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    document["provenance"]["payload_sha256"] = hashlib.sha256(encoded).hexdigest()


def mutate_margin_stability(document: dict) -> None:
    snapshot = document["cases"][0]["snapshots"][0]
    snapshot["margin_stable"][0] = not snapshot["margin_stable"][0]
    snapshot["margins"][0]["stable"] = not snapshot["margins"][0]["stable"]


def run_case(name: str, mutate: Callable[[dict], None], baseline: dict) -> None:
    document = copy.deepcopy(baseline)
    mutate(document)
    refresh_payload_digest(document)
    with tempfile.TemporaryDirectory(prefix="iom-reference-negative-") as directory:
        candidate = Path(directory) / f"{name}.json"
        candidate.write_text(
            json.dumps(document, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--verify", str(candidate)],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 2:
        raise SystemExit(
            f"{name}: verifier returned {result.returncode}, expected 2\n"
            f"{result.stdout}{result.stderr}"
        )
    detail = (result.stderr or result.stdout).strip().splitlines()[-1]
    print(f"{name}: rejected (exit 2): {detail}")


def main() -> int:
    baseline = json.loads(CORPUS.read_text(encoding="utf-8"))
    mutations: list[tuple[str, Callable[[dict], None]]] = [
        ("input-matrix", lambda d: d["cases"][0]["input"].update(prefix_ids=[2])),
        ("model-binding", lambda d: d["cases"][0].update(model_id=d["models"][1]["id"])),
        ("snapshot-phase", lambda d: d["cases"][0]["snapshots"].reverse()),
        (
            "winner",
            lambda d: d["cases"][0]["snapshots"][0]["greedy_id"].__setitem__(
                0,
                (d["cases"][0]["snapshots"][0]["greedy_id"][0] + 1) % 19,
            ),
        ),
        (
            "margin-values",
            lambda d: d["cases"][0]["snapshots"][0]["margins"][0].update(
                winner_error=0.0
            ),
        ),
        ("margin-stable", mutate_margin_stability),
        (
            "cached-consistency",
            lambda d: d["cases"][0]["snapshots"][1]["last_layer_output"][
                "values"
            ].__setitem__(
                0,
                d["cases"][0]["snapshots"][1]["last_layer_output"]["values"][0]
                + 10.0,
            ),
        ),
        (
            "teacher-token-ids",
            lambda d: d["cases"][4]["expected_result"].update(
                token_ids=[0, 0, 0]
            ),
        ),
        (
            "production-greedy",
            lambda d: d["cases"][4]["expected_result"][
                "production_greedy_ids"
            ].__setitem__(
                0,
                (
                    d["cases"][4]["expected_result"]["production_greedy_ids"][0]
                    + 1
                )
                % 19,
            ),
        ),
        (
            "certified-positions",
            lambda d: d["cases"][0]["expected_result"].update(
                reference_margin_certified_positions=[999]
            ),
        ),
        ("duplicate-case", lambda d: d["cases"][1].update(id=d["cases"][0]["id"])),
    ]
    for name, mutate in mutations:
        run_case(name, mutate, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

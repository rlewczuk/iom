#!/usr/bin/env python3
"""Validate remediation evidence and task_ctl lifecycle metadata."""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path
from typing import Any, Callable


TASK_DIR_RE = re.compile(r"^(\d+)-((?:CC|ST|AR|NT|PF)-\d{3})-[a-z0-9]+(?:-[a-z0-9]+)*$")
EVIDENCE_FIELDS = [
    "Finding",
    "Review area",
    "Review severity",
    "Review verification",
    "Review scope",
    "Backend scope",
    "Location",
    "Review source",
]
FORBIDDEN_METADATA_FIELDS = {
    "Order",
    "Type",
    "Priority",
    "Blocked by",
    "Source",
}
REQUIRED_SECTIONS = [
    "## Outcome",
    "## Current problem",
    "## Scope",
    "## Implementation references",
    "## Requirements",
    "## Non-goals",
    "## Acceptance criteria",
    "## Verification",
]


def parse_fields(text: str) -> dict[str, str]:
    field_re = re.compile(r"^\*\*(?P<name>[^*]+):\*\*\s*(?P<value>.*)$", re.M)
    return {match.group("name"): match.group("value").strip() for match in field_re.finditer(text)}


def task_api(repo: Path) -> tuple[Callable[..., dict[str, Any]], Callable[..., list[dict[str, Any]]]]:
    try:
        helper_root = Path(__file__).resolve().parents[3]
        namespace = runpy.run_path(
            str(helper_root / ".omp" / "csw" / "bin" / "task_ctl"),
            run_name="task_ctl_validator",
        )
        return namespace["get_task"], namespace["list_tasks"]
    except (OSError, KeyError, RuntimeError) as exc:
        raise RuntimeError(f"unable to load task_ctl API: {exc}") from exc


def repo_root_for_spec(spec_dir: Path) -> Path:
    for parent in (spec_dir, *spec_dir.parents):
        if parent.name == "changes" and parent.parent.name == "docs":
            return parent.parent.parent
    raise RuntimeError("spec-dir must be beneath docs/changes/")


def canonical_task_id(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def validate_task(
    path: Path,
    spec_dir: Path,
    repo: Path,
    get_task: Callable[..., dict[str, Any]],
) -> tuple[list[str], int | None, str | None, dict[str, Any] | None]:
    errors: list[str] = []
    resolved = path.resolve()
    if not resolved.is_file():
        return [f"file not found: {path}"], None, None, None
    if resolved.name != "spec.md" or resolved.parent.parent != spec_dir:
        errors.append("task-file must be <spec-dir>/<task-directory>/spec.md")

    directory = resolved.parent.name
    directory_match = TASK_DIR_RE.fullmatch(directory)
    if not directory_match:
        errors.append(f"invalid task directory name: {directory}")
        directory_order = None
        directory_id = None
    else:
        directory_prefix = directory_match.group(1)
        if len(directory_prefix) < 2:
            errors.append("task order prefix must be at least two digits")
        directory_order = int(directory_prefix)
        directory_id = directory_match.group(2)

    text = resolved.read_text(encoding="utf-8")
    if not text.startswith("# "):
        errors.append("missing action-oriented level-one title")
    fields = parse_fields(text)
    for field in EVIDENCE_FIELDS:
        if not fields.get(field):
            errors.append(f"missing field: {field}")
    for field in FORBIDDEN_METADATA_FIELDS:
        if field in fields:
            errors.append(f"lifecycle metadata must be stored in task.yml, not spec.md: {field}")

    last = -1
    for section in REQUIRED_SECTIONS:
        position = text.find(section)
        if position < 0:
            errors.append(f"missing section: {section}")
        elif position < last:
            errors.append(f"section out of order: {section}")
        else:
            last = position
    acceptance_start = text.find("## Acceptance criteria")
    verification_start = text.find("## Verification")
    if acceptance_start >= 0:
        acceptance_end = verification_start if verification_start > acceptance_start else len(text)
        if not re.search(r"^- \[ \] \S", text[acceptance_start:acceptance_end], re.M):
            errors.append("Acceptance criteria must contain an observable checklist item")
    if verification_start >= 0 and not re.search(r"^- `[^`\n]+`", text[verification_start:], re.M):
        errors.append("Verification must contain a focused command")

    finding = fields.get("Finding", "").strip("`")
    if not re.fullmatch(r"(?:CC|ST|AR|NT|PF)-\d{3}", finding):
        errors.append(f"invalid Finding value: {fields.get('Finding', '')}")
    elif directory_id is not None and finding != directory_id:
        errors.append(f"Finding {finding} does not match directory ID {directory_id}")
    if fields.get("Review severity") not in {"critical", "high", "medium", "low"}:
        errors.append("Review severity must be critical, high, medium, or low")
    verification = fields.get("Review verification", "")
    confidence = re.fullmatch(r"(verified|strongly-supported|hypothesis), confidence (\d{1,3})", verification)
    if not confidence:
        errors.append("Review verification must include state and confidence 0-100")
    elif not 0 <= int(confidence.group(2)) <= 100:
        errors.append("review confidence must be 0-100")
    if fields.get("Review scope") not in {
        "introduced by selected commit",
        "materially exposed by selected commit",
        "whole-codebase",
    }:
        errors.append("invalid Review scope value")

    metadata: dict[str, Any] | None = None
    if resolved.parent.parent == spec_dir:
        try:
            metadata = get_task(repo, canonical_task_id(repo, resolved.parent))
        except Exception as exc:  # task_ctl owns the precise fail-closed diagnostic
            errors.append(f"task_ctl metadata error: {exc}")
    if metadata is not None:
        if metadata.get("type") != "impl":
            errors.append(f"task type must be impl, got {metadata.get('type', '')!r}")
        if metadata.get("status") != "new":
            errors.append(f"generated task status must be new, got {metadata.get('status', '')!r}")
        order = metadata.get("order")
        if not isinstance(order, int) or isinstance(order, bool):
            errors.append("task_ctl order must be numeric for a written task")
        elif directory_order is not None and order != directory_order:
            errors.append(f"task_ctl order {order} does not match directory prefix {directory_order}")
        priority = metadata.get("priority")
        if not isinstance(priority, str) or not re.fullmatch(r"P[012]", priority):
            errors.append("task_ctl priority must be P0, P1, or P2")
        source = metadata.get("source")
        expected_source = (spec_dir.relative_to(repo).as_posix() + "/spec.md")
        if source != expected_source:
            errors.append(f"task source must be parent spec.md: {expected_source}")
        blocked = metadata.get("blocked-by", [])
        if blocked is None:
            blocked = []
        if not isinstance(blocked, list):
            errors.append("task_ctl blocked-by must be a list of dependency records")
            blocker_ids: list[str] = []
        else:
            blocker_ids = []
            for record in blocked:
                if not isinstance(record, dict) or not isinstance(record.get("task-id"), str):
                    errors.append("task_ctl blocked-by entries must contain canonical task-id records")
                    continue
                blocker_ids.append(record["task-id"])
            metadata["_blocker-ids"] = blocker_ids
    return errors, directory_order, directory_id, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--task-file", action="append", required=True)
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir).resolve()
    errors: list[str] = []
    try:
        repo = repo_root_for_spec(spec_dir)
    except RuntimeError as exc:
        errors.append(str(exc))
        repo = spec_dir
    if not spec_dir.is_dir():
        errors.append(f"spec-dir not found: {spec_dir}")
    try:
        get_task, list_tasks = task_api(repo)
    except RuntimeError as exc:
        errors.append(str(exc))
        get_task = list_tasks = None  # type: ignore[assignment]

    records: list[tuple[Path, int | None, str | None, dict[str, Any] | None]] = []
    seen_paths: set[Path] = set()
    for supplied in args.task_file:
        path = Path(supplied).resolve()
        if path in seen_paths:
            errors.append(f"duplicate --task-file: {supplied}")
            continue
        seen_paths.add(path)
        if get_task is None:
            task_errors, order, finding, metadata = (["task_ctl API unavailable"], None, None, None)
        else:
            task_errors, order, finding, metadata = validate_task(path, spec_dir, repo, get_task)
        errors.extend(f"{path}: {error}" for error in task_errors)
        records.append((path, order, finding, metadata))

    all_task_records: list[dict[str, Any]] = []
    known_task_ids: set[str] = set()
    if list_tasks is not None and spec_dir.is_dir():
        try:
            parent_id = spec_dir.relative_to(repo).as_posix()
            all_task_records = list_tasks(repo, parent_id, impl=False)
            known_task_ids = {
                record["task-id"]
                for record in all_task_records
                if isinstance(record.get("task-id"), str)
            }
        except Exception as exc:
            errors.append(f"task_ctl list error: {exc}")

    numeric_orders = sorted(
        metadata["order"]
        for _, _, _, metadata in records
        if metadata and isinstance(metadata.get("order"), int) and not isinstance(metadata.get("order"), bool)
    )
    if len(numeric_orders) != len(set(numeric_orders)):
        errors.append("generated task orders must be unique")
    if numeric_orders and numeric_orders != list(range(numeric_orders[0], numeric_orders[0] + len(numeric_orders))):
        errors.append("generated task orders must be consecutive")

    findings = [finding for _, _, finding, _ in records if finding is not None]
    if len(findings) != len(set(findings)):
        errors.append("generated task finding IDs must be unique")

    existing_dirs = {child.name for child in spec_dir.iterdir() if child.is_dir()} if spec_dir.is_dir() else set()
    generated_dirs = {path.parent.name for path, _, _, _ in records}
    prior_prefixes = []
    for directory in existing_dirs - generated_dirs:
        match = re.match(r"^(\d+)-", directory)
        if match:
            prior_prefixes.append(match.group(1))
    generated_ids = {
        canonical_task_id(repo, path.parent)
        for path, _, _, _ in records
        if path.parent.parent == spec_dir
    }
    prior_metadata_orders = [
        record["order"]
        for record in all_task_records
        if record.get("task-id") not in generated_ids
        and isinstance(record.get("order"), int)
        and not isinstance(record.get("order"), bool)
    ]
    if numeric_orders:
        prior_maximum = max(
            max((int(prefix) for prefix in prior_prefixes), default=0),
            max(prior_metadata_orders, default=0),
        )
        if numeric_orders[0] != prior_maximum + 1:
            errors.append(f"generated task orders must start at {prior_maximum + 1}")
        required_width = max(
            2,
            len(str(numeric_orders[-1])),
            *(len(prefix) for prefix in prior_prefixes),
            *(len(str(order)) for order in prior_metadata_orders),
        )
        for path, _, _, metadata in records:
            if metadata and len(path.parent.name.split("-", 1)[0]) < required_width:
                errors.append(f"{path}: task order prefix must use width {required_width}")

    for path, order, _, metadata in records:
        if not metadata:
            continue
        for blocker in metadata.get("_blocker-ids", []):
            if blocker not in known_task_ids:
                errors.append(f"{path}: blocker does not name a canonical task: {blocker}")
                continue
            blocker_order = None
            try:
                blocker_record = get_task(repo, blocker) if get_task is not None else {}
                blocker_order = blocker_record.get("order")
            except Exception as exc:
                errors.append(f"{path}: unable to read blocker {blocker}: {exc}")
            if isinstance(order, int) and isinstance(blocker_order, int) and blocker_order >= order:
                errors.append(f"{path}: blocker must point to an earlier task: {blocker}")

    if errors:
        for error in errors:
            print(f"review-task-error: {error}", file=sys.stderr)
        return 2

    print(f"review-tasks-ok: {len(records)} task(s) beneath {spec_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

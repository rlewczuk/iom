#!/usr/bin/env python3
"""Validate remediation task specs emitted directly by cpp-inference review skills."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TASK_DIR_RE = re.compile(r"^(\d+)-((?:CC|ST|AR|NT|PF)-\d{3})-[a-z0-9]+(?:-[a-z0-9]+)*$")
FIELD_RE = re.compile(r"^\*\*(?P<name>[^*]+):\*\*\s*(?P<value>.*)$", re.M)
REQUIRED_FIELDS = [
    "Order",
    "Priority",
    "Blocked by",
    "Review source",
    "Finding",
    "Review area",
    "Review severity",
    "Review verification",
    "Review scope",
    "Backend scope",
    "Location",
]
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
    return {match.group("name"): match.group("value").strip() for match in FIELD_RE.finditer(text)}


def validate_task(path: Path, spec_dir: Path) -> tuple[list[str], int | None, str | None, list[str]]:
    errors: list[str] = []
    resolved = path.resolve()
    if not resolved.is_file():
        return [f"file not found: {path}"], None, None, []
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
    for field in REQUIRED_FIELDS:
        if not fields.get(field):
            errors.append(f"missing field: {field}")

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

    order_value = fields.get("Order", "")
    if not order_value.isdigit():
        errors.append("Order must be a numeric value for a written task")
    elif directory_order is not None and int(order_value) != directory_order:
        errors.append(f"Order {order_value} does not match directory prefix {directory_order}")

    finding = fields.get("Finding", "").strip("`")
    if not re.fullmatch(r"(?:CC|ST|AR|NT|PF)-\d{3}", finding):
        errors.append(f"invalid Finding value: {fields.get('Finding', '')}")
    elif directory_id is not None and finding != directory_id:
        errors.append(f"Finding {finding} does not match directory ID {directory_id}")

    if not re.match(r"P[012]\b", fields.get("Priority", "")):
        errors.append("Priority must start with P0, P1, or P2")
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

    blockers_value = fields.get("Blocked by", "")
    blockers = [] if blockers_value == "None" else [value.strip(" `") for value in blockers_value.split(",") if value.strip()]
    return errors, directory_order, directory_id, blockers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--task-file", action="append", required=True)
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir).resolve()
    errors: list[str] = []
    parts = spec_dir.parts
    try:
        docs_index = parts.index("docs")
    except ValueError:
        docs_index = -1
    if docs_index < 0 or docs_index + 1 >= len(parts) or parts[docs_index + 1] != "changes":
        errors.append("spec-dir must be beneath docs/changes/")
    if not spec_dir.is_dir():
        errors.append(f"spec-dir not found: {spec_dir}")

    records = []
    seen_paths: set[Path] = set()
    for supplied in args.task_file:
        path = Path(supplied).resolve()
        if path in seen_paths:
            errors.append(f"duplicate --task-file: {supplied}")
            continue
        seen_paths.add(path)
        task_errors, order, finding, blockers = validate_task(path, spec_dir)
        errors.extend(f"{path}: {error}" for error in task_errors)
        records.append((path, order, finding, blockers))

    numeric_orders = sorted(order for _, order, _, _ in records if order is not None)
    if len(numeric_orders) != len(set(numeric_orders)):
        errors.append("generated task orders must be unique")
    if numeric_orders and numeric_orders != list(range(numeric_orders[0], numeric_orders[0] + len(numeric_orders))):
        errors.append("generated task orders must be consecutive")

    findings = [finding for _, _, finding, _ in records if finding is not None]
    if len(findings) != len(set(findings)):
        errors.append("generated task finding IDs must be unique")

    known_dirs = {child.name for child in spec_dir.iterdir() if child.is_dir()} if spec_dir.is_dir() else set()
    generated_dirs = {path.parent.name for path, _, _, _ in records}
    prior_prefixes = []
    for directory in known_dirs - generated_dirs:
        match = re.match(r"^(\d+)-", directory)
        if match:
            prior_prefixes.append(match.group(1))
    if numeric_orders:
        prior_maximum = max((int(prefix) for prefix in prior_prefixes), default=0)
        if numeric_orders[0] != prior_maximum + 1:
            errors.append(f"generated task orders must start at {prior_maximum + 1}")
        required_width = max(2, len(str(numeric_orders[-1])), *(len(prefix) for prefix in prior_prefixes))
        for path, _, _, _ in records:
            if len(path.parent.name.split("-", 1)[0]) < required_width:
                errors.append(f"{path}: task order prefix must use width {required_width}")

    order_by_dir = {path.parent.name: order for path, order, _, _ in records if order is not None}
    for path, order, _, blockers in records:
        for blocker in blockers:
            if blocker not in known_dirs:
                errors.append(f"{path}: blocker does not name a task directory: {blocker}")
                continue
            blocker_match = re.match(r"^(\d+)-", blocker)
            if order is not None and blocker_match and int(blocker_match.group(1)) >= order:
                errors.append(f"{path}: blocker must point to an earlier task: {blocker}")
            if blocker in order_by_dir and order is not None and order_by_dir[blocker] >= order:
                errors.append(f"{path}: generated blocker edge points forward: {blocker}")

    if errors:
        for error in errors:
            print(f"review-task-error: {error}", file=sys.stderr)
        return 2

    print(f"review-tasks-ok: {len(records)} task(s) beneath {spec_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

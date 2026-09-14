#!/usr/bin/env python3
"""Resolve a docs/changes task destination and report safe numbering metadata."""

from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any


TASK_DIR_RE = re.compile(r"^(\d+)-")


def git_root(repo: Path) -> Path:
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "not a git repository")
    return Path(process.stdout.strip()).resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_task_ctl(repo: Path) -> Any:
    try:
        helper_root = Path(__file__).resolve().parents[3]
        namespace = runpy.run_path(
            str(helper_root / ".omp" / "csw" / "bin" / "task_ctl"),
            run_name="resolve_spec_path",
        )
        return namespace["list_tasks"]
    except (OSError, KeyError, RuntimeError) as exc:
        raise RuntimeError(f"unable to load task_ctl API: {exc}") from exc


def occupied_task_dirs(spec_dir: Path) -> list[tuple[int, str]]:
    occupied: list[tuple[int, str]] = []
    for child in spec_dir.iterdir():
        if child.is_dir():
            match = TASK_DIR_RE.match(child.name)
            if match:
                occupied.append((int(match.group(1)), child.name))
    return sorted(occupied, key=lambda item: (item[0], item[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--spec", required=True, help="Repository-relative docs/changes/... directory")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    try:
        root = git_root(Path(args.repo))
        supplied = Path(args.spec)
        candidate = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()

        changes_root = (root / "docs" / "changes").resolve()
        if candidate == changes_root or not is_relative_to(candidate, changes_root):
            raise RuntimeError("spec path must identify a child directory beneath docs/changes/")
        if not candidate.is_dir():
            raise RuntimeError(f"spec directory does not exist: {candidate}")

        relative = candidate.relative_to(root)
        list_tasks = load_task_ctl(root)
        task_records = list_tasks(root, relative.as_posix(), impl=False)
        occupied = occupied_task_dirs(candidate)
        metadata_orders: dict[str, int] = {}
        for record in task_records:
            task_id = record.get("task-id")
            order = record.get("order")
            if (
                not isinstance(task_id, str)
                or Path(task_id).parent.as_posix() != relative.as_posix()
                or not isinstance(order, int)
                or isinstance(order, bool)
            ):
                continue
            metadata_orders[Path(task_id).name] = order

        # The filesystem remains part of collision safety: a numbered directory
        # without task.yml/spec.md still reserves its numeric prefix.
        existing = sorted(
            {(metadata_orders.get(name, order), name) for order, name in occupied},
            key=lambda item: (item[0], item[1]),
        )
        metadata_maximum = max((order for order, _ in existing), default=0)
        occupied_maximum = max((order for order, _ in occupied), default=0)
        maximum = max(metadata_maximum, occupied_maximum)
        next_order = maximum + 1
        width = max(
            2,
            len(str(next_order)),
            *(len(name.split("-", 1)[0]) for _, name in occupied),
            *(len(name.split("-", 1)[0]) for _, name in existing),
        )
        result: dict[str, Any] = {
            "repo_root": str(root),
            "spec_dir": relative.as_posix(),
            "task_root": relative.as_posix(),
            "existing_task_dirs": [name for _, name in existing],
            "existing_tasks": task_records,
            "next_task_order": next_order,
            "order_width": width,
        }
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        print(f"spec-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

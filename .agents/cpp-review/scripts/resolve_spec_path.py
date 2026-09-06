#!/usr/bin/env python3
"""Resolve a docs/changes task destination and report safe numbering metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


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


def task_orders(spec_dir: Path) -> list[tuple[int, str]]:
    orders = []
    for child in spec_dir.iterdir():
        if not child.is_dir():
            continue
        match = TASK_DIR_RE.match(child.name)
        if match:
            orders.append((int(match.group(1)), child.name))
    return sorted(orders)


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
        existing = task_orders(candidate)
        maximum = max((order for order, _ in existing), default=0)
        next_order = maximum + 1
        width = max(2, len(str(next_order)), *(len(name.split("-", 1)[0]) for _, name in existing))
        result = {
            "repo_root": str(root),
            "spec_dir": relative.as_posix(),
            "task_root": relative.as_posix(),
            "existing_task_dirs": [name for _, name in existing],
            "next_task_order": next_order,
            "order_width": width,
        }
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except RuntimeError as exc:
        print(f"spec-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resolve and validate a docs/changes/<change>[/<subchange>] specification path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def git_root(repo: Path) -> Path:
    p = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "not a git repository")
    return Path(p.stdout.strip()).resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--spec", required=True, help="Repository-relative docs/changes/... directory")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    try:
        root = git_root(Path(args.repo))
        supplied = Path(args.spec)
        if supplied.is_absolute():
            candidate = supplied.resolve()
        else:
            candidate = (root / supplied).resolve()

        changes_root = (root / "docs" / "changes").resolve()
        if candidate == changes_root or not is_relative_to(candidate, changes_root):
            raise RuntimeError("spec path must identify a child directory beneath docs/changes/")
        if not candidate.is_dir():
            raise RuntimeError(f"spec directory does not exist: {candidate}")

        relative = candidate.relative_to(root)
        result = {
            "repo_root": str(root),
            "spec_dir": relative.as_posix(),
            "review_file": (relative / "review.md").as_posix(),
        }
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except RuntimeError as exc:
        print(f"spec-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

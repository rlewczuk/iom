#!/usr/bin/env python3
"""Resolve a deterministic review scope without changing checkout state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


class ScopeError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode != 0:
        raise ScopeError(p.stderr.strip() or f"git {' '.join(args)} failed")
    return p.stdout


def repo_root(repo: Path) -> Path:
    out = git(repo, "rev-parse", "--show-toplevel").strip()
    return Path(out).resolve()


def commit_info(repo: Path, sha: str) -> dict:
    sha = git(repo, "rev-parse", "--verify", f"{sha}^{{commit}}").strip()
    subject = git(repo, "show", "-s", "--format=%s", sha).rstrip("\n")
    message = git(repo, "show", "-s", "--format=%B", sha).rstrip("\n")
    parents = git(repo, "show", "-s", "--format=%P", sha).strip().split()
    if parents:
        baseline = parents[0]
        baseline_kind = "parent"
    else:
        # Canonical Git empty-tree object; use it as the baseline for a root commit.
        baseline = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        baseline_kind = "empty-tree"
    return {
        "mode": "commit",
        "commit": sha,
        "subject": subject,
        "message": message,
        "parents": parents,
        "baseline": baseline,
        "baseline_kind": baseline_kind,
    }


def all_commit_messages(repo: Path) -> list[tuple[str, str, str]]:
    # Record separator 0x1e, fields 0x1f. %B may contain newlines but practically not these separators.
    raw = git(repo, "log", "--all", "--format=%H%x1f%s%x1f%B%x1e")
    records = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split("\x1f", 2)
        if len(parts) != 3:
            continue
        sha, subject, message = parts
        records.append((sha.strip(), subject.rstrip("\n"), message.rstrip("\n")))
    return records


def resolve_message(repo: Path, selector: str) -> dict:
    selector = selector.rstrip("\n")
    records = all_commit_messages(repo)

    full = [(sha, subject) for sha, subject, message in records if message == selector]
    if len(full) == 1:
        result = commit_info(repo, full[0][0])
        result["selector_kind"] = "exact-full-message"
        return result
    if len(full) > 1:
        raise ScopeError(
            "Commit message is ambiguous: exact full message matches multiple commits: "
            + ", ".join(sha[:12] for sha, _ in full[:10])
        )

    subjects = [(sha, subject) for sha, subject, _ in records if subject == selector]
    if len(subjects) == 1:
        result = commit_info(repo, subjects[0][0])
        result["selector_kind"] = "exact-subject"
        return result
    if len(subjects) > 1:
        details = ", ".join(f"{sha[:12]} {subject}" for sha, subject in subjects[:10])
        raise ScopeError(f"Commit subject is ambiguous: {details}")

    raise ScopeError(
        "No unique exact commit-message match. Supply the exact full message, exact unique subject, or commit hash."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="Repository path (default: current directory)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--whole-codebase", action="store_true")
    group.add_argument("--commit-hash")
    group.add_argument("--commit-message")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    try:
        root = repo_root(Path(args.repo))
        if args.whole_codebase:
            result = {
                "mode": "whole-codebase",
                "repo_root": str(root),
                "head": git(root, "rev-parse", "HEAD").strip(),
            }
        elif args.commit_hash:
            result = commit_info(root, args.commit_hash)
            result["selector_kind"] = "hash"
            result["repo_root"] = str(root)
        else:
            result = resolve_message(root, args.commit_message)
            result["repo_root"] = str(root)

        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except ScopeError as exc:
        print(f"scope-error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

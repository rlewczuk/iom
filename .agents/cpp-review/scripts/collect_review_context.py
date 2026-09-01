#!/usr/bin/env python3
"""Collect compact repository or selected-commit review context."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


BACKEND_HINTS = {
    "cuda": ("cuda", "cublas", "cudnn", ".cu", ".cuh"),
    "hip": ("hip", "rocm", "rocblas", "hipblas"),
    "vulkan": ("vulkan", "spirv", "spir-v", ".comp", ".spv"),
    "sycl": ("sycl", "oneapi", "dpcpp", "dpc++"),
    "metal": ("metal", ".metal"),
    "opencl": ("opencl", ".cl"),
}


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"git {' '.join(args)} failed")
    return p.stdout


def infer_backends(paths: list[str]) -> list[str]:
    lower = "\n".join(paths).lower()
    result = []
    for backend, hints in BACKEND_HINTS.items():
        if any(h in lower for h in hints):
            result.append(backend)
    return result


def file_kind(path: str) -> str:
    p = path.lower()
    if p.endswith((".cpp", ".cc", ".cxx", ".c", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh")):
        return "cpp"
    if p.endswith((".md", ".rst", ".txt")):
        return "docs"
    if p.endswith(("cmakelists.txt", ".cmake", ".bazel", ".bzl", "meson.build")):
        return "build"
    if "/test" in p or p.startswith("test"):
        return "tests"
    return "other"


def commit_context(repo: Path, commit: str) -> dict:
    sha = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    parents = git(repo, "show", "-s", "--format=%P", sha).strip().split()
    baseline = parents[0] if parents else "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    name_status = git(repo, "diff", "--name-status", "--find-renames", baseline, sha)
    rows = []
    paths = []
    for line in name_status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        affected = parts[1:]
        rows.append({"status": status, "paths": affected})
        paths.extend(affected)
    return {
        "mode": "commit",
        "commit": sha,
        "subject": git(repo, "show", "-s", "--format=%s", sha).strip(),
        "baseline": baseline,
        "changed": rows,
        "file_kind_counts": dict(Counter(file_kind(p) for p in paths)),
        "backend_hints": infer_backends(paths),
        "diff_stat": git(repo, "diff", "--stat", baseline, sha).rstrip(),
    }


def whole_context(repo: Path) -> dict:
    paths = git(repo, "ls-files").splitlines()
    top = Counter((Path(p).parts[0] if Path(p).parts else p) for p in paths)
    return {
        "mode": "whole-codebase",
        "head": git(repo, "rev-parse", "HEAD").strip(),
        "tracked_files": len(paths),
        "file_kind_counts": dict(Counter(file_kind(p) for p in paths)),
        "backend_hints": infer_backends(paths),
        "top_level_file_counts": dict(top.most_common(20)),
        "change_specs": sorted({
            "/".join(Path(p).parts[:3])
            for p in paths
            if len(Path(p).parts) >= 3 and Path(p).parts[:2] == ("docs", "changes")
        })[:100],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--whole-codebase", action="store_true")
    group.add_argument("--commit")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    repo = Path(git(Path(args.repo), "rev-parse", "--show-toplevel").strip()).resolve()
    data = whole_context(repo) if args.whole_codebase else commit_context(repo, args.commit)
    data["repo_root"] = str(repo)
    print(json.dumps(data, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Perform deterministic Git and path operations for the spec-run-task skill."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


COMPONENT_RE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
STATUS_RE = re.compile(r"^\s*\*\*Status:\*\*\s*(.*?)\s*$", re.IGNORECASE)
BLOCKED_RE = re.compile(r"^\s*\*\*Blocked by:\*\*\s*(.*?)\s*$", re.IGNORECASE)
GENERATED_HEADING_RE = re.compile(r"^##\s+(Summary|Verification|Errors)\s*$", re.IGNORECASE)
SECTION_BOUNDARY_RE = re.compile(r"^#{1,2}\s+")
FINAL_SUBJECT_RE = re.compile(r"^spec-run-task\(([^)]+)\): (.+)$")
STATE_FILE = "spec-run-task-state.json"
TERMINAL_STATUSES = {"ready", "done", "failed", "blocked"}


class TaskError(RuntimeError):
    pass


class RebaseConflict(TaskError):
    pass


def git(
    repo: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        env=process_env,
    )
    if check and process.returncode != 0:
        stderr = process.stderr if isinstance(process.stderr, str) else process.stderr.decode(errors="replace")
        stdout = process.stdout if isinstance(process.stdout, str) else process.stdout.decode(errors="replace")
        detail = stderr.strip() or stdout.strip() or f"git {' '.join(args)} failed"
        raise TaskError(detail)
    return process


def root_for(repo: Path) -> Path:
    process = git(repo, "rev-parse", "--show-toplevel")
    return Path(process.stdout.strip()).resolve()


def common_git_dir(repo: Path) -> Path:
    process = git(repo, "rev-parse", "--git-common-dir")
    path = Path(process.stdout.strip())
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def absolute_git_dir(repo: Path) -> Path:
    process = git(repo, "rev-parse", "--absolute-git-dir")
    return Path(process.stdout.strip()).resolve()


def current_branch(repo: Path) -> str:
    process = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if process.returncode != 0 or not process.stdout.strip():
        raise TaskError(f"checkout has no named local branch: {repo}")
    return process.stdout.strip()


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD^{commit}").stdout.strip()


def resolve_commit(repo: Path, value: str) -> str:
    process = git(repo, "rev-parse", "--verify", f"{value}^{{commit}}", check=False)
    if process.returncode != 0:
        raise TaskError(f"commit does not resolve: {value}")
    return process.stdout.strip()


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False).returncode == 0


def status_entries(repo: Path) -> list[str]:
    process = git(repo, "status", "--porcelain=v1", "-z", binary=True)
    raw = process.stdout
    assert isinstance(raw, bytes)
    return [entry.decode(errors="surrogateescape") for entry in raw.split(b"\0") if entry]


def require_clean(repo: Path, label: str) -> None:
    entries = status_entries(repo)
    if entries:
        preview = ", ".join(entries[:8])
        raise TaskError(f"{label} is not clean: {preview}")


def validate_target(raw: str, *, branch_safe: bool = False) -> str:
    target = raw.strip()
    if not target:
        raise TaskError("target must be non-empty")
    if target != raw:
        raise TaskError("target must not contain surrounding whitespace")
    if target.startswith("/") or "\\" in target:
        raise TaskError(f"target must be a relative POSIX directory path: {raw}")
    if target.endswith("spec.md") or target.endswith("task.md"):
        raise TaskError("target names a directory, not spec.md or task.md")
    components = target.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise TaskError(f"target contains an empty, '.' or '..' component: {raw}")
    if branch_safe:
        invalid = [component for component in components if not COMPONENT_RE.fullmatch(component)]
        if invalid:
            raise TaskError(
                "task path has a component that cannot form the deterministic branch: "
                + ", ".join(invalid)
            )
    return target


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def changes_root(repo: Path) -> Path:
    return (repo / "docs" / "changes").resolve()


def spec_path(repo: Path, task: str) -> Path:
    root = changes_root(repo)
    candidate = (root / task / "spec.md").resolve()
    if not is_relative_to(candidate, root):
        raise TaskError(f"spec path escapes docs/changes: {task}")
    if not candidate.is_file():
        raise TaskError(f"specification not found: {repo / 'docs' / 'changes' / task / 'spec.md'}")
    return candidate


def annotation_path(repo: Path, task: str, *, require_spec: bool = True) -> Path:
    if require_spec:
        spec = spec_path(repo, task)
    else:
        spec = (changes_root(repo) / task / "spec.md").resolve()
    annotation = spec.with_name("task.md")
    if annotation.parent != spec.parent or not is_relative_to(annotation, changes_root(repo)):
        raise TaskError(f"invalid sibling annotation path for: {spec}")
    return annotation


def deterministic_branch(task: str) -> str:
    task = validate_target(task, branch_safe=True)
    branch = "run-task/" + "--".join(task.split("/"))
    process = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise TaskError(f"derived feature branch is invalid: {branch}")
    return branch


def worktree_path(repo: Path, task: str) -> Path:
    work_root = (repo / ".work").resolve()
    if not is_relative_to(work_root, repo):
        raise TaskError(f".work resolves outside repository: {work_root}")
    candidate = (work_root / task).resolve()
    if not is_relative_to(candidate, work_root):
        raise TaskError(f"worktree path escapes .work: {candidate}")
    return candidate


def require_work_ignored(repo: Path) -> None:
    probe = ".work/.spec-run-task-ignore-probe"
    process = git(repo, "check-ignore", "--quiet", "--no-index", "--", probe, check=False)
    if process.returncode != 0:
        raise TaskError(".work/ is not ignored by Git")


def validate_control(repo: Path, *, require_status_clean: bool = True) -> tuple[str, str]:
    branch = current_branch(repo)
    if require_status_clean:
        require_clean(repo, "integration checkout")
    require_work_ignored(repo)
    return branch, head(repo)


def parse_annotation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "status": None}
    text = path.read_text(encoding="utf-8")
    statuses = [match.group(1).strip().lower() for line in text.splitlines() if (match := STATUS_RE.match(line))]
    if len(statuses) > 1:
        raise TaskError(f"multiple Status fields in annotation: {path}")
    return {"exists": True, "status": statuses[0] if statuses else None}


def blocked_tokens(spec: Path) -> list[str]:
    fields = [match.group(1).strip() for line in spec.read_text(encoding="utf-8").splitlines() if (match := BLOCKED_RE.match(line))]
    if len(fields) > 1:
        raise TaskError(f"multiple Blocked by fields in specification: {spec}")
    if not fields or fields[0].lower() in {"", "none"}:
        return []
    tokens = []
    for value in fields[0].split(","):
        token = value.strip().strip("`").strip()
        if not token:
            raise TaskError(f"empty blocker in specification: {spec}")
        tokens.append(token)
    return tokens




def specs_below(repo: Path, target: str) -> list[Path]:
    root = changes_root(repo)
    target_dir = spec_path(repo, target).parent
    specs = []
    for candidate in target_dir.rglob("spec.md"):
        resolved = candidate.resolve()
        if not is_relative_to(resolved, root):
            raise TaskError(f"specification resolves outside docs/changes: {candidate}")
        specs.append(resolved)
    return sorted(set(specs))


def relative_task(repo: Path, spec: Path) -> str:
    return spec.parent.relative_to(changes_root(repo)).as_posix()


def leaf_specs(repo: Path, target: str) -> tuple[str, list[Path], list[Path]]:
    specs = specs_below(repo, target)
    target_spec = spec_path(repo, target)
    descendants = [spec for spec in specs if spec != target_spec]
    if not descendants:
        return "leaf", [target_spec], specs
    leaves = []
    for candidate in descendants:
        directory = candidate.parent
        if not any(other != candidate and is_relative_to(other.parent, directory) for other in descendants):
            leaves.append(candidate)
    return "container", sorted(leaves), specs


def resolve_blocker(
    repo: Path,
    requested_target: str,
    token: str,
    selected_specs: list[Path],
    repository_specs: list[Path],
) -> str:
    normalized = token
    if normalized.startswith("docs/changes/"):
        normalized = normalized[len("docs/changes/") :]
    if normalized.endswith("/spec.md"):
        normalized = normalized[: -len("/spec.md")]
    by_task = {relative_task(repo, spec): spec for spec in repository_specs}
    direct = [normalized, f"{requested_target}/{normalized}"]
    for candidate in direct:
        if candidate in by_task:
            return candidate
    selected_tasks = [relative_task(repo, spec) for spec in selected_specs]
    basename_matches = [task for task in selected_tasks if task.rsplit("/", 1)[-1] == normalized]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        raise TaskError(f"ambiguous blocker '{token}': {', '.join(basename_matches)}")
    raise TaskError(f"unknown blocker '{token}' beneath target {requested_target}")


def detect_cycle(edges: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            start = stack.index(node)
            raise TaskError("dependency cycle: " + " -> ".join(stack[start:] + [node]))
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for dependency in edges.get(node, []):
            if dependency in edges:
                visit(dependency, stack)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node, [])


def inspect_target(repo: Path, target: str) -> dict[str, Any]:
    target = validate_target(target)
    integration_branch, integration_head = validate_control(repo)
    kind, leaves, selected_specs = leaf_specs(repo, target)
    repository_specs = []
    root = changes_root(repo)
    if root.is_dir():
        for candidate in root.rglob("spec.md"):
            resolved = candidate.resolve()
            if not is_relative_to(resolved, root):
                raise TaskError(f"specification resolves outside docs/changes: {candidate}")
            repository_specs.append(resolved)
    repository_specs = sorted(set(repository_specs))

    leaf_tasks = {relative_task(repo, spec) for spec in leaves}
    records = []
    edges: dict[str, list[str]] = {}
    for leaf in leaves:
        task = relative_task(repo, leaf)
        branch = deterministic_branch(task)
        annotation = annotation_path(repo, task)
        parsed = parse_annotation(annotation)
        dependencies = [
            resolve_blocker(repo, target, token, selected_specs, repository_specs)
            for token in blocked_tokens(leaf)
        ]
        edges[task] = [dependency for dependency in dependencies if dependency in leaf_tasks]
        dependency_records = []
        for dependency in dependencies:
            dependency_annotation = annotation_path(repo, dependency)
            dependency_state = parse_annotation(dependency_annotation)
            dependency_records.append(
                {
                    "task_path": dependency,
                    "status": dependency_state["status"],
                    "selected_leaf": dependency in leaf_tasks,
                    "satisfied": dependency_state["status"] == "done",
                }
            )
        wt = worktree_path(repo, task)
        records.append(
            {
                "task_path": task,
                "spec_path": str(leaf),
                "annotation_path": str(annotation),
                "worktree": str(wt),
                "worktree_spec_path": str(wt / "docs" / "changes" / task / "spec.md"),
                "worktree_annotation_path": str(wt / "docs" / "changes" / task / "task.md"),
                "feature_branch": branch,
                "status": parsed["status"],
                "annotation_exists": parsed["exists"],
                "blocked_by": dependency_records,
                "explicitly_ready": parsed["status"] != "done"
                and all(dependency["satisfied"] for dependency in dependency_records),
            }
        )
    detect_cycle(edges)
    return {
        "repo_root": str(repo),
        "integration_branch": integration_branch,
        "integration_head": integration_head,
        "requested_target": target,
        "target_kind": kind,
        "target_spec_path": str(spec_path(repo, target)),
        "target_annotation_path": str(annotation_path(repo, target)),
        "leaves": records,
    }


def worktree_records(repo: Path) -> list[dict[str, str]]:
    process = git(repo, "worktree", "list", "--porcelain", "-z", binary=True)
    raw = process.stdout
    assert isinstance(raw, bytes)
    records: list[dict[str, str]] = []
    record: dict[str, str] = {}
    for field in raw.split(b"\0"):
        if not field:
            if record:
                records.append(record)
                record = {}
            continue
        text = field.decode(errors="surrogateescape")
        key, _, value = text.partition(" ")
        record[key] = value
    if record:
        records.append(record)
    return records


def rebase_in_progress(worktree: Path) -> bool:
    for name in ("rebase-merge", "rebase-apply"):
        path = Path(git(worktree, "rev-parse", "--git-path", name).stdout.strip())
        if not path.is_absolute():
            path = worktree / path
        if path.exists():
            return True
    return False


def verify_worktree(repo: Path, task: str, *, allow_detached: bool = False) -> tuple[Path, str]:
    branch = deterministic_branch(task)
    wt = worktree_path(repo, task)
    registered = [record for record in worktree_records(repo) if Path(record["worktree"]).resolve() == wt]
    if len(registered) != 1:
        raise TaskError(f"expected registered worktree not found: {wt}")
    if common_git_dir(wt) != common_git_dir(repo):
        raise TaskError(f"worktree belongs to a different Git repository: {wt}")
    if allow_detached:
        # A script-managed rebase detaches HEAD while conflicts are resolved;
        # the worktree identity and repository ownership are still verified.
        return wt, branch
    expected_ref = f"refs/heads/{branch}"
    if registered[0].get("branch") != expected_ref:
        raise TaskError(
            f"worktree {wt} is registered on {registered[0].get('branch', 'detached HEAD')}, expected {expected_ref}"
        )
    if current_branch(wt) != branch:
        raise TaskError(f"worktree branch mismatch at {wt}")
    return wt, branch


def state_path(worktree: Path) -> Path:
    return absolute_git_dir(worktree) / STATE_FILE


def load_state(repo: Path, task: str) -> tuple[Path, dict[str, Any]]:
    probe = worktree_path(repo, task)
    allow_detached = probe.is_dir() and rebase_in_progress(probe)
    wt, branch = verify_worktree(repo, task, allow_detached=allow_detached)
    path = state_path(wt)
    if not path.is_file():
        raise TaskError(f"task worktree has no script state; run prepare first: {wt}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("task_path") != task or state.get("feature_branch") != branch:
        raise TaskError(f"task worktree state does not match {task}: {path}")
    return wt, state


def save_state(worktree: Path, state: dict[str, Any]) -> None:
    path = state_path(worktree)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{STATE_FILE}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_integration(state: dict[str, Any]) -> Path:
    checkout = Path(state["integration_checkout"]).resolve()
    if current_branch(checkout) != state["integration_branch"]:
        raise TaskError(
            f"integration checkout changed branch: expected {state['integration_branch']}, found {current_branch(checkout)}"
        )
    require_clean(checkout, "integration checkout")
    return checkout


def prepare_task(
    repo: Path,
    task: str,
    run_target: str,
    integration_branch: str,
    integration_base: str,
) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    run_target = validate_target(run_target)
    spec_path(repo, task)
    actual_branch, actual_head = validate_control(repo)
    if actual_branch != integration_branch:
        raise TaskError(f"integration branch changed: expected {integration_branch}, found {actual_branch}")
    integration_base = resolve_commit(repo, integration_base)
    if actual_head != integration_base:
        raise TaskError(f"integration branch advanced: expected {integration_base}, found {actual_head}; run inspect again")
    branch = deterministic_branch(task)
    wt = worktree_path(repo, task)
    records = worktree_records(repo)
    at_path = [record for record in records if Path(record["worktree"]).resolve() == wt]
    branch_elsewhere = [record for record in records if record.get("branch") == f"refs/heads/{branch}"]
    created = False

    if at_path:
        if len(at_path) != 1 or at_path[0].get("branch") != f"refs/heads/{branch}":
            found = at_path[0].get("branch", "detached HEAD")
            raise TaskError(f"worktree path already registered on {found}: {wt}")
    elif branch_elsewhere:
        locations = ", ".join(record["worktree"] for record in branch_elsewhere)
        raise TaskError(f"feature branch {branch} is checked out elsewhere: {locations}")
    else:
        if wt.exists() and any(wt.iterdir()):
            raise TaskError(f"unregistered worktree path is not empty: {wt}")
        wt.parent.mkdir(parents=True, exist_ok=True)
        ref_exists = git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0
        if ref_exists:
            git(repo, "worktree", "add", "--", str(wt), branch)
        else:
            git(repo, "worktree", "add", "-b", branch, "--", str(wt), integration_base)
        created = True

    wt, branch = verify_worktree(repo, task)
    path = state_path(wt)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("task_path") != task or state.get("feature_branch") != branch:
            raise TaskError(f"existing worktree state belongs to another task: {path}")
        base = resolve_commit(wt, state["base"])
        if not state.get("pending_rebase_onto") and not is_ancestor(wt, base, head(wt)):
            raise TaskError(f"recorded task base is not an ancestor of its branch: {base}")
    else:
        merge_base = git(wt, "merge-base", "HEAD", integration_base, check=False)
        if merge_base.returncode != 0 or not merge_base.stdout.strip():
            raise TaskError(f"cannot infer a safe task base for reused branch {branch}")
        base = merge_base.stdout.strip()
        state = {}

    state.update(
        {
            "version": 1,
            "task_path": task,
            "run_target": run_target,
            "feature_branch": branch,
            "worktree": str(wt),
            "integration_checkout": str(repo),
            "integration_branch": integration_branch,
            "integration_start": integration_base,
            "base": base,
        }
    )
    save_state(wt, state)
    output = {
        "task_path": task,
        "created": created,
        "reused": not created,
        "feature_branch": branch,
        "worktree": str(wt),
        "spec_path": str(wt / "docs" / "changes" / task / "spec.md"),
        "annotation_path": str(wt / "docs" / "changes" / task / "task.md"),
        "base": base,
        "head": head(wt),
        "pending_rebase_onto": state.get("pending_rebase_onto"),
    }
    output.update(worktree_report(wt, state))
    return output


def task_is_ancestor(ancestor: str, descendant: str) -> bool:
    return descendant == ancestor or descendant.startswith(ancestor + "/")


def validate_annotation_owner(state: dict[str, Any], owner: str, annotated: str) -> None:
    if annotated == owner:
        return
    run_target = state["run_target"]
    if not task_is_ancestor(annotated, owner) or not task_is_ancestor(run_target, annotated):
        raise TaskError(f"{annotated} is not an owned container annotation for leaf {owner}")


def strip_generated_sections(text: str) -> list[str]:
    lines = text.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        if STATUS_RE.match(lines[index]):
            index += 1
            continue
        if GENERATED_HEADING_RE.match(lines[index]):
            index += 1
            while index < len(lines) and not SECTION_BOUNDARY_RE.match(lines[index]):
                index += 1
            continue
        kept.append(lines[index])
        index += 1
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def single_line(value: str, label: str) -> str:
    value = value.strip()
    if not value or "\n" in value or "\r" in value:
        raise TaskError(f"{label} must be one non-empty line")
    return value


def annotate_task(
    repo: Path,
    owner: str,
    annotated: str,
    status: str,
    summary: str,
    verification: list[str],
    errors: list[str],
) -> dict[str, Any]:
    owner = validate_target(owner, branch_safe=True)
    annotated = validate_target(annotated)
    wt, state = load_state(repo, owner)
    validate_integration(state)
    validate_annotation_owner(state, owner, annotated)
    status = status.lower()
    if status not in TERMINAL_STATUSES:
        raise TaskError(f"unsupported persistent status: {status}")
    summary = single_line(summary, "summary")
    verification = [single_line(item, "verification entry") for item in verification]
    errors = [single_line(item, "error entry") for item in errors]
    if status == "done" and not verification:
        raise TaskError("done annotation requires at least one observed verification entry")
    if status in {"failed", "blocked"} and not errors:
        raise TaskError(f"{status} annotation requires at least one concrete error entry")
    if status in {"ready", "done"} and errors:
        raise TaskError(f"{status} annotation cannot retain error entries")

    annotation = annotation_path(wt, annotated)
    existing = annotation.read_text(encoding="utf-8") if annotation.is_file() else ""
    body = strip_generated_sections(existing)
    output = [f"**Status:** {status}"]
    if body:
        output.extend(["", *body])
    output.extend(["", "## Summary", "", summary])
    if verification:
        output.extend(["", "## Verification", ""])
        output.extend(f"- {entry}" for entry in verification)
    if errors:
        output.extend(["", "## Errors", ""])
        output.extend(f"- {entry}" for entry in errors)
    annotation.write_text("\n".join(output) + "\n", encoding="utf-8")
    parsed = parse_annotation(annotation)
    return {
        "owner_task": owner,
        "annotated_task": annotated,
        "annotation_path": str(annotation),
        "status": parsed["status"],
    }


def changed_paths(repo: Path, base: str, revision: str | None = None) -> list[str]:
    args = ["diff", "--name-only", "-z", base]
    if revision is not None:
        args.append(revision)
    process = git(repo, *args, binary=True)
    raw = process.stdout
    assert isinstance(raw, bytes)
    return [item.decode(errors="surrogateescape") for item in raw.split(b"\0") if item]


def annotation_task_from_path(path: str) -> str | None:
    prefix = "docs/changes/"
    suffix = "/task.md"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    task = path[len(prefix) : -len(suffix)]
    return task or None


def annotation_status_at_base(worktree: Path, base: str, task: str) -> str | None:
    process = git(
        worktree,
        "show",
        f"{base}:docs/changes/{task}/task.md",
        check=False,
    )
    if process.returncode != 0:
        return None
    statuses = [m.group(1).strip().lower() for line in process.stdout.splitlines() if (m := STATUS_RE.match(line))]
    return statuses[0] if statuses else None


def validate_annotation_changes(
    worktree: Path,
    state: dict[str, Any],
    paths: Iterable[str],
    owner_status: str | None,
    *,
    require_owner: bool,
) -> None:
    owner = state["task_path"]
    annotation_tasks = []
    for path in paths:
        if Path(path).name != "task.md":
            continue
        annotated = annotation_task_from_path(path)
        if annotated is None:
            raise TaskError(f"misplaced task.md change: {path}")
        validate_annotation_owner(state, owner, annotated)
        if annotated != owner and owner_status != "done":
            raise TaskError(f"container annotation cannot be committed before leaf is done: {path}")
        associated_spec = worktree / "docs" / "changes" / annotated / "spec.md"
        if not associated_spec.is_file():
            raise TaskError(f"annotation has no sibling spec.md: {path}")
        annotation_tasks.append(annotated)
    if require_owner and owner not in annotation_tasks:
        # An annotation-only container roll-up may leave the owner's own leaf
        # annotation unchanged when it already carries the expected status at base.
        if annotation_status_at_base(worktree, state["base"], owner) != owner_status:
            expected = f"docs/changes/{owner}/task.md"
            raise TaskError(f"task commit does not contain its required sibling annotation: {expected}")


def commits_since(repo: Path, base: str) -> list[str]:
    process = git(repo, "rev-list", "--reverse", f"{base}..HEAD")
    return [line for line in process.stdout.splitlines() if line]


def commit_subject(repo: Path, commit: str) -> str:
    return git(repo, "show", "-s", "--format=%s", commit).stdout.rstrip("\n")


def commit_parents(repo: Path, commit: str) -> list[str]:
    return git(repo, "show", "-s", "--format=%P", commit).stdout.strip().split()


def recognized_task_subject(subject: str, task: str) -> bool:
    return subject.startswith(f"spec-run-task({task}): ") or subject.startswith(
        f"spec-run-task checkpoint({task}): "
    )


def require_no_rebase(worktree: Path) -> None:
    if rebase_in_progress(worktree):
        raise TaskError(f"rebase is in progress in {worktree}; use continue-rebase or abort-rebase")


def validate_prior_task_commits(worktree: Path, state: dict[str, Any]) -> list[str]:
    base = resolve_commit(worktree, state["base"])
    branch_head = head(worktree)
    if not is_ancestor(worktree, base, branch_head):
        raise TaskError(f"recorded base {base} is not an ancestor of task branch")
    commits = commits_since(worktree, base)
    for commit in commits:
        parents = commit_parents(worktree, commit)
        if len(parents) != 1:
            raise TaskError(f"task history contains a merge or root commit: {commit}")
        subject = commit_subject(worktree, commit)
        if not recognized_task_subject(subject, state["task_path"]):
            raise TaskError(f"refusing to rewrite commit not owned by task: {commit} {subject}")
    return commits

def worktree_report(worktree: Path, state: dict[str, Any]) -> dict[str, Any]:
    base = resolve_commit(worktree, state["base"])
    branch_head = head(worktree)
    base_is_ancestor = is_ancestor(worktree, base, branch_head)
    commits = []
    if base_is_ancestor:
        for commit in commits_since(worktree, base):
            commits.append(
                {
                    "commit": commit,
                    "parents": commit_parents(worktree, commit),
                    "subject": commit_subject(worktree, commit),
                }
            )
    return {
        "dirty": bool(status_entries(worktree)),
        "status_entries": status_entries(worktree),
        "base_is_ancestor": base_is_ancestor,
        "task_commits": commits,
    }


def show_task(repo: Path, task: str) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    worktree, state = load_state(repo, task)
    validate_integration(state)
    return {
        "task_path": task,
        "feature_branch": state["feature_branch"],
        "worktree": str(worktree),
        "spec_path": str(worktree / "docs" / "changes" / task / "spec.md"),
        "annotation_path": str(worktree / "docs" / "changes" / task / "task.md"),
        "base": state["base"],
        "head": head(worktree),
        "pending_rebase_onto": state.get("pending_rebase_onto"),
        **worktree_report(worktree, state),
    }




def checkpoint_task(repo: Path, task: str) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    wt, state = load_state(repo, task)
    validate_integration(state)
    require_no_rebase(wt)
    if not status_entries(wt):
        return {"task_path": task, "checkpoint": None, "no_change": True}
    git(wt, "add", "-A")
    paths = changed_paths(wt, state["base"])
    owner_state = parse_annotation(annotation_path(wt, task))["status"]
    validate_annotation_changes(wt, state, paths, owner_state, require_owner=False)
    git(wt, "commit", "-m", f"spec-run-task checkpoint({task}): preserve reusable work")
    commit = head(wt)
    return {"task_path": task, "checkpoint": commit, "no_change": False}


def commit_task(repo: Path, task: str, expected_status: str, outcome: str | None) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    expected_status = expected_status.lower()
    if expected_status not in TERMINAL_STATUSES:
        raise TaskError(f"unsupported commit status: {expected_status}")
    wt, state = load_state(repo, task)
    validate_integration(state)
    require_no_rebase(wt)
    prior_commits = validate_prior_task_commits(wt, state)
    if outcome is None:
        if len(prior_commits) != 1:
            raise TaskError("outcome is required until the task has one final task commit")
        prior_subject = commit_subject(wt, prior_commits[0])
        match = FINAL_SUBJECT_RE.fullmatch(prior_subject)
        if not match or match.group(1) != task:
            raise TaskError("outcome is required because the existing commit has no final task subject")
        outcome = match.group(2)
    outcome = single_line(outcome, "outcome")
    annotation = annotation_path(wt, task)
    parsed = parse_annotation(annotation)
    if parsed["status"] != expected_status:
        raise TaskError(
            f"annotation status mismatch at {annotation}: expected {expected_status}, found {parsed['status']}"
        )

    git(wt, "add", "-A")
    paths = changed_paths(wt, state["base"])
    validate_annotation_changes(wt, state, paths, parsed["status"], require_owner=True)
    if not paths:
        raise TaskError("task has no changes relative to its recorded base")

    git(wt, "reset", "--soft", state["base"])
    git(wt, "add", "-A")
    subject = f"spec-run-task({task}): {outcome}"
    git(wt, "commit", "-m", subject)
    require_clean(wt, "task worktree")
    state["final_commit"] = head(wt)
    save_state(wt, state)
    return check_task(repo, task, expected_status)


def check_task(repo: Path, task: str, expected_status: str | None) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    wt, state = load_state(repo, task)
    validate_integration(state)
    require_no_rebase(wt)
    require_clean(wt, "task worktree")
    commits = validate_prior_task_commits(wt, state)
    if len(commits) != 1:
        raise TaskError(f"task branch must contain exactly one commit above {state['base']}; found {len(commits)}")
    commit = commits[0]
    subject = commit_subject(wt, commit)
    match = FINAL_SUBJECT_RE.fullmatch(subject)
    if not match or match.group(1) != task:
        raise TaskError(f"task commit subject is invalid: {subject}")
    annotation = annotation_path(wt, task)
    parsed = parse_annotation(annotation)
    if expected_status and parsed["status"] != expected_status.lower():
        raise TaskError(
            f"annotation status mismatch at {annotation}: expected {expected_status.lower()}, found {parsed['status']}"
        )
    paths = changed_paths(wt, state["base"], "HEAD")
    validate_annotation_changes(wt, state, paths, parsed["status"], require_owner=True)
    return {
        "task_path": task,
        "worktree": str(wt),
        "feature_branch": state["feature_branch"],
        "base": state["base"],
        "commit": commit,
        "subject": subject,
        "status": parsed["status"],
        "annotation_path": str(annotation),
        "changed_paths": paths,
    }


def rebase_task(repo: Path, task: str, onto: str) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    wt, state = load_state(repo, task)
    validate_integration(state)
    require_no_rebase(wt)
    require_clean(wt, "task worktree")
    onto = resolve_commit(wt, onto)
    commits = validate_prior_task_commits(wt, state)
    if len(commits) > 1:
        raise TaskError("consolidate task checkpoints with commit before rebasing")
    if not commits:
        if not is_ancestor(wt, state["base"], onto):
            raise TaskError(
                f"empty task branch can only move forward from {state['base']} to a descendant base"
            )
        git(wt, "reset", "--hard", onto)
        state["base"] = onto
        state.pop("final_commit", None)
        save_state(wt, state)
        return {"task_path": task, "base": onto, "commit": None, "no_task_commit": True}
    subject = commit_subject(wt, commits[0])
    match = FINAL_SUBJECT_RE.fullmatch(subject)
    if not match or match.group(1) != task:
        raise TaskError("task commit must have its final subject before rebasing")
    if onto == state["base"]:
        return check_task(repo, task, None) | {"rebased": False}
    if is_ancestor(wt, commits[0], onto):
        raise TaskError("target base already contains the task commit; refusing to replay it")

    state["pending_rebase_onto"] = onto
    save_state(wt, state)
    process = git(
        wt,
        "rebase",
        "--onto",
        onto,
        state["base"],
        state["feature_branch"],
        check=False,
        env={"GIT_EDITOR": "true"},
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise RebaseConflict(
            f"rebase stopped in {wt}: {detail}; resolve files, then run continue-rebase for {task}"
        )
    state["base"] = onto
    state.pop("pending_rebase_onto", None)
    state["final_commit"] = head(wt)
    save_state(wt, state)
    return check_task(repo, task, None) | {"rebased": True}


def continue_rebase(repo: Path, task: str) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    wt, state = load_state(repo, task)
    validate_integration(state)
    onto = state.get("pending_rebase_onto")
    if not onto or not rebase_in_progress(wt):
        raise TaskError(f"no script-managed rebase is in progress for {task}")
    git(wt, "add", "-A")
    process = git(wt, "rebase", "--continue", check=False, env={"GIT_EDITOR": "true"})
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        raise RebaseConflict(
            f"rebase still stopped in {wt}: {detail}; resolve remaining files, then run continue-rebase again"
        )
    state["base"] = onto
    state.pop("pending_rebase_onto", None)
    state["final_commit"] = head(wt)
    save_state(wt, state)
    return check_task(repo, task, None) | {"rebased": True}


def abort_rebase(repo: Path, task: str) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    wt, state = load_state(repo, task)
    validate_integration(state)
    if not state.get("pending_rebase_onto") or not rebase_in_progress(wt):
        raise TaskError(f"no script-managed rebase is in progress for {task}")
    git(wt, "rebase", "--abort")
    state.pop("pending_rebase_onto", None)
    save_state(wt, state)
    return {"task_path": task, "aborted": True, "base": state["base"], "head": head(wt)}


def commit_changed_paths(repo: Path, commit: str) -> list[str]:
    process = git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        commit,
        binary=True,
    )
    raw = process.stdout
    assert isinstance(raw, bytes)
    return [item.decode(errors="surrogateescape") for item in raw.split(b"\0") if item]

def blob_at_commit(repo: Path, commit: str, path: str) -> str | None:
    process = git(repo, "show", f"{commit}:{path}", check=False)
    if process.returncode != 0:
        return None
    return process.stdout


def annotation_status_in_commit(repo: Path, commit: str, task: str) -> str | None:
    text = blob_at_commit(repo, commit, f"docs/changes/{task}/task.md")
    if text is None:
        return None
    statuses = [m.group(1).strip().lower() for line in text.splitlines() if (m := STATUS_RE.match(line))]
    return statuses[0] if statuses else None


def integrate_task(repo: Path, task: str) -> dict[str, Any]:
    task = validate_target(task, branch_safe=True)
    wt, state = load_state(repo, task)
    integration = validate_integration(state)
    if integration != repo:
        raise TaskError(f"integration checkout mismatch: state records {integration}, command selected {repo}")
    require_no_rebase(wt)
    require_clean(wt, "task worktree")
    integration_head = head(integration)
    feature_head = head(wt)
    if not is_ancestor(integration, integration_head, feature_head):
        raise TaskError(
            f"feature train is not a fast-forward of current integration tip {integration_head}; rebase it first"
        )
    commits = [line for line in git(integration, "rev-list", "--reverse", f"{integration_head}..{feature_head}").stdout.splitlines() if line]
    if not commits:
        raise TaskError("feature branch has no commits to integrate")
    seen_tasks: set[str] = set()
    details = []
    for commit in commits:
        if len(commit_parents(integration, commit)) != 1:
            raise TaskError(f"feature train contains a merge or root commit: {commit}")
        subject = commit_subject(integration, commit)
        match = FINAL_SUBJECT_RE.fullmatch(subject)
        if not match:
            raise TaskError(f"feature train contains a non-task commit: {commit} {subject}")
        commit_task_path = validate_target(match.group(1), branch_safe=True)
        if commit_task_path in seen_tasks:
            raise TaskError(f"feature train contains multiple commits for task: {commit_task_path}")
        seen_tasks.add(commit_task_path)
        paths = commit_changed_paths(integration, commit)
        # The leaf's own annotation must be done in this commit's tree, whether or
        # not this particular commit modified it (an annotation-only roll-up does not).
        if annotation_status_in_commit(integration, commit, commit_task_path) != "done":
            raise TaskError(
                f"task commit lacks a done sibling annotation: {commit} docs/changes/{commit_task_path}/task.md"
            )
        for path in paths:
            if Path(path).name != "task.md":
                continue
            annotated = annotation_task_from_path(path)
            if annotated is None:
                raise TaskError(f"feature train changes misplaced task.md: {path}")
            if annotated != commit_task_path:
                if not task_is_ancestor(annotated, commit_task_path) or not task_is_ancestor(
                    state["run_target"], annotated
                ):
                    raise TaskError(f"task commit changes an unowned annotation: {commit} {path}")
            associated_spec = blob_at_commit(integration, commit, f"docs/changes/{annotated}/spec.md")
            if associated_spec is None:
                raise TaskError(f"feature train annotation lacks sibling spec.md: {path}")
            if annotation_status_in_commit(integration, commit, annotated) != "done":
                raise TaskError(f"feature train annotation is not done: {path}")
        details.append({"task_path": commit_task_path, "commit": commit, "subject": subject})

    git(integration, "merge", "--ff-only", state["feature_branch"])
    require_clean(integration, "integration checkout")
    return {
        "integration_branch": state["integration_branch"],
        "previous_head": integration_head,
        "integration_head": head(integration),
        "feature_branch": state["feature_branch"],
        "commits": details,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", default=".", help="Integration checkout path")
    result.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    commands = result.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="Validate and resolve a requested target")
    inspect.add_argument("target")

    prepare = commands.add_parser("prepare", help="Provision or reuse one exact task worktree")
    prepare.add_argument("task")
    prepare.add_argument("--run-target", required=True)
    prepare.add_argument("--integration-branch", required=True)
    prepare.add_argument("--integration-base", required=True)

    show = commands.add_parser("show", help="Report reusable worktree state without mutation")
    show.add_argument("task")

    annotate = commands.add_parser("annotate", help="Write an annotation at its computed sibling path")
    annotate.add_argument("task", help="Owning executable leaf")
    annotate.add_argument("--for-task", dest="annotated")
    annotate.add_argument("--status", required=True, choices=sorted(TERMINAL_STATUSES))
    annotate.add_argument("--summary", required=True)
    annotate.add_argument("--verification", action="append", default=[])
    annotate.add_argument("--error", action="append", default=[])

    checkpoint = commands.add_parser("checkpoint", help="Preserve dirty reusable task state")
    checkpoint.add_argument("task")

    commit = commands.add_parser("commit", help="Consolidate all task-owned work into one commit")
    commit.add_argument("task")
    commit.add_argument("--status", required=True, choices=sorted(TERMINAL_STATUSES))
    commit.add_argument("--outcome")

    check = commands.add_parser("check", help="Validate the single task commit and annotation")
    check.add_argument("task")
    check.add_argument("--status", choices=sorted(TERMINAL_STATUSES))

    rebase = commands.add_parser("rebase", help="Replay the single task commit onto a new base")
    rebase.add_argument("task")
    rebase.add_argument("--onto", required=True)

    continuation = commands.add_parser("continue-rebase", help="Stage conflict resolutions and continue")
    continuation.add_argument("task")

    abort = commands.add_parser("abort-rebase", help="Abort a script-managed rebase")
    abort.add_argument("task")

    integrate = commands.add_parser("integrate", help="Validate and fast-forward the integration checkout")
    integrate.add_argument("task", help="Final task branch in the verified train")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        repo = root_for(Path(args.repo))
        if args.command == "inspect":
            output = inspect_target(repo, args.target)
        elif args.command == "prepare":
            output = prepare_task(
                repo,
                args.task,
                args.run_target,
                args.integration_branch,
                args.integration_base,
            )
        elif args.command == "show":
            output = show_task(repo, args.task)
        elif args.command == "annotate":
            output = annotate_task(
                repo,
                args.task,
                args.annotated or args.task,
                args.status,
                args.summary,
                args.verification,
                args.error,
            )
        elif args.command == "checkpoint":
            output = checkpoint_task(repo, args.task)
        elif args.command == "commit":
            output = commit_task(repo, args.task, args.status, args.outcome)
        elif args.command == "check":
            output = check_task(repo, args.task, args.status)
        elif args.command == "rebase":
            output = rebase_task(repo, args.task, args.onto)
        elif args.command == "continue-rebase":
            output = continue_rebase(repo, args.task)
        elif args.command == "abort-rebase":
            output = abort_rebase(repo, args.task)
        elif args.command == "integrate":
            output = integrate_task(repo, args.task)
        else:
            raise AssertionError(args.command)
        print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except RebaseConflict as exc:
        print(f"spec-run-task conflict: {exc}", file=sys.stderr)
        return 3
    except (TaskError, OSError, json.JSONDecodeError) as exc:
        print(f"spec-run-task: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

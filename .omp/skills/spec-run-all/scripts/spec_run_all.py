#!/usr/bin/env python3
"""Scan, queue, and prepare direct spec-run-task children."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The task helper is deliberately the single owner of path, control, and
# worktree mechanics.  This script only owns the read-only queue policy.
sys.dont_write_bytecode = True
_HELPER_DIR = Path(__file__).resolve().parents[2] / "spec-run-task" / "scripts"
sys.path.insert(0, str(_HELPER_DIR))
import spec_run_task as helper


class QueueError(RuntimeError):
    pass


def _target_dir(repo: Path, target: str) -> tuple[str, Path, Path]:
    target = helper.validate_target(target)
    root = helper.changes_root(repo)
    directory = (root / target).resolve()
    if not helper.is_relative_to(directory, root):
        raise QueueError(f"target escapes docs/changes: {target}")
    if not directory.is_dir():
        raise QueueError(f"target directory not found: {root / target}")
    return target, root, directory


def _metadata(spec: Path) -> tuple[str | None, str | None]:
    fields = []
    with spec.open(encoding="utf-8") as stream:
        for line in stream:
            if match := helper.BLOCKED_RE.match(line):
                fields.append(match.group(1).strip())
    if len(fields) != 1:
        return None, "expected exactly one Blocked by field"
    return fields[0], None


def _annotation(repo: Path, task: str) -> tuple[Path, dict[str, Any], str | None]:
    path = helper.annotation_path(repo, task, require_spec=False)
    try:
        parsed = helper.parse_annotation(path)
        return path, parsed, None
    except (helper.TaskError, OSError, UnicodeError) as exc:
        return path, {"exists": path.is_file(), "status": None}, str(exc)


def scan(repo: Path, target: str) -> dict[str, Any]:
    target, root, directory = _target_dir(repo, target)
    tasks: list[dict[str, Any]] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        spec = (child / "spec.md").resolve()
        if not spec.is_file():
            continue
        if not helper.is_relative_to(spec, root):
            raise QueueError(f"specification resolves outside docs/changes: {child / 'spec.md'}")
        name = child.name
        task = f"{target}/{name}"
        annotation, parsed, annotation_error = _annotation(repo, task)
        try:
            raw_blocked, metadata_error = _metadata(spec)
        except (OSError, UnicodeError) as exc:
            raw_blocked, metadata_error = None, str(exc)
        errors = [error for error in (metadata_error, annotation_error) if error]
        nested = any(path.resolve() != spec for path in child.rglob("spec.md"))
        record: dict[str, Any] = {
            "name": name,
            "task_path": task,
            "spec_path": str(spec),
            "annotation_path": str(annotation),
            "status": parsed.get("status"),
            "annotation_exists": bool(parsed.get("exists")),
            "blocked_by": raw_blocked,
        }
        if nested:
            record["nested"] = True
        if errors:
            record["error"] = "; ".join(errors)
        tasks.append(record)
    return {"repo_root": str(repo), "target": target, "tasks": tasks}


def _parse_tokens(raw: str | None) -> tuple[list[str], str | None]:
    if raw is None or not raw.strip():
        return [], "empty Blocked by field; use None for no dependencies"
    if raw.strip().lower() == "none":
        return [], None
    tokens = raw.split(",")
    result: list[str] = []
    for token in tokens:
        value = token.strip()
        if value.startswith("`") or value.endswith("`"):
            if not (value.startswith("`") and value.endswith("`") and len(value) >= 2):
                return [], f"malformed dependency token '{value}'"
            value = value[1:-1].strip()
        if not value:
            return [], "empty dependency token"
        if value.lower() == "none":
            return [], "None cannot be combined with other dependencies"
        result.append(value)
    return result, None


def _canonical_token(token: str, target: str, names: set[str], repo: Path) -> tuple[str | None, str | None]:
    """Return canonical task path and an issue kind (waiting or blocked)."""
    value = token
    if value.endswith("/spec.md"):
        value = value[:-len("/spec.md")]
    if value.startswith("docs/changes/"):
        value = value[len("docs/changes/"):]
    if value in names:
        return f"{target}/{value}", None
    if value == target or value == f"{target}/":
        return None, "blocked"
    try:
        helper.validate_target(value)
    except helper.TaskError:
        return None, "blocked"
    if value.startswith(f"{target}/"):
        suffix = value[len(target) + 1:]
        if suffix in names:
            return f"{target}/{suffix}", None
        return None, "waiting"
    # Resolve only explicitly referenced external tasks, never scan other tasks.
    if "/" in value:
        try:
            helper.spec_path(repo, value)
            return value, None
        except helper.TaskError:
            pass
    return None, "waiting"


def _load_state(path_value: str | None, repo: Path, names: set[str]) -> tuple[dict[str, list[str]], dict[str, dict[str, str]], dict[str, str]]:
    if path_value is None:
        return {}, {}, {}
    path = Path(path_value).expanduser().resolve()
    if helper.is_relative_to(path, repo):
        raise QueueError("state file must live outside the integration checkout")
    if not path.is_file():
        raise QueueError(f"state file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        state = json.load(stream)
    if not isinstance(state, dict):
        raise QueueError("state must be a JSON object")
    allowed = {"dependencies", "outcomes"}
    unknown = sorted(set(state) - allowed)
    if unknown:
        raise QueueError("unsupported state fields: " + ", ".join(unknown))
    dependencies = state.get("dependencies", {})
    outcomes = state.get("outcomes", {})
    if not isinstance(dependencies, dict) or not isinstance(outcomes, dict):
        raise QueueError("state dependencies and outcomes must be objects")
    dep_overrides: dict[str, list[str]] = {}
    outcome_values: dict[str, dict[str, str]] = {}
    state_errors: dict[str, str] = {}
    for key, values in dependencies.items():
        if not isinstance(key, str) or key not in names:
            raise QueueError(f"state dependency key is not a direct child: {key!r}")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise QueueError(f"state.dependencies[{key!r}] must be a list of sibling names")
        invalid = [value for value in values if value not in names]
        if invalid:
            state_errors[key] = "state dependency references non-sibling task(s): " + ", ".join(invalid)
        else:
            dep_overrides[key] = list(dict.fromkeys(values))
    for key, value in outcomes.items():
        if not isinstance(key, str) or key not in names:
            raise QueueError(f"state outcome key is not a direct child: {key!r}")
        if not isinstance(value, dict):
            raise QueueError(f"state.outcomes[{key!r}] must be an object")
        status = value.get("status")
        reason = value.get("reason")
        if not isinstance(status, str) or status not in {"blocked", "failed", "ready", "running"} or not isinstance(reason, str) or not reason.strip():
            state_errors[key] = "state outcome requires status blocked, failed, ready, or running and a nonempty reason"
        else:
            outcome_values[key] = {"status": status, "reason": reason.strip()}
    return dep_overrides, outcome_values, state_errors


def _status_for(repo: Path, task: str, cache: dict[str, tuple[str | None, str | None]]) -> tuple[str | None, str | None]:
    if task in cache:
        return cache[task]
    try:
        annotation = helper.annotation_path(repo, task)
        parsed = helper.parse_annotation(annotation)
        cache[task] = (parsed.get("status"), None)
    except (helper.TaskError, OSError, UnicodeError) as exc:
        cache[task] = (None, str(exc))
    return cache[task]


def _cycles(edges: dict[str, list[str]]) -> set[str]:
    visiting: list[str] = []
    visited: set[str] = set()
    cyclic: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cyclic.update(visiting[visiting.index(node):])
            return
        if node in visited:
            return
        visiting.append(node)
        for dep in edges.get(node, []):
            if dep in edges:
                visit(dep)
        visiting.pop()
        visited.add(node)

    for node in edges:
        visit(node)
    return cyclic


def queue(repo: Path, target: str, state_path: str | None) -> dict[str, Any]:
    result = scan(repo, target)
    tasks = result["tasks"]
    names = {task["name"] for task in tasks}
    dep_overrides, outcomes, state_errors = _load_state(state_path, repo, names)
    task_by_name = {task["name"]: task for task in tasks}
    task_by_path = {task["task_path"]: task for task in tasks}
    edges: dict[str, list[str]] = {}
    static_blocked: dict[str, list[str]] = {name: [] for name in names}
    unresolved: dict[str, list[str]] = {name: [] for name in names}
    dependency_paths: dict[str, list[str]] = {name: [] for name in names}

    for task in tasks:
        name = task["name"]
        if task.get("error"):
            static_blocked[name].append("invalid metadata: " + task["error"])
        if task.get("nested"):
            static_blocked[name].append("unsupported nested-container task; run its executable leaves explicitly")
        if name in state_errors:
            static_blocked[name].append(state_errors[name])
        if name in dep_overrides:
            canonical = [f"{result['target']}/{dep}" for dep in dep_overrides[name]]
            dependency_paths[name].extend(canonical)
        else:
            tokens, parse_error = _parse_tokens(task.get("blocked_by"))
            if parse_error:
                static_blocked[name].append("invalid metadata: " + parse_error)
                tokens = []
            for token in tokens:
                canonical, issue = _canonical_token(token, result["target"], names, repo)
                if canonical is None:
                    if issue == "blocked":
                        static_blocked[name].append(f"invalid dependency reference '{token}'")
                    else:
                        unresolved[name].append(token)
                else:
                    dependency_paths[name].append(canonical)
        dependency_paths[name] = list(dict.fromkeys(dependency_paths[name]))
        edges[name] = [
            task_by_path[path]["name"] for path in dependency_paths[name]
            if path in task_by_path and task_by_path[path]["status"] != "done"
        ] if task["status"] != "done" else []

    cyclic = _cycles(edges)

    status_cache: dict[str, tuple[str | None, str | None]] = {
        task["task_path"]: (task.get("status"), task.get("error")) for task in tasks
    }
    already_done = sorted(name for name, task in task_by_name.items() if task.get("status") == "done")
    waiting: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    candidates: list[str] = []
    for name in sorted(names):
        if name in already_done:
            continue
        if static_blocked[name]:
            blocked.append({"name": name, "reason": "; ".join(static_blocked[name])})
            continue
        outcome = outcomes.get(name)
        if outcome:
            if outcome["status"] in {"blocked", "failed"}:
                blocked.append({"name": name, "reason": outcome["reason"]})
            else:
                waiting.append({
                    "name": name,
                    "reason": f"outcome already recorded as {outcome['status']}: {outcome['reason']}",
                })
            continue
        if name in cyclic:
            waiting.append({"name": name, "reason": "dependency cycle involving: " + ", ".join(sorted(cyclic))})
            continue
        if unresolved[name]:
            waiting.append({"name": name, "reason": "unresolved dependency reference(s): " + ", ".join(unresolved[name])})
            continue
        dependency_waits: list[str] = []
        for dependency in dependency_paths[name]:
            status, error = _status_for(repo, dependency, status_cache)
            if status != "done":
                detail = f"{dependency} is {status or 'not done'}"
                if error:
                    detail += f" ({error})"
                dependency_waits.append(detail)
        if dependency_waits:
            waiting.append({"name": name, "reason": "waiting for " + "; ".join(dependency_waits)})
        else:
            candidates.append(name)

    ready = candidates
    finished = len(already_done) == len(tasks) and not blocked and not waiting
    result.update({
        "ready": ready,
        "waiting": waiting,
        "blocked": blocked,
        "already_done": already_done,
        "finished": finished,
    })
    return result


def prepare(repo: Path, target: str, state_path: str | None) -> dict[str, Any]:
    selected = queue(repo, target, state_path)
    integration_branch, integration_head = helper.validate_control(repo)
    prepared: list[dict[str, Any]] = []
    blocked = list(selected["blocked"])
    for name in selected["ready"]:
        task = f"{selected['target']}/{name}"
        try:
            prepared.append({"name": name, **helper.prepare_task(repo, task, task, integration_branch, integration_head)})
        except (helper.TaskError, OSError, ValueError) as exc:
            blocked.append({"name": name, "reason": f"prepare failed: {exc}"})
    selected.update({
        "integration_branch": integration_branch,
        "integration_head": integration_head,
        "prepared": prepared,
        "blocked": blocked,
    })
    return selected

def control(repo: Path) -> dict[str, str]:
    integration_branch, integration_head = helper.validate_control(repo)
    return {
        "repo_root": str(repo),
        "integration_branch": integration_branch,
        "integration_head": integration_head,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", default=".", help="Integration checkout path")
    result.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    commands = result.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan", help="Scan direct child specifications")
    scan_parser.add_argument("target")
    for command in ("queue", "prepare"):
        sub = commands.add_parser(command, help=f"{command.title()} direct child specifications")
        sub.add_argument("target")
        sub.add_argument("--state", help="Caller-owned temporary state JSON")
    commands.add_parser("control", help="Refresh integration checkout control state")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        repo = helper.root_for(Path(args.repo))
        if args.command == "scan":
            output = scan(repo, args.target)
        elif args.command == "queue":
            output = queue(repo, args.target, args.state)
        elif args.command == "prepare":
            output = prepare(repo, args.target, args.state)
        else:
            output = control(repo)
        print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except (QueueError, helper.TaskError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"spec-run-all: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

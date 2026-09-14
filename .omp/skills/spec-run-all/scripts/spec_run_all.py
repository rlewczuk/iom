#!/usr/bin/env python3
"""Scan, queue, and prepare direct spec-run-task children."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# spec-run-task owns Git/worktree mechanics and loads task_ctl's public API.
sys.dont_write_bytecode = True
_HELPER_DIR = Path(__file__).resolve().parents[2] / "spec-run-task" / "scripts"
sys.path.insert(0, str(_HELPER_DIR))
import spec_run_task as helper


class QueueError(RuntimeError):
    pass


def _target(target: str) -> tuple[str, str]:
    target = helper.validate_target(target)
    return target, helper.canonical_task(target)


def scan(repo: Path, target: str) -> dict[str, Any]:
    target, target_id = _target(target)
    direct = helper.task_ctl_scan_tasks(repo, target_id)
    tasks: list[dict[str, Any]] = []
    for config in direct:
        task_id = config["task-id"]
        task_path = helper.relative_task_id(task_id)
        if "error" in config:
            tasks.append({
                "name": Path(task_id).name,
                "task_id": task_id,
                "task_path": task_path,
                "type": None,
                "status": None,
                "blocked_by": [],
                "error": config["error"],
            })
            continue
        task_dir = Path(helper.task_ctl_task_dir(repo, task_id)).resolve()
        spec = task_dir / "spec.md"
        evidence = task_dir / "task.md"
        record: dict[str, Any] = {
            "name": task_dir.name,
            "task_id": task_id,
            "task_path": task_path,
            "spec_path": str(spec),
            "control_path": str(task_dir / "task.yml"),
            "annotation_path": str(evidence),
            "annotation_exists": evidence.is_file(),
            "type": config["type"],
            "status": config["status"],
            "order": config.get("order"),
            "source": config.get("source"),
            "priority": config.get("priority"),
            "blocked_by": config.get("blocked-by", []),
        }
        if not spec.is_file():
            record["error"] = f"specification not found: {spec}"
        if config["type"] != "impl" or helper.task_ctl_scan_tasks(repo, task_id):
            record["nested"] = True
        tasks.append(record)
    return {
        "repo_root": str(repo),
        "target": target,
        "target_id": target_id,
        "tasks": tasks,
    }


def _load_state(
    path_value: str | None,
    repo: Path,
    tasks: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]], dict[str, str]]:
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
    unknown = sorted(set(state) - {"dependencies", "outcomes"})
    if unknown:
        raise QueueError("unsupported state fields: " + ", ".join(unknown))
    dependencies = state.get("dependencies", {})
    outcomes = state.get("outcomes", {})
    if not isinstance(dependencies, dict) or not isinstance(outcomes, dict):
        raise QueueError("state dependencies and outcomes must be objects")

    names = {task["name"] for task in tasks}
    additions: dict[str, list[str]] = {}
    outcome_values: dict[str, dict[str, str]] = {}
    errors: dict[str, str] = {}
    for key, values in dependencies.items():
        if not isinstance(key, str) or key not in names:
            raise QueueError(f"state dependency key is not a direct child: {key!r}")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            errors[key] = f"state.dependencies[{key!r}] must be a list of canonical task IDs"
            continue
        unique = list(dict.fromkeys(values))
        invalid = []
        for value in unique:
            if value in {".cswd/tasks", "docs/changes"}:
                invalid.append(value)
                continue
            try:
                helper.task_ctl_task_dir(repo, value)
            except helper.TaskCtlError:
                invalid.append(value)
        if invalid:
            errors[key] = "state dependency is not a canonical task ID: " + ", ".join(invalid)
        else:
            additions[key] = unique
    for key, value in outcomes.items():
        if not isinstance(key, str) or key not in names:
            raise QueueError(f"state outcome key is not a direct child: {key!r}")
        if not isinstance(value, dict):
            raise QueueError(f"state.outcomes[{key!r}] must be an object")
        status = value.get("status")
        reason = value.get("reason")
        if (
            not isinstance(status, str)
            or status not in {"blocked", "failed", "ready", "running"}
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            errors[key] = (
                "state outcome requires status blocked, failed, ready, or running "
                "and a nonempty reason"
            )
        else:
            outcome_values[key] = {"status": status, "reason": reason.strip()}
    return additions, outcome_values, errors


def _cycles(edges: dict[str, list[str]]) -> set[str]:
    visiting: list[str] = []
    visited: set[str] = set()
    cyclic: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cyclic.update(visiting[visiting.index(node) :])
            return
        if node in visited:
            return
        visiting.append(node)
        for dependency in edges.get(node, []):
            if dependency in edges:
                visit(dependency)
        visiting.pop()
        visited.add(node)

    for node in edges:
        visit(node)
    return cyclic


def queue(repo: Path, target: str, state_path: str | None) -> dict[str, Any]:
    result = scan(repo, target)
    tasks = result["tasks"]
    additions, outcomes, state_errors = _load_state(state_path, repo, tasks)
    by_id = {task["task_id"]: task for task in tasks}
    dependencies: dict[str, list[str]] = {}
    edges: dict[str, list[str]] = {}
    for task in tasks:
        name = task["name"]
        configured = [record["task-id"] for record in task["blocked_by"]]
        dependencies[name] = list(dict.fromkeys([*configured, *additions.get(name, [])]))
        edges[name] = [
            by_id[dependency]["name"]
            for dependency in dependencies[name]
            if dependency in by_id and by_id[dependency]["status"] != "done"
        ] if task["status"] != "done" else []
    cyclic = _cycles(edges)

    already_done: list[str] = []
    waiting: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    ready: list[str] = []
    for task in tasks:
        name = task["name"]
        status = task["status"]
        if status == "done":
            already_done.append(name)
            continue
        static_errors: list[str] = []
        if task.get("error"):
            static_errors.append(task["error"])
        if task.get("nested"):
            static_errors.append("unsupported nested-container task; run its implementation leaves explicitly")
        if name in state_errors:
            static_errors.append(state_errors[name])
        if static_errors:
            blocked.append({"name": name, "reason": "; ".join(static_errors)})
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
            cycle = [record["name"] for record in tasks if record["name"] in cyclic]
            waiting.append({"name": name, "reason": "dependency cycle involving: " + ", ".join(cycle)})
            continue
        dependency_waits: list[str] = []
        for dependency_id in dependencies[name]:
            try:
                dependency_status = helper.task_ctl_get_task(repo, dependency_id)["status"]
                if dependency_status != "done":
                    dependency_waits.append(f"{dependency_id} is {dependency_status}")
            except helper.TaskCtlError as exc:
                dependency_waits.append(f"{dependency_id} is missing ({exc})")
        if dependency_waits:
            waiting.append({"name": name, "reason": "waiting for " + "; ".join(dependency_waits)})
        elif status in {"new", "critic", "planned", "ready"}:
            ready.append(name)
        else:
            waiting.append({"name": name, "reason": f"lifecycle status is {status}; awaiting integration"})

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
    by_name = {task["name"]: task for task in selected["tasks"]}
    prepared: list[dict[str, Any]] = []
    blocked = list(selected["blocked"])
    for name in selected["ready"]:
        task = by_name[name]["task_path"]
        try:
            prepared.append({
                "name": name,
                **helper.prepare_task(repo, task, task, integration_branch, integration_head),
            })
        except (helper.TaskError, helper.TaskCtlError, OSError, ValueError) as exc:
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
    scan_parser = commands.add_parser("scan", help="Scan direct child task controls")
    scan_parser.add_argument("target")
    for command in ("queue", "prepare"):
        sub = commands.add_parser(command, help=f"{command.title()} direct child implementation tasks")
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
    except (
        QueueError,
        helper.TaskError,
        helper.TaskCtlError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"spec-run-all: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

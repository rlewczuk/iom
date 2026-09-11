#!/usr/bin/env python3
"""Check the effective OMP dispatch configuration used by spec-run-all."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

AGENTS = (
    ("spec-run-all-implementer", "@implementer", ("spec-run-debug",)),
    ("spec-run-debug", "@slow", ()),
)
THINKING_LEVELS = frozenset(("off", "minimal", "low", "medium", "high", "xhigh", "max"))
THINKING_SUFFIX = re.compile(r":([A-Za-z][A-Za-z0-9_-]*)$")


class Preflight:
    def __init__(self, repo: Path, omp: str) -> None:
        self.repo = repo
        self.omp = omp
        self.errors: list[str] = []
        self.config: dict[str, Any] | None = None
        self.models: list[dict[str, Any]] = []
        self.settings: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.overrides: dict[str, str] = {}
        self.agent_settings: dict[str, Any] = {}
        self.profile_data: dict[str, dict[str, Any]] = {}

    def run_command(self, args: list[str], label: str) -> Any | None:
        command = [self.omp, *args]
        try:
            result = subprocess.run(
                command,
                cwd=self.repo,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            self.errors.append(f"{label} command failed: {' '.join(command)}: {exc}")
            return None
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().replace("\n", " ")
            if not detail:
                detail = f"exit status {result.returncode}"
            self.errors.append(f"{label} command failed: {detail}")
            return None
        try:
            return json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            self.errors.append(f"{label} returned invalid JSON: {exc}")
            return None

    def load_effective_data(self) -> None:
        config = self.run_command(["config", "list", "--json"], "omp config list --json")
        models = self.run_command(["models", "--json"], "omp models --json")
        if isinstance(config, dict):
            self.config = config
        elif config is not None:
            self.errors.append("omp config list --json returned a JSON value, not an object")
        if isinstance(models, dict):
            raw_models = models.get("models")
            if not isinstance(raw_models, list):
                self.errors.append("omp models --json is missing a models array")
            else:
                for index, model in enumerate(raw_models):
                    if not isinstance(model, dict):
                        self.errors.append(f"omp models --json models[{index}] is not an object")
                        continue
                    missing = [key for key in ("selector", "provider", "id", "thinking") if key not in model]
                    if missing:
                        self.errors.append(
                            f"omp models --json models[{index}] is missing required field(s): {', '.join(missing)}"
                        )
                        continue
                    if not all(isinstance(model[key], str) and model[key] for key in ("selector", "provider", "id")):
                        self.errors.append(f"omp models --json models[{index}] has invalid identity fields")
                        continue
                    self.models.append(model)
        elif models is not None:
            self.errors.append("omp models --json returned a JSON value, not an object")

    @staticmethod
    def _setting(config: dict[str, Any], path: str, errors: list[str]) -> Any:
        if path not in config:
            errors.append(f"effective setting {path!r} is missing")
            return None
        record = config[path]
        if not isinstance(record, dict) or "value" not in record:
            errors.append(f"effective setting {path!r} is malformed (expected an object with value)")
            return None
        return record["value"]

    def parse_settings(self) -> None:
        if self.config is None:
            return
        paths = (
            "modelRoles",
            "task.agentModelOverrides",
            "task.agentAdvisor",
            "task.agentPrewalk",
            "task.disabledAgents",
            "task.maxRecursionDepth",
        )
        for path in paths:
            self.settings[path] = self._setting(self.config, path, self.errors)

        roles = self.settings.get("modelRoles")
        if isinstance(roles, dict):
            for key, value in roles.items():
                if not isinstance(key, str) or not key:
                    self.errors.append("effective setting modelRoles contains an invalid role name")
                elif not isinstance(value, str) or not value.strip():
                    self.errors.append(f"effective setting modelRoles[{key!r}] must be a nonempty string")
                else:
                    self.roles[key.lstrip("@")] = value.strip()
        elif roles is not None:
            self.errors.append("effective setting modelRoles must be an object")

        overrides = self.settings.get("task.agentModelOverrides")
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
                    self.errors.append("effective setting task.agentModelOverrides must map agent names to strings")
                else:
                    self.overrides[key] = value.strip()
        elif overrides is not None:
            self.errors.append("effective setting task.agentModelOverrides must be an object")

        disabled = self.settings.get("task.disabledAgents")
        if isinstance(disabled, list) and all(isinstance(item, str) and item for item in disabled):
            for agent, _, _ in AGENTS:
                if agent in disabled:
                    self.errors.append(f"required agent {agent!r} is disabled by task.disabledAgents")
        elif disabled is not None:
            self.errors.append("effective setting task.disabledAgents must be an array of agent names")

        depth = self.settings.get("task.maxRecursionDepth")
        if isinstance(depth, bool) or not isinstance(depth, int):
            if depth is not None:
                self.errors.append("effective setting task.maxRecursionDepth must be an integer")
        elif depth >= 0 and depth < 2:
            self.errors.append("task.maxRecursionDepth must be at least 2 for root -> implementer -> debugger rescue")

        self._validate_agent_settings()

    @staticmethod
    def _agent_value(value: Any, agent: str) -> tuple[bool, Any]:
        if isinstance(value, dict):
            return agent in value, value.get(agent)
        return True, value

    def _validate_agent_settings(self) -> None:
        advisor = self.settings.get("task.agentAdvisor")
        prewalk = self.settings.get("task.agentPrewalk")
        if advisor is not None and not isinstance(advisor, (dict, bool, str)):
            self.errors.append("effective setting task.agentAdvisor must be an object, boolean, or string")
        if prewalk is not None and not isinstance(prewalk, (dict, bool, str)):
            self.errors.append("effective setting task.agentPrewalk must be an object, boolean, or string")
        self.agent_settings = {"advisor": advisor, "prewalk": prewalk}
        for agent, _, _ in AGENTS:
            advisor_set, advisor_value = self._agent_value(advisor, agent)
            prewalk_set, prewalk_value = self._agent_value(prewalk, agent)
            if advisor_set and self._enabled(advisor_value):
                self.errors.append(
                    f"task.agentAdvisor enables an advisor for required agent {agent!r}; fixed dispatch requires it off"
                )
            if prewalk_set and self._enabled(prewalk_value):
                self.errors.append(
                    f"task.agentPrewalk enables prewalk for required agent {agent!r}; fixed dispatch requires it off"
                )

    @staticmethod
    def _enabled(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in ("", "off", "false", "0", "none", "disabled")
        return bool(value)

    @staticmethod
    def _parse_frontmatter(path: Path) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing YAML frontmatter")
        try:
            end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
        except StopIteration as exc:
            raise ValueError("unterminated YAML frontmatter") from exc
        fields: dict[str, Any] = {}
        index = 1
        while index < end:
            line = lines[index]
            if not line.strip() or line.lstrip().startswith("#"):
                index += 1
                continue
            if ":" not in line or line[0].isspace():
                raise ValueError(f"unsupported frontmatter line {line!r}")
            key, raw = line.split(":", 1)
            key = key.strip()
            raw = raw.strip()
            if not key or key in fields:
                raise ValueError(f"invalid or duplicate frontmatter field {key!r}")
            if raw:
                if raw.startswith("[") and raw.endswith("]"):
                    body = raw[1:-1].strip()
                    value = [] if not body else [item.strip().strip("\"'") for item in body.split(",")]
                else:
                    try:
                        value = ast.literal_eval(raw)
                    except (SyntaxError, ValueError):
                        value = raw.strip('"\'')
                fields[key] = value
                index += 1
                continue
            values: list[str] = []
            cursor = index + 1
            while cursor < end and (not lines[cursor].strip() or lines[cursor].startswith("  ")):
                child = lines[cursor].strip()
                if child:
                    if not child.startswith("-"):
                        raise ValueError(f"unsupported nested frontmatter field {child!r}")
                    values.append(child[1:].strip().strip('"\''))
                cursor += 1
            fields[key] = values
            index = cursor
        return fields

    def load_profiles(self) -> None:
        profile_dir = self.repo / ".omp" / "agents"
        for agent, expected_model, expected_spawns in AGENTS:
            path = profile_dir / f"{agent}.md"
            if not path.is_file():
                self.errors.append(f"required profile is missing: {path}")
                continue
            try:
                fields = self._parse_frontmatter(path)
            except (OSError, ValueError) as exc:
                self.errors.append(f"profile {agent!r} is malformed: {exc}")
                continue
            self.profile_data[agent] = fields
            if fields.get("name") != agent:
                self.errors.append(f"profile {agent!r} must declare name {agent!r}")
            description = fields.get("description")
            if not isinstance(description, str) or not description.strip():
                self.errors.append(f"profile {agent!r} must declare a nonempty description")
            if fields.get("model") != expected_model:
                self.errors.append(
                    f"profile {agent!r} must request exact model alias {expected_model!r}; got {fields.get('model')!r}"
                )
            spawns = fields.get("spawns")
            if not isinstance(spawns, list) or any(not isinstance(item, str) for item in spawns):
                self.errors.append(f"profile {agent!r} must declare a spawns list")
            elif tuple(spawns) != expected_spawns:
                self.errors.append(f"profile {agent!r} must spawn exactly {list(expected_spawns)!r}; got {spawns!r}")
            for setting in ("advisor", "prewalk"):
                if setting in fields and self._enabled(fields[setting]):
                    self.errors.append(f"profile {agent!r} enables {setting}; fixed dispatch requires it off")

    @staticmethod
    def _split_suffix(value: str) -> tuple[str, str | None, str | None]:
        match = THINKING_SUFFIX.search(value)
        if not match:
            return value, None, None
        suffix = match.group(1).lower()
        if suffix not in THINKING_LEVELS:
            return value[: match.start()], suffix, "unsupported thinking suffix"
        return value[: match.start()], suffix, None

    def resolve(self, requested: str, agent: str) -> tuple[str | None, str | None, str | None]:
        seen: list[str] = []

        def visit(value: str, inherited: str | None = None) -> tuple[str | None, str | None, str | None]:
            base, suffix, issue = self._split_suffix(value.strip())
            if issue:
                return None, suffix, f"agent {agent!r} uses unsupported thinking suffix :{suffix}"
            # An explicit suffix on the original request or referring role wins
            # over a suffix carried by the role it aliases.
            effective_suffix = inherited or suffix
            if base == "*":
                base = "@default"
            if not base.startswith("@"):
                return base, effective_suffix, None
            role = base[1:]
            if not role:
                return None, effective_suffix, f"agent {agent!r} has an empty model role alias"
            if role in seen:
                chain = " -> ".join("@" + item for item in (*seen, role))
                return None, effective_suffix, f"agent {agent!r} has a model role cycle: {chain}"
            if role not in self.roles:
                return None, effective_suffix, f"agent {agent!r} references missing model role @{role}"
            seen.append(role)
            resolved = visit(self.roles[role], effective_suffix)
            seen.pop()
            return resolved

        return visit(requested)

    @staticmethod
    def _thinking_levels(model: dict[str, Any]) -> set[str] | None:
        value = model.get("thinking")
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return {item.lower() for item in value}
        if isinstance(value, dict):
            for key in ("efforts", "levels", "supported"):
                candidate = value.get(key)
                if isinstance(candidate, list) and all(isinstance(item, str) for item in candidate):
                    return {item.lower() for item in candidate}
            default = value.get("defaultLevel")
            if isinstance(default, str):
                return {default.lower()}
            return None
        if isinstance(value, str):
            return {value.lower()}
        if value is False or value is None:
            return set()
        return None

    def check_agent_models(self) -> dict[str, Any]:
        agents: dict[str, Any] = {}
        for agent, expected_model, _ in AGENTS:
            profile = self.profile_data.get(agent, {})
            profile_model = profile.get("model")
            requested = self.overrides.get(agent, profile_model)
            detail: dict[str, Any] = {"requested": requested, "available": False}
            if not isinstance(requested, str) or not requested.strip():
                self.errors.append(f"agent {agent!r} has no effective model request")
                agents[agent] = detail
                continue
            if requested != expected_model:
                self.errors.append(
                    f"agent {agent!r} effective model request is {requested!r}; exact contract requires {expected_model!r}"
                )
            resolved, thinking, issue = self.resolve(requested, agent)
            detail["role"] = requested[1:] if requested.startswith("@") else None
            if issue:
                self.errors.append(issue)
            if resolved is not None:
                detail["resolved"] = resolved
                if thinking is not None:
                    detail["thinking"] = thinking
                matches = [
                    model
                    for model in self.models
                    if model.get("selector") == resolved
                    or (model.get("provider") + "/" + model.get("id")) == resolved
                ]
                if thinking is not None and thinking != "off":
                    matches = [model for model in matches if thinking in (self._thinking_levels(model) or set())]
                detail["available"] = bool(matches)
                detail["available_models"] = [
                    {
                        "selector": model["selector"],
                        "provider": model["provider"],
                        "id": model["id"],
                        "thinking": model["thinking"],
                    }
                    for model in matches
                ]
                if not matches:
                    suffix = f":{thinking}" if thinking else ""
                    self.errors.append(f"agent {agent!r} resolved model {resolved!r}{suffix}, which is unavailable")
            agents[agent] = detail
        return agents

    def result(self) -> dict[str, Any]:
        self.load_effective_data()
        self.parse_settings()
        self.load_profiles()
        agents = self.check_agent_models()
        return {
            "ok": not self.errors,
            "settings": self.settings,
            "roles": {key: self.roles[key] for key in sorted(self.roles)},
            "agents": agents,
            "errors": sorted(dict.fromkeys(self.errors)),
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", required=True, type=Path, help="repository whose project OMP configuration is checked")
    result.add_argument("--omp", default="omp", help="omp executable (or test double)")
    result.add_argument("--pretty", action="store_true", help="indent the JSON result")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.repo.is_dir():
        payload = {"ok": False, "settings": {}, "roles": {}, "agents": {}, "errors": [f"repository is not a directory: {args.repo}"]}
    else:
        payload = Preflight(args.repo.resolve(), args.omp).result()
    json.dump(payload, sys.stdout, indent=2 if args.pretty else None, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

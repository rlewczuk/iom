#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=
WORKSPACE_ROOT=
PRIMARY_WORKTREE_ROOT=
WORKSPACE_KIND=

fail() {
  printf 'remote-development: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

resolve_workspace() {
  require_cmd git

  local requested="${REMOTE_DEV_WORKSPACE:-$PWD}"
  [[ -d "$requested" ]] || fail "workspace path is not a directory: $requested"

  local root git_dir common_git_dir
  root="$(git -C "$requested" rev-parse --show-toplevel 2>/dev/null)" ||
    fail "workspace path is not inside a Git worktree: $requested"
  WORKSPACE_ROOT="$(cd -- "$root" && pwd -P)"

  git_dir="$(git -C "$WORKSPACE_ROOT" rev-parse --git-dir)"
  common_git_dir="$(git -C "$WORKSPACE_ROOT" rev-parse --git-common-dir)"
  [[ "$git_dir" == /* ]] || git_dir="$WORKSPACE_ROOT/$git_dir"
  [[ "$common_git_dir" == /* ]] || common_git_dir="$WORKSPACE_ROOT/$common_git_dir"
  git_dir="$(cd -- "$git_dir" && pwd -P)"
  common_git_dir="$(cd -- "$common_git_dir" && pwd -P)"
  PRIMARY_WORKTREE_ROOT="$(cd -- "$common_git_dir/.." && pwd -P)"

  if [[ "$git_dir" == "$common_git_dir" ]]; then
    WORKSPACE_KIND=checkout
  else
    WORKSPACE_KIND=worktree
  fi

  if [[ -n "${REMOTE_DEV_CONFIG:-}" ]]; then
    CONFIG_FILE="$REMOTE_DEV_CONFIG"
  elif [[ -f "$WORKSPACE_ROOT/.remote-hosts.conf" ]]; then
    CONFIG_FILE="$WORKSPACE_ROOT/.remote-hosts.conf"
  elif [[ "$WORKSPACE_KIND" == worktree &&
          -f "$PRIMARY_WORKTREE_ROOT/.remote-hosts.conf" ]]; then
    CONFIG_FILE="$PRIMARY_WORKTREE_ROOT/.remote-hosts.conf"
  else
    CONFIG_FILE="$WORKSPACE_ROOT/.remote-hosts.conf"
  fi
}

validate_task_id() {
  local task_id="$1"
  [[ "$task_id" =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid task id: $task_id"
}

load_profile() {
  local profile="$1"
  [[ -f "$CONFIG_FILE" ]] || fail "config not found: $CONFIG_FILE"

  local line
  line="$(awk -F'|' -v p="$profile" '
    /^[[:space:]]*#/ { next }
    NF >= 3 && $1 == p { print; exit }
  ' "$CONFIG_FILE")"

  [[ -n "$line" ]] || fail "profile not found in $CONFIG_FILE: $profile"

  IFS='|' read -r PROFILE SSH_HOST REMOTE_BASE REMOTE_SETUP <<<"$line"
  [[ -n "$SSH_HOST" ]] || fail "missing ssh host for profile: $profile"
  [[ -n "$REMOTE_BASE" ]] || fail "missing remote base dir for profile: $profile"
}

remote_dir_for() {
  local task_id="$1"
  printf '%s/%s\n' "${REMOTE_BASE%/}" "$task_id"
}

quote_single() {
  # Print a shell-safe single-quoted string.
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\''/g")"
}

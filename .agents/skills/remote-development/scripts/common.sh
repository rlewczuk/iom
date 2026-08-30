#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${REMOTE_DEV_CONFIG:-.remote-hosts.conf}"

fail() {
  printf 'remote-development: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
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

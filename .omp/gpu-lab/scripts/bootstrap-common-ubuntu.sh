#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(uname -m)" != x86_64 ]]; then
  echo "This package supports amd64/x86_64 hosts only." >&2
  exit 1
fi
. /etc/os-release
case "${VERSION_ID:-}" in
  24.04|26.04) ;;
  *) echo "Expected Ubuntu 24.04 or 26.04; found ${PRETTY_NAME:-unknown}." >&2; exit 1 ;;
esac

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential ca-certificates curl wget gnupg lsb-release software-properties-common \
  git openssh-client openssh-server rsync jq \
  cmake ninja-build ccache pkg-config \
  clang clangd clang-format clang-tidy lld lldb \
  gdb gdbserver valgrind \
  python3 python3-pip python3-venv \
  shellcheck tmux util-linux pciutils hwloc numactl

sudo systemctl enable --now ssh
target_user="${SUDO_USER:-$USER}"
target_home="$(getent passwd "$target_user" | cut -d: -f6)"
sudo -u "$target_user" mkdir -p "$target_home/.cache/ccache"

printf '\nCommon development tools installed on %s.\n' "${PRETTY_NAME}"
printf 'Versions:\n'
cmake --version | head -1
ninja --version
clang --version | head -1
clangd --version | head -1
gdb --version | head -1

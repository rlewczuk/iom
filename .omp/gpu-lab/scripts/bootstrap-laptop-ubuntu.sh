#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/bootstrap-common-ubuntu.sh"

cat <<'EOF'

Laptop bootstrap complete.

Still required:
  1. Install Oh My Pi and verify: omp --version
  2. Install uv only when enabling ssh-mcp:
       curl -LsSf https://astral.sh/uv/install.sh | sh
  3. Put SSH aliases and pinned host keys in ~/.ssh/config and ~/.ssh/known_hosts.
  4. Install this package with ./install.sh --project /path/to/project.

A local CUDA/ROCm SDK is optional. Remote SSH-stdio DAP adapters run the
backend debugger on the GPU host, so the laptop needs OMP, OpenSSH, Clang tools,
and the package, not necessarily a GPU SDK.
EOF

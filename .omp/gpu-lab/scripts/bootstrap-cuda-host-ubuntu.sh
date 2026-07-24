#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_toolkit=false
cuda_package="${CUDA_APT_PACKAGE:-cuda-toolkit-13-3}"

usage() {
  cat <<'USAGE'
Usage: bootstrap-cuda-host-ubuntu.sh [--install-toolkit] [--package APT_PACKAGE]

Installs common tools. With --install-toolkit, installs the pinned CUDA 13.3
Toolkit package by default. Pass --package cuda-toolkit to track the repository
meta-package, or another exact package documented by NVIDIA.

This script deliberately does not install or replace the NVIDIA kernel driver.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-toolkit) install_toolkit=true ;;
    --package) cuda_package="${2:?missing package name}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ "$cuda_package" =~ ^[A-Za-z0-9.+-]+$ ]] || {
  echo "Invalid apt package name: $cuda_package" >&2
  exit 2
}

"$SCRIPT_DIR/bootstrap-common-ubuntu.sh"
. /etc/os-release
case "$VERSION_ID" in
  24.04) distro=ubuntu2404 ;;
  26.04) distro=ubuntu2604 ;;
  *) echo "Unsupported Ubuntu version: $VERSION_ID" >&2; exit 1 ;;
esac

if $install_toolkit; then
  temp="$(mktemp -d)"; trap 'rm -rf -- "$temp"' EXIT
  wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb" -O "$temp/cuda-keyring.deb"
  sudo dpkg -i "$temp/cuda-keyring.deb"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$cuda_package"
fi

missing=0
for command in nvidia-smi nvcc cuda-gdb compute-sanitizer; do
  if command -v "$command" >/dev/null 2>&1; then
    printf 'OK      %-20s %s\n' "$command" "$(command -v "$command")"
  else
    printf 'MISSING %-20s\n' "$command"
    missing=1
  fi
done
command -v cuda-gdbserver >/dev/null 2>&1 || echo "OPTIONAL cuda-gdbserver not found"

cat <<EOF2

CUDA host notes:
  * Requested toolkit package: ${cuda_package}
  * This script never installs or changes the NVIDIA driver implicitly.
  * Confirm the installed driver supports the selected CUDA toolkit and GPU.
  * Use a Debug CMake preset with host -g3 -O0 -fno-omit-frame-pointer and
    CUDA -g -G -O0 for device stepping. Never benchmark a -G build.
EOF2
exit "$missing"

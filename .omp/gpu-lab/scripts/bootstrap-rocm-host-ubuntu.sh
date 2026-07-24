#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install_rocm=false
confirm_supported=false
rocm_release="${ROCM_RELEASE:-7.14}"
rocm_package="${ROCM_PACKAGE:-}"

usage() {
  cat <<'USAGE'
Usage:
  bootstrap-rocm-host-ubuntu.sh [--install-rocm] [--confirm-supported]
                                [--rocm-release 7.14]
                                [--package APT_PACKAGE]

The default package is amdrocm<release>, for example amdrocm7.14.
Use --package for an exact architecture-specific meta-package documented by
AMD, for example amdrocm-core-sdk7.14-gfx110x.

Automatic ROCm installation requires --confirm-supported because support is
specific to the exact GPU, Ubuntu point release, kernel, driver, and firmware.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-rocm) install_rocm=true ;;
    --confirm-supported) confirm_supported=true ;;
    --rocm-release) rocm_release="${2:?missing ROCm release}"; shift ;;
    --package) rocm_package="${2:?missing package name}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "$rocm_release" =~ ^[0-9]+\.[0-9]+$ ]] || {
  echo "Invalid ROCm release: $rocm_release" >&2
  exit 2
}
[[ -z "$rocm_package" || "$rocm_package" =~ ^[A-Za-z0-9.+-]+$ ]] || {
  echo "Invalid apt package name: $rocm_package" >&2
  exit 2
}

"$SCRIPT_DIR/bootstrap-common-ubuntu.sh"
. /etc/os-release
case "$VERSION_ID" in
  24.04) repo_suite=ubuntu2404 ;;
  26.04) repo_suite=ubuntu2604 ;;
  *) echo "Unsupported Ubuntu version: $VERSION_ID" >&2; exit 1 ;;
esac

if $install_rocm && ! $confirm_supported; then
  cat >&2 <<EOF2
Refusing automatic ROCm installation without --confirm-supported.
Verify the exact GPU, Ubuntu point release, kernel, amdgpu driver, and firmware
in AMD's current compatibility matrix, then rerun with --confirm-supported.
EOF2
  exit 1
fi

if [[ -z "$rocm_package" ]]; then
  rocm_package="amdrocm${rocm_release}"
fi

if $install_rocm; then
  sudo install -d -m 0755 /etc/apt/keyrings
  temp="$(mktemp -d)"
  trap 'rm -rf -- "$temp"' EXIT

  wget -q https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg \
    -O "$temp/rocm.gpg"
  gpg --dearmor < "$temp/rocm.gpg" > "$temp/amdrocm.gpg"
  sudo install -m 0644 "$temp/amdrocm.gpg" /etc/apt/keyrings/amdrocm.gpg

  printf '%s\n' \
    "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/${repo_suite} stable main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null

  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$rocm_package"
fi

target_user="${SUDO_USER:-$USER}"
sudo usermod -a -G render,video "$target_user"

# The 7.14 Core SDK keeps compatibility symlinks under /opt/rocm for apt
# installs, but include common explicit paths for first-run verification.
export PATH="/opt/rocm/bin:/opt/rocm/core/bin:$PATH"

missing=0
for command in rocminfo hipcc rocgdb; do
  if command -v "$command" >/dev/null 2>&1; then
    printf 'OK      %-20s %s\n' "$command" "$(command -v "$command")"
  else
    printf 'MISSING %-20s\n' "$command"
    missing=1
  fi
done
command -v amd-smi >/dev/null 2>&1 || echo "OPTIONAL amd-smi not found"
command -v rocprofv3 >/dev/null 2>&1 || echo "OPTIONAL rocprofv3 not found"

cat <<EOF2

ROCm host notes:
  * Configured Ubuntu repository: ${repo_suite}; requested package: ${rocm_package}
  * Log out and back in after render/video group changes.
  * Confirm both groups with: id ${target_user}
  * Confirm that the exact GPU/OS/driver/firmware combination appears in AMD's
    current ROCm compatibility matrix. Repository availability alone does not
    imply support for every GPU on that Ubuntu release.
  * Prefer a Debug preset with -g3 -O0 -fno-omit-frame-pointer for diagnosis
    and a separate RelWithDebInfo/Release preset for benchmarks.
EOF2
exit "$missing"

#!/usr/bin/env bash
set -u

tools=(
  cmake ninja ctest
  clang clang++ clang-tidy
  gcc g++
  codeql include-what-you-use iwyu
  compute-sanitizer nsys ncu
  rocprofv3 rocprof-compute
  vulkaninfo spirv-val
)

printf '%-24s %s\n' TOOL STATUS
printf '%-24s %s\n' '------------------------' '------'
for tool in "${tools[@]}"; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%-24s available\n' "$tool"
  else
    printf '%-24s unavailable\n' "$tool"
  fi
done

**Status:** done

## Summary

Aligned the CPU factory's borrowed allocator lifetime wording with the CUDA, ROCm, and SYCL public headers, and amended the umbrella tensor-view specification to require allocator lifetime through device destruction and document deferred quarantine cleanup. Only `include/iom/cpu/device.hpp` and `docs/changes/0001-tensor-view/spec.md` were changed for the implementation.

## Verification

- Contract wording audit across the CPU, CUDA, ROCm, and SYCL headers plus umbrella specification — all required lifetime, exactly-once free, and quarantine-drain wording present; existing runtime quarantine paths remained unchanged.
- `git diff --stat && git diff --check` — exactly two implementation files changed; no whitespace errors.
- CPU-only CMake configure/build and `ctest --test-dir build/cc004-cpu --output-on-failure -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — both tests passed.
- Remote ROCm, CUDA, TTNN, and SYCL conformance builds/tests — each backend conformance test passed 1/1 on its configured remote host; remote mirrors were removed after verification.

**Status:** done

## Summary

Aligned the CPU factory's borrowed allocator lifetime wording with the CUDA, ROCm, and SYCL public headers, and amended the umbrella tensor-view specification to require allocator lifetime through device destruction and document deferred quarantine cleanup. Only `include/iom/cpu/device.hpp` and `docs/changes/0001-tensor-view/spec.md` were changed for the implementation.

## Verification

- Contract wording audit across the CPU, CUDA, ROCm, and SYCL headers plus umbrella specification — all required lifetime, exactly-once free, and quarantine-drain wording present; no task-owned runtime, test, or build files changed.
- `git diff main...HEAD --stat && git diff main...HEAD --check` — the task branch's implementation delta contains only `include/iom/cpu/device.hpp` and `docs/changes/0001-tensor-view/spec.md` plus this annotation; no whitespace errors.
- `cmake --build build/cc004-cpu -j2 --target iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/cc004-cpu --output-on-failure -R '^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — both CPU tests passed after merging the latest integration branch.
- Remote ROCm, CUDA, TTNN, and SYCL conformance builds/tests after the integration merge — `iom_rocm_conformance_tests`, `iom_cuda_conformance_tests`, `iom_ttnn_conformance_tests`, and `iom_sycl_conformance_tests` each built successfully and passed 1/1 on their configured remote hosts; all remote mirrors were removed afterward.

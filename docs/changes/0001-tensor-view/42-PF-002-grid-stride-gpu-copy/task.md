**Status:** done

## Summary

Replaced CUDA and ROCm per-plane device copies with one grid-stride kernel launch driven by queue-owned, bounded metadata slots. Added checked metadata sizing, geometric host/device mirror growth, slot leasing, backend-private fence resources, and event/slot lifetime cleanup. Removed the old `PlanePair`, `view_planes`, `plane_pairs`, and per-submit vector path while preserving shared host-transfer behavior, fault injection, transactional failures, no-op handling, and public APIs.

## Verification

- `cmake -S . -B build-pf002-cpu -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build-pf002-cpu -j && ctest --test-dir build-pf002-cpu --output-on-failure -E iom_cpu_bench` — CPU/common build and 3/3 selected tests passed.
- CUDA remote configure/build and `flock /tmp/agent-gpu0.lock ctest --test-dir build-pf002-cuda --output-on-failure -R "iom_cuda_(smoke|conformance)_tests|iom_backend_coexistence_tests|iom_tests|iom_backend_conformance_cpu_tests"` — build passed and 5/5 tests passed.
- ROCm remote configure/build and `flock /tmp/agent-gpu0.lock ctest --test-dir build-pf002-rocm --output-on-failure -R "iom_rocm_(smoke|conformance)_tests|iom_backend_coexistence_tests|iom_tests|iom_backend_conformance_cpu_tests"` — build passed and 5/5 tests passed.
- `compute-sanitizer --tool memcheck --error-exitcode 1 ./build-pf002-cuda/test/iom_cuda_conformance_tests` on `bv1` — 11/11 test cases and 39,859/39,859 assertions passed; sanitizer reported 0 errors.
- Source audit with repository search — no `PlanePair`, `view_planes`, `plane_pairs`, `launch_copy_plane`, or `std::vector` remains in the CUDA/ROCm/shared GPU copy path; remote mirrors were cleaned after verification.

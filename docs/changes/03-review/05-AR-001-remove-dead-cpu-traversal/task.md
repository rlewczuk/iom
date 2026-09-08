**Status:** done

## Summary

Removed the dead CPU coordinate/plane traversal helpers and four direct worker-only includes while preserving the active tiled traversal and CPU transfer behavior.

## Verification

- exhaustive CPU source search for for_each_coordinate, plane_at, and direct condition_variable/deque/exception/thread includes — no matches in the CPU source tree/device.cpp.
- focused CPU verification: cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/cpu --target iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/cpu -R "^(iom_cpu_tests|iom_backend_conformance_cpu_tests)$" --output-on-failure — 2/2 tests passed.
- final-train CPU combined verification repeated the CPU unit and backend-conformance command — 2/2 tests passed.

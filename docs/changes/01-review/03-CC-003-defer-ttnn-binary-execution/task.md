**Status:** done

## Summary

TTNN binary admission now publishes immutable outcomes before worker-side binary execution, preserving FIFO/API-mutex/lease ownership and repeatable retained failures with a deterministic barrier seam.

## Verification

- cmake -S . -B /tmp/iom-01-review-03-cpu -DCMAKE_BUILD_TYPE=Debug && cmake --build /tmp/iom-01-review-03-cpu -j2 --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests && ctest --test-dir /tmp/iom-01-review-03-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$' — all 3 local CPU tests passed
- .omp/skills/remote-development/scripts/remote-sync ttnn 01-review-03-CC-003-ttnn followed by remote-exec TTNN build and ctest — built TTNN and CPU control targets; both iom_ttnn_conformance_tests and iom_backend_conformance_cpu_tests passed, including deferred barrier and retained-fault cases

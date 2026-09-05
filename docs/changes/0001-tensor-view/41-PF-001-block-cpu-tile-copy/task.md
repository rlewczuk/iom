**Status:** done

## Summary

Replaced CPU per-element transfer and queued-copy traversal with tile-blocked row helpers and lockstep plane traversal. Byte-aligned rows use `memcpy`; sub-byte rows retain LSB-first packing and zero unused bits. Added focused CPU conformance cases for window scope, byte-aligned physical equality, sub-byte packing, lockstep planes, and allocation stability. Added the single-case CPU benchmark target with the required throughput and latency checks.

## Verification

- `cmake --build build -j --target iom_cpu_tests iom_backend_conformance_cpu_tests iom_tests iom_cpu_bench`
- `taskset -c 0 ctest --test-dir build --output-on-failure -R '^iom_backend_conformance_cpu_tests$'` — passed
- `taskset -c 0 ctest --test-dir build --output-on-failure -R '^iom_cpu_tests$'` — passed
- `taskset -c 0 ctest --test-dir build --output-on-failure -R '^iom_tests$'` — passed
- `taskset -c 0 ctest --test-dir build --output-on-failure -R '^iom_cpu_bench$'` — passed
- Focused execution of all four new conformance cases — passed
- CPU benchmark thresholds — passed: F32 host-to-device 28.6562 GB/s, F32 device-to-host 15.8162 GB/s, queued F32 164.094 GB/s, I4 host-to-device 37.7566 GB/s, F32 16x16 submit+wait 2.585 us
- Static audits — legacy coordinate helpers retained only as definitions, CPU allocation delegates to the shared aligned helpers, no backend or top-level CMake files changed, and the benchmark contains exactly one doctest case.

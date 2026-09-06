**Status:** done

## Summary

Unified TTNN quarantine-action failure handling with the other registry backends. TTNN tensors now own their native plane vector through `std::unique_ptr`, release it without destruction on quarantine-action failure, and return normally instead of terminating. Added the internal TTNN fault-injection seam, TTNN conformance regression, test-only CMake definition, and the implementation-generic deliberate-leak exception to umbrella §7.

## Verification

- `cmake -S . -B build/st004-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/st004-cpu -j2 --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests` — configured and built all requested CPU targets successfully.
- `ctest --test-dir build/st004-cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — 3/3 tests passed, including the unchanged CPU quarantine cleanup case.
- TTNN remote workflow on `bv1`: configured `BUILD_TESTING=ON` with `TTNN_ENABLED=ON`, built `iom_ttnn_smoke_tests` and `iom_ttnn_conformance_tests`, and ran `flock /tmp/iom-ttnn.lock ctest --test-dir build/st004-ttnn --output-on-failure -R iom_ttnn` — 2/2 tests passed on TT hardware.
- Source and build audits — no `std::terminate` in `src/ttnn/device.cpp`; only the intended `planes_.release()` member access; the two seam declarations, guarded definitions, single consume site, TTNN `IOM_ENABLE_TESTING` definition, and one umbrella `deliberately leaked` sentence were observed.

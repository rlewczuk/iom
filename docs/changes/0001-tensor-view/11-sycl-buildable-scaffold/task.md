**Status:** done

## Summary

Added the optional SYCL backend scaffold: Intel oneAPI compiler/runtime discovery, target-local SYCL compile and link settings, a public factory header without SYCL types, an owned context-backed device implementation, and real accelerator smoke coverage.

## Verification

- `REMOTE_DEV_CONFIG=/tmp/iom-sycl-host.conf .agents/skills/remote-development/scripts/remote-sync sycl task-11-sycl-scaffold` followed by remote configuration, `cmake --build build/sycl-smoke --target iom_sycl_smoke_tests -j`, and `ctest --test-dir build/sycl-smoke --output-on-failure -R "^iom_sycl_smoke_tests$"` on `bv2` — configuration and build succeeded; 1/1 SYCL smoke test passed on real Intel Arc Pro B60 accelerator hardware.
- `cmake -S . -B build/sycl-disabled -DBUILD_TESTING=ON -DSYCL_ENABLED=OFF -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/sycl-disabled --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j && ctest --test-dir build/sycl-disabled --output-on-failure -R '^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$'` — all 3 CPU/core tests passed; disabled compile commands contain no SYCL flags or runtime links.

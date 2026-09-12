**Status:** done

## Summary

Extracted StagedWorker into a transitive detail header, registered it for installation, and kept both headers within the 499-line cap without changing consumers.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF — reconfigure completed successfully after rebase.
- cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests — all targets built successfully after rebase.
- ctest --test-dir build/split-cpu --output-on-failure -R ^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$ — 4/4 tests passed after rebase.
- c++ -std=c++20 -Iinclude /tmp/iom-staged-worker-consumer.cpp build/split-cpu/libiom.a -pthread -o /tmp/iom-staged-worker-consumer && /tmp/iom-staged-worker-consumer — umbrella-only consumer compiled, linked, and ran successfully.
- wc -l include/iom/iom.hpp include/iom/detail/staged_worker.hpp — 499 and 192 lines respectively.

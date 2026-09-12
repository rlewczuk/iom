**Status:** done

## Summary

Split the outstanding-work registry detail contract into four acyclic responsibility headers while retaining the umbrella transitive include boundary and behavior.

## Verification

- cmake -S . -B build/split-cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF — configure completed successfully.
- cmake --build build/split-cpu --target libiom iom_tests iom_scalar_add_tests iom_cpu_tests iom_backend_conformance_cpu_tests — all targets built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^(iom_tests|iom_scalar_add_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$ — 4/4 tests passed.
- c++ -std=c++20 -Iinclude /tmp/iom-registry-consumer.cpp build/split-cpu/libiom.a -pthread -o /tmp/iom-registry-consumer && /tmp/iom-registry-consumer — umbrella and shard consumer compiled, linked, and ran successfully.
- wc -l include/iom/detail/outstanding_work_registry.hpp include/iom/detail/fence.hpp include/iom/detail/outstanding_work_registry_core.hpp include/iom/detail/outstanding_work_cleanup.hpp include/iom/detail/workspace_registry.hpp — counts 6, 184, 323, 141, and 315, all <=499.

**Status:** done

## Summary

Split core DeviceOps copy and neural families into dedicated translation units and removed their duplicate definitions from iom.cpp.

## Verification

- cmake configure and build targets libiom iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests — all targets built successfully.
- ctest --test-dir build/split-cpu --output-on-failure -R ^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$ — 3/3 tests passed.
- wc -l src/device_ops_copy.cpp src/device_ops_neural.cpp src/iom.cpp — 47, 81, and 673 lines respectively; final iom.cpp cutover is specified for task 05.

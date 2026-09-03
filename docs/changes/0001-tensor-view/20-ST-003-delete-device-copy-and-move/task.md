**Status:** ready

## Summary

Deleted `iom::Device` copy and move construction and assignment, preserving stable owner addresses, and added compile-time assertions for the concrete `FakeDevice` fixture in `test/test_iom.cpp`. An explicitly defaulted default constructor preserves existing derived-device construction because declaring deleted copy/move operations suppresses the implicit default constructor in C++.

## Verification

- `cmake -S . -B build && cmake --build build -j && ctest --test-dir build -R '^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$' --output-on-failure` — configure/build succeeded; all 3 CPU tests passed.
- Negative compile control with only `include/iom/device.hpp` stashed — `iom_tests` failed at all 4 new `FakeDevice` trait assertions; after restoring the header, the target rebuilt and `iom_tests` passed.

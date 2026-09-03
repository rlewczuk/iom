**Status:** done

## Summary

Deleted `iom::Device` copy and move construction and assignment, preserving stable owner addresses, and added compile-time assertions for the concrete `FakeDevice` fixture in `test/test_iom.cpp`. An explicitly defaulted default constructor preserves existing derived-device construction because declaring deleted copy/move operations suppresses the implicit default constructor in C++.

## Verification

- `cmake -S . -B build && cmake --build build -j && ctest --test-dir build -R '^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$' --output-on-failure` — configure/build succeeded; all 3 CPU tests passed.
- Negative compile control with only `include/iom/device.hpp` stashed — `iom_tests` failed at all 4 new `FakeDevice` trait assertions; after restoring the header, the target rebuilt and `iom_tests` passed.
- `remote-exec cuda st003-device-copy-move 'cmake -S . -B build -DCUDA_ENABLED=ON -DROCM_ENABLED=ON -DTTNN_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j'` — CUDA, ROCm, and TTNN libraries and tests compiled successfully.
- `remote-exec cuda st003-device-copy-move-cuda-ttnn 'ctest --test-dir build -R iom_cuda --output-on-failure'` — CUDA smoke and conformance tests passed.
- `remote-exec cuda st003-device-copy-move-cuda-ttnn 'ctest --test-dir build -R iom_ttnn --output-on-failure'` — TTNN smoke and conformance tests passed.
- `remote-exec rocm st003-device-copy-move 'ctest --test-dir build -R iom_rocm --output-on-failure'` — ROCm smoke and conformance tests passed.
- `remote-exec cuda st003-device-copy-move-cuda-ttnn 'ctest --test-dir build -R iom_backend_coexistence_tests --output-on-failure'` — CUDA/TTNN coexistence test passed.
- `remote-exec rocm st003-device-copy-move 'ctest --test-dir build -R iom_backend_coexistence_tests --output-on-failure'` — ROCm coexistence test passed.

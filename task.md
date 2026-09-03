**Status:** done

## Summary

Added a backend-neutral standard-layout storage encoder and `AcceleratorStorageOracle` contract, plus the shared physical-storage sweep across transfer shapes, copy shapes, leaf widths, and transformed views. Added direct CUDA and ROCm allocation oracles, a TTNN owner-plane oracle using native host-buffer/tensor APIs, CPU oracle coverage, and reusable identical-permutation negative fixtures for all accelerator drivers.

## Verification

- `cmake -S . -B build/cpu -DBUILD_TESTING=ON && cmake --build build/cpu --target iom_backend_conformance_cpu_tests iom_cpu_tests -j2 && ctest --test-dir build/cpu --output-on-failure -R '^iom_(cpu|backend_conformance_cpu)_tests$'` — 2/2 tests passed.
- `remote-exec cuda task-nt001-cuda 'cmake --build build -j --target iom_cuda_conformance_tests && ...'` — CUDA negative fixture and full storage-oracle sweep passed; full sweep reported 17,253 assertions passed.
- `remote-exec rocm task-nt001-rocm 'cmake --build build -j --target iom_rocm_conformance_tests && ...'` — ROCm negative fixture and full storage-oracle sweep passed; full sweep reported 17,091 assertions passed.
- `remote-exec ttnn task-nt001-ttnn 'cmake --build build -j --target iom_ttnn_conformance_tests && ...'` — TTNN negative fixture and full storage-oracle sweep passed; full sweep reported 34,948 assertions passed.
- `remote-clean cuda task-nt001-cuda`, `remote-clean rocm task-nt001-rocm`, `remote-clean ttnn task-nt001-ttnn` — all remote mirrors removed.

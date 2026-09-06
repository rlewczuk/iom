**Status:** done

## Summary

Extended `copy_cases_for` with contiguous-to-stepped and permuted-to-contiguous asymmetric-stride cases. Added `{17, 16}` row-tail and `{1, 1}` minimal-extent owners to both shared transfer/copy shape matrices, and wired the CPU asynchronous-copy conformance case to `CpuStorageOracle`. No production files changed.

## Verification

- On the final rebased tree (`3e2bfc7`), `cmake -S . -B build/cpu -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF`, `cmake --build build/cpu --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j`, and `ctest --test-dir build/cpu --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — configured, built, and passed 3/3 tests.
- Temporary CPU mirrored-stride injection: the pre-extension async-copy case passed 1/1 test with 4785/4785 assertions; the extended case failed at `contiguous source to stepped destination`, and an isolated `{2, 2, 2, 3, 17, 33}` run failed at `permuted source to contiguous destination`; the injection and temporary matrix reductions were reverted.
- CUDA final remote build and `ctest --test-dir build --output-on-failure -R ^iom_cuda_conformance_tests$` — passed; focused async-copy and storage-oracle cases passed with 18954/18954 and 21393/21393 assertions. A temporary shared metadata mirrored-stride injection failed at both new labels in isolated runs; revert and final CTest passed.
- ROCm final remote build and `ctest --test-dir build --output-on-failure -R ^iom_rocm_conformance_tests$` — passed; focused async-copy and storage-oracle cases passed with 17389/17389 and 21139/21139 assertions.
- SYCL final remote build and `ctest --test-dir build --output-on-failure -R ^iom_sycl_conformance_tests$` — passed with the interactive oneAPI environment; focused async-copy and storage-oracle cases passed with 16289/16289 and 19510/19510 assertions.
- TTNN final remote build and `ctest --test-dir build --output-on-failure -R ^iom_ttnn_conformance_tests$` — passed; focused async-copy and storage-oracle cases passed with 32310/32310 and 37207/37207 assertions.

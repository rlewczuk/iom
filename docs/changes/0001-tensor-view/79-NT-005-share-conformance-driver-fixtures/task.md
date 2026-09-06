**Status:** done

## Summary

TrafficGate and expect_repeated_runtime_failure now exist once in test/backend/backend_conformance_common.hpp (namespace iom_conformance); the four byte-identical driver copies are deleted and every driver reference is iom_conformance-qualified; conformance behavior unchanged on all four backends.

## Verification

- ctest --test-dir build --output-on-failure -R 'iom_backend_conformance_cpu_tests|iom_tests' (local CPU) — 2/2 passed
- remote-exec sycl: sycl-ls enumerated 2 Level-Zero GPUs, then cmake --build build/sycl --target iom_sycl_conformance_tests && ctest -R '^iom_sycl_conformance_tests$' — passed
- remote-exec cuda: cmake --build build/cuda --target iom_cuda_conformance_tests && ctest -R '^iom_cuda_conformance_tests$' on bv1 — passed
- remote-exec rocm: cmake --build build/rocm --target iom_rocm_conformance_tests && ctest -R '^iom_rocm_conformance_tests$' on bv2 — passed
- static audits: 'class TrafficGate' and 'void expect_repeated_runtime_failure' each match exactly once (shared header); all driver references carry iom_conformance::; git diff --stat -- src include empty

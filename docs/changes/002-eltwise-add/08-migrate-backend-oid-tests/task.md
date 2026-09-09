**Status:** done

## Summary

Migrated shared backend fakes, every backend conformance driver, and coexistence tests to signed OID results with exact 55-bit decoding, complete invalid-wait coverage, skipped-sequence rejection, error mapping, repeat waits, retained failures, and queue reuse semantics.

## Verification

- cmake --build /tmp/iom-002-eltwise-add-08-build --target iom_tests iom_backend_conformance_cpu_tests && ctest --test-dir /tmp/iom-002-eltwise-add-08-build --output-on-failure -R ^(iom_tests|iom_backend_conformance_cpu_tests)$ — both local targets passed
- remote-exec cuda 002-eltwise-add-08-cuda: ctest --test-dir build/cuda --output-on-failure -R ^(iom_cuda_conformance_tests|iom_backend_coexistence_tests)$ — CUDA conformance and coexistence passed
- remote-exec rocm 002-eltwise-add-08-rocm: ctest --test-dir build/rocm --output-on-failure -R ^(iom_rocm_conformance_tests|iom_backend_coexistence_tests)$ — ROCm conformance and coexistence passed
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf remote-exec sycl 002-eltwise-add-08-sycl: ctest --test-dir build/sycl --output-on-failure -R ^(iom_sycl_conformance_tests|iom_backend_coexistence_tests)$ — SYCL conformance and coexistence passed after Level Zero visibility check
- remote-exec ttnn 002-eltwise-add-08-ttnn: ctest --test-dir build/ttnn --output-on-failure -R ^(iom_ttnn_conformance_tests|iom_backend_coexistence_tests)$ — TTNN conformance and coexistence passed

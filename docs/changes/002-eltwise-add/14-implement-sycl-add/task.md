**Status:** done

## Summary

Implemented SYCL ADD with device-USM-safe temporary staging, exact host scalar arithmetic, ordered ownership, and lifetime-safe padded extent metadata.

## Verification

- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf remote-sync sycl 002-eltwise-add-14-sycl && remote-exec sycl 002-eltwise-add-14-sycl cmake --build build/sycl -j — build passed
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf remote-sync sycl 002-eltwise-add-14-sycl && remote-exec sycl 002-eltwise-add-14-sycl sycl-ls — Level Zero GPU devices enumerated
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-remote-hosts.conf bounded remote SYCL ctest -R "^(iom_sycl_smoke_tests|iom_sycl_conformance_tests)$" — smoke and conformance passed 2/2
- Final 16 train CPU/CUDA/ROCm/SYCL/TTNN backend-focused conformance — CPU 3/3, CUDA 2/2, ROCm 2/2, SYCL 2/2, TTNN 2/2 passed

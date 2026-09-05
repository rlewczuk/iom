**Status:** done

## Summary

Added `Device::supported_data_types()` and authoritative capability spans for CPU, CUDA, ROCm, SYCL, and TTNN. Migrated the four accelerator conformance drivers to query the candidate device, added backend membership tests, and added `FakeShrunkDevice` mutation coverage in `test/test_iom.cpp`.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_tests iom_backend_conformance_cpu_tests` — passed.
- Local `ctest` for `iom_tests` and `iom_backend_conformance_cpu_tests`, plus the focused mutation, CPU membership, tensor-owner, and device-owner tests — passed.
- CUDA remote configure/build/conformance and focused membership, compute-capability, and storage tests on `bv1` — passed.
- ROCm remote configure/build/conformance and focused membership, compute-capability, and storage tests on `bv2` — passed.
- TTNN remote configure/build/conformance and focused membership, compute-capability, and storage tests on `bv1` — passed.
- SYCL remote configure/build with `/opt/intel/oneapi/compiler/2026.1/bin/icpx`, full conformance, and focused membership, compute-capability, and storage tests on `bv2` after sourcing `/opt/intel/oneapi/setvars.sh --force` — passed; `sycl-ls` reported two Level Zero GPU devices.

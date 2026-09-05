**Status:** done

## Summary

Replaced CUDA and ROCm bit-granular copy paths with destination-word ownership. Shared GPU helpers now extract fields with 32-bit word loads, use an explicit two-word 64-bit path, merge in registers, preserve untouched bits, and perform one non-atomic store per owned word across device copies and host scatter/gather transfers. PF-002 metadata, launch, fault-injection, and PF-003 pooled transfer interfaces remain unchanged.

## Verification

- Remote CUDA configure/build with `BUILD_TESTING=ON`, CUDA enabled and ROCm/SYCL/TTNN disabled — build succeeded.
- Remote ROCm configure/build with `BUILD_TESTING=ON`, ROCm enabled and CUDA/SYCL/TTNN disabled — build succeeded.
- Remote CUDA `flock /tmp/iom-cuda0.lock ctest --test-dir build-pf006-cuda -E iom_cpu_bench --output-on-failure` — 6/6 tests passed, including CUDA smoke/conformance, CPU/common, and backend coexistence coverage.
- Remote ROCm `flock /tmp/iom-rocm0.lock ctest --test-dir build-pf006-rocm -E iom_cpu_bench --output-on-failure` — 6/6 tests passed, including ROCm smoke/conformance, CPU/common, and backend coexistence coverage.
- Source audit — zero `atomicOr`, `atomicAnd`, `read_bits`, `write_bits`, `copy_value`, and `IOM_GPU_ATOMIC` matches in CUDA/ROCm/shared GPU copy sources; CUDA PTX audit reported zero `atom.`/`red.` instructions and three global stores for the three copy kernels.

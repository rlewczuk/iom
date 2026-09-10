**Status:** done

## Summary

Replaced duplicated CUDA/ROCm binary tile-slot and coordinate arithmetic with the canonical standard_tiled_copy plane_slot and physical_coordinate helpers, preserving launch and operation behavior.

## Verification

- cmake --build build/review-cpu --target iom_scalar_add_tests iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R "^(iom_scalar_add_tests|iom_backend_conformance_cpu_tests)$" — both CPU targets passed
- CUDA iom_cuda_conformance_tests — 46,190 assertions passed after restored canonical mapping
- ROCm iom_rocm_conformance_tests — 45,958 assertions passed after restored canonical mapping
- SYCL iom_sycl_conformance_tests — full conformance passed with icx/icpx configuration
- TTNN iom_ttnn_conformance_tests — full conformance passed
- Temporary canonical plane_slot +1 mutation made CUDA binary ADD/MUL/SUB/DIV fail 42,824 assertions and CUDA storage/host-transfer conformance fail at byte 11 for BOOL; restored exact mapping

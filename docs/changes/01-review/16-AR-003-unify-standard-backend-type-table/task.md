**Status:** done

## Summary

The four duplicated 23-leaf supported-data-type arrays are unified into one shared inline constexpr std::array with a common read-only span helper in src/shared/standard_tiled_copy.hpp; CPU/CUDA/ROCm/SYCL advertise identical capability with zero allocation or runtime query; TTNN native mapping and per-backend conformance expectations unchanged.

## Verification

- ctest -R "^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$" local CPU — 3/3 passed (34.88s)
- iom_cuda_conformance_tests on bv1 — 1/1 passed (34.79s)
- iom_rocm_conformance_tests on bv2 — 1/1 passed (39.41s)
- iom_sycl_conformance_tests on bv2 — 1/1 passed (6.87s)
- combined wave-3 train: CUDA/ROCm/SYCL/TTNN 3/3 each

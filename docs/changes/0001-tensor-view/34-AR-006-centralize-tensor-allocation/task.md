**Status:** done

## Summary

Added the backend-neutral header-only helpers `iom::detail::allocate_aligned_storage` and `iom::detail::release_aligned_storage` in `include/iom/detail/aligned_storage.hpp`. CPU, CUDA, ROCm, and SYCL tensor construction and destruction now use the shared 32-byte alignment and single-free contracts; CUDA and ROCm retain their device-activation callbacks, and SYCL retains its post-allocation USM compatibility cleanup. Documented the 32-byte `Allocator::alloc` obligation in `include/iom/alloc.hpp`. TTNN and all test files remain unchanged.

## Verification

- After merging the latest `main`, `cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_cpu_tests iom_backend_conformance_cpu_tests iom_tests` — all three targets built successfully.
- After merging the latest `main`, `ctest --test-dir build --output-on-failure` — 3/3 local tests passed.
- CPU focused allocation, misalignment, allocator-failure, exhaustion, and supported-data-type mutation tests — passed; the misalignment case passed 12 assertions.
- After merging the latest `main`, remote CUDA configure/build and `flock /tmp/agent-gpu0.lock ctest --test-dir build --output-on-failure -R "iom_cuda_(conformance|smoke)_tests"` — both suites passed; the focused CUDA misaligned-allocation case passed 5 assertions.
- After merging the latest `main`, remote ROCm configure/build and `flock /tmp/agent-gpu1.lock ctest --test-dir build --output-on-failure -R "iom_rocm_(conformance|smoke)_tests"` — both suites passed.
- After merging the latest `main`, remote SYCL configure/build and runtime-path-configured `flock /tmp/agent-gpu2.lock ctest --test-dir build --output-on-failure -R "iom_sycl_(conformance|smoke)_tests"` — both suites passed.
- Static audits — one shared `kStorageAlignment` declaration, four backend alignment-message call sites, helper calls in all four standard-layout backends, and zero helper references in TTNN.

**Status:** done

## Summary

Removed the unused `iom` hello-world executable and `src/main.cpp`, deleted the four unconsumed backend enable compile definitions, and removed `SingleBufferAllocatorBase::owns(void*)` while preserving active SYCL configuration and allocator behavior.

## Verification

- Repository structural search for `IOM_*_ENABLED`, the `iom` executable target, allocator `owns` references, and `.owns(` calls — no dead-surface matches; preserved SYCL and `__HIP_PLATFORM_AMD__` markers remained present.
- `cmake -S . -B build -DBUILD_TESTING=ON` — configured successfully.
- `cmake --build build -j` — built `libiom.a`, `iom_tests`, `iom_cpu_tests`, and `iom_backend_conformance_cpu_tests`; no `iom` executable was produced.
- `ctest --test-dir build --output-on-failure` — 3/3 CPU tests passed.
- `cmake -S . -B build-sycl -DSYCL_ENABLED=ON -DSYCL_COMPILER=/nonexistent` — failed with the existing `SYCL compiler does not exist: /nonexistent` diagnostic.

**Status:** done

## Summary

SYCL synchronous host transfers now use one host-USM staging allocation. Calling-thread `std::memcpy` moves exact logical bytes between caller spans and that device-accessible allocation, so SYCL runtime operations never receive ordinary host pointers and downloads do not depend on remapping shared USM for host access. Only `src/sycl/copy.cpp` changed.

## Verification

- `.agents/skills/remote-development/scripts/remote-sync sycl st005-host-usm-final-20260906` — synchronized the exact task worktree to the SYCL host.
- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF` — configured successfully with IntelLLVM 2026.1.
- `cmake --build build/sycl --target iom_sycl_conformance_tests iom_sycl_smoke_tests iom_tests -j` — built all requested targets successfully.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"` — passed 2/2 tests.
- `flock /tmp/agent-gpu0.lock ./build/sycl/test/iom_sycl_conformance_tests -tc="*storage oracle*"` — passed both cases and 18,774 assertions.
- `flock /tmp/agent-gpu0.lock ./build/sycl/test/iom_sycl_conformance_tests -tc="*transfer failures*"` — passed one case and 705 assertions.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"` — passed 1/1 test.
- Static source audit — `src/sycl/copy.cpp` contains one `sycl::malloc_host`, two host-leg `std::memcpy` calls, and the preserved `std::memset`; no `queue.memcpy` remains under `src/sycl`.

**Status:** done

## Summary

Converged SYCL copy and host-transfer paths onto shared word-oriented kernels, a single metadata-driven queued launch, queue-fenced metadata slots, and device-owned pooled staging with `malloc_device`/`malloc_host` mirrors. Added SYCL staging-pool lifecycle coverage, migrated transfer call sites, and preserved the 55-ST-001 fault and fence contracts.

## Verification

- `cmake -S . -B build/sycl -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF` via remote-development `sycl` — configured successfully with IntelLLVM 2026.1.
- `cmake --build build/sycl --target iom_sycl_smoke_tests iom_sycl_conformance_tests iom_tests -j` via remote-development — all requested targets built successfully.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/sycl --output-on-failure -R "^iom_sycl_(conformance|smoke)_tests$"` — 2/2 tests passed.
- `ctest --test-dir build/sycl --output-on-failure -R "^iom_tests$"` — 1/1 test passed.
- `./build/sycl/test/iom_sycl_conformance_tests -tc="*pre-enqueue failures preserve submission sequences*,*transactional across post-enqueue failures*,*queue destruction fences pending copies*,*SyclFenceState::result() idempotency*"` under exclusive access — 4 cases and 87 assertions passed.
- `valgrind --error-exitcode=99 --leak-check=no ./build/sycl/test/iom_sycl_conformance_tests -tc="*destruction fences queued work*,*transactional across post-enqueue failures*,*queue destruction fences pending copies*,*SyclFenceState::result() idempotency*,*host transfers reuse pooled staging*"` under exclusive access — 5 cases and 101 assertions passed; Memcheck reported `ERROR SUMMARY: 0 errors from 0 contexts`.
- Final static audits — no legacy SYCL atomic/per-plane/malloc_shared/resource symbols; one shared tiled-copy include; exactly one `pool->release(slot)` site; shared-only diff limited to `standard_tiled_copy.inl`; accelerator-unrelated source diff empty; `git diff --check` passed.

**Status:** done

## Summary

ListAllocator::alloc/free and coalescing are strongly exception-safe: complete post-split/post-release geometry prepared before commit with noexcept vector swap and strong-guarantee map insert; FixedSizeAllocator::free appends the released index before clearing in_use_ (push-before-clear) and can never drop a slot; best-fit, alignment, coalescing, stable addresses, std::bad_alloc/std::invalid_argument categories, and standalone public allocator APIs unchanged; no mutexes

## Verification

- cmake --build build --target iom_tests — clean
- ctest --test-dir build --output-on-failure -R "^iom_tests$" — 120/120 doctest cases, 11323/11323 assertions passed (focused rollback: injected bookkeeping failure leaves free_bytes/slot state exact, recovery reuses the same address exactly once)
- combined train: ctest --test-dir build --output-on-failure -R "^(iom_tests|iom_backend_conformance_cpu_tests)$" — 2/2 passed

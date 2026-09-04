**Status:** done

## Summary

Changed `LinearAllocator::alloc` exhaustion to throw `std::bad_alloc` and added direct allocator and CPU `create_tensor` regression coverage. The CPU regression confirms an exhausted real `LinearAllocator` leaves the first tensor's storage and ownership intact while the failed construction returns no tensor.

## Verification

- `cmake -S . -B build/cc004 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF` — configured successfully.
- `cmake --build build/cc004 --target iom_tests iom_cpu_tests` — both targets built successfully.
- `./build/cc004/test/iom_tests '--test-case=LinearAllocator*,ListAllocator rejects invalid frees and out-of-memory allocation,FixedSizeAllocator rejects invalid frees and exhausted allocations'` — 10 test cases and 28 assertions passed.
- `./build/cc004/test/iom_cpu_tests '--test-case=*create_tensor*allocator*,CPU tensors reject null and misaligned allocations exactly once'` — 3 test cases and 20 assertions passed.
- `ctest --test-dir build/cc004 --output-on-failure -R '^(iom_tests|iom_cpu_tests)$'` — 2/2 tests passed.

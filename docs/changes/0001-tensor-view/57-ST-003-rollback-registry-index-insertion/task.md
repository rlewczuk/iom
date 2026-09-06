**Status:** done

## Summary

Restructured `OutstandingWorkRegistry::register_entry` to use one exception handler that rolls back the complete entry through `erase_entry_locked`. Added a deterministic process-global allocation-fault sweep covering every registry index insertion ordinal in `test/test_iom.cpp`.

## Verification

- `cmake -S . -B build/st003-rollback -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/st003-rollback --target iom_tests -j` — configured and built successfully.
- `./build/st003-rollback/test/iom_tests -tc='*rolls back every registry index*'` — 1 test passed, 54 assertions passed.
- `ctest --test-dir build/st003-rollback --output-on-failure -R '^iom_tests$'` — 1/1 test passed.
- `cmake -S . -B build/st003-rollback-debug -DBUILD_TESTING=ON -DCMAKE_CXX_FLAGS='-D_GLIBCXX_DEBUG' -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/st003-rollback-debug --target iom_tests -j && ./build/st003-rollback-debug/test/iom_tests -tc='*rolls back every registry index*'` — debug build succeeded; 1 test and 54 assertions passed.
- `cmake --build build/st003-rollback --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j && ctest --test-dir build/st003-rollback --output-on-failure -R '^(iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests)$'` — all 3 CPU test targets passed.
- Reverting only the rollback restructure reproduced the address-index leak (`try_release_entry` returned true followed by a SIGSEGV) in the default build and singular-iterator SIGABRT under `_GLIBCXX_DEBUG`; restoring the restructure made both focused runs pass.
- Source audit confirmed one rollback `catch`, `erase_entry_locked(entry_it)`, and the helper's `noexcept` signature.

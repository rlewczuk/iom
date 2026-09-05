**Status:** done

## Summary

Reserved submission sequences before invoking `queue_work`, rolled back synchronous failures only when safe, and updated retained-failure validation for reserved-but-not-completed sequences. Added skipped-sequence tracking for the test seam so never-submitted gaps remain rejected, plus inline-completion, rollback, retained-failure, concurrency, exhaustion, and boundary regression coverage in `test/test_iom.cpp`.

## Verification

- `cmake -S . -B build -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_tests iom_cpu_tests` — configured and built both CPU test targets successfully.
- `ctest --test-dir build --output-on-failure -R iom_tests` — 1/1 test passed.
- `ctest --test-dir build --output-on-failure -R iom_cpu_tests` — 1/1 test passed.
- `./build/test/iom_tests -tc='*submit commits the sequence before queue_work runs and survives inline complete*'`, `-tc='*commit_failure accepts a reserved-but-not-completed sequence after submit reservation*'`, `-tc='*commit_failure accepts a reserved sequence under concurrent submit*'`, `-tc='*commit_failure rejects a never-submitted sequence*'`, `-tc='*commit_failure rejects an already-completed sequence*'`, and `-tc='*submit suppresses rollback when queue_work completes inline and then throws*'` — each selected case passed.
- Remote CUDA `cmake -S . -B build -DBUILD_TESTING=ON -DCUDA_ENABLED=ON -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build -j --target iom_cuda_conformance_tests && flock /tmp/agent-gpu0.lock ctest --test-dir build --output-on-failure -R iom_cuda_conformance_tests` on `bv1` — configured, built, and 1/1 CUDA conformance test passed.

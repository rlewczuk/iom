**Status:** done

## Summary

Reduced `OutstandingWorkRegistry` to its `by_id_` and `by_address_` indexes, aliased `EntrySnapshot` to `Entry`, and replaced the intrusive quarantine list with a pinned vector-backed implementation. Relocated the registry header under `iom/detail`, updated all includers, and removed synchronous CPU copy registration and outcome bookkeeping. Expanded registry and quarantine regression coverage, including queue-scoped invalidation and quarantine-growth allocation failure.

## Verification

- `cmake -S . -B build/ar003-registry -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF && cmake --build build/ar003-registry --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests iom_cpu_bench -j` — configured and built all required CPU targets successfully.
- `./build/ar003-registry/test/iom_tests -tc='*OutstandingWorkRegistry*'` and `./build/ar003-registry/test/iom_tests -tc='*Quarantine*'` — 5 focused cases passed, 60 assertions passed.
- `ctest --test-dir build/ar003-registry --output-on-failure -R 'iom_tests|iom_cpu_tests|iom_backend_conformance_cpu_tests'` — 3/3 CPU tests passed.
- CUDA and ROCm remote conformance plus smoke suites — each profile configured, built, and passed its 2/2 CTest targets; deferred-lifetime and queue-destruction focused cases passed.
- TTNN remote conformance plus smoke suite — 2/2 CTest targets passed; the deferred queue-lifetime focused case passed.
- SYCL remote conformance plus smoke suite — 2/2 CTest targets passed; destruction-fence, post-enqueue-failure, queue-destruction, and `SyclFenceState::result()` focused cases passed.
- SYCL remote `valgrind --leak-check=no --error-exitcode=1` runs for those four lifetime cases — all passed with `ERROR SUMMARY: 0 errors from 0 contexts`.
- Source audits in the task worktree — removed registry indexes/APIs and old include paths absent; detail header contains the `EntrySnapshot` alias, two indexes, and pinned `Quarantine::add` release path; CPU registration/outcome symbols absent; `git diff --check` passed.
- Five sequential `iom_cpu_bench` runs on the pre-change archived tree and five on this tree — queued 16x16 submit+wait median improved from 18.955 us to 11.201 us. The benchmark's existing absolute 6 us gate remained red as documented by the specification; the required before/after median comparison passed.

**Status:** done

## Summary

Stored `TensorView` by value in CPU and TTNN queue tasks instead of raw `TensorView*`/`ttnn::Tensor*` to caller-owned temporaries (review finding ST-002). `CpuTask` and `TtnnTask` now carry `TensorView source`/`TensorView destination` value members member-initialized by a new `Task(std::uint64_t, const TensorView&, TensorView&, bool)` constructor invoked inside the submit lambda on the caller thread; the AR-002 `void* fence = nullptr` field is preserved and no `std::exception_ptr failure` field is added. The TTNN task drops the cached `source_planes`/`destination_planes` fields entirely; the worker derives the native plane pointers from the value-copied view's `native_handle()` as stack locals at execution time (spec shape (a)). `StagedWorker` extraction already move-constructs, which is the only legal shape given `TensorView`'s deleted assignment operators. Added `CPU copy survives derived-view temporaries` and `TTNN copy survives derived-view temporaries` tests that submit the copy from a helper frame returning before `wait`, with no named local binding either derived view (the destination rvalue binds through `const_cast` because `copy`'s destination parameter is non-const; the view object is never modified).

Files changed: `src/cpu/device.cpp`, `src/ttnn/device.cpp`, `test/cpu/test_cpu_conformance.cpp`, `test/ttnn/test_ttnn_conformance.cpp`.

## Verification

- `ctest --test-dir build-st002 --output-on-failure -R 'iom_cpu_tests|iom_backend_conformance_cpu_tests'` — 2/2 passed, including the new temporary-view case (10 assertions).
- ASan build (`-fsanitize=address`, `ASAN_OPTIONS=detect_stack_use_after_return=1`), same ctest filter — 2/2 passed, no ASan diagnostic.
- Remote TTNN (host `ttnn`, task `st002-ttnn`): `cmake --build build -j --target iom_ttnn_smoke_tests iom_ttnn_conformance_tests` + `ctest --test-dir build --output-on-failure -R "iom_ttnn_smoke_tests|iom_ttnn_conformance_tests"` — 2/2 passed; `-tc="TTNN copy survives derived-view temporaries"` — 1/1 passed (3 assertions). Remote mirror cleaned.
- Acceptance greps: no `TensorView*`/`ttnn::Tensor*` Task fields remain (only the worker-thread stack locals in `TtnnQueue::execute` per spec shape (a)); no `task.(source|destination) =` assignments; no `enable_shared_from_this`/`shared_ptr<Tensor>`; `StagedWorker` uses `Task current(std::move(...))` move-construction.
- Post-merge of `main` (PF-004 moved the TTNN `finish()` to the wait path, conflicting with the worker block; resolved in the task worktree keeping the value-copy shape without the removed `finish()`): CPU ctest 2/2 passed and remote TTNN ctest 2/2 passed with the new temporary-view case 1/1.
- Note: the spec's literal same-frame test expression (`queue->copy(temp, temp)`) cannot compile because `copy(const TensorView&, TensorView&)` cannot bind a destination rvalue; the tests use a helper frame instead, preserving "no named local for either argument" and the temporary-lifetime hazard.

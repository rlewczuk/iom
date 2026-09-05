**Status:** done

## Summary

Moved TTNN's device-wide `finish()` from per-task staged-worker execution to the wait-boundary `TtnnQueue::fence_through_sequence` hook, serialized by a per-queue high-water mark. `DeviceOps::wait` now releases the completion mutex around the hook and rechecks retained failures; post-completion failures are recorded idempotently. The temporary finish-call counter seam was used for hardware verification and removed before shipping.

## Verification

- `cmake -S . -B build/pf004-cpu -DBUILD_TESTING=ON -DCPU_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=OFF -DTTNN_ENABLED=OFF && cmake --build build/pf004-cpu -j2` — CPU targets built successfully.
- `ctest --test-dir build/pf004-cpu --output-on-failure -E '^iom_cpu_bench$'` — 3/3 CPU/common tests passed.
- Remote TTNN configure/build with `TTNN_ENABLED=ON` through `remote-development` — `iom_ttnn` and all TTNN test targets built successfully on `bv1`.
- Remote `ctest --test-dir build/pf004-ttnn --output-on-failure -E "^iom_cpu_bench$"` under `flock` — 6/6 combined CPU/TTNN tests passed.
- Temporary remote counter smoke with 64 queued copies and repeated waits — printed `finish calls: 1` and exited successfully.
- Shipped-source audit — exactly one TTNN `finish()` call site, one `fence_mutex_` declaration/lock site, no verification counter symbols, and no `include/iom/ttnn/test_counter.hpp` path.

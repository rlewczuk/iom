**Status:** done

## Summary

Added the always-run CPU storage-recycling destruction regressions and shared lifetime-protocol unit coverage in `test/cpu/test_cpu.cpp` and `test/test_iom.cpp`. The tests cover source, destination, and both-operand destruction before wait; queue teardown with unwaited tokens; release-or-quarantine classification; two-phase registry rollback; and destructor-versus-completion race tolerance.

## Verification

- `cmake -S . -B build/nt003-lifetime -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF -DSYCL_ENABLED=OFF` — configured successfully.
- `cmake --build build/nt003-lifetime --target iom_tests iom_cpu_tests -j` — both targets built successfully after the final rebase.
- `ctest --test-dir build/nt003-lifetime --output-on-failure -R '^iom_tests$|^iom_cpu_tests$'` — 2/2 tests passed after the final rebase.
- Focused local doctest runs for destruction-before-wait, queue teardown, release-or-quarantine, registry rollback, and destructor-versus-completion — 5/5 passed (148 assertions) after the final rebase.
- Temporary protocol, registry-rollback, and CPU-release discriminator variants each failed their new assertions; restoring the production code made the focused regressions pass again.
- `REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf ./.agents/skills/remote-development/scripts/remote-sync sycl 69-NT-003-20260906-r3` — synchronized the newest rebased task worktree.
- Remote configure plus `cmake --build build/sycl --target iom_sycl_conformance_tests iom_sycl_smoke_tests iom_tests -j` — configured with IntelLLVM and built all requested targets successfully.
- `flock /tmp/agent-gpu0.lock ctest --test-dir build/sycl --output-on-failure -R '^iom_sycl_(conformance|smoke)_tests$'` and the common `^iom_tests$` ctest run — 2/2 SYCL targets and 1/1 common target passed.
- The three focused SYCL lifetime cases — each passed (14, 23, and 20 assertions).
- Valgrind Memcheck reruns of the three focused SYCL lifetime cases — each passed with `ERROR SUMMARY: 0 errors from 0 contexts`.
- `REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf ./.agents/skills/remote-development/scripts/remote-clean sycl 69-NT-003-20260906-r3` — removed the remote mirror.

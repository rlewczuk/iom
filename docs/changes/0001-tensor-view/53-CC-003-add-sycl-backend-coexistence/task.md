**Status:** done

## Summary

Added SYCL to the backend coexistence target gate and linked the SYCL library, headers, and `-fsycl` compile/link options when enabled. Extended the coexistence translation unit with a context-bound USM allocator, SYCL participant, foreign-device rejection coverage, and guarded second-device ordinal coverage. No production sources or unrelated backend wiring changed.

## Verification

- `git diff --check` — passed with no whitespace errors.
- `REMOTE_DEV_CONFIG=/tmp/task-53-cc003.remote-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl task-53-cc003 'flock /tmp/agent-gpu0.lock -c "cmake -S . -B build/sycl-cc003 -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DTTNN_ENABLED=OFF"'` — configured successfully on `bv2` with IntelLLVM 2026.1.0.
- `REMOTE_DEV_CONFIG=/tmp/task-53-cc003.remote-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl task-53-cc003 'flock /tmp/agent-gpu0.lock -c "cmake --build build/sycl-cc003 --target iom_backend_coexistence_tests -j"'` — built `iom_backend_coexistence_tests` successfully.
- `REMOTE_DEV_CONFIG=/tmp/task-53-cc003.remote-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl task-53-cc003 'flock /tmp/agent-gpu0.lock -c "ctest --test-dir build/sycl-cc003 --output-on-failure -R iom_backend_coexistence_tests"'` — after merging the latest `main`, 1/1 test passed in 0.86 seconds; all CPU+SYCL coexistence assertions passed.
- `.agents/skills/remote-development/scripts/remote-exec rocm task-53-cc003 'flock /tmp/agent-gpu0.lock -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>/dev/null; cmake -S . -B build/sycl-rocm-cc003 -DBUILD_TESTING=ON -DSYCL_ENABLED=ON -DROCM_ENABLED=ON -DCUDA_ENABLED=OFF -DTTNN_ENABLED=OFF"'` — configured successfully on `bv2` with IntelLLVM and HIP enabled.
- `.agents/skills/remote-development/scripts/remote-exec rocm task-53-cc003 'flock /tmp/agent-gpu0.lock -c "source /opt/intel/oneapi/setvars.sh >/dev/null 2>/dev/null; cmake --build build/sycl-rocm-cc003 --target iom_backend_coexistence_tests -j"` — required combined build reached linking but was blocked by the toolchain: `ld.lld: error: cannot open libLLVMSYCL.so: No such file or directory`. The SYCL-only evidence is the delivered combined-coexistence result; no combined test run was claimed.

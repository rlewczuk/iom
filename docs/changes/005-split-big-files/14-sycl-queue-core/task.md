**Status:** done

## Summary

Split SYCL queue lifecycle, submission, regular execution, completion, factory, and testing snapshot into private queue linkage while retaining binary and host-transfer support in copy.cpp.

## Verification

- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-sync sycl 005-split-big-files-14-resume — synchronized the exact assigned worktree.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-14-resume set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build/split-sycl-14 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=ON -DTTNN_ENABLED=OFF && cmake --build build/split-sycl-14 --target libiom iom_sycl iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests — sycl-ls enumerated Level Zero GPUs and all requested targets built.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-14-resume set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && ctest --test-dir build/split-sycl-14 --output-on-failure -R "^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$" — 3/3 tests passed.
- wc -l src/sycl/queue.cpp src/sycl/queue_support.hpp src/sycl/queue_internal.hpp src/sycl/copy.cpp — queue.cpp 469 lines, queue_support.hpp 348, queue_internal.hpp 117; intermediate copy.cpp 728 lines is intentionally retained for task 15.

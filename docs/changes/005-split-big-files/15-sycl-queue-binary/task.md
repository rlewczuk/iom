**Status:** done

## Summary

Moved SYCL binary workspace sizing, checked staging arithmetic, host-staging scalar execution, cleanup, and execute_binary into queue_binary.cpp while preserving queue lifecycle and copy support boundaries.

## Verification

- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-sync sycl 005-split-big-files-15-resume — synchronized the exact assigned worktree.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-15-resume set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build/split-sycl-15 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=ON -DTTNN_ENABLED=OFF && cmake --build build/split-sycl-15 --target libiom iom_sycl iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests — sycl-ls enumerated Level Zero GPUs and all requested targets built.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-15-resume set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && ctest --test-dir build/split-sycl-15 --output-on-failure -R "^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$" — 3/3 tests passed.
- wc -l src/sycl/copy.cpp src/sycl/queue.cpp src/sycl/queue_binary.cpp src/sycl/queue_support.hpp src/sycl/queue_internal.hpp — copy.cpp 335, queue.cpp 469, queue_binary.cpp 415, queue_support.hpp 348, queue_internal.hpp 117; all within the final caps.

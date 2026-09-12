**Status:** done

## Summary

Split SyclDevice declaration and tensor/workspace ownership into private translation-unit components without changing public APIs or runtime behavior.

## Verification

- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-sync sycl 005-split-big-files-06-resume — synchronized the exact assigned worktree.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-06-resume set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build/split-sycl-06 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=ON -DTTNN_ENABLED=OFF && cmake --build build/split-sycl-06 --target libiom iom_sycl iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests — sycl-ls enumerated Level Zero GPUs and all requested targets built.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-exec sycl 005-split-big-files-06-resume set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && ctest --test-dir build/split-sycl-06 --output-on-failure -R "^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$" — 3/3 tests passed.
- wc -l src/sycl/device.cpp src/sycl/device_tensor.cpp src/sycl/device_internal.hpp — 437, 174, and 111 lines; all within the 499-line cap.

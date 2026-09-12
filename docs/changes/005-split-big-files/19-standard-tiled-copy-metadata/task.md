**Status:** done

## Summary

Split standard tiled-copy metadata and grid-stride implementation into the included private fragment while preserving CUDA, ROCm, and SYCL consumers and target registration.

## Verification

- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf .agents/skills/remote-development/scripts/remote-sync sycl 005-split-big-files-19-resume-sycl; .agents/skills/remote-development/scripts/remote-sync cuda 005-split-big-files-19-resume-cuda; .agents/skills/remote-development/scripts/remote-sync rocm 005-split-big-files-19-resume-rocm — synchronized the exact assigned worktree to all required accelerator hosts.
- remote-exec cuda 005-split-big-files-19-resume-cuda cmake configure/build for libiom, iom_cuda, iom_cuda_smoke_tests, iom_cuda_conformance_tests, and iom_backend_coexistence_tests — all requested targets built.
- remote-exec cuda 005-split-big-files-19-resume-cuda ctest --test-dir build/split-cuda-19 --output-on-failure -R "^(iom_cuda_smoke_tests|iom_cuda_conformance_tests|iom_backend_coexistence_tests)$" — 3/3 tests passed.
- remote-exec rocm 005-split-big-files-19-resume-rocm cmake configure/build for libiom, iom_rocm, iom_rocm_smoke_tests, iom_rocm_conformance_tests, and iom_backend_coexistence_tests — all requested targets built.
- remote-exec rocm 005-split-big-files-19-resume-rocm ctest --test-dir build/split-rocm-19 --output-on-failure -R "^(iom_rocm_smoke_tests|iom_rocm_conformance_tests|iom_backend_coexistence_tests)$" — 3/3 tests passed.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf remote-exec sycl 005-split-big-files-19-resume-sycl set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && cmake -S . -B build/split-sycl-19 -DBUILD_TESTING=ON -DCUDA_ENABLED=OFF -DROCM_ENABLED=OFF -DSYCL_ENABLED=ON -DTTNN_ENABLED=OFF && cmake --build build/split-sycl-19 --target libiom iom_sycl iom_sycl_smoke_tests iom_sycl_conformance_tests iom_backend_coexistence_tests — sycl-ls enumerated Level Zero GPUs and all requested targets built.
- REMOTE_DEV_CONFIG=/tmp/iom-sycl-hosts.conf remote-exec sycl 005-split-big-files-19-resume-sycl set +u; source /opt/intel/oneapi/setvars.sh >/tmp/iom-setvars.log 2>&1; set -u; sycl-ls && ctest --test-dir build/split-sycl-19 --output-on-failure -R "^(iom_sycl_smoke_tests|iom_sycl_conformance_tests|iom_backend_coexistence_tests)$" — 3/3 tests passed.
- wc -l src/shared/standard_tiled_copy.inl src/shared/standard_tiled_copy_metadata.inl — 443 and 282 lines; both within the 499-line cap.

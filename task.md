**Status:** done

## Summary

Centralized tensor release-or-quarantine and queue entry resolution in `include/iom/detail/outstanding_work_registry.hpp`. Migrated CPU, CUDA, ROCm, SYCL, and TTNN tensor destructors and queue completion paths to the shared helpers, unified registering-backend state as `detail::RegistryState`, and removed obsolete backend registry-state headers while preserving TTNN native cleanup behavior.

## Verification

- `cmake --build build/ar002-protocol --target iom_tests iom_cpu_tests iom_backend_conformance_cpu_tests -j && ctest --test-dir build/ar002-protocol --output-on-failure -R '^iom_tests$|^iom_cpu_tests$|^iom_backend_conformance_cpu_tests$'` — CPU targets built and 3/3 tests passed.
- CUDA remote-development profile `ar002-protocol-cuda` — configured and built `iom_cuda_conformance_tests`, `iom_cuda_smoke_tests`, and `iom_tests`; backend 2/2 and common 1/1 passed; transactional-failure and queue-destruction lifetime cases passed.
- ROCm remote-development profile `ar002-protocol-rocm` — configured and built `iom_rocm_conformance_tests`, `iom_rocm_smoke_tests`, and `iom_tests`; backend 2/2 and common 1/1 passed; transactional-failure and queue-destruction lifetime cases passed.
- TTNN remote-development profile `ar002-protocol-ttnn` — final worktree rebuild passed; backend 2/2 passed; deferred-lifetime and quarantine-action-allocation-failure cases passed.
- SYCL remote-development profile `ar002-protocol-sycl` — configured with the Intel compiler and UMF runtime environment, built all requested targets, backend 2/2 and common 1/1 passed; destruction-fence, transactional-submission, queue-teardown, pre-enqueue, and no-op lifetime cases passed.
- Source audits and `git diff --check` — one shared `release_or_quarantine` definition with five destructor call sites, one shared `SequenceOutcome`, one SYCL wrapper, obsolete backend state names absent, and deleted state headers absent.

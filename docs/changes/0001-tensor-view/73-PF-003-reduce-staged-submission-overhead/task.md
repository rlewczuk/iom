**Status:** done

## Summary

Replaced heap-erased fence callbacks with a fixed 32-byte inline `detail::Fence`, including explicit copy/move/destroy trampolines and invalidated-fence support. Added cached, intrusive-refcounted CUDA/ROCm event leases; inline TTNN pointer captures and SYCL shared-state captures; pointer-based TTNN task views; and synchronous inline CPU completion. Preserved staged-worker callback ordering and shared registry release/quarantine behavior.

## Verification

- CPU Release configure/build — `iom_tests`, `iom_cpu_tests`, `iom_backend_conformance_cpu_tests`, and `iom_cpu_bench` built.
- CPU CTest — 3/3 selected tests passed after the final common regression test; the CPU destruction-before-wait lifetime case passed.
- Common inline-fence tests — capture copy/move/assignment and snapshot-after-worker-refcount-drop cases passed; the latter passed 16 assertions. ASAN `*Inline*` run passed 4/4 cases and 39/39 assertions.
- Allocation probe — throwaway global `operator new` counter linked against `libiom.a`; 1000 copy+wait cycles reported `copy_allocations=0`, and 1000 identical-window no-op cycles reported `noop_allocations=0`.
- CPU bench recording-only smoke — five ordinary-load runs reported queued copy medians `0.15, 0.13, 0.09, 0.09, 0.09 us` and no-op medians `0.05, 0.05, 0.041, 0.04, 0.04 us`. A separate pre-change tree was not built because task verification remained confined to the assigned worktree; no cross-tree latency claim is made.
- CUDA remote-development profile `pf003-cuda` — final rebuild passed `iom_cuda_smoke_tests` and `iom_cuda_conformance_tests` 2/2 plus `iom_tests` 1/1; transactional and queue-focused lifetime filters passed.
- ROCm remote-development profile `pf003-rocm` — final rebuild passed `iom_rocm_smoke_tests` and `iom_rocm_conformance_tests` 2/2 plus `iom_tests` 1/1; transactional and queue-focused lifetime filters passed 3/3.
- TTNN remote-development profile `pf003-ttnn` — final rebuild passed `iom_ttnn_smoke_tests` and `iom_ttnn_conformance_tests` 2/2 plus `iom_tests` 1/1; deferred-lifetime and quarantine fault filters passed 2/2.
- SYCL remote-development profile `pf003-sycl` — Intel toolchain setup and `sycl-ls` Level Zero enumeration completed; final rebuild passed `iom_sycl_smoke_tests` and `iom_sycl_conformance_tests` 2/2 plus `iom_tests` 1/1; destruction, transactional, and pending-copy filters passed 3/3.
- Source audits and `git diff --check` — legacy `std::function<FenceResult>`, `make_fence`, CUDA/ROCm destroy-resource helpers, CPU `StagedWorker` symbols, TTNN value task views, and protected-file diffs are absent; inline Fence layout/constants and backend lease/capture symbols are present.

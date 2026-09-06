**Status:** done

## Summary

Added iom::detail::FenceCaptureOps<Capture> beside Fence in include/iom/detail/outstanding_work_registry.hpp as the single owner of the mechanical capture-storage ops (copy_construct, move_construct with move-then-destroy_at, destroy) with nothrow and size/align static_asserts; deleted the hand-written cuda_fence_*, hip_fence_*, and sycl_fence_* trios and wired build_cuda_fence, build_hip_fence, and build_sycl_fence to FenceCaptureOps instantiations; kept per-backend invoke functions and capture construction; TTNN and CPU untouched

## Verification

- cmake -S . -B build -DBUILD_TESTING=ON with all backends OFF plus cmake --build build --target iom_tests and ctest -R iom_tests in the task worktree — iom_tests passed 1/1
- remote-exec cuda ar011-fence-capture on bv1, CUDA 13.2.78: built iom_cuda_conformance_tests, iom_cuda_smoke_tests, iom_tests — ctest iom_cuda_ passed 2/2 (smoke 0.44s, conformance 21.76s), iom_tests passed
- remote-exec sycl ar011-fence-capture on bv2 with oneAPI setvars: sycl-ls enumerated 2 Level Zero Arc Pro B60 GPUs, then ctest iom_sycl_ passed 2/2 (smoke 0.86s, conformance 6.68s), iom_tests passed
- remote-exec rocm ar011-fence-capture on bv2, ROCm 10.0 HIP clang 23: ctest iom_rocm_ passed 2/2 (smoke 0.88s, conformance 26.20s), iom_tests passed
- grep audits in worktree: _fence_copy_construct/_fence_move_construct/_fence_storage_destroy zero matches in src; FenceCaptureOps has one definition and exactly three instantiations (cuda, rocm, sycl); one fence_invoke and one build_fence per backend (4 each); git diff --stat src/ttnn src/cpu empty

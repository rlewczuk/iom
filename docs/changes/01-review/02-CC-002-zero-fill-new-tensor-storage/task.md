**Status:** done

## Summary

<<<<<<< HEAD
Zero-filled new tensor storage before exposure across standard tiled backends.

## Verification

- CPU conformance and CPU unit tests passed; CUDA, SYCL, and TTNN conformance passed 1/1; ROCm full conformance passed 1/1 after checked hipDeviceSynchronize was added after hipMemset to order default-stream zeroing against reused storage.
=======
Zero-filled standard tensor storage before exposure across CPU, CUDA, ROCm, and SYCL; CUDA construction now synchronizes the memset before the tensor becomes visible.

## Verification

- CUDA exact 02 worktree: cmake configure with CUDA and ctest -R iom_cuda_conformance_tests — 1/1 passed.
>>>>>>> 5f6915b (spec-run-task(01-review/02-CC-002-zero-fill-new-tensor-storage): zero-fill standard tensor storage before exposure across CPU CUDA ROCm and SYCL)

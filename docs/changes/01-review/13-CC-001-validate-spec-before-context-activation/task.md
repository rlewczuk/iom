**Status:** done

## Summary

CUDA/ROCm create_tensor now validates the spec before any context activation: the standalone activate() is removed from both paths (allocation-time activation stays with the pre_allocate callback), a rejected spec throws the deterministic spec error with zero ctx_set_current/hipSetDevice calls and zero allocator allocations, and ROCm leaves the calling thread current device unchanged on rejection. Driver-call-probe smoke coverage added for both backends.

## Verification

- ctest -R "^iom_cuda_smoke_tests$" on bv1 — 1/1 passed (0.42s)
- ctest -R "^iom_rocm_smoke_tests$" on bv2 — 1/1 passed (0.88s)
- combined wave-3 train: CUDA/ROCm smoke+conformance+coexistence 3/3 each

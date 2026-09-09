**Status:** done

## Summary

Accepted documented ADD ranks through dynamic traversal and metadata storage.

## Verification

- CPU backend conformance passed 1/1; CUDA conformance passed 1/1 after CUDA build; ROCm conformance passed 1/1 after HIP build; SYCL conformance passed 1/1 with sycl-ls Level Zero GPU enumeration; SYCL rank-boundary test passed under host ASan/UBSan; TTNN conformance passed 1/1. Rank 8/9/16/17 shared oracle cases completed with repeated waits.

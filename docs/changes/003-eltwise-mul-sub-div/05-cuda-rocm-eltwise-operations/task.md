**Status:** done

## Summary

Generalized the shared CUDA and ROCm policy queue and standard-tiled binary kernel for ADD, MUL, SUB, and floating DIV with preserved metadata, mapping, ownership, and failure semantics.

## Verification

- Remote CUDA: CUDA-enabled CMake configured with other optional backends OFF; iom_cuda_conformance_tests built and ctest passed 1/1. Remote ROCm: ROCm-enabled CMake configured with other optional backends OFF; iom_rocm_conformance_tests built and ctest passed 1/1.

**Status:** done

## Summary

Retained pooled metadata until event or stream-drain completion proof.

## Verification

- CPU was not in scope. CUDA: cmake --build build/cuda --target iom_cuda_smoke_tests iom_cuda_conformance_tests and ctest -R ^iom_cuda_.*_tests$ passed 2/2; compute-sanitizer memcheck double-fault smoke passed with 0 errors; racecheck passed with 0 hazards. ROCm: corresponding build and ctest -R ^iom_rocm_.*_tests$ passed 2/2.

**Status:** done

## Summary

CUDA and ROCm copy transactions register both owners and reserve completion before native effects; pre-launch failures roll back and post-launch failures retain ownership safely.

## Verification

- CUDA focused remote conformance cases CUDA copy reservation failures roll back before native work and CUDA submission remains transactional across post-enqueue failures — both passed.
- ROCm focused remote conformance cases ROCm copy reservation failures roll back before native work and ROCm submission remains transactional across post-enqueue failures — both passed.
- Combined final train CUDA and ROCm conformance plus backend coexistence — all four targets passed.

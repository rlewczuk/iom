**Status:** done

## Summary

Documented the shipped ADD, MUL, SUB, and DIV APIs, dtype domains, arithmetic, validation, views, ownership, queues, failures, backend boundaries, compatibility, and conformance source map in the approved docs and header comments.

## Verification

- git diff d950ea46d7a2df0488664c92bc88514b03bcf354..HEAD --check -- docs/ARCHITECTURE.md docs/BACKEND_CONTRACT.md README.md include/iom/iom.hpp include/iom/device.hpp include/iom/tensor.hpp include/iom/ttnn/device.hpp — passed.
- Focused review of the listed files for stale MUL-unsupported, ADD-only, native-SDK narrowing, promotion, public-query, and fallback claims; no contradictory claims found.
- Final d950ea46d7a2df0488664c92bc88514b03bcf354 train CPU, CUDA, ROCm, SYCL, TTNN conformance and coexistence gates — all passed.

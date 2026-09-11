**Status:** done

## Summary

Fixed 4*C queue resources on CUDA/ROCm/SYCL: at most four live DeviceOps queues per Device, each published only after atomically reserving one disjoint C-slot partition from the Device-wide FixedSizeAllocator plus exactly C eagerly created completion resources and fixed host mirrors; transactional construction with full rollback and fifth-queue bad_alloc before stream/worker creation; lazy machinery removed (ensure_slot_capacity, SyclMetadataSlotPool, lazy event creation, poisoned buffers, metadata native alloc/free, array resizing); inline/no-metadata paths slot-free, binary/pointer-copy paths at most one immutable slot; compile-time 512-byte/32-alignment descriptor asserts; drained-partition reuse and unknown-completion quarantine until covering proof; six-queue coexistence scenario migrated to the deliberate four-live-queue cap with deadlock-proof barrier drain

## Verification

- focused: ctest -R "^iom_cuda_smoke_tests$" / "^iom_cuda_conformance_tests$" (remote bv1) — Passed 0.59s / 35.02s; ctest -R "^iom_rocm_smoke_tests$" / "^iom_rocm_conformance_tests$" (remote bv2) — Passed 1.03s / 39.81s; ctest -R "^iom_sycl_smoke_tests$" / "^iom_sycl_conformance_tests$" (remote bv2) — Passed 0.95s / 6.55s; CUDA/ROCM/SYCL sources unchanged by later test-only amendments
- combined train gates: local iom_tests+iom_cpu_tests+iom_backend_conformance_cpu_tests 3/3 (0.07s, 0.12s, 30.63s); CUDA smoke 0.32s + conformance 35.93s; ROCm smoke 0.95s + conformance 39.07s; SYCL smoke 0.18s + conformance 6.66s
- combined coexistence after the four-live-queue migration (7a0e0aa0 tree): ctest -R "^iom_backend_coexistence_tests$" — Passed 0.27s (CUDA bv1), 0.84s (ROCm bv2), 0.22s (SYCL bv2); C=1/16/17 partition, fifth-queue rejection, rollback, drained reuse, and quarantine scenarios exercised inside the smoke suites
- TTNN gates not run: tt-nn CMake SDK absent on configured host bv1; TTNN is not an enabled backend in this environment

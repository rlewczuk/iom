**Status:** done

## Summary

Added real-queue MUL/SUB/floating-DIV value coverage for CPU, SYCL, and TTNN. Repaired SYCL F64 binary staging by widening each extracted bit to uint64 before shifts, preserving upper 32 bits in host staging.

## Verification

- cmake --build build/review-cpu --target iom_backend_conformance_cpu_tests && ctest --test-dir build/review-cpu --output-on-failure -R ^iom_backend_conformance_cpu_tests$ — CPU conformance passed
- remote-sync sycl 003-01-review-06-nt003-final-sycl && remote-exec sycl ... ./build/test/iom_sycl_conformance_tests — 20 test cases and 142,328 assertions passed
- remote-sync ttnn 003-01-review-06-nt003-final-ttnn && remote-exec ttnn ... ./build/test/iom_ttnn_conformance_tests — 35 test cases and 295,915 assertions passed
- Temporary SYCL ADD-for-MUL dispatch mutation failed the real-queue operation-value case with 23,369 failed assertions; restored BinaryOp::mul and reran the fixed focused case with 87,661 assertions passed

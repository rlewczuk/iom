**Status:** done

## Summary

Generalized staged SYCL binary execution for ADD, MUL, SUB, and DIV with existing queue and copy semantics.

## Verification

- Remote SYCL: sycl-ls enumerated two Intel Arc Pro B60 Level Zero GPUs; cmake configured with SYCL_ENABLED=ON and other optional backends OFF; iom_sycl_conformance_tests built and ctest passed 1/1.
